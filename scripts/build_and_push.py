#!/usr/bin/env python3
"""Build and push the training container to GCR using Cloud Build.

This script triggers a Cloud Build job to build and push the container image.
It creates a tar archive of the source code and uploads it to Cloud Build.

Usage:
    python scripts/build_and_push.py [--tag TAG]
"""

from __future__ import annotations

import argparse
import io
import os
import tarfile
from pathlib import Path

from google.cloud import storage
from google.cloud.devtools import cloudbuild_v1


def reset_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Reset tarinfo to avoid permission issues.

    Returns None to skip problematic files (symlinks, special files).
    """
    # Skip symlinks and special files
    if tarinfo.issym() or tarinfo.islnk():
        return None
    if not (tarinfo.isfile() or tarinfo.isdir()):
        return None

    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = "root"
    tarinfo.gname = "root"
    tarinfo.mode = 0o644 if tarinfo.isfile() else 0o755
    # Clear any extended attributes
    tarinfo.pax_headers = {}
    return tarinfo


def create_source_archive(project_root: Path) -> bytes:
    """Create a tar.gz archive of the source code."""
    buffer = io.BytesIO()

    # Files/dirs to include
    include_patterns = [
        "alpha_zero/",
        "configs/",
        "dockerfiles/",
        "pyproject.toml",
        "requirements-runtime.txt",
    ]

    # Files/dirs to exclude
    exclude_patterns = [
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".coverage",
        ".tox",
        ".nox",
        ".eggs",
        "*.egg-info",
        "*.pyc",
        "*.pyo",
        "*.so",
        "*.dll",
        "wandb/",
        "checkpoints/",
        "logs/",
        "data/",
        ".env",
        ".venv",
        "venv/",
        "*.lock",
        "*.sqlite",
        "*.db",
    ]

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for pattern in include_patterns:
            path = project_root / pattern.rstrip("/")
            if path.is_file():
                # Single file
                arcname = pattern.rstrip("/")
                tar.add(str(path), arcname=arcname, filter=reset_tarinfo)
            elif path.is_dir():
                # Directory - walk and add files
                for root, dirs, files in os.walk(path):
                    # Filter out excluded directories
                    dirs[:] = [d for d in dirs if not any(excl in d for excl in exclude_patterns)]

                    for file in files:
                        # Skip excluded file patterns
                        if any(excl in file for excl in exclude_patterns):
                            continue

                        file_path = Path(root) / file
                        arcname = str(file_path.relative_to(project_root))
                        tar.add(str(file_path), arcname=arcname, filter=reset_tarinfo)

    buffer.seek(0)
    return buffer.read()


def trigger_cloud_build(
    project_id: str,
    bucket_name: str,
    tag: str = "latest",
) -> str:
    """Trigger Cloud Build and wait for completion."""
    print(f"Building container with tag: {tag}")

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create source archive
    print("Creating source archive...")
    archive_data = create_source_archive(project_root)
    print(f"Archive size: {len(archive_data) / 1024:.1f} KB")

    # Upload to GCS
    gcs_path = f"cloudbuild-source/civ6az-{tag}.tar.gz"
    print(f"Uploading to gs://{bucket_name}/{gcs_path}...")

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(archive_data, content_type="application/gzip")

    # Create Cloud Build client
    build_client = cloudbuild_v1.CloudBuildClient()

    # Build configuration
    build = cloudbuild_v1.Build(
        source=cloudbuild_v1.Source(
            storage_source=cloudbuild_v1.StorageSource(
                bucket=bucket_name,
                object_=gcs_path,
            )
        ),
        steps=[
            # Build the Docker image
            cloudbuild_v1.BuildStep(
                name="gcr.io/cloud-builders/docker",
                args=[
                    "build",
                    "-t",
                    f"gcr.io/{project_id}/civ6az-training:{tag}",
                    "-f",
                    "dockerfiles/Dockerfile.vertex",
                    ".",
                ],
                id="build",
            ),
            # Smoke test
            cloudbuild_v1.BuildStep(
                name=f"gcr.io/{project_id}/civ6az-training:{tag}",
                entrypoint="python",
                args=[
                    "-c",
                    'import sys; sys.path.insert(0, "/app"); from alpha_zero.cloud.gcp.config import GCPConfig; print("Container smoke test passed")',
                ],
                dir_="/app",
                id="smoke-test",
                wait_for=["build"],
            ),
            # Test selfplay worker
            cloudbuild_v1.BuildStep(
                name=f"gcr.io/{project_id}/civ6az-training:{tag}",
                entrypoint="python",
                args=[
                    "-m",
                    "alpha_zero.cloud.gcp.selfplay_worker",
                    "--dry_run",
                    "--worker_id=0",
                    "--model_version=1",
                    "--iteration=1",
                    "--games=1",
                ],
                dir_="/app",
                id="test-selfplay",
                wait_for=["smoke-test"],
            ),
            # Push the image
            cloudbuild_v1.BuildStep(
                name="gcr.io/cloud-builders/docker",
                args=[
                    "push",
                    f"gcr.io/{project_id}/civ6az-training:{tag}",
                ],
                id="push",
                wait_for=["test-selfplay"],
            ),
        ],
        images=[f"gcr.io/{project_id}/civ6az-training:{tag}"],
        options=cloudbuild_v1.BuildOptions(
            logging=cloudbuild_v1.BuildOptions.LoggingMode.CLOUD_LOGGING_ONLY,
            machine_type=cloudbuild_v1.BuildOptions.MachineType.E2_HIGHCPU_8,
        ),
        timeout={"seconds": 1800},
        tags=["civ6az", "training", "vertex-ai"],
    )

    # Start the build
    print("Starting Cloud Build...")
    operation = build_client.create_build(project_id=project_id, build=build)

    # Get build ID
    build_id = operation.metadata.build.id
    print(f"Build ID: {build_id}")
    print(
        f"View logs: https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}"
    )

    # Wait for completion
    print("Waiting for build to complete (this may take 10-15 minutes)...")
    result = operation.result()

    if result.status == cloudbuild_v1.Build.Status.SUCCESS:
        print("\nBuild completed successfully!")
        print(f"Image: gcr.io/{project_id}/civ6az-training:{tag}")
        return f"gcr.io/{project_id}/civ6az-training:{tag}"
    else:
        print(f"\nBuild failed with status: {result.status.name}")
        raise RuntimeError(f"Build failed: {result.status.name}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build and push training container")
    parser.add_argument("--tag", default="latest", help="Container image tag")
    args = parser.parse_args()

    # Load .env file if it exists
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)

    # Get project and bucket from environment with validation
    project_id = os.environ.get("CIV6AZ_VERTEX_PROJECT")
    bucket_name = os.environ.get("CIV6AZ_GCS_BUCKET")

    # Warn about missing configuration
    using_defaults = False
    if not project_id:
        print("WARNING: CIV6AZ_VERTEX_PROJECT not set, using default 'civ6-alphazero'")
        print("  Set this environment variable for your GCP project.")
        project_id = "civ6-alphazero"
        using_defaults = True

    if not bucket_name:
        print("WARNING: CIV6AZ_GCS_BUCKET not set, using default 'civ6az-training'")
        print("  Set this environment variable for your GCS bucket.")
        bucket_name = "civ6az-training"
        using_defaults = True

    if using_defaults:
        print("\nTo configure, set environment variables or create .env file:")
        print("  export CIV6AZ_VERTEX_PROJECT=<your-project-id>")
        print("  export CIV6AZ_GCS_BUCKET=<your-bucket-name>")
        print()

    print(f"Project: {project_id}")
    print(f"Bucket: {bucket_name}")

    try:
        image_uri = trigger_cloud_build(project_id, bucket_name, args.tag)
        print(f"\nDone! Set CIV6AZ_CONTAINER_IMAGE_URI={image_uri}")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
