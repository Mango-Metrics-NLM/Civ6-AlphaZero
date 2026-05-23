"""Build and push a fresh MuZero-World-Model container via Cloud Build.

Creates a project-isolated Artifact Registry repo (`civ6az-muzero-world-model`)
and builds the container remotely — no local Docker required.

Usage:
    python scripts/cloud_build_container.py [--dry-run] [--tag v12-muzero]

Env vars (optional overrides):
    CIV6AZ_VERTEX_PROJECT  - GCP project (default: from ADC quota_project_id)
    CIV6AZ_VERTEX_REGION   - Region (default: us-central1)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import socket
import sys
import time
from typing import Any

# ── IPv4-only patch (fixes Windows IPv6 routing hang) ──────────────────────
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [  # type: ignore[assignment]
    r for r in _orig_getaddrinfo(*a, **k) if r[0] == socket.AF_INET
]

import requests  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
REPO_NAME = "civ6az-muzero-world-model"  # Dedicated AR repo — NOT shared
DEFAULT_IMAGE_NAME = "civ6az-muzero"
DEFAULT_TAG = "v12-muzero"


def get_adc_credentials() -> dict[str, str]:
    """Read ADC credentials from disk."""
    adc_path = pathlib.Path(os.environ["APPDATA"], "gcloud", "application_default_credentials.json")
    if not adc_path.exists():
        raise FileNotFoundError(f"ADC file not found: {adc_path}")
    return json.loads(adc_path.read_text(encoding="utf-8"))


def exchange_token(adc: dict[str, str]) -> str:
    """Exchange refresh token for access token."""
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": adc["client_id"],
            "client_secret": adc["client_secret"],
            "refresh_token": adc["refresh_token"],
            "grant_type": "refresh_token",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return str(resp.json()["access_token"])


def resolve_config() -> dict[str, str]:
    """Resolve project/region config from env or ADC."""
    adc = get_adc_credentials()
    project = os.environ.get("CIV6AZ_VERTEX_PROJECT", adc.get("quota_project_id", ""))
    region = os.environ.get("CIV6AZ_VERTEX_REGION", "us-central1")
    if not project:
        raise ValueError("No project found — set CIV6AZ_VERTEX_PROJECT or ADC quota_project_id")
    return {"project": project, "region": region}


def ensure_ar_repo(token: str, project: str, region: str) -> None:
    """Create Artifact Registry Docker repo if it doesn't exist."""
    url = (
        f"https://artifactregistry.googleapis.com/v1/"
        f"projects/{project}/locations/{region}/repositories/{REPO_NAME}"
    )
    headers = {"Authorization": f"Bearer {token}"}

    # Check if exists
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code == 200:
        logger.info("AR repo '%s' already exists in %s/%s", REPO_NAME, project, region)
        return

    # Create it
    create_url = (
        f"https://artifactregistry.googleapis.com/v1/"
        f"projects/{project}/locations/{region}/repositories"
    )
    payload = {
        "format": "DOCKER",
        "description": "MuZero World Model training containers (project-isolated)",
    }
    resp = requests.post(
        create_url,
        headers={**headers, "Content-Type": "application/json"},
        json=payload,
        params={"repositoryId": REPO_NAME},
        timeout=30,
    )
    if resp.status_code in (200, 409):  # 409 = already exists (race condition)
        logger.info("Created AR repo '%s' in %s/%s", REPO_NAME, project, region)
    elif resp.status_code == 202:
        # Long-running operation — repo creation in progress
        logger.info("AR repo creation started (LRO). Waiting...")
        time.sleep(5)  # Brief wait for repo to become available
    else:
        logger.error("Failed to create AR repo: %s %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()


def create_source_tarball(project_root: pathlib.Path) -> pathlib.Path:
    """Create a gzipped tarball of the project source for Cloud Build."""
    import tarfile
    import tempfile

    # Patterns to exclude from the tarball (save upload time & space)
    exclude_patterns = {
        ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".hypothesis", "node_modules", ".venv", "venv", "wandb", "mlruns",
        "checkpoints", ".tox", "dist", "*.egg-info", ".coverage",
        "htmlcov", "training_data", "models",
    }

    tar_path = pathlib.Path(tempfile.gettempdir()) / "muzero_cloud_build_source.tar.gz"

    def _filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = pathlib.PurePosixPath(tarinfo.name).parts
        for p in parts:
            if p in exclude_patterns or p.endswith(".egg-info"):
                return None
        return tarinfo

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(project_root), arcname=".", filter=_filter)

    size_mb = tar_path.stat().st_size / (1024 * 1024)
    logger.info("Source tarball: %s (%.1f MB)", tar_path, size_mb)
    return tar_path


def upload_source_to_gcs(
    token: str,
    project: str,
    tar_path: pathlib.Path,
    bucket: str = "",
) -> tuple[str, str]:
    """Upload source tarball to GCS for Cloud Build.

    Returns (bucket_name, object_name).
    """
    if not bucket:
        # Cloud Build uses a default bucket: {project_id}_cloudbuild
        bucket = f"{project}_cloudbuild"

    object_name = f"source/civ6az-muzero-world-model-{int(time.time())}.tar.gz"

    # Upload via GCS JSON API (resumable for large files, simple for small)
    url = (
        f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
        f"?uploadType=media&name={object_name}"
    )
    with open(tar_path, "rb") as f:
        data = f.read()

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/gzip",
        },
        data=data,
        timeout=120,
    )

    if resp.status_code not in (200, 201):
        logger.error("GCS upload failed: %s %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()

    logger.info("Uploaded source to gs://%s/%s", bucket, object_name)
    return bucket, object_name


def submit_cloud_build(
    token: str,
    project: str,
    region: str,
    image_tag: str,
    source_bucket: str,
    source_object: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit a Cloud Build job to build and push the container."""
    image_uri = f"{region}-docker.pkg.dev/{project}/{REPO_NAME}/{DEFAULT_IMAGE_NAME}:{image_tag}"

    build_config: dict[str, Any] = {
        "source": {
            "storageSource": {
                "bucket": source_bucket,
                "object": source_object,
            },
        },
        "steps": [
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": [
                    "build",
                    "-t", image_uri,
                    "-f", "dockerfiles/Dockerfile.vertex",
                    ".",
                ],
            },
        ],
        "images": [image_uri],
        "options": {
            "machineType": "E2_HIGHCPU_8",
            "logging": "CLOUD_LOGGING_ONLY",
        },
        "timeout": "1200s",
    }

    logger.info("Container image URI: %s", image_uri)

    if dry_run:
        logger.info("DRY RUN — Cloud Build config:\n%s", json.dumps(build_config, indent=2))
        return {"image_uri": image_uri, "status": "DRY_RUN"}

    url = f"https://cloudbuild.googleapis.com/v1/projects/{project}/builds"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=build_config,
        timeout=30,
    )

    if resp.status_code != 200:
        logger.error("Cloud Build submission failed: %s %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()

    result = resp.json()
    build_id = result.get("metadata", {}).get("build", {}).get("id", "unknown")
    logger.info("Cloud Build submitted! Build ID: %s", build_id)
    logger.info("Monitor: https://console.cloud.google.com/cloud-build/builds/%s?project=%s",
                build_id, project)

    return {"image_uri": image_uri, "build_id": build_id, "status": "SUBMITTED"}


def poll_build_status(token: str, project: str, build_id: str, timeout_minutes: int = 20) -> str:
    """Poll Cloud Build until completion or timeout."""
    url = f"https://cloudbuild.googleapis.com/v1/projects/{project}/builds/{build_id}"
    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.time() + timeout_minutes * 60

    terminal_states = {"SUCCESS", "FAILURE", "INTERNAL_ERROR", "TIMEOUT", "CANCELLED"}

    while time.time() < deadline:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            logger.warning("Failed to poll build: %s", resp.status_code)
            time.sleep(15)
            continue

        data = resp.json()
        status = data.get("status", "UNKNOWN")
        logger.info("Build %s: %s", build_id[:8], status)

        if status in terminal_states:
            if status == "SUCCESS":
                logger.info("Container built and pushed successfully!")
            else:
                log_url = data.get("logUrl", "N/A")
                logger.error("Build failed with status %s — logs: %s", status, log_url)
            return status

        time.sleep(15)

    logger.error("Build timed out after %d minutes", timeout_minutes)
    return "TIMEOUT"


def update_configs(image_uri: str) -> None:
    """Update YAML configs to reference the new image."""
    config_dir = pathlib.Path("configs")
    if not config_dir.exists():
        logger.warning("configs/ directory not found — skipping config update")
        return

    count = 0
    for yaml_file in config_dir.glob("vertex_*.yaml"):
        content = yaml_file.read_text(encoding="utf-8")
        if "imageUri:" in content:
            # Replace any imageUri line with new one
            lines = content.splitlines()
            new_lines = []
            for line in lines:
                if "imageUri:" in line:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(f"{' ' * indent}imageUri: {image_uri}")
                    count += 1
                else:
                    new_lines.append(line)
            yaml_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    logger.info("Updated %d imageUri references in configs/", count)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build MuZero World Model container via Cloud Build")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"Image tag (default: {DEFAULT_TAG})")
    parser.add_argument("--dry-run", action="store_true", help="Show config without submitting")
    parser.add_argument("--no-wait", action="store_true", help="Submit and exit without waiting")
    parser.add_argument("--no-update-configs", action="store_true", help="Skip updating YAML configs")
    args = parser.parse_args()

    cfg = resolve_config()
    logger.info("Project: %s, Region: %s", cfg["project"], cfg["region"])

    adc = get_adc_credentials()
    token = exchange_token(adc)

    # Step 1: Ensure AR repo exists
    logger.info("Step 1: Ensuring Artifact Registry repo '%s' exists...", REPO_NAME)
    if not args.dry_run:
        ensure_ar_repo(token, cfg["project"], cfg["region"])

    # Step 2: Create source tarball and upload to GCS
    logger.info("Step 2: Creating source tarball...")
    project_root = pathlib.Path(__file__).resolve().parent.parent
    tar_path = create_source_tarball(project_root)

    source_bucket = ""
    source_object = ""
    if not args.dry_run:
        logger.info("Step 2b: Uploading source to GCS...")
        source_bucket, source_object = upload_source_to_gcs(
            token, cfg["project"], tar_path,
        )

    # Step 3: Submit Cloud Build
    logger.info("Step 3: Submitting Cloud Build...")
    result = submit_cloud_build(
        token, cfg["project"], cfg["region"], args.tag,
        source_bucket, source_object,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return 0

    image_uri = result["image_uri"]
    build_id = result["build_id"]

    # Step 3: Wait for build (unless --no-wait)
    if args.no_wait:
        logger.info("--no-wait: build submitted, exiting. Monitor at the URL above.")
        return 0

    logger.info("Step 3: Waiting for build to complete...")
    status = poll_build_status(token, cfg["project"], build_id)

    if status != "SUCCESS":
        return 1

    # Step 4: Update config files
    if not args.no_update_configs:
        logger.info("Step 4: Updating config files with new image URI...")
        update_configs(image_uri)

    # Step 5: Update submission script default
    logger.info("New image URI: %s", image_uri)
    logger.info("To submit a job with this image:")
    logger.info("  python scripts/gcp_diagnose_and_submit.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
