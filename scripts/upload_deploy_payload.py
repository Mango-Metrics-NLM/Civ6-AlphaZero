#!/usr/bin/env python3
"""
Pack and upload deployment payload to GCS.
The VM then downloads and installs via gsutil (no git access needed).
"""
from __future__ import annotations
import os, tarfile, pathlib, sys, tempfile

try:
    from google.cloud import storage
except ImportError:
    print("google-cloud-storage not found")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────
BUCKET = os.environ.get("CIV6AZ_GCS_BUCKET", "civ6az-training")
GCS_OBJECT = "deploy/civ6az-orchestrator-payload.tar.gz"
PROJECT_ROOT = pathlib.Path(__file__).parent.parent

INCLUDE = [
    "alpha_zero/cloud/gcp",
    "scripts/deploy_orchestrator_vm.py",
    "scripts/gcp_vm_check.py",
    "scripts/gcp_diagnose.py",
    "scripts/setup_vm_orchestrator.sh",
    "scripts/systemd",
    "requirements.txt",
]

EXCLUDE_SUFFIXES = {".pyc", ".pth"}
EXCLUDE_DIRS = {"__pycache__", ".git", "node_modules"}


def should_include(path: pathlib.Path) -> bool:
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return False
    return True


def create_tarball(tar_path: pathlib.Path) -> int:
    """Create tarball, returns number of files included."""
    count = 0
    with tarfile.open(tar_path, "w:gz") as tf:
        for rel in INCLUDE:
            full = PROJECT_ROOT / rel
            if not full.exists():
                print(f"  SKIP (not found): {rel}")
                continue
            if full.is_file():
                if should_include(full):
                    tf.add(full, arcname=rel)
                    count += 1
                    print(f"  + {rel}")
            elif full.is_dir():
                for child in sorted(full.rglob("*")):
                    if child.is_file() and should_include(child):
                        arc = str(child.relative_to(PROJECT_ROOT)).replace("\\", "/")
                        tf.add(child, arcname=arc)
                        count += 1
    return count


def upload_to_gcs(tar_path: pathlib.Path) -> str:
    """Upload tarball to GCS, return gs:// URI."""
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(GCS_OBJECT)
    size_mb = tar_path.stat().st_size / 1e6
    print(f"  Uploading {size_mb:.1f} MB to gs://{BUCKET}/{GCS_OBJECT}...")
    blob.upload_from_filename(str(tar_path))
    return f"gs://{BUCKET}/{GCS_OBJECT}"


def main() -> int:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        tar_path = pathlib.Path(f.name)

    try:
        print("Creating deployment archive...")
        count = create_tarball(tar_path)
        size_mb = tar_path.stat().st_size / 1e6
        print(f"Archive: {count} files, {size_mb:.2f} MB")

        print("Uploading to GCS...")
        uri = upload_to_gcs(tar_path)
        print(f"[OK] Uploaded: {uri}")

        print()
        print("=" * 60)
        print("Next: SSH into VM and run:")
        print(f"  gsutil cp {uri} ~/civ6az-payload.tar.gz")
        print(f"  mkdir -p ~/Civ6-AlphaZero && tar -xzf ~/civ6az-payload.tar.gz -C ~/Civ6-AlphaZero")
        print(f"  sudo cp ~/Civ6-AlphaZero/scripts/systemd/civ6az-orchestrator.service /etc/systemd/system/")
        print(f"  sudo sed -i 's/civ6az/ianshank/g' /etc/systemd/system/civ6az-orchestrator.service")
        print(f"  sudo systemctl daemon-reload && sudo systemctl enable --now civ6az-orchestrator")
        print("=" * 60)

    finally:
        tar_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
