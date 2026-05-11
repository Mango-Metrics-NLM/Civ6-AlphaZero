"""Provision GCP resources for MuZero-World-Model (IPv4 bypass).

Creates:
1. Artifact Registry repo: civ6az-muzero-world-model
2. GCS bucket: civ6az-muzero-training

Uses direct REST API with IPv4 enforcement to avoid gcloud CLI IPv6 hang.

Usage:
    python scripts/gcp_provision_resources.py [--dry-run]
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import sys
import time
from typing import Any

# --- IPv4 enforcement (before any network imports) ---
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(
    *args: Any, **kwargs: Any
) -> list[tuple[socket.AddressFamily, int, int, str, tuple[str, int]]]:
    """Filter DNS results to IPv4 only."""
    results = _orig_getaddrinfo(*args, **kwargs)
    return [r for r in results if r[0] == socket.AF_INET]


socket.getaddrinfo = _ipv4_only_getaddrinfo  # type: ignore[assignment]

import requests  # noqa: E402

# --- Configuration ---
PROJECT = os.environ.get("CIV6AZ_VERTEX_PROJECT", "hrmtrain")
REGION = "us-central1"
AR_REPO = "civ6az-muzero-world-model"
GCS_BUCKET = "civ6az-muzero-training"

_ADC_PATH = pathlib.Path(
    os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        os.path.join(
            os.environ.get("APPDATA", ""),
            "gcloud",
            "application_default_credentials.json",
        ),
    )
)


def _get_token() -> str:
    """Exchange ADC refresh token for access token."""
    adc = json.loads(_ADC_PATH.read_text(encoding="utf-8"))
    if adc.get("type") != "authorized_user":
        msg = f"Unsupported ADC type: {adc.get('type')}"
        raise ValueError(msg)

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
    if resp.status_code != 200:
        msg = f"Token exchange failed ({resp.status_code}): {resp.text[:200]}"
        raise RuntimeError(msg)
    return resp.json()["access_token"]


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def create_ar_repo(token: str, dry_run: bool = False) -> bool:
    """Create Artifact Registry Docker repo."""
    url = (
        f"https://artifactregistry.googleapis.com/v1/"
        f"projects/{PROJECT}/locations/{REGION}/repositories"
        f"?repositoryId={AR_REPO}"
    )
    body = {"format": "DOCKER", "description": "MuZero-World-Model container images"}

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Creating AR repo: {AR_REPO}")
    print(f"  URL: {url}")

    if dry_run:
        print("  Payload:", json.dumps(body, indent=2))
        return True

    resp = requests.post(url, headers=_headers(token), json=body, timeout=30)

    if resp.status_code == 409:
        print(f"  AR repo '{AR_REPO}' already exists (409) — OK")
        return True
    elif resp.status_code == 200:
        print(f"  AR repo '{AR_REPO}' created successfully!")
        # Wait for LRO to complete
        op = resp.json()
        if "name" in op:
            print(f"  Operation: {op['name']}")
            _wait_for_operation(token, op["name"])
        return True
    else:
        print(f"  FAILED ({resp.status_code}): {resp.text[:300]}")
        return False


def _wait_for_operation(token: str, op_name: str, max_wait: int = 60) -> None:
    """Poll a GCP long-running operation until done."""
    url = f"https://artifactregistry.googleapis.com/v1/{op_name}"
    start = time.time()
    while time.time() - start < max_wait:
        resp = requests.get(url, headers=_headers(token), timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("done"):
                print("  Operation completed!")
                return
        time.sleep(2)
    print(f"  WARNING: Operation polling timed out after {max_wait}s")


def create_gcs_bucket(token: str, dry_run: bool = False) -> bool:
    """Create GCS bucket."""
    url = f"https://storage.googleapis.com/storage/v1/b?project={PROJECT}"
    body = {
        "name": GCS_BUCKET,
        "location": REGION.upper(),
        "storageClass": "STANDARD",
    }

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Creating GCS bucket: gs://{GCS_BUCKET}")
    print(f"  URL: {url}")

    if dry_run:
        print("  Payload:", json.dumps(body, indent=2))
        return True

    resp = requests.post(url, headers=_headers(token), json=body, timeout=30)

    if resp.status_code == 409:
        print(f"  Bucket 'gs://{GCS_BUCKET}' already exists (409) — OK")
        return True
    elif resp.status_code == 200:
        print(f"  Bucket 'gs://{GCS_BUCKET}' created successfully!")
        return True
    else:
        print(f"  FAILED ({resp.status_code}): {resp.text[:300]}")
        return False


def verify_ar_repo(token: str) -> bool:
    """Verify AR repo exists."""
    url = (
        f"https://artifactregistry.googleapis.com/v1/"
        f"projects/{PROJECT}/locations/{REGION}/repositories/{AR_REPO}"
    )
    resp = requests.get(url, headers=_headers(token), timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        print(f"  AR repo verified: {data.get('name', 'unknown')}")
        print(f"  Format: {data.get('format', 'unknown')}")
        return True
    print(f"  AR repo NOT found ({resp.status_code})")
    return False


def verify_gcs_bucket(token: str) -> bool:
    """Verify GCS bucket exists."""
    url = f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}"
    resp = requests.get(url, headers=_headers(token), timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Bucket verified: gs://{data.get('name', 'unknown')}")
        print(f"  Location: {data.get('location', 'unknown')}")
        return True
    print(f"  Bucket NOT found ({resp.status_code})")
    return False


def main() -> None:
    """Provision GCP resources."""
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("  GCP Resource Provisioning (IPv4 bypass)")
    print(f"  Project: {PROJECT} | Region: {REGION}")
    print(f"  Mode: {'DRY-RUN' if dry_run else 'LIVE'}")
    print("=" * 60)

    # Get auth token
    print("\n[1/5] Acquiring access token...")
    token = _get_token()
    print(f"  Token acquired (length={len(token)})")

    # Create AR repo
    print("\n[2/5] Creating Artifact Registry repo...")
    ar_ok = create_ar_repo(token, dry_run)

    # Create GCS bucket
    print("\n[3/5] Creating GCS bucket...")
    gcs_ok = create_gcs_bucket(token, dry_run)

    # Verify
    print("\n[4/5] Verifying AR repo...")
    ar_verified = verify_ar_repo(token) if not dry_run else True

    print("\n[5/5] Verifying GCS bucket...")
    gcs_verified = verify_gcs_bucket(token) if not dry_run else True

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  AR repo ({AR_REPO}):  {'PASS' if ar_ok and ar_verified else 'FAIL'}")
    print(f"  GCS bucket ({GCS_BUCKET}): {'PASS' if gcs_ok and gcs_verified else 'FAIL'}")

    if ar_ok and gcs_ok and ar_verified and gcs_verified:
        print("\n  All resources provisioned successfully!")
        sys.exit(0)
    else:
        print("\n  Some resources failed — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
