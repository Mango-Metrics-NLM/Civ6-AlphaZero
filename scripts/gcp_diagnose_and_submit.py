"""GCP connectivity fix + Vertex AI job submission.

Bypasses all hanging SDK/CLI layers by:
1. Reading refresh token directly from ADC file
2. Exchanging it via oauth2.googleapis.com/token (requests + IPv4 patch)
3. Submitting job via Vertex AI REST API

Usage:
    python scripts/gcp_diagnose_and_submit.py [--dry-run]

Env vars (all optional, with sensible defaults):
    CIV6AZ_VERTEX_PROJECT  - GCP project (default: from ADC)
    CIV6AZ_VERTEX_REGION   - Region (default: us-central1)
    CIV6AZ_CONTAINER_IMAGE_URI - Container image
    CIV6AZ_GCS_BUCKET      - GCS bucket
    CIV6AZ_AGENT_ID        - Agent ID (default: muzero)
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import socket
import ssl
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# IPv4 enforcement — MUST be applied before any network imports
# Root cause: IPv6 routing broken on this machine (WinError 10051)
# ---------------------------------------------------------------------------
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(
    *args: Any, **kwargs: Any
) -> list[tuple[socket.AddressFamily, int, int, str, tuple[str, int]]]:
    """Filter DNS results to IPv4 only."""
    results = _orig_getaddrinfo(*args, **kwargs)
    return [r for r in results if r[0] == socket.AF_INET]


socket.getaddrinfo = _ipv4_only_getaddrinfo  # type: ignore[assignment]

# Now safe to import network libraries
import requests  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from env vars, backwards compatible defaults)
# ---------------------------------------------------------------------------
_ADC_PATH = pathlib.Path(
    os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        os.path.join(
            os.environ.get("APPDATA", ""), "gcloud", "application_default_credentials.json"
        ),
    )
)


def _load_config() -> dict[str, str]:
    """Load configuration from environment with defaults."""
    adc = _read_adc()
    project = os.environ.get("CIV6AZ_VERTEX_PROJECT", adc.get("quota_project_id", "hrmtrain"))
    region = os.environ.get("CIV6AZ_VERTEX_REGION", "us-central1")
    agent_id = os.environ.get("CIV6AZ_AGENT_ID", "muzero")

    return {
        "project": project,
        "region": region,
        "agent_id": agent_id,
        "gcs_bucket": os.environ.get("CIV6AZ_GCS_BUCKET", "civ6az-muzero-training"),
        "container_image": os.environ.get(
            "CIV6AZ_CONTAINER_IMAGE_URI",
            f"{region}-docker.pkg.dev/{project}/civ6az-muzero-world-model/civ6az-muzero:v16-full-fix",
        ),
        "service_account": os.environ.get(
            "CIV6AZ_VERTEX_SERVICE_ACCOUNT",
            "1019219508001-compute@developer.gserviceaccount.com",
        ),
        "api_host": f"{region}-aiplatform.googleapis.com",
        "wandb_project": os.environ.get("WANDB_PROJECT", "civ6az-muzero-world-model"),
        "wandb_enabled": os.environ.get("WANDB_ENABLED", "true"),
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def _step(name: str) -> None:
    logger.info("=" * 50)
    logger.info("  STEP: %s", name)
    logger.info("=" * 50)


def diagnose_dns(host: str) -> str:
    """Resolve API host to IPv4."""
    _step("DNS Resolution (IPv4 only)")
    ais = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    ip = ais[0][4][0]
    logger.info("  Resolved %s -> %s (%d results)", host, ip, len(ais))
    return ip


def diagnose_tcp(ip: str) -> float:
    """Test TCP connectivity, return latency in ms."""
    _step(f"TCP Connect to {ip}:443")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10)
    t = time.time()
    s.connect((ip, 443))
    latency = (time.time() - t) * 1000
    logger.info("  TCP OK in %.1fms", latency)
    s.close()
    return latency


def diagnose_tls(ip: str, host: str) -> str:
    """Test TLS handshake, return protocol version."""
    _step(f"TLS Handshake to {ip}:443")
    ctx = ssl.create_default_context()
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.settimeout(10)
    raw.connect((ip, 443))
    t = time.time()
    ss = ctx.wrap_socket(raw, server_hostname=host)
    latency = (time.time() - t) * 1000
    proto = ss.version() or "unknown"
    logger.info("  TLS OK in %.1fms (protocol: %s)", latency, proto)
    ss.close()
    return proto


def diagnose_https(api_base: str) -> int:
    """Test HTTPS connectivity (unauthenticated, expect 401)."""
    _step("HTTPS GET (unauthenticated)")
    url = f"{api_base}/customJobs?pageSize=1"
    t = time.time()
    resp = requests.get(url, timeout=15)
    latency = (time.time() - t) * 1000
    logger.info("  HTTPS %d in %.1fms", resp.status_code, latency)
    if resp.status_code == 401:
        logger.info("  (401 = expected — HTTPS path works)")
    return resp.status_code


# ---------------------------------------------------------------------------
# Auth — direct token exchange, no SDK
# ---------------------------------------------------------------------------
def _read_adc() -> dict[str, str]:
    """Read Application Default Credentials from disk."""
    if not _ADC_PATH.exists():
        msg = f"ADC not found at {_ADC_PATH}"
        raise FileNotFoundError(msg)
    return json.loads(_ADC_PATH.read_text(encoding="utf-8"))


def get_access_token() -> str:
    """Exchange ADC refresh token for an access token via OAuth2 endpoint."""
    _step("Auth Token (direct OAuth2 exchange)")
    adc = _read_adc()

    if adc.get("type") != "authorized_user":
        msg = f"Unsupported ADC type: {adc.get('type')}. Expected 'authorized_user'."
        raise ValueError(msg)

    t = time.time()
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
    latency = (time.time() - t) * 1000

    if resp.status_code != 200:
        msg = f"Token exchange failed ({resp.status_code}): {resp.text[:200]}"
        raise RuntimeError(msg)

    token = resp.json()["access_token"]
    logger.info("  Token acquired in %.1fms (length=%d)", latency, len(token))
    return token


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------
def build_job_payload(cfg: dict[str, str]) -> dict[str, Any]:
    """Build Vertex AI CustomJob payload."""
    agent_id = cfg["agent_id"]
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    env_vars = {
        "CIV6AZ_VERTEX_PROJECT": cfg["project"],
        "CIV6AZ_VERTEX_REGION": cfg["region"],
        "CIV6AZ_GCS_BUCKET": cfg["gcs_bucket"],
        "CIV6AZ_CONTAINER_IMAGE_URI": cfg["container_image"],
        "CIV6AZ_AGENT_ID": agent_id,
        "WANDB_PROJECT": cfg["wandb_project"],
        "WANDB_RUN_GROUP": agent_id,
        "WANDB_ENABLED": cfg["wandb_enabled"],
    }
    if wandb_api_key:
        env_vars["WANDB_API_KEY"] = wandb_api_key
    payload: dict[str, Any] = {
        "displayName": f"civ6az-mzwm-{agent_id}-orchestrator",
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "machineSpec": {"machineType": "n1-standard-4"},
                    "replicaCount": 1,
                    "containerSpec": {
                        "imageUri": cfg["container_image"],
                        "command": ["python", "-m", "alpha_zero.cloud.gcp.vertex_orchestrator"],
                        "args": [
                            f"--algorithm={agent_id}",
                            "--iterations=50",
                            "--mcts_simulations=50",
                            "--mcts_type=muzero",
                            "--workers=2",
                            "--games_per_worker=10",
                            "--epochs=10",
                            "--model_version_start=0",
                        ],
                        "env": [{"name": k, "value": v} for k, v in env_vars.items()],
                    },
                }
            ],
            # Explicitly set service account so the orchestrator container gets
            # a token with cloud-platform scope, enabling it to create child
            # Vertex AI jobs for self-play/training/evaluation.
            "serviceAccount": cfg["service_account"],
        },
    }
    return payload


def submit_job(token: str, cfg: dict[str, str], *, dry_run: bool = False) -> int:
    """Submit job via Vertex AI REST API. Returns HTTP status code (or 0 for dry-run)."""
    _step("Submit Vertex AI Custom Job")
    api_base = f"https://{cfg['api_host']}/v1/projects/{cfg['project']}/locations/{cfg['region']}"
    url = f"{api_base}/customJobs"
    payload = build_job_payload(cfg)

    if dry_run:
        logger.info("  DRY RUN — payload:\n%s", json.dumps(payload, indent=2))
        return 0

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    logger.info("  Response: %d", resp.status_code)

    if resp.status_code == 200:
        data = resp.json()
        logger.info("  *** JOB SUBMITTED SUCCESSFULLY ***")
        logger.info("  Name:  %s", data.get("name"))
        logger.info("  State: %s", data.get("state"))
        logger.info(
            "  Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=%s",
            cfg["project"],
        )
    else:
        logger.error("  FAILED: %s", resp.text[:500])
    return resp.status_code


# ---------------------------------------------------------------------------
# List existing jobs (smoke test)
# ---------------------------------------------------------------------------
def list_jobs(token: str, cfg: dict[str, str], limit: int = 5) -> list[dict[str, Any]]:
    """List recent Vertex AI custom jobs via REST API."""
    _step(f"List Custom Jobs (limit={limit})")
    api_base = f"https://{cfg['api_host']}/v1/projects/{cfg['project']}/locations/{cfg['region']}"
    url = f"{api_base}/customJobs?pageSize={limit}"

    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    if resp.status_code != 200:
        logger.error("  List failed: %d %s", resp.status_code, resp.text[:200])
        return []

    jobs = resp.json().get("customJobs", [])
    logger.info("  Found %d jobs", len(jobs))
    for job in jobs[:3]:
        logger.info(
            "    %s | %s | %s",
            job.get("displayName", "?"),
            job.get("state", "?"),
            job.get("createTime", "?")[:19],
        )
    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Run diagnostics, authenticate, and submit/list jobs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    dry_run = "--dry-run" in sys.argv
    list_only = "--list" in sys.argv

    logger.info("GCP CONNECTIVITY DIAGNOSTIC & JOB SUBMISSION")
    logger.info("Time: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    if dry_run:
        logger.info("*** DRY RUN MODE ***")

    try:
        cfg = _load_config()
        host = cfg["api_host"]
        api_base = f"https://{host}/v1/projects/{cfg['project']}/locations/{cfg['region']}"

        # Diagnostic sequence
        ip = diagnose_dns(host)
        diagnose_tcp(ip)
        diagnose_tls(ip, host)
        https_status = diagnose_https(api_base)

        if https_status not in (200, 401, 403):
            logger.error("HTTPS probe failed (%d). Cannot proceed.", https_status)
            return 1

        # Auth
        token = get_access_token()

        # Smoke test: list jobs
        jobs = list_jobs(token, cfg)
        logger.info("  Smoke test: %d jobs visible", len(jobs))

        if list_only:
            return 0

        # Submit
        result = submit_job(token, cfg, dry_run=dry_run)
        return 0 if result in (0, 200) else 1

    except Exception:
        logger.exception("FATAL error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
