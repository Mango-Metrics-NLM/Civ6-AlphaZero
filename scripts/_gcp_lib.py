"""Shared helpers for the GCP debug-utility scripts in this directory.

Centralises three patterns that were duplicated across ~8 ad-hoc scripts
(``_cancel_job.py``, ``_fetch_job_logs.py``, ``_fetch_worker_logs.py``,
``_get_job_logs.py``, ``_poll_build.py``, ``fetch_vertex_logs.py``,
``run_tests_ipv4.py``):

1. **IPv4-only DNS** (:func:`force_ipv4_only`) — Windows IPv6 routing
   intermittently breaks (WinError 10051), making ``import torch`` and any
   ``requests`` call hang. Filtering ``socket.getaddrinfo`` to ``AF_INET``
   results before any networked module is imported avoids the hang.
   **Call this BEFORE importing ``requests`` / ``torch`` / google SDKs.**

2. **ADC refresh-token exchange** (:func:`get_access_token`) — the gcloud
   ADC JSON at ``%APPDATA%/gcloud/application_default_credentials.json``
   holds a refresh token; this function POSTs it to ``oauth2.googleapis.com``
   and returns the ``access_token`` string.

3. **Project / region defaults** (:data:`PROJECT`, :data:`REGION`) — read
   from ``CIV6AZ_VERTEX_PROJECT`` / ``CIV6AZ_VERTEX_REGION``, with the
   project defaults documented in MEMORY.md.

The scripts in this directory are *debug utilities*, not part of the
training package; this module deliberately lives here (under ``scripts/``)
rather than in ``alpha_zero.cloud.gcp`` so it isn't pulled into the training
container.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import socket
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project / region defaults
# ---------------------------------------------------------------------------

#: GCP project ID (override via ``CIV6AZ_VERTEX_PROJECT``).
PROJECT: str = os.environ.get("CIV6AZ_VERTEX_PROJECT", "hrmtrain")
#: GCP region (override via ``CIV6AZ_VERTEX_REGION``).
REGION: str = os.environ.get("CIV6AZ_VERTEX_REGION", "us-central1")
#: Vertex AI API base URL for the configured project + region.
VERTEX_API_BASE: str = (
    f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}"
)
#: Cloud Logging entries:list endpoint.
LOGGING_ENTRIES_URL: str = "https://logging.googleapis.com/v2/entries:list"
#: Cloud Build API base URL for the configured project.
CLOUDBUILD_API_BASE: str = f"https://cloudbuild.googleapis.com/v1/projects/{PROJECT}"

# ---------------------------------------------------------------------------
# IPv4 enforcement
# ---------------------------------------------------------------------------

_ipv4_installed: bool = False


def force_ipv4_only() -> None:
    """Patch :func:`socket.getaddrinfo` to return AF_INET results only.

    Idempotent — safe to call from multiple scripts in the same process.
    Must be called BEFORE the first import of any module that issues
    DNS lookups during import time (e.g. ``torch``, ``requests``,
    google-cloud SDKs).
    """
    global _ipv4_installed
    if _ipv4_installed:
        return
    original = socket.getaddrinfo

    def _ipv4_only(*args: Any, **kwargs: Any) -> list[Any]:
        return [r for r in original(*args, **kwargs) if r[0] == socket.AF_INET]

    socket.getaddrinfo = _ipv4_only  # type: ignore[assignment]
    _ipv4_installed = True
    logger.debug("IPv4-only DNS filter installed")


# Auto-install on import so a single ``import _gcp_lib`` BEFORE other
# imports is enough to enable the patch. Scripts that want a different
# ordering can ``from scripts._gcp_lib import force_ipv4_only`` and call
# it manually instead.
force_ipv4_only()


# ---------------------------------------------------------------------------
# ADC token helper
# ---------------------------------------------------------------------------


def _adc_path() -> pathlib.Path:
    """Resolve the Application Default Credentials JSON path.

    Honours ``GOOGLE_APPLICATION_CREDENTIALS`` if set; otherwise falls
    back to the gcloud SDK's Windows default location.
    """
    return pathlib.Path(
        os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
            os.path.join(
                os.environ.get("APPDATA", ""),
                "gcloud",
                "application_default_credentials.json",
            ),
        )
    )


def get_access_token(timeout: float = 15.0) -> str:
    """Exchange the local ADC refresh token for a short-lived access token.

    Args:
        timeout: HTTP timeout in seconds for the token POST.

    Returns:
        A Bearer token string suitable for ``Authorization: Bearer <token>``.

    Raises:
        FileNotFoundError: If the ADC JSON file is missing.
        KeyError: If the ADC JSON lacks the expected refresh-token fields
            (i.e. it's a service-account key, not a user ADC file).
        requests.HTTPError: If the OAuth2 endpoint rejects the refresh.
    """
    # Local import: requests must be imported AFTER force_ipv4_only() runs,
    # which it does at module-import time above.
    import requests

    adc_file = _adc_path()
    if not adc_file.exists():
        raise FileNotFoundError(
            f"ADC credentials not found at {adc_file}. "
            "Run `gcloud auth application-default login` or set "
            "GOOGLE_APPLICATION_CREDENTIALS."
        )
    adc = json.loads(adc_file.read_text(encoding="utf-8"))
    try:
        payload = {
            "client_id": adc["client_id"],
            "client_secret": adc["client_secret"],
            "refresh_token": adc["refresh_token"],
            "grant_type": "refresh_token",
        }
    except KeyError as exc:
        raise KeyError(
            f"ADC file {adc_file} is missing field {exc.args[0]!r}. "
            "Expected a user-mode ADC file from `gcloud auth "
            "application-default login`, not a service-account key."
        ) from exc
    resp = requests.post("https://oauth2.googleapis.com/token", data=payload, timeout=timeout)
    resp.raise_for_status()
    token = resp.json()["access_token"]
    logger.debug("Obtained access token (length=%d)", len(token))
    return token


def auth_headers() -> dict[str, str]:
    """Build the standard ``Authorization`` + ``Content-Type`` headers."""
    return {"Authorization": f"Bearer {get_access_token()}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Logging bootstrap
# ---------------------------------------------------------------------------


def configure_script_logging(level: int = logging.INFO) -> None:
    """Configure a simple stderr logger for utility scripts.

    Scripts in this directory print their *results* to stdout (so they
    pipe nicely) and their *diagnostics* via logger to stderr.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


__all__ = [
    "CLOUDBUILD_API_BASE",
    "LOGGING_ENTRIES_URL",
    "PROJECT",
    "REGION",
    "VERTEX_API_BASE",
    "auth_headers",
    "configure_script_logging",
    "force_ipv4_only",
    "get_access_token",
]
