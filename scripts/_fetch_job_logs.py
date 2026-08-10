"""Fetch Cloud Logging entries for a failed (or completed) Vertex AI job.

Usage:
    python scripts/_fetch_job_logs.py <job_id>
"""

from __future__ import annotations

import logging
import sys

from scripts._gcp_lib import (
    LOGGING_ENTRIES_URL,
    PROJECT,
    auth_headers,
    configure_script_logging,
)

logger = logging.getLogger(__name__)


def fetch_logs(job_id: str, page_size: int = 50) -> int:
    """Print log lines (chronological) for ``job_id``. Returns entry count."""
    import requests

    body = {
        "resourceNames": [f"projects/{PROJECT}"],
        "filter": (
            f'resource.type="ml_job" '
            f'AND resource.labels.job_id="{job_id}" '
            f'AND resource.labels.task_name="workerpool0-0"'
        ),
        "orderBy": "timestamp desc",
        "pageSize": page_size,
    }
    resp = requests.post(LOGGING_ENTRIES_URL, headers=auth_headers(), json=body, timeout=30)
    if resp.status_code != 200:
        logger.error("Log fetch failed: HTTP %d — %s", resp.status_code, resp.text[:300])
        return -1

    entries = resp.json().get("entries", [])
    if not entries:
        logger.info("No ml_job/workerpool0-0 entries for %s — broadening filter", job_id)
        body["filter"] = f'"{job_id}" AND NOT textPayload:"provisioning"'
        resp = requests.post(LOGGING_ENTRIES_URL, headers=auth_headers(), json=body, timeout=30)
        entries = resp.json().get("entries", []) if resp.status_code == 200 else []

    if not entries:
        logger.warning("No log entries found for %s", job_id)
        return 0

    for entry in reversed(entries):
        ts = entry.get("timestamp", "")[11:23]
        msg = entry.get("textPayload", "")
        if not msg:
            jp = entry.get("jsonPayload", {})
            msg = jp.get("message", str(jp) if jp else "")
        if msg:
            print(f"{ts} {msg[:400]}")
    return len(entries)


def main(argv: list[str]) -> int:
    configure_script_logging()
    if len(argv) < 2:
        print("Usage: python scripts/_fetch_job_logs.py <job_id>", file=sys.stderr)
        return 2
    return 0 if fetch_logs(argv[1]) >= 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
