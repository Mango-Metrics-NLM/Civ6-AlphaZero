"""Fetch Vertex AI job details and logs via REST API.

Usage:
    python scripts/fetch_vertex_logs.py [JOB_ID]
    python scripts/fetch_vertex_logs.py --latest
    python scripts/fetch_vertex_logs.py --latest --count=3
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from scripts._gcp_lib import (
    LOGGING_ENTRIES_URL,
    PROJECT,
    VERTEX_API_BASE,
    auth_headers,
    configure_script_logging,
)

logger = logging.getLogger(__name__)


def get_job_details(job_id: str) -> dict[str, Any]:
    """Get full job details including error info."""
    import requests

    url = f"{VERTEX_API_BASE}/customJobs/{job_id}"
    resp = requests.get(url, headers=auth_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json()


def list_recent_jobs(count: int = 5) -> list[dict[str, Any]]:
    """List recent jobs (newest first)."""
    import requests

    url = f"{VERTEX_API_BASE}/customJobs?pageSize={count}"
    resp = requests.get(url, headers=auth_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json().get("customJobs", [])


def get_job_logs(job_id: str, max_entries: int = 100) -> list[dict[str, Any]]:
    """Fetch Cloud Logging entries for a specific Vertex AI job."""
    import requests

    filter_str = (
        f'resource.type="ml_job" '
        f'resource.labels.job_id="{job_id}" '
        f'resource.labels.project_id="{PROJECT}"'
    )
    payload = {
        "resourceNames": [f"projects/{PROJECT}"],
        "filter": filter_str,
        "orderBy": "timestamp desc",
        "pageSize": max_entries,
    }
    resp = requests.post(LOGGING_ENTRIES_URL, headers=auth_headers(), json=payload, timeout=30)
    if resp.status_code != 200:
        logger.warning("Log fetch failed (%d): %s", resp.status_code, resp.text[:300])
        return []
    return resp.json().get("entries", [])


def format_job_summary(job: dict[str, Any]) -> str:
    """Format job details as a readable multi-line summary."""
    lines = [
        f"Job: {job.get('displayName', '?')}",
        f"  State:   {job.get('state', '?')}",
        f"  Created: {job.get('createTime', '?')[:19]}",
        f"  Updated: {job.get('updateTime', '?')[:19]}",
    ]
    error = job.get("error")
    if error:
        lines.append(f"  Error Code: {error.get('code', '?')}")
        lines.append(f"  Error Msg:  {error.get('message', '?')[:300]}")
    spec = job.get("jobSpec", {}).get("workerPoolSpecs", [{}])[0]
    container = spec.get("containerSpec", {})
    lines.append(f"  Image:   {container.get('imageUri', '?')}")
    lines.append(f"  Command: {' '.join(container.get('command', []))}")
    lines.append(f"  Args:    {' '.join(container.get('args', []))}")
    return "\n".join(lines)


def format_log_entries(entries: list[dict[str, Any]]) -> str:
    """Format log entries in chronological order."""
    lines = []
    for entry in reversed(entries):
        ts = entry.get("timestamp", "?")[:19]
        severity = entry.get("severity", "?")
        payload = entry.get("textPayload", "")
        if not payload:
            json_payload = entry.get("jsonPayload", {})
            payload = json_payload.get("message", json.dumps(json_payload)[:200])
        lines.append(f"[{ts}] [{severity}] {payload}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """Fetch and display job details and logs."""
    configure_script_logging()
    logger.info("Authenticated successfully against project %s", PROJECT)

    count = 1
    job_id: str | None = None
    for arg in argv[1:]:
        if arg == "--latest":
            pass
        elif arg.startswith("--count="):
            count = int(arg.split("=", 1)[1])
        elif not arg.startswith("-"):
            job_id = arg

    if job_id:
        job = get_job_details(job_id)
        print(format_job_summary(job))
        print("\n--- LOGS ---")
        entries = get_job_logs(job_id)
        print(format_log_entries(entries) or "  (no log entries found)")
    else:
        for job in list_recent_jobs(count=count):
            jid = job.get("name", "").split("/")[-1] or "?"
            print(format_job_summary(job))
            print(f"\n--- LOGS for {jid} ---")
            entries = get_job_logs(jid)
            print(format_log_entries(entries) or "  (no log entries found)")
            print("\n" + "=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
