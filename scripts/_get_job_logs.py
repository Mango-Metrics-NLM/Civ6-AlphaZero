"""Fetch container logs for the most recent Vertex AI custom job.

Usage:
    python scripts/_get_job_logs.py

Tries multiple Cloud Logging filters in order of specificity, prints the
first non-empty result.
"""

from __future__ import annotations

import json
import logging
import sys

from scripts._gcp_lib import (
    LOGGING_ENTRIES_URL,
    PROJECT,
    VERTEX_API_BASE,
    auth_headers,
    configure_script_logging,
)

logger = logging.getLogger(__name__)


def _try_filter(filter_str: str, page_size: int = 20) -> list[dict]:
    """Run one log query; empty list if non-200 or no entries."""
    import requests

    # Cloud Logging entries:list expects ``resourceNames`` in the form
    # ``projects/<project-id>``. The legacy ``projectIds`` field is rejected
    # by current API versions and silently returns an empty result set;
    # ``fetch_vertex_logs.py`` and ``_fetch_job_logs.py`` already use the
    # correct shape — keep this in sync.
    body = {
        "resourceNames": [f"projects/{PROJECT}"],
        "filter": filter_str,
        "orderBy": "timestamp desc",
        "pageSize": page_size,
    }
    resp = requests.post(LOGGING_ENTRIES_URL, headers=auth_headers(), json=body, timeout=15)
    if resp.status_code != 200:
        logger.debug("Log filter rejected: HTTP %d — %s", resp.status_code, resp.text[:200])
        return []
    return resp.json().get("entries", [])


def main(_argv: list[str]) -> int:
    configure_script_logging()

    import requests

    # Look up the most recent job
    url = f"{VERTEX_API_BASE}/customJobs"
    r = requests.get(url, headers=auth_headers(), params={"pageSize": 1}, timeout=15)
    r.raise_for_status()
    jobs = r.json().get("customJobs", [])
    if not jobs:
        logger.warning("No recent jobs found in project %s", PROJECT)
        return 1

    job = jobs[0]
    job_id = job["name"].split("/")[-1]
    display = job.get("displayName", "?")
    print(f"Job:    {display} ({job_id})")
    print(f"State:  {job.get('state', '?')}")
    print(f"Image:  {job['jobSpec']['workerPoolSpecs'][0]['containerSpec'].get('imageUri', '?')}")
    err = job.get("error", {})
    if err:
        print(f"Error:  {json.dumps(err, indent=2)}")
    print()

    # Try filters in decreasing specificity. Stop as soon as one returns.
    filters_to_try = [
        f'resource.type="ml_job" resource.labels.job_id="{job_id}"',
        f'resource.type="ml_job" textPayload:"{display}"',
        'resource.type="ml_job"',
    ]
    for i, fstr in enumerate(filters_to_try):
        entries = _try_filter(fstr)
        print(f"Filter {i + 1}: {fstr[:80]}... -> {len(entries)} entries")
        for e in entries[:5]:
            ts = e.get("timestamp", "?")[:19]
            payload = e.get("textPayload", "") or json.dumps(e.get("jsonPayload", {}))[:200]
            print(f"  [{ts}] {payload[:200]}")
        if entries:
            return 0
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
