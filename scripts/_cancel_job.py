"""Cancel one or more Vertex AI custom jobs by ID.

Usage:
    python scripts/_cancel_job.py <job_id> [job_id2 ...]

Exits non-zero if any cancellation request returns HTTP != 200.
"""

from __future__ import annotations

import logging
import sys

# Imports `requests` indirectly; force_ipv4_only() runs at import time.
from scripts._gcp_lib import (
    PROJECT,
    REGION,
    auth_headers,
    configure_script_logging,
)

logger = logging.getLogger(__name__)


def cancel_job(job_id: str) -> bool:
    """Cancel a single Vertex job. Returns True on HTTP 200, False otherwise."""
    import requests

    url = (
        f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}"
        f"/locations/{REGION}/customJobs/{job_id}:cancel"
    )
    resp = requests.post(url, headers=auth_headers(), timeout=30)
    if resp.status_code == 200:
        logger.info("Cancelled job %s", job_id)
        print(f"OK   {job_id}")
        return True
    logger.error(
        "Cancel job %s failed: HTTP %d — %s",
        job_id,
        resp.status_code,
        resp.text[:300],
    )
    print(f"FAIL {job_id}  (HTTP {resp.status_code})")
    return False


def main(argv: list[str]) -> int:
    """Cancel every job listed on the CLI; return 0 only if all succeed."""
    configure_script_logging()
    if len(argv) < 2:
        print("Usage: python scripts/_cancel_job.py <job_id> [job_id2 ...]", file=sys.stderr)
        return 2
    failures = sum(1 for jid in argv[1:] if not cancel_job(jid))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
