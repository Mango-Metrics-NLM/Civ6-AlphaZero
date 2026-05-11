"""Poll the status of a Cloud Build run.

Usage:
    python scripts/_poll_build.py <build_id>
"""

from __future__ import annotations

import logging
import sys

from scripts._gcp_lib import (
    CLOUDBUILD_API_BASE,
    auth_headers,
    configure_script_logging,
)

logger = logging.getLogger(__name__)


def poll(build_id: str) -> int:
    """Print build status / steps / log URL. Returns 0 on success."""
    import requests  # noqa: PLC0415

    url = f"{CLOUDBUILD_API_BASE}/builds/{build_id}"
    resp = requests.get(url, headers=auth_headers(), timeout=15)
    if resp.status_code != 200:
        logger.error("Build lookup failed: HTTP %d — %s", resp.status_code, resp.text[:300])
        return 1

    data = resp.json()
    print(f"Build:    {build_id}")
    print(f"Status:   {data.get('status', '?')}")
    print(f"Started:  {data.get('startTime', 'N/A')}")
    print(f"Finished: {data.get('finishTime', 'N/A')}")

    for i, step in enumerate(data.get("steps", [])):
        print(f"Step {i}: {step.get('name')} -> {step.get('status', 'PENDING')}")

    if "statusDetail" in data:
        print(f"Detail:   {data['statusDetail']}")
    if "logUrl" in data:
        print(f"Logs:     {data['logUrl']}")
    return 0


def main(argv: list[str]) -> int:
    configure_script_logging()
    if len(argv) < 2:
        print("Usage: python scripts/_poll_build.py <build_id>", file=sys.stderr)
        return 2
    return poll(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
