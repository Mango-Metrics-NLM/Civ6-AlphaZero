#!/usr/bin/env python3
"""Monitor a Cloud Build job, polling until completion with step-by-step status.

Usage:
    python scripts/monitor_build.py BUILD_ID
    python scripts/monitor_build.py BUILD_ID --project hrmtrain
    python scripts/monitor_build.py BUILD_ID --poll-interval 30
    python scripts/monitor_build.py BUILD_ID --no-show-log
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset(
    {
        "SUCCESS",
        "FAILURE",
        "TIMEOUT",
        "CANCELLED",
        "EXPIRED",
        "INTERNAL_ERROR",
    }
)

POLL_INTERVAL_DEFAULT = 15
CLOUD_BUILD_API_URL = "https://cloudbuild.googleapis.com/v1/projects/{project}/builds/{build_id}"


def _get_auth_token() -> str:
    """Obtain a Bearer token using Application Default Credentials."""
    import google.auth
    import google.auth.transport.requests

    creds, _ = google.auth.default()
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token  # type: ignore[return-value]


def fetch_build(build_id: str, project: str, token: str) -> dict:
    """Fetch build status from Cloud Build REST API.

    Args:
        build_id: Cloud Build ID.
        project: GCP project ID.
        token: Bearer authentication token.

    Returns:
        Parsed JSON response as dict.

    Raises:
        urllib.error.URLError: On network failure.
    """
    url = CLOUD_BUILD_API_URL.format(project=project, build_id=build_id)
    req = urllib.request.Request(  # noqa: S310
        url, headers={"Authorization": f"Bearer {token}"}
    )
    resp = urllib.request.urlopen(req, timeout=30)  # noqa: S310
    return json.loads(resp.read())


def print_step_summary(steps: list[dict]) -> tuple[int, int]:
    """Print step-by-step status table.

    Args:
        steps: List of step dicts from Cloud Build API.

    Returns:
        Tuple of (passed_count, total_count).
    """
    passed = 0
    total = len(steps)
    for i, step in enumerate(steps):
        step_id = step.get("id", f"step-{i}")
        status = step.get("status", "UNKNOWN")
        marker = "OK" if status == "SUCCESS" else status
        if status == "SUCCESS":
            passed += 1
        print(f"  Step {i + 1}/{total}: {step_id:<35} [{marker}]")
    return passed, total


def fetch_build_log_tail(
    logs_bucket: str,
    build_id: str,
    token: str,
    lines: int = 50,
) -> str | None:
    """Fetch last N lines of the build log from GCS.

    Args:
        logs_bucket: GCS bucket URI for logs (gs://...).
        build_id: Cloud Build ID.
        token: Bearer authentication token.
        lines: Number of trailing lines to return.

    Returns:
        Log tail string, or None on failure.
    """
    log_url = f"{logs_bucket}/log-{build_id}.txt"
    if log_url.startswith("gs://"):
        log_url = log_url.replace("gs://", "https://storage.googleapis.com/", 1)

    try:
        req = urllib.request.Request(  # noqa: S310
            log_url, headers={"Authorization": f"Bearer {token}"}
        )
        resp = urllib.request.urlopen(req, timeout=30)  # noqa: S310
        text = resp.read().decode("utf-8", errors="replace")
        tail = text.strip().split("\n")[-lines:]
        return "\n".join(tail)
    except Exception as exc:
        logger.warning("Could not fetch build log: %s", exc)
        return None


def monitor(
    build_id: str,
    project: str,
    poll_interval: int = POLL_INTERVAL_DEFAULT,
    show_log_on_failure: bool = True,
    log_lines: int = 50,
) -> int:
    """Poll build until terminal state, return exit code.

    Args:
        build_id: Cloud Build ID to monitor.
        project: GCP project ID.
        poll_interval: Seconds between polls.
        show_log_on_failure: Whether to fetch logs on failure.
        log_lines: Number of log tail lines to show.

    Returns:
        0 on success, 1 on failure.
    """
    token = _get_auth_token()
    print(f"Monitoring build {build_id} in project {project}...")
    print(f"Poll interval: {poll_interval}s\n")

    while True:
        try:
            data = fetch_build(build_id, project, token)
        except Exception as exc:
            logger.error("Failed to fetch build: %s", exc)
            # Refresh token in case it expired
            try:
                token = _get_auth_token()
            except Exception:
                logger.debug("Token refresh failed, will retry next poll")
            time.sleep(poll_interval)
            continue

        build_status = data.get("status", "UNKNOWN")
        steps = data.get("steps", [])

        print(f"--- Build status: {build_status} ---")
        passed, total = print_step_summary(steps)
        print(f"  Progress: {passed}/{total} steps passed\n")

        if build_status in TERMINAL_STATUSES:
            # Final report
            if build_status == "SUCCESS" and passed == total:
                print(f"BUILD SUCCEEDED: {passed}/{total} steps passed")
                return 0

            print(f"BUILD {build_status}: {passed}/{total} steps passed")

            if "statusDetail" in data:
                print(f"Detail: {data['statusDetail']}")
            if "failureInfo" in data:
                print(f"Failure: {json.dumps(data['failureInfo'], indent=2)}")

            if show_log_on_failure and build_status != "SUCCESS":
                logs_bucket = data.get("logsBucket", "")
                if logs_bucket:
                    print(f"\n=== Build log (last {log_lines} lines) ===")
                    tail = fetch_build_log_tail(logs_bucket, build_id, token, log_lines)
                    if tail:
                        print(tail)

            return 0 if build_status == "SUCCESS" else 1

        time.sleep(poll_interval)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor a Cloud Build job until completion.",
    )
    parser.add_argument("build_id", help="Cloud Build ID to monitor")
    parser.add_argument(
        "--project",
        default=os.environ.get("CIV6AZ_VERTEX_PROJECT", "hrmtrain"),
        help="GCP project ID (default: $CIV6AZ_VERTEX_PROJECT or 'hrmtrain')",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL_DEFAULT,
        help=f"Seconds between polls (default: {POLL_INTERVAL_DEFAULT})",
    )
    parser.add_argument(
        "--no-show-log",
        dest="show_log",
        action="store_false",
        default=True,
        help="Suppress log output on failure",
    )
    parser.add_argument(
        "--log-lines",
        type=int,
        default=50,
        help="Number of log tail lines to show on failure (default: 50)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return monitor(
        build_id=args.build_id,
        project=args.project,
        poll_interval=args.poll_interval,
        show_log_on_failure=args.show_log,
        log_lines=args.log_lines,
    )


if __name__ == "__main__":
    sys.exit(main())
