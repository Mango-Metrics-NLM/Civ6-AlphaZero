#!/usr/bin/env python3
"""Clean up specific trajectory iterations from GCS.

Usage:
    python scripts/cleanup_trajectories.py --iterations 28-33 --list-only
    python scripts/cleanup_trajectories.py --iterations 28-33 --dry-run
    python scripts/cleanup_trajectories.py --iterations 28,29,30 --delete
    python scripts/cleanup_trajectories.py --iterations 28-33 --delete --force
    python scripts/cleanup_trajectories.py --iterations 28-33 --delete --agent-id ab_patch_embed
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure project root is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpha_zero.cloud.gcp.config import GCPConfig, GCSPathConfig
from alpha_zero.cloud.gcp.storage import GCSTrajectoryStorage

logger = logging.getLogger(__name__)


def parse_iteration_range(value: str) -> list[int]:
    """Parse iteration specification into a list of ints.

    Supports dash ranges ('28-33') and comma-separated lists ('28,29,30').

    Args:
        value: Iteration range string.

    Returns:
        Sorted list of iteration numbers.

    Raises:
        argparse.ArgumentTypeError: On invalid input.
    """
    iterations: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            bounds = part.split("-", 1)
            try:
                start, end = int(bounds[0]), int(bounds[1])
            except ValueError as exc:
                msg = f"Invalid range '{part}': {exc}"
                raise argparse.ArgumentTypeError(msg) from exc
            if start > end:
                msg = f"Invalid range '{part}': start > end"
                raise argparse.ArgumentTypeError(msg)
            iterations.extend(range(start, end + 1))
        else:
            try:
                iterations.append(int(part))
            except ValueError as exc:
                msg = f"Invalid iteration '{part}': {exc}"
                raise argparse.ArgumentTypeError(msg) from exc
    return sorted(set(iterations))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean up specific trajectory iterations from GCS.",
    )
    parser.add_argument(
        "--iterations",
        required=True,
        type=parse_iteration_range,
        help="Iteration range '28-33' or comma list '28,29,30'",
    )

    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--list-only",
        action="store_true",
        help="Only list blobs per iteration (no deletions)",
    )
    action_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without acting",
    )
    action_group.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete the blobs",
    )

    parser.add_argument(
        "--agent-id",
        default=None,
        help="Agent ID scope (e.g., 'ab_patch_embed'). Omit for default prefix.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt (for --delete)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build config from environment
    project_id = os.environ.get("CIV6AZ_VERTEX_PROJECT")
    bucket = os.environ.get("CIV6AZ_GCS_BUCKET")
    if not project_id or not bucket:
        logging.error(
            "CIV6AZ_VERTEX_PROJECT and CIV6AZ_GCS_BUCKET must both be set; there are no "
            "built-in defaults (a default bucket name is a takeover vector)."
        )
        return 2
    config = GCPConfig(
        project_id=project_id,
        region=os.environ.get("CIV6AZ_VERTEX_REGION", "us-central1"),
        bucket=bucket,
        paths=GCSPathConfig(agent_id=args.agent_id),
    )
    storage = GCSTrajectoryStorage(config, compress=True)

    iterations: list[int] = args.iterations

    if args.list_only:
        stats = storage.list_iteration_stats(iterations)
        for entry in stats:
            print(f"  iter_{entry['iteration']:03d}: {entry['blob_count']} blob(s)")
            for blob in entry["blob_paths"]:
                print(f"    gs://{config.bucket}/{blob}")
        total = sum(e["blob_count"] for e in stats)
        print(f"\nTotal: {total} blob(s) across {len(iterations)} iterations")
        return 0

    if args.dry_run:
        result = storage.delete_iterations(iterations, dry_run=True)
        print(
            f"\n[DRY RUN] Would delete {result['deleted_count']} blob(s), "
            f"{len(result['skipped_iterations'])} empty iterations skipped"
        )
        return 0

    # --delete mode
    if not args.force:
        answer = input(f"Delete all blobs for iterations {iterations}? [y/N] ")
        if answer.lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    result = storage.delete_iterations(iterations, dry_run=False)
    print(
        f"\nDeleted {result['deleted_count']} blob(s), "
        f"{len(result['skipped_iterations'])} empty iterations skipped, "
        f"{len(result['errors'])} error(s)"
    )
    if result["errors"]:
        for err in result["errors"]:
            print(f"  ERROR: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
