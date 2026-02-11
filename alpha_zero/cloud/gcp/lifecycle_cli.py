"""
CLI for GCS Lifecycle Management operations.

Provides command-line interface for managing GCS data lifecycle,
including storage class transitions, cleanup operations, and cost analysis.

Usage:
    python -m alpha_zero.cloud.gcp.lifecycle_cli --status
    python -m alpha_zero.cloud.gcp.lifecycle_cli --apply-rules --dry-run
    python -m alpha_zero.cloud.gcp.lifecycle_cli --cleanup --iteration 100
    python -m alpha_zero.cloud.gcp.lifecycle_cli --estimate-savings --iteration 100
    python -m alpha_zero.cloud.gcp.lifecycle_cli --storage-summary
    python -m alpha_zero.cloud.gcp.lifecycle_cli --generate-rules --output rules.json

Environment Variables:
    CIV6AZ_VERTEX_BUCKET: GCS bucket name (required)
    CIV6AZ_VERTEX_PROJECT: GCP project ID (required)
    CIV6AZ_LIFECYCLE_ENABLED: Enable lifecycle management (default: true)
    CIV6AZ_LIFECYCLE_DRY_RUN: Enable dry-run mode (default: false)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from alpha_zero.cloud.gcp.lifecycle import (
    GCSLifecycleManager,
    LifecycleConfig,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    """Set up logging configuration.

    Args:
        verbose: Enable verbose logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_bucket() -> str:
    """Get bucket name from environment.

    Returns:
        GCS bucket name.

    Raises:
        SystemExit: If bucket not configured.
    """
    bucket = os.environ.get("CIV6AZ_VERTEX_BUCKET")
    if not bucket:
        print("Error: CIV6AZ_VERTEX_BUCKET environment variable not set", file=sys.stderr)
        sys.exit(1)
    return bucket


def cmd_status(args: argparse.Namespace) -> int:
    """Show lifecycle manager status.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    bucket = get_bucket()
    config = LifecycleConfig.from_env()

    if args.dry_run:
        config.dry_run = True

    manager = GCSLifecycleManager(bucket, config)
    stats = manager.get_statistics()

    print("=" * 60)
    print("GCS Lifecycle Manager Status")
    print("=" * 60)
    print(f"Bucket:           {stats['bucket']}")
    print(f"Enabled:          {stats['enabled']}")
    print(f"Dry Run:          {stats['dry_run']}")
    print(f"Cleanup Interval: Every {stats['cleanup_interval']} iterations")
    print()
    print("Retention Policies:")
    print("-" * 60)

    for name, policy in stats["policies"].items():
        print(f"\n  {name.upper()}:")
        print(f"    Prefix:         {policy['prefix']}")
        keep = policy["keep_recent_iterations"]
        print(f"    Keep Recent:    {keep if keep > 0 else 'All'} iterations")
        print(f"    Archive Class:  {policy['archive_storage_class']}")
        archive = policy["archive_after_days"]
        print(f"    Archive After:  {archive if archive > 0 else 'Never'} days")
        delete = policy["delete_after_days"]
        print(f"    Delete After:   {delete if delete > 0 else 'Never'} days")

    return 0


def cmd_storage_summary(args: argparse.Namespace) -> int:
    """Show storage summary by prefix and storage class.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    bucket = get_bucket()
    config = LifecycleConfig.from_env()
    manager = GCSLifecycleManager(bucket, config)

    print("Analyzing storage usage (this may take a while)...")
    summary = manager.get_storage_summary()

    print()
    print("=" * 70)
    print("Storage Summary by Prefix")
    print("=" * 70)

    total_bytes = 0
    total_count = 0

    for prefix, stats in sorted(summary.items()):
        print(f"\n{prefix}:")
        print(f"  Total Objects: {stats['total_count']:,}")
        print(f"  Total Size:    {_format_bytes(stats['total_bytes'])}")
        total_bytes += stats["total_bytes"]
        total_count += stats["total_count"]

        print("  By Storage Class:")
        for storage_class in ["STANDARD", "NEARLINE", "COLDLINE", "ARCHIVE"]:
            size = stats.get(storage_class, 0)
            if size > 0:
                print(f"    {storage_class:12} {_format_bytes(size)}")

    print()
    print("-" * 70)
    print(f"TOTAL: {total_count:,} objects, {_format_bytes(total_bytes)}")

    if args.output:
        _write_json(args.output, summary)
        print(f"\nJSON output written to: {args.output}")

    return 0


def cmd_estimate_savings(args: argparse.Namespace) -> int:
    """Estimate cost savings from lifecycle policies.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    bucket = get_bucket()
    config = LifecycleConfig.from_env()
    manager = GCSLifecycleManager(bucket, config)

    iteration = args.iteration or 100

    print(f"Estimating cost savings (assuming iteration {iteration})...")
    savings = manager.estimate_cost_savings(iteration)

    print()
    print("=" * 50)
    print("Cost Savings Estimate")
    print("=" * 50)
    print(f"Current Monthly Cost:   ${savings['current_monthly_cost_usd']:.2f}")
    print(f"Optimized Monthly Cost: ${savings['estimated_optimized_cost_usd']:.2f}")
    print(f"Estimated Savings:      ${savings['estimated_monthly_savings_usd']:.2f}")
    print(f"Savings Percentage:     {savings['savings_percentage']:.1f}%")

    if args.output:
        _write_json(args.output, savings)
        print(f"\nJSON output written to: {args.output}")

    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Run iteration-based cleanup.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    bucket = get_bucket()
    config = LifecycleConfig.from_env()

    if args.dry_run:
        config.dry_run = True

    manager = GCSLifecycleManager(bucket, config)

    if not args.iteration:
        print("Error: --iteration is required for cleanup", file=sys.stderr)
        return 1

    iteration = args.iteration

    if config.dry_run:
        print(f"[DRY RUN] Simulating cleanup for iteration {iteration}...")
    else:
        print(f"Running cleanup for iteration {iteration}...")

    result = manager.cleanup_old_iterations(iteration)

    print()
    print("=" * 50)
    print("Cleanup Result")
    print("=" * 50)
    print(f"Dry Run:         {result.dry_run}")
    print(f"Objects Archived: {result.archived_count:,}")
    print(f"Bytes Archived:   {_format_bytes(result.bytes_archived)}")
    print(f"Objects Deleted:  {result.deleted_count:,}")
    print(f"Bytes Deleted:    {_format_bytes(result.bytes_deleted)}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors[:10]:  # Limit to first 10 errors
            print(f"  - {error}")
        if len(result.errors) > 10:
            print(f"  ... and {len(result.errors) - 10} more errors")

    if args.output:
        _write_json(args.output, result.to_dict())
        print(f"\nJSON output written to: {args.output}")

    return 0 if not result.errors else 1


def cmd_generate_rules(args: argparse.Namespace) -> int:
    """Generate GCS lifecycle rules.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    bucket = get_bucket()
    config = LifecycleConfig.from_env()
    manager = GCSLifecycleManager(bucket, config)

    rules = manager.generate_lifecycle_rules()

    print(f"Generated {len(rules)} lifecycle rules:")
    print()

    gcs_rules = []
    for i, rule in enumerate(rules, 1):
        gcs_rule = rule.to_gcs_rule()
        gcs_rules.append(gcs_rule)

        action = gcs_rule["action"]
        condition = gcs_rule["condition"]

        action_str = action["type"]
        if action.get("storageClass"):
            action_str += f" -> {action['storageClass']}"

        print(f"Rule {i}:")
        print(f"  Action:    {action_str}")
        if condition.get("age"):
            print(f"  Age:       {condition['age']} days")
        if condition.get("matchesPrefix"):
            print(f"  Prefixes:  {condition['matchesPrefix']}")
        print()

    if args.output:
        _write_json(args.output, {"rules": gcs_rules})
        print(f"JSON output written to: {args.output}")

    return 0


def cmd_apply_rules(args: argparse.Namespace) -> int:
    """Apply lifecycle rules to bucket.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    bucket = get_bucket()
    config = LifecycleConfig.from_env()

    if args.dry_run:
        config.dry_run = True

    manager = GCSLifecycleManager(bucket, config)

    if config.dry_run:
        print("[DRY RUN] Would apply the following lifecycle rules:")
    else:
        print("Applying lifecycle rules to bucket...")

    rules = manager.generate_lifecycle_rules()
    for i, rule in enumerate(rules, 1):
        gcs_rule = rule.to_gcs_rule()
        action = gcs_rule["action"]
        condition = gcs_rule["condition"]
        print(f"  {i}. {action['type']}", end="")
        if action.get("storageClass"):
            print(f" -> {action['storageClass']}", end="")
        if condition.get("matchesPrefix"):
            print(f" for {condition['matchesPrefix'][0]}", end="")
        if condition.get("age"):
            print(f" after {condition['age']} days", end="")
        print()

    if not config.dry_run:
        if not args.force:
            response = input("\nProceed with applying rules? [y/N]: ")
            if response.lower() != "y":
                print("Aborted.")
                return 0

    try:
        success = manager.apply_bucket_lifecycle_rules()
        if success:
            if config.dry_run:
                print("\n[DRY RUN] Rules would be applied successfully.")
            else:
                print("\nLifecycle rules applied successfully.")
            return 0
        else:
            print("\nFailed to apply rules.", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"\nError applying rules: {e}", file=sys.stderr)
        return 1


def cmd_show_rules(args: argparse.Namespace) -> int:
    """Show current bucket lifecycle rules.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    bucket = get_bucket()
    config = LifecycleConfig.from_env()
    manager = GCSLifecycleManager(bucket, config)

    print(f"Fetching current lifecycle rules from bucket: {bucket}")
    rules = manager.get_current_lifecycle_rules()

    if not rules:
        print("No lifecycle rules configured on bucket.")
        return 0

    print()
    print("=" * 50)
    print(f"Current Lifecycle Rules ({len(rules)})")
    print("=" * 50)

    for i, rule in enumerate(rules, 1):
        print(f"\nRule {i}:")
        print(f"  {json.dumps(rule, indent=4)}")

    if args.output:
        _write_json(args.output, {"rules": rules})
        print(f"\nJSON output written to: {args.output}")

    return 0


def _format_bytes(size: int | float) -> str:
    """Format bytes as human-readable string.

    Args:
        size: Size in bytes.

    Returns:
        Human-readable string.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024.0:
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} PB"


def _write_json(path: str, data: Any) -> None:
    """Write data to JSON file.

    Args:
        path: Output file path.
        data: Data to write.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="lifecycle",
        description="GCS Lifecycle Management CLI for Civ6 AlphaZero",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show status
  python -m alpha_zero.cloud.gcp.lifecycle_cli --status

  # Show storage summary
  python -m alpha_zero.cloud.gcp.lifecycle_cli --storage-summary

  # Estimate cost savings
  python -m alpha_zero.cloud.gcp.lifecycle_cli --estimate-savings --iteration 100

  # Run cleanup (dry run)
  python -m alpha_zero.cloud.gcp.lifecycle_cli --cleanup --iteration 100 --dry-run

  # Apply lifecycle rules (dry run)
  python -m alpha_zero.cloud.gcp.lifecycle_cli --apply-rules --dry-run

  # Generate rules and save to file
  python -m alpha_zero.cloud.gcp.lifecycle_cli --generate-rules --output rules.json

Environment Variables:
  CIV6AZ_VERTEX_BUCKET    GCS bucket name (required)
  CIV6AZ_VERTEX_PROJECT   GCP project ID (required)
  CIV6AZ_LIFECYCLE_ENABLED Enable lifecycle management (default: true)
  CIV6AZ_LIFECYCLE_DRY_RUN Enable dry-run mode (default: false)
""",
    )

    # Commands (mutually exclusive)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument(
        "--status",
        action="store_true",
        help="Show lifecycle manager status and policies",
    )
    commands.add_argument(
        "--storage-summary",
        action="store_true",
        help="Show storage usage by prefix and storage class",
    )
    commands.add_argument(
        "--estimate-savings",
        action="store_true",
        help="Estimate cost savings from lifecycle policies",
    )
    commands.add_argument(
        "--cleanup",
        action="store_true",
        help="Run iteration-based cleanup (requires --iteration)",
    )
    commands.add_argument(
        "--generate-rules",
        action="store_true",
        help="Generate GCS lifecycle rules from config",
    )
    commands.add_argument(
        "--apply-rules",
        action="store_true",
        help="Apply lifecycle rules to bucket",
    )
    commands.add_argument(
        "--show-rules",
        action="store_true",
        help="Show current bucket lifecycle rules",
    )

    # Options
    parser.add_argument(
        "--iteration",
        type=int,
        help="Current training iteration (for cleanup and savings estimation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate operations without making changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for JSON results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    try:
        if args.status:
            return cmd_status(args)
        elif args.storage_summary:
            return cmd_storage_summary(args)
        elif args.estimate_savings:
            return cmd_estimate_savings(args)
        elif args.cleanup:
            return cmd_cleanup(args)
        elif args.generate_rules:
            return cmd_generate_rules(args)
        elif args.apply_rules:
            return cmd_apply_rules(args)
        elif args.show_rules:
            return cmd_show_rules(args)
        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
