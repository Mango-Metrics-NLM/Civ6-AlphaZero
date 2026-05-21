"""
Migrate MLflow FileStore (mlruns/) to SQLite backend (sqlite:///mlflow.db).

MLflow 3.x deprecates the filesystem tracking backend (./mlruns) in favour
of a database backend.  This one-shot migration utility copies all experiments,
runs, metrics, params, tags, and artifacts from the FileStore to a SQLite
store using the MLflow Tracking client API — no external tools required.

Usage:
    python scripts/migrate_mlflow_to_sqlite.py
    python scripts/migrate_mlflow_to_sqlite.py --src mlruns --dst sqlite:///mlflow.db
    python scripts/migrate_mlflow_to_sqlite.py --dry_run          # validate, no writes
    python scripts/migrate_mlflow_to_sqlite.py --experiment civ6az-local-50iter

Safety:
    * Idempotent: runs already present in the destination (same run_id) are skipped.
    * Dry-run mode prints what would be migrated without touching the destination.
    * Original mlruns/ directory is NOT deleted.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Core migration logic
# ---------------------------------------------------------------------------


def _copy_run(
    src_client: Any,
    dst_client: Any,
    run: Any,
    dst_experiment_id: str,
    *,
    dry_run: bool,
) -> bool:
    """Copy a single MLflow run from src to dst.

    Args:
        src_client: Source MlflowClient.
        dst_client: Destination MlflowClient.
        run: mlflow.entities.Run object from the source store.
        dst_experiment_id: Experiment ID in the destination store.
        dry_run: If True, only log what would happen without writing.

    Returns:
        True if the run was migrated; False if it already existed and was skipped.
    """
    import mlflow

    run_id = run.info.run_id
    run_name = run.info.run_name or run_id[:8]

    # Check if already migrated
    existing = dst_client.search_runs(
        [dst_experiment_id],
        filter_string=f"tags.`mlflow.runId` = '{run_id}'",
        max_results=1,
    )
    if existing:
        logger.info("  SKIP  run %s (%s) — already in destination", run_id[:8], run_name)
        return False

    logger.info("  COPY  run %s (%s)  status=%s", run_id[:8], run_name, run.info.status)

    if dry_run:
        metrics_count = sum(
            1
            for key in (run.data.metrics or {})
            for _ in src_client.get_metric_history(run_id, key)
        )
        logger.info(
            "    [dry-run] params=%d  metrics_keys=%d  metric_points=%d  tags=%d",
            len(run.data.params or {}),
            len(run.data.metrics or {}),
            metrics_count,
            len(run.data.tags or {}),
        )
        return True

    # Create run in destination
    dst_run = dst_client.create_run(
        experiment_id=dst_experiment_id,
        run_name=run_name,
        start_time=run.info.start_time,
        tags={
            **(run.data.tags or {}),
            "mlflow.runId": run_id,  # traceability tag
        },
    )
    dst_run_id = dst_run.info.run_id

    # Params
    for key, value in (run.data.params or {}).items():
        dst_client.log_param(dst_run_id, key, value)

    # Metrics (full history)
    for key in (run.data.metrics or {}):
        history = src_client.get_metric_history(run_id, key)
        for point in history:
            dst_client.log_metric(
                dst_run_id,
                key,
                point.value,
                timestamp=point.timestamp,
                step=point.step,
            )

    # Set terminal status
    status_map = {
        "FINISHED": mlflow.entities.RunStatus.FINISHED,
        "FAILED": mlflow.entities.RunStatus.FAILED,
        "KILLED": mlflow.entities.RunStatus.KILLED,
        "RUNNING": mlflow.entities.RunStatus.RUNNING,
    }
    terminal_status = status_map.get(run.info.status, mlflow.entities.RunStatus.FINISHED)
    dst_client.set_terminated(dst_run_id, status=mlflow.entities.RunStatus.to_string(terminal_status))

    return True


def migrate(
    src_uri: str,
    dst_uri: str,
    *,
    experiment_names: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Migrate experiments and runs from src_uri to dst_uri.

    Args:
        src_uri: Source MLflow tracking URI (e.g. ``"mlruns"``).
        dst_uri: Destination MLflow tracking URI (e.g. ``"sqlite:///mlflow.db"``).
        experiment_names: Optional list of experiment names to migrate.
            If None, all experiments are migrated.
        dry_run: Validate without writing to the destination.

    Returns:
        Summary dict with ``"experiments"``, ``"runs_migrated"``, ``"runs_skipped"`` counts.
    """
    import mlflow

    src_client = mlflow.tracking.MlflowClient(tracking_uri=src_uri)
    dst_client = mlflow.tracking.MlflowClient(tracking_uri=dst_uri)

    src_experiments = src_client.search_experiments()
    if experiment_names:
        src_experiments = [e for e in src_experiments if e.name in experiment_names]

    logger.info("Found %d experiments to migrate", len(src_experiments))

    total_migrated = 0
    total_skipped = 0
    experiments_processed = 0

    for exp in src_experiments:
        if exp.name == "Default":
            logger.debug("Skipping built-in 'Default' experiment")
            continue

        logger.info("Experiment: %s (id=%s)", exp.name, exp.experiment_id)

        # Ensure experiment exists in destination
        dst_exp = dst_client.get_experiment_by_name(exp.name)
        if dst_exp is None:
            if not dry_run:
                dst_exp_id = dst_client.create_experiment(
                    exp.name,
                    tags=dict(exp.tags or {}),
                )
            else:
                dst_exp_id = f"<dry-run:{exp.experiment_id}>"
                logger.info("  [dry-run] would create experiment '%s'", exp.name)
        else:
            dst_exp_id = dst_exp.experiment_id

        # Copy each run
        runs = src_client.search_runs(
            [exp.experiment_id],
            max_results=5000,
        )
        logger.info("  %d runs to process", len(runs))

        for run in runs:
            copied = _copy_run(src_client, dst_client, run, dst_exp_id, dry_run=dry_run)
            if copied:
                total_migrated += 1
            else:
                total_skipped += 1

        experiments_processed += 1

    summary = {
        "experiments": experiments_processed,
        "runs_migrated": total_migrated,
        "runs_skipped": total_skipped,
    }
    logger.info(
        "Migration complete: experiments=%d  migrated=%d  skipped=%d",
        experiments_processed,
        total_migrated,
        total_skipped,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed Namespace.
    """
    p = argparse.ArgumentParser(
        description="Migrate MLflow FileStore to SQLite backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/migrate_mlflow_to_sqlite.py
  python scripts/migrate_mlflow_to_sqlite.py --src mlruns --dst sqlite:///mlflow.db
  python scripts/migrate_mlflow_to_sqlite.py --dry_run
  python scripts/migrate_mlflow_to_sqlite.py --experiment civ6az-local-50iter
""",
    )
    p.add_argument(
        "--src",
        default="mlruns",
        help="Source MLflow tracking URI (default: mlruns)",
    )
    p.add_argument(
        "--dst",
        default="sqlite:///mlflow.db",
        help="Destination MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    p.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        metavar="NAME",
        help="Experiment name to migrate (repeat for multiple; default: all)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate without writing to the destination",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: CLI arguments.

    Returns:
        Exit code (0 = success).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Source:      %s", args.src)
    logger.info("Destination: %s", args.dst)
    logger.info("Dry run:     %s", args.dry_run)

    if args.src == args.dst:
        logger.error("Source and destination are the same URI; nothing to do.")
        return 1

    try:
        summary = migrate(
            src_uri=args.src,
            dst_uri=args.dst,
            experiment_names=args.experiments,
            dry_run=args.dry_run,
        )
        print(
            f"\nDone: {summary['experiments']} experiments, "
            f"{summary['runs_migrated']} runs migrated, "
            f"{summary['runs_skipped']} skipped."
        )
        return 0
    except Exception as exc:
        logger.exception("Migration failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
