#!/usr/bin/env python3
"""Export datasets to the Hugging Face Hub.

Supports ELO tournament results and replay buffer datasets.
**Dry-run by default**: pass ``--publish`` to actually push to Hub.

Usage::

    # Export ELO results (dry-run)
    python scripts/export_dataset.py \\
        --type elo \\
        --source-path ./results/ \\
        --output-dir ./exported-elo/

    # Push to Hub
    python scripts/export_dataset.py \\
        --type elo \\
        --source-path ./results/ \\
        --repo-id ianshank/civ6-elo-tournament \\
        --publish
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    from alpha_zero.hub.constants import HF_ORG
    from alpha_zero.hub.datasets import (
        DatasetExportConfig,
        EloDatasetExporter,
        ReplayDatasetExporter,
    )

    _HUB_AVAILABLE = True
except ImportError:
    _HUB_AVAILABLE = False
    HF_ORG = os.environ.get("HF_ORG", "ianshank")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_DATASET_TYPES = {
    "elo": {
        "class": "EloDatasetExporter",
        "description": "ELO tournament ratings and win/loss matrices",
    },
    "replay": {
        "class": "ReplayDatasetExporter",
        "description": "MicroCiv replay buffer episodes",
    },
}


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export datasets to the Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=sorted(_DATASET_TYPES),
        dest="dataset_type",
        help="Dataset type to export",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        required=True,
        help="Directory containing source data files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local output directory (default: ./dataset-export/)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hub repo in 'org/repo' format for push_to_hub",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to include",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        default=False,
        help="Push to Hub (default: local export only)",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Dataset description",
    )

    args = parser.parse_args()

    if not args.source_path.exists():
        print(f"ERROR: Source path not found: {args.source_path}")
        return 1

    repo_id = args.repo_id or f"{HF_ORG}/civ6-{args.dataset_type}-dataset"
    description = args.description or _DATASET_TYPES[args.dataset_type]["description"]

    print("=" * 60)
    print(f"  Dataset Export: {_DATASET_TYPES[args.dataset_type]['description']}")
    print("=" * 60)
    print(f"  Type:           {args.dataset_type}")
    print(f"  Source:         {args.source_path}")
    print(f"  Output:         {args.output_dir or '(default)'}")
    print(f"  Repo ID:        {repo_id}")
    print(f"  Publish:        {args.publish}")
    print()

    if not _HUB_AVAILABLE:
        print("ERROR: Hub extra not installed. Run: pip install -e '.[hub]'")
        return 1

    config = DatasetExportConfig(
        source_dir=args.source_path,
        output_dir=args.output_dir or Path.cwd() / "dataset-export",
        repo_id=repo_id,
        dry_run=not args.publish,
        max_samples=args.max_samples,
        description=description,
    )

    # Use registry to select exporter.
    exporter_map: dict[str, type] = {
        "elo": EloDatasetExporter,
        "replay": ReplayDatasetExporter,
    }
    exporter = exporter_map[args.dataset_type](config)
    out = exporter.export_to_dir()

    print(f"\n  Exported to:    {out}")

    if args.publish:
        if hasattr(exporter, "push_to_hub"):
            exporter.push_to_hub(repo_id)
            print(f"  Published to:   https://huggingface.co/datasets/{repo_id}")
        else:
            print("  WARNING: This exporter does not support push_to_hub yet")
    else:
        print("  [LOCAL ONLY] Use --publish to push to Hub")

    return 0


if __name__ == "__main__":
    sys.exit(main())
