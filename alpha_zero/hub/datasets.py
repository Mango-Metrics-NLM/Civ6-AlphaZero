"""Dataset export utilities for Hugging Face Hub.

Provides exporters for ELO tournament results and MicroCiv replay buffers.
Each exporter reads local data and produces a directory suitable for
``datasets.Dataset.push_to_hub()`` or local inspection.

**Import guard**: the ``datasets`` library is only required when calling
``push_to_hub()``.  The local export paths (``export_to_dir()``) work
with standard-library JSON / CSV only.

Example::

    from alpha_zero.hub.datasets import EloDatasetExporter

    exporter = EloDatasetExporter(source_dir=Path("./results"))
    exporter.export_to_dir(Path("./elo-dataset"))
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetExportConfig",
    "EloDatasetExporter",
    "ReplayDatasetExporter",
]


@dataclass
class DatasetExportConfig:
    """Configuration for dataset export.

    Attributes:
        source_dir: Directory containing source data files.
        output_dir: Target directory for the exported dataset.
        repo_id: Hub ``org/repo`` identifier for push_to_hub.
        dry_run: If True, export locally but don't push.
        max_samples: Maximum number of samples to include (None = all).
        description: Human-readable dataset description.
    """

    source_dir: Path = field(default_factory=lambda: Path("."))
    output_dir: Path = field(default_factory=lambda: Path("./dataset-export"))
    repo_id: str = ""
    dry_run: bool = True
    max_samples: int | None = None
    description: str = "Civ6 AlphaZero dataset"


class EloDatasetExporter:
    """Export ELO tournament results as a Hugging Face dataset.

    Scans *source_dir* for JSON/CSV files containing ELO ratings, win/loss
    matrices, and sequential testing results.

    Args:
        config: Export configuration.
    """

    def __init__(self, config: DatasetExportConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = DatasetExportConfig(**kwargs)
        self.config = config

    def _scan_elo_files(self) -> list[Path]:
        """Find ELO result files in the source directory."""
        patterns = ("*elo*", "*tournament*", "*rating*")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(self.config.source_dir.glob(f"**/{pattern}.json"))
            files.extend(self.config.source_dir.glob(f"**/{pattern}.csv"))
        return sorted(set(files))

    def _load_json_records(self, path: Path) -> list[dict[str, Any]]:
        """Load records from a JSON file (list or dict format)."""
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data  # type: ignore[return-value]
        if isinstance(data, dict):
            # Wrap single-record dicts or extract known keys.
            if "results" in data:
                return data["results"]  # type: ignore[no-any-return]
            return [data]
        return []

    def _load_csv_records(self, path: Path) -> list[dict[str, Any]]:
        """Load records from a CSV file."""
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records

    def export_to_dir(self, output_dir: Path | None = None) -> Path:
        """Export ELO data to a local directory as JSONL.

        Args:
            output_dir: Override output directory.

        Returns:
            Path to the output directory.
        """
        out = (output_dir or self.config.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        files = self._scan_elo_files()
        logger.info("Found %d ELO result files in %s", len(files), self.config.source_dir)

        all_records: list[dict[str, Any]] = []
        for f in files:
            if f.suffix == ".json":
                all_records.extend(self._load_json_records(f))
            elif f.suffix == ".csv":
                all_records.extend(self._load_csv_records(f))

        if self.config.max_samples is not None:
            all_records = all_records[: self.config.max_samples]

        # Write as JSONL.
        output_file = out / "elo_results.jsonl"
        with output_file.open("w", encoding="utf-8") as fp:
            for record in all_records:
                fp.write(json.dumps(record, default=str) + "\n")

        # Write metadata.
        meta = {
            "description": self.config.description,
            "num_records": len(all_records),
            "source_files": [str(f) for f in files],
        }
        (out / "dataset_info.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )

        logger.info("Exported %d ELO records to %s", len(all_records), out)
        return out

    def push_to_hub(self, repo_id: str | None = None) -> None:
        """Export and push the dataset to the Hugging Face Hub.

        Args:
            repo_id: Override Hub repo ID.

        Raises:
            ImportError: If the ``datasets`` library is not installed.
        """
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                "Pushing to Hub requires the 'datasets' extra. "
                "Install with:  pip install -e '.[datasets]'"
            ) from exc

        out = self.export_to_dir()
        jsonl_path = out / "elo_results.jsonl"

        records: list[dict[str, Any]] = []
        with jsonl_path.open(encoding="utf-8") as fp:
            for line in fp:
                records.append(json.loads(line))

        ds = Dataset.from_list(records)
        target = repo_id or self.config.repo_id
        if not target:
            msg = "repo_id must be set for push_to_hub"
            raise ValueError(msg)

        if self.config.dry_run:
            logger.info("[DRY RUN] Would push %d records to %s", len(records), target)
            return

        ds.push_to_hub(target)
        logger.info("Pushed %d records to %s", len(records), target)


class ReplayDatasetExporter:
    """Export MicroCiv replay buffer as a Hugging Face dataset.

    Scans *source_dir* for replay/trajectory files (.pt, .json, .pkl).

    Args:
        config: Export configuration.
    """

    def __init__(self, config: DatasetExportConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = DatasetExportConfig(**kwargs)
        self.config = config

    def _scan_replay_files(self) -> list[Path]:
        """Find replay files in the source directory."""
        extensions = (".pt", ".json", ".pkl", ".jsonl")
        files: list[Path] = []
        for ext in extensions:
            files.extend(self.config.source_dir.glob(f"**/*replay*{ext}"))
            files.extend(self.config.source_dir.glob(f"**/*trajectory*{ext}"))
            files.extend(self.config.source_dir.glob(f"**/*episode*{ext}"))
        return sorted(set(files))

    def export_to_dir(self, output_dir: Path | None = None) -> Path:
        """Export replay data to a local directory as JSONL.

        Args:
            output_dir: Override output directory.

        Returns:
            Path to the output directory.
        """
        out = (output_dir or self.config.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        files = self._scan_replay_files()
        logger.info("Found %d replay files in %s", len(files), self.config.source_dir)

        all_records: list[dict[str, Any]] = []
        for f in files:
            if f.suffix == ".json":
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    all_records.extend(data)
                else:
                    all_records.append(data)
            elif f.suffix == ".jsonl":
                with f.open(encoding="utf-8") as fp:
                    for line in fp:
                        all_records.append(json.loads(line))
            # .pt and .pkl files require torch/pickle and are skipped
            # in the basic export — they'd need schema negotiation.

        if self.config.max_samples is not None:
            all_records = all_records[: self.config.max_samples]

        output_file = out / "replays.jsonl"
        with output_file.open("w", encoding="utf-8") as fp:
            for record in all_records:
                fp.write(json.dumps(record, default=str) + "\n")

        meta = {
            "description": self.config.description,
            "num_records": len(all_records),
            "source_files": [str(f) for f in files],
        }
        (out / "dataset_info.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )

        logger.info("Exported %d replay records to %s", len(all_records), out)
        return out
