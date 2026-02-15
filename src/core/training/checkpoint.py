"""Checkpoint management for saving and restoring training state.

Provides a filesystem-based checkpoint manager that tracks versions,
maintains a 'best' pointer, and stores JSON metadata alongside each
checkpoint.  Works with any object satisfying ``CheckpointableProtocol``.

Layer 3 -- depends on config, types, logging.
"""
from __future__ import annotations

__all__ = ["CheckpointManager", "CheckpointMetadata"]

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.core.config import SDKConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CheckpointMetadata:
    """Metadata stored alongside each checkpoint."""

    iteration: int
    timestamp: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    is_best: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        return cls(
            iteration=data.get("iteration", 0),
            timestamp=data.get("timestamp", ""),
            metrics=data.get("metrics", {}),
            config_snapshot=data.get("config_snapshot", {}),
            is_best=data.get("is_best", False),
            extra=data.get("extra", {}),
        )


class CheckpointManager:
    """Manages checkpoint saving, loading, and version tracking.

    Checkpoints are stored in a directory structure::

        output_dir/
            checkpoints/
                iteration_0000/
                    checkpoint.bin
                    metadata.json
                iteration_0010/
                    checkpoint.bin
                    metadata.json
                best/  -> symlink or copy of best iteration
                latest/  -> symlink or copy of latest iteration
                manifest.json  (tracks all checkpoints)

    Args:
        output_dir: Root directory for checkpoint storage.
        keep_last_n: Maximum number of checkpoints to retain (0 = unlimited).
        checkpoint_interval: Iteration interval between automatic saves.
    """

    _CHECKPOINT_FILE = "checkpoint.bin"
    _METADATA_FILE = "metadata.json"
    _MANIFEST_FILE = "manifest.json"
    _BEST_DIR = "best"
    _LATEST_DIR = "latest"

    def __init__(
        self,
        output_dir: str | Path,
        keep_last_n: int = 0,
        checkpoint_interval: int = 10,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._checkpoints_dir = self._output_dir / "checkpoints"
        self._keep_last_n = keep_last_n
        self._checkpoint_interval = checkpoint_interval
        self._best_metric_value: float | None = None
        self._best_iteration: int | None = None
        self._manifest: list[dict[str, Any]] = []

        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._load_manifest()

    @classmethod
    def from_config(cls, config: SDKConfig) -> CheckpointManager:
        """Create a CheckpointManager from SDK configuration."""
        return cls(
            output_dir=config.output_dir,
            checkpoint_interval=config.training.checkpoint_interval,
        )

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def checkpoint_interval(self) -> int:
        return self._checkpoint_interval

    @property
    def best_iteration(self) -> int | None:
        return self._best_iteration

    @property
    def all_checkpoints(self) -> list[dict[str, Any]]:
        """Return metadata for all saved checkpoints."""
        return list(self._manifest)

    def should_save(self, iteration: int) -> bool:
        """Check if a checkpoint should be saved at this iteration."""
        return self._checkpoint_interval > 0 and (
            iteration % self._checkpoint_interval == 0
        )

    def iteration_dir(self, iteration: int) -> Path:
        """Return the directory path for a specific iteration checkpoint."""
        return self._checkpoints_dir / f"iteration_{iteration:06d}"

    def save(
        self,
        iteration: int,
        save_fn: Any,
        metadata: CheckpointMetadata | None = None,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint at the given iteration.

        Args:
            iteration: Current training iteration number.
            save_fn: Callable that accepts a path string and writes the
                checkpoint data.  Signature: ``(path: str) -> None``.
            metadata: Optional metadata to store alongside the checkpoint.
            is_best: Whether this is the best checkpoint so far.

        Returns:
            Path to the saved checkpoint directory.
        """
        import datetime

        ckpt_dir = self.iteration_dir(iteration)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save the checkpoint data via the provided function
        ckpt_path = ckpt_dir / self._CHECKPOINT_FILE
        save_fn(str(ckpt_path))

        # Build metadata
        if metadata is None:
            metadata = CheckpointMetadata(iteration=iteration)
        metadata.iteration = iteration
        metadata.is_best = is_best
        if not metadata.timestamp:
            metadata.timestamp = datetime.datetime.now(
                tz=datetime.UTC
            ).isoformat()

        # Save metadata
        meta_path = ckpt_dir / self._METADATA_FILE
        meta_path.write_text(json.dumps(metadata.to_dict(), indent=2))

        # Update best pointer
        if is_best:
            self._best_iteration = iteration
            self._best_metric_value = metadata.metrics.get("elo", None)
            self._update_pointer(self._BEST_DIR, ckpt_dir)

        # Update latest pointer
        self._update_pointer(self._LATEST_DIR, ckpt_dir)

        # Update manifest
        self._manifest.append(metadata.to_dict())
        self._save_manifest()

        # Cleanup old checkpoints
        if self._keep_last_n > 0:
            self._cleanup_old_checkpoints()

        logger.info(
            "Saved checkpoint at iteration %d (best=%s) → %s",
            iteration,
            is_best,
            ckpt_dir,
        )
        return ckpt_dir

    def load(
        self,
        load_fn: Any,
        iteration: int | None = None,
        load_best: bool = False,
    ) -> CheckpointMetadata:
        """Load a checkpoint.

        Args:
            load_fn: Callable that accepts a path string and loads the
                checkpoint data.  Signature: ``(path: str) -> None``.
            iteration: Specific iteration to load.  If None, loads latest.
            load_best: If True, load the best checkpoint instead.

        Returns:
            The metadata associated with the loaded checkpoint.

        Raises:
            FileNotFoundError: If the requested checkpoint doesn't exist.
        """
        if load_best:
            ckpt_dir = self._checkpoints_dir / self._BEST_DIR
        elif iteration is not None:
            ckpt_dir = self.iteration_dir(iteration)
        else:
            ckpt_dir = self._checkpoints_dir / self._LATEST_DIR

        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_dir}"
            )

        ckpt_path = ckpt_dir / self._CHECKPOINT_FILE
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {ckpt_path}"
            )

        load_fn(str(ckpt_path))

        # Load metadata
        meta_path = ckpt_dir / self._METADATA_FILE
        if meta_path.exists():
            raw = json.loads(meta_path.read_text())
            metadata = CheckpointMetadata.from_dict(raw)
        else:
            metadata = CheckpointMetadata(iteration=iteration or 0)

        logger.info(
            "Loaded checkpoint from iteration %d → %s",
            metadata.iteration,
            ckpt_dir,
        )
        return metadata

    def get_latest_iteration(self) -> int | None:
        """Return the iteration number of the latest checkpoint, or None."""
        if not self._manifest:
            return None
        return self._manifest[-1].get("iteration")

    def _update_pointer(self, name: str, target_dir: Path) -> None:
        """Update a named pointer (best/latest) to point to a checkpoint dir."""
        pointer_dir = self._checkpoints_dir / name
        if pointer_dir.exists():
            shutil.rmtree(pointer_dir)
        shutil.copytree(target_dir, pointer_dir)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping the last N plus best."""
        if self._keep_last_n <= 0:
            return

        # Get all iteration directories sorted by iteration number
        iter_dirs = sorted(
            (d for d in self._checkpoints_dir.iterdir()
             if d.is_dir() and d.name.startswith("iteration_")),
            key=lambda d: int(d.name.split("_")[1]),
        )

        # Keep last N + best
        to_remove = iter_dirs[:-self._keep_last_n] if len(iter_dirs) > self._keep_last_n else []

        for d in to_remove:
            # Don't remove best
            iteration = int(d.name.split("_")[1])
            if iteration == self._best_iteration:
                continue
            shutil.rmtree(d)
            logger.debug("Removed old checkpoint: %s", d.name)

        # Update manifest to match
        kept_iterations = {
            int(d.name.split("_")[1])
            for d in self._checkpoints_dir.iterdir()
            if d.is_dir() and d.name.startswith("iteration_")
        }
        self._manifest = [
            m for m in self._manifest
            if m.get("iteration") in kept_iterations
        ]
        self._save_manifest()

    def _load_manifest(self) -> None:
        """Load the manifest file if it exists."""
        manifest_path = self._checkpoints_dir / self._MANIFEST_FILE
        if manifest_path.exists():
            self._manifest = json.loads(manifest_path.read_text())
            # Restore best iteration
            for entry in self._manifest:
                if entry.get("is_best"):
                    self._best_iteration = entry["iteration"]
                    self._best_metric_value = entry.get("metrics", {}).get("elo")

    def _save_manifest(self) -> None:
        """Persist the manifest to disk."""
        manifest_path = self._checkpoints_dir / self._MANIFEST_FILE
        manifest_path.write_text(json.dumps(self._manifest, indent=2))
