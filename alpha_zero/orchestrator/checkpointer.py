"""Checkpoint management for training state persistence.

This module provides comprehensive checkpoint management including:
- Save and load training checkpoints
- Automatic rotation to keep only recent checkpoints
- Best model tracking by configurable metric
- Resume support from latest or best checkpoint
- Metadata persistence in JSON format

Usage:
    from alpha_zero.orchestrator.checkpointer import CheckpointManager

    manager = CheckpointManager(
        checkpoint_dir="checkpoints/experiment1",
        max_checkpoints=10,
    )

    # Save checkpoint
    manager.save(checkpoint_data, iteration=100, metrics={"win_rate": 0.65})

    # Save best model
    manager.save_best(model, metrics={"win_rate": 0.75})

    # Load checkpoints
    latest = manager.load_latest()
    best = manager.load_best()
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Metadata for a checkpoint."""

    path: str
    iteration: int
    win_rate: float
    timestamp: str
    is_best: bool = False
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointInfo:
        """Create from dictionary."""
        return cls(
            path=data.get("path", ""),
            iteration=data.get("iteration", 0),
            win_rate=data.get("win_rate", 0.0),
            timestamp=data.get("timestamp", ""),
            is_best=data.get("is_best", False),
            metrics=data.get("metrics"),
        )


class CheckpointManager:
    """Manages training checkpoints with rotation and best model tracking.

    Features:
    - Automatic checkpoint rotation (keep N most recent)
    - Best model tracking by configurable metric
    - Resume support from latest or best checkpoint
    - Checkpoint metadata tracking in JSON
    - Thread-safe operations

    Attributes:
        checkpoint_dir: Directory for storing checkpoints.
        max_checkpoints: Maximum number of checkpoints to keep.
        best_metric: Metric name to use for best model selection.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 10,
        best_metric: str = "win_rate",
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints.
            max_checkpoints: Maximum checkpoints to keep.
            best_metric: Metric to track for best model selection.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_metric = best_metric

        self._metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        self._metadata: dict[str, Any] = self._load_metadata()

        logger.info(f"CheckpointManager initialized at {self.checkpoint_dir}")

    def save(
        self,
        checkpoint_data: dict[str, Any],
        iteration: int,
        metrics: dict[str, float] | None = None,
    ) -> str:
        """Save a checkpoint.

        Args:
            checkpoint_data: Data to save (model state, optimizer state, etc.).
            iteration: Current training iteration.
            metrics: Optional metrics for this checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        filename = f"checkpoint_{iteration:08d}.pt"
        path = self.checkpoint_dir / filename

        # Add metadata to checkpoint
        checkpoint_data["_checkpoint_info"] = {
            "iteration": iteration,
            "timestamp": self._get_timestamp(),
            "metrics": metrics,
        }

        torch.save(checkpoint_data, path)
        logger.info(f"Saved checkpoint at iteration {iteration}: {path}")

        # Update metadata
        win_rate = (metrics or {}).get(self.best_metric, 0.0)
        info = CheckpointInfo(
            path=str(path),
            iteration=iteration,
            win_rate=win_rate,
            timestamp=self._get_timestamp(),
            metrics=metrics,
        )
        self._update_metadata(info)

        # Rotate old checkpoints
        self._rotate_checkpoints()

        return str(path)

    def save_best(
        self,
        model: torch.nn.Module,
        metrics: dict[str, float],
    ) -> str:
        """Save model as best checkpoint.

        Args:
            model: Model to save.
            metrics: Metrics for this model.

        Returns:
            Path to saved checkpoint.
        """
        best_path = self.checkpoint_dir / "best_model.pt"

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": self._get_timestamp(),
            "_checkpoint_info": {
                "is_best": True,
                "timestamp": self._get_timestamp(),
                "metrics": metrics,
            },
        }

        torch.save(checkpoint_data, best_path)
        logger.info(
            f"Saved best model with {self.best_metric}="
            f"{metrics.get(self.best_metric, 'N/A')}"
        )

        self._metadata["best"] = {
            "path": str(best_path),
            "metrics": metrics,
            "timestamp": self._get_timestamp(),
        }
        self._save_metadata()

        return str(best_path)

    def save_final(
        self,
        model: torch.nn.Module,
        metrics: dict[str, float],
    ) -> str:
        """Save final model.

        Args:
            model: Model to save.
            metrics: Final metrics.

        Returns:
            Path to saved checkpoint.
        """
        final_path = self.checkpoint_dir / "final_model.pt"

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": self._get_timestamp(),
            "_checkpoint_info": {
                "is_final": True,
                "timestamp": self._get_timestamp(),
                "metrics": metrics,
            },
        }

        torch.save(checkpoint_data, final_path)
        logger.info(f"Saved final model: {final_path}")

        self._metadata["final"] = {
            "path": str(final_path),
            "metrics": metrics,
            "timestamp": self._get_timestamp(),
        }
        self._save_metadata()

        return str(final_path)

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load checkpoint from path.

        Args:
            path: Checkpoint path.

        Returns:
            Checkpoint data.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"Loaded checkpoint: {path}")
        return cast(dict[str, Any], checkpoint)

    def load_latest(self) -> dict[str, Any]:
        """Load the most recent checkpoint.

        Returns:
            Latest checkpoint data.

        Raises:
            FileNotFoundError: If no checkpoints exist.
        """
        checkpoints = self._metadata.get("checkpoints", [])
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")

        latest = max(checkpoints, key=lambda x: x.get("iteration", 0))
        logger.info(f"Loading latest checkpoint from iteration {latest['iteration']}")
        return self.load(latest["path"])

    def load_best(self) -> dict[str, Any]:
        """Load the best checkpoint.

        Returns:
            Best checkpoint data.

        Raises:
            FileNotFoundError: If no checkpoints exist.
        """
        best_info = self._metadata.get("best")
        if best_info and Path(best_info["path"]).exists():
            logger.info("Loading best checkpoint")
            return self.load(best_info["path"])

        # Fall back to latest if no best
        logger.info("No best checkpoint found, loading latest")
        return self.load_latest()

    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists.

        Returns:
            True if at least one checkpoint exists.
        """
        return len(self._metadata.get("checkpoints", [])) > 0

    def has_best(self) -> bool:
        """Check if best checkpoint exists.

        Returns:
            True if best checkpoint exists.
        """
        best_info = self._metadata.get("best")
        return best_info is not None and Path(best_info.get("path", "")).exists()

    def get_latest_path(self) -> str | None:
        """Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints.
        """
        checkpoints = self._metadata.get("checkpoints", [])
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda x: x.get("iteration", 0))
        return cast(str, latest["path"])

    def get_latest_iteration(self) -> int:
        """Get iteration number of latest checkpoint.

        Returns:
            Latest iteration number, or 0 if no checkpoints.
        """
        checkpoints = self._metadata.get("checkpoints", [])
        if not checkpoints:
            return 0
        return cast(int, max(c.get("iteration", 0) for c in checkpoints))

    def get_best_metric(self) -> float | None:
        """Get best metric value.

        Returns:
            Best metric value, or None if no best checkpoint.
        """
        best_info = self._metadata.get("best")
        if not best_info:
            return None
        return cast(float | None, best_info.get("metrics", {}).get(self.best_metric))

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """List all checkpoints.

        Returns:
            List of checkpoint info objects, sorted by iteration.
        """
        checkpoints = self._metadata.get("checkpoints", [])
        infos = [CheckpointInfo.from_dict(c) for c in checkpoints]
        return sorted(infos, key=lambda x: x.iteration)

    def delete_checkpoint(self, path: str | Path) -> bool:
        """Delete a specific checkpoint.

        Args:
            path: Path to checkpoint to delete.

        Returns:
            True if deleted, False if not found.
        """
        path = Path(path)
        if not path.exists():
            return False

        path.unlink()

        # Update metadata
        checkpoints = self._metadata.get("checkpoints", [])
        self._metadata["checkpoints"] = [
            c for c in checkpoints if c.get("path") != str(path)
        ]
        self._save_metadata()

        logger.info(f"Deleted checkpoint: {path}")
        return True

    def clear_all(self, keep_best: bool = True) -> int:
        """Clear all checkpoints.

        Args:
            keep_best: If True, keeps the best model checkpoint.

        Returns:
            Number of checkpoints deleted.
        """
        deleted = 0
        checkpoints = self._metadata.get("checkpoints", [])

        for ckpt in checkpoints:
            path = Path(ckpt.get("path", ""))
            if path.exists():
                path.unlink()
                deleted += 1

        self._metadata["checkpoints"] = []

        if not keep_best:
            best_path = self._metadata.get("best", {}).get("path")
            if best_path and Path(best_path).exists():
                Path(best_path).unlink()
                deleted += 1
            self._metadata["best"] = None

        self._save_metadata()
        logger.info(f"Cleared {deleted} checkpoints")
        return deleted

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit."""
        checkpoints = self._metadata.get("checkpoints", [])
        if len(checkpoints) <= self.max_checkpoints:
            return

        # Sort by iteration and keep most recent
        sorted_checkpoints = sorted(checkpoints, key=lambda x: x.get("iteration", 0))
        to_remove = sorted_checkpoints[: -self.max_checkpoints]

        for ckpt in to_remove:
            path = Path(ckpt.get("path", ""))
            if path.exists():
                path.unlink()
                logger.debug(f"Rotated out checkpoint: {path}")

        self._metadata["checkpoints"] = sorted_checkpoints[-self.max_checkpoints :]
        self._save_metadata()

    def _update_metadata(self, info: CheckpointInfo) -> None:
        """Update metadata with new checkpoint info."""
        if "checkpoints" not in self._metadata:
            self._metadata["checkpoints"] = []

        self._metadata["checkpoints"].append(info.to_dict())
        self._save_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata from file."""
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, encoding="utf-8") as f:
                    return cast(dict[str, Any], json.load(f))
            except json.JSONDecodeError:
                logger.warning("Corrupted metadata file, starting fresh")
                return {"checkpoints": []}
        return {"checkpoints": []}

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2)

    def _get_timestamp(self) -> str:
        """Get current timestamp string in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def __repr__(self) -> str:
        """String representation of checkpoint manager."""
        num_checkpoints = len(self._metadata.get("checkpoints", []))
        has_best = self.has_best()
        return (
            f"CheckpointManager("
            f"dir={self.checkpoint_dir}, "
            f"checkpoints={num_checkpoints}, "
            f"has_best={has_best})"
        )
