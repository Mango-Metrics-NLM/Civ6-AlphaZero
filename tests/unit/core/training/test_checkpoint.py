"""Tests for CheckpointManager and CheckpointMetadata.

Covers:
- CheckpointMetadata: to_dict/from_dict round-trip, defaults, custom values.
- CheckpointManager: constructor, from_config, should_save, iteration_dir.
- save: checkpoint file, metadata JSON, manifest, best pointer, latest pointer.
- load: specific iteration, best, latest, missing checkpoint handling.
- Cleanup: keep_last_n, preserving best checkpoint during cleanup.
- get_latest_iteration, all_checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.core.config import SDKConfig
from src.core.training.checkpoint import CheckpointManager, CheckpointMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_METRICS = {"elo": 1500.0, "win_rate": 0.65}
SAMPLE_CONFIG_SNAPSHOT = {"lr": 0.001, "batch_size": 256}
SAMPLE_EXTRA = {"notes": "experiment-42", "phase": 3}
SAMPLE_TIMESTAMP = "2026-01-20T12:00:00+00:00"


def _make_save_fn(payload: dict[str, Any] | None = None) -> Any:
    """Return a save function that writes JSON to the given path.

    Args:
        payload: Optional dictionary to serialize.  Defaults to a
            simple marker payload.

    Returns:
        A callable ``(path: str) -> None``.
    """
    data = payload if payload is not None else {"saved": True}

    def _save(path: str) -> None:
        Path(path).write_text(json.dumps(data))

    return _save


def _make_load_sink() -> tuple[Any, list[str]]:
    """Return a (load_fn, captured_paths) pair.

    The load function appends the path it receives to *captured_paths*
    so tests can assert which file was loaded.
    """
    captured: list[str] = []

    def _load(path: str) -> None:
        captured.append(path)

    return _load, captured


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Return a fresh temporary output directory."""
    return tmp_path / "training_output"


@pytest.fixture()
def manager(output_dir: Path) -> CheckpointManager:
    """Return a CheckpointManager with default settings."""
    return CheckpointManager(output_dir=output_dir)


@pytest.fixture()
def manager_with_cleanup(output_dir: Path) -> CheckpointManager:
    """Return a CheckpointManager that keeps only the last 3 checkpoints."""
    return CheckpointManager(
        output_dir=output_dir,
        keep_last_n=3,
        checkpoint_interval=10,
    )


# ---------------------------------------------------------------------------
# CheckpointMetadata
# ---------------------------------------------------------------------------


class TestCheckpointMetadata:
    """Tests for the CheckpointMetadata dataclass."""

    def test_defaults(self) -> None:
        """Metadata with only iteration should have sensible defaults."""
        meta = CheckpointMetadata(iteration=0)

        assert meta.iteration == 0
        assert meta.timestamp == ""
        assert meta.metrics == {}
        assert meta.config_snapshot == {}
        assert meta.is_best is False
        assert meta.extra == {}

    def test_custom_values(self) -> None:
        """All fields should accept custom values."""
        meta = CheckpointMetadata(
            iteration=42,
            timestamp=SAMPLE_TIMESTAMP,
            metrics=SAMPLE_METRICS,
            config_snapshot=SAMPLE_CONFIG_SNAPSHOT,
            is_best=True,
            extra=SAMPLE_EXTRA,
        )

        assert meta.iteration == 42
        assert meta.timestamp == SAMPLE_TIMESTAMP
        assert meta.metrics == SAMPLE_METRICS
        assert meta.config_snapshot == SAMPLE_CONFIG_SNAPSHOT
        assert meta.is_best is True
        assert meta.extra == SAMPLE_EXTRA

    def test_to_dict(self) -> None:
        """to_dict should return a plain dictionary with all fields."""
        meta = CheckpointMetadata(
            iteration=5,
            metrics={"elo": 1200.0},
            is_best=True,
        )
        result = meta.to_dict()

        assert isinstance(result, dict)
        assert result["iteration"] == 5
        assert result["metrics"] == {"elo": 1200.0}
        assert result["is_best"] is True
        assert "timestamp" in result
        assert "config_snapshot" in result
        assert "extra" in result

    def test_from_dict_full(self) -> None:
        """from_dict should reconstruct metadata from a complete dictionary."""
        data = {
            "iteration": 99,
            "timestamp": SAMPLE_TIMESTAMP,
            "metrics": SAMPLE_METRICS,
            "config_snapshot": SAMPLE_CONFIG_SNAPSHOT,
            "is_best": True,
            "extra": SAMPLE_EXTRA,
        }
        meta = CheckpointMetadata.from_dict(data)

        assert meta.iteration == 99
        assert meta.timestamp == SAMPLE_TIMESTAMP
        assert meta.metrics == SAMPLE_METRICS
        assert meta.config_snapshot == SAMPLE_CONFIG_SNAPSHOT
        assert meta.is_best is True
        assert meta.extra == SAMPLE_EXTRA

    def test_from_dict_partial(self) -> None:
        """from_dict should use defaults for missing keys."""
        meta = CheckpointMetadata.from_dict({"iteration": 7})

        assert meta.iteration == 7
        assert meta.timestamp == ""
        assert meta.metrics == {}
        assert meta.config_snapshot == {}
        assert meta.is_best is False
        assert meta.extra == {}

    def test_from_dict_empty(self) -> None:
        """from_dict on an empty dict should fall back to all defaults."""
        meta = CheckpointMetadata.from_dict({})

        assert meta.iteration == 0
        assert meta.is_best is False

    def test_round_trip(self) -> None:
        """to_dict followed by from_dict should produce an equal object."""
        original = CheckpointMetadata(
            iteration=10,
            timestamp=SAMPLE_TIMESTAMP,
            metrics=SAMPLE_METRICS,
            config_snapshot=SAMPLE_CONFIG_SNAPSHOT,
            is_best=True,
            extra=SAMPLE_EXTRA,
        )
        reconstructed = CheckpointMetadata.from_dict(original.to_dict())

        assert reconstructed.iteration == original.iteration
        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.metrics == original.metrics
        assert reconstructed.config_snapshot == original.config_snapshot
        assert reconstructed.is_best == original.is_best
        assert reconstructed.extra == original.extra

    def test_round_trip_defaults(self) -> None:
        """Round-trip should work correctly with default values."""
        original = CheckpointMetadata(iteration=0)
        reconstructed = CheckpointMetadata.from_dict(original.to_dict())

        assert reconstructed.iteration == original.iteration
        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.metrics == original.metrics
        assert reconstructed.is_best == original.is_best


# ---------------------------------------------------------------------------
# CheckpointManager — construction
# ---------------------------------------------------------------------------


class TestCheckpointManagerConstruction:
    """Tests for CheckpointManager initialization and configuration."""

    def test_constructor_creates_directory(self, output_dir: Path) -> None:
        """Constructor should create the checkpoints subdirectory."""
        mgr = CheckpointManager(output_dir=output_dir)
        checkpoints_dir = output_dir / "checkpoints"

        assert checkpoints_dir.is_dir()
        assert mgr.output_dir == output_dir

    def test_constructor_defaults(self, output_dir: Path) -> None:
        """Default keep_last_n=0 and checkpoint_interval=10."""
        mgr = CheckpointManager(output_dir=output_dir)

        assert mgr.checkpoint_interval == 10
        assert mgr.best_iteration is None
        assert mgr.all_checkpoints == []

    def test_constructor_custom_interval(self, output_dir: Path) -> None:
        """Custom checkpoint_interval should be stored."""
        mgr = CheckpointManager(
            output_dir=output_dir,
            checkpoint_interval=25,
        )

        assert mgr.checkpoint_interval == 25

    def test_from_config(self, tmp_path: Path) -> None:
        """from_config should create a manager from an SDKConfig."""
        config = SDKConfig(
            output_dir=str(tmp_path / "cfg_output"),
            training={"checkpoint_interval": 50},
        )
        mgr = CheckpointManager.from_config(config)

        assert mgr.output_dir == Path(config.output_dir)
        assert mgr.checkpoint_interval == 50

    def test_constructor_accepts_string_path(self, output_dir: Path) -> None:
        """Constructor should accept a string path as well as a Path."""
        mgr = CheckpointManager(output_dir=str(output_dir))

        assert mgr.output_dir == output_dir
        assert (output_dir / "checkpoints").is_dir()

    def test_constructor_idempotent_on_existing_dir(
        self, output_dir: Path
    ) -> None:
        """Creating a manager twice on the same directory should not fail."""
        CheckpointManager(output_dir=output_dir)
        mgr2 = CheckpointManager(output_dir=output_dir)

        assert mgr2.output_dir == output_dir


# ---------------------------------------------------------------------------
# should_save / iteration_dir
# ---------------------------------------------------------------------------


class TestShouldSaveAndIterationDir:
    """Tests for should_save and iteration_dir helper methods."""

    def test_should_save_at_interval(self, output_dir: Path) -> None:
        """should_save returns True at multiples of checkpoint_interval."""
        mgr = CheckpointManager(output_dir=output_dir, checkpoint_interval=10)

        assert mgr.should_save(0) is True
        assert mgr.should_save(10) is True
        assert mgr.should_save(20) is True
        assert mgr.should_save(100) is True

    def test_should_save_between_intervals(self, output_dir: Path) -> None:
        """should_save returns False between checkpoint intervals."""
        mgr = CheckpointManager(output_dir=output_dir, checkpoint_interval=10)

        assert mgr.should_save(1) is False
        assert mgr.should_save(5) is False
        assert mgr.should_save(11) is False
        assert mgr.should_save(99) is False

    def test_should_save_interval_one(self, output_dir: Path) -> None:
        """With interval=1, every iteration should trigger a save."""
        mgr = CheckpointManager(output_dir=output_dir, checkpoint_interval=1)

        assert mgr.should_save(0) is True
        assert mgr.should_save(1) is True
        assert mgr.should_save(7) is True

    def test_iteration_dir_format(self, manager: CheckpointManager) -> None:
        """iteration_dir should return a zero-padded path."""
        result = manager.iteration_dir(0)
        assert result.name == "iteration_000000"

        result = manager.iteration_dir(42)
        assert result.name == "iteration_000042"

        result = manager.iteration_dir(999999)
        assert result.name == "iteration_999999"

    def test_iteration_dir_is_under_checkpoints(
        self, manager: CheckpointManager
    ) -> None:
        """iteration_dir should be nested under output_dir/checkpoints."""
        result = manager.iteration_dir(5)
        expected_parent = manager.output_dir / "checkpoints"

        assert result.parent == expected_parent


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class TestSave:
    """Tests for the save method."""

    def test_save_creates_checkpoint_file(
        self, manager: CheckpointManager
    ) -> None:
        """save should invoke save_fn and create checkpoint.bin."""
        iteration = 10
        manager.save(iteration, save_fn=_make_save_fn())

        ckpt_file = manager.iteration_dir(iteration) / "checkpoint.bin"
        assert ckpt_file.exists()

        data = json.loads(ckpt_file.read_text())
        assert data == {"saved": True}

    def test_save_creates_metadata_json(
        self, manager: CheckpointManager
    ) -> None:
        """save should write a metadata.json alongside the checkpoint."""
        iteration = 20
        meta = CheckpointMetadata(
            iteration=iteration,
            metrics=SAMPLE_METRICS,
        )
        manager.save(iteration, save_fn=_make_save_fn(), metadata=meta)

        meta_file = manager.iteration_dir(iteration) / "metadata.json"
        assert meta_file.exists()

        raw = json.loads(meta_file.read_text())
        assert raw["iteration"] == iteration
        assert raw["metrics"] == SAMPLE_METRICS

    def test_save_auto_generates_metadata(
        self, manager: CheckpointManager
    ) -> None:
        """save without explicit metadata should create default metadata."""
        iteration = 30
        manager.save(iteration, save_fn=_make_save_fn())

        meta_file = manager.iteration_dir(iteration) / "metadata.json"
        assert meta_file.exists()

        raw = json.loads(meta_file.read_text())
        assert raw["iteration"] == iteration

    def test_save_sets_timestamp_if_empty(
        self, manager: CheckpointManager
    ) -> None:
        """save should auto-fill the timestamp when not provided."""
        iteration = 5
        meta = CheckpointMetadata(iteration=iteration)
        manager.save(iteration, save_fn=_make_save_fn(), metadata=meta)

        meta_file = manager.iteration_dir(iteration) / "metadata.json"
        raw = json.loads(meta_file.read_text())

        assert raw["timestamp"] != ""
        assert len(raw["timestamp"]) > 0

    def test_save_preserves_existing_timestamp(
        self, manager: CheckpointManager
    ) -> None:
        """save should keep a pre-set timestamp untouched."""
        iteration = 5
        meta = CheckpointMetadata(
            iteration=iteration,
            timestamp=SAMPLE_TIMESTAMP,
        )
        manager.save(iteration, save_fn=_make_save_fn(), metadata=meta)

        meta_file = manager.iteration_dir(iteration) / "metadata.json"
        raw = json.loads(meta_file.read_text())

        assert raw["timestamp"] == SAMPLE_TIMESTAMP

    def test_save_updates_manifest(
        self, manager: CheckpointManager
    ) -> None:
        """Each save should append an entry to the manifest."""
        manager.save(10, save_fn=_make_save_fn())
        manager.save(20, save_fn=_make_save_fn())

        manifest_path = (
            manager.output_dir / "checkpoints" / "manifest.json"
        )
        assert manifest_path.exists()

        entries = json.loads(manifest_path.read_text())
        assert len(entries) == 2
        assert entries[0]["iteration"] == 10
        assert entries[1]["iteration"] == 20

    def test_save_updates_latest_pointer(
        self, manager: CheckpointManager
    ) -> None:
        """save should create/update the latest/ directory."""
        manager.save(10, save_fn=_make_save_fn({"v": 1}))
        manager.save(20, save_fn=_make_save_fn({"v": 2}))

        latest_dir = manager.output_dir / "checkpoints" / "latest"
        assert latest_dir.is_dir()

        ckpt_file = latest_dir / "checkpoint.bin"
        data = json.loads(ckpt_file.read_text())
        assert data == {"v": 2}

    def test_save_best_updates_best_pointer(
        self, manager: CheckpointManager
    ) -> None:
        """save with is_best=True should create/update the best/ directory."""
        manager.save(10, save_fn=_make_save_fn({"v": 1}), is_best=True)

        best_dir = manager.output_dir / "checkpoints" / "best"
        assert best_dir.is_dir()

        ckpt_file = best_dir / "checkpoint.bin"
        data = json.loads(ckpt_file.read_text())
        assert data == {"v": 1}

        assert manager.best_iteration == 10

    def test_save_best_pointer_changes(
        self, manager: CheckpointManager
    ) -> None:
        """A later best should replace the previous best pointer."""
        manager.save(10, save_fn=_make_save_fn({"v": 1}), is_best=True)
        manager.save(20, save_fn=_make_save_fn({"v": 2}), is_best=False)
        manager.save(30, save_fn=_make_save_fn({"v": 3}), is_best=True)

        best_dir = manager.output_dir / "checkpoints" / "best"
        ckpt_file = best_dir / "checkpoint.bin"
        data = json.loads(ckpt_file.read_text())

        assert data == {"v": 3}
        assert manager.best_iteration == 30

    def test_save_returns_checkpoint_dir(
        self, manager: CheckpointManager
    ) -> None:
        """save should return the path to the checkpoint directory."""
        result = manager.save(10, save_fn=_make_save_fn())

        assert result == manager.iteration_dir(10)
        assert result.is_dir()

    def test_save_with_custom_payload(
        self, manager: CheckpointManager
    ) -> None:
        """save_fn receives the correct path and writes custom data."""
        payload = {"weights": [1.0, 2.0, 3.0], "step": 100}
        manager.save(10, save_fn=_make_save_fn(payload))

        ckpt_file = manager.iteration_dir(10) / "checkpoint.bin"
        data = json.loads(ckpt_file.read_text())

        assert data == payload

    def test_save_metadata_is_best_flag(
        self, manager: CheckpointManager
    ) -> None:
        """The is_best flag should be recorded in the metadata file."""
        manager.save(10, save_fn=_make_save_fn(), is_best=True)

        meta_file = manager.iteration_dir(10) / "metadata.json"
        raw = json.loads(meta_file.read_text())

        assert raw["is_best"] is True

    def test_save_overrides_metadata_iteration(
        self, manager: CheckpointManager
    ) -> None:
        """save should overwrite metadata.iteration with the actual iteration."""
        meta = CheckpointMetadata(iteration=999)
        manager.save(10, save_fn=_make_save_fn(), metadata=meta)

        meta_file = manager.iteration_dir(10) / "metadata.json"
        raw = json.loads(meta_file.read_text())

        assert raw["iteration"] == 10


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


class TestLoad:
    """Tests for the load method."""

    def test_load_specific_iteration(
        self, manager: CheckpointManager
    ) -> None:
        """load with a specific iteration should load that checkpoint."""
        payload = {"model": "v1"}
        manager.save(10, save_fn=_make_save_fn(payload))

        load_fn, captured = _make_load_sink()
        meta = manager.load(load_fn, iteration=10)

        assert len(captured) == 1
        assert "iteration_000010" in captured[0]
        assert "checkpoint.bin" in captured[0]
        assert meta.iteration == 10

    def test_load_latest(self, manager: CheckpointManager) -> None:
        """load without iteration should load the latest checkpoint."""
        manager.save(10, save_fn=_make_save_fn({"v": 1}))
        manager.save(20, save_fn=_make_save_fn({"v": 2}))

        load_fn, captured = _make_load_sink()
        meta = manager.load(load_fn)

        assert len(captured) == 1
        assert "latest" in captured[0]
        assert meta.iteration == 20

    def test_load_best(self, manager: CheckpointManager) -> None:
        """load with load_best=True should load the best checkpoint."""
        manager.save(10, save_fn=_make_save_fn(), is_best=False)
        manager.save(20, save_fn=_make_save_fn(), is_best=True)
        manager.save(30, save_fn=_make_save_fn(), is_best=False)

        load_fn, captured = _make_load_sink()
        meta = manager.load(load_fn, load_best=True)

        assert len(captured) == 1
        assert "best" in captured[0]
        assert meta.iteration == 20

    def test_load_returns_metadata(self, manager: CheckpointManager) -> None:
        """load should return the stored CheckpointMetadata."""
        meta_in = CheckpointMetadata(
            iteration=10,
            metrics=SAMPLE_METRICS,
            config_snapshot=SAMPLE_CONFIG_SNAPSHOT,
        )
        manager.save(10, save_fn=_make_save_fn(), metadata=meta_in)

        load_fn, _ = _make_load_sink()
        meta_out = manager.load(load_fn, iteration=10)

        assert meta_out.iteration == 10
        assert meta_out.metrics == SAMPLE_METRICS
        assert meta_out.config_snapshot == SAMPLE_CONFIG_SNAPSHOT

    def test_load_missing_iteration_raises(
        self, manager: CheckpointManager
    ) -> None:
        """load should raise FileNotFoundError for a nonexistent iteration."""
        load_fn, _ = _make_load_sink()

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            manager.load(load_fn, iteration=999)

    def test_load_missing_best_raises(
        self, manager: CheckpointManager
    ) -> None:
        """load with load_best=True should raise if no best exists."""
        load_fn, _ = _make_load_sink()

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            manager.load(load_fn, load_best=True)

    def test_load_missing_latest_raises(
        self, manager: CheckpointManager
    ) -> None:
        """load without arguments should raise if nothing saved yet."""
        load_fn, _ = _make_load_sink()

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            manager.load(load_fn)

    def test_load_missing_checkpoint_file_raises(
        self, manager: CheckpointManager
    ) -> None:
        """load should raise if the directory exists but checkpoint.bin does not."""
        # Create the directory without the checkpoint file
        ckpt_dir = manager.iteration_dir(10)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "metadata.json").write_text("{}")

        load_fn, _ = _make_load_sink()

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            manager.load(load_fn, iteration=10)

    def test_load_without_metadata_file(
        self, manager: CheckpointManager
    ) -> None:
        """load should return default metadata when metadata.json is missing."""
        # Save normally, then delete the metadata file
        manager.save(10, save_fn=_make_save_fn())
        meta_file = manager.iteration_dir(10) / "metadata.json"
        meta_file.unlink()

        load_fn, _ = _make_load_sink()
        meta = manager.load(load_fn, iteration=10)

        assert meta.iteration == 10
        assert meta.metrics == {}


# ---------------------------------------------------------------------------
# Cleanup (keep_last_n)
# ---------------------------------------------------------------------------


class TestCleanup:
    """Tests for checkpoint cleanup via keep_last_n."""

    def test_cleanup_removes_old_checkpoints(
        self, manager_with_cleanup: CheckpointManager
    ) -> None:
        """Saving more than keep_last_n should remove the oldest."""
        keep_last_n = 3
        iterations = [10, 20, 30, 40, 50]

        for it in iterations:
            manager_with_cleanup.save(it, save_fn=_make_save_fn())

        # Only the last 3 iteration dirs should remain
        checkpoints_dir = manager_with_cleanup.output_dir / "checkpoints"
        iter_dirs = sorted(
            d.name
            for d in checkpoints_dir.iterdir()
            if d.is_dir() and d.name.startswith("iteration_")
        )

        assert len(iter_dirs) == keep_last_n
        assert "iteration_000010" not in iter_dirs
        assert "iteration_000020" not in iter_dirs
        assert "iteration_000030" in iter_dirs
        assert "iteration_000040" in iter_dirs
        assert "iteration_000050" in iter_dirs

    def test_cleanup_preserves_best(self, output_dir: Path) -> None:
        """Cleanup should never remove the best checkpoint."""
        mgr = CheckpointManager(
            output_dir=output_dir,
            keep_last_n=2,
            checkpoint_interval=1,
        )
        # Save 5 checkpoints, marking the first as best
        mgr.save(1, save_fn=_make_save_fn(), is_best=True)
        mgr.save(2, save_fn=_make_save_fn())
        mgr.save(3, save_fn=_make_save_fn())
        mgr.save(4, save_fn=_make_save_fn())
        mgr.save(5, save_fn=_make_save_fn())

        # Best (iteration 1) should still exist even though keep_last_n=2
        checkpoints_dir = output_dir / "checkpoints"
        iter_dirs = sorted(
            d.name
            for d in checkpoints_dir.iterdir()
            if d.is_dir() and d.name.startswith("iteration_")
        )

        assert "iteration_000001" in iter_dirs  # best preserved
        assert "iteration_000004" in iter_dirs  # kept
        assert "iteration_000005" in iter_dirs  # kept
        assert mgr.best_iteration == 1

    def test_cleanup_updates_manifest(self, output_dir: Path) -> None:
        """After cleanup, the manifest should only list remaining checkpoints."""
        mgr = CheckpointManager(
            output_dir=output_dir,
            keep_last_n=2,
            checkpoint_interval=1,
        )
        for it in range(1, 6):
            mgr.save(it, save_fn=_make_save_fn())

        manifest_path = output_dir / "checkpoints" / "manifest.json"
        entries = json.loads(manifest_path.read_text())
        manifest_iterations = {e["iteration"] for e in entries}

        # Only the last 2 should remain in the manifest
        assert len(entries) == 2
        assert 4 in manifest_iterations
        assert 5 in manifest_iterations

    def test_no_cleanup_when_keep_last_n_zero(
        self, manager: CheckpointManager
    ) -> None:
        """With keep_last_n=0 (default), no cleanup should occur."""
        for it in range(0, 60, 10):
            manager.save(it, save_fn=_make_save_fn())

        checkpoints_dir = manager.output_dir / "checkpoints"
        iter_dirs = [
            d
            for d in checkpoints_dir.iterdir()
            if d.is_dir() and d.name.startswith("iteration_")
        ]

        assert len(iter_dirs) == 6

    def test_cleanup_does_not_remove_when_under_limit(
        self, manager_with_cleanup: CheckpointManager
    ) -> None:
        """No cleanup when fewer checkpoints than keep_last_n."""
        manager_with_cleanup.save(10, save_fn=_make_save_fn())
        manager_with_cleanup.save(20, save_fn=_make_save_fn())

        checkpoints_dir = manager_with_cleanup.output_dir / "checkpoints"
        iter_dirs = [
            d
            for d in checkpoints_dir.iterdir()
            if d.is_dir() and d.name.startswith("iteration_")
        ]

        assert len(iter_dirs) == 2


# ---------------------------------------------------------------------------
# get_latest_iteration / all_checkpoints
# ---------------------------------------------------------------------------


class TestQueryMethods:
    """Tests for get_latest_iteration and all_checkpoints."""

    def test_get_latest_iteration_empty(
        self, manager: CheckpointManager
    ) -> None:
        """get_latest_iteration should return None when no checkpoints exist."""
        assert manager.get_latest_iteration() is None

    def test_get_latest_iteration_after_saves(
        self, manager: CheckpointManager
    ) -> None:
        """get_latest_iteration should return the most recent iteration."""
        manager.save(10, save_fn=_make_save_fn())
        assert manager.get_latest_iteration() == 10

        manager.save(20, save_fn=_make_save_fn())
        assert manager.get_latest_iteration() == 20

    def test_all_checkpoints_empty(
        self, manager: CheckpointManager
    ) -> None:
        """all_checkpoints should return an empty list initially."""
        assert manager.all_checkpoints == []

    def test_all_checkpoints_returns_copies(
        self, manager: CheckpointManager
    ) -> None:
        """all_checkpoints should return a copy, not the internal list."""
        manager.save(10, save_fn=_make_save_fn())
        checkpoints = manager.all_checkpoints

        checkpoints.clear()
        assert len(manager.all_checkpoints) == 1

    def test_all_checkpoints_accumulates(
        self, manager: CheckpointManager
    ) -> None:
        """all_checkpoints should grow with each save."""
        manager.save(10, save_fn=_make_save_fn())
        manager.save(20, save_fn=_make_save_fn())
        manager.save(30, save_fn=_make_save_fn())

        checkpoints = manager.all_checkpoints
        assert len(checkpoints) == 3

        iterations = [c["iteration"] for c in checkpoints]
        assert iterations == [10, 20, 30]

    def test_all_checkpoints_entries_contain_metadata(
        self, manager: CheckpointManager
    ) -> None:
        """Each entry in all_checkpoints should contain metadata fields."""
        meta = CheckpointMetadata(
            iteration=10,
            metrics=SAMPLE_METRICS,
            is_best=True,
        )
        manager.save(10, save_fn=_make_save_fn(), metadata=meta, is_best=True)

        entries = manager.all_checkpoints
        assert len(entries) == 1

        entry = entries[0]
        assert entry["iteration"] == 10
        assert entry["metrics"] == SAMPLE_METRICS
        assert entry["is_best"] is True


# ---------------------------------------------------------------------------
# Manifest persistence across manager instances
# ---------------------------------------------------------------------------


class TestManifestPersistence:
    """Tests for manifest persistence when re-creating a CheckpointManager."""

    def test_manifest_survives_restart(self, output_dir: Path) -> None:
        """A new manager on the same directory should reload the manifest."""
        mgr1 = CheckpointManager(output_dir=output_dir)
        mgr1.save(10, save_fn=_make_save_fn())
        mgr1.save(20, save_fn=_make_save_fn(), is_best=True)

        mgr2 = CheckpointManager(output_dir=output_dir)

        assert mgr2.get_latest_iteration() == 20
        assert len(mgr2.all_checkpoints) == 2
        assert mgr2.best_iteration == 20

    def test_best_iteration_restored_on_restart(
        self, output_dir: Path
    ) -> None:
        """Reloading should restore best_iteration from the manifest."""
        mgr1 = CheckpointManager(output_dir=output_dir)
        mgr1.save(10, save_fn=_make_save_fn(), is_best=True)
        mgr1.save(20, save_fn=_make_save_fn(), is_best=False)

        mgr2 = CheckpointManager(output_dir=output_dir)

        assert mgr2.best_iteration == 10

    def test_load_after_restart(self, output_dir: Path) -> None:
        """Checkpoints saved by one manager should be loadable by another."""
        payload = {"weights": [0.1, 0.2]}
        mgr1 = CheckpointManager(output_dir=output_dir)
        mgr1.save(10, save_fn=_make_save_fn(payload))

        mgr2 = CheckpointManager(output_dir=output_dir)
        load_fn, captured = _make_load_sink()
        meta = mgr2.load(load_fn, iteration=10)

        assert meta.iteration == 10
        assert len(captured) == 1
