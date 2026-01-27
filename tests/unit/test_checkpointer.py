"""Unit tests for the CheckpointManager module.

Tests checkpoint management functionality including:
- Save and load checkpoints
- Checkpoint rotation
- Best model tracking
- Resume support
- Metadata persistence
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from alpha_zero.orchestrator.checkpointer import CheckpointInfo, CheckpointManager


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size: int = 10, output_size: int = 5):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create checkpoint manager with temp directory."""
    return CheckpointManager(
        checkpoint_dir=temp_checkpoint_dir,
        max_checkpoints=5,
        best_metric="win_rate",
    )


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return SimpleModel()


@pytest.fixture
def sample_checkpoint_data(simple_model):
    """Create sample checkpoint data."""
    return {
        "model_state_dict": simple_model.state_dict(),
        "optimizer_state_dict": {"step": 100},
        "scheduler_state": {"learning_rate": 0.001},
        "iteration": 100,
    }


class TestCheckpointInfo:
    """Tests for CheckpointInfo dataclass."""

    def test_default_values(self):
        """Test CheckpointInfo default values."""
        info = CheckpointInfo(
            path="/path/to/checkpoint.pt",
            iteration=100,
            win_rate=0.65,
            timestamp="2024-01-01T00:00:00Z",
        )

        assert info.path == "/path/to/checkpoint.pt"
        assert info.iteration == 100
        assert info.win_rate == 0.65
        assert info.is_best is False
        assert info.metrics is None

    def test_to_dict(self):
        """Test CheckpointInfo.to_dict()."""
        info = CheckpointInfo(
            path="/path/to/checkpoint.pt",
            iteration=100,
            win_rate=0.65,
            timestamp="2024-01-01T00:00:00Z",
            is_best=True,
            metrics={"loss": 0.5},
        )

        data = info.to_dict()

        assert data["path"] == "/path/to/checkpoint.pt"
        assert data["iteration"] == 100
        assert data["win_rate"] == 0.65
        assert data["is_best"] is True
        assert data["metrics"] == {"loss": 0.5}

    def test_from_dict(self):
        """Test CheckpointInfo.from_dict()."""
        data = {
            "path": "/path/to/checkpoint.pt",
            "iteration": 100,
            "win_rate": 0.65,
            "timestamp": "2024-01-01T00:00:00Z",
            "is_best": False,
            "metrics": {"loss": 0.5},
        }

        info = CheckpointInfo.from_dict(data)

        assert info.path == "/path/to/checkpoint.pt"
        assert info.iteration == 100
        assert info.win_rate == 0.65
        assert info.metrics == {"loss": 0.5}

    def test_from_dict_with_missing_keys(self):
        """Test CheckpointInfo.from_dict() with missing keys."""
        data = {}
        info = CheckpointInfo.from_dict(data)

        assert info.path == ""
        assert info.iteration == 0
        assert info.win_rate == 0.0


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_creates_directory(self, tmp_path):
        """Test manager creates checkpoint directory."""
        checkpoint_dir = tmp_path / "new_checkpoints"
        assert not checkpoint_dir.exists()

        _ = CheckpointManager(checkpoint_dir=checkpoint_dir)

        assert checkpoint_dir.exists()

    def test_loads_existing_metadata(self, temp_checkpoint_dir):
        """Test manager loads existing metadata."""
        # Create metadata file
        metadata_path = temp_checkpoint_dir / "checkpoint_metadata.json"
        metadata = {
            "checkpoints": [
                {"path": str(temp_checkpoint_dir / "ckpt1.pt"), "iteration": 100}
            ]
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        assert manager._metadata["checkpoints"][0]["iteration"] == 100

    def test_handles_corrupted_metadata(self, temp_checkpoint_dir):
        """Test manager handles corrupted metadata file."""
        metadata_path = temp_checkpoint_dir / "checkpoint_metadata.json"
        with open(metadata_path, "w") as f:
            f.write("not valid json{{{")

        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        assert manager._metadata == {"checkpoints": []}


class TestSaveCheckpoint:
    """Tests for save() method."""

    def test_saves_checkpoint_file(
        self, checkpoint_manager, sample_checkpoint_data, temp_checkpoint_dir
    ):
        """Test save() creates checkpoint file."""
        path = checkpoint_manager.save(
            sample_checkpoint_data,
            iteration=100,
            metrics={"win_rate": 0.65},
        )

        assert Path(path).exists()
        assert "checkpoint_00000100.pt" in path

    def test_saves_with_metadata(self, checkpoint_manager, sample_checkpoint_data):
        """Test save() updates metadata."""
        checkpoint_manager.save(
            sample_checkpoint_data,
            iteration=100,
            metrics={"win_rate": 0.65},
        )

        assert checkpoint_manager.has_checkpoint()
        assert checkpoint_manager.get_latest_iteration() == 100

    def test_saves_multiple_checkpoints(
        self, checkpoint_manager, sample_checkpoint_data
    ):
        """Test saving multiple checkpoints."""
        for i in range(3):
            checkpoint_manager.save(
                sample_checkpoint_data,
                iteration=(i + 1) * 100,
                metrics={"win_rate": 0.5 + i * 0.1},
            )

        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 3

    def test_adds_checkpoint_info_to_data(
        self, checkpoint_manager, sample_checkpoint_data
    ):
        """Test save() adds _checkpoint_info to data."""
        path = checkpoint_manager.save(
            sample_checkpoint_data,
            iteration=100,
            metrics={"win_rate": 0.65},
        )

        loaded = torch.load(path, weights_only=False)
        assert "_checkpoint_info" in loaded
        assert loaded["_checkpoint_info"]["iteration"] == 100


class TestSaveBest:
    """Tests for save_best() method."""

    def test_saves_best_model(self, checkpoint_manager, simple_model, temp_checkpoint_dir):
        """Test save_best() creates best model file."""
        path = checkpoint_manager.save_best(
            simple_model,
            metrics={"win_rate": 0.75},
        )

        assert Path(path).exists()
        assert "best_model.pt" in path

    def test_updates_best_metadata(self, checkpoint_manager, simple_model):
        """Test save_best() updates best metadata."""
        checkpoint_manager.save_best(
            simple_model,
            metrics={"win_rate": 0.75},
        )

        assert checkpoint_manager.has_best()
        assert checkpoint_manager.get_best_metric() == 0.75

    def test_overwrites_previous_best(self, checkpoint_manager, simple_model):
        """Test save_best() overwrites previous best."""
        checkpoint_manager.save_best(simple_model, metrics={"win_rate": 0.6})
        checkpoint_manager.save_best(simple_model, metrics={"win_rate": 0.8})

        assert checkpoint_manager.get_best_metric() == 0.8


class TestSaveFinal:
    """Tests for save_final() method."""

    def test_saves_final_model(self, checkpoint_manager, simple_model, temp_checkpoint_dir):
        """Test save_final() creates final model file."""
        path = checkpoint_manager.save_final(
            simple_model,
            metrics={"win_rate": 0.70},
        )

        assert Path(path).exists()
        assert "final_model.pt" in path


class TestLoadCheckpoint:
    """Tests for load() method."""

    def test_loads_checkpoint(self, checkpoint_manager, sample_checkpoint_data):
        """Test load() retrieves checkpoint data."""
        path = checkpoint_manager.save(
            sample_checkpoint_data,
            iteration=100,
        )

        loaded = checkpoint_manager.load(path)

        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded

    def test_load_nonexistent_raises(self, checkpoint_manager):
        """Test load() raises for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load("/nonexistent/path.pt")


class TestLoadLatest:
    """Tests for load_latest() method."""

    def test_loads_most_recent(self, checkpoint_manager, sample_checkpoint_data):
        """Test load_latest() returns most recent checkpoint."""
        checkpoint_manager.save(sample_checkpoint_data, iteration=100)
        checkpoint_manager.save(sample_checkpoint_data, iteration=200)
        checkpoint_manager.save(sample_checkpoint_data, iteration=150)

        loaded = checkpoint_manager.load_latest()

        assert loaded["_checkpoint_info"]["iteration"] == 200

    def test_load_latest_no_checkpoints_raises(self, checkpoint_manager):
        """Test load_latest() raises when no checkpoints exist."""
        with pytest.raises(FileNotFoundError, match="No checkpoints found"):
            checkpoint_manager.load_latest()


class TestLoadBest:
    """Tests for load_best() method."""

    def test_loads_best_checkpoint(self, checkpoint_manager, simple_model, sample_checkpoint_data):
        """Test load_best() returns best checkpoint."""
        checkpoint_manager.save(sample_checkpoint_data, iteration=100)
        checkpoint_manager.save_best(simple_model, metrics={"win_rate": 0.8})

        loaded = checkpoint_manager.load_best()

        assert loaded["metrics"]["win_rate"] == 0.8

    def test_falls_back_to_latest(self, checkpoint_manager, sample_checkpoint_data):
        """Test load_best() falls back to latest if no best."""
        checkpoint_manager.save(sample_checkpoint_data, iteration=100)

        loaded = checkpoint_manager.load_best()

        assert loaded["_checkpoint_info"]["iteration"] == 100


class TestCheckpointRotation:
    """Tests for checkpoint rotation."""

    def test_rotates_old_checkpoints(self, temp_checkpoint_dir, sample_checkpoint_data):
        """Test manager rotates old checkpoints."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            max_checkpoints=3,
        )

        # Save more than max_checkpoints
        for i in range(5):
            manager.save(sample_checkpoint_data, iteration=(i + 1) * 100)

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3

        # Should keep the most recent
        iterations = [c.iteration for c in checkpoints]
        assert 300 in iterations
        assert 400 in iterations
        assert 500 in iterations

    def test_old_checkpoint_files_deleted(self, temp_checkpoint_dir, sample_checkpoint_data):
        """Test old checkpoint files are actually deleted."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            max_checkpoints=2,
        )

        paths = []
        for i in range(4):
            path = manager.save(sample_checkpoint_data, iteration=(i + 1) * 100)
            paths.append(path)

        # First two should be deleted
        assert not Path(paths[0]).exists()
        assert not Path(paths[1]).exists()
        # Last two should exist
        assert Path(paths[2]).exists()
        assert Path(paths[3]).exists()


class TestHasCheckpoint:
    """Tests for has_checkpoint() method."""

    def test_false_when_empty(self, checkpoint_manager):
        """Test has_checkpoint() returns False when no checkpoints."""
        assert not checkpoint_manager.has_checkpoint()

    def test_true_after_save(self, checkpoint_manager, sample_checkpoint_data):
        """Test has_checkpoint() returns True after saving."""
        checkpoint_manager.save(sample_checkpoint_data, iteration=100)
        assert checkpoint_manager.has_checkpoint()


class TestHasBest:
    """Tests for has_best() method."""

    def test_false_when_no_best(self, checkpoint_manager):
        """Test has_best() returns False when no best checkpoint."""
        assert not checkpoint_manager.has_best()

    def test_true_after_save_best(self, checkpoint_manager, simple_model):
        """Test has_best() returns True after saving best."""
        checkpoint_manager.save_best(simple_model, metrics={"win_rate": 0.7})
        assert checkpoint_manager.has_best()


class TestGetLatestPath:
    """Tests for get_latest_path() method."""

    def test_returns_none_when_empty(self, checkpoint_manager):
        """Test get_latest_path() returns None when no checkpoints."""
        assert checkpoint_manager.get_latest_path() is None

    def test_returns_latest_path(self, checkpoint_manager, sample_checkpoint_data):
        """Test get_latest_path() returns path to latest."""
        checkpoint_manager.save(sample_checkpoint_data, iteration=100)
        path2 = checkpoint_manager.save(sample_checkpoint_data, iteration=200)

        assert checkpoint_manager.get_latest_path() == path2


class TestGetLatestIteration:
    """Tests for get_latest_iteration() method."""

    def test_returns_zero_when_empty(self, checkpoint_manager):
        """Test get_latest_iteration() returns 0 when no checkpoints."""
        assert checkpoint_manager.get_latest_iteration() == 0

    def test_returns_latest_iteration(self, checkpoint_manager, sample_checkpoint_data):
        """Test get_latest_iteration() returns latest iteration number."""
        checkpoint_manager.save(sample_checkpoint_data, iteration=100)
        checkpoint_manager.save(sample_checkpoint_data, iteration=200)

        assert checkpoint_manager.get_latest_iteration() == 200


class TestListCheckpoints:
    """Tests for list_checkpoints() method."""

    def test_returns_empty_list_when_empty(self, checkpoint_manager):
        """Test list_checkpoints() returns empty list."""
        checkpoints = checkpoint_manager.list_checkpoints()
        assert checkpoints == []

    def test_returns_sorted_list(self, checkpoint_manager, sample_checkpoint_data):
        """Test list_checkpoints() returns sorted by iteration."""
        checkpoint_manager.save(sample_checkpoint_data, iteration=300)
        checkpoint_manager.save(sample_checkpoint_data, iteration=100)
        checkpoint_manager.save(sample_checkpoint_data, iteration=200)

        checkpoints = checkpoint_manager.list_checkpoints()

        assert len(checkpoints) == 3
        assert checkpoints[0].iteration == 100
        assert checkpoints[1].iteration == 200
        assert checkpoints[2].iteration == 300


class TestDeleteCheckpoint:
    """Tests for delete_checkpoint() method."""

    def test_deletes_checkpoint(self, checkpoint_manager, sample_checkpoint_data):
        """Test delete_checkpoint() removes checkpoint."""
        path = checkpoint_manager.save(sample_checkpoint_data, iteration=100)
        assert Path(path).exists()

        result = checkpoint_manager.delete_checkpoint(path)

        assert result is True
        assert not Path(path).exists()

    def test_returns_false_for_nonexistent(self, checkpoint_manager):
        """Test delete_checkpoint() returns False for nonexistent file."""
        result = checkpoint_manager.delete_checkpoint("/nonexistent/path.pt")
        assert result is False

    def test_updates_metadata(self, checkpoint_manager, sample_checkpoint_data):
        """Test delete_checkpoint() updates metadata."""
        path = checkpoint_manager.save(sample_checkpoint_data, iteration=100)
        checkpoint_manager.delete_checkpoint(path)

        assert not checkpoint_manager.has_checkpoint()


class TestClearAll:
    """Tests for clear_all() method."""

    def test_clears_all_checkpoints(self, checkpoint_manager, sample_checkpoint_data):
        """Test clear_all() removes all checkpoints."""
        for i in range(3):
            checkpoint_manager.save(sample_checkpoint_data, iteration=(i + 1) * 100)

        deleted = checkpoint_manager.clear_all()

        assert deleted == 3
        assert not checkpoint_manager.has_checkpoint()

    def test_keeps_best_by_default(self, checkpoint_manager, simple_model, sample_checkpoint_data):
        """Test clear_all() keeps best checkpoint by default."""
        checkpoint_manager.save(sample_checkpoint_data, iteration=100)
        checkpoint_manager.save_best(simple_model, metrics={"win_rate": 0.8})

        checkpoint_manager.clear_all(keep_best=True)

        assert checkpoint_manager.has_best()

    def test_clears_best_when_requested(self, checkpoint_manager, simple_model, sample_checkpoint_data):
        """Test clear_all() clears best when requested."""
        checkpoint_manager.save(sample_checkpoint_data, iteration=100)
        checkpoint_manager.save_best(simple_model, metrics={"win_rate": 0.8})

        checkpoint_manager.clear_all(keep_best=False)

        assert not checkpoint_manager.has_best()


class TestRepr:
    """Tests for __repr__ method."""

    def test_repr_contains_info(self, checkpoint_manager):
        """Test __repr__ contains useful information."""
        repr_str = repr(checkpoint_manager)

        assert "CheckpointManager" in repr_str
        assert "checkpoints=" in repr_str
        assert "has_best=" in repr_str


class TestMetadataPersistence:
    """Tests for metadata persistence across manager instances."""

    def test_metadata_persists(self, temp_checkpoint_dir, sample_checkpoint_data):
        """Test metadata persists between manager instances."""
        # First manager
        manager1 = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        manager1.save(sample_checkpoint_data, iteration=100)

        # Second manager (same directory)
        manager2 = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        assert manager2.has_checkpoint()
        assert manager2.get_latest_iteration() == 100
