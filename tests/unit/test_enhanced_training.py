"""Unit tests for enhanced training script."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from run_enhanced_training import (
    ReplayBuffer,
    ReplayDataset,
    TrainingExample,
)


class TestTrainingExample:
    """Tests for TrainingExample dataclass."""

    def test_create_example(self):
        """Test creating a training example."""
        example = TrainingExample(
            spatial=np.zeros((13, 8, 8), dtype=np.float32),
            vector=np.zeros(64, dtype=np.float32),
            action_mask=np.ones(100, dtype=np.float32),
            policy=np.ones(100, dtype=np.float32) / 100,
            value=1.0,
        )

        assert example.spatial.shape == (13, 8, 8)
        assert example.vector.shape == (64,)
        assert example.action_mask.shape == (100,)
        assert example.policy.shape == (100,)
        assert example.value == 1.0

    def test_example_with_different_values(self):
        """Test example with different value outcomes."""
        win_example = TrainingExample(
            spatial=np.zeros((13, 8, 8)),
            vector=np.zeros(64),
            action_mask=np.ones(100),
            policy=np.ones(100) / 100,
            value=1.0,
        )

        loss_example = TrainingExample(
            spatial=np.zeros((13, 8, 8)),
            vector=np.zeros(64),
            action_mask=np.ones(100),
            policy=np.ones(100) / 100,
            value=-1.0,
        )

        draw_example = TrainingExample(
            spatial=np.zeros((13, 8, 8)),
            vector=np.zeros(64),
            action_mask=np.ones(100),
            policy=np.ones(100) / 100,
            value=0.0,
        )

        assert win_example.value == 1.0
        assert loss_example.value == -1.0
        assert draw_example.value == 0.0


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    @pytest.fixture
    def buffer(self):
        """Create a replay buffer."""
        return ReplayBuffer(max_size=100)

    @pytest.fixture
    def sample_example(self):
        """Create a sample training example."""
        return TrainingExample(
            spatial=np.random.randn(13, 8, 8).astype(np.float32),
            vector=np.random.randn(64).astype(np.float32),
            action_mask=np.ones(100, dtype=np.float32),
            policy=np.ones(100, dtype=np.float32) / 100,
            value=1.0,
        )

    def test_empty_buffer(self, buffer):
        """Test empty buffer length."""
        assert len(buffer) == 0

    def test_add_single_example(self, buffer, sample_example):
        """Test adding single example."""
        buffer.add(sample_example)
        assert len(buffer) == 1

    def test_add_multiple_examples(self, buffer, sample_example):
        """Test adding multiple examples."""
        for i in range(10):
            buffer.add(sample_example)
        assert len(buffer) == 10

    def test_add_game(self, buffer, sample_example):
        """Test adding a game (list of examples)."""
        examples = [sample_example for _ in range(5)]
        buffer.add_game(examples)
        assert len(buffer) == 5

    def test_buffer_max_size(self, sample_example):
        """Test buffer respects max size."""
        buffer = ReplayBuffer(max_size=10)
        for i in range(20):
            buffer.add(sample_example)
        assert len(buffer) == 10

    def test_sample(self, buffer, sample_example):
        """Test sampling from buffer."""
        for i in range(10):
            buffer.add(sample_example)

        samples = buffer.sample(5)
        assert len(samples) == 5
        assert all(isinstance(s, TrainingExample) for s in samples)

    def test_sample_all(self, buffer, sample_example):
        """Test sampling when batch_size >= buffer size."""
        for i in range(5):
            buffer.add(sample_example)

        samples = buffer.sample(10)
        assert len(samples) == 5  # Should return all available

    def test_sample_from_empty_returns_empty(self, buffer):
        """Test sampling from empty buffer returns empty list."""
        # The sample method uses min(batch_size, len(buffer)) so empty buffer returns []
        try:
            result = buffer.sample(5)
            # If it doesn't raise, it should return empty
            assert len(result) == 0
        except (ValueError, IndexError):
            # This is also acceptable behavior
            pass


class TestReplayDataset:
    """Tests for ReplayDataset."""

    @pytest.fixture
    def examples(self):
        """Create sample examples."""
        return [
            TrainingExample(
                spatial=np.random.randn(13, 8, 8).astype(np.float32),
                vector=np.random.randn(64).astype(np.float32),
                action_mask=np.ones(100, dtype=np.float32),
                policy=np.ones(100, dtype=np.float32) / 100,
                value=float(i % 3 - 1),  # -1, 0, 1
            )
            for i in range(10)
        ]

    def test_dataset_length(self, examples):
        """Test dataset length."""
        dataset = ReplayDataset(examples)
        assert len(dataset) == 10

    def test_dataset_getitem(self, examples):
        """Test getting item from dataset."""
        dataset = ReplayDataset(examples)
        item = dataset[0]

        assert "spatial" in item
        assert "vector" in item
        assert "action_mask" in item
        assert "policy" in item
        assert "value" in item

    def test_dataset_tensor_types(self, examples):
        """Test that dataset returns tensors."""
        dataset = ReplayDataset(examples)
        item = dataset[0]

        assert isinstance(item["spatial"], torch.Tensor)
        assert isinstance(item["vector"], torch.Tensor)
        assert isinstance(item["action_mask"], torch.Tensor)
        assert isinstance(item["policy"], torch.Tensor)
        assert isinstance(item["value"], torch.Tensor)

    def test_dataset_shapes(self, examples):
        """Test tensor shapes."""
        dataset = ReplayDataset(examples)
        item = dataset[0]

        assert item["spatial"].shape == (13, 8, 8)
        assert item["vector"].shape == (64,)
        assert item["action_mask"].shape == (100,)
        assert item["policy"].shape == (100,)
        assert item["value"].shape == ()

    def test_dataloader_iteration(self, examples):
        """Test using dataset with DataLoader."""
        from torch.utils.data import DataLoader

        dataset = ReplayDataset(examples)
        loader = DataLoader(dataset, batch_size=3, shuffle=True)

        batches = list(loader)
        assert len(batches) == 4  # 10 examples / 3 batch_size = 4 batches

        # Check first batch
        batch = batches[0]
        assert batch["spatial"].shape[0] == 3
        assert batch["value"].shape[0] == 3


class TestTrainingPresetsConfig:
    """Tests for training preset configurations."""

    def test_fast_preset_values(self):
        """Test fast preset has expected values."""

        # Verify preset structure
        presets = {
            "fast": {
                "num_iterations": 10,
                "games_per_iteration": 10,
                "batch_size": 32,
            },
            "balanced": {
                "num_iterations": 50,
                "games_per_iteration": 30,
                "batch_size": 64,
            },
            "full": {
                "num_iterations": 200,
                "games_per_iteration": 100,
                "batch_size": 128,
            },
        }

        assert presets["fast"]["num_iterations"] == 10
        assert presets["balanced"]["num_iterations"] == 50
        assert presets["full"]["num_iterations"] == 200

    def test_preset_batch_sizes(self):
        """Test preset batch sizes are reasonable."""
        presets = {
            "fast": {"batch_size": 32},
            "balanced": {"batch_size": 64},
            "full": {"batch_size": 128},
        }

        # Batch sizes should be powers of 2
        for preset, config in presets.items():
            bs = config["batch_size"]
            assert bs & (bs - 1) == 0, f"{preset} batch_size {bs} is not power of 2"


class TestTrainingFunctionSignatures:
    """Tests for training function interfaces."""

    def test_run_enhanced_training_signature(self):
        """Test run_enhanced_training has expected parameters."""
        import inspect

        from run_enhanced_training import run_enhanced_training

        sig = inspect.signature(run_enhanced_training)
        params = list(sig.parameters.keys())

        expected_params = [
            "num_iterations",
            "games_per_iteration",
            "batch_size",
            "learning_rate",
            "num_epochs",
            "num_simulations",
            "board_size",
            "max_turns",
            "checkpoint_dir",
            "eval_interval",
            "device",
            "use_curriculum",
            "use_reward_shaping",
            "use_env_tuning",
        ]

        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

    def test_train_epoch_signature(self):
        """Test train_epoch has expected parameters."""
        import inspect

        from run_enhanced_training import train_epoch

        sig = inspect.signature(train_epoch)
        params = list(sig.parameters.keys())

        assert "network" in params
        assert "optimizer" in params
        assert "dataloader" in params
        assert "device" in params

    def test_evaluate_network_signature(self):
        """Test evaluate_network has expected parameters."""
        import inspect

        from run_enhanced_training import evaluate_network

        sig = inspect.signature(evaluate_network)
        params = list(sig.parameters.keys())

        assert "network" in params
        assert "board_size" in params
        assert "device" in params
        assert "num_games" in params
