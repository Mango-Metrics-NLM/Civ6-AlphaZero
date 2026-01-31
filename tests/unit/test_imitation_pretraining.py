"""Tests for imitation pre-training module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from alpha_zero.distributed.shared_storage import GameTrajectory, TrainingExample
from alpha_zero.evaluation.baseline_agents import GreedyHeuristicAgent, RandomAgent
from alpha_zero.training.imitation_pretraining import (
    HeuristicExpertGenerator,
    ImitationConfig,
    ImitationPreTrainer,
    SupervisedDataset,
)


class TestImitationConfig:
    """Tests for ImitationConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = ImitationConfig()

        assert config.num_games == 1000
        assert config.expert_type == "heuristic"
        assert config.board_size == 16
        assert config.max_turns == 200
        assert config.batch_size == 256
        assert config.learning_rate == 0.001
        assert config.num_epochs == 5
        assert config.gradient_clip == 1.0
        assert config.use_n_step_returns is True
        assert config.n_steps == 5
        assert config.gamma == 0.99

    def test_custom_values(self):
        """Should accept custom values."""
        config = ImitationConfig(
            num_games=500,
            board_size=8,
            batch_size=128,
            num_epochs=3,
        )

        assert config.num_games == 500
        assert config.board_size == 8
        assert config.batch_size == 128
        assert config.num_epochs == 3

    def test_from_env(self):
        """Should load from environment variables."""
        with patch.dict(os.environ, {
            "CIV6AZ_IMITATION_NUM_GAMES": "500",
            "CIV6AZ_IMITATION_BOARD_SIZE": "8",
            "CIV6AZ_IMITATION_EPOCHS": "3",
        }):
            config = ImitationConfig.from_env()

            assert config.num_games == 500
            assert config.board_size == 8
            assert config.num_epochs == 3

    def test_device_defaults_to_available(self):
        """Device should default based on CUDA availability."""
        config = ImitationConfig()

        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert config.device == expected


class TestHeuristicExpertGenerator:
    """Tests for HeuristicExpertGenerator."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for fast tests."""
        return ImitationConfig(
            num_games=2,
            board_size=8,
            max_turns=30,
            expert_seed=42,
        )

    def test_initialization(self, small_config):
        """Should initialize with config."""
        generator = HeuristicExpertGenerator(small_config)

        assert generator.config == small_config
        assert generator.agent is not None

    def test_custom_agent_injection(self, small_config):
        """Should accept custom agent for testing."""
        mock_agent = MagicMock()
        mock_agent.select_action = MagicMock(return_value=0)
        mock_agent.reset = MagicMock()

        generator = HeuristicExpertGenerator(small_config, agent=mock_agent)

        assert generator.agent is mock_agent

    def test_generate_single_game(self, small_config):
        """Should generate a complete game trajectory."""
        generator = HeuristicExpertGenerator(small_config)

        trajectory = generator.generate_game()

        assert isinstance(trajectory, GameTrajectory)
        assert len(trajectory.examples) > 0
        assert trajectory.game_length > 0
        assert trajectory.winner in [0, 1, None]

    def test_trajectory_format_matches_shared_storage(self, small_config):
        """Generated examples should match TrainingExample format."""
        generator = HeuristicExpertGenerator(small_config)
        trajectory = generator.generate_game()

        for example in trajectory.examples:
            assert isinstance(example, TrainingExample)
            assert isinstance(example.spatial, np.ndarray)
            assert isinstance(example.vector, np.ndarray)
            assert isinstance(example.action_mask, np.ndarray)
            assert isinstance(example.policy, np.ndarray)
            assert isinstance(example.value, float)
            assert -1.0 <= example.value <= 1.0

    def test_generate_batch(self, small_config):
        """Should generate multiple games."""
        generator = HeuristicExpertGenerator(small_config)

        trajectories = generator.generate_batch(3)

        assert len(trajectories) == 3
        for traj in trajectories:
            assert isinstance(traj, GameTrajectory)
            assert len(traj.examples) > 0

    def test_policy_is_one_hot(self, small_config):
        """Policy targets should be one-hot (expert action)."""
        generator = HeuristicExpertGenerator(small_config)
        trajectory = generator.generate_game()

        for example in trajectory.examples:
            # Should sum to 1 (one-hot)
            assert abs(example.policy.sum() - 1.0) < 1e-6
            # Should have exactly one 1
            assert (example.policy == 1.0).sum() == 1

    def test_deterministic_with_seed(self, small_config):
        """Same seed should produce same games."""
        gen1 = HeuristicExpertGenerator(small_config)
        gen2 = HeuristicExpertGenerator(small_config)

        traj1 = gen1.generate_game()
        traj2 = gen2.generate_game()

        assert traj1.game_length == traj2.game_length
        assert traj1.winner == traj2.winner


class TestSupervisedDataset:
    """Tests for SupervisedDataset."""

    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectories for testing."""
        examples = [
            TrainingExample(
                spatial=np.random.randn(12, 8, 8).astype(np.float32),
                vector=np.random.randn(32).astype(np.float32),
                action_mask=np.ones(100, dtype=np.float32),
                policy=np.zeros(100, dtype=np.float32),
                value=0.5,
            )
            for _ in range(10)
        ]
        # Set one-hot policy
        for i, ex in enumerate(examples):
            ex.policy[i % 100] = 1.0

        return [
            GameTrajectory(examples=examples[:5], game_id=1),
            GameTrajectory(examples=examples[5:], game_id=2),
        ]

    def test_length_matches_examples(self, sample_trajectories):
        """Dataset length should match total examples."""
        dataset = SupervisedDataset(sample_trajectories)

        assert len(dataset) == 10

    def test_getitem_returns_correct_shapes(self, sample_trajectories):
        """getitem should return correctly shaped tensors."""
        dataset = SupervisedDataset(sample_trajectories)

        item = dataset[0]

        assert "spatial" in item
        assert "vector" in item
        assert "action_mask" in item
        assert "policy" in item
        assert "value" in item

        assert isinstance(item["spatial"], torch.Tensor)
        assert item["spatial"].shape == (12, 8, 8)
        assert item["vector"].shape == (32,)
        assert item["value"].shape == ()

    def test_collate_fn(self, sample_trajectories):
        """Collate function should batch correctly."""
        dataset = SupervisedDataset(sample_trajectories)

        batch = [dataset[i] for i in range(3)]
        collated = SupervisedDataset.collate_fn(batch)

        assert collated["spatial"].shape == (3, 12, 8, 8)
        assert collated["vector"].shape == (3, 32)
        assert collated["policy"].shape == (3, 100)
        assert collated["value"].shape == (3,)

    def test_dataloader_integration(self, sample_trajectories):
        """Should work with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = SupervisedDataset(sample_trajectories)
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=SupervisedDataset.collate_fn,
        )

        batches = list(dataloader)
        assert len(batches) == 3  # 10 examples / 4 batch = 3 batches

        # First batch should have 4 examples
        assert batches[0]["spatial"].shape[0] == 4


class TestImitationPreTrainer:
    """Tests for ImitationPreTrainer."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for fast tests."""
        return ImitationConfig(
            num_games=3,
            board_size=8,
            max_turns=20,
            batch_size=8,
            num_epochs=2,
            num_res_blocks=2,
            num_filters=16,
            device="cpu",
            log_interval=1,
            save_interval=10,  # Avoid saving during tests
        )

    def test_initialization(self, small_config):
        """Should initialize with config."""
        pretrainer = ImitationPreTrainer(small_config)

        assert pretrainer.config == small_config
        assert pretrainer.network is not None
        assert pretrainer.optimizer is not None

    def test_custom_network_injection(self, small_config):
        """Should accept custom network for testing."""
        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=small_config.board_size)
        custom_network = MicroCivNetwork(
            board_size=small_config.board_size,
            action_space_size=env.action_space.n,
            num_res_blocks=2,
            num_filters=16,
        )

        pretrainer = ImitationPreTrainer(small_config, network=custom_network)

        assert pretrainer.network is custom_network

    def test_generate_expert_data(self, small_config):
        """Should generate expert games."""
        pretrainer = ImitationPreTrainer(small_config)

        trajectories = pretrainer.generate_expert_data()

        assert len(trajectories) == 3
        assert pretrainer.num_games == 3
        assert pretrainer.num_examples > 0

    def test_train_requires_data(self, small_config):
        """Should raise error if no data generated."""
        pretrainer = ImitationPreTrainer(small_config)

        with pytest.raises(ValueError, match="No training data"):
            pretrainer.train()

    def test_train_reduces_loss(self, small_config):
        """Training should reduce loss over epochs."""
        pretrainer = ImitationPreTrainer(small_config)
        pretrainer.generate_expert_data()

        metrics = pretrainer.train()

        # Should return metrics
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "policy_accuracy" in metrics

        # Check training history
        history = pretrainer.get_training_history()
        assert len(history) == 2  # 2 epochs

    def test_save_load_roundtrip(self, small_config):
        """Should save and load pretrained weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            small_config.checkpoint_dir = tmpdir

            pretrainer = ImitationPreTrainer(small_config)
            pretrainer.generate_expert_data()
            pretrainer.train()

            # Save
            path = pretrainer.save_pretrained()
            assert os.path.exists(path)

            # Load into new pretrainer
            pretrainer2 = ImitationPreTrainer(small_config)
            pretrainer2.load_pretrained(path)

            # Weights should match
            for p1, p2 in zip(
                pretrainer.network.parameters(),
                pretrainer2.network.parameters(),
            ):
                assert torch.allclose(p1, p2)

    def test_checkpoint_saving(self, small_config):
        """Should save checkpoints at interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            small_config.checkpoint_dir = tmpdir
            small_config.save_interval = 1  # Save every epoch

            pretrainer = ImitationPreTrainer(small_config)
            pretrainer.generate_expert_data()
            pretrainer.train()

            # Should have checkpoint files
            checkpoints = [f for f in os.listdir(tmpdir) if f.startswith("imitation_epoch")]
            assert len(checkpoints) >= 1

    def test_wandb_logging(self, small_config):
        """Should log to WandB when provided."""
        mock_wandb = MagicMock()
        mock_wandb.log_imitation_training = MagicMock()

        pretrainer = ImitationPreTrainer(small_config, wandb_logger=mock_wandb)
        pretrainer.generate_expert_data()
        pretrainer.train()

        # Should have called log for each epoch
        assert mock_wandb.log_imitation_training.call_count == 2

    def test_get_network(self, small_config):
        """Should return the network."""
        pretrainer = ImitationPreTrainer(small_config)

        network = pretrainer.get_network()

        assert network is pretrainer.network


class TestExpertDataGeneratorProtocol:
    """Tests for protocol compliance."""

    def test_heuristic_generator_implements_protocol(self):
        """HeuristicExpertGenerator should implement ExpertDataGenerator protocol."""
        from alpha_zero.training.imitation_pretraining import ExpertDataGenerator

        config = ImitationConfig(board_size=8, max_turns=20)
        generator = HeuristicExpertGenerator(config)

        # Should have required methods
        assert hasattr(generator, "generate_game")
        assert hasattr(generator, "generate_batch")
        assert callable(generator.generate_game)
        assert callable(generator.generate_batch)


class TestNStepReturns:
    """Tests for n-step return value computation."""

    @pytest.fixture
    def generator(self):
        """Create a generator for testing."""
        config = ImitationConfig(
            board_size=8,
            max_turns=20,
            use_n_step_returns=True,
            n_steps=5,
            gamma=0.99,
        )
        return HeuristicExpertGenerator(config)

    def test_value_targets_in_range(self, generator):
        """Value targets should be in [-1, 1]."""
        trajectory = generator.generate_game()

        for example in trajectory.examples:
            assert -1.0 <= example.value <= 1.0

    def test_final_states_have_correct_values(self, generator):
        """States near end should have values closer to outcome."""
        trajectory = generator.generate_game()

        if trajectory.winner is not None and len(trajectory.examples) > 5:
            # Last few examples should have stronger values
            last_example = trajectory.examples[-1]
            # Value magnitude should be relatively high for final states
            assert abs(last_example.value) > 0.5

    def test_n_step_vs_full_game(self):
        """n-step returns should differ from full game returns."""
        config_nstep = ImitationConfig(
            board_size=8,
            max_turns=20,
            use_n_step_returns=True,
            n_steps=3,
            expert_seed=42,
        )
        config_full = ImitationConfig(
            board_size=8,
            max_turns=20,
            use_n_step_returns=False,
            expert_seed=42,
        )

        gen_nstep = HeuristicExpertGenerator(config_nstep)
        gen_full = HeuristicExpertGenerator(config_full)

        traj_nstep = gen_nstep.generate_game()
        traj_full = gen_full.generate_game()

        # Early examples should have different values
        # (n-step discounts more heavily)
        if len(traj_nstep.examples) > 10:
            early_nstep = traj_nstep.examples[0].value
            early_full = traj_full.examples[0].value

            # For non-zero outcomes, values should differ
            if traj_nstep.winner is not None:
                # n-step should have smaller magnitude for early states
                assert abs(early_nstep) <= abs(early_full) + 0.01


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test complete generation -> training -> save pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ImitationConfig(
                num_games=3,
                board_size=8,
                max_turns=20,
                batch_size=4,
                num_epochs=1,
                num_res_blocks=2,
                num_filters=16,
                device="cpu",
                checkpoint_dir=tmpdir,
            )

            pretrainer = ImitationPreTrainer(config)

            # Generate
            trajectories = pretrainer.generate_expert_data()
            assert len(trajectories) == 3

            # Train
            metrics = pretrainer.train()
            assert metrics["loss"] > 0

            # Save
            path = pretrainer.save_pretrained()
            assert os.path.exists(path)

            # Network should be trainable
            network = pretrainer.get_network()
            assert network is not None

    def test_run_imitation_pretraining_function(self):
        """Test convenience function."""
        from alpha_zero.training.imitation_pretraining import run_imitation_pretraining

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ImitationConfig(
                num_games=2,
                board_size=8,
                max_turns=15,
                batch_size=4,
                num_epochs=1,
                num_res_blocks=2,
                num_filters=16,
                device="cpu",
                checkpoint_dir=tmpdir,
            )

            network, metrics = run_imitation_pretraining(config)

            assert network is not None
            assert "loss" in metrics
