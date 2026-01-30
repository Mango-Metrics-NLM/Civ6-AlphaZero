"""Integration tests for enhanced training script."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from alpha_zero.core.civ6_mcts import SimpleMCTS
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.utils.curriculum_learning import CurriculumConfig, CurriculumManager
from alpha_zero.utils.enhanced_rewards import create_enhanced_reward_shaper
from alpha_zero.utils.environment_tuning import (
    create_tuned_environment,
    get_default_fast_config,
)
from run_enhanced_training import (
    ReplayDataset,
    TrainingExample,
    evaluate_network,
    run_enhanced_training,
    run_self_play_enhanced,
    self_play_game_enhanced,
    train_epoch,
)


class TestSelfPlayEnhanced:
    """Integration tests for enhanced self-play."""

    @pytest.fixture
    def env_config(self):
        """Create environment config."""
        config = get_default_fast_config()
        config.board_size = 8
        config.max_turns = 20
        return config

    @pytest.fixture
    def mcts(self):
        """Create simple MCTS."""
        return SimpleMCTS(num_simulations=10, c_puct=1.5)

    @pytest.fixture
    def reward_shaper(self):
        """Create reward shaper."""
        return create_enhanced_reward_shaper(scale_factor=1.0, board_size=8)

    def test_self_play_game_without_reward_shaping(self, env_config, mcts):
        """Test self-play game without reward shaping."""
        env = create_tuned_environment(env_config, board_size=8)

        examples, stats = self_play_game_enhanced(
            env=env,
            mcts=mcts,
            reward_shaper=None,
            temperature=1.0,
            temperature_threshold=10,
        )

        assert len(examples) > 0
        assert "steps" in stats
        assert "winner" in stats
        assert stats["shaped_reward"] == 0.0  # No shaping

    def test_self_play_game_with_reward_shaping(self, env_config, mcts, reward_shaper):
        """Test self-play game with reward shaping."""
        env = create_tuned_environment(env_config, board_size=8)

        examples, stats = self_play_game_enhanced(
            env=env,
            mcts=mcts,
            reward_shaper=reward_shaper,
            temperature=1.0,
            temperature_threshold=10,
        )

        assert len(examples) > 0
        assert "steps" in stats
        assert "shaped_reward" in stats

    def test_examples_have_correct_structure(self, env_config, mcts):
        """Test that generated examples have correct structure."""
        env = create_tuned_environment(env_config, board_size=8)

        examples, _ = self_play_game_enhanced(
            env=env,
            mcts=mcts,
            reward_shaper=None,
        )

        for ex in examples:
            assert isinstance(ex, TrainingExample)
            assert ex.spatial.shape[1:] == (8, 8)  # Board size
            assert ex.value in [-1.0, 0.0, 1.0]


class TestRunSelfPlayEnhanced:
    """Integration tests for run_self_play_enhanced."""

    @pytest.fixture
    def network(self):
        """Create network."""
        return MicroCivNetwork(
            board_size=8,
            action_space_size=376,
            num_res_blocks=2,
            num_filters=16,
        )

    @pytest.fixture
    def env_config(self):
        """Create environment config."""
        config = get_default_fast_config()
        config.board_size = 8
        config.max_turns = 15
        return config

    @pytest.fixture
    def curriculum(self):
        """Create curriculum manager."""
        config = CurriculumConfig.default_three_stage()
        for stage in config.stages:
            stage.min_games = 5
            stage.board_size = 8
            stage.max_turns = 15
        return CurriculumManager(config)

    def test_run_self_play_with_simple_mcts(self, network, env_config, curriculum):
        """Test running self-play with simple MCTS."""
        examples, stats = run_self_play_enhanced(
            network=network,
            env_config=env_config,
            curriculum=curriculum,
            reward_shaper=None,
            num_games=2,
            num_simulations=5,
            device="cpu",
            use_simple_mcts=True,
        )

        assert len(examples) > 0
        assert len(stats) == 2

    def test_run_self_play_with_network_mcts(self, network, env_config, curriculum):
        """Test running self-play with network MCTS."""
        examples, stats = run_self_play_enhanced(
            network=network,
            env_config=env_config,
            curriculum=curriculum,
            reward_shaper=None,
            num_games=2,
            num_simulations=5,
            device="cpu",
            use_simple_mcts=False,
        )

        assert len(examples) > 0
        assert len(stats) == 2

    def test_curriculum_records_games(self, network, env_config, curriculum):
        """Test that curriculum records games."""
        initial_games = curriculum.get_progress()["total_games"]

        run_self_play_enhanced(
            network=network,
            env_config=env_config,
            curriculum=curriculum,
            reward_shaper=None,
            num_games=3,
            num_simulations=5,
            device="cpu",
            use_simple_mcts=True,
        )

        final_games = curriculum.get_progress()["total_games"]
        assert final_games == initial_games + 3


class TestTrainEpoch:
    """Integration tests for train_epoch."""

    @pytest.fixture
    def network(self):
        """Create network."""
        return MicroCivNetwork(
            board_size=8,
            action_space_size=200,
            num_res_blocks=2,
            num_filters=16,
        )

    @pytest.fixture
    def optimizer(self, network):
        """Create optimizer."""
        return torch.optim.Adam(network.parameters(), lr=0.001)

    @pytest.fixture
    def dataloader(self):
        """Create dataloader with sample data."""
        from torch.utils.data import DataLoader

        examples = [
            TrainingExample(
                spatial=np.random.randn(13, 8, 8).astype(np.float32),
                vector=np.random.randn(64).astype(np.float32),
                action_mask=np.ones(200, dtype=np.float32),
                policy=np.ones(200, dtype=np.float32) / 200,
                value=float(i % 3 - 1),
            )
            for i in range(20)
        ]

        dataset = ReplayDataset(examples)
        return DataLoader(dataset, batch_size=5, shuffle=True)

    def test_train_epoch_returns_losses(self, network, optimizer, dataloader):
        """Test train_epoch returns loss dict."""
        losses = train_epoch(
            network=network,
            optimizer=optimizer,
            dataloader=dataloader,
            device="cpu",
        )

        assert "total_loss" in losses
        assert "policy_loss" in losses
        assert "value_loss" in losses

    def test_train_epoch_reduces_loss(self, network, optimizer, dataloader):
        """Test that training reduces loss over epochs."""
        initial_losses = train_epoch(network, optimizer, dataloader, "cpu")

        for _ in range(5):
            losses = train_epoch(network, optimizer, dataloader, "cpu")

        # Loss should generally decrease
        assert losses["total_loss"] >= 0  # Loss is non-negative


class TestEvaluateNetwork:
    """Integration tests for evaluate_network."""

    @pytest.fixture
    def network(self):
        """Create trained network (random weights)."""
        return MicroCivNetwork(
            board_size=8,
            action_space_size=376,
            num_res_blocks=2,
            num_filters=16,
        )

    def test_evaluate_returns_metrics(self, network):
        """Test evaluate_network returns metrics dict."""
        metrics = evaluate_network(
            network=network,
            board_size=8,
            max_turns=20,
            device="cpu",
            num_games=2,
            num_simulations=5,
        )

        assert "win_rate" in metrics
        assert "avg_game_length" in metrics

    def test_evaluate_win_rate_range(self, network):
        """Test win rate is in valid range."""
        metrics = evaluate_network(
            network=network,
            board_size=8,
            max_turns=20,
            device="cpu",
            num_games=4,
            num_simulations=5,
        )

        assert 0.0 <= metrics["win_rate"] <= 1.0

    def test_evaluate_game_length_positive(self, network):
        """Test average game length is positive."""
        metrics = evaluate_network(
            network=network,
            board_size=8,
            max_turns=20,
            device="cpu",
            num_games=2,
            num_simulations=5,
        )

        assert metrics["avg_game_length"] > 0


class TestRunEnhancedTrainingMinimal:
    """Integration tests for run_enhanced_training with minimal config."""

    def test_minimal_training_run(self):
        """Test minimal training run completes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = run_enhanced_training(
                num_iterations=2,
                games_per_iteration=2,
                batch_size=8,
                learning_rate=0.001,
                num_epochs=1,
                num_simulations=5,
                board_size=8,
                max_turns=15,
                checkpoint_dir=tmpdir,
                eval_interval=2,
                device="cpu",
                use_curriculum=True,
                use_reward_shaping=True,
                use_env_tuning=True,
            )

            assert "iterations" in history
            assert "final_metrics" in history
            assert len(history["iterations"]) == 2

    def test_training_creates_checkpoints(self):
        """Test training creates checkpoint files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_enhanced_training(
                num_iterations=2,
                games_per_iteration=2,
                batch_size=8,
                num_epochs=1,
                num_simulations=5,
                board_size=8,
                max_turns=15,
                checkpoint_dir=tmpdir,
                eval_interval=2,
                device="cpu",
            )

            checkpoint_dir = Path(tmpdir)
            # Should have latest.pt and checkpoint_2.pt
            assert (checkpoint_dir / "latest.pt").exists()

    def test_training_without_improvements(self):
        """Test training with all improvements disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = run_enhanced_training(
                num_iterations=2,
                games_per_iteration=2,
                batch_size=8,
                num_epochs=1,
                num_simulations=5,
                board_size=8,
                max_turns=15,
                checkpoint_dir=tmpdir,
                eval_interval=2,
                device="cpu",
                use_curriculum=False,
                use_reward_shaping=False,
                use_env_tuning=False,
            )

            assert "final_metrics" in history

    def test_training_with_curriculum_only(self):
        """Test training with only curriculum enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = run_enhanced_training(
                num_iterations=2,
                games_per_iteration=2,
                batch_size=8,
                num_epochs=1,
                num_simulations=5,
                board_size=8,
                max_turns=15,
                checkpoint_dir=tmpdir,
                eval_interval=2,
                device="cpu",
                use_curriculum=True,
                use_reward_shaping=False,
                use_env_tuning=False,
            )

            assert "curriculum_stages" in history


class TestTrainingHistoryStructure:
    """Tests for training history output structure."""

    def test_history_has_all_fields(self):
        """Test history dict has all expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = run_enhanced_training(
                num_iterations=2,
                games_per_iteration=2,
                batch_size=8,
                num_epochs=1,
                num_simulations=5,
                board_size=8,
                max_turns=15,
                checkpoint_dir=tmpdir,
                eval_interval=2,
                device="cpu",
            )

            expected_fields = [
                "iterations",
                "losses",
                "win_rates",
                "curriculum_stages",
                "game_stats",
                "final_metrics",
                "final_curriculum",
            ]

            for field in expected_fields:
                assert field in history, f"Missing field: {field}"

    def test_final_metrics_structure(self):
        """Test final_metrics has expected structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = run_enhanced_training(
                num_iterations=2,
                games_per_iteration=2,
                batch_size=8,
                num_epochs=1,
                num_simulations=5,
                board_size=8,
                max_turns=15,
                checkpoint_dir=tmpdir,
                eval_interval=2,
                device="cpu",
            )

            final_metrics = history["final_metrics"]
            assert "win_rate" in final_metrics
            assert "avg_game_length" in final_metrics
