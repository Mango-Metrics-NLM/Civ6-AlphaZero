"""Coverage tests for alpha_zero.training.imitation_pretraining module."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from alpha_zero.distributed.shared_storage import GameTrajectory, TrainingExample
from alpha_zero.training.imitation_pretraining import (
    HeuristicExpertGenerator,
    ImitationConfig,
    ImitationPreTrainer,
    SupervisedDataset,
    run_imitation_pretraining,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_example(spatial_shape=(13, 16, 16), vector_dim=64, action_size=376):
    """Create a single TrainingExample with random data."""
    return TrainingExample(
        spatial=np.random.randn(*spatial_shape).astype(np.float32),
        vector=np.random.randn(vector_dim).astype(np.float32),
        action_mask=np.ones(action_size, dtype=np.float32),
        policy=np.zeros(action_size, dtype=np.float32),
        value=0.5,
    )


def _make_trajectory(num_examples=5, winner=0):
    """Create a GameTrajectory with random data."""
    examples = [_make_example() for _ in range(num_examples)]
    examples[0].policy[0] = 1.0  # At least one valid action
    return GameTrajectory(
        examples=examples,
        game_id=1,
        game_length=num_examples,
        winner=winner,
    )


def _small_config(tmp_path):
    """Create a small config for fast testing."""
    return ImitationConfig(
        num_games=2,
        board_size=8,
        max_turns=20,
        batch_size=4,
        learning_rate=0.01,
        num_epochs=1,
        num_res_blocks=1,
        num_filters=16,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_interval=1,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# ImitationConfig
# ---------------------------------------------------------------------------


class TestImitationConfig:
    def test_defaults(self):
        config = ImitationConfig()
        assert config.num_games == 1000
        assert config.expert_type == "heuristic"
        assert config.expert_seed == 42
        assert config.board_size == 16
        assert config.batch_size == 256
        assert config.learning_rate == 0.001
        assert config.num_epochs == 5
        assert config.gradient_clip == 1.0
        assert config.use_n_step_returns is True
        assert config.n_steps == 5
        assert config.gamma == 0.99
        assert config.save_interval == 1
        assert config.log_interval == 10

    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = ImitationConfig.from_env()
            assert config.num_games == 1000
            assert config.expert_type == "heuristic"
            assert config.board_size == 16

    def test_from_env_overrides(self):
        env = {
            "CIV6AZ_IMITATION_NUM_GAMES": "50",
            "CIV6AZ_IMITATION_EXPERT_TYPE": "random",
            "CIV6AZ_IMITATION_EXPERT_SEED": "123",
            "CIV6AZ_IMITATION_BOARD_SIZE": "8",
            "CIV6AZ_IMITATION_MAX_TURNS": "100",
            "CIV6AZ_IMITATION_BATCH_SIZE": "32",
            "CIV6AZ_IMITATION_LR": "0.01",
            "CIV6AZ_IMITATION_EPOCHS": "3",
            "CIV6AZ_IMITATION_GRAD_CLIP": "0.5",
            "CIV6AZ_IMITATION_N_STEP": "false",
            "CIV6AZ_IMITATION_N_STEPS": "10",
            "CIV6AZ_IMITATION_GAMMA": "0.95",
            "CIV6AZ_IMITATION_RES_BLOCKS": "4",
            "CIV6AZ_IMITATION_FILTERS": "32",
            "CIV6AZ_IMITATION_DEVICE": "cpu",
            "CIV6AZ_IMITATION_CHECKPOINT_DIR": "/tmp/test_ckpt",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ImitationConfig.from_env()
            assert config.num_games == 50
            assert config.expert_type == "random"
            assert config.expert_seed == 123
            assert config.board_size == 8
            assert config.batch_size == 32
            assert config.use_n_step_returns is False
            assert config.gamma == 0.95
            assert config.device == "cpu"


# ---------------------------------------------------------------------------
# HeuristicExpertGenerator
# ---------------------------------------------------------------------------


class TestHeuristicExpertGenerator:
    def test_generate_game(self):
        config = ImitationConfig(board_size=8, max_turns=20, expert_seed=42)
        gen = HeuristicExpertGenerator(config)
        trajectory = gen.generate_game()
        assert isinstance(trajectory, GameTrajectory)
        assert len(trajectory.examples) > 0
        assert trajectory.game_length >= 0

    def test_generate_game_with_custom_agent(self):
        config = ImitationConfig(board_size=8, max_turns=20)
        mock_agent = MagicMock()
        mock_agent.select_action = MagicMock(return_value=0)
        mock_agent.reset = MagicMock()

        gen = HeuristicExpertGenerator(config, agent=mock_agent)
        trajectory = gen.generate_game()
        assert isinstance(trajectory, GameTrajectory)
        mock_agent.reset.assert_called()

    def test_generate_batch(self):
        config = ImitationConfig(board_size=8, max_turns=10, expert_seed=42)
        gen = HeuristicExpertGenerator(config)
        trajectories = gen.generate_batch(3)
        assert len(trajectories) == 3
        for t in trajectories:
            assert isinstance(t, GameTrajectory)

    def test_game_counter_increments(self):
        config = ImitationConfig(board_size=8, max_turns=10, expert_seed=42)
        gen = HeuristicExpertGenerator(config)
        assert gen._game_counter == 0
        gen.generate_game()
        assert gen._game_counter == 1
        gen.generate_game()
        assert gen._game_counter == 2

    def test_create_examples_with_n_step(self):
        config = ImitationConfig(board_size=8, max_turns=20, use_n_step_returns=True)
        gen = HeuristicExpertGenerator(config)
        trajectory = gen.generate_game()
        # All examples should have value in [-1, 1]
        for ex in trajectory.examples:
            assert -1.0 <= ex.value <= 1.0

    def test_create_examples_without_n_step(self):
        config = ImitationConfig(board_size=8, max_turns=20, use_n_step_returns=False)
        gen = HeuristicExpertGenerator(config)
        trajectory = gen.generate_game()
        for ex in trajectory.examples:
            assert ex.value in (-1.0, 0.0, 1.0)

    def test_n_step_value_near_end(self):
        config = ImitationConfig(n_steps=5, gamma=0.99)
        gen = HeuristicExpertGenerator(config)
        # Create a short history where steps_remaining <= n_steps
        history = [{"player": 0} for _ in range(3)]
        value = gen._compute_n_step_value(history, 0, 0, 0, 5, 0.99)
        # steps_remaining=3 <= n_steps=5, so should return final_value=1.0
        assert value == 1.0

    def test_n_step_value_far_from_end(self):
        config = ImitationConfig(n_steps=3, gamma=0.5)
        gen = HeuristicExpertGenerator(config)
        history = [{"player": 0} for _ in range(20)]
        value = gen._compute_n_step_value(history, 0, 0, 0, 3, 0.5)
        # steps_remaining=20 > n_steps=3, discount = 0.5^3 = 0.125
        assert value == pytest.approx(0.125)

    def test_n_step_value_draw(self):
        config = ImitationConfig()
        gen = HeuristicExpertGenerator(config)
        history = [{"player": 0} for _ in range(20)]
        value = gen._compute_n_step_value(history, 0, 0, None, 5, 0.99)
        # final_value = 0.0 for draw
        assert value == 0.0

    def test_n_step_value_loser(self):
        config = ImitationConfig()
        gen = HeuristicExpertGenerator(config)
        history = [{"player": 0} for _ in range(3)]
        # winner=1, player=0 → final_value=-1.0, steps_remaining=3 <= n_steps=5
        value = gen._compute_n_step_value(history, 0, 0, 1, 5, 0.99)
        assert value == -1.0

    def test_none_expert_seed(self):
        config = ImitationConfig(board_size=8, max_turns=10, expert_seed=None)
        gen = HeuristicExpertGenerator(config)
        assert gen._game_seed == 0  # None or 0


# ---------------------------------------------------------------------------
# SupervisedDataset
# ---------------------------------------------------------------------------


class TestSupervisedDataset:
    def test_from_trajectories(self):
        t1 = _make_trajectory(3, winner=0)
        t2 = _make_trajectory(5, winner=1)
        ds = SupervisedDataset([t1, t2])
        assert len(ds) == 8

    def test_getitem(self):
        t = _make_trajectory(3)
        ds = SupervisedDataset([t])
        item = ds[0]
        assert "spatial" in item
        assert "vector" in item
        assert "action_mask" in item
        assert "policy" in item
        assert "value" in item
        assert isinstance(item["spatial"], torch.Tensor)
        assert isinstance(item["value"], torch.Tensor)

    def test_empty_trajectories(self):
        ds = SupervisedDataset([])
        assert len(ds) == 0

    def test_collate_fn(self):
        t = _make_trajectory(4)
        ds = SupervisedDataset([t])
        batch = [ds[i] for i in range(3)]
        collated = SupervisedDataset.collate_fn(batch)
        assert collated["spatial"].shape[0] == 3
        assert collated["value"].shape[0] == 3


# ---------------------------------------------------------------------------
# ImitationPreTrainer
# ---------------------------------------------------------------------------


class TestImitationPreTrainer:
    def test_init_default(self, tmp_path):
        config = _small_config(tmp_path)
        trainer = ImitationPreTrainer(config=config)
        assert trainer.network is not None
        assert trainer.action_space_size > 0

    def test_init_with_network(self, tmp_path):
        from alpha_zero.core.civ6_network import MicroCivNetwork

        config = _small_config(tmp_path)
        net = MicroCivNetwork(board_size=8, action_space_size=376, num_res_blocks=1, num_filters=16)
        trainer = ImitationPreTrainer(config=config, network=net)
        assert trainer.network is net

    def test_init_with_generator(self, tmp_path):
        config = _small_config(tmp_path)
        mock_gen = MagicMock()
        mock_gen.generate_batch.return_value = [_make_trajectory()]
        trainer = ImitationPreTrainer(config=config, expert_generator=mock_gen)
        assert trainer._generator is mock_gen

    def test_generate_expert_data(self, tmp_path):
        config = _small_config(tmp_path)
        trainer = ImitationPreTrainer(config=config)
        trajectories = trainer.generate_expert_data(n_games=2)
        assert len(trajectories) == 2
        assert trainer.num_games == 2
        assert trainer.num_examples > 0
        assert trainer._dataset is not None

    def test_train_without_data_raises(self, tmp_path):
        config = _small_config(tmp_path)
        trainer = ImitationPreTrainer(config=config)
        with pytest.raises(ValueError, match="No training data"):
            trainer.train()

    def test_train(self, tmp_path):
        config = _small_config(tmp_path)
        trainer = ImitationPreTrainer(config=config)
        trainer.generate_expert_data(n_games=2)
        metrics = trainer.train(epochs=1)
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "policy_accuracy" in metrics
        assert metrics["epoch"] == 1

    def test_train_with_wandb(self, tmp_path):
        config = _small_config(tmp_path)
        mock_wandb = MagicMock()
        trainer = ImitationPreTrainer(config=config, wandb_logger=mock_wandb)
        trainer.generate_expert_data(n_games=2)
        trainer.train(epochs=1)
        mock_wandb.log_imitation_training.assert_called()

    def test_save_and_load_checkpoint(self, tmp_path):
        config = _small_config(tmp_path)
        trainer = ImitationPreTrainer(config=config)
        trainer.generate_expert_data(n_games=1)
        trainer.train(epochs=1)
        path = trainer._save_checkpoint(1)
        assert os.path.exists(path)

    def test_save_pretrained_default_path(self, tmp_path):
        config = _small_config(tmp_path)
        trainer = ImitationPreTrainer(config=config)
        path = trainer.save_pretrained()
        assert os.path.exists(path)
        assert "pretrained.pt" in path

    def test_save_pretrained_custom_path(self, tmp_path):
        config = _small_config(tmp_path)
        trainer = ImitationPreTrainer(config=config)
        path = str(tmp_path / "custom" / "model.pt")
        result = trainer.save_pretrained(path)
        assert os.path.exists(result)

    def test_load_pretrained(self, tmp_path):
        config = _small_config(tmp_path)
        trainer = ImitationPreTrainer(config=config)
        path = trainer.save_pretrained()

        trainer2 = ImitationPreTrainer(config=config)
        trainer2.load_pretrained(path)
        # Verify weights match
        for k in trainer.network.state_dict():
            assert torch.equal(
                trainer.network.state_dict()[k],
                trainer2.network.state_dict()[k],
            )

    def test_get_network(self, tmp_path):
        config = _small_config(tmp_path)
        trainer = ImitationPreTrainer(config=config)
        assert trainer.get_network() is trainer.network

    def test_get_training_history(self, tmp_path):
        config = _small_config(tmp_path)
        trainer = ImitationPreTrainer(config=config)
        assert trainer.get_training_history() == []
        trainer.generate_expert_data(n_games=1)
        trainer.train(epochs=1)
        history = trainer.get_training_history()
        assert len(history) == 1
        # Verify it's a copy
        history.append({"fake": True})
        assert len(trainer.get_training_history()) == 1

    def test_num_examples_no_data(self, tmp_path):
        config = _small_config(tmp_path)
        trainer = ImitationPreTrainer(config=config)
        assert trainer.num_examples == 0

    def test_num_games_no_data(self, tmp_path):
        config = _small_config(tmp_path)
        trainer = ImitationPreTrainer(config=config)
        assert trainer.num_games == 0

    def test_gradient_clip_zero(self, tmp_path):
        config = _small_config(tmp_path)
        config.gradient_clip = 0.0
        trainer = ImitationPreTrainer(config=config)
        trainer.generate_expert_data(n_games=1)
        # Should not crash even with gradient_clip=0
        trainer.train(epochs=1)


# ---------------------------------------------------------------------------
# run_imitation_pretraining
# ---------------------------------------------------------------------------


class TestRunImitationPretraining:
    def test_run(self, tmp_path):
        config = _small_config(tmp_path)
        config.num_games = 2
        network, metrics = run_imitation_pretraining(config=config)
        assert network is not None
        assert "loss" in metrics
