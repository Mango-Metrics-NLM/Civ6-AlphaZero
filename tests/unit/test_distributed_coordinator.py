"""Tests for distributed training coordinator.

Tests cover:
- CoordinatorConfig defaults and YAML loading
- TrainingCoordinator initialization
- Shutdown behavior
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest
import yaml

from alpha_zero.distributed.coordinator import (
    CoordinatorConfig,
    TrainingCoordinator,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def coordinator_config() -> CoordinatorConfig:
    """Create a minimal coordinator configuration for testing."""
    return CoordinatorConfig(
        board_size=8,
        max_turns=50,
        num_res_blocks=2,
        num_filters=16,
        num_simulations=10,
        num_actors=2,
        batch_size=32,
        replay_buffer_size=1000,
        min_replay_size=100,
        max_games=5,
        max_training_steps=10,
        device="cpu",
    )


@pytest.fixture
def sample_yaml_config() -> str:
    """Create a sample YAML configuration."""
    config = {
        "environment": {
            "board_size": 8,
            "max_turns": 100,
        },
        "network": {
            "num_res_blocks": 4,
            "num_filters": 32,
        },
        "mcts": {
            "num_simulations": 25,
            "c_puct": 2.0,
            "dirichlet_alpha": 0.5,
        },
        "training": {
            "batch_size": 128,
            "learning_rate": 0.002,
            "weight_decay": 0.0002,
            "replay_buffer_size": 100000,
            "min_replay_size": 5000,
        },
        "selfplay": {
            "num_actors": 8,
            "games_per_checkpoint": 100,
        },
    }
    return yaml.dump(config)


# =============================================================================
# CoordinatorConfig Tests
# =============================================================================


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CoordinatorConfig()

        # Environment
        assert config.board_size == 16
        assert config.max_turns == 200

        # Network
        assert config.num_res_blocks == 6
        assert config.num_filters == 64

        # MCTS
        assert config.num_simulations == 50
        assert config.c_puct == 1.5
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25
        assert config.temperature == 1.0
        assert config.temperature_threshold == 30

        # Distributed
        assert config.num_actors == 64
        assert config.games_per_checkpoint == 500

        # Training
        assert config.batch_size == 256
        assert config.learning_rate == 0.001
        assert config.weight_decay == 0.0001
        assert config.gradient_clip == 1.0
        assert config.lr_decay_steps == [100000, 300000, 500000]
        assert config.lr_decay_rate == 0.1

        # Replay buffer
        assert config.replay_buffer_size == 500_000
        assert config.min_replay_size == 10_000

        # Limits
        assert config.max_games == 0
        assert config.max_training_steps == 0

        # Checkpointing
        assert config.checkpoint_dir == "checkpoints/civ6"
        assert config.log_dir == "logs/civ6"
        assert config.save_interval == 1000

        # Evaluation
        assert config.eval_interval == 500
        assert config.eval_games == 20

        # Monitoring
        assert config.monitor_interval == 10.0
        assert config.restart_dead_actors is True
        assert config.actor_pool_stop_timeout == 10.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = CoordinatorConfig(
            board_size=8,
            num_actors=4,
            batch_size=64,
            learning_rate=0.005,
            max_games=100,
            device="cuda:1",
        )

        assert config.board_size == 8
        assert config.num_actors == 4
        assert config.batch_size == 64
        assert config.learning_rate == 0.005
        assert config.max_games == 100
        assert config.device == "cuda:1"

    def test_lr_decay_steps_mutable_default(self) -> None:
        """Test that lr_decay_steps default is properly isolated."""
        config1 = CoordinatorConfig()
        config2 = CoordinatorConfig()

        # Modify config1's lr_decay_steps
        config1.lr_decay_steps.append(999)

        # config2 should not be affected
        assert 999 not in config2.lr_decay_steps

    def test_from_yaml(self, sample_yaml_config: str) -> None:
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(sample_yaml_config)
            temp_path = f.name

        try:
            config = CoordinatorConfig.from_yaml(temp_path)

            # Check values from YAML
            assert config.board_size == 8
            assert config.max_turns == 100
            assert config.num_res_blocks == 4
            assert config.num_filters == 32
            assert config.num_simulations == 25
            assert config.c_puct == 2.0
            assert config.dirichlet_alpha == 0.5
            assert config.batch_size == 128
            assert config.learning_rate == 0.002
            assert config.num_actors == 8
        finally:
            os.unlink(temp_path)

    def test_from_yaml_partial(self) -> None:
        """Test YAML loading with partial config uses defaults."""
        partial_config = {"environment": {"board_size": 12}}
        yaml_str = yaml.dump(partial_config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_str)
            temp_path = f.name

        try:
            config = CoordinatorConfig.from_yaml(temp_path)

            # Specified value
            assert config.board_size == 12

            # Default values
            assert config.num_res_blocks == 6
            assert config.num_simulations == 50
        finally:
            os.unlink(temp_path)

    def test_from_yaml_with_microciv_override(self) -> None:
        """Test YAML loading with microciv section overrides."""
        config_dict = {
            "environment": {"board_size": 16},
            "microciv": {
                "board_size": 8,
                "num_res_blocks": 2,
                "num_simulations": 20,
            },
        }
        yaml_str = yaml.dump(config_dict)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_str)
            temp_path = f.name

        try:
            config = CoordinatorConfig.from_yaml(temp_path)

            # MicroCiv values should override
            assert config.board_size == 8
            assert config.num_res_blocks == 2
            assert config.num_simulations == 20
        finally:
            os.unlink(temp_path)

    def test_config_to_dict(self) -> None:
        """Test config can be converted to dict."""
        config = CoordinatorConfig(num_actors=10)
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert config_dict["num_actors"] == 10
        assert "batch_size" in config_dict


# =============================================================================
# TrainingCoordinator Tests
# =============================================================================


class TestTrainingCoordinator:
    """Tests for TrainingCoordinator class.

    These tests mock heavy dependencies to test coordinator logic
    without spawning processes or loading real networks.
    """

    @patch("alpha_zero.distributed.coordinator.mp.set_start_method")
    @patch("alpha_zero.distributed.coordinator.SharedStorage")
    @patch("alpha_zero.distributed.coordinator.DistributedReplayBuffer")
    def test_initialization(
        self,
        mock_buffer_class: MagicMock,
        mock_storage_class: MagicMock,
        mock_set_start_method: MagicMock,
        coordinator_config: CoordinatorConfig,
    ) -> None:
        """Test TrainingCoordinator initializes correctly."""
        # Setup mocks
        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage

        mock_buffer = MagicMock()
        mock_buffer_class.return_value = mock_buffer

        # Use temp directories
        with tempfile.TemporaryDirectory() as tmpdir:
            coordinator_config.checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            coordinator_config.log_dir = os.path.join(tmpdir, "logs")

            coordinator = TrainingCoordinator(coordinator_config)

            # Verify initialization
            assert coordinator.config == coordinator_config
            assert coordinator.shared_storage is mock_storage
            assert coordinator.replay_buffer is mock_buffer
            assert coordinator.actor_pool is None
            assert coordinator.learner is None
            assert coordinator.eval_network is None
            assert coordinator._running is False
            assert coordinator._shutdown_requested is False

            # Verify directories created
            assert os.path.exists(coordinator_config.checkpoint_dir)
            assert os.path.exists(coordinator_config.log_dir)

    @patch("alpha_zero.distributed.coordinator.mp.set_start_method")
    @patch("alpha_zero.distributed.coordinator.SharedStorage")
    @patch("alpha_zero.distributed.coordinator.DistributedReplayBuffer")
    def test_shutdown_cleans_up_resources(
        self,
        mock_buffer_class: MagicMock,
        mock_storage_class: MagicMock,
        mock_set_start_method: MagicMock,
        coordinator_config: CoordinatorConfig,
    ) -> None:
        """Test shutdown properly cleans up resources."""
        mock_storage = MagicMock()
        mock_storage.get_games_played.return_value = 10
        mock_storage_class.return_value = mock_storage

        mock_buffer = MagicMock()
        mock_buffer_class.return_value = mock_buffer

        with tempfile.TemporaryDirectory() as tmpdir:
            coordinator_config.checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            coordinator_config.log_dir = os.path.join(tmpdir, "logs")

            coordinator = TrainingCoordinator(coordinator_config)

            # Call shutdown
            coordinator.shutdown()

            # Verify cleanup
            assert coordinator._running is False
            mock_storage.request_stop.assert_called()
            mock_buffer.shutdown.assert_called()
            mock_storage.shutdown.assert_called()

    @patch("alpha_zero.distributed.coordinator.mp.set_start_method")
    @patch("alpha_zero.distributed.coordinator.SharedStorage")
    @patch("alpha_zero.distributed.coordinator.DistributedReplayBuffer")
    @patch("alpha_zero.distributed.coordinator.MicroCivNetwork")
    @patch("alpha_zero.distributed.coordinator.MicroCivEnv")
    def test_init_evaluation_network(
        self,
        mock_env_class: MagicMock,
        mock_network_class: MagicMock,
        mock_buffer_class: MagicMock,
        mock_storage_class: MagicMock,
        mock_set_start_method: MagicMock,
        coordinator_config: CoordinatorConfig,
    ) -> None:
        """Test _init_evaluation_network creates network."""
        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage

        mock_buffer = MagicMock()
        mock_buffer_class.return_value = mock_buffer

        mock_env = MagicMock()
        mock_env.action_space.n = 100
        mock_env_class.return_value = mock_env

        mock_network = MagicMock()
        mock_network.to.return_value = mock_network
        mock_network_class.return_value = mock_network

        with tempfile.TemporaryDirectory() as tmpdir:
            coordinator_config.checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            coordinator_config.log_dir = os.path.join(tmpdir, "logs")

            coordinator = TrainingCoordinator(coordinator_config)
            coordinator._init_evaluation_network()

            assert coordinator.eval_network is mock_network
            mock_network.to.assert_called_with(coordinator_config.device)
            mock_network.eval.assert_called()

    @patch("alpha_zero.distributed.coordinator.mp.set_start_method")
    @patch("alpha_zero.distributed.coordinator.SharedStorage")
    @patch("alpha_zero.distributed.coordinator.DistributedReplayBuffer")
    def test_update_eval_network_no_network(
        self,
        mock_buffer_class: MagicMock,
        mock_storage_class: MagicMock,
        mock_set_start_method: MagicMock,
        coordinator_config: CoordinatorConfig,
    ) -> None:
        """Test _update_eval_network returns False with no network."""
        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage

        mock_buffer = MagicMock()
        mock_buffer_class.return_value = mock_buffer

        with tempfile.TemporaryDirectory() as tmpdir:
            coordinator_config.checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            coordinator_config.log_dir = os.path.join(tmpdir, "logs")

            coordinator = TrainingCoordinator(coordinator_config)

            # No eval network yet
            assert coordinator._update_eval_network() is False

    @patch("alpha_zero.distributed.coordinator.mp.set_start_method")
    @patch("alpha_zero.distributed.coordinator.SharedStorage")
    @patch("alpha_zero.distributed.coordinator.DistributedReplayBuffer")
    def test_update_eval_network_no_weights(
        self,
        mock_buffer_class: MagicMock,
        mock_storage_class: MagicMock,
        mock_set_start_method: MagicMock,
        coordinator_config: CoordinatorConfig,
    ) -> None:
        """Test _update_eval_network returns False with no weights."""
        mock_storage = MagicMock()
        mock_storage.get_model_weights.return_value = None
        mock_storage_class.return_value = mock_storage

        mock_buffer = MagicMock()
        mock_buffer_class.return_value = mock_buffer

        with tempfile.TemporaryDirectory() as tmpdir:
            coordinator_config.checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            coordinator_config.log_dir = os.path.join(tmpdir, "logs")

            coordinator = TrainingCoordinator(coordinator_config)
            coordinator.eval_network = MagicMock()  # Fake network

            assert coordinator._update_eval_network() is False

    @patch("alpha_zero.distributed.coordinator.mp.set_start_method")
    @patch("alpha_zero.distributed.coordinator.SharedStorage")
    @patch("alpha_zero.distributed.coordinator.DistributedReplayBuffer")
    def test_actor_config_created(
        self,
        mock_buffer_class: MagicMock,
        mock_storage_class: MagicMock,
        mock_set_start_method: MagicMock,
        coordinator_config: CoordinatorConfig,
    ) -> None:
        """Test actor_config is created with correct values."""
        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage

        mock_buffer = MagicMock()
        mock_buffer_class.return_value = mock_buffer

        with tempfile.TemporaryDirectory() as tmpdir:
            coordinator_config.checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            coordinator_config.log_dir = os.path.join(tmpdir, "logs")

            coordinator = TrainingCoordinator(coordinator_config)

            # Verify actor config
            assert coordinator.actor_config.board_size == coordinator_config.board_size
            assert coordinator.actor_config.max_turns == coordinator_config.max_turns
            assert coordinator.actor_config.num_simulations == coordinator_config.num_simulations
            assert coordinator.actor_config.c_puct == coordinator_config.c_puct
            assert coordinator.actor_config.device == "cpu"  # Actors on CPU

    @patch("alpha_zero.distributed.coordinator.mp.set_start_method")
    @patch("alpha_zero.distributed.coordinator.SharedStorage")
    @patch("alpha_zero.distributed.coordinator.DistributedReplayBuffer")
    def test_learner_config_created(
        self,
        mock_buffer_class: MagicMock,
        mock_storage_class: MagicMock,
        mock_set_start_method: MagicMock,
        coordinator_config: CoordinatorConfig,
    ) -> None:
        """Test learner_config is created with correct values."""
        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage

        mock_buffer = MagicMock()
        mock_buffer_class.return_value = mock_buffer

        with tempfile.TemporaryDirectory() as tmpdir:
            coordinator_config.checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            coordinator_config.log_dir = os.path.join(tmpdir, "logs")

            coordinator = TrainingCoordinator(coordinator_config)

            # Verify learner config
            assert coordinator.learner_config.board_size == coordinator_config.board_size
            assert coordinator.learner_config.batch_size == coordinator_config.batch_size
            assert coordinator.learner_config.learning_rate == coordinator_config.learning_rate
            assert coordinator.learner_config.weight_decay == coordinator_config.weight_decay
            assert coordinator.learner_config.device == coordinator_config.device
