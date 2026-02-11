"""Unit tests for MuZero integration bridge module.

Tests cover:
- MuZeroIntegrationConfig: defaults, validation, serialization
- MuZeroTrainingBridge: component creation, training step execution
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml

from alpha_zero.training.muzero_integration import (
    MuZeroIntegrationConfig,
    MuZeroTrainingBridge,
)

# =============================================================================
# MuZeroIntegrationConfig Tests
# =============================================================================


class TestMuZeroIntegrationConfigDefaults:
    """Test default configuration values."""

    def test_default_values(self):
        """Test that defaults match documented values."""
        config = MuZeroIntegrationConfig()
        assert config.unroll_steps == 5
        assert config.td_steps == 10
        assert config.discount == 0.997
        assert config.use_reanalyze is False
        assert config.reanalyze_fraction == 0.5
        assert config.consistency_loss_weight == 2.0
        assert config.reward_loss_weight == 1.0
        assert config.value_prefix is False

    def test_initialization_with_custom_values(self):
        """Test custom initialization."""
        config = MuZeroIntegrationConfig(
            unroll_steps=10,
            td_steps=20,
            discount=0.99,
            use_reanalyze=True,
        )
        assert config.unroll_steps == 10
        assert config.td_steps == 20
        assert config.discount == 0.99
        assert config.use_reanalyze is True


class TestMuZeroIntegrationConfigValidation:
    """Test configuration validation."""

    def test_negative_unroll_steps_raises_error(self):
        """Test that negative unroll_steps raises ValueError."""
        with pytest.raises(ValueError, match="unroll_steps must be >= 1"):
            MuZeroIntegrationConfig(unroll_steps=0)

    def test_negative_td_steps_raises_error(self):
        """Test that negative td_steps raises ValueError."""
        with pytest.raises(ValueError, match="td_steps must be >= 1"):
            MuZeroIntegrationConfig(td_steps=-5)

    def test_discount_below_zero_raises_error(self):
        """Test that discount < 0 raises ValueError."""
        with pytest.raises(ValueError, match="discount must be in"):
            MuZeroIntegrationConfig(discount=-0.1)

    def test_discount_above_one_raises_error(self):
        """Test that discount > 1 raises ValueError."""
        with pytest.raises(ValueError, match="discount must be in"):
            MuZeroIntegrationConfig(discount=1.5)

    def test_reanalyze_fraction_out_of_range_raises_error(self):
        """Test that reanalyze_fraction outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="reanalyze_fraction must be"):
            MuZeroIntegrationConfig(reanalyze_fraction=1.2)

    def test_negative_consistency_loss_weight_raises_error(self):
        """Test that negative consistency_loss_weight raises ValueError."""
        with pytest.raises(ValueError, match="consistency_loss_weight"):
            MuZeroIntegrationConfig(consistency_loss_weight=-1.0)

    def test_negative_reward_loss_weight_raises_error(self):
        """Test that negative reward_loss_weight raises ValueError."""
        with pytest.raises(ValueError, match="reward_loss_weight"):
            MuZeroIntegrationConfig(reward_loss_weight=-2.0)


class TestMuZeroIntegrationConfigSerialization:
    """Test configuration serialization and deserialization."""

    def test_to_dict_round_trip(self):
        """Test to_dict contains all fields."""
        config = MuZeroIntegrationConfig(
            unroll_steps=7,
            td_steps=15,
            discount=0.95,
        )
        config_dict = config.to_dict()

        assert config_dict["unroll_steps"] == 7
        assert config_dict["td_steps"] == 15
        assert config_dict["discount"] == 0.95
        assert config_dict["use_reanalyze"] is False
        assert config_dict["reanalyze_fraction"] == 0.5
        assert config_dict["consistency_loss_weight"] == 2.0
        assert config_dict["reward_loss_weight"] == 1.0
        assert config_dict["value_prefix"] is False

    def test_from_dict_reconstructs_config(self):
        """Test from_dict reconstructs config correctly."""
        data = {
            "unroll_steps": 8,
            "td_steps": 12,
            "discount": 0.98,
            "use_reanalyze": True,
            "reanalyze_fraction": 0.6,
            "consistency_loss_weight": 3.0,
            "reward_loss_weight": 1.5,
            "value_prefix": True,
        }
        config = MuZeroIntegrationConfig.from_dict(data)

        assert config.unroll_steps == 8
        assert config.td_steps == 12
        assert config.discount == 0.98
        assert config.use_reanalyze is True
        assert config.reanalyze_fraction == 0.6
        assert config.consistency_loss_weight == 3.0
        assert config.reward_loss_weight == 1.5
        assert config.value_prefix is True

    def test_from_dict_with_missing_keys_uses_defaults(self):
        """Test from_dict uses defaults for missing keys."""
        data = {"unroll_steps": 3}
        config = MuZeroIntegrationConfig.from_dict(data)

        assert config.unroll_steps == 3
        assert config.td_steps == 10  # default
        assert config.discount == 0.997  # default

    def test_to_dict_from_dict_round_trip(self):
        """Test serialization round trip preserves values."""
        original = MuZeroIntegrationConfig(
            unroll_steps=6,
            td_steps=18,
            use_reanalyze=True,
        )
        reconstructed = MuZeroIntegrationConfig.from_dict(original.to_dict())

        assert reconstructed.unroll_steps == original.unroll_steps
        assert reconstructed.td_steps == original.td_steps
        assert reconstructed.use_reanalyze == original.use_reanalyze


class TestMuZeroIntegrationConfigFromEnv:
    """Test configuration from environment variables."""

    def test_from_env_with_defaults(self):
        """Test from_env uses defaults when env vars not set."""
        # Clear any existing vars
        for key in os.environ.copy():
            if key.startswith("CIV6AZ_MUZERO_"):
                del os.environ[key]

        config = MuZeroIntegrationConfig.from_env()
        assert config.unroll_steps == 5
        assert config.td_steps == 10

    def test_from_env_reads_unroll_steps(self):
        """Test from_env reads CIV6AZ_MUZERO_UNROLL_STEPS."""
        os.environ["CIV6AZ_MUZERO_UNROLL_STEPS"] = "8"
        try:
            config = MuZeroIntegrationConfig.from_env()
            assert config.unroll_steps == 8
        finally:
            del os.environ["CIV6AZ_MUZERO_UNROLL_STEPS"]

    def test_from_env_reads_td_steps(self):
        """Test from_env reads CIV6AZ_MUZERO_TD_STEPS."""
        os.environ["CIV6AZ_MUZERO_TD_STEPS"] = "15"
        try:
            config = MuZeroIntegrationConfig.from_env()
            assert config.td_steps == 15
        finally:
            del os.environ["CIV6AZ_MUZERO_TD_STEPS"]

    def test_from_env_reads_discount(self):
        """Test from_env reads CIV6AZ_MUZERO_DISCOUNT."""
        os.environ["CIV6AZ_MUZERO_DISCOUNT"] = "0.99"
        try:
            config = MuZeroIntegrationConfig.from_env()
            assert config.discount == 0.99
        finally:
            del os.environ["CIV6AZ_MUZERO_DISCOUNT"]

    def test_from_env_reads_boolean_reanalyze(self):
        """Test from_env parses boolean CIV6AZ_MUZERO_REANALYZE."""
        os.environ["CIV6AZ_MUZERO_REANALYZE"] = "true"
        try:
            config = MuZeroIntegrationConfig.from_env()
            assert config.use_reanalyze is True
        finally:
            del os.environ["CIV6AZ_MUZERO_REANALYZE"]


class TestMuZeroIntegrationConfigYAML:
    """Test YAML serialization."""

    def test_to_yaml_creates_file(self):
        """Test to_yaml creates a valid YAML file."""
        config = MuZeroIntegrationConfig(unroll_steps=4, td_steps=8)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml_path = f.name

        try:
            config.to_yaml(yaml_path)
            assert Path(yaml_path).exists()

            # Verify content
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            assert data["unroll_steps"] == 4
            assert data["td_steps"] == 8
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_loads_config(self):
        """Test from_yaml loads configuration from file."""
        data = {
            "unroll_steps": 9,
            "td_steps": 16,
            "discount": 0.995,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            yaml_path = f.name

        try:
            config = MuZeroIntegrationConfig.from_yaml(yaml_path)
            assert config.unroll_steps == 9
            assert config.td_steps == 16
            assert config.discount == 0.995
        finally:
            Path(yaml_path).unlink()


# =============================================================================
# MuZeroTrainingBridge Tests
# =============================================================================


class TestMuZeroTrainingBridgeInit:
    """Test MuZeroTrainingBridge initialization."""

    def test_default_initialization(self):
        """Test bridge initializes with default config."""
        bridge = MuZeroTrainingBridge()
        assert bridge.config is not None
        assert isinstance(bridge.config, MuZeroIntegrationConfig)
        assert bridge.mcts_factory_fn is None

    def test_initialization_with_config(self):
        """Test bridge initializes with custom config."""
        config = MuZeroIntegrationConfig(unroll_steps=7)
        bridge = MuZeroTrainingBridge(config=config)
        assert bridge.config.unroll_steps == 7

    def test_initialization_with_factory_function(self):
        """Test bridge accepts factory function."""
        mock_factory = Mock()
        bridge = MuZeroTrainingBridge(mcts_factory_fn=mock_factory)
        assert bridge.mcts_factory_fn is mock_factory


@patch("alpha_zero.training.muzero_integration.TORCH_AVAILABLE", True)
class TestMuZeroTrainingBridgeCreateNetwork:
    """Test network creation."""

    def test_create_network_calls_muzero_network(self):
        """Test create_network instantiates MuZeroNetwork."""
        bridge = MuZeroTrainingBridge()

        with patch(
            "alpha_zero.core.muzero_network.MuZeroNetwork"
        ) as mock_network_cls:
            mock_network = Mock()
            mock_network_cls.return_value = mock_network

            result = bridge.create_network(
                spatial_channels=13,
                vector_dim=64,
                action_size=376,
            )

            # Verify MuZeroNetwork was called with correct args
            mock_network_cls.assert_called_once()
            call_kwargs = mock_network_cls.call_args.kwargs
            assert call_kwargs["spatial_channels"] == 13
            assert call_kwargs["vector_dim"] == 64
            assert call_kwargs["action_size"] == 376
            assert call_kwargs["discount"] == 0.997
            assert result is mock_network

    def test_create_network_passes_extra_kwargs(self):
        """Test create_network forwards additional kwargs."""
        bridge = MuZeroTrainingBridge()

        with patch(
            "alpha_zero.core.muzero_network.MuZeroNetwork"
        ) as mock_network_cls:
            bridge.create_network(
                spatial_channels=13,
                vector_dim=64,
                action_size=376,
                num_res_blocks=8,
            )

            call_kwargs = mock_network_cls.call_args.kwargs
            assert call_kwargs["num_res_blocks"] == 8


@patch("alpha_zero.training.muzero_integration.TORCH_AVAILABLE", True)
class TestMuZeroTrainingBridgeCreateTrainer:
    """Test trainer creation."""

    def test_create_trainer_calls_muzero_trainer(self):
        """Test create_trainer instantiates MuZeroTrainer."""
        bridge = MuZeroTrainingBridge()
        mock_network = Mock()

        with patch(
            "alpha_zero.training.muzero_trainer.MuZeroTrainer"
        ) as mock_trainer_cls:
            mock_trainer = Mock()
            mock_trainer_cls.return_value = mock_trainer

            result = bridge.create_trainer(
                network=mock_network,
                device="cpu",
            )

            mock_trainer_cls.assert_called_once()
            call_args = mock_trainer_cls.call_args
            assert call_args.kwargs["network"] is mock_network
            assert call_args.kwargs["device"] == "cpu"
            assert result is mock_trainer

    def test_create_trainer_builds_training_config(self):
        """Test create_trainer converts integration config to training."""
        config = MuZeroIntegrationConfig(
            unroll_steps=6,
            td_steps=12,
            consistency_loss_weight=3.0,
        )
        bridge = MuZeroTrainingBridge(config=config)
        mock_network = Mock()

        with patch(
            "alpha_zero.training.muzero_trainer.MuZeroTrainer"
        ) as mock_trainer_cls:
            bridge.create_trainer(network=mock_network)

            # Get the config passed to trainer
            training_config = mock_trainer_cls.call_args.kwargs["config"]
            assert training_config.unroll_steps == 6
            assert training_config.td_steps == 12
            assert training_config.consistency_loss_weight == 3.0


class TestMuZeroTrainingBridgeCreateMCTS:
    """Test MCTS creation."""

    def test_create_mcts_with_factory_function(self):
        """Test create_mcts uses factory function when provided."""
        mock_factory = Mock(return_value=Mock())
        bridge = MuZeroTrainingBridge(mcts_factory_fn=mock_factory)
        mock_network = Mock()

        result = bridge.create_mcts(
            network=mock_network,
            num_simulations=100,
            c_puct=1.5,
            device="cpu",
        )

        mock_factory.assert_called_once_with(
            variant="muzero",
            network=mock_network,
            num_simulations=100,
            c_puct=1.5,
        )
        assert result is mock_factory.return_value

    @patch("alpha_zero.core.civ6_mcts.MuZeroMCTSConfig")
    @patch("alpha_zero.core.civ6_mcts.MuZeroMCTS")
    def test_create_mcts_direct_instantiation(
        self, mock_mcts_cls, mock_config_cls
    ):
        """Test create_mcts creates MuZeroMCTS directly when no factory."""
        bridge = MuZeroTrainingBridge()
        mock_network = Mock()
        mock_config = Mock()
        mock_config_cls.return_value = mock_config
        mock_mcts = Mock()
        mock_mcts_cls.return_value = mock_mcts

        result = bridge.create_mcts(
            network=mock_network,
            num_simulations=80,
            c_puct=2.0,
            device="cuda",
        )

        # Verify config creation
        mock_config_cls.assert_called_once_with(
            num_simulations=80,
            c_puct=2.0,
            discount=0.997,
        )

        # Verify MCTS instantiation
        mock_mcts_cls.assert_called_once_with(
            network=mock_network,
            config=mock_config,
            device="cuda",
        )
        assert result is mock_mcts

    @patch("alpha_zero.core.civ6_mcts.MuZeroMCTSConfig")
    @patch("alpha_zero.core.civ6_mcts.MuZeroMCTS")
    def test_create_mcts_uses_config_discount(self, mock_mcts_cls, mock_config_cls):
        """Test create_mcts passes discount from integration config."""
        config = MuZeroIntegrationConfig(discount=0.99)
        bridge = MuZeroTrainingBridge(config=config)
        mock_network = Mock()

        bridge.create_mcts(network=mock_network)

        # Verify discount was passed to MCTS config
        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs["discount"] == 0.99


@patch("alpha_zero.training.muzero_integration.TORCH_AVAILABLE", True)
class TestMuZeroTrainingBridgeToTrainingConfig:
    """Test training config conversion."""

    def test_to_training_config_maps_fields_correctly(self):
        """Test to_training_config maps integration config fields."""
        config = MuZeroIntegrationConfig(
            unroll_steps=7,
            td_steps=14,
            discount=0.98,
            consistency_loss_weight=2.5,
            reward_loss_weight=1.2,
            value_prefix=True,
        )
        bridge = MuZeroTrainingBridge(config=config)

        with patch(
            "alpha_zero.training.muzero_trainer.MuZeroTrainingConfig"
        ) as mock_config_cls:
            bridge.to_training_config()

            mock_config_cls.assert_called_once()
            call_kwargs = mock_config_cls.call_args.kwargs
            assert call_kwargs["unroll_steps"] == 7
            assert call_kwargs["td_steps"] == 14
            assert call_kwargs["discount"] == 0.98
            assert call_kwargs["consistency_loss_weight"] == 2.5
            assert call_kwargs["reward_loss_weight"] == 1.2
            assert call_kwargs["use_value_prefix"] is True

    def test_to_training_config_with_overrides(self):
        """Test to_training_config applies overrides."""
        bridge = MuZeroTrainingBridge()

        with patch(
            "alpha_zero.training.muzero_trainer.MuZeroTrainingConfig"
        ) as mock_config_cls:
            bridge.to_training_config(
                batch_size=128,
                learning_rate=0.0005,
            )

            call_kwargs = mock_config_cls.call_args.kwargs
            assert call_kwargs["batch_size"] == 128
            assert call_kwargs["learning_rate"] == 0.0005
            # Base values still present
            assert call_kwargs["unroll_steps"] == 5


@patch("alpha_zero.training.muzero_integration.TORCH_AVAILABLE", True)
class TestMuZeroTrainingBridgeSelfplayStep:
    """Test self-play step execution."""

    def test_selfplay_step_produces_trajectory(self):
        """Test selfplay_step returns GameTrajectory."""
        bridge = MuZeroTrainingBridge()
        mock_network = Mock()
        mock_mcts = Mock()
        mock_env = Mock()

        # Mock environment behavior
        mock_obs = {
            "spatial": np.zeros((13, 8, 8)),
            "vector": np.zeros(64),
        }
        mock_env.reset.return_value = mock_obs
        mock_env.action_mask.return_value = np.ones(376, dtype=bool)
        mock_env.step.return_value = (
            mock_obs,
            0.0,
            True,
            {},
        )

        # Mock MCTS behavior
        mock_mcts.search.return_value = (
            np.zeros(376),
            0.5,
        )
        mock_mcts.select_action.return_value = 0

        with patch(
            "alpha_zero.training.muzero_trainer.GameTrajectory"
        ) as mock_trajectory_cls:
            mock_trajectory = Mock()
            mock_trajectory_cls.return_value = mock_trajectory

            result = bridge.selfplay_step(
                network=mock_network,
                env=mock_env,
                mcts=mock_mcts,
                temperature=1.0,
                max_moves=200,
            )

            mock_trajectory_cls.assert_called_once()
            assert result is mock_trajectory

    def test_selfplay_step_respects_max_moves(self):
        """Test selfplay_step stops after max_moves."""
        bridge = MuZeroTrainingBridge()
        mock_network = Mock()
        mock_mcts = Mock()
        mock_env = Mock()

        # Mock environment to never terminate
        mock_obs = {"spatial": np.zeros((13, 8, 8)), "vector": np.zeros(64)}
        mock_env.reset.return_value = mock_obs
        mock_env.action_mask.return_value = np.ones(376, dtype=bool)
        mock_env.step.return_value = (mock_obs, 0.0, False, {})

        mock_mcts.search.return_value = (np.zeros(376), 0.5)
        mock_mcts.select_action.return_value = 0

        with patch(
            "alpha_zero.training.muzero_trainer.GameTrajectory"
        ):
            bridge.selfplay_step(
                network=mock_network,
                env=mock_env,
                mcts=mock_mcts,
                max_moves=10,
            )

            # Should have called step exactly 10 times
            assert mock_env.step.call_count == 10

    def test_selfplay_step_handles_action_mask_from_env(self):
        """Test selfplay_step gets action_mask from env.action_mask()."""
        bridge = MuZeroTrainingBridge()
        mock_network = Mock()
        mock_mcts = Mock()
        mock_env = Mock()

        # Mock with action_mask method
        mock_obs = {"spatial": np.zeros((13, 8, 8)), "vector": np.zeros(64)}
        action_mask = np.ones(376, dtype=bool)
        mock_env.reset.return_value = mock_obs
        mock_env.action_mask.return_value = action_mask
        mock_env.step.return_value = (mock_obs, 0.0, True, {})

        mock_mcts.search.return_value = (np.zeros(376), 0.5)
        mock_mcts.select_action.return_value = 0

        with patch(
            "alpha_zero.training.muzero_trainer.GameTrajectory"
        ):
            bridge.selfplay_step(
                network=mock_network,
                env=mock_env,
                mcts=mock_mcts,
            )

            # Verify action_mask was retrieved
            mock_env.action_mask.assert_called()


class TestMuZeroTrainingBridgeTrainingStep:
    """Test training step execution."""

    def test_training_step_returns_metrics_dict(self):
        """Test training_step returns dict with metrics."""
        bridge = MuZeroTrainingBridge()
        mock_trainer = Mock()
        mock_batch = Mock()

        # Mock trainer return value
        mock_trainer.train_step.return_value = {
            "total_loss": 1.5,
            "policy_loss": 0.5,
            "value_loss": 1.0,
        }

        result = bridge.training_step(
            trainer=mock_trainer,
            batch=mock_batch,
            step=42,
        )

        mock_trainer.train_step.assert_called_once_with(mock_batch)
        assert isinstance(result, dict)
        assert result["total_loss"] == 1.5
        assert result["policy_loss"] == 0.5
        assert result["value_loss"] == 1.0

    def test_training_step_adds_step_to_metrics(self):
        """Test training_step adds step number to metrics."""
        bridge = MuZeroTrainingBridge()
        mock_trainer = Mock()
        mock_trainer.train_step.return_value = {"loss": 2.0}

        result = bridge.training_step(
            trainer=mock_trainer,
            batch=Mock(),
            step=100,
        )

        assert "step" in result
        assert result["step"] == 100.0
