"""Unit tests for auxiliary policy targets."""

import numpy as np
import pytest

from alpha_zero.optimizations.auxiliary_targets import (
    AuxiliaryLossCalculator,
    AuxiliaryTargetGenerator,
    AuxiliaryTargets,
    AuxiliaryTargetsConfig,
    GamePhase,
    create_auxiliary_system,
)


class TestAuxiliaryTargetsConfig:
    """Tests for AuxiliaryTargetsConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AuxiliaryTargetsConfig()
        assert config.enable_attention_targets is True
        assert config.enable_opponent_modeling is True
        assert config.attention_weight == 0.1
        assert config.opponent_weight == 0.2

    def test_validation_weights(self):
        """Test validation of loss weights."""
        with pytest.raises(ValueError, match="attention_weight must be >= 0"):
            AuxiliaryTargetsConfig(attention_weight=-0.1)
        with pytest.raises(ValueError, match="opponent_weight must be >= 0"):
            AuxiliaryTargetsConfig(opponent_weight=-0.1)
        with pytest.raises(ValueError, match="phase_weight must be >= 0"):
            AuxiliaryTargetsConfig(phase_weight=-0.1)
        with pytest.raises(ValueError, match="future_value_weight must be >= 0"):
            AuxiliaryTargetsConfig(future_value_weight=-0.1)

    def test_validation_future_value_steps(self):
        """Test validation of future_value_steps."""
        with pytest.raises(ValueError, match="future_value_steps must be >= 1"):
            AuxiliaryTargetsConfig(future_value_steps=0)

    def test_validation_phase_thresholds(self):
        """Test validation of phase_thresholds."""
        with pytest.raises(ValueError, match="phase_thresholds must have exactly 2"):
            AuxiliaryTargetsConfig(phase_thresholds=(10,))
        with pytest.raises(ValueError, match=r"phase_thresholds.*must be <"):
            AuxiliaryTargetsConfig(phase_thresholds=(50, 20))

    def test_validation_attention_blur_sigma(self):
        """Test validation of attention_blur_sigma."""
        with pytest.raises(ValueError, match="attention_blur_sigma must be >= 0"):
            AuxiliaryTargetsConfig(attention_blur_sigma=-1)

    def test_validation_opponent_history_length(self):
        """Test validation of opponent_history_length."""
        with pytest.raises(ValueError, match="opponent_history_length must be >= 1"):
            AuxiliaryTargetsConfig(opponent_history_length=0)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "enable_attention_targets": False,
            "attention_weight": 0.2,
            "phase_thresholds": [15, 45],
        }
        config = AuxiliaryTargetsConfig.from_dict(data)
        assert config.enable_attention_targets is False
        assert config.attention_weight == 0.2
        assert config.phase_thresholds == (15, 45)

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = AuxiliaryTargetsConfig(enable_future_value=False)
        data = config.to_dict()
        assert data["enable_future_value"] is False
        assert isinstance(data["phase_thresholds"], list)

    def test_get_total_weight(self):
        """Test computing total weight."""
        config = AuxiliaryTargetsConfig(
            enable_attention_targets=True,
            enable_opponent_modeling=True,
            enable_phase_prediction=False,
            enable_future_value=False,
            attention_weight=0.1,
            opponent_weight=0.2,
        )
        assert config.get_total_weight() == pytest.approx(0.3)

    def test_any_enabled(self):
        """Test checking if any target is enabled."""
        config_all = AuxiliaryTargetsConfig()
        assert config_all.any_enabled() is True

        config_none = AuxiliaryTargetsConfig(
            enable_attention_targets=False,
            enable_opponent_modeling=False,
            enable_phase_prediction=False,
            enable_future_value=False,
        )
        assert config_none.any_enabled() is False


class TestGamePhase:
    """Tests for GamePhase enum."""

    def test_phases_exist(self):
        """Test all phases are defined."""
        assert GamePhase.EARLY.value == "early"
        assert GamePhase.MID.value == "mid"
        assert GamePhase.LATE.value == "late"


class TestAuxiliaryTargets:
    """Tests for AuxiliaryTargets dataclass."""

    def test_default_values(self):
        """Test default values are None."""
        targets = AuxiliaryTargets()
        assert targets.attention_map is None
        assert targets.opponent_action is None
        assert targets.game_phase is None
        assert targets.future_value is None

    def test_to_dict(self):
        """Test converting to dictionary."""
        targets = AuxiliaryTargets(
            attention_map=np.ones((4, 4)) / 16,
            future_value=0.5,
        )
        data = targets.to_dict()
        assert "attention_map" in data
        assert data["future_value"] == 0.5
        assert "opponent_action" not in data

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "attention_map": [[0.25] * 4] * 4,
            "future_value": -0.3,
        }
        targets = AuxiliaryTargets.from_dict(data)
        assert targets.attention_map is not None
        assert targets.attention_map.shape == (4, 4)
        assert targets.future_value == -0.3


class TestAuxiliaryTargetGenerator:
    """Tests for AuxiliaryTargetGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator with default config."""
        return AuxiliaryTargetGenerator(
            config=AuxiliaryTargetsConfig(),
            board_size=8,
            action_space_size=100,
            seed=42,
        )

    @pytest.fixture
    def sample_state(self):
        """Create sample game state."""
        return {
            "board": np.random.rand(8, 8),
            "turn": 30,
            "units": [{"position": (2, 3)}, {"position": (5, 5)}],
            "cities": [{"position": (4, 4)}],
            "action_mask": np.ones(100, dtype=bool),
        }

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.board_size == 8
        assert generator.action_space_size == 100
        assert generator._targets_generated == 0

    def test_generate_targets(self, generator, sample_state):
        """Test generating all targets."""
        targets = generator.generate_targets(sample_state)

        assert isinstance(targets, AuxiliaryTargets)
        assert targets.attention_map is not None
        assert targets.opponent_action is not None
        assert targets.game_phase is not None
        assert targets.future_value is not None

    def test_attention_map_normalized(self, generator, sample_state):
        """Test attention map sums to 1."""
        targets = generator.generate_targets(sample_state)
        assert abs(targets.attention_map.sum() - 1.0) < 1e-6

    def test_attention_map_shape(self, generator, sample_state):
        """Test attention map has correct shape."""
        targets = generator.generate_targets(sample_state)
        assert targets.attention_map.shape == (8, 8)

    def test_opponent_action_normalized(self, generator, sample_state):
        """Test opponent action sums to 1."""
        targets = generator.generate_targets(sample_state)
        assert abs(targets.opponent_action.sum() - 1.0) < 1e-6

    def test_opponent_action_shape(self, generator, sample_state):
        """Test opponent action has correct shape."""
        targets = generator.generate_targets(sample_state)
        assert targets.opponent_action.shape == (100,)

    def test_game_phase_one_hot(self, generator, sample_state):
        """Test game phase is one-hot encoded."""
        targets = generator.generate_targets(sample_state)
        assert targets.game_phase.shape == (3,)
        assert targets.game_phase.sum() == 1.0

    def test_phase_early(self, generator):
        """Test early game phase classification."""
        state = {"turn": 5}
        targets = generator.generate_targets(state)
        assert targets.game_phase[0] == 1.0  # Early

    def test_phase_mid(self, generator):
        """Test mid game phase classification."""
        state = {"turn": 40}
        targets = generator.generate_targets(state)
        assert targets.game_phase[1] == 1.0  # Mid

    def test_phase_late(self, generator):
        """Test late game phase classification."""
        state = {"turn": 80}
        targets = generator.generate_targets(state)
        assert targets.game_phase[2] == 1.0  # Late

    def test_future_value_range(self, generator, sample_state):
        """Test future value is in valid range."""
        future_states = [{"value": 0.5}]
        targets = generator.generate_targets(
            sample_state,
            future_states=future_states,
        )
        assert -1.0 <= targets.future_value <= 1.0

    def test_opponent_with_history(self, generator, sample_state):
        """Test opponent modeling uses history."""
        history = [
            {"opponent_action": 5},
            {"opponent_action": 5},
            {"opponent_action": 10},
        ]
        targets = generator.generate_targets(
            sample_state,
            game_history=history,
        )
        # Action 5 should have higher probability
        assert targets.opponent_action[5] > targets.opponent_action[50]

    def test_selective_generation(self):
        """Test generating only enabled targets."""
        config = AuxiliaryTargetsConfig(
            enable_attention_targets=True,
            enable_opponent_modeling=False,
            enable_phase_prediction=False,
            enable_future_value=False,
        )
        generator = AuxiliaryTargetGenerator(config, seed=42)

        targets = generator.generate_targets({"turn": 10})
        assert targets.attention_map is not None
        assert targets.opponent_action is None
        assert targets.game_phase is None
        assert targets.future_value is None

    def test_get_stats(self, generator, sample_state):
        """Test getting statistics."""
        for _ in range(10):
            generator.generate_targets(sample_state)

        stats = generator.get_stats()
        assert stats["targets_generated"] == 10
        assert "config" in stats


class TestAuxiliaryLossCalculator:
    """Tests for AuxiliaryLossCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator with default config."""
        return AuxiliaryLossCalculator(AuxiliaryTargetsConfig())

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        return {
            "attention": np.ones((8, 8)) / 64,
            "opponent": np.ones(100) / 100,
            "phase": np.array([0.7, 0.2, 0.1]),
            "future_value": 0.3,
        }

    @pytest.fixture
    def sample_targets(self):
        """Create sample targets."""
        attention = np.ones((8, 8)) / 64
        phase = np.zeros(3)
        phase[0] = 1.0
        return AuxiliaryTargets(
            attention_map=attention,
            opponent_action=np.ones(100) / 100,
            game_phase=phase,
            future_value=0.5,
        )

    def test_compute_loss(self, calculator, sample_predictions, sample_targets):
        """Test computing auxiliary losses."""
        losses = calculator.compute_loss(sample_predictions, sample_targets)

        assert "attention_loss" in losses
        assert "opponent_loss" in losses
        assert "phase_loss" in losses
        assert "future_value_loss" in losses
        assert "total_auxiliary_loss" in losses

    def test_total_loss_is_weighted_sum(self, calculator, sample_predictions, sample_targets):
        """Test that total loss is weighted sum of components."""
        losses = calculator.compute_loss(sample_predictions, sample_targets)

        expected_total = (
            losses["attention_loss"] * calculator.config.attention_weight
            + losses["opponent_loss"] * calculator.config.opponent_weight
            + losses["phase_loss"] * calculator.config.phase_weight
            + losses["future_value_loss"] * calculator.config.future_value_weight
        )
        assert abs(losses["total_auxiliary_loss"] - expected_total) < 1e-6

    def test_attention_loss_kl(self, calculator):
        """Test attention loss is KL divergence."""
        pred = np.array([[0.5, 0.5], [0.0, 0.0]])
        pred = pred / pred.sum()

        target = np.array([[1.0, 0.0], [0.0, 0.0]])

        targets = AuxiliaryTargets(attention_map=target)
        predictions = {"attention": pred}

        losses = calculator.compute_loss(predictions, targets)

        # KL divergence should be positive
        assert losses["attention_loss"] > 0

    def test_perfect_prediction(self, calculator, sample_targets):
        """Test losses are near zero for perfect predictions."""
        predictions = {
            "attention": sample_targets.attention_map.copy(),
            "opponent": sample_targets.opponent_action.copy(),
            "phase": sample_targets.game_phase.copy(),
            "future_value": sample_targets.future_value,
        }

        losses = calculator.compute_loss(predictions, sample_targets)

        assert losses["attention_loss"] < 1e-6
        assert losses["future_value_loss"] < 1e-6

    def test_get_average_losses(self, calculator, sample_predictions, sample_targets):
        """Test getting average losses."""
        for _ in range(50):
            calculator.compute_loss(sample_predictions, sample_targets)

        avg = calculator.get_average_losses(window=50)
        assert "avg_total_auxiliary_loss" in avg
        assert avg["avg_total_auxiliary_loss"] >= 0

    def test_reset_history(self, calculator, sample_predictions, sample_targets):
        """Test resetting loss history."""
        for _ in range(20):
            calculator.compute_loss(sample_predictions, sample_targets)

        calculator.reset_history()
        avg = calculator.get_average_losses()
        assert len(avg) == 0

    def test_selective_loss_calculation(self):
        """Test calculating only enabled losses."""
        config = AuxiliaryTargetsConfig(
            enable_attention_targets=True,
            enable_opponent_modeling=False,
            enable_phase_prediction=False,
            enable_future_value=False,
        )
        calculator = AuxiliaryLossCalculator(config)

        targets = AuxiliaryTargets(
            attention_map=np.ones((8, 8)) / 64,
        )
        predictions = {"attention": np.ones((8, 8)) / 64}

        losses = calculator.compute_loss(predictions, targets)

        assert "attention_loss" in losses
        assert "opponent_loss" not in losses
        assert "phase_loss" not in losses
        assert "future_value_loss" not in losses


class TestCreateAuxiliarySystem:
    """Tests for create_auxiliary_system factory."""

    def test_create_with_defaults(self):
        """Test creating system with defaults."""
        generator, calculator = create_auxiliary_system()
        assert isinstance(generator, AuxiliaryTargetGenerator)
        assert isinstance(calculator, AuxiliaryLossCalculator)

    def test_create_with_config(self):
        """Test creating system with custom config."""
        config = AuxiliaryTargetsConfig(attention_weight=0.5)
        _generator, calculator = create_auxiliary_system(config)
        assert calculator.config.attention_weight == 0.5

    def test_create_with_dict(self):
        """Test creating system with dict config."""
        config_dict = {"enable_opponent_modeling": False}
        generator, _calculator = create_auxiliary_system(config_dict)
        assert generator.config.enable_opponent_modeling is False

    def test_create_with_params(self):
        """Test creating system with custom parameters."""
        generator, _calculator = create_auxiliary_system(
            board_size=12,
            action_space_size=500,
            seed=123,
        )
        assert generator.board_size == 12
        assert generator.action_space_size == 500
