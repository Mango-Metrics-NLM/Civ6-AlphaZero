"""Tests for alpha_zero.utils.constants module.

Validates all constant definitions, dataclass functionality, and
helper methods for encoding/configuration constants.
"""

from __future__ import annotations

import pytest

from alpha_zero.utils.constants import (
    ENCODER,
    GAME,
    MCTS,
    NETWORK,
    TRAINING,
    EncoderConstants,
    GameConstants,
    MCTSConstants,
    NetworkConstants,
    TrainingConstants,
)


class TestEncoderConstants:
    """Tests for EncoderConstants dataclass."""

    def test_channel_counts(self) -> None:
        """Test channel count constants."""
        ec = EncoderConstants()
        assert ec.MICROCIV_CHANNELS == 13
        assert ec.FULL_ENCODER_CHANNELS == 84

    def test_microciv_channel_ranges(self) -> None:
        """Test MicroCiv channel range definitions."""
        ec = EncoderConstants()
        # Terrain channels
        assert ec.TERRAIN_CHANNELS == (0, 3)
        # Unit channels
        assert ec.OWN_UNIT_CHANNELS == (3, 6)
        assert ec.ENEMY_UNIT_CHANNELS == (6, 9)
        # City channels
        assert ec.OWN_CITY_CHANNEL == 9
        assert ec.ENEMY_CITY_CHANNEL == 10
        # Special channels
        assert ec.VISIBILITY_CHANNEL == 11
        assert ec.BIAS_CHANNEL == 12

    def test_full_encoder_channel_ranges(self) -> None:
        """Test full encoder channel ranges are contiguous."""
        ec = EncoderConstants()
        # Check ranges are contiguous
        ranges = [
            ec.TERRAIN_TYPE_RANGE,
            ec.TERRAIN_FEATURE_RANGE,
            ec.RESOURCE_RANGE,
            ec.OWN_UNITS_RANGE,
            ec.ENEMY_UNITS_RANGE,
            ec.OWN_CITIES_RANGE,
            ec.ENEMY_CITIES_RANGE,
            ec.BORDER_RANGE,
            ec.FOG_RANGE,
            ec.YIELDS_RANGE,
            ec.IMPROVEMENTS_RANGE,
            ec.MOVEMENT_RANGE,
        ]
        # Verify starting at 0
        assert ranges[0][0] == 0
        # Verify contiguous (each start == previous end)
        for i in range(1, len(ranges)):
            assert ranges[i][0] == ranges[i - 1][1], f"Gap between {ranges[i - 1]} and {ranges[i]}"
        # Verify ending at 84
        assert ranges[-1][1] == ec.FULL_ENCODER_CHANNELS

    def test_normalization_factors_positive(self) -> None:
        """Test normalization factors are positive."""
        ec = EncoderConstants()
        assert ec.YIELD_NORMALIZATION > 0
        assert ec.POPULATION_NORMALIZATION > 0
        assert ec.HEALTH_NORMALIZATION > 0
        assert ec.STRENGTH_NORMALIZATION > 0
        assert ec.TURN_NORMALIZATION > 0
        assert ec.PRODUCTION_NORMALIZATION > 0
        assert ec.GOLD_NORMALIZATION > 0
        assert ec.SCIENCE_NORMALIZATION > 0
        assert ec.CULTURE_NORMALIZATION > 0
        assert ec.FAITH_NORMALIZATION > 0

    def test_vector_dimensions(self) -> None:
        """Test vector dimension constants."""
        ec = EncoderConstants()
        assert ec.MICROCIV_VECTOR_DIM == 64
        assert ec.FULL_VECTOR_DIM == 512

    def test_get_channel_range_known_types(self) -> None:
        """Test get_channel_range for known types."""
        ec = EncoderConstants()
        assert ec.get_channel_range("terrain_type") == (0, 7)
        assert ec.get_channel_range("own_units") == (27, 39)
        assert ec.get_channel_range("fog") == (63, 66)
        assert ec.get_channel_range("movement") == (79, 84)

    def test_get_channel_range_unknown_type(self) -> None:
        """Test get_channel_range returns (0, 0) for unknown types."""
        ec = EncoderConstants()
        assert ec.get_channel_range("unknown") == (0, 0)
        assert ec.get_channel_range("invalid_type") == (0, 0)

    def test_dataclass_frozen(self) -> None:
        """Test that EncoderConstants is frozen (immutable)."""
        ec = EncoderConstants()
        with pytest.raises(AttributeError):
            ec.MICROCIV_CHANNELS = 20  # type: ignore[misc]


class TestNetworkConstants:
    """Tests for NetworkConstants dataclass."""

    def test_default_architecture_params(self) -> None:
        """Test default architecture parameters."""
        nc = NetworkConstants()
        assert nc.DEFAULT_NUM_FILTERS == 64
        assert nc.DEFAULT_NUM_RES_BLOCKS == 6
        assert nc.DEFAULT_FC_DIM == 256

    def test_attention_params(self) -> None:
        """Test attention mechanism parameters."""
        nc = NetworkConstants()
        assert nc.DEFAULT_NUM_HEADS == 8
        assert nc.DEFAULT_ATTENTION_DIM == 64
        assert nc.DEFAULT_MLP_RATIO == 4.0
        assert nc.DEFAULT_DROPOUT == 0.1

    def test_action_masking_values(self) -> None:
        """Test action masking constants."""
        nc = NetworkConstants()
        # Invalid action logit should be very negative
        assert nc.INVALID_ACTION_LOGIT < -1e8
        # Temperature range
        assert nc.SOFTMAX_TEMPERATURE_MIN > 0
        assert nc.SOFTMAX_TEMPERATURE_MAX > nc.SOFTMAX_TEMPERATURE_MIN

    def test_initialization_values(self) -> None:
        """Test weight initialization constants."""
        nc = NetworkConstants()
        assert nc.WEIGHT_INIT_STD > 0
        assert nc.WEIGHT_INIT_STD < 0.1  # Should be small
        assert nc.BIAS_INIT_VALUE == 0.0

    def test_gradient_clipping(self) -> None:
        """Test gradient clipping constant."""
        nc = NetworkConstants()
        assert nc.MAX_GRADIENT_NORM > 0


class TestMCTSConstants:
    """Tests for MCTSConstants dataclass."""

    def test_puct_params(self) -> None:
        """Test PUCT formula parameters."""
        mc = MCTSConstants()
        assert mc.DEFAULT_C_PUCT > 0
        assert mc.DEFAULT_C_PUCT_BASE > 0
        assert mc.DEFAULT_C_PUCT_INIT > 0

    def test_exploration_noise(self) -> None:
        """Test Dirichlet exploration noise parameters."""
        mc = MCTSConstants()
        assert mc.DEFAULT_DIRICHLET_ALPHA > 0
        assert mc.DEFAULT_DIRICHLET_EPSILON >= 0
        assert mc.DEFAULT_DIRICHLET_EPSILON <= 1

    def test_temperature_bounds(self) -> None:
        """Test temperature parameter bounds."""
        mc = MCTSConstants()
        assert mc.DEFAULT_TEMPERATURE > 0
        assert mc.DEFAULT_TEMPERATURE_THRESHOLD >= 0
        assert mc.DEFAULT_TEMPERATURE_FINAL >= 0

    def test_simulation_constraints(self) -> None:
        """Test simulation count constraints."""
        mc = MCTSConstants()
        assert mc.DEFAULT_NUM_SIMULATIONS > 0
        assert mc.MIN_SIMULATIONS > 0
        assert mc.DEFAULT_NUM_SIMULATIONS >= mc.MIN_SIMULATIONS


class TestTrainingConstants:
    """Tests for TrainingConstants dataclass."""

    def test_learning_rate_params(self) -> None:
        """Test learning rate parameters."""
        tc = TrainingConstants()
        assert tc.DEFAULT_LEARNING_RATE > 0
        assert tc.DEFAULT_LEARNING_RATE < 1.0
        assert tc.MIN_LEARNING_RATE > 0
        assert tc.LR_DECAY_FACTOR > 0
        assert tc.LR_DECAY_FACTOR < 1.0

    def test_batch_params(self) -> None:
        """Test batch size parameters."""
        tc = TrainingConstants()
        assert tc.DEFAULT_BATCH_SIZE > 0
        assert tc.MIN_BATCH_SIZE > 0
        assert tc.DEFAULT_BATCH_SIZE >= tc.MIN_BATCH_SIZE

    def test_buffer_params(self) -> None:
        """Test replay buffer parameters."""
        tc = TrainingConstants()
        assert tc.DEFAULT_BUFFER_SIZE > 0
        assert tc.DEFAULT_MIN_REPLAY_SIZE > 0
        assert tc.DEFAULT_BUFFER_SIZE >= tc.DEFAULT_MIN_REPLAY_SIZE

    def test_regularization_params(self) -> None:
        """Test regularization parameters."""
        tc = TrainingConstants()
        assert tc.WEIGHT_DECAY >= 0
        # TrainingConstants doesn't have gradient clip, check via NetworkConstants
        from alpha_zero.utils.constants import NETWORK

        assert NETWORK.MAX_GRADIENT_NORM > 0


class TestGameConstants:
    """Tests for GameConstants dataclass."""

    def test_board_constraints(self) -> None:
        """Test board size constraints."""
        gc = GameConstants()
        assert gc.MIN_BOARD_SIZE > 0
        assert gc.MAX_BOARD_SIZE > gc.MIN_BOARD_SIZE
        assert gc.DEFAULT_BOARD_SIZE >= gc.MIN_BOARD_SIZE
        assert gc.DEFAULT_BOARD_SIZE <= gc.MAX_BOARD_SIZE

    def test_unit_constraints(self) -> None:
        """Test unit count constraints."""
        gc = GameConstants()
        assert gc.MAX_UNITS_PER_PLAYER > 0
        assert gc.MAX_CITIES_PER_PLAYER > 0

    def test_turn_constraints(self) -> None:
        """Test turn count constraints."""
        gc = GameConstants()
        assert gc.DEFAULT_MAX_TURNS > 0


class TestGlobalInstances:
    """Tests for global constant instances."""

    def test_global_instances_exist(self) -> None:
        """Test that global instances are created."""
        assert ENCODER is not None
        assert NETWORK is not None
        assert MCTS is not None
        assert TRAINING is not None
        assert GAME is not None

    def test_global_instances_are_correct_type(self) -> None:
        """Test that global instances have correct types."""
        assert isinstance(ENCODER, EncoderConstants)
        assert isinstance(NETWORK, NetworkConstants)
        assert isinstance(MCTS, MCTSConstants)
        assert isinstance(TRAINING, TrainingConstants)
        assert isinstance(GAME, GameConstants)


class TestConstantsConsistency:
    """Tests for consistency across constant definitions."""

    def test_microciv_channels_consistent(self) -> None:
        """Test MicroCiv channel count matches range definitions."""
        ec = EncoderConstants()
        # BIAS_CHANNEL is the last channel (12), so total is 13
        assert ec.BIAS_CHANNEL + 1 == ec.MICROCIV_CHANNELS

    def test_channel_ranges_no_overlap(self) -> None:
        """Test full encoder channel ranges don't overlap."""
        ec = EncoderConstants()
        ranges = [
            ec.TERRAIN_TYPE_RANGE,
            ec.TERRAIN_FEATURE_RANGE,
            ec.RESOURCE_RANGE,
            ec.OWN_UNITS_RANGE,
            ec.ENEMY_UNITS_RANGE,
            ec.OWN_CITIES_RANGE,
            ec.ENEMY_CITIES_RANGE,
            ec.BORDER_RANGE,
            ec.FOG_RANGE,
            ec.YIELDS_RANGE,
            ec.IMPROVEMENTS_RANGE,
            ec.MOVEMENT_RANGE,
        ]
        # Check no overlapping ranges
        for i, (start_i, end_i) in enumerate(ranges):
            for j, (start_j, end_j) in enumerate(ranges):
                if i != j:
                    # Ranges should not overlap
                    assert end_i <= start_j or end_j <= start_i, (
                        f"Ranges {ranges[i]} and {ranges[j]} overlap"
                    )

    def test_normalization_factors_reasonable(self) -> None:
        """Test normalization factors have reasonable values."""
        ec = EncoderConstants()
        # All normalization factors should be in reasonable range
        factors = [
            ec.YIELD_NORMALIZATION,
            ec.POPULATION_NORMALIZATION,
            ec.HEALTH_NORMALIZATION,
            ec.STRENGTH_NORMALIZATION,
            ec.TURN_NORMALIZATION,
            ec.PRODUCTION_NORMALIZATION,
            ec.GOLD_NORMALIZATION,
            ec.SCIENCE_NORMALIZATION,
            ec.CULTURE_NORMALIZATION,
            ec.FAITH_NORMALIZATION,
        ]
        for factor in factors:
            assert factor >= 1.0, "Normalization factors should be >= 1"
            assert factor <= 10000.0, "Normalization factors should be reasonable"


class TestConstantsValidation:
    """Tests for validating constant relationships."""

    def test_training_lr_decay_steps_increasing(self) -> None:
        """Test learning rate decay steps are in increasing order."""
        tc = TrainingConstants()
        if hasattr(tc, "LR_DECAY_STEPS") and tc.LR_DECAY_STEPS:
            steps = tc.LR_DECAY_STEPS
            for i in range(1, len(steps)):
                assert steps[i] > steps[i - 1], "LR decay steps must be increasing"

    def test_mcts_simulations_bounds(self) -> None:
        """Test MCTS simulation bounds are consistent."""
        mc = MCTSConstants()
        assert mc.DEFAULT_NUM_SIMULATIONS >= mc.MIN_SIMULATIONS
        if hasattr(mc, "MAX_SIMULATIONS"):
            assert mc.DEFAULT_NUM_SIMULATIONS <= mc.MAX_SIMULATIONS

    def test_temperature_bounds_consistent(self) -> None:
        """Test temperature bounds are consistent."""
        mc = MCTSConstants()
        nc = NetworkConstants()
        # Temperature should be within softmax temperature range
        assert mc.DEFAULT_TEMPERATURE >= nc.SOFTMAX_TEMPERATURE_MIN
        assert mc.DEFAULT_TEMPERATURE <= nc.SOFTMAX_TEMPERATURE_MAX
