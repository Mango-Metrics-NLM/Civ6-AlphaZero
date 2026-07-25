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
    PATHS,
    TRAINING,
    EncoderConstants,
    GameConstants,
    MCTSConstants,
    NetworkConstants,
    PathConstants,
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

    def test_per_variant_simulation_budgets(self) -> None:
        """Per-variant budgets are positive and within global bounds."""
        mc = MCTSConstants()
        for budget in (
            mc.CLASSIC_NUM_SIMULATIONS,
            mc.GUMBEL_NUM_SIMULATIONS,
            mc.MUZERO_NUM_SIMULATIONS,
            mc.ORCHESTRATOR_NUM_SIMULATIONS,
        ):
            assert mc.MIN_SIMULATIONS <= budget <= mc.MAX_SIMULATIONS

    def test_gumbel_params(self) -> None:
        """Gumbel-specific constants are internally consistent."""
        mc = MCTSConstants()
        assert mc.GUMBEL_NUM_SAMPLED_ACTIONS > 0
        assert mc.GUMBEL_MAX_CONSIDERED_ACTIONS >= mc.GUMBEL_NUM_SAMPLED_ACTIONS
        assert mc.GUMBEL_MIN_ACTIONS_AFTER_HALVING >= 1
        assert mc.GUMBEL_C_SCALE > 0

    def test_muzero_params(self) -> None:
        """MuZero-specific constants are sane."""
        mc = MCTSConstants()
        assert 0 < mc.MUZERO_DISCOUNT <= 1.0
        assert mc.MUZERO_REWARD_SUPPORT_SIZE > 0


class TestMCTSConfigValidation:
    """Tests for runtime validation on the MCTS config dataclasses.

    These validations are new boundary checks added to prevent silently
    accepting invalid hyperparameter combinations.
    """

    def test_core_mctsconfig_rejects_negative_simulations(self) -> None:
        import pytest

        from alpha_zero.core.civ6_mcts import MCTSConfig

        with pytest.raises(ValueError, match="num_simulations"):
            MCTSConfig(num_simulations=-1)

    def test_core_mctsconfig_rejects_excessive_simulations(self) -> None:
        import pytest

        from alpha_zero.core.civ6_mcts import MCTSConfig
        from alpha_zero.utils.constants import MCTS

        with pytest.raises(ValueError, match="num_simulations"):
            MCTSConfig(num_simulations=MCTS.MAX_SIMULATIONS + 1)

    def test_core_mctsconfig_rejects_invalid_dirichlet_epsilon(self) -> None:
        import pytest

        from alpha_zero.core.civ6_mcts import MCTSConfig

        with pytest.raises(ValueError, match="dirichlet_epsilon"):
            MCTSConfig(dirichlet_epsilon=1.5)
        with pytest.raises(ValueError, match="dirichlet_epsilon"):
            MCTSConfig(dirichlet_epsilon=-0.1)

    def test_core_mctsconfig_rejects_nonpositive_c_puct(self) -> None:
        import pytest

        from alpha_zero.core.civ6_mcts import MCTSConfig

        with pytest.raises(ValueError, match="c_puct"):
            MCTSConfig(c_puct=0.0)

    def test_core_mctsconfig_rejects_negative_temperature(self) -> None:
        import pytest

        from alpha_zero.core.civ6_mcts import MCTSConfig

        with pytest.raises(ValueError, match="temperature"):
            MCTSConfig(temperature=-0.5)

    def test_gumbel_config_rejects_inconsistent_action_counts(self) -> None:
        import pytest

        from alpha_zero.core.civ6_mcts import GumbelMCTSConfig

        with pytest.raises(ValueError, match="max_considered_actions"):
            GumbelMCTSConfig(num_sampled_actions=32, max_considered_actions=8)

    def test_muzero_config_rejects_invalid_discount(self) -> None:
        import pytest

        from alpha_zero.core.civ6_mcts import MuZeroMCTSConfig

        with pytest.raises(ValueError, match="discount"):
            MuZeroMCTSConfig(discount=0.0)
        with pytest.raises(ValueError, match="discount"):
            MuZeroMCTSConfig(discount=1.5)

    def test_muzero_config_rejects_nonpositive_sampled_actions_under_gumbel(self) -> None:
        """MuZeroMCTSConfig with use_gumbel=True must reject non-positive
        num_sampled_actions. Without this guard, _gumbel_search() would
        pass a non-positive value into gumbel_sample_actions and crash at
        search time (PR #77 review feedback)."""
        import pytest

        from alpha_zero.core.civ6_mcts import MuZeroMCTSConfig

        with pytest.raises(ValueError, match="num_sampled_actions"):
            MuZeroMCTSConfig(use_gumbel=True, num_sampled_actions=0)
        with pytest.raises(ValueError, match="num_sampled_actions"):
            MuZeroMCTSConfig(use_gumbel=True, num_sampled_actions=-1)

    def test_muzero_config_skips_sampled_actions_check_when_gumbel_disabled(
        self,
    ) -> None:
        """The num_sampled_actions field is only used by Gumbel search, so
        when use_gumbel=False a non-positive value is harmless."""
        from alpha_zero.core.civ6_mcts import MuZeroMCTSConfig

        # Should not raise.
        MuZeroMCTSConfig(use_gumbel=False, num_sampled_actions=0)

    def test_muzero_from_network_preserves_temperature_threshold(self) -> None:
        """MuZeroMCTS.from_network() must propagate every base_config field
        when it syncs reward settings from the network. Previously
        temperature_threshold and reward_support_size were silently reset
        to class defaults, losing user overrides (PR #77 review feedback)."""
        from unittest.mock import MagicMock

        from alpha_zero.core.civ6_mcts import MuZeroMCTS, MuZeroMCTSConfig

        # Network stub with its own config (triggers the synced_config path)
        fake_network = MagicMock(spec=["config", "to", "eval"])
        fake_network.config = MagicMock()
        fake_network.config.use_categorical_reward = True
        fake_network.config.reward_support_size = 401  # non-default
        fake_network.to.return_value = fake_network

        user_config = MuZeroMCTSConfig(
            temperature_threshold=99,  # non-default
            temperature=0.5,
            num_simulations=33,
        )
        mcts = MuZeroMCTS.from_network(fake_network, config=user_config)

        # Every field of base_config must round-trip into the synced config.
        assert mcts.config.temperature_threshold == 99, "from_network dropped temperature_threshold"
        assert mcts.config.temperature == 0.5
        assert mcts.config.num_simulations == 33
        # Reward settings come from the network when present.
        assert mcts.config.reward_support_size == 401

    def test_orchestrator_mctsconfig_validation(self) -> None:
        import pytest

        from alpha_zero.orchestrator.config import MCTSConfig as OrchMCTSConfig

        with pytest.raises(ValueError, match="num_simulations"):
            OrchMCTSConfig(num_simulations=0)
        with pytest.raises(ValueError, match="dirichlet_epsilon"):
            OrchMCTSConfig(dirichlet_epsilon=2.0)

    def test_temperature_threshold_validated_across_variants(self) -> None:
        """Negative temperature_threshold must be rejected by every variant.

        Field added to MuZeroMCTSConfig in response to PR #77 review;
        validation added to GumbelMCTSConfig at the same time so all four
        config classes (classic / gumbel / muzero / orchestrator) reject
        negative thresholds consistently.
        """
        import pytest

        from alpha_zero.core.civ6_mcts import (
            GumbelMCTSConfig,
            MCTSConfig,
            MuZeroMCTSConfig,
        )
        from alpha_zero.orchestrator.config import MCTSConfig as OrchMCTSConfig

        for cls in (MCTSConfig, GumbelMCTSConfig, MuZeroMCTSConfig, OrchMCTSConfig):
            with pytest.raises(ValueError, match="temperature_threshold"):
                cls(temperature_threshold=-1)

    def test_muzero_config_has_temperature_threshold(self) -> None:
        """MuZeroMCTSConfig exposes temperature_threshold like the other variants.

        Previously absent — would have prevented turn-based temperature
        schedules during MuZero self-play.
        """
        from alpha_zero.core.civ6_mcts import MuZeroMCTSConfig
        from alpha_zero.utils.constants import MCTS

        cfg = MuZeroMCTSConfig()
        assert cfg.temperature_threshold == MCTS.DEFAULT_TEMPERATURE_THRESHOLD

    def test_orchestrator_mctsconfig_from_dict_no_drift(self) -> None:
        """from_dict({}) must agree with the class default constructor.

        This guards against the entire drift bug class where a class default
        and its from_dict literal fallback get out of sync over time, silently
        producing different behaviour depending on whether a field is present
        in the YAML being loaded.
        """
        from alpha_zero.orchestrator.config import MCTSConfig as OrchMCTSConfig

        default = OrchMCTSConfig()
        from_empty = OrchMCTSConfig.from_dict({})

        assert from_empty.num_simulations == default.num_simulations
        assert from_empty.c_puct == default.c_puct
        assert from_empty.dirichlet_alpha == default.dirichlet_alpha
        assert from_empty.dirichlet_epsilon == default.dirichlet_epsilon
        assert from_empty.temperature == default.temperature
        assert from_empty.temperature_threshold == default.temperature_threshold

    def test_per_variant_defaults_sourced_from_constants(self) -> None:
        """All four MCTS config classes draw defaults from utils.constants.MCTS.

        Pinning this in a test guards against future drift back into
        hard-coded literals.
        """
        from alpha_zero.core.civ6_mcts import (
            GumbelMCTSConfig,
            MCTSConfig,
            MuZeroMCTSConfig,
        )
        from alpha_zero.orchestrator.config import MCTSConfig as OrchMCTSConfig
        from alpha_zero.utils.constants import MCTS

        assert MCTSConfig().num_simulations == MCTS.CLASSIC_NUM_SIMULATIONS
        assert GumbelMCTSConfig().num_simulations == MCTS.GUMBEL_NUM_SIMULATIONS
        assert MuZeroMCTSConfig().num_simulations == MCTS.MUZERO_NUM_SIMULATIONS
        assert OrchMCTSConfig().num_simulations == MCTS.ORCHESTRATOR_NUM_SIMULATIONS


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


class TestPathConstants:
    """Tests for PathConstants and the configs that consume them.

    Pinning these in tests guards against the entire class of bugs where
    a hardcoded path appears as a class default AND as a from_dict
    fallback, then drifts when only one of the two is updated.
    """

    def test_path_constants_are_non_empty_strings(self) -> None:
        pc = PathConstants()
        for name in (
            "RAG_PERSIST_DIR",
            "RAG_KNOWLEDGE_DIR",
            "RAG_COLLECTION_NAME",
            "REFLECTION_PERSIST_DIR",
            "REFLECTION_COLLECTION_NAME",
            "ORCHESTRATOR_CHECKPOINT_DIR",
            "ORCHESTRATOR_LOG_DIR",
            "DEFAULT_LOCAL_STORAGE_ROOT",
        ):
            value = getattr(pc, name)
            assert isinstance(value, str)
            assert value, f"{name} must be a non-empty string"

    def test_rag_config_paths_sourced_from_constants(self) -> None:
        from alpha_zero.orchestrator.config import RAGConfig

        c = RAGConfig()
        assert c.persist_directory == PATHS.RAG_PERSIST_DIR
        assert c.knowledge_dir == PATHS.RAG_KNOWLEDGE_DIR
        assert c.collection_name == PATHS.RAG_COLLECTION_NAME

    def test_reflection_config_paths_sourced_from_constants(self) -> None:
        from alpha_zero.orchestrator.config import ReflectionIntegrationConfig

        c = ReflectionIntegrationConfig()
        assert c.persist_directory == PATHS.REFLECTION_PERSIST_DIR
        assert c.collection_name == PATHS.REFLECTION_COLLECTION_NAME

    def test_orchestrator_config_paths_sourced_from_constants(self) -> None:
        from alpha_zero.orchestrator.config import OrchestratorConfig

        c = OrchestratorConfig()
        assert c.checkpoint_dir == PATHS.ORCHESTRATOR_CHECKPOINT_DIR
        assert c.log_dir == PATHS.ORCHESTRATOR_LOG_DIR

    def test_storage_config_base_path_sourced_from_constants(self) -> None:
        from alpha_zero.storage.config import StorageConfig

        assert StorageConfig().base_path == PATHS.DEFAULT_LOCAL_STORAGE_ROOT
        # for_local() factory honors override but defaults to the constant.
        assert StorageConfig.for_local().base_path == PATHS.DEFAULT_LOCAL_STORAGE_ROOT

    def test_path_consuming_configs_no_drift_on_from_dict(self) -> None:
        """from_dict({}) must agree with the class default constructor for
        every config class that holds a path. Guards the drift bug class —
        a class default and its from_dict literal fallback getting out of
        sync — at the path layer (RAGConfig, ReflectionIntegrationConfig,
        StorageConfig) and on TrainingConfig.games_per_iteration."""
        from alpha_zero.orchestrator.config import (
            RAGConfig,
            ReflectionIntegrationConfig,
            TrainingConfig,
        )
        from alpha_zero.storage.config import StorageConfig

        for cls, attrs in (
            (RAGConfig, ("persist_directory", "knowledge_dir", "collection_name")),
            (
                ReflectionIntegrationConfig,
                ("persist_directory", "collection_name"),
            ),
            (StorageConfig, ("base_path",)),
            (TrainingConfig, ("games_per_iteration",)),
        ):
            default = cls()
            from_empty = cls.from_dict({})
            for attr in attrs:
                assert getattr(default, attr) == getattr(from_empty, attr), (
                    f"{cls.__name__}.{attr} drift: "
                    f"class={getattr(default, attr)!r} from_dict={getattr(from_empty, attr)!r}"
                )

    def test_training_config_games_per_iteration_drift_fixed(self) -> None:
        """Regression test for the silent-drift bug class fixed in this branch.

        The class default and the from_dict fallback must agree regardless
        of what value either currently holds — the bug was structural (the
        two paths used independent literals that could drift apart over
        time), not about any specific numeric pair. Embedding specific
        values in the docstring would itself drift if either default is
        retuned later, so we describe the bug class instead.
        """
        from alpha_zero.orchestrator.config import TrainingConfig

        class_default = TrainingConfig().games_per_iteration
        from_empty = TrainingConfig.from_dict({}).games_per_iteration
        assert class_default == from_empty, (
            f"games_per_iteration drift: class={class_default!r} from_dict={from_empty!r}"
        )


class TestGlobalInstances:
    """Tests for global constant instances."""

    def test_global_instances_exist(self) -> None:
        """Test that global instances are created."""
        assert ENCODER is not None
        assert NETWORK is not None
        assert MCTS is not None
        assert TRAINING is not None
        assert GAME is not None
        assert PATHS is not None

    def test_global_instances_are_correct_type(self) -> None:
        """Test that global instances have correct types."""
        assert isinstance(ENCODER, EncoderConstants)
        assert isinstance(NETWORK, NetworkConstants)
        assert isinstance(MCTS, MCTSConstants)
        assert isinstance(TRAINING, TrainingConstants)
        assert isinstance(GAME, GameConstants)
        assert isinstance(PATHS, PathConstants)


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
