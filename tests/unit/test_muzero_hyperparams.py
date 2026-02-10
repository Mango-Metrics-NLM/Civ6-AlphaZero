"""Unit tests for MuZero-specific hyperparameter space configuration.

Tests cover:
- get_muzero_custom_params: parameter definitions
- create_muzero_hyperparameter_space: complete space creation
- merge_with_base_space: space merging and overrides
- Sampling and mutation behavior
"""

import numpy as np

from alpha_zero.pbt.config import HyperparameterRange, HyperparameterSpace
from alpha_zero.pbt.muzero_hyperparams import (
    MUZERO_PARAM_NAMES,
    create_muzero_hyperparameter_space,
    get_muzero_custom_params,
    merge_with_base_space,
)

# =============================================================================
# get_muzero_custom_params Tests
# =============================================================================


class TestGetMuZeroCustomParams:
    """Test MuZero custom parameter definitions."""

    def test_returns_dict_with_all_muzero_params(self):
        """Test returns dict with all MUZERO_PARAM_NAMES."""
        params = get_muzero_custom_params()

        assert isinstance(params, dict)
        assert len(params) == len(MUZERO_PARAM_NAMES)

        for param_name in MUZERO_PARAM_NAMES:
            assert param_name in params

    def test_all_values_are_hyperparameter_ranges(self):
        """Test all returned values are HyperparameterRange instances."""
        params = get_muzero_custom_params()

        for param_range in params.values():
            assert isinstance(param_range, HyperparameterRange)

    def test_unroll_steps_definition(self):
        """Test unroll_steps parameter is correctly defined."""
        params = get_muzero_custom_params()
        unroll = params["unroll_steps"]

        assert unroll.name == "unroll_steps"
        assert unroll.min_value == 3
        assert unroll.max_value == 10
        assert unroll.scale == "linear"
        assert unroll.discrete is True

    def test_td_steps_definition(self):
        """Test td_steps parameter is correctly defined."""
        params = get_muzero_custom_params()
        td_steps = params["td_steps"]

        assert td_steps.name == "td_steps"
        assert td_steps.min_value == 5
        assert td_steps.max_value == 20
        assert td_steps.scale == "linear"
        assert td_steps.discrete is True

    def test_discount_definition(self):
        """Test discount parameter is correctly defined."""
        params = get_muzero_custom_params()
        discount = params["discount"]

        assert discount.name == "discount"
        assert discount.min_value == 0.95
        assert discount.max_value == 0.999
        assert discount.scale == "linear"
        assert discount.discrete is False

    def test_consistency_loss_weight_definition(self):
        """Test consistency_loss_weight parameter is correctly defined."""
        params = get_muzero_custom_params()
        consistency = params["consistency_loss_weight"]

        assert consistency.name == "consistency_loss_weight"
        assert consistency.min_value == 0.5
        assert consistency.max_value == 5.0
        assert consistency.scale == "linear"

    def test_reward_loss_weight_definition(self):
        """Test reward_loss_weight parameter is correctly defined."""
        params = get_muzero_custom_params()
        reward = params["reward_loss_weight"]

        assert reward.name == "reward_loss_weight"
        assert reward.min_value == 0.5
        assert reward.max_value == 2.0
        assert reward.scale == "linear"


# =============================================================================
# create_muzero_hyperparameter_space Tests
# =============================================================================


class TestCreateMuZeroHyperparameterSpace:
    """Test MuZero hyperparameter space creation."""

    def test_returns_hyperparameter_space(self):
        """Test returns HyperparameterSpace instance."""
        space = create_muzero_hyperparameter_space()
        assert isinstance(space, HyperparameterSpace)

    def test_contains_all_muzero_custom_params(self):
        """Test space includes all MuZero custom parameters."""
        space = create_muzero_hyperparameter_space()
        all_params = space.get_all_params()

        for param_name in MUZERO_PARAM_NAMES:
            assert param_name in all_params

    def test_contains_base_parameters(self):
        """Test space includes standard base parameters."""
        space = create_muzero_hyperparameter_space()
        all_params = space.get_all_params()

        # Check for key base parameters
        assert "learning_rate" in all_params
        assert "weight_decay" in all_params
        assert "batch_size" in all_params
        assert "num_simulations" in all_params

    def test_batch_size_has_log_scale(self):
        """Test batch_size uses log scale for MuZero."""
        space = create_muzero_hyperparameter_space()
        assert space.batch_size.scale == "log"

    def test_batch_size_max_reduced(self):
        """Test batch_size max is 256 (reduced from base 512)."""
        space = create_muzero_hyperparameter_space()
        assert space.batch_size.max_value == 256

    def test_num_simulations_has_log_scale(self):
        """Test num_simulations uses log scale for MuZero."""
        space = create_muzero_hyperparameter_space()
        assert space.num_simulations.scale == "log"

    def test_c_puct_wider_range(self):
        """Test c_puct has wider range (0.5-5.0) for learned dynamics."""
        space = create_muzero_hyperparameter_space()
        assert space.c_puct.min_value == 0.5
        assert space.c_puct.max_value == 5.0

    def test_sample_all_returns_all_params(self):
        """Test sample_all returns dict with all parameters."""
        space = create_muzero_hyperparameter_space()
        rng = np.random.default_rng(42)

        samples = space.sample_all(rng)

        assert isinstance(samples, dict)
        # Should have base + custom params
        all_params = space.get_all_params()
        assert len(samples) == len(all_params)

    def test_sample_all_values_within_ranges(self):
        """Test sampled values are within defined ranges."""
        space = create_muzero_hyperparameter_space()
        rng = np.random.default_rng(42)

        samples = space.sample_all(rng)

        # Check MuZero-specific params
        assert 3 <= samples["unroll_steps"] <= 10
        assert 5 <= samples["td_steps"] <= 20
        assert 0.95 <= samples["discount"] <= 0.999
        assert 0.5 <= samples["consistency_loss_weight"] <= 5.0
        assert 0.5 <= samples["reward_loss_weight"] <= 2.0

    def test_discrete_params_are_integers(self):
        """Test discrete parameters return integer values."""
        space = create_muzero_hyperparameter_space()
        rng = np.random.default_rng(42)

        # Sample multiple times to check consistency
        for _ in range(10):
            samples = space.sample_all(rng)

            assert isinstance(samples["unroll_steps"], (int, np.integer))
            assert isinstance(samples["td_steps"], (int, np.integer))
            assert isinstance(samples["batch_size"], (int, np.integer))
            assert isinstance(samples["num_simulations"], (int, np.integer))

    def test_log_scale_params_use_log_sampling(self):
        """Test log scale parameters sample across orders of magnitude."""
        space = create_muzero_hyperparameter_space()
        rng = np.random.default_rng(42)

        # Sample many times to get distribution
        learning_rates = []
        batch_sizes = []

        for _ in range(100):
            samples = space.sample_all(rng)
            learning_rates.append(samples["learning_rate"])
            batch_sizes.append(samples["batch_size"])

        # Log scale should produce values spanning multiple orders
        lr_min = min(learning_rates)
        lr_max = max(learning_rates)
        assert lr_max / lr_min > 10  # At least 1 order of magnitude

        bs_min = min(batch_sizes)
        bs_max = max(batch_sizes)
        assert bs_max / bs_min > 2  # Significant range


# =============================================================================
# merge_with_base_space Tests
# =============================================================================


class TestMergeWithBaseSpace:
    """Test merging MuZero params with existing base space."""

    def test_combined_space_has_base_and_muzero_params(self):
        """Test merged space contains both base and MuZero parameters."""
        base_space = HyperparameterSpace()
        merged = merge_with_base_space(base_space)

        all_params = merged.get_all_params()

        # Should have base params
        assert "learning_rate" in all_params
        assert "batch_size" in all_params

        # Should have MuZero params
        for param_name in MUZERO_PARAM_NAMES:
            assert param_name in all_params

    def test_muzero_params_override_base_when_duplicate(self):
        """Test MuZero-specific overrides take precedence."""
        base_space = HyperparameterSpace()
        merged = merge_with_base_space(base_space)

        # c_puct should be overridden with wider range
        assert merged.c_puct.min_value == 0.5
        assert merged.c_puct.max_value == 5.0

        # batch_size should use log scale and reduced max
        assert merged.batch_size.scale == "log"
        assert merged.batch_size.max_value == 256

        # num_simulations should use log scale
        assert merged.num_simulations.scale == "log"

    def test_original_base_space_not_modified(self):
        """Test merge doesn't modify the original base space."""
        base_space = HyperparameterSpace()
        original_c_puct_max = base_space.c_puct.max_value
        original_batch_size_scale = base_space.batch_size.scale

        merge_with_base_space(base_space)

        # Original should be unchanged
        assert base_space.c_puct.max_value == original_c_puct_max
        assert base_space.batch_size.scale == original_batch_size_scale

    def test_base_params_not_in_muzero_preserved(self):
        """Test base parameters not overridden by MuZero are preserved."""
        base_space = HyperparameterSpace()
        merged = merge_with_base_space(base_space)

        # These should match base exactly
        assert merged.learning_rate.min_value == base_space.learning_rate.min_value
        assert merged.learning_rate.max_value == base_space.learning_rate.max_value
        assert merged.learning_rate.scale == base_space.learning_rate.scale

        assert merged.weight_decay.min_value == base_space.weight_decay.min_value
        assert merged.weight_decay.max_value == base_space.weight_decay.max_value

    def test_custom_params_merged_correctly(self):
        """Test custom_params dict combines base and MuZero."""
        base_space = HyperparameterSpace()
        base_space.custom_params["custom_param"] = HyperparameterRange(
            name="custom_param",
            min_value=0.0,
            max_value=1.0,
        )

        merged = merge_with_base_space(base_space)

        # Should have both base custom and MuZero params
        assert "custom_param" in merged.custom_params
        for param_name in MUZERO_PARAM_NAMES:
            assert param_name in merged.custom_params


# =============================================================================
# Mutation Tests
# =============================================================================


class TestMuZeroParamMutations:
    """Test mutation behavior for MuZero parameters."""

    def test_mutate_unroll_steps_produces_valid_int(self):
        """Test mutating unroll_steps produces valid integer."""
        params = get_muzero_custom_params()
        unroll_range = params["unroll_steps"]
        rng = np.random.default_rng(42)

        current_value = 5
        for _ in range(20):
            mutated = unroll_range.mutate(current_value, rng)
            assert isinstance(mutated, (int, np.integer))
            assert 3 <= mutated <= 10

    def test_mutate_td_steps_produces_valid_int(self):
        """Test mutating td_steps produces valid integer."""
        params = get_muzero_custom_params()
        td_range = params["td_steps"]
        rng = np.random.default_rng(42)

        current_value = 10
        for _ in range(20):
            mutated = td_range.mutate(current_value, rng)
            assert isinstance(mutated, (int, np.integer))
            assert 5 <= mutated <= 20

    def test_mutate_discount_stays_in_range(self):
        """Test mutating discount stays within bounds."""
        params = get_muzero_custom_params()
        discount_range = params["discount"]
        rng = np.random.default_rng(42)

        current_value = 0.997
        for _ in range(20):
            mutated = discount_range.mutate(current_value, rng)
            assert 0.95 <= mutated <= 0.999

    def test_mutate_consistency_loss_weight_stays_in_range(self):
        """Test mutating consistency_loss_weight stays within bounds."""
        params = get_muzero_custom_params()
        consistency_range = params["consistency_loss_weight"]
        rng = np.random.default_rng(42)

        current_value = 2.0
        for _ in range(20):
            mutated = consistency_range.mutate(current_value, rng)
            assert 0.5 <= mutated <= 5.0

    def test_mutations_produce_diversity(self):
        """Test mutations explore the parameter space."""
        params = get_muzero_custom_params()
        unroll_range = params["unroll_steps"]
        rng = np.random.default_rng(42)

        # Start at middle value
        current_value = 6
        mutated_values = set()

        # Mutate many times
        for _ in range(100):
            mutated = unroll_range.mutate(current_value, rng)
            mutated_values.add(mutated)

        # Should explore multiple values
        assert len(mutated_values) > 3


# =============================================================================
# Integration Tests
# =============================================================================


class TestMuZeroHyperparameterSpaceIntegration:
    """Integration tests for complete MuZero hyperparameter workflow."""

    def test_create_sample_mutate_workflow(self):
        """Test complete workflow: create space, sample, mutate."""
        space = create_muzero_hyperparameter_space()
        rng = np.random.default_rng(42)

        # Sample initial hyperparameters
        initial_params = space.sample_all(rng)

        # Verify all required params present
        assert "unroll_steps" in initial_params
        assert "td_steps" in initial_params
        assert "learning_rate" in initial_params

        # Mutate some parameters
        all_params = space.get_all_params()
        mutated_params = {}

        for name, param_range in all_params.items():
            current_value = initial_params[name]
            mutated_params[name] = param_range.mutate(current_value, rng)

        # Verify mutations are valid
        assert 3 <= mutated_params["unroll_steps"] <= 10
        assert 5 <= mutated_params["td_steps"] <= 20

    def test_merge_and_sample_workflow(self):
        """Test merging with base space and sampling."""
        base_space = HyperparameterSpace()
        merged_space = merge_with_base_space(base_space)
        rng = np.random.default_rng(123)

        # Sample from merged space
        samples = merged_space.sample_all(rng)

        # Should have both base and MuZero params
        assert "learning_rate" in samples
        assert "unroll_steps" in samples
        assert "td_steps" in samples

        # All values should be valid
        assert samples["learning_rate"] > 0
        assert isinstance(samples["unroll_steps"], (int, np.integer))

    def test_reproducibility_with_seed(self):
        """Test sampling is reproducible with same seed."""
        space = create_muzero_hyperparameter_space()

        # Sample twice with same seed
        rng1 = np.random.default_rng(42)
        samples1 = space.sample_all(rng1)

        rng2 = np.random.default_rng(42)
        samples2 = space.sample_all(rng2)

        # Should produce identical samples
        for key in samples1:
            assert samples1[key] == samples2[key]

    def test_different_seeds_produce_different_samples(self):
        """Test different seeds produce different samples."""
        space = create_muzero_hyperparameter_space()

        rng1 = np.random.default_rng(42)
        samples1 = space.sample_all(rng1)

        rng2 = np.random.default_rng(99)
        samples2 = space.sample_all(rng2)

        # Should differ in at least some values
        differences = sum(
            1 for key in samples1 if samples1[key] != samples2[key]
        )
        assert differences > 0
