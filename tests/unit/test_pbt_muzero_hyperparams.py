"""Tests for alpha_zero.pbt.muzero_hyperparams module."""

from __future__ import annotations

import pytest

from alpha_zero.pbt.config import HyperparameterRange, HyperparameterSpace
from alpha_zero.pbt.muzero_hyperparams import (
    MUZERO_PARAM_NAMES,
    create_muzero_hyperparameter_space,
    get_muzero_custom_params,
    merge_with_base_space,
)


class TestMuzeroParamNames:
    """Test the MUZERO_PARAM_NAMES constant."""

    def test_is_frozenset(self) -> None:
        assert isinstance(MUZERO_PARAM_NAMES, frozenset)

    def test_contains_expected_names(self) -> None:
        expected = {
            "unroll_steps",
            "td_steps",
            "discount",
            "consistency_loss_weight",
            "reward_loss_weight",
        }
        assert MUZERO_PARAM_NAMES == expected

    def test_has_five_params(self) -> None:
        assert len(MUZERO_PARAM_NAMES) == 5


class TestGetMuzeroCustomParams:
    """Test get_muzero_custom_params factory function."""

    def test_returns_dict(self) -> None:
        params = get_muzero_custom_params()
        assert isinstance(params, dict)

    def test_keys_match_param_names(self) -> None:
        params = get_muzero_custom_params()
        assert set(params.keys()) == MUZERO_PARAM_NAMES

    def test_all_values_are_hyperparameter_ranges(self) -> None:
        params = get_muzero_custom_params()
        for name, param_range in params.items():
            assert isinstance(param_range, HyperparameterRange), (
                f"{name} is not HyperparameterRange"
            )

    def test_unroll_steps_is_discrete(self) -> None:
        params = get_muzero_custom_params()
        assert params["unroll_steps"].discrete is True
        assert params["unroll_steps"].min_value == 3
        assert params["unroll_steps"].max_value == 10

    def test_td_steps_is_discrete(self) -> None:
        params = get_muzero_custom_params()
        assert params["td_steps"].discrete is True
        assert params["td_steps"].min_value == 5
        assert params["td_steps"].max_value == 20

    def test_discount_is_continuous(self) -> None:
        params = get_muzero_custom_params()
        assert params["discount"].discrete is False
        assert params["discount"].min_value == pytest.approx(0.95)
        assert params["discount"].max_value == pytest.approx(0.999)

    def test_consistency_loss_weight_range(self) -> None:
        params = get_muzero_custom_params()
        assert params["consistency_loss_weight"].min_value == pytest.approx(0.5)
        assert params["consistency_loss_weight"].max_value == pytest.approx(5.0)

    def test_reward_loss_weight_range(self) -> None:
        params = get_muzero_custom_params()
        assert params["reward_loss_weight"].min_value == pytest.approx(0.5)
        assert params["reward_loss_weight"].max_value == pytest.approx(2.0)

    def test_all_have_linear_scale(self) -> None:
        params = get_muzero_custom_params()
        for name, param_range in params.items():
            assert param_range.scale == "linear", f"{name} should have linear scale"


class TestCreateMuzeroHyperparameterSpace:
    """Test create_muzero_hyperparameter_space factory function."""

    def test_returns_hyperparameter_space(self) -> None:
        space = create_muzero_hyperparameter_space()
        assert isinstance(space, HyperparameterSpace)

    def test_has_custom_params(self) -> None:
        space = create_muzero_hyperparameter_space()
        assert len(space.custom_params) == 5
        assert set(space.custom_params.keys()) == MUZERO_PARAM_NAMES

    def test_c_puct_wider_range(self) -> None:
        space = create_muzero_hyperparameter_space()
        assert space.c_puct.min_value == pytest.approx(0.5)
        assert space.c_puct.max_value == pytest.approx(5.0)

    def test_batch_size_smaller_max(self) -> None:
        space = create_muzero_hyperparameter_space()
        assert space.batch_size.max_value == 256
        assert space.batch_size.scale == "log"
        assert space.batch_size.discrete is True

    def test_num_simulations_log_scale(self) -> None:
        space = create_muzero_hyperparameter_space()
        assert space.num_simulations.scale == "log"
        assert space.num_simulations.min_value == 50
        assert space.num_simulations.max_value == 400

    def test_learning_rate_log_scale(self) -> None:
        space = create_muzero_hyperparameter_space()
        assert space.learning_rate.scale == "log"
        assert space.learning_rate.min_value == pytest.approx(1e-5)
        assert space.learning_rate.max_value == pytest.approx(1e-2)

    def test_get_all_params_includes_custom(self) -> None:
        space = create_muzero_hyperparameter_space()
        all_params = space.get_all_params()
        for name in MUZERO_PARAM_NAMES:
            assert name in all_params, f"Missing custom param: {name}"

    def test_total_param_count(self) -> None:
        space = create_muzero_hyperparameter_space()
        all_params = space.get_all_params()
        # 11 base params + 5 custom = 16 total
        assert len(all_params) == 16


class TestMergeWithBaseSpace:
    """Test merge_with_base_space function."""

    def test_returns_new_space(self) -> None:
        base = HyperparameterSpace()
        merged = merge_with_base_space(base)
        assert isinstance(merged, HyperparameterSpace)
        assert merged is not base

    def test_preserves_base_learning_rate(self) -> None:
        base = HyperparameterSpace()
        merged = merge_with_base_space(base)
        assert merged.learning_rate.min_value == base.learning_rate.min_value
        assert merged.learning_rate.max_value == base.learning_rate.max_value

    def test_overrides_c_puct(self) -> None:
        base = HyperparameterSpace()
        merged = merge_with_base_space(base)
        assert merged.c_puct.min_value == pytest.approx(0.5)
        assert merged.c_puct.max_value == pytest.approx(5.0)

    def test_overrides_batch_size(self) -> None:
        base = HyperparameterSpace()
        merged = merge_with_base_space(base)
        assert merged.batch_size.max_value == 256
        assert merged.batch_size.scale == "log"

    def test_adds_muzero_custom_params(self) -> None:
        base = HyperparameterSpace()
        merged = merge_with_base_space(base)
        for name in MUZERO_PARAM_NAMES:
            assert name in merged.custom_params

    def test_preserves_base_custom_params(self) -> None:
        base = HyperparameterSpace(
            custom_params={
                "my_param": HyperparameterRange(name="my_param", min_value=0.0, max_value=1.0)
            }
        )
        merged = merge_with_base_space(base)
        assert "my_param" in merged.custom_params
        # Also has MuZero params
        for name in MUZERO_PARAM_NAMES:
            assert name in merged.custom_params

    def test_does_not_mutate_base(self) -> None:
        base = HyperparameterSpace()
        original_custom_count = len(base.custom_params)
        merge_with_base_space(base)
        assert len(base.custom_params) == original_custom_count

    def test_merged_total_params(self) -> None:
        base = HyperparameterSpace()
        merged = merge_with_base_space(base)
        all_params = merged.get_all_params()
        assert len(all_params) == 16  # 11 base + 5 MuZero custom
