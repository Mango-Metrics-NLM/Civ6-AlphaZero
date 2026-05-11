"""Tests for the LLM-prior blending seam in :class:`Civ6MCTS`.

Critical contract under test: when ``llm_prior_fn=None``, behaviour must be
byte-identical to the pre-extension code path. This is the hard backwards-
compatibility guarantee for the MCTS modification.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig


def _make_mcts(**kwargs):  # type: ignore[no-untyped-def]
    network = MagicMock()
    return Civ6MCTS(network=network, config=MCTSConfig(num_simulations=1), **kwargs)


class TestBlendingMath:
    def test_no_prior_is_identity(self) -> None:
        mcts = _make_mcts()
        policy = np.array([0.5, 0.3, 0.2, 0.0], dtype=np.float32)
        mask = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        result = mcts._blend_llm_prior(env=None, network_policy=policy, action_mask=mask)
        np.testing.assert_array_equal(result, policy)

    def test_blend_weight_zero_recovers_network(self) -> None:
        mcts = _make_mcts(
            llm_prior_fn=lambda env, mask: np.array([0.0, 1.0, 0.0, 0.0]),
            llm_prior_weight=0.0,
        )
        policy = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        mask = np.ones(4, dtype=np.float32)
        result = mcts._blend_llm_prior(env=None, network_policy=policy, action_mask=mask)
        np.testing.assert_allclose(result, policy, rtol=1e-6)

    def test_blend_weight_one_recovers_llm(self) -> None:
        llm_prior = np.array([0.0, 1.0, 0.0, 0.0])
        mcts = _make_mcts(
            llm_prior_fn=lambda env, mask: llm_prior,
            llm_prior_weight=1.0,
        )
        policy = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        mask = np.ones(4, dtype=np.float32)
        result = mcts._blend_llm_prior(env=None, network_policy=policy, action_mask=mask)
        np.testing.assert_allclose(result, llm_prior, rtol=1e-6)

    def test_blended_distribution_sums_to_one(self) -> None:
        mcts = _make_mcts(
            llm_prior_fn=lambda env, mask: np.array([0.5, 0.5, 0.0, 0.0]),
            llm_prior_weight=0.5,
        )
        policy = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        mask = np.ones(4, dtype=np.float32)
        result = mcts._blend_llm_prior(env=None, network_policy=policy, action_mask=mask)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_mask_zeroes_propagate(self) -> None:
        mcts = _make_mcts(
            llm_prior_fn=lambda env, mask: np.array([0.5, 0.5, 0.0, 0.0]),
            llm_prior_weight=0.5,
        )
        policy = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        mask = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        result = mcts._blend_llm_prior(env=None, network_policy=policy, action_mask=mask)
        assert result[2] == 0.0
        assert result[3] == 0.0
        assert abs(result.sum() - 1.0) < 1e-6


class TestErrorPaths:
    def test_callable_exception_falls_back_to_network(self) -> None:
        def boom(env, mask):  # type: ignore[no-untyped-def]
            raise RuntimeError("network unavailable")

        mcts = _make_mcts(llm_prior_fn=boom)
        policy = np.array([0.5, 0.5], dtype=np.float32)
        mask = np.ones(2, dtype=np.float32)
        result = mcts._blend_llm_prior(env=None, network_policy=policy, action_mask=mask)
        np.testing.assert_array_equal(result, policy)

    def test_shape_mismatch_falls_back_to_network(self) -> None:
        mcts = _make_mcts(
            llm_prior_fn=lambda env, mask: np.array([0.5, 0.5, 0.0]),
        )
        policy = np.array([0.5, 0.5], dtype=np.float32)
        mask = np.ones(2, dtype=np.float32)
        result = mcts._blend_llm_prior(env=None, network_policy=policy, action_mask=mask)
        np.testing.assert_array_equal(result, policy)

    def test_zero_legal_llm_mass_falls_back(self) -> None:
        # LLM returns mass only on masked-out actions.
        mcts = _make_mcts(
            llm_prior_fn=lambda env, mask: np.array([0.0, 1.0]),
        )
        policy = np.array([0.5, 0.5], dtype=np.float32)
        mask = np.array([1.0, 0.0], dtype=np.float32)
        result = mcts._blend_llm_prior(env=None, network_policy=policy, action_mask=mask)
        np.testing.assert_array_equal(result, policy)

    def test_invalid_weight_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="llm_prior_weight"):
            _make_mcts(
                llm_prior_fn=lambda env, mask: np.array([1.0]),
                llm_prior_weight=1.5,
            )
