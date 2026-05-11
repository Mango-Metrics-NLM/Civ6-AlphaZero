"""Tests for :class:`NetworkOpponent`."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from alpha_zero.envs.chess_actions import ACTION_SPACE_SIZE
from alpha_zero.envs.chess_env import ChessEnv
from alpha_zero.opponents.network_opponent import NetworkOpponent


def _make_network_returning(probabilities: np.ndarray) -> MagicMock:
    """Return a mock network whose forward() yields logits ≈ ``probabilities``."""
    import torch

    logits = torch.log(torch.from_numpy(probabilities).float().clamp_min(1e-9))
    logits = logits.unsqueeze(0)
    value = torch.zeros((1, 1))
    network = MagicMock()
    network.forward.return_value = (logits, value)
    network.eval.return_value = None
    return network


@pytest.fixture
def env() -> ChessEnv:
    e = ChessEnv(history_depth=1)
    e.reset()
    return e


class TestGreedy:
    def test_argmax_over_legal(self, env: ChessEnv) -> None:
        # Construct policy with all mass on action index 0 (a1-pawn-N, illegal)
        # and slight mass on a legal action. Legal mask should force the
        # legal action to win.
        torch = pytest.importorskip("torch")
        del torch
        probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
        probs[0] = 0.99
        # Find a known-legal action from the starting mask.
        mask = env.get_valid_actions()
        legal_indices = np.flatnonzero(mask > 0)
        target = int(legal_indices[0])
        probs[target] = 0.01
        # Normalize so it's a proper distribution.
        probs /= probs.sum()
        net = _make_network_returning(probs.astype(np.float32))
        op = NetworkOpponent(network=net, temperature=0.0)
        action = op.get_action(env, mask)
        assert mask[action] == 1.0
        # Since action 0 is masked out, the only non-zero masked mass is target.
        assert action == target

    def test_uniform_fallback_on_zero_mass(self, env: ChessEnv) -> None:
        pytest.importorskip("torch")
        mask = env.get_valid_actions()
        probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        probs[~(mask > 0)] = 1e-3  # Put mass only on illegal actions.
        # Re-normalize.
        probs = probs / probs.sum() if probs.sum() > 0 else probs
        net = _make_network_returning(probs)
        op = NetworkOpponent(network=net)
        action = op.get_action(env, mask)
        assert mask[action] == 1.0


class TestTemperature:
    def test_negative_temperature_raises(self) -> None:
        net = MagicMock()
        with pytest.raises(ValueError, match="temperature"):
            NetworkOpponent(network=net, temperature=-0.1)

    def test_temperature_one_samples_legal(self, env: ChessEnv) -> None:
        pytest.importorskip("torch")
        mask = env.get_valid_actions()
        # Uniform legal-only distribution.
        probs = mask / mask.sum()
        net = _make_network_returning(probs.astype(np.float32))
        op = NetworkOpponent(network=net, temperature=1.0)
        actions = {op.get_action(env, mask) for _ in range(10)}
        assert all(mask[a] == 1.0 for a in actions)


class TestLifecycle:
    def test_name_default(self) -> None:
        net = MagicMock()
        op = NetworkOpponent(network=net)
        assert op.name == "network"

    def test_reset_and_close_are_noops(self) -> None:
        net = MagicMock()
        op = NetworkOpponent(network=net)
        assert op.reset() is None
        assert op.close() is None
