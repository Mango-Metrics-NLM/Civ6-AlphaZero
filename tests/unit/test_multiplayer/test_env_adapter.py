"""Tests for N-player environment adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from alpha_zero.multiplayer.env_adapter import NPlayerEnvAdapter


class SimpleActionSpace:
    def __init__(self, n: int = 4):
        self.n = n


class MockEnv:
    """Minimal mock environment for testing the adapter."""

    def __init__(self, num_players: int = 2, action_space_size: int = 4):
        self.action_space = SimpleActionSpace(action_space_size)
        self.current_player = 0
        self._num_players = num_players
        self._step_count = 0

    @property
    def num_players(self):
        return self._num_players

    def reset(self, seed=None):
        self.current_player = 0
        self._step_count = 0
        return self.get_observation(), {}

    def step(self, action: int):
        self._step_count += 1
        reward = 1.0 if action == 0 else 0.0
        self.current_player = (self.current_player + 1) % self._num_players
        terminated = self._step_count >= 10
        info: dict[str, Any] = {}
        if terminated:
            info["winner"] = 0
        return self.get_observation(), reward, terminated, False, info

    def get_observation(self):
        return {
            "spatial": np.zeros((13, 4, 4), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
            "action_mask": np.ones(self.action_space.n, dtype=np.float32),
        }

    def get_info(self):
        return {}

    def get_valid_actions(self):
        return np.ones(self.action_space.n, dtype=np.float32)

    def clone(self):
        cloned = MockEnv(self._num_players, self.action_space.n)
        cloned.current_player = self.current_player
        cloned._step_count = self._step_count
        return cloned


class TestNPlayerEnvAdapter:
    """Tests for NPlayerEnvAdapter."""

    @pytest.fixture
    def env(self):
        return MockEnv(num_players=3)

    @pytest.fixture
    def adapter(self, env):
        return NPlayerEnvAdapter(env, num_players=3)

    def test_num_players(self, adapter):
        assert adapter.num_players == 3

    def test_action_space(self, adapter):
        assert adapter.action_space.n == 4

    def test_current_player(self, adapter):
        adapter.reset()
        assert adapter.current_player == 0

    def test_reset(self, adapter):
        obs, info = adapter.reset()
        assert "spatial" in obs
        assert "num_players" in info
        assert info["num_players"] == 3

    def test_step_advances_player(self, adapter):
        adapter.reset()
        assert adapter.current_player == 0
        adapter.step(0)
        assert adapter.current_player == 1
        adapter.step(0)
        assert adapter.current_player == 2
        adapter.step(0)
        assert adapter.current_player == 0  # Wraps around

    def test_step_returns_enriched_info(self, adapter):
        adapter.reset()
        _, _, _, _, info = adapter.step(0)
        assert "num_players" in info
        assert "player_scores" in info
        assert "eliminated" in info

    def test_score_tracking(self, adapter):
        adapter.reset()
        adapter.step(0)  # Player 0, reward 1.0
        adapter.step(1)  # Player 1, reward 0.0

        _, _, _, _, info = adapter.step(0)  # Player 2, reward 1.0
        scores = info["player_scores"]
        assert scores[0] == 1.0
        assert scores[1] == 0.0

    def test_clone(self, adapter):
        adapter.reset()
        adapter.step(0)

        cloned = adapter.clone()
        assert cloned.num_players == adapter.num_players
        assert cloned.current_player == adapter.current_player

    def test_clone_independence(self, adapter):
        """Cloned adapter is independent of original."""
        adapter.reset()
        cloned = adapter.clone()

        adapter.step(0)
        assert adapter.current_player != cloned.current_player

    def test_get_observation(self, adapter):
        adapter.reset()
        obs = adapter.get_observation()
        assert "spatial" in obs
        assert "vector" in obs
        assert "action_mask" in obs

    def test_get_valid_actions(self, adapter):
        adapter.reset()
        mask = adapter.get_valid_actions()
        assert mask.shape == (4,)
        assert np.all(mask == 1)

    def test_get_info_enriched(self, adapter):
        adapter.reset()
        info = adapter.get_info()
        assert "num_players" in info
        assert info["num_players"] == 3

    def test_move_history(self, adapter):
        adapter.reset()
        adapter.step(0)
        adapter.step(2)

        history = adapter.move_history
        assert len(history) == 2
        assert history[0]["player"] == 0
        assert history[0]["action"] == 0
        assert history[1]["player"] == 1
        assert history[1]["action"] == 2

    def test_terminal_state_has_rankings(self, adapter):
        """Terminal state info includes rankings."""
        adapter.reset()
        # Step until terminal
        for _ in range(20):
            _, _, terminated, _, info = adapter.step(0)
            if terminated:
                break

        if terminated:
            assert "rankings" in info

    def test_repr(self, adapter):
        """repr includes useful info."""
        s = repr(adapter)
        assert "NPlayerEnvAdapter" in s
        assert "num_players=3" in s

    def test_score_tracking_disabled(self):
        """Can disable score tracking."""
        env = MockEnv(num_players=2)
        adapter = NPlayerEnvAdapter(env, num_players=2, track_scores=False)
        adapter.reset()
        adapter.step(0)
        # Scores should still be initialized but at 0
        info = adapter.get_info()
        assert all(v == 0.0 for v in info["player_scores"].values())

    def test_reset_clears_state(self, adapter):
        """Reset clears all tracked state."""
        adapter.reset()
        adapter.step(0)
        adapter.step(0)

        adapter.reset()
        assert len(adapter.move_history) == 0
        info = adapter.get_info()
        assert info["turn_count"] == 0
        assert all(v == 0.0 for v in info["player_scores"].values())
