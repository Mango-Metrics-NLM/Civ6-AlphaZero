"""Tests for N-player MCTS search."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from alpha_zero.multiplayer.config import NPlayerMCTSConfig
from alpha_zero.multiplayer.mcts import NPlayerMCTS


class SimpleActionSpace:
    """Simple action space for testing."""

    def __init__(self, n: int = 4):
        self.n = n


class MockMultiplayerEnv:
    """Mock multiplayer environment for testing MCTS."""

    def __init__(
        self,
        num_players: int = 2,
        action_space_size: int = 4,
        board_size: int = 4,
        max_steps: int = 20,
    ):
        self.num_players = num_players
        self.action_space = SimpleActionSpace(action_space_size)
        self._board_size = board_size
        self._max_steps = max_steps
        self._current_player = 0
        self._step_count = 0
        self._terminated = False

    @property
    def current_player(self) -> int:
        return self._current_player

    def clone(self) -> MockMultiplayerEnv:
        cloned = MockMultiplayerEnv(
            num_players=self.num_players,
            action_space_size=self.action_space.n,
            board_size=self._board_size,
            max_steps=self._max_steps,
        )
        cloned._current_player = self._current_player
        cloned._step_count = self._step_count
        cloned._terminated = self._terminated
        return cloned

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        self._current_player = (self._current_player + 1) % self.num_players

        terminated = self._step_count >= self._max_steps
        self._terminated = terminated

        info: dict[str, Any] = {}
        if terminated:
            info["winner"] = 0  # Player 0 always wins in mock

        return self.get_observation(), 0.0, terminated, False, info

    def get_observation(self) -> dict[str, np.ndarray]:
        spatial = np.zeros((13, self._board_size, self._board_size), dtype=np.float32)
        vector = np.zeros(64, dtype=np.float32)
        mask = np.ones(self.action_space.n, dtype=np.float32)
        return {"spatial": spatial, "vector": vector, "action_mask": mask}

    def get_info(self) -> dict[str, Any]:
        return {"winner": None if not self._terminated else 0}

    def get_valid_actions(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=np.float32)

    def reset(self, seed=None):
        self._current_player = 0
        self._step_count = 0
        self._terminated = False
        return self.get_observation(), {}


class MockMultiplayerNetwork:
    """Mock network that returns uniform policy and zero values."""

    def __init__(self, action_space_size: int = 4, num_players: int = 2):
        self._action_space_size = action_space_size
        self._num_players = num_players

    @property
    def num_players(self) -> int:
        return self._num_players

    def predict(
        self, spatial: torch.Tensor, vector: torch.Tensor, action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = spatial.shape[0]
        policy = torch.ones(batch_size, self._action_space_size) / self._action_space_size
        values = torch.zeros(batch_size, self._num_players)
        return policy, values

    def eval(self):
        pass


class TestNPlayerMCTS:
    """Tests for NPlayerMCTS search."""

    @pytest.fixture
    def env(self):
        return MockMultiplayerEnv(num_players=3, action_space_size=4)

    @pytest.fixture
    def network(self):
        return MockMultiplayerNetwork(action_space_size=4, num_players=3)

    @pytest.fixture
    def config(self):
        return NPlayerMCTSConfig(num_simulations=20, c_puct=1.5)

    @pytest.fixture
    def mcts(self, network, config):
        return NPlayerMCTS(network, config, num_players=3)

    def test_search_returns_valid_probs(self, mcts, env):
        """Search returns valid probability distribution."""
        env.reset()
        action_probs, _values = mcts.search(env)
        assert action_probs.shape == (env.action_space.n,)
        assert np.all(action_probs >= 0)
        assert action_probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_search_returns_per_player_values(self, mcts, env):
        """Search returns value dict with entry per player."""
        env.reset()
        _, values = mcts.search(env)
        assert isinstance(values, dict)
        assert len(values) == 3
        assert all(p in values for p in range(3))

    def test_search_values_bounded(self, mcts, env):
        """All values are in reasonable range."""
        env.reset()
        _, values = mcts.search(env)
        for v in values.values():
            assert -2.0 <= v <= 2.0

    def test_search_respects_action_mask(self, mcts, env):
        """Search doesn't assign probability to masked actions."""
        # Override valid actions to only allow action 0 and 2
        original_get_obs = env.get_observation

        def restricted_obs():
            obs = original_get_obs()
            obs["action_mask"] = np.array([1, 0, 1, 0], dtype=np.float32)
            return obs

        env.get_observation = restricted_obs
        env.get_valid_actions = lambda: np.array([1, 0, 1, 0], dtype=np.float32)

        env.reset()
        action_probs, _ = mcts.search(env)
        assert action_probs[1] == pytest.approx(0.0, abs=1e-6)
        assert action_probs[3] == pytest.approx(0.0, abs=1e-6)

    def test_select_action_greedy(self):
        """Temperature=0 selects most probable action."""
        probs = np.array([0.1, 0.7, 0.1, 0.1])
        action = NPlayerMCTS.select_action(probs, temperature=0.0)
        assert action == 1

    def test_select_action_with_mask(self):
        """Action selection respects mask."""
        probs = np.array([0.7, 0.3, 0.0, 0.0])
        mask = np.array([0, 1, 0, 0])
        action = NPlayerMCTS.select_action(probs, temperature=0.0, action_mask=mask)
        assert action == 1

    def test_select_action_stochastic(self):
        """Stochastic selection samples from distribution."""
        probs = np.array([0.5, 0.5, 0.0, 0.0])
        np.random.seed(42)
        actions = [NPlayerMCTS.select_action(probs, temperature=1.0) for _ in range(100)]
        assert 0 in actions
        assert 1 in actions
        assert 2 not in actions
        assert 3 not in actions

    def test_search_with_paranoid_strategy(self):
        """Paranoid search completes without error."""
        env = MockMultiplayerEnv(num_players=3, action_space_size=4)
        network = MockMultiplayerNetwork(action_space_size=4, num_players=3)
        config = NPlayerMCTSConfig(
            num_simulations=10, search_strategy="paranoid"
        )
        mcts = NPlayerMCTS(network, config, num_players=3)

        env.reset()
        action_probs, values = mcts.search(env)
        assert action_probs.shape == (4,)
        assert len(values) == 3

    def test_search_with_best_reply_strategy(self):
        """Best-reply search completes without error."""
        env = MockMultiplayerEnv(num_players=4, action_space_size=4)
        network = MockMultiplayerNetwork(action_space_size=4, num_players=4)
        config = NPlayerMCTSConfig(
            num_simulations=10, search_strategy="best_reply"
        )
        mcts = NPlayerMCTS(network, config, num_players=4)

        env.reset()
        action_probs, values = mcts.search(env)
        assert action_probs.shape == (4,)
        assert len(values) == 4

    def test_search_no_noise(self):
        """Search works without Dirichlet noise."""
        env = MockMultiplayerEnv(num_players=2, action_space_size=4)
        network = MockMultiplayerNetwork(action_space_size=4, num_players=2)
        config = NPlayerMCTSConfig(num_simulations=10)
        mcts = NPlayerMCTS(network, config, num_players=2)

        env.reset()
        action_probs, _ = mcts.search(env, add_noise=False)
        assert action_probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_two_player_backward_compatible(self):
        """2-player MCTS produces valid results."""
        env = MockMultiplayerEnv(num_players=2, action_space_size=4)
        network = MockMultiplayerNetwork(action_space_size=4, num_players=2)
        config = NPlayerMCTSConfig(num_simulations=10)
        mcts = NPlayerMCTS(network, config, num_players=2)

        env.reset()
        action_probs, values = mcts.search(env)
        assert len(values) == 2
        assert action_probs.shape == (4,)

    def test_legacy_scalar_network(self):
        """MCTS handles legacy networks that return scalar values."""

        class LegacyNetwork:
            @property
            def num_players(self):
                return 2

            def predict(self, spatial, vector, mask):
                batch_size = spatial.shape[0]
                policy = torch.ones(batch_size, 4) / 4
                value = torch.zeros(batch_size)  # Scalar, not vector
                return policy, value

            def eval(self):
                pass

        env = MockMultiplayerEnv(num_players=2, action_space_size=4)
        config = NPlayerMCTSConfig(num_simulations=5)
        mcts = NPlayerMCTS(LegacyNetwork(), config, num_players=2)

        env.reset()
        _action_probs, values = mcts.search(env)
        assert len(values) == 2

    def test_multiple_searches_consistent(self, mcts, env):
        """Multiple searches don't corrupt internal state."""
        env.reset()
        for _ in range(5):
            probs, _values = mcts.search(env, add_noise=False)
            assert probs.shape == (4,)
            assert probs.sum() == pytest.approx(1.0, abs=1e-5)
