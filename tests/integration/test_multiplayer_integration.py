"""Integration tests for multiplayer module components working together.

Tests end-to-end workflows combining MCTS, environment adapters,
network heads, turn management, and opponent modeling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from alpha_zero.multiplayer.config import (
    NPlayerMCTSConfig,
    OpponentModelConfig,
    TurnManagerConfig,
)
from alpha_zero.multiplayer.env_adapter import NPlayerEnvAdapter
from alpha_zero.multiplayer.mcts import NPlayerMCTS
from alpha_zero.multiplayer.opponent_model import OpponentModel
from alpha_zero.multiplayer.scoring import (
    ScoringMethod,
    compute_player_values,
    extract_game_result,
)
from alpha_zero.multiplayer.turn_manager import TurnManager


class SimpleActionSpace:
    """Minimal action space for integration tests."""

    def __init__(self, n: int = 4):
        self.n = n


class IntegrationMockEnv:
    """Mock environment with richer state for integration testing."""

    def __init__(self, num_players: int = 3, action_space_size: int = 4):
        self.action_space = SimpleActionSpace(action_space_size)
        self.current_player = 0
        self.num_players = num_players
        self._step_count = 0
        self._max_steps = 15
        self._scores: dict[int, float] = dict.fromkeys(range(num_players), 0.0)

    def clone(self) -> IntegrationMockEnv:
        cloned = IntegrationMockEnv(self.num_players, self.action_space.n)
        cloned.current_player = self.current_player
        cloned._step_count = self._step_count
        cloned._scores = dict(self._scores)
        return cloned

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        player = self.current_player
        reward = float(action % 2)  # Odd actions give reward
        self._scores[player] += reward
        self._step_count += 1
        self.current_player = (self.current_player + 1) % self.num_players

        terminated = self._step_count >= self._max_steps
        info: dict[str, Any] = {"player_scores": dict(self._scores)}
        if terminated:
            winner = max(self._scores, key=lambda k: self._scores[k])
            info["winner"] = winner
            info["scores"] = self._scores

        return self.get_observation(), reward, terminated, False, info

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "spatial": np.zeros((13, 4, 4), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
            "action_mask": np.ones(self.action_space.n, dtype=np.float32),
        }

    def get_info(self) -> dict[str, Any]:
        return {"player_scores": dict(self._scores)}

    def get_valid_actions(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.current_player = 0
        self._step_count = 0
        self._scores = dict.fromkeys(range(self.num_players), 0.0)
        return self.get_observation(), {}


class IntegrationMockNetwork:
    """Mock network for integration tests."""

    def __init__(self, action_space_size: int = 4, num_players: int = 3):
        self._action_space_size = action_space_size
        self._num_players = num_players

    @property
    def num_players(self) -> int:
        return self._num_players

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = spatial.shape[0]
        policy = torch.ones(b, self._action_space_size) / self._action_space_size
        values = torch.zeros(b, self._num_players)
        return policy, values

    def eval(self) -> None:
        pass


class TestMCTSWithEnvAdapterIntegration:
    """Integration: MCTS search using NPlayerEnvAdapter."""

    def test_mcts_search_with_adapter(self):
        """Full MCTS search through env adapter produces valid trajectory."""
        env = IntegrationMockEnv(num_players=3, action_space_size=4)
        network = IntegrationMockNetwork(action_space_size=4, num_players=3)
        adapter = NPlayerEnvAdapter(env, num_players=3)

        adapter.reset()
        config = NPlayerMCTSConfig(num_simulations=10)
        mcts = NPlayerMCTS(network, config, num_players=3)

        action_probs, values = mcts.search(adapter)
        assert action_probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert len(values) == 3

        # Take the best action
        action = int(np.argmax(action_probs))
        _obs, _reward, _terminated, _truncated, info = adapter.step(action)
        assert "player_scores" in info

    def test_multiple_turns_with_scoring(self):
        """Multiple MCTS turns produce valid game results."""
        env = IntegrationMockEnv(num_players=2, action_space_size=4)
        network = IntegrationMockNetwork(action_space_size=4, num_players=2)
        adapter = NPlayerEnvAdapter(env, num_players=2)

        adapter.reset()
        config = NPlayerMCTSConfig(num_simulations=5)
        mcts = NPlayerMCTS(network, config, num_players=2)

        # Play several turns
        for _ in range(10):
            action_probs, _ = mcts.search(adapter)
            action = int(np.argmax(action_probs))
            _, _, terminated, _, info = adapter.step(action)
            if terminated:
                break

        # Game should reach terminal state
        info = adapter.get_info()
        assert "player_scores" in info


class TestTurnManagerWithOpponentModel:
    """Integration: TurnManager tracks turns while OpponentModel learns."""

    def test_sequential_turns_with_opponent_tracking(self):
        """Turn manager correctly sequences and opponent model tracks actions."""
        config = TurnManagerConfig(mode="sequential")
        manager = TurnManager(config, num_players=3)

        opp_config = OpponentModelConfig(enabled=True, history_window=20)
        opponent_model = OpponentModel(opp_config, num_players=3, action_space_size=10)

        for _ in range(6):
            state = manager.begin_turn()
            active_player = state.active_players[0]

            # Simulate deterministic action selection
            action = active_player * 2  # Each player picks different action
            opponent_model.update(active_player, {}, action)
            manager.submit_action(active_player, action)
            manager.end_turn()

        # Each player should have 2 observations
        for p in range(3):
            summary = opponent_model.get_player_summary(p)
            assert summary["total_observations"] == 2

    def test_dynamic_mode_war_transition(self):
        """Dynamic mode transitions with opponent model still functional."""
        config = TurnManagerConfig(mode="dynamic")
        manager = TurnManager(config, num_players=3)

        opp_config = OpponentModelConfig(enabled=True, history_window=10)
        opponent_model = OpponentModel(opp_config, num_players=3, action_space_size=10)

        # Start simultaneous
        state = manager.begin_turn()
        assert len(state.active_players) == 3

        # War switches to sequential
        manager.declare_war(0, 1)
        manager.end_turn()

        state = manager.begin_turn()
        assert len(state.active_players) == 1

        # Can still track opponents
        opponent_model.update(state.active_players[0], {}, 5)
        manager.end_turn()

        # Peace restores simultaneous
        manager.make_peace(0, 1)
        state = manager.begin_turn()
        assert len(state.active_players) == 3


class TestScoringPipelineIntegration:
    """Integration: Full scoring pipeline from game result to training values."""

    def test_extract_and_score_game_result(self):
        """Game info dict flows through ranking -> value computation."""
        info = {
            "winner": 0,
            "scores": {0: 100.0, 1: 75.0, 2: 50.0},
            "eliminated": {2},
        }
        num_players = 3

        # Full pipeline
        values = extract_game_result(info, num_players)

        assert len(values) == 3
        assert values[0] > values[1]  # Winner > 2nd place
        assert values[2] < values[1]  # Eliminated < 2nd place

    def test_draw_produces_equal_values(self):
        """Draw game produces equal values for all players."""
        info = {"winner": None}
        num_players = 4

        values = extract_game_result(info, num_players)
        assert all(abs(v - values[0]) < 1e-6 for v in values.values())

    def test_rankings_to_values_roundtrip(self):
        """Rankings correctly map to training values."""
        rankings = {0: 0, 1: 1, 2: 2, 3: 3}  # 0=best, 3=worst
        values = compute_player_values(rankings, num_players=4, method=ScoringMethod.LINEAR_RANKING)

        assert values[0] == pytest.approx(1.0)
        assert values[3] == pytest.approx(-1.0)
        # Middle players should be between
        assert -1.0 < values[1] < 1.0
        assert -1.0 < values[2] < 1.0


class TestMCTSStrategiesIntegration:
    """Integration: Different MCTS strategies produce valid results."""

    @pytest.fixture
    def env(self):
        return IntegrationMockEnv(num_players=4, action_space_size=4)

    @pytest.fixture
    def network(self):
        return IntegrationMockNetwork(action_space_size=4, num_players=4)

    @pytest.mark.parametrize("strategy", ["max_n", "paranoid", "best_reply"])
    def test_all_strategies_produce_valid_results(self, env, network, strategy):
        """Each search strategy produces valid action probs and values."""
        env.reset()
        config = NPlayerMCTSConfig(num_simulations=10, search_strategy=strategy)
        mcts = NPlayerMCTS(network, config, num_players=4)

        action_probs, values = mcts.search(env)
        assert action_probs.shape == (4,)
        assert np.all(action_probs >= 0)
        assert action_probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert len(values) == 4
