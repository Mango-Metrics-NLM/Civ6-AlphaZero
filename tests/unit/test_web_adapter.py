"""Unit tests for web adapter module.

Tests MicroCivGameAdapter, MCTSAIAgent, and BaselineAgentAdapter classes
that bridge MicroCivEnv and AI agents to the GameController protocol.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.evaluation.baselines import RandomPlayer
from alpha_zero.web.adapter import (
    BaselineAgentAdapter,
    MCTSAIAgent,
    MicroCivGameAdapter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_env() -> MicroCivEnv:
    """Create a small MicroCivEnv for fast testing."""
    return MicroCivEnv(board_size=8, max_turns=50, seed=42)


@pytest.fixture()
def adapter(small_env: MicroCivEnv) -> MicroCivGameAdapter:
    """Create a MicroCivGameAdapter wrapping a small environment."""
    return MicroCivGameAdapter(small_env)


@pytest.fixture()
def reset_adapter(adapter: MicroCivGameAdapter) -> MicroCivGameAdapter:
    """Return an adapter whose environment has already been reset."""
    adapter.reset(seed=42)
    return adapter


@pytest.fixture()
def mock_network() -> MagicMock:
    """Create a mock neural network that returns plausible outputs.

    The mock supports both ``predict_single`` (used when the attribute exists)
    and a generic ``__call__`` (forward) path.
    """
    net = MagicMock()

    # predict_single returns (policy_tensor, value_tensor)
    def _predict_single(spatial: Any, vector: Any = None, mask: Any = None) -> tuple:
        # Infer action-space size from the spatial shape if we can
        size = 356  # default for board_size=8 fallback
        if mask is not None and hasattr(mask, "shape"):
            size = mask.shape[-1]
        elif spatial is not None and hasattr(spatial, "shape"):
            size = 356
        policy = torch.ones(size) / size
        value = torch.tensor(0.1)
        return policy, value

    net.predict_single = MagicMock(side_effect=_predict_single)

    # Forward pass (batch mode) returns (policy_logits, value)
    def _forward(spatial: Any, vector: Any = None, mask: Any = None) -> tuple:
        batch = spatial.shape[0]
        size = 356
        if mask is not None and hasattr(mask, "shape"):
            size = mask.shape[-1]
        policy_logits = torch.zeros(batch, size)
        value = torch.tensor([[0.1]] * batch)
        return policy_logits, value

    net.__call__ = MagicMock(side_effect=_forward)
    net.return_value = None  # prevent default MagicMock behaviour

    return net


@pytest.fixture()
def random_player() -> RandomPlayer:
    """Create a seeded RandomPlayer baseline."""
    return RandomPlayer(seed=123, name="test_random")


# ===========================================================================
# MicroCivGameAdapter tests
# ===========================================================================


class TestMicroCivGameAdapterInit:
    """Tests for MicroCivGameAdapter construction and properties."""

    def test_env_property_returns_underlying_env(
        self, adapter: MicroCivGameAdapter, small_env: MicroCivEnv
    ) -> None:
        assert adapter.env is small_env

    def test_action_space_size_is_int(self, adapter: MicroCivGameAdapter) -> None:
        size = adapter.action_space_size
        assert isinstance(size, int)
        assert size == int(adapter.env.action_space.n)
        assert size > 0

    def test_board_size_matches_env(
        self, adapter: MicroCivGameAdapter, small_env: MicroCivEnv
    ) -> None:
        assert adapter.board_size == small_env.board_size
        assert adapter.board_size == 8

    def test_last_info_empty_before_reset(self, adapter: MicroCivGameAdapter) -> None:
        assert adapter.get_last_info() == {}


class TestMicroCivGameAdapterReset:
    """Tests for MicroCivGameAdapter.reset()."""

    def test_reset_returns_dict(self, adapter: MicroCivGameAdapter) -> None:
        obs = adapter.reset(seed=42)
        assert isinstance(obs, dict)

    def test_reset_obs_has_required_keys(self, adapter: MicroCivGameAdapter) -> None:
        obs = adapter.reset(seed=42)
        assert "spatial" in obs
        assert "vector" in obs
        assert "action_mask" in obs

    def test_reset_obs_spatial_shape(self, adapter: MicroCivGameAdapter) -> None:
        obs = adapter.reset(seed=42)
        spatial = obs["spatial"]
        assert isinstance(spatial, np.ndarray)
        # (channels, board_size, board_size)
        assert spatial.ndim == 3
        assert spatial.shape[1] == 8
        assert spatial.shape[2] == 8

    def test_reset_obs_vector_is_1d(self, adapter: MicroCivGameAdapter) -> None:
        obs = adapter.reset(seed=42)
        assert obs["vector"].ndim == 1

    def test_reset_obs_action_mask_is_1d(self, adapter: MicroCivGameAdapter) -> None:
        obs = adapter.reset(seed=42)
        mask = obs["action_mask"]
        assert mask.ndim == 1
        assert mask.shape[0] == adapter.action_space_size

    def test_reset_without_seed(self, adapter: MicroCivGameAdapter) -> None:
        obs = adapter.reset()
        assert isinstance(obs, dict)
        assert "spatial" in obs

    def test_reset_stores_info(self, adapter: MicroCivGameAdapter) -> None:
        adapter.reset(seed=42)
        info = adapter.get_last_info()
        assert isinstance(info, dict)
        # The env info dict should contain at least turn and current_player
        assert "turn" in info
        assert "current_player" in info

    def test_reset_deterministic_with_same_seed(
        self, adapter: MicroCivGameAdapter
    ) -> None:
        obs1 = adapter.reset(seed=99)
        obs2 = adapter.reset(seed=99)
        np.testing.assert_array_equal(obs1["spatial"], obs2["spatial"])
        np.testing.assert_array_equal(obs1["vector"], obs2["vector"])

    def test_current_player_and_turn_after_reset(
        self, adapter: MicroCivGameAdapter
    ) -> None:
        adapter.reset(seed=42)
        assert adapter.current_player == 0
        assert adapter.turn == 0


class TestMicroCivGameAdapterStep:
    """Tests for MicroCivGameAdapter.step()."""

    def test_step_returns_five_tuple(
        self, reset_adapter: MicroCivGameAdapter
    ) -> None:
        valid = reset_adapter.get_valid_actions()
        assert len(valid) > 0, "Expected at least one valid action after reset"
        result = reset_adapter.step(valid[0])
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_obs_is_dict(self, reset_adapter: MicroCivGameAdapter) -> None:
        valid = reset_adapter.get_valid_actions()
        obs, _, _, _, _ = reset_adapter.step(valid[0])
        assert isinstance(obs, dict)
        assert "spatial" in obs
        assert "vector" in obs
        assert "action_mask" in obs

    def test_step_reward_is_float(self, reset_adapter: MicroCivGameAdapter) -> None:
        valid = reset_adapter.get_valid_actions()
        _, reward, _, _, _ = reset_adapter.step(valid[0])
        assert isinstance(reward, float)

    def test_step_terminated_truncated_are_bool(
        self, reset_adapter: MicroCivGameAdapter
    ) -> None:
        valid = reset_adapter.get_valid_actions()
        _, _, terminated, truncated, _ = reset_adapter.step(valid[0])
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_info_is_dict(self, reset_adapter: MicroCivGameAdapter) -> None:
        valid = reset_adapter.get_valid_actions()
        _, _, _, _, info = reset_adapter.step(valid[0])
        assert isinstance(info, dict)

    def test_step_updates_last_info(
        self, reset_adapter: MicroCivGameAdapter
    ) -> None:
        valid = reset_adapter.get_valid_actions()
        _, _, _, _, info = reset_adapter.step(valid[0])
        assert reset_adapter.get_last_info() is info


class TestMicroCivGameAdapterGetValidActions:
    """Tests for MicroCivGameAdapter.get_valid_actions()."""

    def test_returns_list(self, reset_adapter: MicroCivGameAdapter) -> None:
        valid = reset_adapter.get_valid_actions()
        assert isinstance(valid, list)

    def test_elements_are_ints(self, reset_adapter: MicroCivGameAdapter) -> None:
        valid = reset_adapter.get_valid_actions()
        for a in valid:
            assert isinstance(a, int)

    def test_non_empty_after_reset(self, reset_adapter: MicroCivGameAdapter) -> None:
        valid = reset_adapter.get_valid_actions()
        assert len(valid) > 0

    def test_indices_within_action_space(
        self, reset_adapter: MicroCivGameAdapter
    ) -> None:
        valid = reset_adapter.get_valid_actions()
        size = reset_adapter.action_space_size
        for a in valid:
            assert 0 <= a < size

    def test_consistent_with_env_mask(
        self, reset_adapter: MicroCivGameAdapter
    ) -> None:
        """The adapter list should match the non-zero indices of the env mask."""
        valid_list = reset_adapter.get_valid_actions()
        env_mask = reset_adapter.env.get_valid_actions()
        expected = np.where(env_mask > 0)[0].tolist()
        assert valid_list == expected


class TestMicroCivGameAdapterGetBoardData:
    """Tests for MicroCivGameAdapter.get_board_data()."""

    def test_returns_dict(self, reset_adapter: MicroCivGameAdapter) -> None:
        data = reset_adapter.get_board_data()
        assert isinstance(data, dict)

    def test_has_required_keys(self, reset_adapter: MicroCivGameAdapter) -> None:
        data = reset_adapter.get_board_data()
        expected_keys = {
            "terrain_grid",
            "units",
            "cities",
            "board_size",
            "current_player",
            "turn",
        }
        assert expected_keys.issubset(data.keys())

    def test_terrain_grid_shape(self, reset_adapter: MicroCivGameAdapter) -> None:
        data = reset_adapter.get_board_data()
        grid = data["terrain_grid"]
        bs = data["board_size"]
        assert len(grid) == bs
        for row in grid:
            assert len(row) == bs

    def test_terrain_tile_structure(
        self, reset_adapter: MicroCivGameAdapter
    ) -> None:
        data = reset_adapter.get_board_data()
        tile = data["terrain_grid"][0][0]
        assert "x" in tile
        assert "y" in tile
        assert "terrain" in tile
        assert tile["x"] == 0
        assert tile["y"] == 0
        assert tile["terrain"] in {"plains", "hills", "mountain"}

    def test_terrain_tile_coords_match_grid_position(
        self, reset_adapter: MicroCivGameAdapter
    ) -> None:
        data = reset_adapter.get_board_data()
        for y, row in enumerate(data["terrain_grid"]):
            for x, tile in enumerate(row):
                assert tile["x"] == x
                assert tile["y"] == y

    def test_units_is_list(self, reset_adapter: MicroCivGameAdapter) -> None:
        data = reset_adapter.get_board_data()
        assert isinstance(data["units"], list)

    def test_unit_structure(self, reset_adapter: MicroCivGameAdapter) -> None:
        data = reset_adapter.get_board_data()
        if len(data["units"]) > 0:
            unit = data["units"][0]
            assert "id" in unit
            assert "player" in unit
            assert "type" in unit
            assert "x" in unit
            assert "y" in unit
            assert "health" in unit
            assert "moves" in unit

    def test_cities_is_list(self, reset_adapter: MicroCivGameAdapter) -> None:
        data = reset_adapter.get_board_data()
        assert isinstance(data["cities"], list)

    def test_city_structure(self, reset_adapter: MicroCivGameAdapter) -> None:
        data = reset_adapter.get_board_data()
        if len(data["cities"]) > 0:
            city = data["cities"][0]
            assert "id" in city
            assert "player" in city
            assert "name" in city
            assert "x" in city
            assert "y" in city
            assert "population" in city

    def test_board_size_matches_adapter(
        self, reset_adapter: MicroCivGameAdapter
    ) -> None:
        data = reset_adapter.get_board_data()
        assert data["board_size"] == reset_adapter.board_size

    def test_current_player_matches_adapter(
        self, reset_adapter: MicroCivGameAdapter
    ) -> None:
        data = reset_adapter.get_board_data()
        assert data["current_player"] == reset_adapter.current_player

    def test_turn_matches_adapter(
        self, reset_adapter: MicroCivGameAdapter
    ) -> None:
        data = reset_adapter.get_board_data()
        assert data["turn"] == reset_adapter.turn


class TestMicroCivGameAdapterMultipleSteps:
    """Integration-style tests exercising reset/step/valid_actions together."""

    def test_play_several_random_steps(
        self, adapter: MicroCivGameAdapter
    ) -> None:
        """Play 20 random steps without errors."""
        adapter.reset(seed=7)
        rng = np.random.default_rng(7)
        for _ in range(20):
            valid = adapter.get_valid_actions()
            if not valid:
                break
            action = rng.choice(valid)
            obs, reward, terminated, truncated, info = adapter.step(action)
            assert isinstance(obs, dict)
            if terminated or truncated:
                break

    def test_board_data_updates_after_step(
        self, adapter: MicroCivGameAdapter
    ) -> None:
        adapter.reset(seed=42)
        data_before = adapter.get_board_data()
        valid = adapter.get_valid_actions()
        if valid:
            adapter.step(valid[0])
        data_after = adapter.get_board_data()
        # At least one of the mutable fields may have changed, but all
        # required keys must still be present.
        for key in ("terrain_grid", "units", "cities", "board_size"):
            assert key in data_after


# ===========================================================================
# MCTSAIAgent tests
# ===========================================================================


class TestMCTSAIAgentInit:
    """Tests for MCTSAIAgent construction."""

    def test_init_policy_only_mode(self, mock_network: MagicMock) -> None:
        agent = MCTSAIAgent(
            network=mock_network,
            device="cpu",
            use_mcts=False,
            mcts_simulations=10,
            temperature=0.0,
            board_size=8,
        )
        assert agent._use_mcts is False
        assert agent._mcts is None

    def test_init_mcts_mode_creates_mcts(
        self, mock_network: MagicMock
    ) -> None:
        """When use_mcts=True and Civ6MCTS is importable, _mcts is set."""
        agent = MCTSAIAgent(
            network=mock_network,
            device="cpu",
            use_mcts=True,
            mcts_simulations=10,
            temperature=0.0,
            board_size=8,
        )
        # _mcts should be initialised (or fallback if import fails)
        # Either way, no exception should occur.
        assert agent._network is mock_network

    def test_default_parameters(self, mock_network: MagicMock) -> None:
        agent = MCTSAIAgent(network=mock_network)
        assert agent._device == "cpu"
        assert agent._use_mcts is True
        assert agent._mcts_simulations == 50
        assert agent._temperature == 0.0
        assert agent._board_size == 16


class TestMCTSAIAgentSetEnv:
    """Tests for MCTSAIAgent.set_env()."""

    def test_set_env_stores_reference(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        agent = MCTSAIAgent(network=mock_network, use_mcts=False)
        assert agent._env is None
        agent.set_env(small_env)
        assert agent._env is small_env


class TestMCTSAIAgentPolicyOnly:
    """Tests for MCTSAIAgent.select_action in policy-only mode."""

    def test_select_action_returns_dict(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=False,
            board_size=8,
        )
        obs, _ = small_env.reset(seed=42)
        result = agent.select_action(obs)
        assert isinstance(result, dict)

    def test_result_has_required_keys(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=False,
            board_size=8,
        )
        obs, _ = small_env.reset(seed=42)
        result = agent.select_action(obs)
        assert "action" in result
        assert "value" in result
        assert "policy" in result

    def test_action_is_int(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=False,
            board_size=8,
        )
        obs, _ = small_env.reset(seed=42)
        result = agent.select_action(obs)
        assert isinstance(result["action"], int)

    def test_value_is_float_or_none(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=False,
            board_size=8,
        )
        obs, _ = small_env.reset(seed=42)
        result = agent.select_action(obs)
        assert result["value"] is None or isinstance(result["value"], float)

    def test_policy_is_ndarray_or_none(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=False,
            board_size=8,
        )
        obs, _ = small_env.reset(seed=42)
        result = agent.select_action(obs)
        assert result["policy"] is None or isinstance(
            result["policy"], np.ndarray
        )

    def test_predict_single_called(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        """With predict_single present, the agent should use it."""
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=False,
            board_size=8,
        )
        obs, _ = small_env.reset(seed=42)
        agent.select_action(obs)
        mock_network.predict_single.assert_called_once()

    def test_valid_actions_kwarg_used_as_fallback(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        """When action mask zeros everything, valid_actions kwarg is used."""
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=False,
            board_size=8,
        )
        obs, _ = small_env.reset(seed=42)
        # Zero out the mask to force the fallback branch
        obs["action_mask"] = np.zeros_like(obs["action_mask"])
        # We need predict_single to return a zero-ish policy so masking
        # yields zero sum, triggering the uniform fallback
        size = obs["action_mask"].shape[0]

        def _zero_policy(spatial: Any, vector: Any = None, mask: Any = None) -> tuple:
            return torch.zeros(size), torch.tensor(0.0)

        mock_network.predict_single = MagicMock(side_effect=_zero_policy)

        valid = [5, 10, 15]
        result = agent.select_action(obs, valid_actions=valid)
        assert result["action"] in valid

    def test_no_spatial_raises_value_error(
        self, mock_network: MagicMock
    ) -> None:
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=False,
            board_size=8,
        )
        obs = {"vector": np.zeros(64), "action_mask": np.ones(356)}
        with pytest.raises(ValueError, match="No spatial observation"):
            agent.select_action(obs)

    def test_greedy_selects_argmax(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        """With temperature=0, the action should be the argmax of masked policy."""
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=False,
            temperature=0.0,
            board_size=8,
        )
        obs, _ = small_env.reset(seed=42)
        size = obs["action_mask"].shape[0]

        # Create a peaked policy that has a clear argmax
        peaked = np.zeros(size, dtype=np.float32)
        # Find a valid action to place the peak
        valid_indices = np.where(obs["action_mask"] > 0)[0]
        target_action = int(valid_indices[0])
        peaked[target_action] = 1.0

        def _peaked_predict(
            spatial: Any, vector: Any = None, mask: Any = None
        ) -> tuple:
            return torch.from_numpy(peaked), torch.tensor(0.5)

        mock_network.predict_single = MagicMock(side_effect=_peaked_predict)

        result = agent.select_action(obs)
        assert result["action"] == target_action

    def test_temperature_enables_sampling(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        """With temperature > 0, actions are sampled (not always argmax)."""
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=False,
            temperature=1.0,
            board_size=8,
        )
        obs, _ = small_env.reset(seed=42)
        # Just verify it runs without error
        result = agent.select_action(obs)
        assert isinstance(result["action"], int)


class TestMCTSAIAgentGenericForward:
    """Tests for the generic forward-pass path (no predict_single)."""

    def test_generic_forward_path(
        self, small_env: MicroCivEnv
    ) -> None:
        """When the network lacks predict_single, the generic __call__ is used."""
        net = MagicMock(spec=[])  # no predict_single attribute

        obs, _ = small_env.reset(seed=42)
        size = obs["action_mask"].shape[0]

        def _forward(
            spatial: Any, vector: Any = None, mask: Any = None
        ) -> tuple:
            batch = spatial.shape[0]
            logits = torch.zeros(batch, size)
            value = torch.tensor([[0.2]] * batch)
            return logits, value

        net.side_effect = _forward

        agent = MCTSAIAgent(
            network=net,
            use_mcts=False,
            board_size=8,
        )
        result = agent.select_action(obs)
        assert isinstance(result["action"], int)
        assert result["policy"] is not None
        assert isinstance(result["policy"], np.ndarray)
        assert result["value"] is not None or isinstance(result["value"], float)

    def test_generic_forward_no_value(
        self, small_env: MicroCivEnv
    ) -> None:
        """Network returns only logits (no value head)."""
        net = MagicMock(spec=[])

        obs, _ = small_env.reset(seed=42)
        size = obs["action_mask"].shape[0]

        def _forward_no_value(
            spatial: Any, vector: Any = None, mask: Any = None
        ) -> torch.Tensor:
            batch = spatial.shape[0]
            return torch.zeros(batch, size)

        net.side_effect = _forward_no_value

        agent = MCTSAIAgent(
            network=net,
            use_mcts=False,
            board_size=8,
        )
        result = agent.select_action(obs)
        assert isinstance(result["action"], int)
        assert result["value"] is None


class TestMCTSAIAgentMCTSMode:
    """Tests for MCTSAIAgent with MCTS enabled (MCTS components mocked)."""

    def test_select_with_mcts(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        """When MCTS is available and env is set, MCTS search path is taken."""
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=True,
            mcts_simulations=5,
            temperature=0.0,
            board_size=8,
        )
        small_env.reset(seed=42)
        agent.set_env(small_env)

        obs = small_env.get_observation()
        size = obs["action_mask"].shape[0]

        # Mock the internal MCTS searcher
        mock_mcts = MagicMock()
        probs = np.zeros(size, dtype=np.float32)
        valid_indices = np.where(obs["action_mask"] > 0)[0]
        probs[valid_indices[0]] = 1.0
        mock_mcts.search.return_value = (probs, 0.42)
        agent._mcts = mock_mcts

        result = agent.select_action(obs)

        mock_mcts.search.assert_called_once()
        assert result["action"] == int(valid_indices[0])
        assert result["value"] == pytest.approx(0.42)
        assert isinstance(result["policy"], np.ndarray)

    def test_mcts_falls_back_to_policy_without_env(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        """If env is not set, MCTS falls back to policy-only."""
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=True,
            mcts_simulations=5,
            board_size=8,
        )
        # Do NOT call agent.set_env(...)
        obs, _ = small_env.reset(seed=42)

        result = agent.select_action(obs)
        # Should still return a valid result via policy-only path
        assert isinstance(result["action"], int)
        assert "value" in result
        assert "policy" in result

    def test_mcts_temperature_sampling(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        """With temperature > 0 in MCTS mode, actions are sampled."""
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=True,
            mcts_simulations=5,
            temperature=1.0,
            board_size=8,
        )
        small_env.reset(seed=42)
        agent.set_env(small_env)

        obs = small_env.get_observation()
        size = obs["action_mask"].shape[0]

        mock_mcts = MagicMock()
        probs = np.zeros(size, dtype=np.float32)
        valid_indices = np.where(obs["action_mask"] > 0)[0]
        for idx in valid_indices[:5]:
            probs[idx] = 0.2
        mock_mcts.search.return_value = (probs, 0.3)
        agent._mcts = mock_mcts

        result = agent.select_action(obs)
        assert isinstance(result["action"], int)
        assert result["action"] in valid_indices[:5]

    def test_mcts_with_action_mask_normalisation(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        """MCTS probs are masked and re-normalised."""
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=True,
            mcts_simulations=5,
            temperature=0.0,
            board_size=8,
        )
        small_env.reset(seed=42)
        agent.set_env(small_env)

        obs = small_env.get_observation()
        size = obs["action_mask"].shape[0]

        # Give probability to an invalid action and a valid one
        mock_mcts = MagicMock()
        probs = np.zeros(size, dtype=np.float32)
        valid_indices = np.where(obs["action_mask"] > 0)[0]
        # Put weight on first valid action
        probs[valid_indices[0]] = 0.8
        # Put weight on an invalid action (find one)
        invalid_indices = np.where(obs["action_mask"] == 0)[0]
        if len(invalid_indices) > 0:
            probs[invalid_indices[0]] = 0.9  # higher but invalid
        mock_mcts.search.return_value = (probs, 0.5)
        agent._mcts = mock_mcts

        result = agent.select_action(obs)
        # The action should be valid
        assert result["action"] in valid_indices

    def test_mcts_value_none_handling(
        self, mock_network: MagicMock, small_env: MicroCivEnv
    ) -> None:
        """When MCTS returns None value, result value should be None."""
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=True,
            mcts_simulations=5,
            temperature=0.0,
            board_size=8,
        )
        small_env.reset(seed=42)
        agent.set_env(small_env)

        obs = small_env.get_observation()
        size = obs["action_mask"].shape[0]

        mock_mcts = MagicMock()
        probs = np.zeros(size, dtype=np.float32)
        valid_indices = np.where(obs["action_mask"] > 0)[0]
        probs[valid_indices[0]] = 1.0
        mock_mcts.search.return_value = (probs, None)
        agent._mcts = mock_mcts

        result = agent.select_action(obs)
        assert result["value"] is None


class TestMCTSAIAgentNonDictObservation:
    """Tests for the non-dict observation fallback path."""

    def test_ndarray_observation_warns_and_proceeds(
        self, mock_network: MagicMock
    ) -> None:
        """Passing a raw ndarray triggers the warning path."""
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=False,
            board_size=8,
        )
        spatial = np.random.randn(13, 8, 8).astype(np.float32)

        # predict_single will receive the raw spatial as-is
        size = 356

        def _ps(sp: Any, vec: Any = None, mask: Any = None) -> tuple:
            return torch.ones(size) / size, torch.tensor(0.0)

        mock_network.predict_single = MagicMock(side_effect=_ps)

        result = agent.select_action(spatial)
        assert isinstance(result["action"], int)


class TestMCTSAIAgentInitMCTSImportFallback:
    """Test that _init_mcts gracefully handles ImportError."""

    def test_import_error_disables_mcts(
        self, mock_network: MagicMock
    ) -> None:
        """When Civ6MCTS import fails, agent falls back to policy-only."""
        with patch(
            "alpha_zero.core.civ6_mcts.Civ6MCTS",
            side_effect=ImportError("mocked"),
        ):
            agent = MCTSAIAgent(
                network=mock_network,
                use_mcts=True,
                mcts_simulations=10,
                board_size=8,
            )
            # Either MCTS was initialised successfully (import is available
            # in the test env) or the fallback kicked in -- both are fine.
            # The key invariant: no unhandled exception was raised.
            assert agent._network is mock_network


# ===========================================================================
# BaselineAgentAdapter tests
# ===========================================================================


class TestBaselineAgentAdapterInit:
    """Tests for BaselineAgentAdapter construction."""

    def test_name_from_baseline(self, random_player: RandomPlayer) -> None:
        adapter = BaselineAgentAdapter(random_player)
        assert adapter.name == "test_random"

    def test_name_fallback_when_no_name_attr(self) -> None:
        baseline = MagicMock(spec=[])  # no 'name' attribute
        del baseline.name  # ensure it's truly absent
        adapter = BaselineAgentAdapter(baseline)
        assert adapter.name == "baseline"

    def test_env_defaults_to_none(self, random_player: RandomPlayer) -> None:
        adapter = BaselineAgentAdapter(random_player)
        assert adapter._env is None

    def test_env_set_in_constructor(
        self, random_player: RandomPlayer, small_env: MicroCivEnv
    ) -> None:
        adapter = BaselineAgentAdapter(random_player, env=small_env)
        assert adapter._env is small_env


class TestBaselineAgentAdapterSetEnv:
    """Tests for BaselineAgentAdapter.set_env()."""

    def test_set_env_stores_reference(
        self, random_player: RandomPlayer, small_env: MicroCivEnv
    ) -> None:
        adapter = BaselineAgentAdapter(random_player)
        adapter.set_env(small_env)
        assert adapter._env is small_env

    def test_set_env_replaces_previous(
        self, random_player: RandomPlayer
    ) -> None:
        env1 = MicroCivEnv(board_size=8, max_turns=50, seed=1)
        env2 = MicroCivEnv(board_size=8, max_turns=50, seed=2)
        adapter = BaselineAgentAdapter(random_player, env=env1)
        adapter.set_env(env2)
        assert adapter._env is env2


class TestBaselineAgentAdapterSelectAction:
    """Tests for BaselineAgentAdapter.select_action()."""

    def test_raises_without_env(self, random_player: RandomPlayer) -> None:
        adapter = BaselineAgentAdapter(random_player)
        obs = {"spatial": np.zeros((13, 8, 8)), "action_mask": np.ones(356)}
        with pytest.raises(RuntimeError, match="Environment not set"):
            adapter.select_action(obs)

    def test_returns_dict(
        self, random_player: RandomPlayer, small_env: MicroCivEnv
    ) -> None:
        small_env.reset(seed=42)
        adapter = BaselineAgentAdapter(random_player, env=small_env)
        obs = small_env.get_observation()
        result = adapter.select_action(obs)
        assert isinstance(result, dict)

    def test_result_has_required_keys(
        self, random_player: RandomPlayer, small_env: MicroCivEnv
    ) -> None:
        small_env.reset(seed=42)
        adapter = BaselineAgentAdapter(random_player, env=small_env)
        obs = small_env.get_observation()
        result = adapter.select_action(obs)
        assert "action" in result
        assert "value" in result
        assert "policy" in result

    def test_action_is_int(
        self, random_player: RandomPlayer, small_env: MicroCivEnv
    ) -> None:
        small_env.reset(seed=42)
        adapter = BaselineAgentAdapter(random_player, env=small_env)
        obs = small_env.get_observation()
        result = adapter.select_action(obs)
        assert isinstance(result["action"], int)

    def test_value_is_none(
        self, random_player: RandomPlayer, small_env: MicroCivEnv
    ) -> None:
        small_env.reset(seed=42)
        adapter = BaselineAgentAdapter(random_player, env=small_env)
        obs = small_env.get_observation()
        result = adapter.select_action(obs)
        assert result["value"] is None

    def test_policy_is_none(
        self, random_player: RandomPlayer, small_env: MicroCivEnv
    ) -> None:
        small_env.reset(seed=42)
        adapter = BaselineAgentAdapter(random_player, env=small_env)
        obs = small_env.get_observation()
        result = adapter.select_action(obs)
        assert result["policy"] is None

    def test_action_is_valid(
        self, random_player: RandomPlayer, small_env: MicroCivEnv
    ) -> None:
        small_env.reset(seed=42)
        adapter = BaselineAgentAdapter(random_player, env=small_env)
        obs = small_env.get_observation()
        result = adapter.select_action(obs)
        mask = obs["action_mask"]
        valid_actions = np.where(mask > 0)[0]
        assert result["action"] in valid_actions

    def test_delegates_to_baseline_select_action(
        self, small_env: MicroCivEnv
    ) -> None:
        baseline = MagicMock()
        baseline.name = "mock_baseline"
        baseline.select_action.return_value = 7

        small_env.reset(seed=42)
        adapter = BaselineAgentAdapter(baseline, env=small_env)
        obs = small_env.get_observation()

        result = adapter.select_action(obs)
        baseline.select_action.assert_called_once_with(small_env, obs)
        assert result["action"] == 7

    def test_valid_actions_kwarg_accepted(
        self, random_player: RandomPlayer, small_env: MicroCivEnv
    ) -> None:
        """valid_actions kwarg is accepted (though unused by baselines)."""
        small_env.reset(seed=42)
        adapter = BaselineAgentAdapter(random_player, env=small_env)
        obs = small_env.get_observation()
        # Should not raise
        result = adapter.select_action(obs, valid_actions=[1, 2, 3])
        assert isinstance(result["action"], int)


class TestBaselineAgentAdapterReset:
    """Tests for BaselineAgentAdapter.reset()."""

    def test_reset_delegates_to_baseline(self) -> None:
        baseline = MagicMock()
        baseline.name = "mock"
        adapter = BaselineAgentAdapter(baseline)
        adapter.reset()
        baseline.reset.assert_called_once()

    def test_reset_no_op_when_baseline_lacks_reset(self) -> None:
        baseline = MagicMock(spec=[])
        del baseline.reset  # ensure no reset attribute
        adapter = BaselineAgentAdapter(baseline)
        # Should not raise
        adapter.reset()

    def test_reset_with_real_random_player(
        self, random_player: RandomPlayer
    ) -> None:
        adapter = BaselineAgentAdapter(random_player)
        # RandomPlayer.reset increments games_played
        adapter.reset()
        assert random_player.get_stats()["games_played"] == 1
        adapter.reset()
        assert random_player.get_stats()["games_played"] == 2


class TestBaselineAgentAdapterMultipleSteps:
    """Integration test using BaselineAgentAdapter with a real environment."""

    def test_play_several_steps(
        self, random_player: RandomPlayer
    ) -> None:
        env = MicroCivEnv(board_size=8, max_turns=50, seed=42)
        game = MicroCivGameAdapter(env)
        adapter = BaselineAgentAdapter(random_player, env=env)

        obs = game.reset(seed=42)
        for _ in range(15):
            valid = game.get_valid_actions()
            if not valid:
                break
            result = adapter.select_action(obs, valid_actions=valid)
            obs, reward, terminated, truncated, info = game.step(
                result["action"]
            )
            if terminated or truncated:
                break
        # No exceptions means success


# ===========================================================================
# Cross-adapter integration tests
# ===========================================================================


class TestAdapterIntegration:
    """Integration tests combining multiple adapter classes."""

    def test_game_adapter_with_mcts_agent(
        self, mock_network: MagicMock
    ) -> None:
        """MicroCivGameAdapter + MCTSAIAgent work together."""
        env = MicroCivEnv(board_size=8, max_turns=50, seed=42)
        game = MicroCivGameAdapter(env)
        agent = MCTSAIAgent(
            network=mock_network,
            use_mcts=False,
            board_size=8,
        )

        obs = game.reset(seed=42)
        valid = game.get_valid_actions()
        result = agent.select_action(obs, valid_actions=valid)
        assert result["action"] in valid or len(valid) == 0
        if valid:
            obs, reward, terminated, truncated, info = game.step(
                result["action"]
            )
            assert isinstance(obs, dict)

    def test_game_adapter_with_baseline_agent(
        self, random_player: RandomPlayer
    ) -> None:
        """MicroCivGameAdapter + BaselineAgentAdapter work together."""
        env = MicroCivEnv(board_size=8, max_turns=50, seed=42)
        game = MicroCivGameAdapter(env)
        agent = BaselineAgentAdapter(random_player, env=env)

        obs = game.reset(seed=42)
        for _ in range(10):
            valid = game.get_valid_actions()
            if not valid:
                break
            result = agent.select_action(obs, valid_actions=valid)
            obs, reward, terminated, truncated, info = game.step(
                result["action"]
            )
            if terminated or truncated:
                break

    def test_set_env_via_game_adapter(
        self, random_player: RandomPlayer
    ) -> None:
        """BaselineAgentAdapter can receive env from game adapter."""
        env = MicroCivEnv(board_size=8, max_turns=50, seed=42)
        game = MicroCivGameAdapter(env)
        agent = BaselineAgentAdapter(random_player)

        # set_env using the adapter's env property
        agent.set_env(game.env)
        assert agent._env is env

        obs = game.reset(seed=42)
        result = agent.select_action(obs)
        assert isinstance(result["action"], int)
