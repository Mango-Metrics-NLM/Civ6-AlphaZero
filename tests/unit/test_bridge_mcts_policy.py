"""Unit tests for the Bridge MCTS policy adapter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv, Civ6MctsPolicy
from alpha_zero.core.civ6_mcts import GumbelMCTSConfig


@pytest.fixture
def mock_converted_state() -> dict:
    return {
        "turn": 10,
        "current_player": 1,
        "_map_size": (8, 8),
        "terrain": {
            (0, 0): {"type": "grassland", "features": ["forest"], "yields": {"food": 2, "production": 1}},
            (1, 1): {"type": "plains", "features": [], "yields": {"food": 1, "production": 1}},
        },
        "units": [
            {
                "_id": 100,
                "player": 0,
                "type": "warrior",
                "position": (0, 0),
                "health": 100,
                "strength": 20,
                "moves_remaining": 2,
            }
        ],
        "cities": [
            {
                "_id": 200,
                "player": 1,
                "name": "Rome",
                "position": (1, 1),
                "population": 2,
                "defense": 15,
                "production": 5,
            }
        ],
    }


def test_bridge_mcts_env_init(mock_converted_state: dict) -> None:
    """Test initialization of BridgeMCTSEnv from converted state."""
    env = BridgeMCTSEnv(mock_converted_state, max_turns=100)

    assert env.turn == 10
    assert env.current_player == 1

    # Check underlying MicroCivEnv
    assert env.env.board_size == 8
    assert env.env.max_turns == 100
    assert env.env.num_players == 2

    assert len(env.env.units) == 1
    unit = env.env.units[0]
    assert unit.id == 100
    assert unit.player == 0
    assert unit.unit_type == "warrior"
    assert unit.position == (0, 0)
    assert unit.health == 100

    assert len(env.env.cities) == 1
    city = env.env.cities[0]
    assert city.id == 200
    assert city.player == 1
    assert city.name == "Rome"
    assert city.position == (1, 1)

    # Check terrain
    assert (0, 0) in env.env.terrain
    assert env.env.terrain[(0, 0)].terrain_type == "grassland"
    assert env.env.terrain[(0, 0)].features == ["forest"]

    # IDs should be advanced beyond max present
    assert env.env._next_unit_id == 101
    assert env.env._next_city_id == 201


def test_bridge_mcts_env_clone(mock_converted_state: dict) -> None:
    """Test that clone creates a deep copy of the environment."""
    env = BridgeMCTSEnv(mock_converted_state)
    cloned = env.clone()

    assert cloned is not env
    assert cloned.env is not env.env
    assert cloned.turn == env.turn
    assert cloned.current_player == env.current_player

    # Modify original to verify deep copy
    env.env.turn = 99
    env.env.units[0].health = 50
    assert cloned.turn == 10
    assert cloned.env.units[0].health == 100


def test_bridge_mcts_env_reset_raises(mock_converted_state: dict) -> None:
    """Test that reset is not supported."""
    env = BridgeMCTSEnv(mock_converted_state)
    with pytest.raises(NotImplementedError, match="does not support reset"):
        env.reset()


def test_bridge_mcts_env_delegates(mock_converted_state: dict) -> None:
    """Test that environment methods delegate to MicroCivEnv."""
    env = BridgeMCTSEnv(mock_converted_state)

    obs = env.get_observation()
    assert "spatial" in obs
    assert "vector" in obs
    assert "action_mask" in obs

    info = env.get_info()
    assert info["turn"] == 10

    mask = env.get_valid_actions()
    assert isinstance(mask, np.ndarray)


@pytest.mark.asyncio
async def test_civ6_mcts_policy_requires_converted_state() -> None:
    """Test that select_action requires converted_state."""
    policy = Civ6MctsPolicy(network=MagicMock(), config=GumbelMCTSConfig())
    obs = {"spatial": np.zeros((13, 8, 8)), "vector": np.zeros(64)}
    mask = np.ones(10)

    with pytest.raises(ValueError, match="requires converted_state"):
        await policy.select_action(obs, mask, converted_state=None)


@pytest.mark.asyncio
async def test_civ6_mcts_policy_select_action(mock_converted_state: dict) -> None:
    """Test successful MCTS action selection."""
    config = GumbelMCTSConfig(num_simulations=2)
    policy = Civ6MctsPolicy(network=MagicMock(), config=config)

    # Mock the internal MCTS search to avoid actual network/tree expansion
    mock_search_stats = {"root_value": 0.75}
    policy.mcts.search = MagicMock(return_value=(42, mock_search_stats))  # type: ignore

    obs = {"spatial": np.zeros((13, 8, 8)), "vector": np.zeros(64)}
    mask = np.ones(10)

    action_idx, diagnostics = await policy.select_action(
        obs, mask, converted_state=mock_converted_state
    )

    assert action_idx == 42
    assert diagnostics["value_estimate"] == 0.75
    assert diagnostics["policy_kind"] == 1.0
    assert diagnostics["mcts_sims"] == 2.0

    # Verify search was called with a BridgeMCTSEnv
    policy.mcts.search.assert_called_once()
    args, _ = policy.mcts.search.call_args
    passed_env = args[0]
    assert isinstance(passed_env, BridgeMCTSEnv)
    assert passed_env.turn == 10
