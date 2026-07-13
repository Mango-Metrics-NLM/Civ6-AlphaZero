"""Unit tests for the Bridge MCTS policy adapter."""

from unittest.mock import MagicMock

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
            (0, 0): {
                "type": "grassland",
                "features": ["forest"],
                "yields": {"food": 2, "production": 1},
            },
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


def test_high_player_ids_are_clamped(mock_converted_state: dict) -> None:
    """Barbarians / city-states with player ids >= num_players must be clamped
    to num_players-1 so MicroCivEnv per-player indexing can't IndexError."""
    state = dict(mock_converted_state)
    state["units"] = [
        {"_id": 1, "player": 0, "type": "warrior", "position": (0, 0), "health": 100},
        {"_id": 2, "player": 63, "type": "warrior", "position": (2, 2), "health": 100},  # barbarian
    ]
    state["cities"] = [
        {"_id": 10, "player": 1, "name": "A", "position": (1, 1), "population": 1},
        {"_id": 11, "player": 99, "name": "CityState", "position": (3, 3), "population": 1},
    ]
    env = BridgeMCTSEnv(state)  # must not raise
    max_idx = env.env.num_players - 1
    assert all(0 <= u.player <= max_idx for u in env.env.units)
    assert all(0 <= c.player <= max_idx for c in env.env.cities)
    # The clamped entities land on the last valid slot.
    assert {u.id: u.player for u in env.env.units}[2] == max_idx
    assert {c.id: c.player for c in env.env.cities}[11] == max_idx


def test_clamp_player_helper() -> None:
    """The clamp helper is pure and bounds into [0, max]."""
    assert BridgeMCTSEnv._clamp_player(3, 7) == 3
    assert BridgeMCTSEnv._clamp_player(63, 7) == 7
    assert BridgeMCTSEnv._clamp_player(-1, 7) == 0


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

    # Mock the internal MCTS search to avoid actual network/tree expansion.
    # GumbelMCTS.search returns (policy_probs, root_value) — mirror the real
    # contract so this test catches unpack drift in select_action.
    mock_policy_probs = np.zeros(64, dtype=np.float32)
    mock_policy_probs[42] = 1.0
    policy.mcts.search = MagicMock(return_value=(mock_policy_probs, 0.75))  # type: ignore

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
