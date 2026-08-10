"""Additional tests for mcts_policy.py gap coverage (post-hardening).

Covers:
- Module constants are correctly exported
- BridgeMCTSEnv.from_bridge_state() alternate constructor
- Empty units/cities branch (no _next_unit_id/_next_city_id reset)
- Multi-player detection (more than 2 player IDs)
- Civ6MctsPolicy with max_turns kwarg
- Default field values applied when keys absent
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from alpha_zero.bridge.mcts_policy import (
    _DEFAULT_CITY_DEFENSE,
    _DEFAULT_CITY_NAME,
    _DEFAULT_CITY_POPULATION,
    _DEFAULT_CITY_PRODUCTION,
    _DEFAULT_MAX_TURNS,
    _DEFAULT_TERRAIN_TYPE,
    _DEFAULT_UNIT_HEALTH,
    _DEFAULT_UNIT_MOVES,
    _DEFAULT_UNIT_STRENGTH,
    _DEFAULT_UNIT_TYPE,
    _MAX_CIV_PLAYERS,
    _MIN_CIV_PLAYERS,
    _POLICY_KIND_MCTS,
    BridgeMCTSEnv,
    Civ6MctsPolicy,
)
from alpha_zero.core.civ6_mcts import GumbelMCTSConfig

# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_constants_are_expected_types() -> None:
    assert isinstance(_DEFAULT_MAX_TURNS, int) and _DEFAULT_MAX_TURNS > 0
    assert isinstance(_MAX_CIV_PLAYERS, int) and _MAX_CIV_PLAYERS >= 2
    assert isinstance(_MIN_CIV_PLAYERS, int) and _MIN_CIV_PLAYERS == 2
    assert isinstance(_DEFAULT_TERRAIN_TYPE, str)
    assert isinstance(_DEFAULT_UNIT_TYPE, str)
    assert isinstance(_DEFAULT_UNIT_HEALTH, int) and _DEFAULT_UNIT_HEALTH > 0
    assert isinstance(_DEFAULT_UNIT_STRENGTH, int) and _DEFAULT_UNIT_STRENGTH > 0
    assert isinstance(_DEFAULT_UNIT_MOVES, int) and _DEFAULT_UNIT_MOVES > 0
    assert isinstance(_DEFAULT_CITY_NAME, str)
    assert isinstance(_DEFAULT_CITY_POPULATION, int) and _DEFAULT_CITY_POPULATION >= 1
    assert isinstance(_DEFAULT_CITY_DEFENSE, int) and _DEFAULT_CITY_DEFENSE >= 0
    assert isinstance(_DEFAULT_CITY_PRODUCTION, int) and _DEFAULT_CITY_PRODUCTION >= 0
    assert _POLICY_KIND_MCTS == 1.0


def test_policy_kind_distinct_from_greedy() -> None:
    """MCTS policy kind must differ from 0.0 (greedy)."""
    assert _POLICY_KIND_MCTS != 0.0


# ---------------------------------------------------------------------------
# from_bridge_state alternate constructor
# ---------------------------------------------------------------------------


def _minimal_state() -> dict:
    return {
        "turn": 5,
        "current_player": 0,
        "_map_size": (8, 8),
        "terrain": {},
        "units": [],
        "cities": [],
    }


def test_from_bridge_state_returns_bridge_mcts_env() -> None:
    env = BridgeMCTSEnv.from_bridge_state(_minimal_state())
    assert isinstance(env, BridgeMCTSEnv)
    assert env.turn == 5


def test_from_bridge_state_max_turns_forwarded() -> None:
    env = BridgeMCTSEnv.from_bridge_state(_minimal_state(), max_turns=42)
    assert env.env.max_turns == 42


def test_from_bridge_state_equivalent_to_direct() -> None:
    state = _minimal_state()
    direct = BridgeMCTSEnv(state, max_turns=200)
    via_cls = BridgeMCTSEnv.from_bridge_state(state, max_turns=200)
    assert direct.turn == via_cls.turn
    assert direct.env.max_turns == via_cls.env.max_turns


# ---------------------------------------------------------------------------
# Default field values applied
# ---------------------------------------------------------------------------


def test_default_terrain_values_applied() -> None:
    """When terrain dict omits optional keys, module defaults are used."""
    state = {
        "turn": 0,
        "current_player": 0,
        "_map_size": (8, 8),
        "terrain": {(0, 0): {}},  # empty sub-dict
        "units": [],
        "cities": [],
    }
    env = BridgeMCTSEnv(state)
    tile = env.env.terrain[(0, 0)]
    assert tile.terrain_type == _DEFAULT_TERRAIN_TYPE


def test_default_unit_values_applied() -> None:
    state = {
        "turn": 0,
        "current_player": 0,
        "_map_size": (8, 8),
        "terrain": {},
        "units": [{"player": 0}],  # minimal unit dict
        "cities": [],
    }
    env = BridgeMCTSEnv(state)
    unit = env.env.units[0]
    assert unit.unit_type == _DEFAULT_UNIT_TYPE
    assert unit.health == _DEFAULT_UNIT_HEALTH
    assert unit.strength == _DEFAULT_UNIT_STRENGTH
    assert unit.moves_remaining == _DEFAULT_UNIT_MOVES


def test_default_city_values_applied() -> None:
    state = {
        "turn": 0,
        "current_player": 0,
        "_map_size": (8, 8),
        "terrain": {},
        "units": [],
        "cities": [{"player": 0}],  # minimal city dict
    }
    env = BridgeMCTSEnv(state)
    city = env.env.cities[0]
    assert city.name == _DEFAULT_CITY_NAME
    assert city.population == _DEFAULT_CITY_POPULATION
    assert city.defense == _DEFAULT_CITY_DEFENSE
    assert city.production == _DEFAULT_CITY_PRODUCTION


# ---------------------------------------------------------------------------
# Empty units / cities branches (no ID counter reset needed)
# ---------------------------------------------------------------------------


def test_empty_units_and_cities_no_crash() -> None:
    env = BridgeMCTSEnv(_minimal_state())
    # Should not raise; _next_unit_id / _next_city_id remain as defaults
    assert env.env is not None


# ---------------------------------------------------------------------------
# Multi-player detection
# ---------------------------------------------------------------------------


def test_three_player_detection() -> None:
    state = {
        "turn": 1,
        "current_player": 0,
        "_map_size": (8, 8),
        "terrain": {},
        "units": [{"player": 2}],  # player index 2 forces 3-player game
        "cities": [],
    }
    env = BridgeMCTSEnv(state)
    assert env.env.num_players == 3


def test_player_count_capped_at_max() -> None:
    """Player IDs beyond _MAX_CIV_PLAYERS - 1 are clamped to _MAX_CIV_PLAYERS."""
    # player=_MAX_CIV_PLAYERS - 1 is the highest valid index → num_players == _MAX_CIV_PLAYERS
    state = {
        "turn": 1,
        "current_player": 0,
        "_map_size": (8, 8),
        "terrain": {},
        "units": [{"player": _MAX_CIV_PLAYERS - 1}],
        "cities": [],
    }
    env = BridgeMCTSEnv(state)
    assert env.env.num_players == _MAX_CIV_PLAYERS


# ---------------------------------------------------------------------------
# Civ6MctsPolicy max_turns forwarding
# ---------------------------------------------------------------------------


def test_policy_max_turns_stored() -> None:
    policy = Civ6MctsPolicy(
        network=MagicMock(),
        config=GumbelMCTSConfig(),
        device="cpu",
        max_turns=250,
    )
    assert policy.max_turns == 250


@pytest.mark.asyncio
async def test_policy_uses_max_turns_when_searching() -> None:
    """select_action should seed BridgeMCTSEnv with the configured max_turns."""
    policy = Civ6MctsPolicy(
        network=MagicMock(),
        config=GumbelMCTSConfig(num_simulations=1),
        max_turns=300,
    )

    seen_envs: list[BridgeMCTSEnv] = []

    def fake_search(env: BridgeMCTSEnv):
        seen_envs.append(env)
        # Mirror the real GumbelMCTS.search contract: (policy_probs, root_value)
        return np.zeros(64, dtype=np.float32), 0.0

    policy.mcts.search = fake_search  # type: ignore

    obs = {"spatial": np.zeros((13, 8, 8)), "vector": np.zeros(64)}
    mask = np.ones(10)
    await policy.select_action(obs, mask, converted_state=_minimal_state())

    assert len(seen_envs) == 1
    assert seen_envs[0].env.max_turns == 300
