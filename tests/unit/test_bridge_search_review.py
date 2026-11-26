"""What the independent review of GM-B6a found, each pinned.

* a city is busy whatever the type of its current production (the mod passes
  on what the game's build queue returns);
* a busy city's placeholder build starts from nothing;
* an archer's attack is translated from the game's ``ranged_attack`` offer;
* on a map at least as large as the board an attack is "into the cell" by the
  converter's own scaling, also at the clamped east edge;
* ``select_command`` needs a converted state, and says so.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv, Civ6MctsPolicy
from alpha_zero.bridge.protocol import GameStateData
from alpha_zero.bridge.search_translator import SearchTranslator
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.core.civ6_mcts import GumbelMCTSConfig

BOARD = 8


def live_state(
    *,
    width: int = 84,
    height: int = 54,
    unit: tuple[int, int] = (40, 27),
    offers: list[dict[str, Any]] | None = None,
    current_production: Any = None,
) -> GameStateData:
    return GameStateData.from_dict(
        {
            "turn_id": 5,
            "player_id": 0,
            "map": {"width": width, "height": height},
            "units": [
                {
                    "id": 7,
                    "owner": 0,
                    "x": unit[0],
                    "y": unit[1],
                    "unit_type": "UNIT_ARCHER",
                    "moves_remaining": 2,
                }
            ],
            "cities": [
                {"id": 301, "owner": 0, "x": 5, "y": 5, "current_production": current_production}
            ],
            "players": [{"id": 0}],
            "legal_actions": [{"action": "end_turn"}, *(offers or [])],
        }
    )


def converted(state: GameStateData) -> dict[str, Any]:
    return StateConverter(board_size=BOARD).convert(state)


@pytest.mark.parametrize("current", ["UNIT_WARRIOR", 123456789, 0, True])
def test_a_city_building_anything_is_busy_whatever_the_type(current: Any) -> None:
    env = BridgeMCTSEnv(converted(live_state(current_production=current)), max_turns=30)
    assert env.proxy_stats["busy_cities"] == 1
    assert env.env.cities[0].production_queue is not None


@pytest.mark.parametrize("current", [None, ""])
def test_a_city_building_nothing_is_idle(current: Any) -> None:
    env = BridgeMCTSEnv(converted(live_state(current_production=current)), max_turns=30)
    assert env.proxy_stats["busy_cities"] == 0
    assert env.env.cities[0].production_queue is None


def test_a_busy_city_s_placeholder_build_starts_from_nothing() -> None:
    state = live_state(current_production="UNIT_WARRIOR")
    state.cities[0].production_progress = 29  # the mod fills this with turns left
    env = BridgeMCTSEnv(converted(state), max_turns=30)
    assert env.env.cities[0].production == 0


def _translator(state: GameStateData) -> SearchTranslator:
    conv = converted(state)
    return SearchTranslator(conv, [7], [301])


def test_an_archer_s_attack_uses_the_ranged_offer() -> None:
    # Own archer at (47, 27) is in board cell (4, 4); (48, 27) is in cell (5, 4).
    state = live_state(
        unit=(47, 27),
        offers=[{"action": "ranged_attack", "unit_id": 7, "targets": [[48, 27]]}],
    )
    result = _translator(state).translate({"type": "attack", "unit_idx": 0, "direction": (1, 0)})
    assert result.command == {
        "action_type": "unit_ranged_attack",
        "unit_id": 7,
        "target": {"x": 48, "y": 27},
    }


def test_a_melee_offer_is_preferred_to_a_ranged_one() -> None:
    state = live_state(
        unit=(47, 27),
        offers=[
            {"action": "ranged_attack", "unit_id": 7, "targets": [[48, 27]]},
            {"action": "attack", "unit_id": 7, "targets": [[48, 27]]},
        ],
    )
    result = _translator(state).translate({"type": "attack", "unit_idx": 0, "direction": (1, 0)})
    assert result.command is not None and result.command["action_type"] == "unit_attack"


def test_a_move_never_uses_a_ranged_offer() -> None:
    state = live_state(
        unit=(47, 27),
        offers=[{"action": "ranged_attack", "unit_id": 7, "targets": [[48, 27]]}],
    )
    result = _translator(state).translate({"type": "move", "unit_idx": 0, "direction": (1, 0)})
    assert result.command is None


def test_an_attack_at_the_clamped_east_edge_is_into_the_last_cell() -> None:
    """Board 8 on a map 84 wide: x 79..83 all scale to cell 7 (round, then clamp)."""
    state = live_state(
        unit=(68, 27),  # round(68 * 8 / 84) = 6: the cell west of the last one
        offers=[{"action": "attack", "unit_id": 7, "targets": [[83, 27]]}],
    )
    conv = converted(state)
    cell = conv["units"][0]["position"]
    target_cell = (min(BOARD - 1, round(83 * BOARD / 84)), round(27 * BOARD / 54))
    direction = (target_cell[0] - cell[0], target_cell[1] - cell[1])
    if max(abs(direction[0]), abs(direction[1])) != 1:
        pytest.fail(f"fixture: unit cell {cell} is not next to the target's cell {target_cell}")
    result = SearchTranslator(conv, [7], [301]).translate(
        {"type": "attack", "unit_idx": 0, "direction": direction}
    )
    assert result.command is not None, result.reason
    assert result.command["target"] == {"x": 83, "y": 27}


def test_select_command_without_a_state_says_what_it_needs() -> None:
    policy = Civ6MctsPolicy(
        object(), GumbelMCTSConfig(num_simulations=8, num_sampled_actions=4), max_turns=30
    )
    with pytest.raises(ValueError, match="converted_state"):
        asyncio.run(policy.select_command(None))  # type: ignore[arg-type]
