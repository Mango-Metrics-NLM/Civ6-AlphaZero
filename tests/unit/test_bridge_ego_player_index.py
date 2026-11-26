"""The acting player's index in a converted state (GM-B5, Part C).

``StateConverter.convert()`` puts the AI's own units at player index 0 in two-player mode,
whatever the AI's game player id is. ``convert()["current_player"]`` is what the encoder,
the action mask and the simulator read to find "the acting player"; it used to be the raw game
id, so an AI playing as game player 1 had its own units read as the enemy's.

These tests build two mirrored two-player states (the same units and tiles, the AI as game
player 0 in one and as game player 1 in the other) and hold every consumer to giving the same
answer for both. Everything is the real converter, encoder, action converter and simulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from alpha_zero.bridge.action_converter import ActionConverter
from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv
from alpha_zero.bridge.protocol import LegalAction, PlayerData
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.core.civ6_actions import ActionSpaceConfig
from tests.fixtures.bridge_search_states import city, make_state, unit

BOARD = 16
#: Own units, by id and square; the enemy's is in a corner so that it has fewer moves.
OWN_WARRIOR_A = (2, 5, 5)
OWN_WARRIOR_B = (3, 8, 8)
ENEMY_WARRIOR = (9, 0, 0)
OWN_CITY = (20, 2, 2)
ENEMY_CITY = (21, 12, 12)


def _mirrored(ai_id: int, enemy_id: int) -> Any:
    """The same position with the AI as ``ai_id`` and the other side as ``enemy_id``."""
    own = {
        ai_id: {"gold": 111.0, "score": 10},
        enemy_id: {"gold": 222.0, "score": 20},
    }
    players = [PlayerData(id=pid, civilization=f"CIV_{pid}", **own[pid]) for pid in sorted(own)]
    return make_state(
        player_id=ai_id,
        size=BOARD,
        units=[
            unit(OWN_WARRIOR_A[0], "UNIT_WARRIOR", ai_id, *OWN_WARRIOR_A[1:]),
            unit(OWN_WARRIOR_B[0], "UNIT_WARRIOR", ai_id, *OWN_WARRIOR_B[1:]),
            unit(ENEMY_WARRIOR[0], "UNIT_WARRIOR", enemy_id, *ENEMY_WARRIOR[1:]),
        ],
        cities=[
            city(OWN_CITY[0], ai_id, *OWN_CITY[1:], name="Own"),
            city(ENEMY_CITY[0], enemy_id, *ENEMY_CITY[1:], name="Other"),
        ],
        legal_actions=[
            LegalAction(action_type="end_turn", unit_id=0, city_id=0),
            LegalAction(action_type="move", unit_id=OWN_WARRIOR_A[0], targets=[(5, 6)]),
            LegalAction(action_type="move", unit_id=OWN_WARRIOR_B[0], targets=[(8, 9)]),
            LegalAction(action_type="skip", unit_id=OWN_WARRIOR_A[0]),
        ],
        players=players,
    )


@pytest.fixture(scope="module")
def pair() -> dict[str, Any]:
    """Both mirrored states, converted by one converter each (so no shared caches)."""
    out: dict[str, Any] = {}
    for key, (ai_id, enemy_id) in {"p0": (0, 1), "p1": (1, 0)}.items():
        state = _mirrored(ai_id, enemy_id)
        converter = StateConverter(board_size=BOARD, research_registry=False)
        out[key] = {
            "state": state,
            "converter": converter,
            "converted": converter.convert(state),
        }
    return out


def _action_converter() -> ActionConverter:
    return ActionConverter(
        board_size=BOARD,
        action_space_config=ActionSpaceConfig(
            board_size=BOARD, max_units=20, max_cities=5, num_production_items=20
        ),
        research_registry=False,
    )


class TestCurrentPlayerIndex:
    def test_two_player_current_player_is_zero_for_both(self, pair: dict[str, Any]) -> None:
        assert pair["p0"]["converted"]["current_player"] == 0
        assert pair["p1"]["converted"]["current_player"] == 0

    def test_raw_game_id_is_still_reported(self, pair: dict[str, Any]) -> None:
        assert pair["p0"]["converted"]["_player_id"] == 0
        assert pair["p1"]["converted"]["_player_id"] == 1

    @pytest.mark.parametrize(("ai_id", "enemy_id"), [(0, 1), (1, 0), (5, 7), (7, 2), (3, 0)])
    def test_two_player_whatever_the_ids(self, ai_id: int, enemy_id: int) -> None:
        state = _mirrored(ai_id, enemy_id)
        converted = StateConverter(board_size=BOARD, research_registry=False).convert(state)

        assert converted["current_player"] == 0
        assert converted["_player_id"] == ai_id
        own = {u["_id"] for u in converted["units"] if u["player"] == converted["current_player"]}
        assert own == {OWN_WARRIOR_A[0], OWN_WARRIOR_B[0]}
        own_cities = {
            c["_id"] for c in converted["cities"] if c["player"] == converted["current_player"]
        }
        assert own_cities == {OWN_CITY[0]}

    def test_acting_player_owns_the_ai_units_in_both(self, pair: dict[str, Any]) -> None:
        for key in ("p0", "p1"):
            converted = pair[key]["converted"]
            mine = sorted(
                u["_id"] for u in converted["units"] if u["player"] == converted["current_player"]
            )
            assert mine == [OWN_WARRIOR_A[0], OWN_WARRIOR_B[0]], key

    def test_mirrored_states_convert_to_the_same_units_and_cities(
        self, pair: dict[str, Any]
    ) -> None:
        def strip(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{k: v for k, v in i.items() if k != "_owner_id"} for i in items]

        assert strip(pair["p0"]["converted"]["units"]) == strip(pair["p1"]["converted"]["units"])
        assert strip(pair["p0"]["converted"]["cities"]) == strip(pair["p1"]["converted"]["cities"])

    @pytest.mark.parametrize("player_id", [0, 1, 2, 3])
    def test_more_than_two_players_keeps_the_raw_id(self, player_id: int) -> None:
        converter = StateConverter(board_size=BOARD, num_players=4, research_registry=False)
        state = make_state(
            player_id=player_id,
            size=BOARD,
            units=[unit(10 + p, "UNIT_WARRIOR", p, 2 + p, 2 + p) for p in range(4)],
        )
        converted = converter.convert(state)

        assert converted["current_player"] == player_id
        assert converted["_player_id"] == player_id
        assert {u["_id"]: u["player"] for u in converted["units"]} == {10 + p: p for p in range(4)}


class TestEncoderSeesTheAiAsOwn:
    def test_spatial_tensors_are_equal(self, pair: dict[str, Any]) -> None:
        spatial = {}
        for key in ("p0", "p1"):
            spatial[key], _ = pair[key]["converter"].convert_and_encode(pair[key]["state"])
        assert torch.equal(spatial["p0"], spatial["p1"])

    def test_own_and_enemy_warriors_are_in_different_channels(self, pair: dict[str, Any]) -> None:
        """Equal tensors are not enough if both were swapped; pin the channels apart."""
        channels = {}
        for key in ("p0", "p1"):
            spatial, _ = pair[key]["converter"].convert_and_encode(pair[key]["state"])
            own = set(
                torch.nonzero(spatial[:, OWN_WARRIOR_A[2], OWN_WARRIOR_A[1]]).flatten().tolist()
            )
            enemy = set(
                torch.nonzero(spatial[:, ENEMY_WARRIOR[2], ENEMY_WARRIOR[1]]).flatten().tolist()
            )
            channels[key] = (own, enemy)
        assert channels["p0"] == channels["p1"]
        own_channels, enemy_channels = channels["p1"]
        assert own_channels
        assert enemy_channels
        assert own_channels != enemy_channels

    def test_player_data_comes_from_the_ais_own_row(self, pair: dict[str, Any]) -> None:
        for key in ("p0", "p1"):
            converted = pair[key]["converted"]
            assert converted["gold"] == 111.0, key
            assert converted["score"] == 10, key


class TestActionMask:
    def test_same_actions_offered_for_both(self, pair: dict[str, Any]) -> None:
        converter = _action_converter()
        masks = {
            key: converter.generate_action_mask(
                pair[key]["state"].legal_actions, pair[key]["converted"]
            )
            for key in ("p0", "p1")
        }
        # End of turn, two moves and a skip: the unit ids are the AI's in both states.
        assert int(masks["p0"].sum()) == 4
        assert int(masks["p1"].sum()) == 4
        assert np.array_equal(masks["p0"], masks["p1"])


class TestSimulatorSeed:
    def test_same_number_of_legal_actions(self, pair: dict[str, Any]) -> None:
        legal = {}
        for key in ("p0", "p1"):
            env = BridgeMCTSEnv.from_bridge_state(pair[key]["converted"])
            legal[key] = int(env.get_valid_actions().sum())
        # Two interior warriors move eight ways each, plus two skips and the end of turn
        # (plus the city's production choices): the AI's side, not the corner warrior's.
        assert legal["p0"] == legal["p1"]
        assert legal["p1"] >= 2 * 8 + 2 + 1

    def test_the_acting_simulator_player_owns_the_ai_units(self, pair: dict[str, Any]) -> None:
        for key in ("p0", "p1"):
            env = BridgeMCTSEnv.from_bridge_state(pair[key]["converted"])
            mine = sorted(u.id for u in env.env.units if u.player == env.current_player)
            assert mine == [OWN_WARRIOR_A[0], OWN_WARRIOR_B[0]], key
            cities = sorted(c.id for c in env.env.cities if c.player == env.current_player)
            assert cities == [OWN_CITY[0]], key

    def test_the_search_acts_for_the_ai_in_both(self, pair: dict[str, Any]) -> None:
        """A legal move of the simulator's own unit 0 starts from the AI's square."""
        for key in ("p0", "p1"):
            env = BridgeMCTSEnv.from_bridge_state(pair[key]["converted"])
            handler = env.env.action_space_handler
            move = handler.encode_action({"type": "move", "unit_idx": 0, "direction": (1, 0)})
            assert env.get_valid_actions()[move] == 1.0, key
            env.step(move)
            moved = next(u for u in env.env.units if u.id == OWN_WARRIOR_A[0])
            assert moved.position == (OWN_WARRIOR_A[1] + 1, OWN_WARRIOR_A[2]), key
