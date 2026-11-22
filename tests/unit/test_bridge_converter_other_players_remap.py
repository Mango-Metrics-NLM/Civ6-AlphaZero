"""Renumbering player ids must renumber ``other_players`` too (review of PR #164).

``StateConverter.convert_multiplayer_state`` renumbers non-contiguous player ids. It rewrote
the owners of units and cities and left ``other_players[*].id`` alone, so a barbarian whose
units became owner 2 still had its entry under 63: no entry for owner 2, relation unknown, its
units left out. It also renumbered only the ``owner_id`` key, while the mod writes ``owner``.
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.bridge.state_converter import StateConverter

MAPPING = {0: 0, 3: 1, 63: 2}


def _raw(owner_key: str) -> dict[str, Any]:
    def unit(uid: int, owner: int, x: int) -> dict[str, Any]:
        return {"id": uid, owner_key: owner, "x": x, "y": 1, "type_name": "UNIT_WARRIOR"}

    return {
        "turn_id": 4,
        "player_id": 0,
        "map": {"width": 16, "height": 16},
        "players": [{"id": 0}, {"id": 3}, {"id": 63}],
        "units": [unit(1, 0, 1), unit(2, 3, 5), unit(3, 63, 9)],
        "cities": [{"id": 7, owner_key: 63, "x": 12, "y": 12, "name": "Camp"}],
        "observes_others": True,
        "other_players": [
            {"id": 3, "kind": "major", "at_war": False},
            {"id": 63, "kind": "barbarian", "at_war": True},
        ],
    }


def _convert(owner_key: str, mapping: dict[int, int] | None = None) -> dict[str, Any]:
    converter = StateConverter(board_size=16, num_players=3, create_encoder=False)
    return converter.convert_multiplayer_state(
        _raw(owner_key), num_players=3, player_mapping=mapping or MAPPING
    )


@pytest.mark.parametrize("owner_key", ["owner", "owner_id"])
class TestRenumberedOwnersKeepTheirRelation:
    def test_the_barbarian_stays_on_the_enemy_side(self, owner_key: str) -> None:
        converted = _convert(owner_key)
        assert [(u["_id"], u["player"]) for u in converted["units"]] == [(1, 0), (3, 2)]
        assert [(c["_id"], c["player"]) for c in converted["cities"]] == [(7, 2)]

    def test_the_counts_follow(self, owner_key: str) -> None:
        others = _convert(owner_key)["_others"]
        assert others["hostile_units"] == 1
        assert others["hostile_cities"] == 1
        assert others["left_out_not_hostile"] == 1
        assert others["left_out_unknown"] == 0

    def test_the_caller_s_state_is_not_changed(self, owner_key: str) -> None:
        raw = _raw(owner_key)
        converter = StateConverter(board_size=16, num_players=3, create_encoder=False)
        converter.convert_multiplayer_state(raw, num_players=3, player_mapping=MAPPING)
        assert [entry["id"] for entry in raw["other_players"]] == [3, 63]
        assert [unit[owner_key] for unit in raw["units"]] == [0, 3, 63]


class TestRemapIds:
    def test_an_owner_outside_the_mapping_is_left_alone(self) -> None:
        remapped = StateConverter._remap_player_ids(_raw("owner"), {0: 0, 3: 1})
        assert [u["owner"] for u in remapped["units"]] == [0, 1, 63]
        assert [entry["id"] for entry in remapped["other_players"]] == [1, 63]

    def test_a_state_without_other_players_is_unchanged_in_that_respect(self) -> None:
        raw = _raw("owner_id")
        del raw["other_players"]
        remapped = StateConverter._remap_player_ids(raw, MAPPING)
        assert "other_players" not in remapped
        assert [u["owner_id"] for u in remapped["units"]] == [0, 1, 2]

    def test_entries_that_are_not_dictionaries_do_not_break_it(self) -> None:
        raw = _raw("owner")
        raw["other_players"] = [{"id": 63, "kind": "barbarian"}, "junk", None]
        remapped = StateConverter._remap_player_ids(raw, MAPPING)
        assert remapped["other_players"][0]["id"] == 2
