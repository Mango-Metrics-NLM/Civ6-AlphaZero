"""GM-B6a: translator rules for ``allowed``, malformed input and ``root_mask``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from alpha_zero.bridge.search_translator import SearchTranslator
from tests.fixtures.bridge_search_translation import build, convert, is_offered, rich_state


def _move(unit_idx: int, dx: int, dy: int) -> dict[str, Any]:
    return {"type": "move", "unit_idx": unit_idx, "direction": (dx, dy)}


def _translator(state: Any, units: list[int], cities: list[int] | None = None, **kw: Any) -> Any:
    return SearchTranslator(convert(state), units, cities or [], **kw)


def _warrior_state(targets: list[tuple[int, int]]) -> Any:
    from alpha_zero.bridge.protocol import LegalAction
    from tests.fixtures.bridge_search_states import make_state, unit

    return make_state(
        width=84,
        height=54,
        units=[unit(2, "UNIT_WARRIOR", 0, 40, 27)],
        legal_actions=[LegalAction("end_turn"), LegalAction("move", unit_id=2, targets=targets)],
    )


class TestAllowedAndMalformed:
    def test_allowed_rules_a_command_out(self) -> None:
        seen: list[dict[str, Any]] = []

        def allowed(command: dict[str, Any]) -> bool:
            seen.append(command)
            return command["action_type"] != "unit_skip"

        t = _translator(rich_state(), [2, 4, 5], allowed=allowed)
        skip = t.translate({"type": "skip", "unit_idx": 0})
        assert skip.command is None and "ruled out" in (skip.reason or "")
        assert seen == [{"action_type": "unit_skip", "unit_id": 2}]
        assert t.translate({"type": "end_turn"}).command == {"action_type": "end_turn"}

    @pytest.mark.parametrize(
        "action",
        [
            None,
            {},
            "move",
            {"type": "teleport"},
            {"type": "move"},
            {"type": "move", "unit_idx": 0},
            {"type": "move", "unit_idx": "0", "direction": (1, 0)},
            {"type": "move", "unit_idx": 0, "direction": (1,)},
            {"type": "move", "unit_idx": 0, "direction": None},
            {"type": "attack", "unit_idx": 0, "direction": ("a", "b")},
            {"type": "skip", "unit_idx": None},
            {"type": "produce", "city_idx": 0},
            {"type": "produce", "city_idx": 0, "unit_type": "dragon"},
        ],
    )
    def test_malformed_action_does_not_raise(self, action: Any) -> None:
        result = _translator(rich_state(), [2, 4, 5], [10]).translate(action)
        assert result.command is None
        assert isinstance(result.reason, str) and result.reason

    @pytest.mark.parametrize(
        "converted",
        [
            {},
            {"_raw_state": None},
            {"_raw_state": object(), "units": None, "cities": 3},
            {"_map_size": (0, 0), "_board_size": 0, "units": [], "cities": []},
            {"_map_size": "big", "_board_size": "x", "_raw_state": None},
        ],
    )
    def test_malformed_state_does_not_raise(self, converted: Any) -> None:
        t = SearchTranslator(converted, [2], [10])
        for action in (
            {"type": "end_turn"},
            {"type": "skip", "unit_idx": 0},
            _move(0, 1, 0),
            {"type": "produce", "city_idx": 0, "unit_type": "warrior"},
        ):
            result = t.translate(action)
            assert result.command is None and result.reason

    def test_malformed_ids_do_not_raise(self) -> None:
        t = SearchTranslator(convert(rich_state()), [None, "x", 2], [None])  # type: ignore[list-item]
        assert t.translate({"type": "skip", "unit_idx": 0}).command is None
        assert t.translate({"type": "skip", "unit_idx": 2}).command == {
            "action_type": "unit_skip",
            "unit_id": 2,
        }


class TestRootMask:
    def test_every_command_is_one_the_game_offered(self) -> None:
        """Checked by ``is_offered``, which reads the raw offers and shares no code."""
        state = rich_state()
        converted, env = build(state)
        translator = SearchTranslator(converted, env.own_unit_ids(), env.own_city_ids())
        valid = env.get_valid_actions()
        before = valid.copy()
        mask, commands, reasons = translator.root_mask(
            valid, env.env.action_space_handler.decode_action
        )

        assert commands, "the rich state must offer something"
        kinds = {c["action_type"] for c in commands.values()}
        assert kinds >= {"end_turn", "unit_skip", "unit_move", "unit_found_city", "city_produce"}
        for index, command in commands.items():
            assert is_offered(command, state.legal_actions), (index, command)
        # Negative control for the checker itself: it does reject things that are not offered.
        assert not is_offered({"action_type": "unit_skip", "unit_id": 99}, state.legal_actions)
        assert not is_offered(
            {"action_type": "unit_move", "unit_id": 2, "target": {"x": 0, "y": 0}},
            state.legal_actions,
        )
        # Mask, commands and reasons describe one partition of the valid indexes.
        assert np.array_equal(valid, before), "the input mask must not be changed"
        assert mask.shape == valid.shape
        assert set(np.nonzero(mask)[0].tolist()) == set(commands)
        assert set(commands) <= set(np.nonzero(valid)[0].tolist())
        assert len(commands) + sum(reasons.values()) == int(np.count_nonzero(valid))
        assert sum(reasons.values()) > 0, "some simulator moves have no offered tile"
        assert all(isinstance(k, str) and isinstance(v, int) for k, v in reasons.items())

    def test_the_builder_is_never_commanded(self) -> None:
        """Builder 7 has offers, but the simulator has no such unit, so no index reaches it."""
        state = rich_state()
        converted, env = build(state)
        translator = SearchTranslator(converted, env.own_unit_ids(), env.own_city_ids())
        _, commands, _ = translator.root_mask(
            env.get_valid_actions(), env.env.action_space_handler.decode_action
        )
        assert all(c.get("unit_id") != 7 for c in commands.values())

    def test_two_actions_with_one_command_both_stay(self) -> None:
        state = _warrior_state([(41, 27)])
        converted, env = build(state)
        translator = SearchTranslator(converted, env.own_unit_ids(), env.own_city_ids())
        handler = env.env.action_space_handler
        _, commands, _ = translator.root_mask(env.get_valid_actions(), handler.decode_action)
        same = [i for i, c in commands.items() if c.get("target") == {"x": 41, "y": 27}]
        assert len(same) >= 2
        assert len({repr(commands[i]) for i in same}) == 1

    def test_allowed_removes_commands_from_the_mask(self) -> None:
        state = rich_state()
        converted, env = build(state)
        handler = env.env.action_space_handler
        translator = SearchTranslator(
            converted,
            env.own_unit_ids(),
            env.own_city_ids(),
            allowed=lambda c: c["action_type"] == "end_turn",
        )
        mask, commands, reasons = translator.root_mask(
            env.get_valid_actions(), handler.decode_action
        )
        assert set(commands) == {0}
        assert np.nonzero(mask)[0].tolist() == [0]
        assert any("ruled out" in key for key in reasons)

    def test_empty_valid_mask_gives_an_empty_result(self) -> None:
        converted, env = build(rich_state())
        zeros = np.zeros(env.action_space.n, dtype=np.float32)
        mask, commands, reasons = SearchTranslator(converted, [2], [10]).root_mask(
            zeros, env.env.action_space_handler.decode_action
        )
        assert not mask.any() and commands == {} and reasons == {}
