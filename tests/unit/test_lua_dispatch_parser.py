"""Tests for ``tests.fixtures.lua_dispatch_parser``.

The parser underpins the action-mask invariant test (W2.1 step 6). If the
parser silently returns partial data, the invariant test passes with
holes — which is exactly the failure mode the invariant is meant to
prevent. So the parser's edge cases are tested here against synthetic
Lua snippets in ``tests/fixtures/lua_dispatch_samples/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.lua_dispatch_parser import (
    NOT_IMPLEMENTED_HANDLER,
    LuaParseError,
    implemented_action_values,
    load_dispatch_from_file,
    parse_action_types,
)

SAMPLES_DIR = Path(__file__).parent.parent / "fixtures" / "lua_dispatch_samples"
REAL_LUA = (
    Path(__file__).parent.parent.parent / "mod" / "AlphaZeroAI" / "Scripts" / "ActionExecutor.lua"
)


class TestParseActionTypes:
    def test_handles_comments_and_trailing_commas(self) -> None:
        types, _ = load_dispatch_from_file(SAMPLES_DIR / "with_comments_and_trailing_commas.lua")
        assert types == {
            "END_TURN": "end_turn",
            "UNIT_MOVE": "unit_move",
            "UNIT_ATTACK": "unit_attack",
        }

    def test_raises_on_missing_block(self) -> None:
        text = 'local OTHER_TABLE = { FOO = "bar" }'
        with pytest.raises(LuaParseError, match="ACTION_TYPES block not found"):
            parse_action_types(text)

    def test_raises_on_empty_block(self) -> None:
        text = "local ACTION_TYPES = { }"
        with pytest.raises(LuaParseError, match="contained no entries"):
            parse_action_types(text)

    def test_double_dash_in_string_literal_fails_loud(self) -> None:
        """Gap-analysis I4 documented behaviour: a string value containing
        ``--`` corrupts the affected line via the comment-stripper, but the
        failure mode is *loud* — the line is dropped, the table appears
        empty / malformed, and we raise ``LuaParseError`` rather than
        silently returning truncated data. The downstream invariant test
        also catches the resulting entry-count mismatch."""
        # Single-line table: '--' makes the closing brace get stripped.
        text = 'local ACTION_TYPES = { OPTION = "foo--bar" }'
        with pytest.raises(LuaParseError):
            parse_action_types(text)
        # Multi-line table: '--' on one entry drops just that entry; if it
        # was the only entry, the parser raises "no entries".
        multi = '\n'.join([
            "local ACTION_TYPES = {",
            '    OPTION = "foo--bar",',
            "}",
        ])
        with pytest.raises(LuaParseError, match="no entries"):
            parse_action_types(multi)


class TestParseActionHandlers:
    def test_distinguishes_real_handler_from_stub(self) -> None:
        types, handlers = load_dispatch_from_file(
            SAMPLES_DIR / "with_comments_and_trailing_commas.lua"
        )
        assert handlers == {
            "END_TURN": "executeEndTurn",
            "UNIT_MOVE": "executeUnitMove",
            "UNIT_ATTACK": NOT_IMPLEMENTED_HANDLER,
        }
        assert implemented_action_values(types, handlers) == {"end_turn", "unit_move"}

    def test_raises_on_missing_handlers_block(self) -> None:
        with pytest.raises(LuaParseError, match="ACTION_HANDLERS block not found"):
            load_dispatch_from_file(SAMPLES_DIR / "missing_handlers_block.lua")

    def test_raises_on_duplicate_key(self) -> None:
        with pytest.raises(LuaParseError, match="Duplicate ACTION_HANDLERS"):
            load_dispatch_from_file(SAMPLES_DIR / "duplicate_key.lua")


class TestRealLuaFile:
    """Spot-check against the actual mod file shipped in this repo."""

    def test_parses_real_action_executor(self) -> None:
        types, handlers = load_dispatch_from_file(REAL_LUA)
        assert "END_TURN" in types
        assert types["END_TURN"] == "end_turn"
        assert handlers["END_TURN"] == "executeEndTurn"

    def test_v1_8_0_actions_have_real_handlers(self) -> None:
        """The 5 W2.1 actions (4 non-diplomacy + DIPLOMACY) must NOT route
        to executeNotImplemented after W2.1 steps 4-5 land."""
        types, handlers = load_dispatch_from_file(REAL_LUA)
        for key in ("CITY_BUY", "CITY_CITIZEN", "UNIT_PROMOTE", "UNIT_DELETE", "DIPLOMACY"):
            assert key in handlers, f"{key} missing from ACTION_HANDLERS"
            assert (
                handlers[key] != NOT_IMPLEMENTED_HANDLER
            ), f"{key} still routes to {NOT_IMPLEMENTED_HANDLER} — W2.1 not done"

    def test_keys_match_between_action_types_and_handlers(self) -> None:
        """Every ACTION_TYPES entry must have a corresponding ACTION_HANDLERS
        entry (and vice-versa). Parser-correctness only — the "all handlers
        are real (non-stub)" assertion lives in
        :mod:`tests.unit.test_bridge_action_invariant` where it belongs.

        Gap-analysis I2 on commit 3572832 surfaced the previous version of
        this test as conflating parser correctness with implementation
        completeness: that conflation would have failed at the wrong time
        (during development of a deferred handler) and the proper "no stubs"
        gate is the invariant test."""
        types, handlers = load_dispatch_from_file(REAL_LUA)
        assert set(types.keys()) == set(handlers.keys()), (
            "ACTION_TYPES and ACTION_HANDLERS keys must match exactly. "
            "Symmetric difference: " + str(set(types.keys()) ^ set(handlers.keys()))
        )

    def test_implemented_action_values_helper_returns_only_non_stubs(self) -> None:
        """Helper correctness — the function MUST NOT include action values
        whose handler is ``executeNotImplemented`` (this is the property the
        invariant test depends on)."""
        types, handlers = load_dispatch_from_file(
            SAMPLES_DIR / "with_comments_and_trailing_commas.lua"
        )
        impl = implemented_action_values(types, handlers)
        # UNIT_ATTACK in the sample maps to executeNotImplemented; it must
        # NOT appear in the implemented set.
        assert "unit_attack" not in impl
        assert impl == {"end_turn", "unit_move"}
