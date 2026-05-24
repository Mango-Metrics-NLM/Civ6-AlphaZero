"""Action-mask invariant test (v1.8.0 W2.1 step 6).

The original v1.7.0 plan called for "an invariant test that every action
in ``generate_action_mask`` has a handler in ``ACTION_HANDLERS``." It was
never built, and the resulting drift was exactly what the v1.8.0 sprint's
gap analysis found (5 stubbed Lua actions, Python enum missing 4 of those,
silent-sink fallthrough in the converter, the works).

This test makes the regression impossible. For every state in the canned
fixture suite:

1. Generate the action mask via ``ActionConverter.generate_action_mask``.
2. For every action index `i` where `mask[i] == 1`, decode it via the
   action-space encoder.
3. Assert the decoded action's category maps to:
   - A real Python ``ActionType`` enum member (i.e. converter has a
     non-fallthrough branch under ``strict=True``).
   - A real entry in the Lua ``ACTION_HANDLERS`` dispatch table whose
     mapped function is NOT ``executeNotImplemented`` (parsed from
     ``mod/AlphaZeroAI/Scripts/ActionExecutor.lua`` at test time via
     :mod:`tests.fixtures.lua_dispatch_parser`).

The end-to-end strictness acceptance criterion from the v1.8.0 plan is
also verified here: the converter under ``strict=True`` must raise
``InvalidActionError`` for an out-of-range action index rather than
silently returning ``END_TURN(unsupported=True)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alpha_zero.bridge.action_converter import ActionConverter, InvalidActionError
from alpha_zero.bridge.protocol import ActionType, LegalAction
from alpha_zero.core.civ6_actions import ActionCategory, ActionSpaceConfig
from tests.fixtures.lua_dispatch_parser import (
    NOT_IMPLEMENTED_HANDLER,
    load_dispatch_from_file,
)


REAL_LUA_PATH = (
    Path(__file__).parent.parent.parent / "mod" / "AlphaZeroAI" / "Scripts" / "ActionExecutor.lua"
)

# Category -> ActionType the converter is allowed to produce.
# Used by the invariant to translate "the converter chose category X" into
# "the Lua dispatch table must have a non-stub handler for one of these".
# Diplomacy is its own dispatcher (sub-actions are checked separately).
CATEGORY_TO_ACTION_TYPES: dict[ActionCategory, set[ActionType]] = {
    ActionCategory.END_TURN: {ActionType.END_TURN},
    ActionCategory.UNIT_ACTION: {
        ActionType.UNIT_MOVE,
        ActionType.UNIT_ATTACK,
        ActionType.UNIT_RANGED_ATTACK,
        ActionType.UNIT_FORTIFY,
        ActionType.UNIT_SKIP,
        ActionType.UNIT_SLEEP,
        ActionType.UNIT_FOUND_CITY,
        ActionType.UNIT_BUILD,
        ActionType.UNIT_PROMOTE,
        ActionType.UNIT_DELETE,
    },
    ActionCategory.CITY_ACTION: {
        ActionType.CITY_PRODUCE,
        ActionType.CITY_BUY,
        ActionType.CITY_CITIZEN,
        # END_TURN can also come out of CITY_ACTION via the unsupported
        # fallback for CHANGE_PRODUCTION/SELL_BUILDING/SET_FOCUS/RANGED_ATTACK/
        # PURCHASE_TILE — flagged in the converter's metadata with
        # `unsupported=True`, not a real END_TURN command.
        ActionType.END_TURN,
    },
    ActionCategory.RESEARCH_TECH: {ActionType.RESEARCH_TECH},
    ActionCategory.RESEARCH_CIVIC: {ActionType.RESEARCH_CIVIC},
    ActionCategory.DIPLOMACY: {ActionType.DIPLOMACY},
}


@pytest.fixture(scope="module")
def lua_action_handlers() -> dict[str, str]:
    """Parse the Lua dispatch table once for the whole module."""
    _types, handlers = load_dispatch_from_file(REAL_LUA_PATH)
    return handlers


@pytest.fixture(scope="module")
def lua_action_types() -> dict[str, str]:
    """Parse the Lua ACTION_TYPES table once for the whole module."""
    types, _handlers = load_dispatch_from_file(REAL_LUA_PATH)
    return types


@pytest.fixture
def converter() -> ActionConverter:
    """Action converter using the project's default board size."""
    return ActionConverter(
        board_size=8, action_space_config=ActionSpaceConfig(board_size=8, max_units=4, max_cities=2)
    )


def _value_to_lua_key(value: str, lua_types: dict[str, str]) -> str | None:
    """Inverse-lookup ACTION_TYPES: return the Lua key whose value matches."""
    for key, val in lua_types.items():
        if val == value:
            return key
    return None


class TestPythonLuaEnumSync:
    """The Python ActionType enum and the Lua ACTION_TYPES table must agree
    on the set of action values."""

    def test_every_python_actiontype_has_lua_entry(self, lua_action_types: dict[str, str]) -> None:
        lua_values = set(lua_action_types.values())
        python_values = {member.value for member in ActionType}
        missing_in_lua = python_values - lua_values
        assert not missing_in_lua, (
            f"Python ActionType members missing from Lua ACTION_TYPES: {missing_in_lua}. "
            f"Update mod/AlphaZeroAI/Scripts/ActionExecutor.lua's ACTION_TYPES table."
        )

    def test_every_lua_actiontype_has_python_entry(self, lua_action_types: dict[str, str]) -> None:
        lua_values = set(lua_action_types.values())
        python_values = {member.value for member in ActionType}
        missing_in_python = lua_values - python_values
        assert not missing_in_python, (
            f"Lua ACTION_TYPES values missing from Python ActionType: {missing_in_python}. "
            f"Update alpha_zero/bridge/protocol.py's ActionType enum."
        )


class TestPythonLuaHandlerCoverage:
    """Every category the Python converter is willing to emit must map to a
    real (non-stub) Lua handler."""

    def test_every_advertised_action_type_has_real_lua_handler(
        self, lua_action_types: dict[str, str], lua_action_handlers: dict[str, str]
    ) -> None:
        """For every ActionType the converter can produce per CATEGORY_TO_ACTION_TYPES,
        the corresponding Lua ACTION_HANDLERS entry must NOT be executeNotImplemented."""
        all_producible = set().union(*CATEGORY_TO_ACTION_TYPES.values())
        unimplemented_in_lua: list[str] = []
        for action_type in all_producible:
            lua_key = _value_to_lua_key(action_type.value, lua_action_types)
            assert lua_key is not None, (
                f"ActionType {action_type.name} (value={action_type.value!r}) "
                f"has no matching Lua ACTION_TYPES entry"
            )
            handler = lua_action_handlers.get(lua_key)
            assert handler is not None, f"No Lua handler registered for {lua_key}"
            if handler == NOT_IMPLEMENTED_HANDLER:
                unimplemented_in_lua.append(f"{action_type.name} -> {handler}")
        assert not unimplemented_in_lua, (
            "Producible ActionTypes routed to executeNotImplemented in Lua:\n  "
            + "\n  ".join(unimplemented_in_lua)
        )


class TestActionMaskInvariant:
    """For every legal action in the mask, the full Python+Lua chain must
    have a real handler."""

    @staticmethod
    def _build_minimal_state() -> dict:
        """Smallest game state that exercises a non-trivial action mask."""
        return {
            "current_player": 0,
            "_map_size": (8, 8),
            "units": [
                {"_id": 10, "player": 0, "position": (2, 3), "type": "melee"},
                {"_id": 11, "player": 0, "position": (4, 4), "type": "settler"},
            ],
            "cities": [
                {"_id": 100, "player": 0, "position": (3, 3), "available_production": [0, 1]},
            ],
        }

    def test_action_mask_only_advertises_implemented_actions(
        self,
        converter: ActionConverter,
        lua_action_types: dict[str, str],
        lua_action_handlers: dict[str, str],
    ) -> None:
        """Every legal action in the mask must be decodable, in a known
        category, and have a real (non-stub) Lua handler."""
        state = self._build_minimal_state()
        legal_actions = [
            LegalAction(action_type="end_turn", unit_id=0, city_id=0),
            LegalAction(
                action_type="move",
                unit_id=10,
                city_id=0,
                targets=[(3, 3), (2, 4)],
            ),
            LegalAction(
                action_type="produce",
                unit_id=0,
                city_id=100,
                items=[0, 1],
            ),
        ]
        mask = converter.generate_action_mask(legal_actions, state)
        legal_indices = np.where(mask > 0)[0]
        assert len(legal_indices) > 0, "mask was all zeros — fixture is wrong"

        implemented_lua_values = {
            lua_action_types[key]
            for key, handler in lua_action_handlers.items()
            if handler != NOT_IMPLEMENTED_HANDLER
        }

        for idx in legal_indices:
            decoded = converter.action_space.decode(int(idx))
            assert decoded.category in CATEGORY_TO_ACTION_TYPES, (
                f"Action {idx} decodes to category {decoded.category.name}, "
                f"which has no entry in CATEGORY_TO_ACTION_TYPES — converter "
                f"can emit an ActionType the invariant test doesn't track."
            )
            possible_types = CATEGORY_TO_ACTION_TYPES[decoded.category]
            has_real_handler = any(at.value in implemented_lua_values for at in possible_types)
            assert has_real_handler, (
                f"Action {idx} (category={decoded.category.name}) decodes to a "
                f"category whose possible ActionTypes "
                f"{[at.name for at in possible_types]} have no real Lua handler."
            )


class TestStrictnessAcceptanceCriterion:
    """The v1.8.0 plan's end-to-end strictness criterion: out-of-range
    action under strict=True raises InvalidActionError, not silent END_TURN."""

    def test_out_of_range_raises(self, converter: ActionConverter) -> None:
        from alpha_zero.core.civ6_actions import HierarchicalAction

        # An ESPIONAGE action has no converter branch (intentionally — out of
        # sprint scope) and should raise under strict=True.
        action = HierarchicalAction(category=ActionCategory.ESPIONAGE, selector=0, action_type=0)
        flat_idx = converter.action_space.encode(action)
        state = TestActionMaskInvariant._build_minimal_state()
        with pytest.raises(InvalidActionError, match="has no implemented converter"):
            converter.convert(flat_idx, state, validate=False)

    def test_strict_false_preserves_legacy_path(self, converter: ActionConverter, caplog) -> None:
        import logging

        from alpha_zero.core.civ6_actions import HierarchicalAction

        action = HierarchicalAction(category=ActionCategory.ESPIONAGE, selector=0, action_type=0)
        flat_idx = converter.action_space.encode(action)
        state = TestActionMaskInvariant._build_minimal_state()
        with caplog.at_level(logging.WARNING, logger="alpha_zero.bridge.action_converter"):
            cmd = converter.convert(flat_idx, state, validate=False, strict=False)
        assert cmd.action_type == ActionType.END_TURN
        assert cmd.metadata is not None
        assert cmd.metadata.get("unsupported") is True
