"""Backwards-compatibility and AST layering invariants for the split bridge protocol."""

from __future__ import annotations

import ast
from pathlib import Path

from alpha_zero import bridge as bridge_pkg
from alpha_zero.bridge import protocol, protocol_entities, protocol_validation

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = REPO_ROOT / "alpha_zero" / "bridge"

PROTOCOL_LAYERS = {
    "protocol_validation": 0,
    "protocol_entities": 1,
    "protocol": 2,
}


def test_reexports_preserve_object_identity() -> None:
    """Every re-exported symbol in protocol.py and bridge/__init__.py must be identical."""
    validation_symbols = (
        "ProtocolValidationError",
        "MessageType",
        "ActionType",
        "ErrorCode",
        "END_TURN_BLOCKED",
        "LEGACY_STATE_SCHEMA_VERSION",
        "ERA_NAME_TO_INDEX",
        "_validate_required",
        "_validate_type",
        "_validate_int",
        "_validate_float",
        "_validate_string",
        "_validate_list",
        "_validate_dict",
        "_optional_text",
        "_optional_number",
        "_optional_int",
        "_parse_era",
    )
    for name in validation_symbols:
        assert getattr(protocol, name) is getattr(protocol_validation, name), name

    entity_symbols = (
        "Position",
        "_NAMED_ITEM_ACTIONS",
        "_TYPE_NAME",
        "LuaCommand",
        "TileData",
        "UnitData",
        "CityData",
        "ResearchStatus",
        "_parse_research_status",
        "PlayerData",
        "LegalAction",
        "MapData",
    )
    for name in entity_symbols:
        assert getattr(protocol, name) is getattr(protocol_entities, name), name

    pkg_exports = (
        "AckMessage",
        "ActionType",
        "CityData",
        "CommandMessage",
        "ErrorCode",
        "ErrorMessage",
        "GameStateData",
        "LegalAction",
        "LuaCommand",
        "MapData",
        "MessageType",
        "PlayerData",
        "Position",
        "StateMessage",
        "TileData",
        "UnitData",
        "create_city_produce_command",
        "create_end_turn_command",
        "create_unit_move_command",
        "validate_state_json",
    )
    for name in pkg_exports:
        assert getattr(bridge_pkg, name) is getattr(protocol, name), name


def test_split_modules_stay_below_god_file_limit_and_acyclic() -> None:
    """All three protocol modules stay < 1000 LOC and strictly layered 0 -> 1 -> 2."""
    for mod_name, layer in PROTOCOL_LAYERS.items():
        path = BRIDGE_DIR / f"{mod_name}.py"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) < 1000, f"{mod_name}.py grew to {len(lines)} lines (>= 1000)"

        tree = ast.parse("\n".join(lines))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("alpha_zero.bridge."):
                    sibling = node.module.removeprefix("alpha_zero.bridge.")
                    if sibling in PROTOCOL_LAYERS:
                        assert PROTOCOL_LAYERS[sibling] < layer, (
                            f"{mod_name}.py (layer {layer}) must not import "
                            f"{sibling}.py (layer {PROTOCOL_LAYERS[sibling]})"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("alpha_zero.bridge."):
                        sibling = alias.name.removeprefix("alpha_zero.bridge.")
                        if sibling in PROTOCOL_LAYERS:
                            assert PROTOCOL_LAYERS[sibling] < layer
