"""Shared constants and plain helpers for the GM-B3 research-wiring tests.

Not collected as tests. Pytest fixtures are defined in each test module that uses them.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from alpha_zero.bridge.action_converter import ActionConverter
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.core.civ6_actions import ActionCategory
from alpha_zero.rules import parse_ruleset

TECH_IDS = ["writing", "bronze_working", "pottery", "mining"]
# Slots differ from the stub's GameInfo row indexes for every tech and civic used below, so a
# converter that passes the row index through (or the other way round) cannot pass by luck.
#   techs  : stub rows pottery 0, mining 1, writing 2, bronze_working 3
#   civics : stub rows code_of_laws 0, craftsmanship 1, foreign_trade 2, early_empire 3
CIVIC_TYPES = (
    "CIVIC_FOREIGN_TRADE",
    "CIVIC_CODE_OF_LAWS",
    "CIVIC_EARLY_EMPIRE",
    "CIVIC_CRAFTSMANSHIP",
)
RULESET = """\
schema_version: 1
id: research-test
techs:
{techs}
units:
  - {{id: sword, cost: 5, strength: 3, moves: 2}}
buildings:
  - id: hut
    cost: 5
    modifiers:
      - {{yield: food, amount: 1}}
"""
END_TURN = {"action_type": "end_turn"}
TECH_RANGE = ActionCategory.RESEARCH_TECH
CIVIC_RANGE = ActionCategory.RESEARCH_CIVIC
GAME_STATE: dict[str, Any] = {"units": [], "cities": [], "current_player": 0}


def big_registry(tr: ModuleType, techs: int, civics: int) -> Any:
    return tr.ResearchRegistry(
        ruleset_id="big",
        fingerprint="0" * 64,
        tech_ids=tuple(f"t{i}" for i in range(techs)),
        civic_ids=tuple(f"c{i}" for i in range(civics)),
    )


def payload(
    techs: tuple[str, ...] = (),
    civics: tuple[str, ...] = (),
    turn: int = 3,
    legal: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "turn_id": turn,
        "player_id": 0,
        "map": {"width": 8, "height": 8, "tiles": []},
        "units": [],
        "cities": [],
        "legal_actions": legal if legal is not None else [{"action": "end_turn"}],
        "players": [
            {
                "id": 0,
                "civilization": "CIVILIZATION_STUB_A",
                "techs_researched": list(techs),
                "civics_researched": list(civics),
            }
        ],
    }


def encode(conv: StateConverter, state: Any) -> np.ndarray:
    _, vector = conv.convert_and_encode(state)
    return np.asarray(vector.detach().cpu().numpy())


def research_bits(conv: ActionConverter, mask: np.ndarray, category: ActionCategory) -> set[int]:
    offset = conv.action_space.category_offsets[category]
    size = conv.action_space.category_sizes[category]
    return {int(i) for i in np.nonzero(mask[offset : offset + size])[0]}


def tech_action(names: list[str] | None, items: list[int] | None = None) -> dict[str, Any]:
    action: dict[str, Any] = {"action": "research_tech", "items": items or [1, 2]}
    if names is not None:
        action["item_names"] = names
    return action


def civic_action(names: list[str] | None, items: list[int] | None = None) -> dict[str, Any]:
    action: dict[str, Any] = {"action": "research_civic", "items": items or [1, 2]}
    if names is not None:
        action["item_names"] = names
    return action


END_ONLY: list[dict[str, Any]] = [{"action": "end_turn"}]


def action_converter(registry: Any = None) -> ActionConverter:
    return ActionConverter(board_size=8, research_registry=registry)


def flat(conv: ActionConverter, category: ActionCategory, slot: int) -> int:
    return conv.action_space.category_offsets[category] + slot


STATE_LOGGER = "alpha_zero.bridge.state_converter"
ACTION_LOGGER = "alpha_zero.bridge.action_converter"
REPO_ROOT = Path(__file__).resolve().parents[2]


def registry_from(
    tr: ModuleType, tech_ids: list[str], civics: tuple[str, ...] = CIVIC_TYPES
) -> Any:
    techs = "\n".join(f"  - id: {i}\n    era: 1\n    cost: 10" for i in tech_ids)
    return tr.from_ruleset(parse_ruleset(RULESET.format(techs=techs)), civics)


def messages(caplog: pytest.LogCaptureFixture, logger: str, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == logger and r.levelno == level]


def new_state(registry: Any) -> StateConverter:
    return StateConverter(research_registry=registry)


def new_actions(registry: Any) -> ActionConverter:
    return ActionConverter(board_size=8, research_registry=registry)


FACTORIES = [new_state, new_actions]
