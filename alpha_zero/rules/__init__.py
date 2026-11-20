"""Data-driven game rule definitions (the rules kernel, GM-K1).

A leaf package: the standard library and ``yaml`` only, no other ``alpha_zero``
import, no torch or numpy and no per-game state. It defines what a game rule set
*is* (frozen integer-only specs), loads one strictly from YAML, validates its
structure, answers queries over it and gives it an identity:
``(id, sha256 of the canonical parsed form)``.

Nothing on the release play path imports this package, and no environment uses
it yet: it adds no rule to any game. See ``docs/GAME_RULES.md``.
"""

from __future__ import annotations

from alpha_zero.rules.builtin import BUILTIN, RulesetHashMismatchError, builtin_ids, load_builtin
from alpha_zero.rules.loader import (
    RulesetLoadError,
    canonical_json,
    content_sha256,
    load_ruleset,
    parse_ruleset,
)
from alpha_zero.rules.queries import available, prerequisite_closure, research_cost
from alpha_zero.rules.registry import Registry, build_registry
from alpha_zero.rules.spec import (
    BuildingSpec,
    ModifierSpec,
    RulesetSpec,
    TechSpec,
    UnitSpec,
    VictorySpec,
)
from alpha_zero.rules.validate import Issue, RulesetValidationError, ensure_valid

__all__ = [
    "BUILTIN",
    "BuildingSpec",
    "Issue",
    "ModifierSpec",
    "Registry",
    "RulesetHashMismatchError",
    "RulesetLoadError",
    "RulesetSpec",
    "RulesetValidationError",
    "TechSpec",
    "UnitSpec",
    "VictorySpec",
    "available",
    "build_registry",
    "builtin_ids",
    "canonical_json",
    "content_sha256",
    "ensure_valid",
    "load_builtin",
    "load_ruleset",
    "parse_ruleset",
    "prerequisite_closure",
    "research_cost",
]
