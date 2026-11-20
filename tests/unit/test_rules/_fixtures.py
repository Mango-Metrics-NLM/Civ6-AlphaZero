"""Shared fixtures for the rules-kernel tests: a tiny valid rule set as YAML text."""

from __future__ import annotations

from alpha_zero.rules.loader import parse_ruleset
from alpha_zero.rules.spec import RulesetSpec

BASE = """\
schema_version: 1
id: tiny-test
techs:
  - id: alpha
    era: 1
    cost: 10
  - id: beta
    era: 2
    cost: 20
    prerequisites: [alpha]
units:
  - {id: sword, cost: 5, strength: 3, moves: 2}
buildings:
  - id: hut
    cost: 5
    modifiers:
      - {yield: food, amount: 1}
victory:
  capstone_tech: beta
"""


def mutate(old: str, new: str, text: str = BASE) -> str:
    """``text`` with ``old`` replaced by ``new``; fails loudly if ``old`` is absent."""
    assert old in text, f"fixture is vacuous: {old!r} not in the base text"
    changed = text.replace(old, new, 1)
    assert changed != text
    return changed


def tiny() -> RulesetSpec:
    return parse_ruleset(BASE)
