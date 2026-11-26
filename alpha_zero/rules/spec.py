"""Frozen, integer-only descriptions of a game rule set.

These are plain data. They hold no per-game state and do no validation beyond
their types: structure is checked by :mod:`alpha_zero.rules.validate`, and strict
construction from a parsed file by :mod:`alpha_zero.rules.loader`.

Every number is an ``int``. Percentages are integer percent. Sequences are tuples
in declared order, which is the order registries index them in.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Version of the file schema this module describes.
SCHEMA_VERSION: int = 1

#: What a modifier may change.
MODIFIER_YIELDS: tuple[str, ...] = ("food", "production", "science", "population_cap")

#: What a modifier's amount is counted per. ``city``: once per city.
#: ``worked_terrain``: once per worked tile of terrain ``ref``.
#: ``building``: once per building ``ref`` in the city.
MODIFIER_SCOPES: tuple[str, ...] = ("city", "worked_terrain", "building")


@dataclass(frozen=True)
class ModifierSpec:
    """A passive integer bonus: ``amount`` of ``yield_`` per ``per`` (of ``ref``)."""

    yield_: str
    amount: int
    per: str = "city"
    ref: str | None = None


@dataclass(frozen=True)
class UnitSpec:
    """A unit a city can build. ``cost`` is in production."""

    id: str
    cost: int
    strength: int
    moves: int


@dataclass(frozen=True)
class BuildingSpec:
    """A building a city can build. ``cost`` is in production."""

    id: str
    cost: int
    modifiers: tuple[ModifierSpec, ...] = ()


@dataclass(frozen=True)
class TechSpec:
    """A technology. ``cost`` is in science; ``boost_percent`` is integer percent."""

    id: str
    era: int
    cost: int
    prerequisites: tuple[str, ...] = ()
    unlocks_units: tuple[str, ...] = ()
    unlocks_buildings: tuple[str, ...] = ()
    modifiers: tuple[ModifierSpec, ...] = ()
    boost_percent: int = 0


@dataclass(frozen=True)
class VictorySpec:
    """How the rule set can be won beyond the base game.

    ``capstone_tech`` is ``None`` when the rule set declares no science victory
    (an imported real tree, for example).
    """

    capstone_tech: str | None = None


@dataclass(frozen=True)
class RulesetSpec:
    """One game rule set, as declared."""

    id: str
    techs: tuple[TechSpec, ...] = ()
    units: tuple[UnitSpec, ...] = ()
    buildings: tuple[BuildingSpec, ...] = ()
    victory: VictorySpec = VictorySpec()
    schema_version: int = SCHEMA_VERSION

    def tech(self, tech_id: str) -> TechSpec:
        """Return the tech named ``tech_id``; ``KeyError`` when there is none."""
        for tech in self.techs:
            if tech.id == tech_id:
                return tech
        raise KeyError(tech_id)
