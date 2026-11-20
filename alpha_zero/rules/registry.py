"""Name-to-index maps for a rule set, in declared order.

An encoder or an action space needs a stable integer for each tech, unit and
building. The index is the position in the rule-set file, so the maps are equal
in every process, and ``fingerprint`` lets two sides check they agree without
exchanging the maps.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from alpha_zero.rules.spec import RulesetSpec


@dataclass(frozen=True)
class Registry:
    """Read-only index maps of one rule set."""

    ruleset_id: str
    tech_index: Mapping[str, int]
    unit_index: Mapping[str, int]
    building_index: Mapping[str, int]
    fingerprint: str

    def __hash__(self) -> int:
        # The generated hash would hash the (unhashable) mapping proxies.
        return hash((self.ruleset_id, self.fingerprint))

    @property
    def tech_names(self) -> tuple[str, ...]:
        """Tech ids by index."""
        return tuple(self.tech_index)


def _index(names: list[str]) -> Mapping[str, int]:
    return MappingProxyType({name: i for i, name in enumerate(names)})


def build_registry(spec: RulesetSpec) -> Registry:
    """Index ``spec``'s techs, units and buildings in declared order.

    The fingerprint is the sha256 of the three ordered name lists. It does not
    cover costs: two rule sets that differ only in numbers index identically.
    """
    techs = [tech.id for tech in spec.techs]
    units = [unit.id for unit in spec.units]
    buildings = [building.id for building in spec.buildings]
    payload = json.dumps(
        {"techs": techs, "units": units, "buildings": buildings},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return Registry(
        ruleset_id=spec.id,
        tech_index=_index(techs),
        unit_index=_index(units),
        building_index=_index(buildings),
        fingerprint=hashlib.sha256(payload.encode("ascii")).hexdigest(),
    )
