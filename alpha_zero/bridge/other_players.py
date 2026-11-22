"""Who the other owners in a live state are to the agent (GM-B8a).

The mod exports what the player currently sees of other players, and for each
such owner what it is (``kind``: major, minor, barbarian, free) and whether the
player is at war with it (``at_war``; absent when the game could not be asked).
The agent has two sides, its own and the enemy's, so only owners that are
hostile go on the enemy side; a neutral scout is not an enemy. The rest are
left out and counted, never shown as hostile.

A state from an older mod carries none of this (``observes_others is None``)
and is handled as before: every other owner's unit and city is on the enemy
side.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

__all__ = ["is_hostile", "owner_relations", "partition", "shown_to_the_agent"]


def optional_bool(value: Any) -> bool | None:
    """A boolean, or None for anything else (an absent key)."""
    return value if isinstance(value, bool) else None


def dict_entries(value: Any) -> list[dict[str, Any]]:
    """The dictionaries of a list; anything else (an empty Lua table sent as ``{}``) is none."""
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []


HOSTILE = "hostile"
NOT_HOSTILE = "not_hostile"
UNKNOWN = "unknown"

#: Kinds that are hostile whatever the diplomacy says.
_ALWAYS_HOSTILE = frozenset({"barbarian", "free"})


def is_hostile(entry: Any) -> bool:
    """Whether an ``other_players`` entry is an enemy: a barbarian, a free city, or at war."""
    if not isinstance(entry, Mapping):
        return False
    return entry.get("kind") in _ALWAYS_HOSTILE or entry.get("at_war") is True


def _relation(entry: Any) -> str:
    if is_hostile(entry):
        return HOSTILE
    if isinstance(entry, Mapping) and entry.get("at_war") is False:
        return NOT_HOSTILE
    return UNKNOWN


def _foreign(state: Any) -> list[Any]:
    """Every unit and city in the state that is not the player's own."""
    items = list(getattr(state, "units", None) or []) + list(getattr(state, "cities", None) or [])
    return [item for item in items if item.owner_id != state.player_id]


def owner_relations(state: Any) -> dict[int, str]:
    """Owner id -> ``hostile`` / ``not_hostile`` / ``unknown`` for every foreign owner present.

    An owner the mod said nothing about is unknown.
    """
    entries: dict[Any, Any] = {}
    for entry in getattr(state, "other_players", None) or []:
        if isinstance(entry, Mapping):
            entries[entry.get("id")] = entry
    return {
        owner: _relation(entries.get(owner))
        for owner in {item.owner_id for item in _foreign(state)}
    }


def partition(state: Any) -> tuple[set[int], dict[str, int]]:
    """The hostile owners, and how many units and cities of the others are left out."""
    relations = owner_relations(state)
    left_out = {NOT_HOSTILE: 0, UNKNOWN: 0}
    for item in _foreign(state):
        relation = relations[item.owner_id]
        if relation != HOSTILE:
            left_out[relation] += 1
    return {owner for owner, relation in relations.items() if relation == HOSTILE}, left_out


def shown_to_the_agent(state: Any) -> tuple[Any, dict[str, Any] | None]:
    """The state with only the player's own and hostile owners' units and cities, and the counts.

    Returns ``(state, None)`` unchanged for a state from an older mod.
    """
    if getattr(state, "observes_others", None) is None:
        return state, None
    hostile, left_out = partition(state)
    keep = hostile | {state.player_id}
    units = [unit for unit in state.units if unit.owner_id in keep]
    cities = [city for city in state.cities if city.owner_id in keep]
    counts = {
        "observes_others": bool(state.observes_others),
        "hostile_units": sum(1 for unit in units if unit.owner_id != state.player_id),
        "hostile_cities": sum(1 for city in cities if city.owner_id != state.player_id),
        "left_out_not_hostile": left_out[NOT_HOSTILE],
        "left_out_unknown": left_out[UNKNOWN],
    }
    return replace(state, units=units, cities=cities), counts
