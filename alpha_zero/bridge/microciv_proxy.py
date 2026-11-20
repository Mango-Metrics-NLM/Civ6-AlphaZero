"""How a live game state is stood in for inside the MicroCiv simulator (GM-B5).

The search on the real-game bridge does not search the real game: it searches
``MicroCivEnv``, seeded from the bridge's converted state
(:class:`alpha_zero.bridge.mcts_policy.BridgeMCTSEnv`). The converted state
speaks the full-game encoder's vocabulary (unit categories such as ``melee``
and ``civilian``, terrain such as ``grassland`` with a ``mountains`` feature),
and MicroCiv knows three unit types and three terrain types. Passed through
unchanged, a settler could not found a city and a mountain could be walked
over, because MicroCiv's rules test for its own words.

This module is that translation and nothing else. It adds no rule: every unit
becomes one of MicroCiv's existing unit types or is left out, and every tile
becomes one of its existing terrain types, the impassable one included. What a
rule would be needed for (water, embarking, ranged attacks, builders) belongs
to the simulator rungs of the game-mechanics plan, not here.

It does not make the search usable on a live game. The search's network and
its actions are the simulator's; there is no translator from a simulator
action to a live command yet (GM-B6), so ``civ6_local`` refuses the mode.

Standard library only, so importing it does not load torch or numpy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SIM_WARRIOR = "warrior"
SIM_SETTLER = "settler"
SIM_ARCHER = "archer"
SIMULATOR_UNIT_CLASSES: frozenset[str] = frozenset({SIM_WARRIOR, SIM_SETTLER, SIM_ARCHER})

SIM_PLAINS = "plains"
SIM_HILLS = "hills"
#: The one terrain type MicroCiv refuses to enter.
SIM_MOUNTAIN = "mountain"
SIMULATOR_TERRAIN: frozenset[str] = frozenset({SIM_PLAINS, SIM_HILLS, SIM_MOUNTAIN})

#: The game's name for the unit that founds cities.
_RAW_SETTLER = "UNIT_SETTLER"

#: Converter unit categories (``UnitMapping``) that have a MicroCiv stand-in.
#: A category that is not here has none: naval and air units (MicroCiv has no
#: water and no air), support, religious and great-person units, and civilians
#: other than the settler.
_CLASS_OF_CATEGORY: dict[str, str] = {
    SIM_WARRIOR: SIM_WARRIOR,
    SIM_SETTLER: SIM_SETTLER,
    SIM_ARCHER: SIM_ARCHER,
    "melee": SIM_WARRIOR,
    "cavalry": SIM_WARRIOR,
    "ranged": SIM_ARCHER,
    "siege": SIM_ARCHER,
}

#: Converted terrain types a land unit cannot enter. Water is here because
#: MicroCiv has neither water nor embarking; the only way it can say "a unit
#: does not go there" is its mountain.
_IMPASSABLE_TYPES: frozenset[str] = frozenset({"mountain", "mountains", "coast", "ocean"})
_IMPASSABLE_FEATURES: frozenset[str] = frozenset({"mountain", "mountains", "ice"})
_HILLS = "hills"


def _word(value: Any) -> str:
    """``value`` as a lower-case word, or the empty string when it is not a string."""
    return value.strip().lower() if isinstance(value, str) else ""


def _attr(obj: Any, name: str) -> Any:
    """``obj.name``, or None when reading it fails for any reason."""
    try:
        return getattr(obj, name)
    except Exception:
        return None


def simulator_unit_class(unit_type: Any, raw_type: Any = None) -> str | None:
    """The MicroCiv unit type that stands for a unit, or None when none does.

    Args:
        unit_type: The converter's category for the unit (``"melee"``,
            ``"civilian"``, ...) or a MicroCiv unit type.
        raw_type: The game's own name for the unit (``"UNIT_SETTLER"``), when
            known. The converter's ``civilian`` covers settlers, builders,
            traders and scouts alike, so only this tells a settler apart.

    Returns:
        ``"warrior"``, ``"settler"`` or ``"archer"``; None for a unit MicroCiv
        has nothing for. Never raises.
    """
    if _word(raw_type) == _RAW_SETTLER.lower():
        return SIM_SETTLER
    return _CLASS_OF_CATEGORY.get(_word(unit_type))


def simulator_terrain(tile: Any) -> str:
    """The MicroCiv terrain type of a converted tile.

    Args:
        tile: The converter's tile dict: ``"type"`` (a string) and
            ``"features"`` (a list of strings).

    Returns:
        ``"mountain"`` for a tile a land unit cannot enter (a mountain, water,
        ice), else ``"hills"``, else ``"plains"``. Anything unreadable is
        ``"plains"``. Never raises.
    """
    if not isinstance(tile, Mapping):
        return SIM_PLAINS
    terrain = _word(tile.get("type"))
    raw_features = tile.get("features")
    features = (
        {_word(feature) for feature in raw_features}
        if isinstance(raw_features, (list, tuple, set, frozenset))
        else set()
    )
    if terrain in _IMPASSABLE_TYPES or features & _IMPASSABLE_FEATURES:
        return SIM_MOUNTAIN
    if terrain == _HILLS or _HILLS in features:
        return SIM_HILLS
    return SIM_PLAINS


def raw_unit_types(converted_state: Any) -> dict[tuple[int, int], str]:
    """The game's own unit names, by ``(owner id, unit id)``.

    The converted units carry a category, not the game's name; the name is
    still on the state the converter was given, which it passes along as
    ``_raw_state``.

    Args:
        converted_state: Output of ``StateConverter.convert``.

    Returns:
        ``{(owner_id, id): unit_type}``; empty when there is no raw state to
        read. Units without a name are left out. Never raises.
    """
    # Whatever stands where the raw state or one of its units should be, the
    # caller falls back to the converter's category: a settler is then not told
    # from the other civilians and is left out, which the caller counts.
    try:
        units = list(converted_state["_raw_state"].units)
    except Exception:
        return {}
    names: dict[tuple[int, int], str] = {}
    for unit in units:
        owner, unit_id, name = _attr(unit, "owner_id"), _attr(unit, "id"), _attr(unit, "unit_type")
        if not isinstance(name, str) or not name:
            continue
        if isinstance(owner, bool) or isinstance(unit_id, bool):
            continue
        if isinstance(owner, int) and isinstance(unit_id, int):
            names[(owner, unit_id)] = name
    return names
