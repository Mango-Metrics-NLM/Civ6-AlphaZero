"""Whether a command is one the game offered in a state (GM-B6b review).

The agent loop checks a command policy's answer with this before sending it.
It is a check on the command itself, not on a bridge action index: the search's
translator can name commands outside the bridge's fixed action space.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["is_offered"]

_PREFIXES = ("unit_", "city_")


def _kinds(action: str) -> set[str]:
    """The offer kinds a command's action can appear under (``unit_move`` -> ``move``)."""
    kinds = {action}
    for prefix in _PREFIXES:
        if action.startswith(prefix):
            kinds.add(action[len(prefix) :])
    return kinds


def _target(command: Mapping[str, Any]) -> tuple[Any, Any] | None:
    target = command.get("target")
    if target is None:
        return None
    if isinstance(target, Mapping):
        return (target.get("x"), target.get("y"))
    return (getattr(target, "x", None), getattr(target, "y", None))


def _names(offer: Any) -> set[str]:
    items = list(getattr(offer, "items", None) or []) + list(
        getattr(offer, "item_names", None) or []
    )
    return {str(item) for item in items}


def is_offered(command: Any, legal_actions: Iterable[Any] | None) -> bool:
    """Whether ``legal_actions`` (the mod's offers) contains ``command``.

    An offer matches when its kind is the command's action (with or without the
    ``unit_`` / ``city_`` prefix) and every operand the command names agrees:
    the unit, the city, the target tile among the offered targets, the item
    among the offered items or item names. Anything that cannot be read is not
    offered.
    """
    try:
        data = command if isinstance(command, Mapping) else command.to_dict()
        action = data.get("action_type")
        if not isinstance(action, str) or not action:
            return False
        kinds = _kinds(action)
        target = _target(data)
        item = data.get("item_id")
        for offer in legal_actions or ():
            if getattr(offer, "action_type", None) not in kinds:
                continue
            if data.get("unit_id") is not None and offer.unit_id != data["unit_id"]:
                continue
            if data.get("city_id") is not None and offer.city_id != data["city_id"]:
                continue
            if target is not None and target not in {tuple(t) for t in offer.targets or ()}:
                continue
            if item is not None and str(item) not in _names(offer):
                continue
            return True
    except Exception:
        return False
    return False
