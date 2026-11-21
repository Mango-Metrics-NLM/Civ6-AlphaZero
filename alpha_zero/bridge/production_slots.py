"""Which action slot stands for which production item a city was offered.

The live action space has a fixed number of production slots per city and per kind
(unit, building, wonder, district). The game offers a city its choices by type name
(``"UNIT_WARRIOR"``). A slot is an item's position among the offered items of its kind,
in sorted order: every offered item gets a slot (up to the action space's bound), the
mapping is the same in every process, and the command built from a slot carries the
item's name, which is what the mod resolves.

The mapping this replaces hashed a name with Python's ``hash()`` into 200 buckets: the
bucket differed between processes (string hashing is salted), and any item whose bucket
was at or above the action space's bound could never be chosen.

Limit: a slot's meaning depends on what was offered, so slot 0 is not the same item in
two different states. That is enough for a mask and the command built from it; a policy
that is to learn which *item* to build needs stable slots (a production registry, as the
research registry is for technologies), which is not built.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["DEFAULT_KIND", "KINDS", "item_at", "legacy_slot", "offered", "slot_of"]

#: The kinds of production item, as the mod names them in ``item_types``.
KINDS: tuple[str, ...] = ("unit", "building", "wonder", "district")
DEFAULT_KIND = "unit"
#: Buckets of the old name-to-number mapping, kept for :func:`legacy_slot` only.
_LEGACY_BUCKETS = 200


def _sort_key(item: Any) -> tuple[int, Any]:
    """Integers before names, each in its own order."""
    return (0, item) if isinstance(item, int) and not isinstance(item, bool) else (1, str(item))


def offered(
    items: Iterable[Any],
    item_types: Mapping[str, str] | None = None,
    default_type: str | None = DEFAULT_KIND,
) -> dict[str, tuple[Any, ...]]:
    """The offered items grouped by kind, each group sorted and without duplicates."""
    kinds = item_types or {}
    fallback = default_type or DEFAULT_KIND
    groups: dict[str, set[Any]] = {}
    for item in items:
        if item is None or isinstance(item, bool):
            continue
        groups.setdefault(kinds.get(str(item), fallback), set()).add(item)
    return {kind: tuple(sorted(group, key=_sort_key)) for kind, group in groups.items()}


def slot_of(
    item: Any,
    items: Iterable[Any],
    item_types: Mapping[str, str] | None = None,
    default_type: str | None = DEFAULT_KIND,
) -> tuple[str, int] | None:
    """``(kind, slot)`` of an offered item, or None when it was not offered."""
    for kind, group in offered(items, item_types, default_type).items():
        if item in group:
            return kind, group.index(item)
    return None


def item_at(
    kind: str,
    slot: int,
    items: Iterable[Any],
    item_types: Mapping[str, str] | None = None,
    default_type: str | None = DEFAULT_KIND,
) -> Any | None:
    """The offered item of ``kind`` at ``slot``, or None when nothing was offered there."""
    group = offered(items, item_types, default_type).get(kind, ())
    return group[slot] if 0 <= slot < len(group) else None


def legacy_slot(item: Any) -> int | None:
    """The number the old mapping gave an item id, made the same in every process.

    An integer is itself, a numeric string its value, a name a CRC bucket. Kept for
    callers outside the mask (purchases); production slots use :func:`slot_of`.
    """
    if isinstance(item, bool) or item is None:
        return None
    if isinstance(item, int):
        return item
    if isinstance(item, str) and item:
        try:
            return int(item)
        except ValueError:
            return zlib.crc32(item.encode("utf-8")) % _LEGACY_BUCKETS
    return None
