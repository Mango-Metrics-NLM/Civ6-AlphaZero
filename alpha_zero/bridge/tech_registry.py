"""Slot numbers for technologies and civics on the real-game bridge (GM-B3).

The full-game encoder has a position per technology and per civic, and the
action space has a research action per technology and per civic. Both need the
same answer to "which number is Mining?", and it must be the answer the network
was trained with. That answer comes from a game rule set
(:mod:`alpha_zero.rules`): a technology's slot is its position in the rule set,
through :func:`alpha_zero.rules.build_registry`.

The rules kernel has no civic tree yet (it arrives with the ``civics-v1`` rule
set), so civic slots come from an ordered list given beside the rule set: the
``Civics`` table of a GameInfo export, in row order. Without one, civics have no
slots.

Nothing here is on by default. The converters ask :func:`resolve_registry`,
which reads two environment variables:

``CIV6AZ_BRIDGE_RULESET``   a built-in rule-set id, or the path of a rule-set file
``CIV6AZ_BRIDGE_GAMEINFO``  optional: the path of a GameInfo export, for the civic order

A rule set that is named but cannot be loaded is an error, never "no registry":
a fallback that looks like success would hide a misconfigured run.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from alpha_zero.rules import (
    RulesetSpec,
    build_registry,
    builtin_ids,
    ensure_valid,
    load_builtin,
    load_ruleset,
)

RULESET_ENV = "CIV6AZ_BRIDGE_RULESET"
GAMEINFO_ENV = "CIV6AZ_BRIDGE_GAMEINFO"

_TECH_PREFIX = "tech_"
_CIVIC_PREFIX = "civic_"
_CIVICS_TABLE = "Civics"
_CIVIC_TYPE_COLUMN = "CivicType"


class ResearchRegistryError(ValueError):
    """The research registry was asked for and cannot be built."""


def _slot_map(ids: tuple[str, ...]) -> dict[str, int]:
    return {name: slot for slot, name in enumerate(ids)}


def _lookup(slots: Mapping[str, int], type_name: Any, prefix: str) -> int | None:
    """Slot of a game type name: its lowercase form, else that form without ``prefix``."""
    if not isinstance(type_name, str) or not type_name:
        return None
    lowered = type_name.lower()
    slot = slots.get(lowered)
    if slot is None and lowered.startswith(prefix):
        slot = slots.get(lowered[len(prefix) :])
    return slot


def _type_name(ids: tuple[str, ...], slot: Any, prefix: str) -> str | None:
    """Game type name of a slot: upper case, with ``prefix`` whether or not the id has it."""
    if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < len(ids):
        return None
    name: str = ids[slot]
    if not name.startswith(prefix):
        name = prefix + name
    return name.upper()


@dataclass(frozen=True)
class ResearchRegistry:
    """Which slot each technology and civic has, for one rule set.

    Attributes:
        ruleset_id: The rule set the technology slots come from.
        fingerprint: sha256 over the rule set's own registry fingerprint and
            the civic list. Two sides that agree on it agree on every slot.
        tech_ids: Rule-set technology ids by slot.
        civic_ids: Civic ids by slot (lowercase); empty when no civic list was given.
    """

    ruleset_id: str
    fingerprint: str
    tech_ids: tuple[str, ...]
    civic_ids: tuple[str, ...] = ()
    _tech_slots: dict[str, int] = field(init=False, repr=False, compare=False, hash=False)
    _civic_slots: dict[str, int] = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_tech_slots", _slot_map(self.tech_ids))
        object.__setattr__(self, "_civic_slots", _slot_map(self.civic_ids))

    def describe(self) -> str:
        """One line for a log: the rule set, the start of the fingerprint, the slot counts."""
        return (
            f"rule set {self.ruleset_id!r} (fingerprint {self.fingerprint[:12]}): "
            f"{len(self.tech_ids)} technology slots, {len(self.civic_ids)} civic slots"
        )

    def tech_slot(self, type_name: Any) -> int | None:
        """Slot of a technology named as the game names it (``"TECH_MINING"``), or None."""
        return _lookup(self._tech_slots, type_name, _TECH_PREFIX)

    def civic_slot(self, type_name: Any) -> int | None:
        """Slot of a civic named as the game names it (``"CIVIC_CODE_OF_LAWS"``), or None."""
        return _lookup(self._civic_slots, type_name, _CIVIC_PREFIX)

    def tech_type(self, slot: Any) -> str | None:
        """The game's type name for a technology slot, or None when no technology has it."""
        return _type_name(self.tech_ids, slot, _TECH_PREFIX)

    def civic_type(self, slot: Any) -> str | None:
        """The game's type name for a civic slot, or None when no civic has it."""
        return _type_name(self.civic_ids, slot, _CIVIC_PREFIX)


def _reject_same_game_type(ids: tuple[str, ...], prefix: str, kind: str) -> None:
    """Refuse two ids that are the same thing to the game (``mining`` and ``tech_mining``).

    Both would answer to one game type name, so one slot could never come from
    the game and the way back from a slot to a name would not be one-to-one.
    """
    first_by_game_type: dict[str, str] = {}
    for name in ids:
        game_type = name[len(prefix) :] if name.startswith(prefix) else name
        other = first_by_game_type.setdefault(game_type, name)
        if other != name:
            raise ResearchRegistryError(
                f"{kind} ids {other!r} and {name!r} are the same {kind} to the game"
            )


def civic_ids(civic_types: Iterable[Any]) -> tuple[str, ...]:
    """The civic ids, by slot, that ``civic_types`` gives: lower case, in order.

    This is the whole rule for what may number civic slots. The snapshot
    capture applies it before filing, so a snapshot that is filed is one this
    registry accepts.

    Raises:
        ResearchRegistryError: If one is not a non-empty string, one is given
            twice (compared without case), or two are the same civic to the
            game (``code_of_laws`` and ``civic_code_of_laws``).
    """
    ids: list[str] = []
    seen: set[str] = set()
    for civic in civic_types:
        if not isinstance(civic, str) or not civic:
            raise ResearchRegistryError(f"a civic type must be a non-empty string, got {civic!r}")
        lowered = civic.lower()
        if lowered in seen:
            raise ResearchRegistryError(f"civic given twice: {civic!r}")
        seen.add(lowered)
        ids.append(lowered)
    _reject_same_game_type(tuple(ids), _CIVIC_PREFIX, "civic")
    return tuple(ids)


def civic_types_of(document: Any, where: str) -> tuple[str, ...]:
    """The ``CivicType`` values of a GameInfo document's ``Civics`` table, in row order.

    Raises:
        ResearchRegistryError: If the document is not an object, has no
            ``Civics`` table, or a row has no string ``CivicType``. The
            message starts with ``where``.
    """
    if not isinstance(document, dict):
        raise ResearchRegistryError(f"{where} is not a JSON object")
    rows = document.get(_CIVICS_TABLE)
    if not isinstance(rows, list):
        raise ResearchRegistryError(f"{where} has no {_CIVICS_TABLE} table")

    civics: list[str] = []
    for number, row in enumerate(rows):
        civic = row.get(_CIVIC_TYPE_COLUMN) if isinstance(row, dict) else None
        if not isinstance(civic, str) or not civic:
            raise ResearchRegistryError(
                f"{where}: {_CIVICS_TABLE} row {number} has no {_CIVIC_TYPE_COLUMN}"
            )
        civics.append(civic)
    return tuple(civics)


def from_ruleset(spec: RulesetSpec, civic_types: Iterable[str] = ()) -> ResearchRegistry:
    """Build the registry for ``spec``, with civic slots in the order of ``civic_types``.

    Raises:
        ResearchRegistryError: If a civic is given twice, or one is not a non-empty string.
    """
    rules = build_registry(spec)
    tech_ids = tuple(rules.tech_index)
    _reject_same_game_type(tech_ids, _TECH_PREFIX, "technology")

    civics = list(civic_ids(civic_types))

    payload = json.dumps(
        {"rules": rules.fingerprint, "civics": civics},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return ResearchRegistry(
        ruleset_id=spec.id,
        fingerprint=hashlib.sha256(payload.encode("ascii")).hexdigest(),
        tech_ids=tech_ids,
        civic_ids=tuple(civics),
    )


def civic_types_from_gameinfo(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """The ``CivicType`` values of a GameInfo export's ``Civics`` table, in row order.

    Raises:
        ResearchRegistryError: If the file cannot be read, is not a JSON object,
            has no ``Civics`` table, or a row has no string ``CivicType``.
    """
    source = Path(path).expanduser()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # ValueError covers a decode error, bad UTF-8 and a number too long to convert.
        raise ResearchRegistryError(f"cannot read the GameInfo export {source}: {exc}") from exc
    return civic_types_of(document, f"the GameInfo export {source}")


def _load_spec(ruleset: str) -> RulesetSpec:
    """A built-in rule set by id, or one from a file, validated."""
    try:
        if ruleset in builtin_ids():
            return load_builtin(ruleset)
        source = Path(ruleset).expanduser()
        if not source.is_file():
            raise ResearchRegistryError(
                f"{ruleset!r} is neither a built-in rule set ({', '.join(builtin_ids())}) "
                "nor an existing file"
            )
        spec = load_ruleset(source)
        ensure_valid(spec)
        return spec
    except ResearchRegistryError:
        raise
    except (ValueError, OSError) as exc:
        raise ResearchRegistryError(f"cannot load the rule set {ruleset!r}: {exc}") from exc


def load_registry(
    ruleset: str | os.PathLike[str], gameinfo: str | os.PathLike[str] | None = None
) -> ResearchRegistry:
    """Build the registry from a rule set (built-in id or file) and an optional GameInfo export.

    Raises:
        ResearchRegistryError: If the rule set or the export cannot be loaded.
    """
    spec = _load_spec(os.fspath(ruleset))
    civics = civic_types_from_gameinfo(gameinfo) if gameinfo is not None else ()
    return from_ruleset(spec, civics)


def choose_registry(given: ResearchRegistry | Literal[False] | None) -> ResearchRegistry | None:
    """The registry a converter uses for its ``research_registry`` argument.

    A registry is used as given; ``None`` asks the environment
    (:func:`resolve_registry`); ``False`` means none, without asking.
    """
    if given is False:
        return None
    if given is None:
        return resolve_registry()
    return given


def resolve_registry(env: Mapping[str, str] | None = None) -> ResearchRegistry | None:
    """The registry the environment asks for, or None when it asks for none.

    Raises:
        ResearchRegistryError: If one is asked for and cannot be loaded, or a
            GameInfo export is named without a rule set.
    """
    source = os.environ if env is None else env
    ruleset = source.get(RULESET_ENV, "").strip()
    gameinfo = source.get(GAMEINFO_ENV, "").strip()
    if not ruleset:
        if gameinfo:
            raise ResearchRegistryError(
                f"{GAMEINFO_ENV} is set but {RULESET_ENV} is not: "
                "the civic order means nothing without the rule set"
            )
        return None
    return load_registry(ruleset, gameinfo or None)
