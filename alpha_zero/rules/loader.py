"""Strict loading of a rule-set file, and its canonical identity.

``yaml.safe_load`` alone keeps the last of two duplicate keys and reads YAML 1.1
booleans (``yes``, ``on``) where a number was meant. This loader rejects a
duplicate key, an unknown key, and a float, boolean or negative number where a
non-negative integer is expected. It also rejects what would make two spellings
of one file disagree: integers that are not plain decimal (``020`` is octal 16 to
PyYAML, ``1_000`` is 1000), anchors, aliases, merge keys and explicit tags.
Every error names the offending key by path, or the line.

Identity is ``content_sha256``: the sha256 of the canonical JSON of the *parsed*
rule set. Line endings and key order cannot move it; any value can. Two choices
keep a pin stable over time: the rule set's ``id`` is not part of the hashed
content (freezing a draft under a new id keeps its hash), and a field equal to
its default is left out (a later optional field does not move old hashes).
Declared order of techs, units and buildings is part of the content, because
registries index by it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.rules.spec import (
    MODIFIER_SCOPES,
    MODIFIER_YIELDS,
    SCHEMA_VERSION,
    BuildingSpec,
    ModifierSpec,
    RulesetSpec,
    TechSpec,
    UnitSpec,
    VictorySpec,
)


class RulesetLoadError(ValueError):
    """A rule-set file is not well formed. The message names the offending key."""


#: Ids of techs, units and buildings: ASCII, so no Unicode normalisation question.
ITEM_ID = re.compile(r"[a-z][a-z0-9_]*")
#: Ids of rule sets, e.g. ``science-v1-draft``. Never contains a dot or a slash,
#: so the CLI can tell an id from a path.
RULESET_ID = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_PLAIN_DECIMAL = re.compile(r"-?(0|[1-9][0-9]*)")
_INT_TAG = "tag:yaml.org,2002:int"
_MERGE_TAG = "tag:yaml.org,2002:merge"
#: Terrain names a ``worked_terrain`` modifier may refer to (MicroCiv's terrain types).
TERRAIN_NAMES: tuple[str, ...] = ("plains", "hills", "mountain")


class _StrictLoader(yaml.SafeLoader):
    """``SafeLoader`` that refuses a repeated key, a merge key and a non-decimal integer."""

    def construct_yaml_int(self, node: yaml.ScalarNode) -> int:
        text = str(node.value)
        if not _PLAIN_DECIMAL.fullmatch(text):
            line = node.start_mark.line + 1
            raise RulesetLoadError(f"integer {text!r} at line {line} is not plain decimal")
        return int(text)

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            if key_node.tag == _MERGE_TAG:
                line = key_node.start_mark.line + 1
                raise RulesetLoadError(f"merge key '<<' at line {line} is not allowed")
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                line = key_node.start_mark.line + 1
                raise RulesetLoadError(f"mapping key at line {line} is not a scalar")
            if key in seen:
                line = key_node.start_mark.line + 1
                raise RulesetLoadError(f"duplicate key {key!r} at line {line}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


_StrictLoader.add_constructor(_INT_TAG, _StrictLoader.construct_yaml_int)


def _reject_yaml_features(text: str) -> None:
    """Refuse anchors, aliases and explicit tags before anything is constructed."""
    for event in yaml.parse(text, Loader=yaml.SafeLoader):
        mark = event.start_mark
        line = mark.line + 1 if mark is not None else 0
        anchor = getattr(event, "anchor", None)
        if isinstance(event, yaml.AliasEvent):
            raise RulesetLoadError(f"alias '*{anchor}' at line {line} is not allowed")
        if anchor is not None:
            raise RulesetLoadError(f"anchor '&{anchor}' at line {line} is not allowed")
        tag = getattr(event, "tag", None)
        if tag is not None:
            raise RulesetLoadError(f"explicit tag {tag!r} at line {line} is not allowed")


def _mapping(node: Any, path: str, allowed: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(node, dict):
        raise RulesetLoadError(f"{path}: expected a mapping, got {type(node).__name__}")
    for key in node:
        if key not in allowed:
            raise RulesetLoadError(f"{path}: unknown key {key!r}")
    return node


def _sequence(node: Any, path: str) -> list[Any]:
    if node is None:
        return []
    if not isinstance(node, list):
        raise RulesetLoadError(f"{path}: expected a list, got {type(node).__name__}")
    return node


def _int(node: Any, path: str) -> int:
    # bool is a subclass of int, so the exact type is checked.
    if type(node) is not int:
        raise RulesetLoadError(f"{path}: expected an integer, got {type(node).__name__}")
    if node < 0:
        raise RulesetLoadError(f"{path}: expected a non-negative integer, got {node}")
    return node


def _str(node: Any, path: str) -> str:
    if type(node) is not str or not node:
        raise RulesetLoadError(f"{path}: expected a non-empty string, got {type(node).__name__}")
    return node


def _item_id(node: Any, path: str) -> str:
    name = _str(node, path)
    if not ITEM_ID.fullmatch(name):
        raise RulesetLoadError(f"{path}: id {name!r} must match {ITEM_ID.pattern}")
    return name


def _required(node: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in node:
        raise RulesetLoadError(f"{path}: missing key {key!r}")
    return node[key]


def _names(node: Any, path: str) -> tuple[str, ...]:
    return tuple(_item_id(item, f"{path}[{i}]") for i, item in enumerate(_sequence(node, path)))


def _modifier(node: Any, path: str) -> ModifierSpec:
    raw = _mapping(node, path, ("yield", "amount", "per", "ref"))
    yield_ = _str(_required(raw, "yield", path), f"{path}.yield")
    if yield_ not in MODIFIER_YIELDS:
        raise RulesetLoadError(f"{path}.yield: {yield_!r} is not one of {MODIFIER_YIELDS}")
    per = _str(raw.get("per", "city"), f"{path}.per")
    if per not in MODIFIER_SCOPES:
        raise RulesetLoadError(f"{path}.per: {per!r} is not one of {MODIFIER_SCOPES}")
    ref = raw.get("ref")
    if ref is not None:
        ref = _item_id(ref, f"{path}.ref")
    if (per == "city") != (ref is None):
        raise RulesetLoadError(f"{path}.ref: required unless per is 'city', and only then")
    if per == "worked_terrain" and ref not in TERRAIN_NAMES:
        raise RulesetLoadError(f"{path}.ref: terrain {ref!r} is not one of {TERRAIN_NAMES}")
    return ModifierSpec(
        yield_=yield_,
        amount=_int(_required(raw, "amount", path), f"{path}.amount"),
        per=per,
        ref=ref,
    )


def _modifiers(node: Any, path: str) -> tuple[ModifierSpec, ...]:
    return tuple(_modifier(item, f"{path}[{i}]") for i, item in enumerate(_sequence(node, path)))


def _tech(node: Any, path: str) -> TechSpec:
    raw = _mapping(
        node,
        path,
        (
            "id",
            "era",
            "cost",
            "prerequisites",
            "unlocks_units",
            "unlocks_buildings",
            "modifiers",
            "boost_percent",
        ),
    )
    boost = _int(raw.get("boost_percent", 0), f"{path}.boost_percent")
    if boost > 100:
        raise RulesetLoadError(f"{path}.boost_percent: {boost} is above 100")
    return TechSpec(
        id=_item_id(_required(raw, "id", path), f"{path}.id"),
        era=_int(_required(raw, "era", path), f"{path}.era"),
        cost=_int(_required(raw, "cost", path), f"{path}.cost"),
        prerequisites=_names(raw.get("prerequisites"), f"{path}.prerequisites"),
        unlocks_units=_names(raw.get("unlocks_units"), f"{path}.unlocks_units"),
        unlocks_buildings=_names(raw.get("unlocks_buildings"), f"{path}.unlocks_buildings"),
        modifiers=_modifiers(raw.get("modifiers"), f"{path}.modifiers"),
        boost_percent=boost,
    )


def _unit(node: Any, path: str) -> UnitSpec:
    raw = _mapping(node, path, ("id", "cost", "strength", "moves"))
    return UnitSpec(
        id=_item_id(_required(raw, "id", path), f"{path}.id"),
        cost=_int(_required(raw, "cost", path), f"{path}.cost"),
        strength=_int(_required(raw, "strength", path), f"{path}.strength"),
        moves=_int(_required(raw, "moves", path), f"{path}.moves"),
    )


def _building(node: Any, path: str) -> BuildingSpec:
    raw = _mapping(node, path, ("id", "cost", "modifiers"))
    return BuildingSpec(
        id=_item_id(_required(raw, "id", path), f"{path}.id"),
        cost=_int(_required(raw, "cost", path), f"{path}.cost"),
        modifiers=_modifiers(raw.get("modifiers"), f"{path}.modifiers"),
    )


def _victory(node: Any, path: str) -> VictorySpec:
    if node is None:
        return VictorySpec()
    raw = _mapping(node, path, ("capstone_tech",))
    capstone = raw.get("capstone_tech")
    if capstone is not None:
        capstone = _item_id(capstone, f"{path}.capstone_tech")
    return VictorySpec(capstone_tech=capstone)


def parse_ruleset(text: str) -> RulesetSpec:
    """Build a :class:`RulesetSpec` from YAML text, strictly.

    Raises:
        RulesetLoadError: on a duplicate or unknown key, a missing key, or a
            value of the wrong type. Structure (cycles, unresolved names) is
            :func:`alpha_zero.rules.validate.validate`'s job, not this one's.
    """
    try:
        _reject_yaml_features(text)
        # _StrictLoader derives from SafeLoader: no arbitrary object construction.
        loader = _StrictLoader(text)
        try:
            document = loader.get_single_data()
        finally:
            loader.dispose()
    except yaml.YAMLError as exc:
        raise RulesetLoadError(f"not valid YAML: {exc}") from exc
    raw = _mapping(
        document, "$", ("schema_version", "id", "techs", "units", "buildings", "victory")
    )
    version = _int(_required(raw, "schema_version", "$"), "$.schema_version")
    if version != SCHEMA_VERSION:
        raise RulesetLoadError(f"$.schema_version: {version} is not {SCHEMA_VERSION}")
    ruleset_id = _str(_required(raw, "id", "$"), "$.id")
    if not RULESET_ID.fullmatch(ruleset_id):
        raise RulesetLoadError(f"$.id: {ruleset_id!r} must match {RULESET_ID.pattern}")
    return RulesetSpec(
        id=ruleset_id,
        techs=tuple(
            _tech(item, f"$.techs[{i}]")
            for i, item in enumerate(_sequence(raw.get("techs"), "$.techs"))
        ),
        units=tuple(
            _unit(item, f"$.units[{i}]")
            for i, item in enumerate(_sequence(raw.get("units"), "$.units"))
        ),
        buildings=tuple(
            _building(item, f"$.buildings[{i}]")
            for i, item in enumerate(_sequence(raw.get("buildings"), "$.buildings"))
        ),
        victory=_victory(raw.get("victory"), "$.victory"),
        schema_version=version,
    )


def load_ruleset(path: Path) -> RulesetSpec:
    """Read and parse the rule-set file at ``path`` (UTF-8).

    Raises:
        RulesetLoadError: the file is not UTF-8, or :func:`parse_ruleset` refuses it.
        OSError: the file cannot be read.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RulesetLoadError(f"{path}: not UTF-8 text ({exc.reason})") from exc
    return parse_ruleset(text)


def _without_defaults(instance: Any) -> dict[str, Any]:
    """A spec as plain data, leaving out every field equal to its default."""
    out: dict[str, Any] = {}
    for field in dataclasses.fields(instance):
        value = getattr(instance, field.name)
        if field.default is not dataclasses.MISSING and value == field.default:
            continue
        key = field.name.rstrip("_")
        if dataclasses.is_dataclass(value):
            out[key] = _without_defaults(value)
        elif isinstance(value, tuple):
            out[key] = [
                _without_defaults(item) if dataclasses.is_dataclass(item) else item
                for item in value
            ]
        else:
            out[key] = value
    return out


def to_canonical_dict(spec: RulesetSpec) -> dict[str, Any]:
    """The rule set's content as plain data: no ``id``, no field at its default.

    Lists keep their declared order; only mapping keys are sorted (by the JSON
    encoder), because list order is what registries index by.
    """
    content = _without_defaults(spec)
    del content["id"]
    return content


def canonical_json(spec: RulesetSpec) -> str:
    """Canonical JSON of the parsed rule set: sorted keys, no spaces, ASCII."""
    return json.dumps(
        to_canonical_dict(spec), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def content_sha256(spec: RulesetSpec) -> str:
    """sha256 of :func:`canonical_json`: the rule set's content identity."""
    return hashlib.sha256(canonical_json(spec).encode("ascii")).hexdigest()
