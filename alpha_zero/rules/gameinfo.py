"""Turn an export of the game's own GameInfo tables into a rule-set file.

The real technology tree is the user's own game data. It is never committed:
the bridge can snapshot it from a running game (slice GM-B4), and this module
turns such a snapshot into a rule set the kernel can load, so the same engine
that runs the small built-in tree can run the real one.

Input: one JSON object whose keys are table names and whose values are lists of
rows. Recognised tables and the columns read from them:

``Technologies``       ``TechnologyType``, ``Cost``, ``EraType`` (or integer ``Era``)
``TechnologyPrereqs``  ``Technology``, ``PrereqTech``
``Buildings``          ``BuildingType``, ``Cost``, optional ``PrereqTech``
``Units``              ``UnitType``, ``Cost``, optional ``Combat``, ``BaseMoves``, ``PrereqTech``
``Civics``, ``CivicPrereqs``  accepted and counted, **not imported**: schema
                       version 1 has no civic tree (it arrives with ``civics-v1``).

Every value must be an identifier, an integer or null: the snapshot holds names
and numbers, no text. Rows whose cost is not positive (the game marks unbuildable
entries so) are skipped and counted. No modifier and no victory is imported.

Rules the importer applies, so that nothing is changed silently: two rows that
give the same id (after lowercasing) are an error; a repeated prerequisite row is
dropped; a technology row gives ``Era`` or ``EraType``, not both; a reference to
a technology that is unknown, or was skipped for its cost, is an error naming the
table, row and column. A negative ``Combat`` or ``BaseMoves`` (the game's "none")
is imported as 0.

The output must lie outside the repository checkout. A leaf package cannot use
``alpha_zero.utils.data_root``, so this module reads ``CIV6AZ_DATA_ROOT`` from
the environment itself and finds the checkout as the nearest ancestor of this
package that holds a ``pyproject.toml``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.rules.loader import RULESET_ID, parse_ruleset, to_canonical_dict
from alpha_zero.rules.spec import SCHEMA_VERSION
from alpha_zero.rules.validate import ensure_valid

DATA_ROOT_ENV = "CIV6AZ_DATA_ROOT"
DEFAULT_SUBDIR = "gameinfo"
DEFAULT_RULESET_ID = "gameinfo-import"

IMPORTED_TABLES: tuple[str, ...] = ("Technologies", "TechnologyPrereqs", "Buildings", "Units")
COUNTED_TABLES: tuple[str, ...] = ("Civics", "CivicPrereqs")
#: Tables that must be present (possibly empty): without them an incomplete
#: capture would silently change the tree.
REQUIRED_TABLES: tuple[str, ...] = ("Technologies", "TechnologyPrereqs")

#: The game's era identifiers, in order; an integer ``Era`` column is also accepted.
ERA_ORDER: tuple[str, ...] = (
    "ERA_ANCIENT",
    "ERA_CLASSICAL",
    "ERA_MEDIEVAL",
    "ERA_RENAISSANCE",
    "ERA_INDUSTRIAL",
    "ERA_MODERN",
    "ERA_ATOMIC",
    "ERA_INFORMATION",
    "ERA_FUTURE",
)

_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


class GameInfoError(ValueError):
    """The export is not in the expected format. The message names the place."""


class RefusedPathError(ValueError):
    """The output path is not allowed (inside the checkout, or no data root)."""


@dataclass(frozen=True)
class ImportSummary:
    """What an import read, skipped and wrote."""

    ruleset_id: str
    techs: int
    units: int
    buildings: int
    skipped_non_positive_cost: int
    civics_not_imported: int
    text: str


def checkout_root() -> Path | None:
    """The repository root this package sits in, or ``None`` (an installed wheel)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def enclosing_checkout(path: Path) -> Path | None:
    """The git checkout ``path`` lies in, or ``None``.

    This does not depend on where the package is installed: from an installed
    wheel :func:`checkout_root` knows no repository, yet the destination may
    still be inside one (the working tree the user is standing in, say).
    """
    for parent in path.parents:
        if (parent / ".git").exists():
            return parent
    return None


def resolve_output(out: str | None, ruleset_id: str) -> Path:
    """Where the rule-set file goes: ``out``, or ``<CIV6AZ_DATA_ROOT>/gameinfo/<id>.yaml``.

    Raises:
        RefusedPathError: no ``out`` and no data root, or the path resolves to a
            place inside this repository's checkout or inside any git checkout.
    """
    if out is not None:
        path = Path(out).expanduser()
    else:
        root = os.environ.get(DATA_ROOT_ENV, "").strip()
        if not root:
            raise RefusedPathError(f"no --out given and {DATA_ROOT_ENV} is not set")
        # `~` is expanded, as alpha_zero.utils.data_root does for the same variable.
        path = Path(root).expanduser() / DEFAULT_SUBDIR / f"{ruleset_id}.yaml"
    resolved = path.resolve()
    repo = checkout_root()
    if repo is None or not resolved.is_relative_to(repo):
        repo = enclosing_checkout(resolved)
    if repo is not None:
        raise RefusedPathError(
            f"{resolved} is inside the repository checkout ({repo}); game data is never written there"
        )
    return resolved


def _item_id(value: Any, where: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise GameInfoError(f"{where}: expected an identifier, got {value!r}")
    return value.lower()


def _integer(value: Any, where: str) -> int:
    if type(value) is not int:
        raise GameInfoError(f"{where}: expected an integer, got {value!r}")
    return value


def _rows(document: dict[str, Any], table: str) -> list[dict[str, Any]]:
    rows = document.get(table, [])
    if not isinstance(rows, list):
        raise GameInfoError(f"{table}: expected a list of rows")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise GameInfoError(f"{table}[{index}]: expected an object")
        for column, value in row.items():
            if value is None or type(value) is int:
                continue
            if type(value) is str and _IDENTIFIER.fullmatch(value):
                continue
            raise GameInfoError(
                f"{table}[{index}].{column}: expected an identifier, an integer or null, "
                f"got {value!r}"
            )
    return rows


def _required(row: dict[str, Any], column: str, where: str) -> Any:
    if row.get(column) is None:
        raise GameInfoError(f"{where}: missing column {column!r}")
    return row[column]


def _era(row: dict[str, Any], where: str) -> int:
    if row.get("Era") is not None:
        if row.get("EraType") is not None:
            raise GameInfoError(f"{where}: give Era or EraType, not both")
        era = _integer(row["Era"], f"{where}.Era")
        if era < 0:
            raise GameInfoError(f"{where}.Era: expected a non-negative integer, got {era}")
        return era
    era_type = _required(row, "EraType", where)
    if era_type not in ERA_ORDER:
        raise GameInfoError(f"{where}.EraType: unknown era {era_type!r}")
    return ERA_ORDER.index(era_type) + 1


def convert(document: Any, ruleset_id: str = DEFAULT_RULESET_ID) -> ImportSummary:
    """Convert a parsed export into rule-set YAML text that loads and validates.

    Raises:
        GameInfoError: the export is malformed; the message names the table and row.
        RulesetLoadError, RulesetValidationError: the converted rule set is not
            valid (an unknown prerequisite or a cycle in the game data, say).
    """
    if not RULESET_ID.fullmatch(ruleset_id):
        raise GameInfoError(f"rule-set id {ruleset_id!r} must match {RULESET_ID.pattern}")
    if not isinstance(document, dict):
        raise GameInfoError("expected a JSON object of tables")
    for table in document:
        if table not in IMPORTED_TABLES + COUNTED_TABLES:
            raise GameInfoError(f"unknown table {table!r}")
    for table in REQUIRED_TABLES:
        if table not in document:
            raise GameInfoError(
                f"missing table {table!r}: an export without it would import every technology "
                "with no prerequisites; write an empty list if the table really is empty"
            )

    skipped = 0
    skipped_techs: set[str] = set()
    techs: dict[str, dict[str, Any]] = {}

    def known_tech(value: Any, where: str) -> str:
        """An id that names an imported technology; ``where`` names table, row and column."""
        tech_id = _item_id(value, where)
        if tech_id in skipped_techs:
            raise GameInfoError(
                f"{where}: technology {tech_id!r} was skipped for its non-positive cost"
            )
        if tech_id not in techs:
            raise GameInfoError(f"{where}: unknown technology {tech_id!r}")
        return tech_id

    def new_id(seen: set[str], value: Any, where: str) -> str:
        item_id = _item_id(value, where)
        if item_id in seen:
            raise GameInfoError(f"{where}: duplicate id {item_id!r}")
        seen.add(item_id)
        return item_id

    tech_ids: set[str] = set()
    for index, row in enumerate(_rows(document, "Technologies")):
        where = f"Technologies[{index}]"
        cost = _integer(_required(row, "Cost", where), f"{where}.Cost")
        tech_id = new_id(
            tech_ids, _required(row, "TechnologyType", where), f"{where}.TechnologyType"
        )
        if cost <= 0:
            skipped += 1
            skipped_techs.add(tech_id)
            continue
        techs[tech_id] = {
            "id": tech_id,
            "era": _era(row, where),
            "cost": cost,
            "prerequisites": [],
            "unlocks_units": [],
            "unlocks_buildings": [],
        }
    for index, row in enumerate(_rows(document, "TechnologyPrereqs")):
        where = f"TechnologyPrereqs[{index}]"
        tech_id = known_tech(_required(row, "Technology", where), f"{where}.Technology")
        prerequisite = known_tech(_required(row, "PrereqTech", where), f"{where}.PrereqTech")
        if prerequisite not in techs[tech_id]["prerequisites"]:
            techs[tech_id]["prerequisites"].append(prerequisite)

    def unlock(kind: str, item_id: str, row: dict[str, Any], where: str) -> None:
        if row.get("PrereqTech") is None:
            return
        techs[known_tech(row["PrereqTech"], f"{where}.PrereqTech")][kind].append(item_id)

    building_ids: set[str] = set()
    buildings: list[dict[str, Any]] = []
    for index, row in enumerate(_rows(document, "Buildings")):
        where = f"Buildings[{index}]"
        cost = _integer(_required(row, "Cost", where), f"{where}.Cost")
        building_id = new_id(
            building_ids, _required(row, "BuildingType", where), f"{where}.BuildingType"
        )
        if cost <= 0:
            skipped += 1
            continue
        buildings.append({"id": building_id, "cost": cost})
        unlock("unlocks_buildings", building_id, row, where)

    unit_ids: set[str] = set()
    units: list[dict[str, Any]] = []
    for index, row in enumerate(_rows(document, "Units")):
        where = f"Units[{index}]"
        cost = _integer(_required(row, "Cost", where), f"{where}.Cost")
        unit_id = new_id(unit_ids, _required(row, "UnitType", where), f"{where}.UnitType")
        if cost <= 0:
            skipped += 1
            continue
        units.append(
            {
                "id": unit_id,
                "cost": cost,
                # The game writes a negative number for "none".
                "strength": max(0, _integer(row.get("Combat") or 0, f"{where}.Combat")),
                "moves": max(0, _integer(row.get("BaseMoves") or 0, f"{where}.BaseMoves")),
            }
        )
        unlock("unlocks_units", unit_id, row, where)

    civics = len(_rows(document, "Civics"))
    _rows(document, "CivicPrereqs")

    raw = {
        "schema_version": SCHEMA_VERSION,
        "id": ruleset_id,
        "techs": list(techs.values()),
        "units": units,
        "buildings": buildings,
    }
    # Round-trip through the strict loader and the validator: what is written is
    # exactly what the kernel will accept, in its canonical field layout.
    spec = ensure_valid(parse_ruleset(yaml.safe_dump(raw, sort_keys=False)))
    canonical = {"schema_version": SCHEMA_VERSION, "id": ruleset_id, **to_canonical_dict(spec)}
    text = (
        "# Imported from a GameInfo export by `python -m alpha_zero.rules import-gameinfo`.\n"
        "# Game data: keep it outside the repository.\n"
        + yaml.safe_dump(canonical, sort_keys=False)
    )
    ensure_valid(parse_ruleset(text))
    return ImportSummary(
        ruleset_id=ruleset_id,
        techs=len(spec.techs),
        units=len(spec.units),
        buildings=len(spec.buildings),
        skipped_non_positive_cost=skipped,
        civics_not_imported=civics,
        text=text,
    )


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``json`` keeps the last of two equal keys; a GameInfo export may not repeat one."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise GameInfoError(f"duplicate key {key!r}")
        seen[key] = value
    return seen


def import_gameinfo(
    source: Path, out: str | None, ruleset_id: str = DEFAULT_RULESET_ID
) -> tuple[Path, ImportSummary]:
    """Read ``source``, convert it and write the rule-set file.

    Returns the written path and what was imported, skipped and left out.

    The destination is checked **before** anything is read or written, and the
    file is published through a temporary file, so a refusal or a failure leaves
    nothing behind.

    Raises:
        RefusedPathError: see :func:`resolve_output`.
        GameInfoError: the export is malformed or is not JSON.
        OSError: the source cannot be read or the destination cannot be written.
    """
    destination = resolve_output(out, ruleset_id)
    try:
        document = json.loads(
            Path(source).read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys
        )
    except GameInfoError as exc:
        raise GameInfoError(f"{source}: {exc}") from exc
    except ValueError as exc:
        # JSONDecodeError and UnicodeDecodeError are ValueErrors; so is Python's
        # refusal to convert an integer beyond its digit limit.
        raise GameInfoError(f"{source}: not UTF-8 JSON ({exc})") from exc
    summary = convert(document, ruleset_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A fresh, exclusively created temporary file: a predictable sibling name
    # could be a symlink into a checkout, the input file itself, or another
    # import's staging file.
    handle, name = tempfile.mkstemp(
        dir=destination.parent, prefix=destination.name + ".", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(summary.text)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, summary
