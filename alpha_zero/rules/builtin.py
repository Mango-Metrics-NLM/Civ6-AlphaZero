"""Built-in rule sets: package data, pinned by content hash.

Each entry maps an id to a resource under ``alpha_zero/rules/data/``, the sha256
its content must have, and a status:

``frozen``  immutable. A content change is a new id, never a re-pin. Every
            frozen entry has a row in the log in ``docs/GAME_RULES.md``.
``draft``   a calibration draft. Its id ends in ``-draft``; it may be re-pinned,
            it is never logged, and nothing confirmatory runs on it.

An id ends in ``-draft`` exactly when its status is ``draft``. Loading checks the
hash either way, so an edit to the data without a re-pin raises.

The data is read through ``importlib.resources``, so it works from an installed
wheel (``configs/`` is not installed; this package's ``data/`` is).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

from alpha_zero.rules.loader import content_sha256, parse_ruleset
from alpha_zero.rules.spec import RulesetSpec
from alpha_zero.rules.validate import ensure_valid

STATUS_DRAFT = "draft"
STATUS_FROZEN = "frozen"
DRAFT_SUFFIX = "-draft"


@dataclass(frozen=True)
class BuiltinRuleset:
    """Where a built-in rule set lives and what it must hash to."""

    resource: str
    sha256: str
    status: str


class RulesetHashMismatchError(ValueError):
    """A built-in rule set's data does not hash to its pin."""


#: First-pass costs; ``science-v1`` is reserved for the calibrated tree.
BUILTIN: dict[str, BuiltinRuleset] = {
    "science-v1-draft": BuiltinRuleset(
        resource="science_v1_draft.yaml",
        sha256="0c336c21e209a76a63fd9b70d2c2e8dff18fb0d326562e8b362989e159d18c3c",
        status=STATUS_DRAFT,
    ),
}


def builtin_ids() -> tuple[str, ...]:
    """Ids of the built-in rule sets, in declared order."""
    return tuple(BUILTIN)


def frozen_pins() -> dict[str, str]:
    """``id -> sha256`` of the frozen built-ins: what the rule-set log must list."""
    return {name: entry.sha256 for name, entry in BUILTIN.items() if entry.status == STATUS_FROZEN}


def read_builtin_text(ruleset_id: str) -> str:
    """The raw YAML of a built-in rule set; ``KeyError`` when there is none."""
    entry = BUILTIN[ruleset_id]
    return resources.files("alpha_zero.rules").joinpath("data", entry.resource).read_text("utf-8")


def load_builtin(ruleset_id: str) -> RulesetSpec:
    """Load, hash-check and validate a built-in rule set.

    Raises:
        KeyError: ``ruleset_id`` is not a built-in.
        RulesetHashMismatchError: the data's content hash differs from the pin, or the
            file declares another id than the table.
        RulesetLoadError, RulesetValidationError: the data is malformed.
    """
    entry = BUILTIN[ruleset_id]
    spec = parse_ruleset(read_builtin_text(ruleset_id))
    if spec.id != ruleset_id:
        raise RulesetHashMismatchError(
            f"built-in {ruleset_id!r}: {entry.resource} declares id {spec.id!r}"
        )
    actual = content_sha256(spec)
    if actual != entry.sha256:
        raise RulesetHashMismatchError(
            f"built-in {ruleset_id!r}: {entry.resource} hashes to {actual}, pinned {entry.sha256}. "
            "A frozen rule set is never edited (a change is a new id); a draft must be re-pinned."
        )
    return ensure_valid(spec)
