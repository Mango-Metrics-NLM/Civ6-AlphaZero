"""Structural rules a rule set must satisfy.

Each rule has a stable name, and :func:`validate` reports every violation as an
:class:`Issue` naming its rule, so a caller (and a test) can tell them apart.

Rules:

``duplicate-id``              two techs, two units or two buildings share an id
``unresolved-prerequisite``   a prerequisite names no tech
``prerequisite-cycle``        the prerequisite graph has a cycle
``non-positive-cost``         a tech, unit or building costs zero or less
``unresolved-unlock``         an unlock or a modifier ``ref`` names nothing
``unreachable-capstone``      the declared capstone tech can never be researched
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from alpha_zero.rules.spec import ModifierSpec, RulesetSpec

RULE_DUPLICATE_ID = "duplicate-id"
RULE_UNRESOLVED_PREREQUISITE = "unresolved-prerequisite"
RULE_PREREQUISITE_CYCLE = "prerequisite-cycle"
RULE_NON_POSITIVE_COST = "non-positive-cost"
RULE_UNRESOLVED_UNLOCK = "unresolved-unlock"
RULE_UNREACHABLE_CAPSTONE = "unreachable-capstone"

RULES: tuple[str, ...] = (
    RULE_DUPLICATE_ID,
    RULE_UNRESOLVED_PREREQUISITE,
    RULE_PREREQUISITE_CYCLE,
    RULE_NON_POSITIVE_COST,
    RULE_UNRESOLVED_UNLOCK,
    RULE_UNREACHABLE_CAPSTONE,
)


@dataclass(frozen=True)
class Issue:
    """One violated rule."""

    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


class RulesetValidationError(ValueError):
    """A rule set violates at least one structural rule."""

    def __init__(self, ruleset_id: str, issues: tuple[Issue, ...]) -> None:
        self.ruleset_id = ruleset_id
        self.issues = issues
        lines = "; ".join(str(issue) for issue in issues)
        super().__init__(f"rule set {ruleset_id!r} is invalid: {lines}")


def _duplicate_ids(spec: RulesetSpec) -> list[Issue]:
    issues = []
    for kind, ids in (
        ("tech", [t.id for t in spec.techs]),
        ("unit", [u.id for u in spec.units]),
        ("building", [b.id for b in spec.buildings]),
    ):
        for name, count in Counter(ids).items():
            if count > 1:
                issues.append(Issue(RULE_DUPLICATE_ID, f"{kind} id {name!r} appears {count} times"))
    return issues


def _unresolved_prerequisites(spec: RulesetSpec) -> list[Issue]:
    known = {t.id for t in spec.techs}
    return [
        Issue(RULE_UNRESOLVED_PREREQUISITE, f"tech {tech.id!r} requires unknown tech {name!r}")
        for tech in spec.techs
        for name in tech.prerequisites
        if name not in known
    ]


def _cycles(spec: RulesetSpec) -> list[Issue]:
    """Report the techs left over when prerequisite-free techs are peeled away.

    Only edges between declared techs count, so an unresolved prerequisite is
    not also reported as a cycle. Techs that merely depend on a cycle are
    peeled off the other end, so only the cycle's own members are named.
    """
    known = {t.id for t in spec.techs}
    requires = {t.id: {p for p in t.prerequisites if p in known} for t in spec.techs}
    remaining = set(requires)
    changed = True
    while changed:
        changed = False
        for tech_id in sorted(remaining):
            if not requires[tech_id] & remaining:
                remaining.discard(tech_id)
                changed = True
    # Drop techs nothing in the remainder depends on: downstream of a cycle, not in it.
    changed = True
    while changed:
        changed = False
        for tech_id in sorted(remaining):
            if not any(tech_id in requires[other] for other in remaining):
                remaining.discard(tech_id)
                changed = True
    if not remaining:
        return []
    members = ", ".join(repr(name) for name in sorted(remaining))
    return [Issue(RULE_PREREQUISITE_CYCLE, f"prerequisite cycle among techs {members}")]


def _non_positive_costs(spec: RulesetSpec) -> list[Issue]:
    priced: Iterable[tuple[str, str, int]] = (
        [("tech", t.id, t.cost) for t in spec.techs]
        + [("unit", u.id, u.cost) for u in spec.units]
        + [("building", b.id, b.cost) for b in spec.buildings]
    )
    return [
        Issue(RULE_NON_POSITIVE_COST, f"{kind} {name!r} has cost {cost}; a cost must be positive")
        for kind, name, cost in priced
        if cost <= 0
    ]


def _unresolved_unlocks(spec: RulesetSpec) -> list[Issue]:
    units = {u.id for u in spec.units}
    buildings = {b.id for b in spec.buildings}
    issues = []

    def check_modifiers(owner: str, modifiers: tuple[ModifierSpec, ...]) -> None:
        for modifier in modifiers:
            if modifier.per == "building" and modifier.ref not in buildings:
                issues.append(
                    Issue(
                        RULE_UNRESOLVED_UNLOCK,
                        f"{owner} has a modifier per unknown building {modifier.ref!r}",
                    )
                )

    for tech in spec.techs:
        for name in tech.unlocks_units:
            if name not in units:
                issues.append(
                    Issue(RULE_UNRESOLVED_UNLOCK, f"tech {tech.id!r} unlocks unknown unit {name!r}")
                )
        for name in tech.unlocks_buildings:
            if name not in buildings:
                issues.append(
                    Issue(
                        RULE_UNRESOLVED_UNLOCK,
                        f"tech {tech.id!r} unlocks unknown building {name!r}",
                    )
                )
        check_modifiers(f"tech {tech.id!r}", tech.modifiers)
    for building in spec.buildings:
        check_modifiers(f"building {building.id!r}", building.modifiers)
    return issues


def _unreachable_capstone(spec: RulesetSpec) -> list[Issue]:
    capstone = spec.victory.capstone_tech
    if capstone is None:
        return []
    researched: set[str] = set()
    changed = True
    while changed:
        changed = False
        for tech in spec.techs:
            if tech.id not in researched and all(p in researched for p in tech.prerequisites):
                researched.add(tech.id)
                changed = True
    if capstone in researched:
        return []
    return [
        Issue(
            RULE_UNREACHABLE_CAPSTONE,
            f"capstone tech {capstone!r} can never be researched from an empty start",
        )
    ]


def validate(spec: RulesetSpec) -> tuple[Issue, ...]:
    """Return every structural rule ``spec`` violates; empty when it is valid."""
    issues: list[Issue] = []
    issues += _duplicate_ids(spec)
    issues += _unresolved_prerequisites(spec)
    issues += _cycles(spec)
    issues += _non_positive_costs(spec)
    issues += _unresolved_unlocks(spec)
    issues += _unreachable_capstone(spec)
    return tuple(issues)


def ensure_valid(spec: RulesetSpec) -> RulesetSpec:
    """Return ``spec`` when it is valid; raise :class:`RulesetValidationError` otherwise."""
    issues = validate(spec)
    if issues:
        raise RulesetValidationError(spec.id, issues)
    return spec
