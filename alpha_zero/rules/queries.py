"""Pure queries over a valid rule set. Integer arithmetic only.

The functions assume :func:`alpha_zero.rules.validate.validate` found nothing:
an unknown tech name raises ``KeyError``.
"""

from __future__ import annotations

from collections.abc import Collection

from alpha_zero.rules.spec import RulesetSpec

_PERCENT = 100


def prerequisite_closure(spec: RulesetSpec, tech_id: str) -> tuple[str, ...]:
    """Every tech that must be researched before ``tech_id``, in declared order.

    ``tech_id`` itself is not included.
    """
    by_id = {tech.id: tech for tech in spec.techs}
    needed: set[str] = set()
    stack = list(by_id[tech_id].prerequisites)
    while stack:
        name = stack.pop()
        if name not in needed:
            needed.add(name)
            stack.extend(by_id[name].prerequisites)
    return tuple(tech.id for tech in spec.techs if tech.id in needed)


def available(spec: RulesetSpec, researched: Collection[str]) -> tuple[str, ...]:
    """The unresearched techs whose prerequisites are all researched, in declared order."""
    done = set(researched)
    return tuple(
        tech.id
        for tech in spec.techs
        if tech.id not in done and all(p in done for p in tech.prerequisites)
    )


def research_cost(spec: RulesetSpec, tech_id: str, boosted: bool = False) -> int:
    """Science needed for ``tech_id``; with ``boosted``, less its ``boost_percent``.

    The discount is ``cost * boost_percent // 100``, rounded **down**, so a boost
    never removes more than its percentage (cost 21 at 40%: 21 - 8 = 13).

    Raises:
        KeyError: ``tech_id`` names no tech.
        ValueError: ``boosted`` is set for a tech that declares no boost.
    """
    tech = spec.tech(tech_id)
    if not boosted:
        return tech.cost
    if tech.boost_percent == 0:
        raise ValueError(f"tech {tech_id!r} declares no boost")
    return tech.cost - (tech.cost * tech.boost_percent) // _PERCENT
