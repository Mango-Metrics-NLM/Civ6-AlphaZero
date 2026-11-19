"""A zero-game pacing model: what a tech tree costs, and how fast it must be paid.

Before any game is played on a rule set, these numbers say whether its costs are
in a sensible range: how much science the capstone needs in total, and what
average science per turn would finish it in a chosen share of the game. It is
arithmetic on the rule set, not a simulation: it knows nothing about how a
player earns science. Calibration sweeps (``docs/GAME_RULES.md`` section 3) are
the measurement; this is the first sanity check before them.

Integer arithmetic only. Shares of the game are integer percent.
"""

from __future__ import annotations

from dataclasses import dataclass

from alpha_zero.rules.queries import prerequisite_closure
from alpha_zero.rules.spec import RulesetSpec

_PERCENT = 100

#: The share of ``max_turns`` in which a scripted science-first player should
#: finish the capstone (band B1 of the probe protocol), in integer percent.
CAPSTONE_BAND_PERCENT: tuple[int, int] = (60, 70)


def full_tree_cost(spec: RulesetSpec) -> int:
    """Science needed to research every tech."""
    return sum(tech.cost for tech in spec.techs)


def prerequisite_set_cost(spec: RulesetSpec, tech_id: str) -> int:
    """Science needed to research ``tech_id``: its own cost plus every prerequisite's.

    This is the cost of the whole prerequisite *set*, each tech counted once. A
    tech with two prerequisite branches needs both, so it is not the cost of one
    path through the tree.
    """
    needed = prerequisite_closure(spec, tech_id)
    return spec.tech(tech_id).cost + sum(spec.tech(name).cost for name in needed)


def turns_to_research(cost: int, science_per_turn: int) -> int:
    """Turns to accumulate ``cost`` at a constant rate, overflow carried: a ceiling.

    Raises:
        ValueError: ``science_per_turn`` is not positive.
    """
    if science_per_turn <= 0:
        raise ValueError(f"science per turn must be positive, got {science_per_turn}")
    return -(-cost // science_per_turn)


def band_science_rates(
    cost: int, max_turns: int, band_percent: tuple[int, int] = CAPSTONE_BAND_PERCENT
) -> tuple[int, int] | None:
    """The constant science rates that finish ``cost`` inside the band, as ``(low, high)``.

    The band is a share of ``max_turns`` in integer percent, both ends included:
    a rate ``r`` is in when ``low% * max_turns <= turns_to_research(cost, r) <=
    high% * max_turns``. Returns ``None`` when no whole rate lands in the band
    (a short game or a cheap tree).

    Rates above ``cost`` are not listed: every one of them also takes one turn,
    so when a one-turn finish is inside the band the high end reads ``cost``.

    Raises:
        ValueError: ``max_turns`` is not positive.
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")
    low_percent, high_percent = band_percent
    rates = [
        rate
        for rate in range(1, max(cost, 1) + 1)
        if low_percent * max_turns
        <= turns_to_research(cost, rate) * _PERCENT
        <= high_percent * max_turns
    ]
    if not rates:
        return None
    return (rates[0], rates[-1])


@dataclass(frozen=True)
class PacingReport:
    """The pacing numbers of one rule set at one game length."""

    ruleset_id: str
    max_turns: int
    full_tree_cost: int
    capstone_tech: str | None
    capstone_cost: int | None
    band_percent: tuple[int, int]
    band_rates: tuple[int, int] | None


def pacing_report(spec: RulesetSpec, max_turns: int) -> PacingReport:
    """Compute the pacing numbers for ``spec`` in a game of ``max_turns`` turns.

    A rule set that declares no capstone has a full-tree cost and nothing else.

    Raises:
        ValueError: ``max_turns`` is not positive.
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")
    capstone = spec.victory.capstone_tech
    capstone_cost = prerequisite_set_cost(spec, capstone) if capstone is not None else None
    return PacingReport(
        ruleset_id=spec.id,
        max_turns=max_turns,
        full_tree_cost=full_tree_cost(spec),
        capstone_tech=capstone,
        capstone_cost=capstone_cost,
        band_percent=CAPSTONE_BAND_PERCENT,
        band_rates=(
            band_science_rates(capstone_cost, max_turns) if capstone_cost is not None else None
        ),
    )
