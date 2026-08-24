"""Single source of truth for turning ``(wins, draws, losses)`` into a decision.

Three incompatible definitions of "win rate" were live in this repository at the
same time, each gating something different:

===============================================  ==================================
Site                                             Statistic
===============================================  ==================================
``cloud/gcp/arena.py``                           ``wins / (wins + draws + losses)``
``scripts/run_encoder_ab.py``                    ``(wins + 0.5 * draws) / total``
``evaluation/sequential_testing.py``             ``wins / (wins + losses)``
===============================================  ==================================

The first scores **every draw as a loss**. Its ceiling is ``1 - draw_rate``, so
against the conventional ``0.55`` threshold promotion becomes arithmetically
impossible once the draw rate exceeds 45%.

This module makes **score rate** (draws = 0.5) the gating statistic. That choice
is not arbitrary: ``envs/microciv.py`` assigns a terminal reward of ``+1 / -1 / 0``,
placing a draw exactly midway between a win and a loss. Gating on score rate makes
the promotion criterion agree with the objective the agent is trained on.

Variance
--------
A per-game score is **ternary** (``X in {0, 0.5, 1}``), not Bernoulli. Applying a
binomial interval to fractional successes overstates the variance by exactly
``draw_rate / 4`` -- see :func:`score_variance`. That error made the published A/B
interval roughly twice as wide as the data supports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "PromotionDecision",
    "ScoreSummary",
    "decisive_win_rate",
    "effective_n",
    "minimum_detectable_effect",
    "promotion_decision",
    "score_interval",
    "score_rate",
    "score_variance",
    "summarize",
]

# Two-sided normal quantiles for the confidence levels we actually use. Computed
# rather than approximated so there is no repeat of the sign defect that made
# ``sequential_testing._get_z_score`` return a negative quantile.
_Z_TWO_SIDED = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


class PromotionDecision(str, Enum):
    """Outcome of a promotion test.

    ``INCONCLUSIVE`` is distinct from ``REJECT``: "not enough evidence" and
    "evidence of no improvement" are different states, and collapsing them hides
    under-powered evaluations behind an apparent verdict.
    """

    PROMOTE = "promote"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


def _total(wins: int, draws: int, losses: int) -> int:
    if min(wins, draws, losses) < 0:
        raise ValueError(f"counts must be non-negative, got {(wins, draws, losses)}")
    return wins + draws + losses


def score_rate(wins: int, draws: int, losses: int) -> float:
    """Fraction of available points taken, scoring a draw as half a point.

    Returns ``0.5`` for an empty record -- the no-information value, so that a
    zero-game evaluation cannot look like a loss.
    """
    total = _total(wins, draws, losses)
    if total == 0:
        return 0.5
    return (wins + 0.5 * draws) / total


def decisive_win_rate(wins: int, losses: int) -> float | None:
    """``wins / (wins + losses)`` -- a secondary diagnostic, never the gate.

    A score rate driven by draws is not the same result as one driven by wins, so
    report this alongside :func:`score_rate`. Returns ``None`` when no game was
    decisive.
    """
    decisive = wins + losses
    if decisive <= 0:
        return None
    return wins / decisive


def effective_n(wins: int, draws: int, losses: int) -> int:
    """Number of decisive games -- the sample size that carries win/loss signal."""
    _total(wins, draws, losses)
    return wins + losses


def score_variance(wins: int, draws: int, losses: int) -> float:
    """Per-game variance of the ternary score ``X in {0, 0.5, 1}``.

    ``Var(X) = E[X^2] - E[X]^2`` where ``E[X^2] = p_win + 0.25 * p_draw``.

    Draws sit near the mean, so they *reduce* variance on the score scale. The
    binomial formula ``mu * (1 - mu)`` overstates it by exactly ``p_draw / 4``,
    which is why a binomial interval on fractional successes is too wide.
    """
    total = _total(wins, draws, losses)
    if total == 0:
        return 0.0
    p_win = wins / total
    p_draw = draws / total
    mean = p_win + 0.5 * p_draw
    return (p_win + 0.25 * p_draw) - mean * mean


def score_interval(
    wins: int, draws: int, losses: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Two-sided confidence interval for the score rate.

    Normal approximation to the mean of a bounded variable using the plug-in
    ternary variance. Bounds are ordered and clipped to ``[0, 1]``.
    """
    if confidence not in _Z_TWO_SIDED:
        raise ValueError(f"confidence must be one of {sorted(_Z_TWO_SIDED)}, got {confidence}")
    if _total(wins, draws, losses) == 0:
        return (0.0, 1.0)

    z = _Z_TWO_SIDED[confidence]

    # Agresti-Coull style adjustment, generalized to the ternary score: add
    # z^2 / 2 pseudo-wins and z^2 / 2 pseudo-losses. Without it a degenerate
    # record (every game the same outcome) has zero plug-in variance and the
    # normal approximation collapses to a zero-width interval -- a 9-0 record
    # would report perfect certainty. The pseudo-counts are symmetric, so they
    # shrink the estimate toward 0.5 rather than toward a win or a loss, and
    # their influence vanishes as n grows.
    pseudo = z * z / 2.0
    adj_wins = wins + pseudo
    adj_losses = losses + pseudo
    adj_total = adj_wins + draws + adj_losses

    p_win = adj_wins / adj_total
    p_draw = draws / adj_total
    center = p_win + 0.5 * p_draw
    variance = (p_win + 0.25 * p_draw) - center * center

    half = z * math.sqrt(variance / adj_total)
    return (max(0.0, center - half), min(1.0, center + half))


def minimum_detectable_effect(
    n_games: int, draw_rate: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest score-rate difference detectable at ``alpha`` with ``power``.

    ``MDE = (z_alpha + z_power) * sigma / sqrt(n)`` with ``sigma^2 = (1 - d) / 4``
    evaluated at parity. Note this *grows* as the draw rate falls: at fixed ``n``,
    converting draws into near-noise decisive results widens the MDE rather than
    narrowing it.
    """
    if n_games <= 0:
        raise ValueError(f"n_games must be positive, got {n_games}")
    if not 0.0 <= draw_rate <= 1.0:
        raise ValueError(f"draw_rate must be in [0, 1], got {draw_rate}")
    z_alpha = {0.05: 1.6448536269514722, 0.01: 2.3263478740408408}.get(alpha)
    z_power = {0.80: 0.8416212335729143, 0.90: 1.2815515655446004}.get(power)
    if z_alpha is None or z_power is None:
        raise ValueError(f"unsupported alpha={alpha} / power={power}")
    sigma = math.sqrt((1.0 - draw_rate) / 4.0)
    return (z_alpha + z_power) * sigma / math.sqrt(n_games)


@dataclass(frozen=True)
class ScoreSummary:
    """Everything a promotion decision should record, so it can be re-audited."""

    wins: int
    draws: int
    losses: int
    total_games: int
    score_rate: float
    draw_rate: float
    decisive_games: int
    decisive_win_rate: float | None
    ci_low: float
    ci_high: float
    confidence: float

    def to_dict(self) -> dict[str, float | int | None]:
        """Serialize for artifacts and experiment tracking."""
        return {
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "total_games": self.total_games,
            "score_rate": self.score_rate,
            "draw_rate": self.draw_rate,
            "decisive_games": self.decisive_games,
            "decisive_win_rate": self.decisive_win_rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
        }


def summarize(wins: int, draws: int, losses: int, confidence: float = 0.95) -> ScoreSummary:
    """Build a :class:`ScoreSummary` from raw counts."""
    total = _total(wins, draws, losses)
    low, high = score_interval(wins, draws, losses, confidence)
    return ScoreSummary(
        wins=wins,
        draws=draws,
        losses=losses,
        total_games=total,
        score_rate=score_rate(wins, draws, losses),
        draw_rate=(draws / total) if total else 0.0,
        decisive_games=effective_n(wins, draws, losses),
        decisive_win_rate=decisive_win_rate(wins, losses),
        ci_low=low,
        ci_high=high,
        confidence=confidence,
    )


def promotion_decision(
    wins: int,
    draws: int,
    losses: int,
    threshold: float = 0.5,
    confidence: float = 0.95,
    min_games: int = 1,
) -> PromotionDecision:
    """Decide promotion from the confidence bound, not the point estimate.

    Promotes when the lower bound of the score-rate interval clears ``threshold``;
    rejects when the upper bound falls below it; otherwise reports
    ``INCONCLUSIVE``. A point-estimate gate promotes on noise at small ``n``,
    which is what a raw ``score >= 0.55`` comparison does.

    ``threshold`` defaults to ``0.5`` ("better than the champion") rather than the
    inherited ``0.55``: that constant comes from AlphaGo Zero, where Go with komi
    7.5 is draw-free by construction, and a threshold-gated test has zero power
    *at* its own threshold by definition.
    """
    total = _total(wins, draws, losses)
    if total < max(1, min_games):
        return PromotionDecision.INCONCLUSIVE
    low, high = score_interval(wins, draws, losses, confidence)
    if low > threshold:
        return PromotionDecision.PROMOTE
    if high < threshold:
        return PromotionDecision.REJECT
    return PromotionDecision.INCONCLUSIVE
