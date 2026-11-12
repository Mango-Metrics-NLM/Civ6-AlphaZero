"""Single source of truth for turning ``(wins, draws, losses)`` into a decision.

Three incompatible definitions of "win rate" were live in this repository at the
same time, each gating something different:

===============================================  ==================================
Site                                             Statistic
===============================================  ==================================
``cloud/gcp/arena.py``                           ``wins / (wins + draws + losses)``
``scripts/run_encoder_ab.py``                    ``(wins + 0.5 * draws) / total``
``evaluation/sequential_testing.py`` (deleted)   ``wins / (wins + losses)``
===============================================  ==================================

(``sequential_testing.py`` was removed in the 2026-09-02 dead-code pass; its
decisive-rate statistic survives here as :func:`decisive_win_rate`, as a
diagnostic rather than a gate.)

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
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "PromotionDecision",
    "ScoreSummary",
    "decisive_win_rate",
    "effective_n",
    "elo_from_score_rate",
    "elo_interval",
    "games_for_score_rate",
    "interval_hull",
    "minimum_detectable_effect",
    "pair_score_interval",
    "promotion_decision",
    "score_interval",
    "score_rate",
    "score_rate_from_elo",
    "score_variance",
    "summarize",
    "win_rate",
]

# Two-sided normal quantiles for the confidence levels we actually use. Computed
# rather than approximated so there is no repeat of the sign defect that made
# the (since deleted) ``sequential_testing._get_z_score`` return a negative
# quantile.
_Z_TWO_SIDED = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}

#: Confidence levels :func:`score_interval` can evaluate. Exposed so callers can
#: validate configuration up front rather than failing mid-evaluation.
SUPPORTED_CONFIDENCE_LEVELS: tuple[float, ...] = tuple(sorted(_Z_TWO_SIDED))

# One-sided upper quantiles for the significance levels the MDE supports. 0.025
# is the one-sided half of a 95% two-sided interval (same quantile).
_Z_ONE_SIDED = {0.05: 1.6448536269514722, 0.025: 1.959963984540054, 0.01: 2.3263478740408408}


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


def win_rate(wins: int, draws: int, losses: int) -> float:
    """``wins / (wins + draws + losses)`` -- the legacy statistic, draws as losses.

    Kept **only** so that every site which still logs a ``win_rate`` for
    artifact compatibility computes it in one place, next to the draw-aware
    :func:`score_rate` it must be reported alongside. Its ceiling is
    ``1 - draw_rate``; it is never the gate (see the module docstring). Returns
    ``0.0`` for an empty record, matching every legacy call site.
    """
    total = _total(wins, draws, losses)
    if total == 0:
        return 0.0
    return wins / total


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

    The pseudo-count adjustment below applies to *every* degenerate record,
    all-draw ones included: observing only draws does not prove the population
    draw rate is 1.0, so the interval deliberately keeps positive width around
    0.5 rather than claiming perfect certainty from a finite sample. (An
    all-draw record still can never promote -- its lower bound sits below any
    threshold above 0.5.) The width vanishes as ``n`` grows.
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

    ``alpha`` is one-sided: ``0.05``, ``0.025`` (the one-sided reading of a 95%
    two-sided interval, which is how the A1 protocol decides) or ``0.01``.
    """
    if n_games <= 0:
        raise ValueError(f"n_games must be positive, got {n_games}")
    if not 0.0 <= draw_rate <= 1.0:
        raise ValueError(f"draw_rate must be in [0, 1], got {draw_rate}")
    z_alpha = _Z_ONE_SIDED.get(alpha)
    z_power = {0.80: 0.8416212335729143, 0.90: 1.2815515655446004}.get(power)
    if z_alpha is None or z_power is None:
        raise ValueError(f"unsupported alpha={alpha} / power={power}")
    sigma = math.sqrt((1.0 - draw_rate) / 4.0)
    return (z_alpha + z_power) * sigma / math.sqrt(n_games)


def pair_score_interval(
    pair_scores: Sequence[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Two-sided interval for the score rate with seat-swapped pairs as the unit.

    The benchmark plays every map twice, once from each seat. The two games of a
    pair share a map, so they are not independent: :func:`score_interval`, which
    treats games as iid, is too narrow when a map decides both games the same
    way and too wide when the seat decides them. Taking the *pair* as the sample
    unit (the pentanomial or pair-clustered view) uses the observed spread of
    pair scores instead. Each pair score is the mean of its two game scores, so
    it lies in ``{0, 0.25, 0.5, 0.75, 1}``.

    Formula -- the estimator of :func:`score_interval` with the pair as the unit.
    With ``P`` pair scores ``s_1 .. s_P``, ``z`` the two-sided normal quantile
    for ``confidence``, and ``k = z^2 / 2`` pseudo-pairs added at score 1 and
    ``k`` at score 0 (the same Agresti-Coull style adjustment, so a record where
    every pair has one score still has positive width)::

        n' = P + 2k
        m  = (sum_p s_p + k) / n'
        v  = (sum_p s_p^2 + k) / n' - m^2
        CI = [m - z * sqrt(v / n'), m + z * sqrt(v / n')]   clipped to [0, 1]

    When every unit is a single game (scores in ``{0, 0.5, 1}``) this is exactly
    :func:`score_interval`'s ternary-variance interval, which the tests pin.
    Returns ``(0.0, 1.0)`` for no pairs.
    """
    if confidence not in _Z_TWO_SIDED:
        raise ValueError(f"confidence must be one of {sorted(_Z_TWO_SIDED)}, got {confidence}")
    for s in pair_scores:
        if not 0.0 <= s <= 1.0:  # also rejects NaN
            raise ValueError(f"pair score must be in [0, 1], got {s}")
    if not pair_scores:
        return (0.0, 1.0)

    z = _Z_TWO_SIDED[confidence]
    pseudo = z * z / 2.0
    adj_total = len(pair_scores) + 2.0 * pseudo
    center = (math.fsum(pair_scores) + pseudo) / adj_total
    variance = (math.fsum(s * s for s in pair_scores) + pseudo) / adj_total - center * center
    half = z * math.sqrt(max(variance, 0.0) / adj_total)
    return (max(0.0, center - half), min(1.0, center + half))


def interval_hull(*intervals: tuple[float, float]) -> tuple[float, float]:
    """``(min of the lows, max of the highs)`` -- the conservative union of intervals.

    The A1 protocol decides on the hull of the iid and the pair-clustered
    interval, so a verdict does not depend on which variance model is right.
    """
    if not intervals:
        raise ValueError("interval_hull needs at least one interval")
    return (min(low for low, _ in intervals), max(high for _, high in intervals))


# Clip for the Elo transform: a score rate of exactly 0 or 1 maps to an
# infinite rating difference, which no finite record supports.
_ELO_CLIP = 1e-9


def elo_from_score_rate(p: float) -> float:
    """Elo difference implied by an expected score rate ``p``.

    The standard logistic relation ``Elo = -400 * log10(1 / p - 1)``, so that
    ``p = 0.5`` is ``0`` Elo, ``p = 0.52`` is about ``+13.9`` and ``p = 0.55``
    about ``+34.9``. The transform is monotone, which is what lets
    :func:`elo_interval` map a score-rate interval bound-for-bound.

    ``p`` is clipped to ``(0, 1)`` exclusive before the logarithm; the
    endpoints would otherwise be ``-inf``/``+inf``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"score rate must be in [0, 1], got {p}")
    clipped = min(max(p, _ELO_CLIP), 1.0 - _ELO_CLIP)
    return -400.0 * math.log10(1.0 / clipped - 1.0)


def score_rate_from_elo(elo: float) -> float:
    """Inverse of :func:`elo_from_score_rate`: ``1 / (1 + 10 ** (-elo / 400))``."""
    return 1.0 / (1.0 + math.pow(10.0, -elo / 400.0))


def elo_interval(
    wins: int, draws: int, losses: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Confidence interval for the implied Elo difference.

    A monotone transform of :func:`score_interval`, so it inherits that
    interval's ternary variance and pseudo-count behaviour: symmetric records
    give a symmetric interval around 0, and the width shrinks with ``n``.
    """
    low, high = score_interval(wins, draws, losses, confidence)
    return (elo_from_score_rate(low), elo_from_score_rate(high))


def games_for_score_rate(
    target_score_rate: float, draw_rate: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Games needed so that ``target_score_rate`` is the MDE at ``alpha``/``power``.

    Inverts :func:`minimum_detectable_effect`: ``n = ((z_alpha + z_power) *
    sigma / delta) ** 2`` with ``delta = target_score_rate - 0.5`` and
    ``sigma ** 2 = (1 - draw_rate) / 4``. Rounded up. This is the sample size
    for a one-sided superiority test at parity, not for a threshold above it.
    """
    delta = target_score_rate - 0.5
    if delta <= 0.0:
        raise ValueError(f"target_score_rate must exceed 0.5, got {target_score_rate}")
    # Reuse the quantile tables through the forward function at n=1:
    # MDE(1) = (z_alpha + z_power) * sigma.
    unit = minimum_detectable_effect(1, draw_rate, alpha=alpha, power=power)
    return math.ceil((unit / delta) ** 2)


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
