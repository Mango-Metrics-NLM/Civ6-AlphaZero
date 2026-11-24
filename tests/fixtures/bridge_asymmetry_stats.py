"""Pure helpers for the three-arm asymmetry probe: hex distance, sign test, bootstrap, verdict.

No torch, no simulator: everything here is arithmetic on numbers, so each function has a
fast unit test with hand-made inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import comb
from typing import Any

import numpy as np

__all__ = [
    "VERDICTS",
    "classify",
    "cluster_bootstrap",
    "hex_distance",
    "hex_neighbours",
    "sign_test_p",
]

VERDICTS = ("avoids contact", "no measurable difference", "inconclusive", "vacuous")

#: Smallest contact-rate drop that counts as avoidance, and the equivalence margin.
MIN_EFFECT = 0.10
ALPHA = 0.05


def _cube(x: int, y: int) -> tuple[int, int]:
    """Cube (q, r) of a tile on the game's grid: odd rows sit half a tile to the right."""
    return x - (y - (y & 1)) // 2, y


def hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Steps between two tiles in the game's row-offset ("odd-r") convention."""
    q1, r1 = _cube(*a)
    q2, r2 = _cube(*b)
    dq, dr = q1 - q2, r1 - r2
    return max(abs(dq), abs(dr), abs(dq + dr))


def hex_neighbours(x: int, y: int) -> list[tuple[int, int]]:
    """The six tiles one step away, odd rows shifted right."""
    shift = 1 if y & 1 else 0
    return [
        (x + 1, y),
        (x - 1, y),
        (x - 1 + shift, y + 1),
        (x + shift, y + 1),
        (x - 1 + shift, y - 1),
        (x + shift, y - 1),
    ]


def sign_test_p(first: int, second: int) -> float:
    """Exact two-sided sign test on the two counts of discordant pairs."""
    n = first + second
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(first, second) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def cluster_bootstrap(
    clusters: Mapping[Any, Sequence[float]],
    *,
    seed: int,
    n_boot: int = 4000,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean of all values and its percentile interval, resampling whole clusters.

    Seeded and deterministic: the same clusters and seed give the same interval.
    """
    keys = sorted(clusters, key=repr)
    sums = np.array([float(sum(clusters[k])) for k in keys])
    sizes = np.array([float(len(clusters[k])) for k in keys])
    mean = float(sums.sum() / sizes.sum())
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boots = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(boots, [tail, 1.0 - tail])
    return mean, float(lo), float(hi)


def classify(
    *,
    diff: float,
    ci: tuple[float, float],
    p: float,
    per_init_diffs: Sequence[float],
    own_elimination_in_a: int,
) -> str:
    """One of :data:`VERDICTS` for the a-vs-c comparison.

    ``diff`` is contact rate (c) minus contact rate (a); a positive value means the merged
    rule (a) makes contact less often.
    """
    if own_elimination_in_a <= 0:
        return "vacuous"  # (a) and (c) are identical by construction
    if (
        diff >= MIN_EFFECT
        and p < ALPHA
        and len(per_init_diffs) > 0
        and all(d > 0 for d in per_init_diffs)
    ):
        return "avoids contact"
    if ci[0] >= -MIN_EFFECT and ci[1] <= MIN_EFFECT:
        return "no measurable difference"
    return "inconclusive"
