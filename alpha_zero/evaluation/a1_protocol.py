"""The registered A1 protocol: its hash, and the positive-control calibration (LF-A1).

``docs/EVALUATION.md`` section 7.1 is the pre-registered protocol for the first
post-fix learning run (A1) and its replication (A1-R). Two things about it must
be computable by code rather than by hand:

* **Its sha256.** The A1 artifact records the hash of the registered text, so a
  later edit to the protocol cannot pass for the protocol the run was judged
  under. The hashed region is everything strictly between the line holding
  :data:`PROTOCOL_BEGIN` and the line holding :data:`PROTOCOL_END`, with CRLF
  normalised to LF, encoded as UTF-8. The hash is never written inside that
  region (it could not be: writing it would change it).
* **The positive-control calibration.** Section 7.1 sizes the positive control
  (heuristic vs random) by the probability that the arm VOIDs the run although
  the heuristic really is stronger: ``P(hull L <= 0.5)`` at the registered game
  count, where ``L`` is the low end of the 95% hull the verdict reads (T11 review
  I2: the hull, not the iid ``ci_low``). :func:`positive_control_void_probability`
  is the registered method: a seeded **pair bootstrap** over T0b's measured
  seat-swapped pairs, so the calibration inherits the real map clustering the
  hull exists for (T13 review I1); the 40/60 branch reads the upper bound of
  :func:`positive_control_void_probability_interval`, a bootstrap over the
  reference itself (T13 rereview N3). :func:`iid_positive_control_void_probability`
  draws games independently; it ignores clustering, so it is reported only as an
  informational lower bound. Both read the hull from the benchmark's own
  :func:`~alpha_zero.evaluation.benchmark_evidence.interval_summary`.

* **The verdict.** :func:`compute_verdict` applies the section 7.1 verdict table
  (VOID, PASS, FAIL, INCONCLUSIVE; the first matching row wins) to an A1 run's
  per-game records. It is the one implementation: ``scripts/verify_release_evidence.py``
  (LF-D8) recomputes every stored verdict with it, and the table itself lives in
  :func:`verdict_from_bounds` so its edges can be tested on exact bounds.

``python scripts/check_learning_trend.py --print-protocol-sha`` prints the hash.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alpha_zero.evaluation.benchmark_evidence import facing_hull_ratio, interval_summary
from alpha_zero.evaluation.scoring import minimum_detectable_effect, summarize

__all__ = [
    "POSITIVE_CONTROL_CONFIDENCE",
    "PRIMARY_CONFIDENCE",
    "PROTOCOL_BEGIN",
    "PROTOCOL_END",
    "SYMMETRY_CONFIDENCE",
    "VERDICT_FAIL",
    "VERDICT_INCONCLUSIVE",
    "VERDICT_ORDER",
    "VERDICT_PASS",
    "VERDICT_VOID",
    "A1Verdict",
    "ProtocolMarkerError",
    "compute_verdict",
    "iid_positive_control_void_probability",
    "pairs_from_games",
    "positive_control_void_probability",
    "positive_control_void_probability_interval",
    "protocol_sha256",
    "read_registered_protocol",
    "registered_protocol",
    "verdict_from_bounds",
]

#: Marker opening the hashed region of ``docs/EVALUATION.md`` (a whole HTML-comment line).
PROTOCOL_BEGIN = "<!-- A1-PROTOCOL-BEGIN"
#: Marker closing it.
PROTOCOL_END = "<!-- A1-PROTOCOL-END"

#: Per-game outcome labels, in the order the simulation draws them.
_RESULTS = ("win", "draw", "loss")


class ProtocolMarkerError(ValueError):
    """The document does not hold exactly one well-ordered pair of protocol markers."""


def registered_protocol(text: str) -> str:
    """The registered region of ``text``: the lines strictly between the two markers.

    CRLF is normalised to LF first, so a Windows checkout and a Linux checkout of
    one commit hash identically. Each returned line keeps its terminating LF.

    Raises:
        ProtocolMarkerError: unless each marker occurs on exactly one line and
            the begin marker comes first.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    begins = [i for i, line in enumerate(lines) if PROTOCOL_BEGIN in line]
    ends = [i for i, line in enumerate(lines) if PROTOCOL_END in line]
    if len(begins) != 1 or len(ends) != 1:
        raise ProtocolMarkerError(
            f"expected exactly one {PROTOCOL_BEGIN!r} and one {PROTOCOL_END!r} line, "
            f"found {len(begins)} and {len(ends)}"
        )
    begin, end = begins[0], ends[0]
    if end <= begin:
        raise ProtocolMarkerError("the protocol end marker precedes its begin marker")
    return "".join(f"{line}\n" for line in lines[begin + 1 : end])


def read_registered_protocol(path: str | Path) -> str:
    """:func:`registered_protocol` of the file at ``path`` (read as UTF-8)."""
    return registered_protocol(Path(path).read_text(encoding="utf-8"))


def protocol_sha256(text: str) -> str:
    """Hex sha256 of :func:`registered_protocol` of ``text``, encoded as UTF-8."""
    return hashlib.sha256(registered_protocol(text).encode("utf-8")).hexdigest()


def pairs_from_games(games: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """``[(seat-0 result, seat-1 result)]`` per pair, in pair order, from a result row's games.

    ``games`` is a benchmark result row's ``games`` list (``{pair, seat, result}``,
    results from the candidate's side). Pools from several result files are made
    by concatenating the lists this returns, file by file.

    Raises:
        ValueError: if a pair lacks a seat, repeats one, or a result is unknown.
    """
    by_pair: dict[int, dict[int, str]] = {}
    for game in games:
        result = str(game["result"])
        if result not in _RESULTS:
            raise ValueError(f"unknown result {result!r}")
        seats = by_pair.setdefault(int(game["pair"]), {})
        seat = int(game["seat"])
        if seat in seats:
            raise ValueError(f"pair {game['pair']} has two games from seat {seat}")
        seats[seat] = result
    out = []
    for pair in sorted(by_pair):
        if set(by_pair[pair]) != {0, 1}:
            raise ValueError(f"pair {pair} is incomplete: seats {sorted(by_pair[pair])}")
        out.append((by_pair[pair][0], by_pair[pair][1]))
    return out


def _hull_low_share(replicates: Sequence[Sequence[tuple[str, str]]], confidence: float) -> float:
    at_or_below = 0
    for sample in replicates:
        games = [
            {"pair": p, "seat": seat, "result": result}
            for p, pair in enumerate(sample)
            for seat, result in enumerate(pair)
        ]
        if interval_summary(games, confidence)["hull"][0] <= 0.5:
            at_or_below += 1
    return at_or_below / len(replicates)


def _check_size(n_games: int, reps: int) -> None:
    if n_games <= 0 or n_games % 2:
        raise ValueError(f"n_games must be a positive even number, got {n_games}")
    if reps <= 0:
        raise ValueError(f"reps must be positive, got {reps}")


def positive_control_void_probability(
    pairs: Sequence[tuple[str, str]],
    n_games: int,
    *,
    reps: int = 20_000,
    seed: int = 1111,
    confidence: float = 0.95,
) -> float:
    """``P(hull low <= 0.5)`` for an arm of ``n_games`` games, by pair bootstrap.

    **The registered method** (``docs/EVALUATION.md`` section 7.1). Each
    replicate draws ``n_games // 2`` whole seat-swapped pairs with replacement
    from ``pairs`` -- T0b's measured pairs (:func:`pairs_from_games` of each of
    its result files, concatenated) -- and reads the low end of the
    ``confidence`` hull from
    :func:`~alpha_zero.evaluation.benchmark_evidence.interval_summary`. Resampling
    pairs, not games, keeps whatever the maps do to both games of a pair, which
    is what the pair-clustered half of the hull answers to. The default seed
    makes the number reproducible; it is not a game seed.

    Raises:
        ValueError: for an odd or non-positive ``n_games``, non-positive
            ``reps``, fewer than 2 pairs, or a malformed pair.
    """
    _check_size(n_games, reps)
    if len(pairs) < 2:
        raise ValueError(f"need at least 2 measured pairs, got {len(pairs)}")
    for pair in pairs:
        if len(pair) != 2 or any(r not in _RESULTS for r in pair):
            raise ValueError(f"malformed pair {pair!r}")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(pairs), size=(reps, n_games // 2))
    return _hull_low_share([[pairs[int(i)] for i in row] for row in picks], confidence)


def positive_control_void_probability_interval(
    pairs: Sequence[tuple[str, str]],
    n_games: int,
    *,
    outer: int = 200,
    inner: int = 2_000,
    seed: int = 1111,
    level: float = 0.95,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for :func:`positive_control_void_probability`.

    The point estimate reads one finite reference (about 300 pairs from T0b), so
    it is itself noisy (T13 rereview N3: 0.8%-8.8% across references from one
    model). Each of ``outer`` replicates resamples ``len(pairs)`` pairs from
    ``pairs`` with replacement -- a plausible alternative reference -- and
    computes the VOID probability on it by the same pair bootstrap with
    ``inner`` replicates. The interval is the central ``level`` range of those
    ``outer`` values. Seeded throughout: the outer draw and each inner seed
    come from one generator seeded with ``seed``.

    Raises:
        ValueError: as :func:`positive_control_void_probability`, or for
            non-positive ``outer`` / ``inner`` or ``level`` outside (0, 1).
    """
    _check_size(n_games, inner)
    if outer <= 0:
        raise ValueError(f"outer must be positive, got {outer}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if len(pairs) < 2:
        raise ValueError(f"need at least 2 measured pairs, got {len(pairs)}")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(outer):
        picks = rng.integers(0, len(pairs), size=len(pairs))
        reference = [pairs[int(i)] for i in picks]
        inner_seed = int(rng.integers(0, 2**31 - 1))
        values.append(
            positive_control_void_probability(
                reference, n_games, reps=inner, seed=inner_seed, confidence=confidence
            )
        )
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(np.asarray(values), [tail, 1.0 - tail])
    return float(low), float(high)


def iid_positive_control_void_probability(
    p_win: float,
    p_draw: float,
    n_games: int,
    *,
    reps: int = 20_000,
    seed: int = 1111,
    confidence: float = 0.95,
) -> float:
    """``P(hull low <= 0.5)`` with every game drawn independently -- informational only.

    Outcomes are iid with probabilities ``(p_win, p_draw, 1 - p_win - p_draw)``
    and seated as consecutive pairs. Independence leaves the pair interval no
    excess width, so this ignores map clustering and is a **lower bound** on the
    VOID risk (T13 review I1 measured 2.75% here against 7.2% under per-map
    clustering at d = 0.3). Section 7.1 reports it beside the registered
    :func:`positive_control_void_probability`; the branch rule never reads it.

    Raises:
        ValueError: for an odd or non-positive ``n_games``, non-positive
            ``reps``, or probabilities outside ``[0, 1]`` / summing above 1.
    """
    _check_size(n_games, reps)
    p_loss = 1.0 - p_win - p_draw
    if not (0.0 <= p_win <= 1.0 and 0.0 <= p_draw <= 1.0 and p_loss >= -1e-12):
        raise ValueError(f"invalid outcome probabilities: win {p_win}, draw {p_draw}")
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(_RESULTS), size=(reps, n_games), p=[p_win, p_draw, max(0.0, p_loss)])
    replicates = [
        [(_RESULTS[int(row[2 * p])], _RESULTS[int(row[2 * p + 1])]) for p in range(n_games // 2)]
        for row in draws
    ]
    return _hull_low_share(replicates, confidence)


# ---------------------------------------------------------------------------
# The verdict (section 7.1, "Verdict")
# ---------------------------------------------------------------------------

#: The verdict table's rows, in the order they are tried; the first match wins.
VERDICT_ORDER: tuple[str, str, str, str] = ("VOID", "PASS", "FAIL", "INCONCLUSIVE")
VERDICT_VOID, VERDICT_PASS, VERDICT_FAIL, VERDICT_INCONCLUSIVE = VERDICT_ORDER

#: Confidence each judged arm is read at (section 7.1, "Intervals").
PRIMARY_CONFIDENCE = 0.95
SYMMETRY_CONFIDENCE = 0.99
POSITIVE_CONTROL_CONFIDENCE = 0.95

#: One-sided alpha of the INCONCLUSIVE report's MDE.
_MDE_ALPHA = 0.025


def verdict_from_bounds(
    *,
    interruptions: Sequence[str],
    symmetry_hull: Sequence[float],
    positive_control_hull: Sequence[float],
    primary_hull: Sequence[float],
) -> tuple[str, tuple[str, ...]]:
    """The section 7.1 verdict table applied to decision bounds: ``(verdict, reasons)``.

    Args:
        interruptions: Why the run counts as interrupted or resumed (``run.json``
            not ``COMPLETED``, a resumed or superseding ``iterations.jsonl`` line,
            an ``MCTS_`` variable); empty for a clean run.
        symmetry_hull: The symmetry arm's 99% hull ``(L, U)``.
        positive_control_hull: The positive control's 95% hull.
        primary_hull: The primary arm's 95% hull.

    Rows, first match wins:

    1. VOID -- any interruption; OR the symmetry hull excludes 0.5 (``L > 0.5``
       or ``U < 0.5``); OR the positive control's ``L <= 0.5``. Every VOID
       condition that holds is listed in ``reasons``.
    2. PASS -- the primary ``L > 0.5``.
    3. FAIL -- the primary ``U < 0.5``.
    4. INCONCLUSIVE -- otherwise.

    The comparisons are exact: a bound of exactly 0.5 does not exclude 0.5, so
    the symmetry arm does not VOID, the positive control does, and the primary
    arm is not a PASS (nor a FAIL).
    """
    void = [f"interrupted or resumed: {reason}" for reason in interruptions]
    sym_low, sym_high = float(symmetry_hull[0]), float(symmetry_hull[1])
    if sym_low > 0.5 or sym_high < 0.5:
        void.append(
            f"the symmetry arm's {SYMMETRY_CONFIDENCE:.0%} hull "
            f"[{sym_low:.4f}, {sym_high:.4f}] excludes 0.5"
        )
    pc_low = float(positive_control_hull[0])
    if pc_low <= 0.5:
        void.append(
            f"the positive control's {POSITIVE_CONTROL_CONFIDENCE:.0%} L = {pc_low:.4f} <= 0.5"
        )
    if void:
        return VERDICT_VOID, tuple(void)
    low, high = float(primary_hull[0]), float(primary_hull[1])
    bounds = f"[{low:.4f}, {high:.4f}]"
    if low > 0.5:
        return VERDICT_PASS, (f"the primary arm's 95% L > 0.5: hull {bounds}",)
    if high < 0.5:
        return VERDICT_FAIL, (f"the primary arm's 95% U < 0.5: hull {bounds}",)
    return VERDICT_INCONCLUSIVE, (f"the primary arm's 95% hull {bounds} contains 0.5",)


@dataclass(frozen=True)
class A1Verdict:
    """A recomputed A1 verdict and everything section 7.1 says it reports.

    The MDE fields are set only for INCONCLUSIVE, the one verdict that reports
    them; they are ``None`` otherwise.
    """

    verdict: str
    reasons: tuple[str, ...]
    primary_score: float
    primary_draw_rate: float
    primary_hull: tuple[float, float]
    symmetry_hull: tuple[float, float]
    positive_control_hull: tuple[float, float]
    mde_alpha025_80pct: float | None
    hull_ratio: float | None
    mde_alpha025_80pct_hull: float | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (tuples become lists)."""
        return {
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "primary_score": self.primary_score,
            "primary_draw_rate": self.primary_draw_rate,
            "primary_hull": list(self.primary_hull),
            "symmetry_hull": list(self.symmetry_hull),
            "positive_control_hull": list(self.positive_control_hull),
            "mde_alpha025_80pct": self.mde_alpha025_80pct,
            "hull_ratio": self.hull_ratio,
            "mde_alpha025_80pct_hull": self.mde_alpha025_80pct_hull,
        }


def _hull(games: Sequence[Mapping[str, Any]], confidence: float) -> tuple[float, float]:
    low, high = interval_summary(games, confidence)["hull"]
    return float(low), float(high)


def compute_verdict(
    primary_games: Sequence[Mapping[str, Any]],
    symmetry_games: Sequence[Mapping[str, Any]],
    positive_control_games: Sequence[Mapping[str, Any]],
    *,
    interruptions: Sequence[str] = (),
) -> A1Verdict:
    """Recompute the section 7.1 verdict from the judged arms' per-game records.

    Each ``*_games`` argument is a benchmark result row's ``games`` list
    (``{pair, seat, result}``, results from the candidate's side). The hulls come
    from :func:`~alpha_zero.evaluation.benchmark_evidence.interval_summary` --
    the same code that wrote the rows -- at 95% (primary, positive control) and
    99% (symmetry); the table is :func:`verdict_from_bounds`. An INCONCLUSIVE
    verdict also reports ``minimum_detectable_effect(n, d, alpha=0.025)`` at the
    primary arm's game count and draw rate, and that MDE widened by
    :func:`~alpha_zero.evaluation.benchmark_evidence.facing_hull_ratio`.

    Raises:
        ValueError: if an arm's records are not whole seat-swapped pairs or
            hold an unknown result.
    """
    results = [str(g["result"]) for g in primary_games]
    unknown = sorted(set(results) - set(_RESULTS))
    if unknown:
        raise ValueError(f"unknown result(s) {unknown}")
    wins, draws, losses = (results.count(r) for r in _RESULTS)
    summary = summarize(wins, draws, losses, PRIMARY_CONFIDENCE)
    primary = interval_summary(primary_games, PRIMARY_CONFIDENCE)
    primary_hull = (float(primary["hull"][0]), float(primary["hull"][1]))
    symmetry_hull = _hull(symmetry_games, SYMMETRY_CONFIDENCE)
    positive_hull = _hull(positive_control_games, POSITIVE_CONTROL_CONFIDENCE)
    verdict, reasons = verdict_from_bounds(
        interruptions=interruptions,
        symmetry_hull=symmetry_hull,
        positive_control_hull=positive_hull,
        primary_hull=primary_hull,
    )
    mde = ratio = widened = None
    if verdict == VERDICT_INCONCLUSIVE:
        mde = minimum_detectable_effect(len(results), summary.draw_rate, alpha=_MDE_ALPHA)
        ratio = facing_hull_ratio(summary.score_rate, primary["iid"], primary["hull"])
        widened = mde * ratio
    return A1Verdict(
        verdict=verdict,
        reasons=reasons,
        primary_score=summary.score_rate,
        primary_draw_rate=summary.draw_rate,
        primary_hull=primary_hull,
        symmetry_hull=symmetry_hull,
        positive_control_hull=positive_hull,
        mde_alpha025_80pct=mde,
        hull_ratio=ratio,
        mde_alpha025_80pct_hull=widened,
    )
