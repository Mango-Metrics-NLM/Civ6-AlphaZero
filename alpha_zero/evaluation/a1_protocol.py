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
  is that number by seeded simulation through the benchmark's own
  :func:`~alpha_zero.evaluation.benchmark_evidence.interval_summary`.

``python scripts/check_learning_trend.py --print-protocol-sha`` prints the hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from alpha_zero.evaluation.benchmark_evidence import interval_summary

__all__ = [
    "PROTOCOL_BEGIN",
    "PROTOCOL_END",
    "ProtocolMarkerError",
    "positive_control_void_probability",
    "protocol_sha256",
    "read_registered_protocol",
    "registered_protocol",
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


def positive_control_void_probability(
    p_win: float,
    p_draw: float,
    n_games: int,
    *,
    reps: int = 20_000,
    seed: int = 1111,
    confidence: float = 0.95,
) -> float:
    """``P(hull low <= 0.5)`` for an arm of ``n_games`` independent games.

    Each replicate draws ``n_games`` outcomes iid with probabilities
    ``(p_win, p_draw, 1 - p_win - p_draw)``, seats them as consecutive
    seat-swapped pairs (game ``g`` is pair ``g // 2``, seat ``g % 2``), and reads
    the low end of the ``confidence`` hull from
    :func:`~alpha_zero.evaluation.benchmark_evidence.interval_summary` -- the
    statistic the A1 verdict reads for the positive control. The default seed
    makes the number reproducible; it is not a game seed.

    Raises:
        ValueError: for an odd or non-positive ``n_games``, non-positive
            ``reps``, or probabilities outside ``[0, 1]`` / summing above 1.
    """
    if n_games <= 0 or n_games % 2:
        raise ValueError(f"n_games must be a positive even number, got {n_games}")
    if reps <= 0:
        raise ValueError(f"reps must be positive, got {reps}")
    p_loss = 1.0 - p_win - p_draw
    if not (0.0 <= p_win <= 1.0 and 0.0 <= p_draw <= 1.0 and p_loss >= -1e-12):
        raise ValueError(f"invalid outcome probabilities: win {p_win}, draw {p_draw}")
    probabilities = [p_win, p_draw, max(0.0, p_loss)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(_RESULTS), size=(reps, n_games), p=probabilities)
    at_or_below = 0
    for row in draws:
        games = [
            {"pair": g // 2, "seat": g % 2, "result": _RESULTS[int(outcome)]}
            for g, outcome in enumerate(row)
        ]
        if interval_summary(games, confidence)["hull"][0] <= 0.5:
            at_or_below += 1
    return at_or_below / reps
