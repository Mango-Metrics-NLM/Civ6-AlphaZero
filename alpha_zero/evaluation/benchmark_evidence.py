"""The benchmark's evidence block and its statistics over per-game records (LF-D6).

Split from :mod:`alpha_zero.evaluation.benchmark`, which re-exports every public
name here: this module holds what does not depend on the protocol or the arena
-- the ``docs/evidence/SCHEMA.md`` block, the per-pair scores and the
iid / pair-clustered / hull interval block, and the shard partition.
"""

from __future__ import annotations

import os
import platform
import shlex
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alpha_zero.evaluation.scoring import interval_hull, pair_score_interval, score_interval

__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "SEARCH_EPOCH",
    "SEARCH_SEMANTICS",
    "benchmark_claims",
    "cli_command",
    "collect_evidence",
    "facing_hull_ratio",
    "hardware_summary",
    "interval_summary",
    "pair_scores",
    "parse_shard",
    "shard_pairs",
]

#: ``schema_version`` of the ``evidence`` block (``docs/evidence/SCHEMA.md``).
EVIDENCE_SCHEMA_VERSION = 1

#: The ``docs/evidence/SCHEMA.md`` epoch. Results from different epochs are not
#: comparable: the 2026-09-04 fixes changed the value backup sign rule and the
#: policy target, which changes how strong a given checkpoint *plays*. This code
#: post-dates them by construction.
SEARCH_EPOCH = "post-2026-09-04"

#: Finer label for the search semantics, recorded beside the epoch. Update when
#: the search changes in a way that alters play at fixed weights.
SEARCH_SEMANTICS = "2026-09-04.to_play-backup+completed-q"

_GAME_SCORE = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    fallback = platform.processor() or None
    if not cpuinfo.exists():
        return fallback
    try:
        text = cpuinfo.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return fallback


def hardware_summary() -> dict[str, Any]:
    """CPU, GPU, core count and thread cap -- the things that move timings."""
    gpu_model: str | None = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch missing or CUDA probe failing
        gpu_model = None
    return {
        "cpu_model": _cpu_model(),
        "gpu_model": gpu_model,
        "logical_cores": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def _git_sha(start: Path | None = None) -> str | None:
    """Commit at ``HEAD`` without a subprocess; ``None`` when it cannot be resolved.

    The implementation lives in ``utils.checkpoint.read_git_sha`` so checkpoint
    provenance and the benchmark envelope record the same value.
    """
    from alpha_zero.utils.checkpoint import read_git_sha

    return read_git_sha(start)


def _version_of(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


#: File name of the operator-facing wrapper over the benchmark CLI.
WRAPPER_SCRIPT = "benchmark_suite.py"


def cli_command(argv: Sequence[str] | None = None) -> str:
    """The command that ran, ``shlex``-joined: entry point plus arguments.

    The entry point is the one actually used -- ``python scripts/benchmark_suite.py``
    (as typed) when the wrapper ran, ``python -m alpha_zero.evaluation.benchmark``
    otherwise (including ``run_card`` children and in-process calls). The two
    take the same arguments.
    """
    argv0 = sys.argv[0] if sys.argv else ""
    if Path(argv0).name == WRAPPER_SCRIPT:
        entry = ["python", Path(argv0).as_posix()]
    else:
        entry = ["python", "-m", "alpha_zero.evaluation.benchmark"]
    return shlex.join([*entry, *(sys.argv[1:] if argv is None else argv)])


def collect_evidence(
    seed: int | None,
    games_total: int,
    command: str | None = None,
    *,
    claims: Sequence[str] = (),
    non_claims: Sequence[str] = (),
    device: str | None = None,
    search: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``docs/evidence/SCHEMA.md`` block: enough to reproduce or dismiss ``results``.

    Beyond the contract it records ``search_semantics``, the torch ``device`` the
    searches ran on (confirmatory runs are CPU-only) and the ``search`` config, so
    shards that differ in any of them cannot be merged.
    """
    block: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "git_sha": _git_sha(),
        "generated_utc": datetime.now(UTC).isoformat(),
        "search_epoch": SEARCH_EPOCH,
        "search_semantics": SEARCH_SEMANTICS,
        "seed": seed,
        "command": command if command is not None else cli_command(),
        "device": device,
        "search": dict(search) if search is not None else None,
        "hardware": hardware_summary(),
        "python": platform.python_version(),
        "torch": _version_of("torch"),
        "numpy": _version_of("numpy"),
        "games_total": games_total,
        "claims": list(claims),
        "non_claims": list(non_claims),
    }
    if seed is None:
        block["seed_null_reason"] = "the benchmark was run without --seed"
    if block["git_sha"] is None:
        block["git_sha_null_reason"] = (
            "no valid CIV6AZ_GIT_SHA and no resolvable .git at the package's source root"
        )
    return block


def benchmark_claims(
    candidate_id: str, opponents: Sequence[str], n_games: int, shard: Mapping[str, Any] | None
) -> tuple[list[str], list[str]]:
    """What the envelope supports, and what it must not be quoted for."""
    claims = [
        f"score rate of {candidate_id} against {opponent} over {n_games} games "
        f"({n_games // 2} seat-swapped map pairs) under the recorded protocol, with the "
        "iid ternary and pair-clustered intervals and their hull at 95% and 99%"
        for opponent in opponents
    ]
    non_claims = [
        "strength under any other board, turn limit, search budget or search epoch",
        "an Elo rating against a pool: elo_* re-express one pairwise score rate",
        "a verdict: the pre-registered decision rule lives in docs/EVALUATION.md",
    ]
    if shard is not None:
        non_claims.insert(
            0,
            f"a complete result: this is shard {shard['index']}/{shard['count']}; "
            "merge every shard (--merge) before quoting it",
        )
    return claims, non_claims


def pair_scores(games: Sequence[Mapping[str, Any]]) -> list[float]:
    """Per-pair score (mean of the two seat-swapped games), in pair order.

    Raises :class:`ValueError` unless every pair has exactly one game per seat:
    the pair is the sample unit, so a half pair is not a smaller sample, it is a
    different statistic.
    """
    by_pair: dict[int, dict[int, float]] = {}
    for game in games:
        seats = by_pair.setdefault(int(game["pair"]), {})
        seat = int(game["seat"])
        if seat in seats:
            raise ValueError(f"pair {game['pair']} has two games from seat {seat}")
        seats[seat] = _GAME_SCORE[game["result"]]
    scores = []
    for pair in sorted(by_pair):
        if set(by_pair[pair]) != {0, 1}:
            raise ValueError(f"pair {pair} is incomplete: seats {sorted(by_pair[pair])}")
        scores.append((by_pair[pair][0] + by_pair[pair][1]) / 2.0)
    return scores


def interval_summary(
    games: Sequence[Mapping[str, Any]], confidence: float
) -> dict[str, list[float]]:
    """The iid ternary interval, the pair-clustered interval and their hull.

    The hull ``[min(lows), max(highs)]`` is what the A1 decision rule reads, so
    a verdict does not hinge on whether games within a pair are independent.
    """
    results = [g["result"] for g in games]
    iid = score_interval(
        results.count("win"), results.count("draw"), results.count("loss"), confidence
    )
    pair = pair_score_interval(pair_scores(games), confidence)
    return {"iid": list(iid), "pair": list(pair), "hull": list(interval_hull(iid, pair))}


def facing_hull_ratio(score: float, iid: Sequence[float], hull: Sequence[float]) -> float:
    """How much further the hull reaches toward 0.5 than the iid interval does.

    ``(score - hull_low) / (score - iid_low)`` when ``score > 0.5`` (the low side
    faces 0.5), ``(hull_high - score) / (iid_high - score)`` when ``score < 0.5``,
    and the larger of the two at exactly 0.5. Distances are from the point
    estimate (the score rate). The verdict and the MDE concern the bound that
    faces 0.5, so a hull wider only on the far side does not widen the MDE.
    Always ``>= 1`` because the hull contains the iid interval; a degenerate
    denominator gives 1.
    """

    def ratio(reach_hull: float, reach_iid: float) -> float:
        return reach_hull / reach_iid if reach_iid > 0 else 1.0

    low = ratio(score - hull[0], score - iid[0])
    high = ratio(hull[1] - score, iid[1] - score)
    if score > 0.5:
        return low
    if score < 0.5:
        return high
    return max(low, high)


def parse_shard(spec: str) -> tuple[int, int]:
    """``"i/N"`` -> ``(i, N)`` with ``1 <= i <= N`` (shards are 1-based)."""
    parts = spec.strip().split("/")
    try:
        index, count = (int(x) for x in parts)
    except ValueError:
        raise ValueError(f"--shard must be i/N with integers, got {spec!r}") from None
    if count < 1 or not 1 <= index <= count:
        raise ValueError(f"--shard i/N needs 1 <= i <= N (1-based), got {spec!r}")
    return index, count


def shard_pairs(num_pairs: int, index: int, count: int) -> range:
    """The ``index``-th of ``count`` contiguous, near-equal blocks of ``range(num_pairs)``.

    The pair is the unit: a shard never holds one seat of a map without the other.
    """
    if count > num_pairs:
        raise ValueError(f"{count} shards for {num_pairs} pairs would leave a shard empty")
    return range((index - 1) * num_pairs // count, index * num_pairs // count)
