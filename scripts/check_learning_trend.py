#!/usr/bin/env python3
"""Describe a training run's learning trend (LF-A2). Descriptive only: it decides nothing.

Reads what a ``training_civ6`` run and its arena arms leave behind:

* ``<checkpoint_dir>/iterations.jsonl`` -- one record per iteration
  (:mod:`alpha_zero.training.run_record`); the last line for an iteration wins;
* ``<checkpoint_dir>/eval_<iter>.json`` -- in-loop evaluations vs a random agent;
* benchmark ``result.json`` envelopes (``scripts/benchmark_suite.py``), each
  placed on the iteration of its ``--checkpoint checkpoint_<N>.pt`` candidate.

and reports, as JSON on stdout:

* **score-rate series** against iteration -- in-loop evaluation vs random, and
  one series per benchmark opponent -- with their intervals and a least-squares
  slope;
* **diagnostic series** from ``iterations.jsonl`` (losses, prequential value
  metrics, target sharpness), with slopes;
* the **Option E target-sharpness trigger** registered in ``docs/EVALUATION.md``
  section 7.1 (T1 / T2), evaluated on the run's own ``iterations.jsonl``.

The A1 verdict is section 7.1's verdict table, not anything here: a slope over a
handful of checkpoints is a description, not a test.

Exit codes:

* **0** -- a trend was computed: at least one score-rate series has
  :data:`MIN_POINTS` or more iterations;
* **1** -- malformed input: an unparseable line or file, a record without an
  integer iteration, counts that do not add up, a score rate that does not match
  its counts, a benchmark envelope that fails
  :func:`alpha_zero.evaluation.benchmark.validate_envelope`, or two inputs
  claiming one point (malformed wins over everything else);
* **2** -- insufficient data: no score-rate series reaches :data:`MIN_POINTS`
  (a 2-point run exits 2), or a named input does not exist.

``--print-protocol-sha`` prints the sha256 of ``docs/EVALUATION.md`` section 7.1
(:func:`alpha_zero.evaluation.a1_protocol.protocol_sha256`) and exits 0; 1 if the
markers are malformed, 2 if the file is missing.

Usage::

    python scripts/check_learning_trend.py --run-dir <D>/ckpt --benchmark <D>/arena
    python scripts/check_learning_trend.py --eval eval_5.json eval_10.json eval_15.json
    python scripts/check_learning_trend.py --print-protocol-sha
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shlex
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run THIS checkout's package, never a stale editable install of another one.
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("check_learning_trend")

EXIT_TREND = 0
EXIT_MALFORMED = 1
EXIT_INSUFFICIENT = 2

#: A score-rate series needs this many distinct iterations before a slope is reported.
MIN_POINTS = 3

REPORT_SCHEMA_VERSION = 1
DEFAULT_EVALUATION_MD = REPO_ROOT / "docs" / "EVALUATION.md"
ITERATIONS_FILE = "iterations.jsonl"

#: ``iterations.jsonl`` fields reported as diagnostic series: (label, section, key).
DIAGNOSTIC_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("train.total_loss", "train", "total_loss"),
    ("train.policy_loss", "train", "policy_loss"),
    ("train.value_loss", "train", "value_loss"),
    ("prequential.value_mse", "prequential", "value_mse"),
    ("prequential.r2_oos", "prequential", "r2_oos"),
    ("prequential.sign_accuracy", "prequential", "sign_accuracy"),
    ("prequential.target_entropy", "prequential", "target_entropy"),
    ("prequential.target_max_p_mean", "prequential", "target_max_p_mean"),
    (
        "prequential.target_share_above_greedy_threshold",
        "prequential",
        "target_share_above_greedy_threshold",
    ),
)

#: Section 7.1's Option E trigger constants.
T1_FLOOR = 0.25
T1_MULTIPLIER = 5.0
T2_MAX_P = 0.80

_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pt$")
_SCORE_TOLERANCE = 1e-9


class MalformedInputError(ValueError):
    """An input exists but cannot be read as the format it claims to be."""


@dataclass(frozen=True)
class Point:
    """One score-rate observation at one iteration."""

    iteration: int
    score_rate: float
    ci_low: float | None
    ci_high: float | None
    games: int
    source: str


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MalformedInputError(f"{path}: not readable JSON ({exc})") from None


def _iteration_of(record: Mapping[str, Any], where: str) -> int:
    iteration = record.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise MalformedInputError(f"{where}: 'iteration' must be a non-negative integer")
    return iteration


def load_iteration_records(path: Path) -> dict[int, dict[str, Any]]:
    """``iterations.jsonl`` as ``{iteration: record}``; the last line for an iteration wins.

    Strict, unlike :func:`alpha_zero.training.run_record.read_iteration_records`:
    a line that does not parse is malformed input here, because a torn record
    means the run did not finish cleanly and its description should say so.
    """
    records: dict[int, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MalformedInputError(f"{path}: not readable ({exc})") from None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            raise MalformedInputError(f"{where}: not valid JSON (a torn write?)") from None
        if not isinstance(record, dict):
            raise MalformedInputError(f"{where}: a record must be a JSON object")
        records[_iteration_of(record, where)] = record
    return records


def _counted_point(
    result: Any, iteration: int, source: str, *, low_key: str, high_key: str
) -> Point:
    """A :class:`Point` from a ``ScoreSummary``-shaped dict, checking its arithmetic."""
    if not isinstance(result, Mapping):
        raise MalformedInputError(f"{source}: 'result' must be an object")
    counts = [result.get(k) for k in ("wins", "draws", "losses")]
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in counts):
        raise MalformedInputError(f"{source}: wins/draws/losses must be non-negative integers")
    wins, draws, losses = counts
    games = wins + draws + losses
    if games == 0:
        raise MalformedInputError(f"{source}: no games")
    score = result.get("score_rate")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise MalformedInputError(f"{source}: 'score_rate' must be a number")
    if abs(float(score) - (wins + 0.5 * draws) / games) > _SCORE_TOLERANCE:
        raise MalformedInputError(f"{source}: score_rate {score} does not match its counts")
    low, high = result.get(low_key), result.get(high_key)
    return Point(
        iteration=iteration,
        score_rate=float(score),
        ci_low=None if low is None else float(low),
        ci_high=None if high is None else float(high),
        games=games,
        source=source,
    )


def eval_point_from_file(path: Path) -> Point:
    """One ``eval_<iter>.json`` (``training_civ6.write_eval_json``) as a :class:`Point`."""
    data = _read_json(path)
    if not isinstance(data, dict):
        raise MalformedInputError(f"{path}: an evaluation file must be a JSON object")
    iteration = _iteration_of(data, str(path))
    return _counted_point(
        data.get("result"), iteration, str(path), low_key="ci_low", high_key="ci_high"
    )


def eval_points_from_records(records: Mapping[int, Mapping[str, Any]], source: str) -> list[Point]:
    """The in-loop evaluations embedded in ``iterations.jsonl`` (``"eval"``, when not null)."""
    points = []
    for iteration, record in sorted(records.items()):
        result = record.get("eval")
        if result is None:
            continue
        points.append(
            _counted_point(
                result,
                iteration,
                f"{source} iteration {iteration}",
                low_key="ci_low",
                high_key="ci_high",
            )
        )
    return points


def _candidate_checkpoint(command: str) -> tuple[int | None, str | None]:
    """``(iteration, file name)`` of the ``--checkpoint`` in a benchmark's recorded command."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None, None
    path: str | None = None
    for i, token in enumerate(tokens):
        if token == "--checkpoint" and i + 1 < len(tokens):
            path = tokens[i + 1]
        elif token.startswith("--checkpoint="):
            path = token.split("=", 1)[1]
    if path is None:
        return None, None
    name = re.split(r"[\\/]", path)[-1]
    match = _CHECKPOINT_RE.search(name)
    return (int(match.group(1)) if match else None), name


def _opponent_checkpoint_names(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    return [
        re.split(r"[\\/]", tokens[i + 1])[-1]
        for i, token in enumerate(tokens[:-1])
        if token == "--opponent-checkpoint"
    ]


def benchmark_files(paths: Iterable[Path]) -> list[Path]:
    """Expand each path: a file is itself; a directory holds ``result.json`` files."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            direct = path / "result.json"
            found.extend([direct] if direct.is_file() else sorted(path.rglob("result.json")))
        else:
            found.append(path)
    return found


def benchmark_points(
    paths: Sequence[Path],
) -> tuple[dict[str, list[Point]], list[dict[str, Any]]]:
    """Score-rate points per benchmark series, plus the rows no iteration can be given to.

    A series is keyed by opponent; a checkpoint opponent named by exactly one
    ``--opponent-checkpoint`` is labelled with its file name too, so the A1
    "vs checkpoint_0" series reads as such.
    """
    from alpha_zero.evaluation.benchmark import validate_envelope

    series: dict[str, list[Point]] = {}
    unplaced: list[dict[str, Any]] = []
    for path in paths:
        envelope = _read_json(path)
        if not isinstance(envelope, dict):
            raise MalformedInputError(f"{path}: a benchmark result must be a JSON object")
        try:
            validate_envelope(envelope)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise MalformedInputError(f"{path}: {exc}") from None
        command = str(envelope["evidence"].get("command", ""))
        iteration, candidate_file = _candidate_checkpoint(command)
        opponent_files = _opponent_checkpoint_names(command)
        for row in envelope["results"]:
            opponent = str(row["opponent_id"])
            if opponent.startswith("checkpoint:") and len(opponent_files) == 1:
                opponent = f"{opponent_files[0]} ({opponent})"
            hull = row["intervals"]["0.95"]["hull"]
            if iteration is None:
                unplaced.append(
                    {
                        "file": str(path),
                        "candidate": envelope["candidate"].get("id"),
                        "candidate_file": candidate_file,
                        "opponent": opponent,
                        "score_rate": row["score_rate"],
                        "hull95": hull,
                        "games": row["num_games"],
                    }
                )
                continue
            series.setdefault(opponent, []).append(
                Point(
                    iteration=iteration,
                    score_rate=float(row["score_rate"]),
                    ci_low=float(hull[0]),
                    ci_high=float(hull[1]),
                    games=int(row["num_games"]),
                    source=str(path),
                )
            )
    return series, unplaced


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------


def least_squares(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    """``(slope, intercept)`` of the ordinary least-squares line, or ``None`` if x is constant."""
    n = len(xs)
    mean_x, mean_y = math.fsum(xs) / n, math.fsum(ys) / n
    sxx = math.fsum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def describe(values: Sequence[tuple[int, float]]) -> dict[str, Any]:
    """Points, first/last/delta and the least-squares slope per iteration (``None`` below 3)."""
    ordered = sorted(values)
    summary: dict[str, Any] = {"n_points": len(ordered), "slope_per_iteration": None}
    if not ordered:
        return summary
    summary.update(
        first_iteration=ordered[0][0],
        last_iteration=ordered[-1][0],
        first=ordered[0][1],
        last=ordered[-1][1],
        delta=ordered[-1][1] - ordered[0][1],
    )
    if len(ordered) >= MIN_POINTS:
        fit = least_squares([float(x) for x, _ in ordered], [y for _, y in ordered])
        if fit is not None:
            summary["slope_per_iteration"], summary["intercept"] = fit
    return summary


def _score_series(label: str, kind: str, points: Sequence[Point]) -> dict[str, Any]:
    seen: dict[int, str] = {}
    for point in points:
        if point.iteration in seen:
            raise MalformedInputError(
                f"{label}: iteration {point.iteration} appears in both "
                f"{seen[point.iteration]} and {point.source}"
            )
        seen[point.iteration] = point.source
    ordered = sorted(points, key=lambda p: p.iteration)
    return {
        "label": label,
        "kind": kind,
        "points": [asdict(p) for p in ordered],
        "trend": describe([(p.iteration, p.score_rate) for p in ordered]),
    }


def diagnostic_series(records: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One descriptive series per :data:`DIAGNOSTIC_FIELDS` entry; null and error records skipped."""
    out = []
    for label, section, key in DIAGNOSTIC_FIELDS:
        values: list[tuple[int, float]] = []
        for iteration, record in records.items():
            block = record.get(section)
            if not isinstance(block, Mapping) or "error" in block:
                continue
            value = block.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append((iteration, float(value)))
        out.append({"label": label, "trend": describe(values), "values": sorted(values)})
    return out


def _share_and_max_p(record: Mapping[str, Any]) -> tuple[float | None, float | None]:
    block = record.get("prequential")
    if not isinstance(block, Mapping) or "error" in block:
        return None, None

    def number(key: str) -> float | None:
        value = block.get(key)
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        return float(value) if ok else None

    return number("target_share_above_greedy_threshold"), number("target_max_p_mean")


def option_e_trigger(records: Mapping[int, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Section 7.1's T1 / T2 on ``iterations.jsonl``; ``None`` without records.

    ``s0`` is the share from loop index 0 (``"iteration": 1``), used only when
    that record's prequential block has ``generator == "network"``, no
    ``error`` and a non-null share; otherwise ``s0 = 0``. T1 fires when a later
    iteration's share exceeds ``max(0.25, 5 * s0)``; T2 when any iteration's
    mean max-p exceeds 0.80. Records with an error or null value contribute
    nothing.
    """
    if not records:
        return None
    first = records.get(1)
    s0, s0_source = 0.0, "default 0: no record for loop index 0 (iteration 1)"
    if first is not None:
        block = first.get("prequential")
        share, _ = _share_and_max_p(first)
        if not isinstance(block, Mapping) or "error" in block:
            s0_source = "default 0: the iteration-1 record has no error-free prequential block"
        elif block.get("generator") != "network":
            s0_source = "default 0: the iteration-1 record was not network-generated"
        elif share is None:
            s0_source = "default 0: the iteration-1 share is null"
        else:
            s0, s0_source = share, "iteration-1 record (loop index 0)"
    t1_threshold = max(T1_FLOOR, T1_MULTIPLIER * s0)
    t1_hits, t2_hits = [], []
    for iteration, record in sorted(records.items()):
        share, max_p = _share_and_max_p(record)
        if iteration >= 2 and share is not None and share > t1_threshold:
            t1_hits.append(iteration)
        if max_p is not None and max_p > T2_MAX_P:
            t2_hits.append(iteration)
    return {
        "s0": s0,
        "s0_source": s0_source,
        "t1_threshold": t1_threshold,
        "t1_fired_at": t1_hits,
        "t2_threshold": T2_MAX_P,
        "t2_fired_at": t2_hits,
        "fired": bool(t1_hits or t2_hits),
        "note": "descriptive: changes nothing during A1 (docs/EVALUATION.md section 7.1)",
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Inputs:
    """Resolved input files."""

    iterations: Path | None
    evals: list[Path]
    benchmarks: list[Path]


def resolve_inputs(args: argparse.Namespace) -> tuple[Inputs, list[str]]:
    """Inputs named by the CLI, and the problems that make the data insufficient (missing)."""
    missing: list[str] = []
    iterations: Path | None = args.iterations
    evals = list(args.eval or [])
    if args.run_dir is not None:
        if not args.run_dir.is_dir():
            missing.append(f"--run-dir {args.run_dir} is not a directory")
        else:
            candidate = args.run_dir / ITERATIONS_FILE
            if iterations is None and candidate.is_file():
                iterations = candidate
            evals.extend(sorted(args.run_dir.glob("eval_*.json")))
    for path in [*([iterations] if iterations else []), *evals, *(args.benchmark or [])]:
        if not path.exists():
            missing.append(f"{path} does not exist")
    benchmarks = benchmark_files(p for p in (args.benchmark or []) if p.exists())
    existing_evals = [p for p in evals if p.is_file()]
    if iterations is not None and not iterations.is_file():
        iterations = None
    return Inputs(iterations, existing_evals, benchmarks), missing


def build_report(inputs: Inputs) -> tuple[dict[str, Any], list[str]]:
    """The descriptive report and the malformed-input problems found building it."""
    problems: list[str] = []
    records: dict[int, dict[str, Any]] = {}
    if inputs.iterations is not None:
        try:
            records = load_iteration_records(inputs.iterations)
        except MalformedInputError as exc:
            problems.append(str(exc))

    eval_points: dict[int, Point] = {}
    for path in inputs.evals:
        try:
            point = eval_point_from_file(path)
        except MalformedInputError as exc:
            problems.append(str(exc))
            continue
        if point.iteration in eval_points:
            problems.append(f"{path}: a second evaluation file for iteration {point.iteration}")
        eval_points[point.iteration] = point
    try:
        for point in eval_points_from_records(records, str(inputs.iterations)):
            filed = eval_points.get(point.iteration)
            if filed is None:
                eval_points[point.iteration] = point
            elif abs(filed.score_rate - point.score_rate) > _SCORE_TOLERANCE:
                problems.append(
                    f"iteration {point.iteration}: {filed.source} and {point.source} disagree"
                )
    except MalformedInputError as exc:
        problems.append(str(exc))

    score_series: list[dict[str, Any]] = []
    unplaced: list[dict[str, Any]] = []
    try:
        if eval_points:
            score_series.append(
                _score_series("in-loop evaluation vs random", "eval", list(eval_points.values()))
            )
        if inputs.benchmarks:
            bench, unplaced = benchmark_points(inputs.benchmarks)
            for opponent, points in sorted(bench.items()):
                score_series.append(_score_series(f"benchmark vs {opponent}", "benchmark", points))
    except MalformedInputError as exc:
        problems.append(str(exc))

    report: dict[str, Any] = {
        "checker": "check_learning_trend",
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "descriptive_only": True,
        "min_points": MIN_POINTS,
        "inputs": {
            "iterations": None if inputs.iterations is None else str(inputs.iterations),
            "evals": [str(p) for p in inputs.evals],
            "benchmarks": [str(p) for p in inputs.benchmarks],
        },
        "score_series": score_series,
        "unplaced_benchmark_rows": unplaced,
        "diagnostic_series": diagnostic_series(records),
        "option_e_trigger": option_e_trigger(records),
    }
    return report, problems


def classify(report: Mapping[str, Any], problems: Sequence[str], missing: Sequence[str]) -> int:
    """Exit code: malformed (1) beats insufficient (2) beats a computed trend (0)."""
    if problems:
        return EXIT_MALFORMED
    if missing:
        return EXIT_INSUFFICIENT
    enough = any(s["trend"]["slope_per_iteration"] is not None for s in report["score_series"])
    return EXIT_TREND if enough else EXIT_INSUFFICIENT


_STATUS = {EXIT_TREND: "trend", EXIT_MALFORMED: "malformed", EXIT_INSUFFICIENT: "insufficient"}


def print_protocol_sha(path: Path) -> int:
    """Print section 7.1's sha256; 0 ok, 1 malformed markers, 2 missing file."""
    from alpha_zero.evaluation.a1_protocol import ProtocolMarkerError, protocol_sha256

    if not path.is_file():
        print(f"MISSING: {path}", file=sys.stderr)
        return EXIT_INSUFFICIENT
    try:
        print(protocol_sha256(path.read_text(encoding="utf-8")))
    except ProtocolMarkerError as exc:
        print(f"MALFORMED: {path}: {exc}", file=sys.stderr)
        return EXIT_MALFORMED
    return EXIT_TREND


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_learning_trend",
        description="Describe a run's score-rate trend (descriptive only; exit 0/1/2).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="A training checkpoint directory: reads its iterations.jsonl and eval_*.json.",
    )
    parser.add_argument("--iterations", type=Path, default=None, help="An iterations.jsonl.")
    parser.add_argument(
        "--eval", type=Path, nargs="+", default=None, help="eval_<iter>.json files."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        nargs="+",
        default=None,
        help="Benchmark result.json files, or directories searched for them.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Also write the report here.")
    parser.add_argument(
        "--print-protocol-sha",
        action="store_true",
        help="Print the sha256 of docs/EVALUATION.md section 7.1 and exit.",
    )
    parser.add_argument(
        "--evaluation-md",
        type=Path,
        default=DEFAULT_EVALUATION_MD,
        help="The EVALUATION.md --print-protocol-sha reads (default: this checkout's).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; see the module docstring for the exit codes."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_protocol_sha:
        return print_protocol_sha(args.evaluation_md)
    if args.run_dir is None and args.iterations is None and not args.eval and not args.benchmark:
        parser.error("name at least one of --run-dir, --iterations, --eval, --benchmark")

    inputs, missing = resolve_inputs(args)
    report, problems = build_report(inputs)
    code = classify(report, problems, missing)
    report["status"] = _STATUS[code]
    report["problems"] = list(problems)
    report["missing"] = list(missing)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    for problem in [*problems, *missing]:
        logger.warning(problem)
    return code


if __name__ == "__main__":
    sys.exit(main())
