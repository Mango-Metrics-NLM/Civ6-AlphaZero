#!/usr/bin/env python3
"""Release Evidence: verify a release's evidence bundle (plan section 6, LF-D8).

``docs/evidence/release_vX_Y_Z/bundle.json`` (schema: ``docs/evidence/SCHEMA.md``,
"Release evidence bundle") lists the component runs of a release -- A1 and, when
it ran, A1-R -- and, for each, its ``run.json``, ``iterations.jsonl`` and the
``result.json`` of every arena arm, each with a sha256. This script trusts none
of the stored numbers:

* **Integrity.** Every component is re-hashed. Each arm's row is re-validated by
  the benchmark's own ``validate_envelope``, and its counts,
  ``scoring.summarize`` and both intervals (iid ternary and plain pentanomial,
  with their hull, at 95% and 99%) are recomputed from the per-game records and
  compared with the row and with the bundle. The verdict is **recomputed** with
  ``a1_protocol.compute_verdict`` (the section 7.1 order: VOID, PASS, FAIL,
  INCONCLUSIVE) and must equal the stored one.
* **Check runs.** The rc commit's CI is read the way ``wait-for-ci`` reads it
  (``release_checks.ci_snapshot``, REST API 2022-11-28): the producing PR into
  ``qa``, its Branch Direction, and every push check green.
* **Budget.** Each component run (by ``run_id``) must be priced at <= 1,000 games
  or carry its own ``approval_ref`` in ``run.json``. The bundle total is reported,
  not gated (A1 + A1-R = 1,960).
* **Freshness.** Every component's ``git_sha`` is the run's pinned commit and an
  ancestor of ``rc_sha``; ``git diff <git_sha> <rc_sha>`` touches nothing in
  ``scripts/release_play_path.txt``; ``run.json`` names no ``MCTS_*`` variable.
* **Learning.** ``learning_recorded`` is required: the bundle is schema-valid, the
  protocol hash is section 7.1's at the run's commit and at HEAD, the verdict is
  recomputed, and ``docs/evidence/README.md`` has the row. ``learning_claimed``
  must equal "A1 and A1-R both PASS" -- the only thing that permits "learns"
  wording, so a FAIL ships the release with the claim withheld.

Exit status: 0 pass, 1 fail, 2 missing (no bundle, a missing component, or a git
object or API answer the check cannot do without). ``--scope`` decides instead
whether the CI job applies at all (base ``main``, head not ``hotfix/*``) and
always exits 0.

Like Branch Direction, the CI job runs this file from the PR's own checkout, so
it is an accident net; the controls are maintainer-only merges into main and the
release gates (``wait-for-ci`` requires this check for a non-hotfix final).

Usage:
    python scripts/verify_release_evidence.py [BUNDLE] [--repo OWNER/NAME]
    python scripts/verify_release_evidence.py --scope   # CI: EVENT_NAME, BASE_REF, HEAD_REF
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

# Run THIS checkout's code (as benchmark_suite.py does): the repository root
# heads sys.path whether this runs as a script or as ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_branch_direction import Api, Git, GitError, gh_api
from scripts.release_checks import ci_snapshot
from scripts.release_version import ReleaseError, parse_tag, pyproject_version

EXIT_PASS, EXIT_FAIL, EXIT_MISSING = 0, 1, 2
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_FILE = "bundle.json"
REPORT_FILE = "report.md"
PLAY_PATH_FILE = Path(__file__).with_name("release_play_path.txt")
EVALUATION_MD = "docs/EVALUATION.md"
EVIDENCE_INDEX = "docs/evidence/README.md"
SEARCH_EPOCH = "post-2026-09-04"
#: Game cap per component run without an ``approval_ref`` (CLAUDE.md section 3.2).
MAX_GAMES_WITHOUT_APPROVAL = 1000
TOLERANCE = 1e-9

#: Component roles and their confirmatory seeds (section 7.1, "Seeds").
ROLE_SEEDS = {"A1": 20261111, "A1-R": 20261112}
#: Both must PASS before any "learns" claim (section 7.1, "Follow-ups").
CLAIM_ROLES = ("A1", "A1-R")
#: Arena arms (section 7.1, "Registered commands"): allowed game counts. The
#: positive control and "final vs random" move together (the 40/60 branch).
ARM_GAMES: dict[str, tuple[int, ...]] = {
    "primary": (200,),
    "symmetry": (40,),
    "positive_control": (40, 60),
    "secondary_5": (60,),
    "secondary_10": (60,),
    "secondary_15": (60,),
    "final_vs_random": (40, 20),
}
BRANCH_FINAL_GAMES = {40: 40, 60: 20}
#: The confidence each arm's bundle summary quotes its hull at.
ARM_CONFIDENCE = {"symmetry": 0.99}
#: The arena search and recipe every arm must record (section 7.1, "Arena search").
REGISTERED_PROTOCOL = {
    "board_size": 8,
    "max_turns": 60,
    "num_simulations": 24,
    "gumbel_k": 8,
    "mcts_type": "gumbel",
    "gameplay_preset": "default",
}
TRAINING_ITERATIONS = 20
CHECKPOINT = "checkpoint:"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MENTION = re.compile(r"`[^`]*`|\"[^\"]*\"")
_LEARNS = re.compile(r"\blearns\b", re.IGNORECASE)


class MissingEvidenceError(Exception):
    """Something the verdict needs is absent: exit 2, not a judgement."""


@dataclass
class Findings:
    problems: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def exit_code(self) -> int:
        if self.missing:
            return EXIT_MISSING
        return EXIT_FAIL if self.problems else EXIT_PASS


# ---------------------------------------------------------------------------
# Scope (the CI job's n/a decision)
# ---------------------------------------------------------------------------


def scope(event_name: str, base_ref: str, head_ref: str) -> tuple[bool, str]:
    """Does this event need release evidence? ``(applies, why)``.

    Only a pull request into ``main`` from anything but ``hotfix/*`` -- a release
    PR -- needs it. Pushes and every other base are n/a.
    """
    if event_name != "pull_request":
        return False, f"n/a: event {event_name or '(none)'!r} is not a pull request"
    if base_ref != "main":
        return False, f"n/a: base {base_ref or '(none)'!r} is not main"
    if head_ref.startswith("hotfix/"):
        return False, f"n/a: head {head_ref!r} is a hotfix"
    return True, f"applies: {head_ref!r} -> main is a release"


def _report_scope(env: Mapping[str, str]) -> int:
    applies, why = scope(
        env.get("EVENT_NAME", ""), env.get("BASE_REF", ""), env.get("HEAD_REF", "")
    )
    print(f"Release Evidence: {why}")
    for name, text in (
        ("GITHUB_OUTPUT", f"applies={'true' if applies else 'false'}\n"),
        ("GITHUB_STEP_SUMMARY", f"## Release Evidence\n\n{why}\n"),
    ):
        target = env.get(name)
        if target:
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(text)
    return EXIT_PASS


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_play_path(path: Path = PLAY_PATH_FILE) -> list[str]:
    """The play-path pathspecs, one per non-comment line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [s for s in (line.strip() for line in lines) if s and not s.startswith("#")]


def bundle_dir_name(release: str) -> str:
    """``2.1.0`` -> ``release_v2_1_0`` (evidence directory names use underscores)."""
    return "release_v" + release.replace(".", "_")


def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=TOLERANCE)
    except (TypeError, ValueError):
        return False


def _close_pair(a: Any, b: Any) -> bool:
    return (
        isinstance(a, list | tuple)
        and isinstance(b, list | tuple)
        and len(a) == len(b) == 2
        and all(_close(x, y) for x, y in zip(a, b, strict=True))
    )


def _component_path(bundle_dir: Path, rel: Any) -> Path:
    """A bundle-relative POSIX path; absolute paths and ``..`` are refused."""
    if not isinstance(rel, str) or not rel:
        raise ValueError(f"component path {rel!r} is not a non-empty string")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts or "\\" in rel or ":" in rel:
        raise ValueError(f"component path {rel!r} must be relative to the bundle, without '..'")
    return bundle_dir.joinpath(*pure.parts)


def _load_component(bundle_dir: Path, entry: Any, label: str, f: Findings) -> Path | None:
    """Resolve, check for, and re-hash one component; ``None`` if it cannot be read."""
    if not isinstance(entry, Mapping):
        f.problems.append(f"{label}: expected {{path, sha256}}, got {type(entry).__name__}")
        return None
    try:
        path = _component_path(bundle_dir, entry.get("path"))
    except ValueError as exc:
        f.problems.append(f"{label}: {exc}")
        return None
    if not path.is_file():
        f.missing.append(f"{label}: {entry.get('path')} does not exist")
        return None
    recorded = str(entry.get("sha256", ""))
    if not _SHA256.match(recorded):
        f.problems.append(f"{label}: sha256 {recorded!r} is not 64 hex")
    elif sha256_file(path) != recorded:
        f.problems.append(f"{label}: {entry.get('path')} does not hash to the recorded sha256")
    return path


def _read_json(path: Path, label: str, f: Findings) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        f.problems.append(f"{label}: not readable JSON: {exc}")
        return None


# ---------------------------------------------------------------------------
# run.json and iterations.jsonl
# ---------------------------------------------------------------------------


@dataclass
class RunFacts:
    """What the run's own records say, before any arm is read."""

    sha: str | None = None
    games: int | None = None
    approval_ref: str | None = None
    priced: dict[str, int] = field(default_factory=dict)
    #: Invocation name -> the argument lists it may have been launched with
    #: (``argv``, and ``launched_argv`` with run_card's ``--max_games`` appended).
    argv: dict[str, list[list[str]]] = field(default_factory=dict)
    interruptions: list[str] = field(default_factory=list)
    mcts_names: list[str] = field(default_factory=list)


def check_run_json(record: Any, run_id: str, label: str, f: Findings) -> RunFacts:
    facts = RunFacts()
    if not isinstance(record, Mapping):
        f.problems.append(f"{label}: run.json is not an object")
        return facts
    if record.get("schema_version") != 1:
        f.problems.append(f"{label}: run.json schema_version {record.get('schema_version')!r} != 1")
    if record.get("card_id") != run_id:
        f.problems.append(f"{label}: run.json card_id {record.get('card_id')!r} != run_id")
    sha = record.get("sha")
    if isinstance(sha, str) and _SHA40.match(sha):
        facts.sha = sha
    else:
        f.problems.append(f"{label}: run.json sha {sha!r} is not a full commit sha")
    status = record.get("status")
    if status != "COMPLETED":
        # RUNNING left behind by a hard kill is read as VOID (section 7.1).
        facts.interruptions.append(f"run.json status is {status!r}, not 'COMPLETED'")
    names = record.get("env_var_names")
    if not isinstance(names, list):
        f.problems.append(f"{label}: run.json env_var_names is not a list")
        names = []
    facts.mcts_names = sorted(str(n) for n in names if str(n).startswith("MCTS_"))
    if facts.mcts_names:
        facts.interruptions.append(f"run.json names MCTS_ variable(s) {facts.mcts_names}")
    approval = record.get("approval_ref")
    facts.approval_ref = (str(approval).strip() or None) if approval is not None else None
    invocations = record.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        f.problems.append(f"{label}: run.json records no invocations")
        return facts
    for inv in invocations:
        if not isinstance(inv, Mapping) or not isinstance(inv.get("priced_games"), int):
            f.problems.append(f"{label}: run.json invocation without integer priced_games")
            return facts
        facts.priced[str(inv.get("name"))] = int(inv["priced_games"])
        facts.argv[str(inv.get("name"))] = [
            [str(a) for a in inv[key]]
            for key in ("argv", "launched_argv")
            if isinstance(inv.get(key), list)
        ]
        if inv.get("exit_code") != 0:
            f.problems.append(
                f"{label}: invocation {inv.get('name')!r} exit_code {inv.get('exit_code')!r}"
            )
    facts.games = sum(facts.priced.values())
    if record.get("total_priced_games") != facts.games:
        f.problems.append(
            f"{label}: total_priced_games {record.get('total_priced_games')!r} != "
            f"the invocations' sum {facts.games}"
        )
    return facts


def check_iterations(
    text: str, seed: int, sha: str | None, label: str, f: Findings, facts: RunFacts
) -> None:
    seen: set[int] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            f.problems.append(f"{label}: iterations.jsonl line {number} is not JSON")
            continue
        if not isinstance(rec, dict):
            f.problems.append(f"{label}: iterations.jsonl line {number} is not an object")
            continue
        if rec.get("resumed_from") is not None:
            facts.interruptions.append(f"iterations.jsonl line {number} was resumed")
        if rec.get("supersedes_earlier_record"):
            facts.interruptions.append(f"iterations.jsonl line {number} supersedes an earlier one")
        if rec.get("seed") != seed:
            f.problems.append(f"{label}: iterations.jsonl line {number} seed {rec.get('seed')!r}")
        if sha is not None and rec.get("git_sha") != sha:
            f.problems.append(
                f"{label}: iterations.jsonl line {number} git_sha {rec.get('git_sha')!r} "
                "is not the run's pinned sha"
            )
        if isinstance(rec.get("iteration"), int):
            seen.add(rec["iteration"])
    expected = set(range(1, TRAINING_ITERATIONS + 1))
    if seen != expected:
        f.problems.append(
            f"{label}: iterations.jsonl covers {sorted(seen)}, not 1..{TRAINING_ITERATIONS}"
        )


# ---------------------------------------------------------------------------
# Arms: integrity and recomputation
# ---------------------------------------------------------------------------


def command_args(command: str) -> list[str] | None:
    """The arguments of a recorded benchmark command, without its entry point.

    ``evidence.command`` is ``shlex``-joined and starts ``python -m
    alpha_zero.evaluation.benchmark`` under run_card or ``python
    scripts/benchmark_suite.py`` by hand; both take the same arguments
    (section 7.1: compare the arguments, not the leading tokens).
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if len(tokens) >= 3 and tokens[1] == "-m":
        return tokens[3:]
    return tokens[2:] if len(tokens) >= 2 else None


def recompute_row(row: Mapping[str, Any]) -> list[str]:
    """Differences between a result row's stored statistics and its per-game records."""
    from alpha_zero.evaluation.benchmark import REPORTED_CONFIDENCE_LEVELS
    from alpha_zero.evaluation.benchmark_evidence import facing_hull_ratio, interval_summary
    from alpha_zero.evaluation.scoring import minimum_detectable_effect, summarize

    games = row["games"]
    results = [g["result"] for g in games]
    wins, draws, losses = (results.count(r) for r in ("win", "draw", "loss"))
    out = []
    counts = (row.get("candidate_wins"), row.get("draws"), row.get("champion_wins"))
    if counts != (wins, draws, losses) or row.get("num_games") != len(results):
        out.append(f"stored counts {counts} != per-game records {(wins, draws, losses)}")
    summary = summarize(wins, draws, losses, float(row.get("confidence", 0.95)))
    for key in ("score_rate", "ci_low", "ci_high", "draw_rate", "decisive_win_rate"):
        if not _close(row.get(key), getattr(summary, key)):
            out.append(f"{key} {row.get(key)!r} != recomputed {getattr(summary, key)!r}")
    intervals = {str(c): interval_summary(games, c) for c in REPORTED_CONFIDENCE_LEVELS}
    stored = row.get("intervals") or {}
    for level, block in intervals.items():
        for kind, bounds in block.items():
            if not _close_pair((stored.get(level) or {}).get(kind), bounds):
                out.append(f"intervals[{level}][{kind}] != recomputed {bounds}")
    ratio = facing_hull_ratio(
        summary.score_rate, intervals["0.95"]["iid"], intervals["0.95"]["hull"]
    )
    mde = minimum_detectable_effect(len(results), summary.draw_rate, alpha=0.025)
    for key, value in (
        ("hull_ratio", ratio),
        ("mde_alpha025_80pct", mde),
        ("mde_alpha025_80pct_hull", mde * ratio),
    ):
        if not _close(row.get(key), value):
            out.append(f"{key} {row.get(key)!r} != recomputed {value!r}")
    return out


def arm_summary(row: Mapping[str, Any], confidence: float) -> dict[str, Any]:
    """What the bundle quotes for one arm, recomputed from its per-game records."""
    from alpha_zero.evaluation.benchmark_evidence import interval_summary

    results = [g["result"] for g in row["games"]]
    wins, draws, losses = (results.count(r) for r in ("win", "draw", "loss"))
    total = len(results)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_rate": (wins + 0.5 * draws) / total if total else 0.0,
        "confidence": confidence,
        "hull": list(interval_summary(row["games"], confidence)["hull"]),
    }


@dataclass
class ArmRead:
    games: list[dict[str, Any]]
    candidate: str
    opponent: str


def check_arm(
    name: str,
    entry: Mapping[str, Any],
    envelope: Any,
    *,
    seed: int,
    sha: str | None,
    facts: RunFacts,
    label: str,
    f: Findings,
) -> ArmRead | None:
    from alpha_zero.evaluation.benchmark import validate_envelope

    try:
        validate_envelope(envelope)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        f.problems.append(f"{label}: {exc}")
        return None
    rows = envelope["results"]
    if len(rows) != 1:
        f.problems.append(f"{label}: expected one result row, found {len(rows)}")
        return None
    if envelope["shard"] is not None:
        f.problems.append(f"{label}: a shard, not a merged result")
    row = rows[0]
    f.problems.extend(f"{label}: {p}" for p in recompute_row(row))
    protocol = envelope["protocol"]
    for key, want in REGISTERED_PROTOCOL.items():
        if protocol.get(key) != want:
            f.problems.append(f"{label}: protocol {key} {protocol.get(key)!r} != {want!r}")
    if protocol.get("n_games") not in ARM_GAMES[name]:
        f.problems.append(f"{label}: n_games {protocol.get('n_games')!r} not in {ARM_GAMES[name]}")
    invocation = name.replace("_", "-")
    if facts.priced.get(invocation) != row["num_games"]:
        f.problems.append(
            f"{label}: {row['num_games']} games, but run.json priced {invocation!r} at "
            f"{facts.priced.get(invocation)!r}"
        )
    evidence = envelope["evidence"]
    if command_args(str(evidence.get("command", ""))) not in facts.argv.get(invocation, []):
        f.problems.append(
            f"{label}: evidence.command is not run.json's {invocation!r} invocation "
            "(was this result written by another arm or run?)"
        )
    if evidence.get("seed") != seed:
        f.problems.append(f"{label}: evidence.seed {evidence.get('seed')!r} != {seed}")
    if sha is not None and evidence.get("git_sha") != sha:
        f.problems.append(f"{label}: evidence.git_sha is not the run's pinned sha")
    if evidence.get("device") != "cpu":
        f.problems.append(f"{label}: device {evidence.get('device')!r}; confirmatory runs are CPU")
    search = evidence.get("search") or {}
    for key in ("mcts_type", "num_simulations", "gumbel_k"):
        if search.get(key) != REGISTERED_PROTOCOL[key]:
            f.problems.append(f"{label}: evidence.search {key} {search.get(key)!r}")
    confidence = ARM_CONFIDENCE.get(name, 0.95)
    stored = {
        k: entry.get(k) for k in ("wins", "draws", "losses", "score_rate", "confidence", "hull")
    }
    recomputed = arm_summary(row, confidence)
    for key, value in recomputed.items():
        if key == "hull":
            same = _close_pair(stored[key], value)
        elif key in ("score_rate", "confidence"):
            same = _close(stored[key], value)
        else:
            same = stored[key] == value
        if not same:
            f.problems.append(f"{label}: bundle {key} {stored[key]!r} != recomputed {value!r}")
    return ArmRead(
        games=list(row["games"]),
        candidate=str(envelope["candidate"].get("id")),
        opponent=str(row.get("opponent_id")),
    )


def check_arm_identities(arms: Mapping[str, ArmRead], label: str, f: Findings) -> None:
    """The arms play the checkpoints and baselines their commands name."""
    primary, symmetry = arms["primary"], arms["symmetry"]
    c0, c20 = primary.opponent, primary.candidate
    expect = {
        "primary": c20.startswith(CHECKPOINT) and c0.startswith(CHECKPOINT) and c0 != c20,
        "symmetry": symmetry.candidate == symmetry.opponent == c0,
        "positive_control": (arms["positive_control"].candidate, arms["positive_control"].opponent)
        == ("heuristic", "random"),
        "final_vs_random": (arms["final_vs_random"].candidate, arms["final_vs_random"].opponent)
        == (c20, "random"),
    }
    for n in ("secondary_5", "secondary_10", "secondary_15"):
        expect[n] = (
            arms[n].opponent == c0
            and arms[n].candidate.startswith(CHECKPOINT)
            and (arms[n].candidate not in (c0, c20))
        )
    for name, ok in expect.items():
        if not ok:
            f.problems.append(
                f"{label}: arm {name} plays {arms[name].candidate} vs {arms[name].opponent}, "
                "not what its registered command names"
            )
    pc = len(arms["positive_control"].games)
    if BRANCH_FINAL_GAMES.get(pc) != len(arms["final_vs_random"].games):
        f.problems.append(
            f"{label}: positive control {pc} games with final vs random "
            f"{len(arms['final_vs_random'].games)}: not a registered branch (40/40 or 60/20)"
        )


# ---------------------------------------------------------------------------
# One component run
# ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    role: str
    run_id: str
    verdict: str | None
    games: int | None
    sha: str | None


def verify_run(
    run: Any, bundle_dir: Path, git: Git, rc_sha: str, play_path: Sequence[str], f: Findings
) -> RunOutcome | None:
    from alpha_zero.evaluation.a1_protocol import compute_verdict

    if not isinstance(run, Mapping):
        f.problems.append("runs[]: a run entry is not an object")
        return None
    run_id, role = str(run.get("run_id", "")), str(run.get("role", ""))
    label = f"run {run_id or '?'} ({role or '?'})"
    if role not in ROLE_SEEDS:
        f.problems.append(f"{label}: role must be one of {sorted(ROLE_SEEDS)}")
        return None
    seed = ROLE_SEEDS[role]
    if run.get("seed") != seed:
        f.problems.append(f"{label}: seed {run.get('seed')!r} != {seed} for {role}")

    run_path = _load_component(bundle_dir, run.get("run_json"), f"{label} run.json", f)
    facts = RunFacts()
    if run_path is not None:
        facts = check_run_json(_read_json(run_path, label, f), run_id, label, f)
    if facts.sha is not None and run.get("git_sha") != facts.sha:
        f.problems.append(f"{label}: bundle git_sha {run.get('git_sha')!r} != run.json sha")
    sha = facts.sha
    iter_path = _load_component(bundle_dir, run.get("iterations"), f"{label} iterations", f)
    if iter_path is not None:
        check_iterations(iter_path.read_text(encoding="utf-8"), seed, sha, label, f, facts)

    arms_in = run.get("arms")
    if not isinstance(arms_in, Mapping) or set(arms_in) != set(ARM_GAMES):
        f.problems.append(f"{label}: arms must be exactly {sorted(ARM_GAMES)}")
        arms_in = arms_in if isinstance(arms_in, Mapping) else {}
    arms: dict[str, ArmRead] = {}
    for name in ARM_GAMES:
        entry = arms_in.get(name)
        path = _load_component(bundle_dir, entry, f"{label} arm {name}", f)
        if path is None or not isinstance(entry, Mapping):
            continue
        envelope = _read_json(path, f"{label} arm {name}", f)
        read = check_arm(
            name, entry, envelope, seed=seed, sha=sha, facts=facts,
            label=f"{label} arm {name}", f=f,
        )  # fmt: skip
        if read is not None:
            arms[name] = read

    check_budget(run, facts, label, f)
    if sha is not None:
        check_freshness(git, sha, rc_sha, play_path, label, f)
    if facts.mcts_names:
        f.problems.append(f"{label}: MCTS_* overrides {facts.mcts_names} (section 7.1)")

    verdict: str | None = None
    if set(arms) == set(ARM_GAMES):
        check_arm_identities(arms, label, f)
        computed = compute_verdict(
            arms["primary"].games,
            arms["symmetry"].games,
            arms["positive_control"].games,
            interruptions=facts.interruptions,
        )
        verdict = computed.verdict
        compare_verdict(run.get("verdict"), computed.to_dict(), label, f)
        f.notes.append(f"{label}: verdict {verdict} ({'; '.join(computed.reasons)})")
    return RunOutcome(role, run_id, verdict, facts.games, sha)


def compare_verdict(stored: Any, computed: Mapping[str, Any], label: str, f: Findings) -> None:
    if not isinstance(stored, Mapping):
        f.problems.append(f"{label}: no stored verdict object")
        return
    if stored.get("verdict") != computed["verdict"]:
        f.problems.append(
            f"{label}: stored verdict {stored.get('verdict')!r} != recomputed "
            f"{computed['verdict']!r} ({'; '.join(computed['reasons'])})"
        )
        return
    for key in ("mde_alpha025_80pct", "hull_ratio", "mde_alpha025_80pct_hull"):
        if not _close(stored.get(key), computed[key]):
            f.problems.append(f"{label}: verdict {key} {stored.get(key)!r} != {computed[key]!r}")
    if not _close_pair(stored.get("primary_hull"), computed["primary_hull"]):
        f.problems.append(f"{label}: verdict primary_hull != {computed['primary_hull']}")


def check_budget(run: Mapping[str, Any], facts: RunFacts, label: str, f: Findings) -> None:
    if facts.games is None:
        return
    if run.get("games") != facts.games:
        f.problems.append(f"{label}: bundle games {run.get('games')!r} != run.json {facts.games}")
    if (run.get("approval_ref") or None) != facts.approval_ref:
        f.problems.append(f"{label}: bundle approval_ref differs from run.json's")
    if facts.games > MAX_GAMES_WITHOUT_APPROVAL and not facts.approval_ref:
        f.problems.append(
            f"{label}: {facts.games:,} games exceeds {MAX_GAMES_WITHOUT_APPROVAL:,} without an "
            "approval_ref in its own run.json"
        )


def check_freshness(
    git: Git, sha: str, rc_sha: str, play_path: Sequence[str], label: str, f: Findings
) -> None:
    if not git.exists(sha):
        raise MissingEvidenceError(f"{label}: commit {sha[:12]} is not in this repository")
    if not git.is_ancestor(sha, rc_sha):
        f.problems.append(f"{label}: git_sha {sha[:12]} is not an ancestor of rc_sha")
        return
    specs = [f":(glob){p}" for p in play_path]
    changed = git.lines("diff", "--name-only", "--no-renames", sha, rc_sha, "--", *specs)
    if changed:
        f.problems.append(
            f"{label}: the play path changed between {sha[:12]} and rc_sha: {', '.join(changed)}"
        )


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------

_TOP_KEYS = (
    "bundle_schema_version",
    "release",
    "rc_tag",
    "rc_sha",
    "protocol_sha256",
    "runs",
    "games_total",
    "learning_recorded",
    "learning_claimed",
    "evidence",
)
_EVIDENCE_KEYS = (
    "schema_version",
    "git_sha",
    "generated_utc",
    "search_epoch",
    "seed",
    "command",
    "hardware",
    "games_total",
    "claims",
    "non_claims",
)


def check_top_level(bundle: Mapping[str, Any], bundle_dir: Path, f: Findings) -> bool:
    missing_keys = [k for k in _TOP_KEYS if k not in bundle]
    if missing_keys:
        f.problems.append(f"bundle.json is missing {missing_keys}")
        return False
    if bundle["bundle_schema_version"] != BUNDLE_SCHEMA_VERSION:
        f.problems.append(f"bundle_schema_version {bundle['bundle_schema_version']!r} != 1")
    release = str(bundle["release"])
    if bundle_dir.name != bundle_dir_name(release):
        f.problems.append(f"bundle directory {bundle_dir.name!r} != {bundle_dir_name(release)!r}")
    try:
        tag = parse_tag(str(bundle["rc_tag"]))
    except ReleaseError as exc:
        f.problems.append(f"rc_tag: {exc}")
        return False
    if not tag.is_rc or tag.base != release:
        f.problems.append(f"rc_tag {tag.name} is not an rc of {release}")
    if not _SHA40.match(str(bundle["rc_sha"])):
        f.problems.append(f"rc_sha {bundle['rc_sha']!r} is not a full commit sha")
        return False
    if not isinstance(bundle["runs"], list) or not bundle["runs"]:
        f.problems.append("runs must be a non-empty list")
        return False
    evidence = bundle["evidence"]
    if not isinstance(evidence, Mapping):
        f.problems.append("evidence must be an object")
        return True
    absent = [k for k in _EVIDENCE_KEYS if k not in evidence]
    if absent:
        f.problems.append(f"evidence block is missing {absent}")
    if evidence.get("search_epoch") != SEARCH_EPOCH:
        f.problems.append(
            f"evidence.search_epoch {evidence.get('search_epoch')!r} != {SEARCH_EPOCH}"
        )
    if evidence.get("games_total") != bundle["games_total"]:
        f.problems.append("evidence.games_total != games_total")
    return True


def check_release_refs(
    bundle: Mapping[str, Any], git: Git, head: str, repo_root: Path, f: Findings
) -> None:
    rc_tag, rc_sha = str(bundle["rc_tag"]), str(bundle["rc_sha"])
    if not git.exists(rc_sha):
        raise MissingEvidenceError(
            f"rc_sha {rc_sha[:12]} is not in this repository (fetch-depth 0?)"
        )
    if not git.exists(f"refs/tags/{rc_tag}"):
        raise MissingEvidenceError(f"tag {rc_tag} is not in this repository (fetch tags)")
    if git.resolve(f"refs/tags/{rc_tag}") != rc_sha:
        f.problems.append(f"{rc_tag} does not point at rc_sha {rc_sha[:12]}")
    if not git.is_ancestor(rc_sha, head):
        f.problems.append(f"rc_sha {rc_sha[:12]} is not an ancestor of {head}")
    try:
        declared = pyproject_version((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, ReleaseError) as exc:
        f.problems.append(f"cannot read the version from pyproject.toml: {exc}")
        return
    if declared != str(bundle["release"]):
        f.problems.append(f"bundle release {bundle['release']!r} != pyproject.toml {declared!r}")


def check_protocol(
    bundle: Mapping[str, Any], git: Git, shas: Sequence[str], repo_root: Path, f: Findings
) -> None:
    """The bundle's protocol hash is section 7.1's at HEAD and at every run's commit."""
    from alpha_zero.evaluation.a1_protocol import ProtocolMarkerError, protocol_sha256

    recorded = str(bundle["protocol_sha256"])
    sources: list[tuple[str, Callable[[], str]]] = [
        ("HEAD", lambda: (repo_root / EVALUATION_MD).read_text(encoding="utf-8")),
    ]
    for sha in dict.fromkeys(shas):
        sources.append((sha[:12], functools.partial(git.out, "show", f"{sha}:{EVALUATION_MD}")))
    for where, read in sources:
        try:
            digest = protocol_sha256(read())
        except (OSError, GitError, ProtocolMarkerError) as exc:
            f.problems.append(f"protocol hash at {where}: {exc}")
            continue
        if digest != recorded:
            f.problems.append(
                f"protocol_sha256 {recorded[:12]}... != section 7.1 at {where} ({digest[:12]}...)"
            )


def check_learning(
    bundle: Mapping[str, Any], outcomes: Sequence[RunOutcome], bundle_dir: Path, f: Findings
) -> None:
    roles = [o.role for o in outcomes]
    if len(set(roles)) != len(roles):
        f.problems.append(f"each role appears at most once; got {roles}")
    if "A1" not in roles:
        f.problems.append("the bundle has no A1 run")
    by_role = {o.role: o.verdict for o in outcomes}
    claimed = all(by_role.get(r) == "PASS" for r in CLAIM_ROLES)
    if bundle["learning_recorded"] is not True:
        f.problems.append(
            "learning_recorded is required (true): the verdict is recorded either way"
        )
    if bundle["learning_claimed"] is not claimed:
        f.problems.append(
            f"learning_claimed is {bundle['learning_claimed']!r}, but A1/A1-R verdicts "
            f"{by_role} {'permit' if claimed else 'do not permit'} the claim"
        )
    report = bundle_dir / REPORT_FILE
    if not report.is_file():
        f.missing.append(f"{REPORT_FILE} is missing from {bundle_dir.name}/")
    elif not claimed:
        used = [
            line.strip()
            for line in report.read_text(encoding="utf-8").splitlines()
            if _LEARNS.search(_MENTION.sub("", line))
        ]
        if used:
            f.problems.append(
                f"{REPORT_FILE} uses 'learns' wording without learning_claimed: {used[0]!r}"
            )


def check_index(repo_root: Path, bundle_dir: Path, f: Findings) -> None:
    index = repo_root / EVIDENCE_INDEX
    if not index.is_file():
        f.missing.append(f"{EVIDENCE_INDEX} does not exist")
    elif f"]({bundle_dir.name}/" not in index.read_text(encoding="utf-8"):
        f.problems.append(f"{EVIDENCE_INDEX} has no row linking {bundle_dir.name}/")


def verify(
    bundle_path: Path,
    *,
    repo_root: Path,
    git: Git,
    api: Api | None,
    repo: str,
    head: str = "HEAD",
    play_path: Sequence[str] | None = None,
) -> Findings:
    """Every section 6 check on one bundle. The returned findings decide the exit code."""
    f = Findings()
    if not bundle_path.is_file():
        f.missing.append(
            f"no release evidence bundle at {bundle_path}; a release PR into main needs "
            "docs/evidence/release_vX_Y_Z/bundle.json (docs/evidence/SCHEMA.md)"
        )
        return f
    bundle = _read_json(bundle_path, BUNDLE_FILE, f)
    if not isinstance(bundle, Mapping):
        f.problems.append("bundle.json is not an object")
        return f
    bundle_dir = bundle_path.parent
    if not check_top_level(bundle, bundle_dir, f):
        return f
    play = list(play_path) if play_path is not None else read_play_path()
    try:
        check_release_refs(bundle, git, head, repo_root, f)
        outcomes = [
            o
            for run in bundle["runs"]
            if (o := verify_run(run, bundle_dir, git, str(bundle["rc_sha"]), play, f)) is not None
        ]
        check_protocol(bundle, git, [o.sha for o in outcomes if o.sha], repo_root, f)
    except (MissingEvidenceError, GitError) as exc:
        f.missing.append(str(exc))
        return f
    total = sum(o.games or 0 for o in outcomes)
    if bundle["games_total"] != total:
        f.problems.append(f"games_total {bundle['games_total']!r} != the runs' sum {total}")
    f.notes.append(
        f"bundle total: {total:,} games across {len(outcomes)} run(s) (reported, not gated)"
    )
    check_learning(bundle, outcomes, bundle_dir, f)
    check_index(repo_root, bundle_dir, f)
    check_runs(bundle, api, repo, f)
    return f


def check_runs(bundle: Mapping[str, Any], api: Api | None, repo: str, f: Findings) -> None:
    """The rc commit's checks, read as ``wait-for-ci`` reads them (one pass, no polling)."""
    if api is None or not repo:
        f.missing.append("check runs: --repo (or GITHUB_REPOSITORY) is required to read them")
        return
    tag = parse_tag(str(bundle["rc_tag"]))
    failures, waiting = ci_snapshot(api, repo, str(bundle["rc_sha"]), tag)
    f.problems.extend(f"check runs: {p}" for p in failures)
    f.problems.extend(f"check runs: not green: {w}" for w in waiting)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_bundle(repo_root: Path) -> Path:
    """``docs/evidence/release_vX_Y_Z/bundle.json`` for pyproject.toml's version."""
    version = pyproject_version((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    return repo_root / "docs" / "evidence" / bundle_dir_name(version) / BUNDLE_FILE


def main(
    argv: Sequence[str] | None = None,
    *,
    api: Api | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", nargs="?", type=Path, help="default: from pyproject.toml")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--repo", default=None, help="OWNER/NAME (default: GITHUB_REPOSITORY)")
    parser.add_argument("--head", default="HEAD", help="the commit rc_sha must be an ancestor of")
    parser.add_argument("--scope", action="store_true", help="CI: print n/a or applies; exit 0")
    args = parser.parse_args(argv)
    env = os.environ if env is None else env
    if args.scope:
        return _report_scope(env)
    repo_root = args.repo_root.resolve()
    try:
        bundle = args.bundle if args.bundle is not None else default_bundle(repo_root)
    except (OSError, ReleaseError) as exc:
        print(f"::error::Release Evidence: {exc}", file=sys.stderr)
        return EXIT_MISSING
    if not bundle.is_absolute():
        bundle = (Path.cwd() / bundle).resolve()
    findings = verify(
        bundle,
        repo_root=repo_root,
        git=Git(repo_root),
        api=api or gh_api,
        repo=args.repo if args.repo is not None else env.get("GITHUB_REPOSITORY", ""),
        head=args.head,
    )
    for note in findings.notes:
        print(f"Release Evidence: {note}")
    for item in findings.missing:
        print(f"::error::Release Evidence (missing): {item}", file=sys.stderr)
    for item in findings.problems:
        print(f"::error::Release Evidence: {item}", file=sys.stderr)
    code = findings.exit_code()
    print(f"Release Evidence: {('pass', 'FAIL', 'MISSING')[code]} ({bundle})")
    return code


if __name__ == "__main__":
    sys.exit(main())
