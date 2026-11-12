#!/usr/bin/env python3
"""Release Evidence: verify a release's evidence bundle (plan section 6, LF-D8).

``docs/evidence/release_vX_Y_Z/bundle.json`` (schema: ``docs/evidence/SCHEMA.md``,
"Release evidence bundle") lists the component runs of a release -- every
attempt at A1 and, when it ran, A1-R -- and, for each, its ``run.json``,
``iterations.jsonl`` and the ``result.json`` of every arena arm, each with a
sha256. This script trusts none of the stored numbers:

* **Registration.** Each attempt's ``run.json`` invocations must be, token for
  token and in order, the commands ``docs/EVALUATION.md`` section 7.1 registers
  **at that run's commit** (``a1_registration.registered_invocations``: the base
  block, the 40/60 branch in force, ``<D>`` for the run's ``card_id``, the role's
  seed). An unregistered card -- other training flags, another checkpoint, the
  wrong branch, a branch still "pending T0b" -- fails. The iteration count, the
  arena recipe and each arm's game count are read from those commands.
* **Integrity.** Every component is re-hashed. Each arm's ``evidence.command``
  must be its ``run.json`` invocation as launched (``launched_argv``). Each row is
  re-validated by the benchmark's own ``validate_envelope``; its counts,
  ``scoring.summarize`` and both intervals (iid ternary and plain pentanomial,
  with their hull, at 95% and 99%) are recomputed from the per-game records. The
  verdict is **recomputed** with ``a1_protocol.compute_verdict`` (section 7.1
  order: VOID, PASS, FAIL, INCONCLUSIVE) and must equal the stored one.
* **Check runs.** The rc commit's CI is read the way ``wait-for-ci`` reads it
  (``release_checks.ci_snapshot``, REST API 2022-11-28): the producing PR into
  ``qa``, its Branch Direction, and every push check green.
* **Budget.** Each component run (by ``run_id``) must be priced at <= 1,000 games
  or carry its own ``approval_ref`` in ``run.json``. The bundle total is reported,
  not gated (A1 + A1-R = 1,960).
* **Freshness.** ``rc_tag`` is the highest rc of the release and points at
  ``rc_sha``, an ancestor of the head under verification. Every attempt's
  ``git_sha`` is an ancestor of ``rc_sha``. For each judged attempt, ``git diff
  --no-renames`` from its ``git_sha`` to ``rc_sha`` and to the head touches
  nothing in ``scripts/release_play_path.txt``. An ``MCTS_*`` variable makes the
  attempt VOID.
* **Learning.** ``learning_recorded`` is required. A **VOID records no learning
  result**: every role needs exactly one non-VOID attempt, and that one is
  judged; VOID attempts stay listed beside it (section 7.1, "After a VOID"). An
  attempt whose ``run.json`` is not COMPLETED is VOID by status and lists what it
  left: its invocations must be a prefix of the registered list, its total the
  card's price, and each arm it lists is recomputed (honesty, not completeness). The
  protocol hash is section 7.1's at HEAD and at every attempt's commit, and
  ``docs/evidence/README.md`` has the row. ``learning_claimed`` must equal "the
  judged A1 and A1-R both PASS"; without it, "learns"/"learned"/"learnt" may not
  appear (outside backticks or quotes) in ``report.md``, the index row or the
  release's CHANGELOG section. A FAIL or INCONCLUSIVE ships with the claim
  withheld.

Exit status: 0 pass, 1 fail, 2 missing (no bundle, a missing component, a git
object, or no way to read the check runs: no ``--repo``, ``gh`` not runnable).
A GitHub API error while reading the check runs is a failure (1), as in
``wait-for-ci``. ``--scope`` decides instead whether the CI job applies at all
(base ``main``, head not ``hotfix/*``), always exits 0, and imports only the
standard library, so it runs on a tree that has no other release script.

Like Branch Direction, the CI job (``.github/workflows/release-evidence.yml``)
runs this file from the PR's own checkout, so it is an accident net; the
controls are maintainer-only merges into main and the release gates
(``wait-for-ci`` requires this check for a non-hotfix final).

Usage:
    python scripts/verify_release_evidence.py [BUNDLE] [--repo OWNER/NAME] [--head REV]
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
from typing import TYPE_CHECKING, Any

# Run THIS checkout's code (as benchmark_suite.py does): the repository root
# heads sys.path whether this runs as a script or as ``python -m``. Everything
# below that is not stdlib is imported lazily, after --scope has returned.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if TYPE_CHECKING:
    from scripts.check_branch_direction import Api, Git

EXIT_PASS, EXIT_FAIL, EXIT_MISSING = 0, 1, 2
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_FILE = "bundle.json"
REPORT_FILE = "report.md"
PLAY_PATH_FILE = Path(__file__).with_name("release_play_path.txt")
EVALUATION_MD = "docs/EVALUATION.md"
EVIDENCE_INDEX = "docs/evidence/README.md"
CHANGELOG = "CHANGELOG.md"
SEARCH_EPOCH = "post-2026-09-04"
#: Game cap per component run without an ``approval_ref`` (CLAUDE.md section 3.2).
MAX_GAMES_WITHOUT_APPROVAL = 1000
TOLERANCE = 1e-9
VOID = "VOID"

#: The roles a bundle may hold. A1's seed is the registered training command's;
#: A1-R's is ``a1_registration.A1R_SEED``.
ROLES = ("A1", "A1-R")
#: Both judged attempts must PASS before any "learns" claim (section 7.1, "Follow-ups").
CLAIM_ROLES = ROLES
TRAIN = "train"
#: The arms the verdict reads (section 7.1, "Verdict"); every other registered
#: arm is reported, not judged.
PRIMARY, SYMMETRY, POSITIVE_CONTROL = "primary", "symmetry", "positive_control"
#: The confidence each arm's bundle summary quotes its hull at.
ARM_CONFIDENCE = {SYMMETRY: 0.99}
#: The gameplay preset section 7.1's "Arena search" registers (no flag sets it).
GAMEPLAY_PRESET = "default"
#: Benchmark protocol fields and the registered flag each is read from.
PROTOCOL_FLAGS = {
    "board_size": "--board-size",
    "max_turns": "--max-turns",
    "num_simulations": "--sims",
    "gumbel_k": "--gumbel-k",
    "mcts_type": "--mcts-type",
    "n_games": "--n-games",
}
CHECKPOINT = "checkpoint:"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MENTION = re.compile(r"`[^`]*`|\"[^\"]*\"")
#: A learning claim in prose. "learning" names fields and runs, so it is allowed.
_LEARNS = re.compile(r"\blearn(?:s|ed|t)\b", re.IGNORECASE)


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
# Scope (the CI job's n/a decision) -- standard library only
# ---------------------------------------------------------------------------


def scope(event_name: str, base_ref: str, head_ref: str) -> tuple[bool, str]:
    """Does this event need release evidence? ``(applies, why)``.

    Only a pull request into ``main`` from anything but ``hotfix/*`` -- a release
    PR -- needs it. Every other event and base is n/a.
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


def arm_name(invocation: str) -> str:
    """A registered invocation name as a bundle arm key: ``secondary-5`` -> ``secondary_5``."""
    return invocation.replace("-", "_")


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


# ---------------------------------------------------------------------------
# The registration at a run's commit
# ---------------------------------------------------------------------------


@dataclass
class Registration:
    """What section 7.1 registers for one attempt, read at the attempt's commit."""

    invocations: dict[str, list[str]]
    seed: int

    def flag(self, invocation: str, name: str) -> str | None:
        from alpha_zero.evaluation.a1_registration import flag_value

        return flag_value(self.invocations[invocation], name)

    @property
    def arms(self) -> list[str]:
        return [arm_name(n) for n in self.invocations if n != TRAIN]


def role_seed(section: str, role: str) -> int:
    """A1's seed from the registered training command; A1-R's from ``A1R_SEED``."""
    from alpha_zero.evaluation.a1_registration import (
        A1R_SEED,
        RegistrationError,
        command_block,
        flag_value,
    )

    if role == "A1-R":
        return A1R_SEED
    train = dict(command_block(section, "base")).get(TRAIN)
    value = flag_value(train or [], "--seed")
    if value is None:
        raise RegistrationError("the registered training command has no --seed")
    return int(value)


def read_registration(git: Git, sha: str, run_id: str, role: str) -> Registration:
    from alpha_zero.evaluation.a1_protocol import registered_protocol
    from alpha_zero.evaluation.a1_registration import registered_invocations

    section = registered_protocol(git.out("show", f"{sha}:{EVALUATION_MD}"))
    seed = role_seed(section, role)
    return Registration(registered_invocations(section, card_id=run_id, seed=seed), seed)


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
    #: Invocation name -> ``launched_argv`` (the card's argv plus ``--max_games N``).
    launched: dict[str, list[str]] = field(default_factory=dict)
    argv: dict[str, list[str]] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)
    interruptions: list[str] = field(default_factory=list)
    #: ``run.json`` says COMPLETED. Anything else (VOID, FAILED, a lingering
    #: RUNNING) is VOID by status and is checked for honesty, not completeness:
    #: run_card appends an invocation only once it finishes, so an interrupted
    #: attempt records a prefix of the card and only the arms that ran.
    completed: bool = False


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
    facts.completed = status == "COMPLETED"
    if not facts.completed:
        # RUNNING left behind by a hard kill is read as VOID (section 7.1).
        facts.interruptions.append(f"run.json status is {status!r}, not 'COMPLETED'")
    names = record.get("env_var_names")
    if not isinstance(names, list):
        f.problems.append(f"{label}: run.json env_var_names is not a list")
        names = []
    mcts = sorted(str(n) for n in names if str(n).startswith("MCTS_"))
    if mcts:
        facts.interruptions.append(f"run.json names MCTS_* override(s) {mcts}")
    approval = record.get("approval_ref")
    facts.approval_ref = (str(approval).strip() or None) if approval is not None else None
    invocations = record.get("invocations")
    total = record.get("total_priced_games")
    if not isinstance(invocations, list) or (facts.completed and not invocations):
        f.problems.append(f"{label}: run.json records no invocations")
        return facts
    for index, inv in enumerate(invocations):
        if not isinstance(inv, Mapping) or not isinstance(inv.get("priced_games"), int):
            f.problems.append(f"{label}: run.json invocation without integer priced_games")
            return facts
        name = str(inv.get("name"))
        facts.names.append(name)
        facts.priced[name] = int(inv["priced_games"])
        facts.argv[name] = [str(a) for a in inv.get("argv") or []]
        launched = inv.get("launched_argv")
        expected = [*facts.argv[name], "--max_games", str(inv["priced_games"])]
        if not isinstance(launched, list) or [str(a) for a in launched] != expected:
            f.problems.append(
                f"{label}: invocation {name!r} launched_argv is not its argv plus "
                "--max_games <priced> (was it run by run_card?)"
            )
        facts.launched[name] = expected
        # A FAILED card stops on the invocation that failed, so only its last
        # recorded invocation may carry a non-zero exit code.
        last_of_failed = not facts.completed and index == len(invocations) - 1
        if inv.get("exit_code") != 0 and not (
            last_of_failed and isinstance(inv.get("exit_code"), int)
        ):
            f.problems.append(f"{label}: invocation {name!r} exit_code {inv.get('exit_code')!r}")
    if facts.completed:
        facts.games = sum(facts.priced.values())
        if total != facts.games:
            f.problems.append(
                f"{label}: total_priced_games {total!r} != the invocations' sum {facts.games}"
            )
    elif isinstance(total, int) and not isinstance(total, bool):
        facts.games = total  # the whole card's price, written at start (check_registered)
    else:
        f.problems.append(f"{label}: total_priced_games {total!r} is not an integer")
    return facts


def check_registered(facts: RunFacts, reg: Registration, label: str, f: Findings) -> None:
    """``run.json``'s invocations are section 7.1's, in order, token for token.

    A COMPLETED attempt records every registered invocation. Any other attempt
    records a prefix of them (those that finished), and its ``total_priced_games``
    must be the whole registered card's price, which run_card writes at start.
    """
    registered = list(reg.invocations)
    expected_names = registered if facts.completed else registered[: len(facts.names)]
    if facts.names != expected_names:
        what = "" if facts.completed else "a prefix of "
        f.problems.append(
            f"{label}: run.json invokes {facts.names}, but section 7.1 registers {what}{registered}"
        )
    if not facts.completed and facts.games is not None:
        price = registered_price(reg)
        if facts.games != sum(price.values()):
            f.problems.append(
                f"{label}: total_priced_games {facts.games} != the registered card's price "
                f"{sum(price.values())}"
            )
        for name in facts.names:
            if name in price and facts.priced.get(name) != price[name]:
                f.problems.append(
                    f"{label}: invocation {name!r} priced {facts.priced.get(name)} != "
                    f"registered {price[name]}"
                )
    for name, registered in reg.invocations.items():
        argv = facts.argv.get(name)
        if argv is not None and argv != registered:
            diff = [f"{a!r}!={b!r}" for a, b in zip(argv, registered, strict=False) if a != b]
            f.problems.append(
                f"{label}: invocation {name!r} is not the registered command "
                f"({', '.join(diff[:3]) or 'different length'}): an unregistered card"
            )


def registered_price(reg: Registration) -> dict[str, int]:
    """Each registered invocation's price by its runner's own parser, as run_card prices it."""
    from alpha_zero.local.run_card import Invocation, _price_invocation

    return {
        name: _price_invocation(
            Invocation(name, "training_civ6" if name == TRAIN else "benchmark", tuple(args))
        )
        for name, args in reg.invocations.items()
    }


def check_iterations(
    text: str, reg: Registration, sha: str | None, label: str, f: Findings, facts: RunFacts
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
        if rec.get("seed") != reg.seed:
            f.problems.append(f"{label}: iterations.jsonl line {number} seed {rec.get('seed')!r}")
        if sha is not None and rec.get("git_sha") != sha:
            f.problems.append(
                f"{label}: iterations.jsonl line {number} git_sha {rec.get('git_sha')!r} "
                "is not the run's pinned sha"
            )
        if isinstance(rec.get("iteration"), int):
            seen.add(rec["iteration"])
    iterations = int(reg.flag(TRAIN, "--num_iterations") or 0)
    expected = set(range(1, iterations + 1))
    if not facts.completed:
        # An interrupted attempt holds the iterations it finished, and no others.
        if not seen <= expected:
            f.problems.append(f"{label}: iterations.jsonl holds {sorted(seen - expected)}")
    elif seen != expected:
        f.problems.append(f"{label}: iterations.jsonl covers {sorted(seen)}, not 1..{iterations}")


# ---------------------------------------------------------------------------
# Arms: integrity and recomputation
# ---------------------------------------------------------------------------


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


def _check_protocol(
    name: str, envelope: Mapping[str, Any], reg: Registration, label: str, f: Findings
) -> None:
    """The envelope's recipe and search are the ones its registered command names."""
    from alpha_zero.evaluation.benchmark_evidence import SEARCH_SEMANTICS

    invocation = name.replace("_", "-")
    protocol = envelope["protocol"]
    for key, flag in PROTOCOL_FLAGS.items():
        want: Any = reg.flag(invocation, flag)
        if key != "mcts_type" and want is not None:
            want = int(want)
        if protocol.get(key) != want:
            f.problems.append(
                f"{label}: protocol {key} {protocol.get(key)!r} != registered {want!r}"
            )
    if protocol.get("gameplay_preset") != GAMEPLAY_PRESET:
        f.problems.append(f"{label}: gameplay_preset {protocol.get('gameplay_preset')!r}")
    evidence = envelope["evidence"]
    if evidence.get("device") != reg.flag(invocation, "--device"):
        f.problems.append(f"{label}: device {evidence.get('device')!r} is not the registered one")
    search = evidence.get("search") or {}
    expected = {
        "mcts_type": protocol.get("mcts_type"),
        "num_simulations": protocol.get("num_simulations"),
        "gumbel_k": protocol.get("gumbel_k"),
        "search_semantics": SEARCH_SEMANTICS,
    }
    for key, want in expected.items():
        if search.get(key) != want:
            f.problems.append(f"{label}: evidence.search {key} {search.get(key)!r} != {want!r}")


def check_arm(
    name: str,
    entry: Mapping[str, Any],
    envelope: Any,
    *,
    reg: Registration,
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
    _check_protocol(name, envelope, reg, label, f)
    invocation = name.replace("_", "-")
    priced, launched = facts.priced.get(invocation), facts.launched.get(invocation)
    if priced is None and not facts.completed:
        # Interrupted after the arm wrote result.json but before run_card recorded
        # it: hold the result to the registered command as run_card would launch it.
        n_games = reg.flag(invocation, "--n-games") or ""
        priced = int(n_games) if n_games.isdigit() else None
        launched = [*reg.invocations[invocation], "--max_games", n_games]
    if priced != row["num_games"]:
        f.problems.append(
            f"{label}: {row['num_games']} games, but run.json priced {invocation!r} at {priced!r}"
        )
    evidence = envelope["evidence"]
    if command_args(str(evidence.get("command", ""))) != launched:
        f.problems.append(
            f"{label}: evidence.command is not run.json's {invocation!r} invocation as "
            "launched (another arm's or run's result, or run by hand?)"
        )
    if evidence.get("seed") != reg.seed:
        f.problems.append(f"{label}: evidence.seed {evidence.get('seed')!r} != {reg.seed}")
    if sha is not None and evidence.get("git_sha") != sha:
        f.problems.append(f"{label}: evidence.git_sha is not the run's pinned sha")
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


def _opponent_flag(args: Sequence[str]) -> str | None:
    """The last non-empty ``--opponents=X`` (``--opponents=`` alone clears the default)."""
    values = [a.split("=", 1)[1] for a in args if a.startswith("--opponents=")]
    values = [v for v in values if v]
    return values[-1] if values else None


def check_arm_identities(
    arms: Mapping[str, ArmRead], reg: Registration, label: str, f: Findings
) -> None:
    """Each arm plays what its registered command names, one digest per checkpoint path."""
    digests: dict[str, str] = {}
    for name, arm in arms.items():
        args = reg.invocations[name.replace("_", "-")]
        for side, path_flag, other in (
            ("candidate", "--checkpoint", lambda a: _flag(a, "--candidate")),
            ("opponent", "--opponent-checkpoint", _opponent_flag),
        ):
            played = arm.candidate if side == "candidate" else arm.opponent
            path = _flag(args, path_flag)
            if path is None:
                want = other(args)
                if played != want:
                    f.problems.append(f"{label}: arm {name} {side} is {played}, registered {want}")
                continue
            if not played.startswith(CHECKPOINT):
                f.problems.append(f"{label}: arm {name} {side} {played} is not a checkpoint")
            elif digests.setdefault(path, played) != played:
                f.problems.append(
                    f"{label}: arm {name} plays a different checkpoint for {path} than another arm"
                )
    by_digest: dict[str, set[str]] = {}
    for path, digest in digests.items():
        by_digest.setdefault(digest, set()).add(path)
    for digest, paths in by_digest.items():
        if len(paths) > 1:
            f.problems.append(
                f"{label}: one checkpoint {digest[:24]}... stands for {sorted(paths)}"
            )


def _flag(args: Sequence[str], name: str) -> str | None:
    from alpha_zero.evaluation.a1_registration import flag_value

    return flag_value(list(args), name)


# ---------------------------------------------------------------------------
# One attempt
# ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    role: str
    run_id: str
    verdict: str | None
    games: int | None
    sha: str | None
    reasons: tuple[str, ...] = ()


def verify_run(run: Any, bundle_dir: Path, git: Git, rc_sha: str, f: Findings) -> RunOutcome | None:
    """Every per-attempt check except the play-path diff (judged attempts only)."""
    from alpha_zero.evaluation.a1_protocol import ProtocolMarkerError, compute_verdict
    from alpha_zero.evaluation.a1_registration import RegistrationError

    if not isinstance(run, Mapping):
        f.problems.append("runs[]: a run entry is not an object")
        return None
    run_id, role = str(run.get("run_id", "")), str(run.get("role", ""))
    label = f"run {run_id or '?'} ({role or '?'})"
    if role not in ROLES:
        f.problems.append(f"{label}: role must be one of {list(ROLES)}")
        return None

    run_path = _load_component(bundle_dir, run.get("run_json"), f"{label} run.json", f)
    if run_path is None:
        return None
    facts = check_run_json(_read_json(run_path, label, f), run_id, label, f)
    sha = facts.sha
    if sha is None:
        return None
    if run.get("git_sha") != sha:
        f.problems.append(f"{label}: bundle git_sha {run.get('git_sha')!r} != run.json sha")
    if not git.exists(sha):
        raise MissingEvidenceError(f"{label}: commit {sha[:12]} is not in this repository")
    if not git.is_ancestor(sha, rc_sha):
        f.problems.append(f"{label}: git_sha {sha[:12]} is not an ancestor of rc_sha")
    try:
        reg = read_registration(git, sha, run_id, role)
    except (RegistrationError, ProtocolMarkerError, ValueError) as exc:
        f.problems.append(f"{label}: section 7.1 at {sha[:12]} registers no runnable card: {exc}")
        return RunOutcome(role, run_id, None, facts.games, sha)
    if run.get("seed") != reg.seed:
        f.problems.append(f"{label}: seed {run.get('seed')!r} != {reg.seed} for {role}")
    check_registered(facts, reg, label, f)
    # An attempt killed before its first iteration finished has no record to list.
    if facts.completed or run.get("iterations") is not None:
        iter_path = _load_component(bundle_dir, run.get("iterations"), f"{label} iterations", f)
        if iter_path is not None:
            check_iterations(iter_path.read_text(encoding="utf-8"), reg, sha, label, f, facts)

    arms_in = run.get("arms")
    if not isinstance(arms_in, Mapping):
        f.problems.append(f"{label}: arms must be an object")
        arms_in = {}
    if facts.completed and set(arms_in) != set(reg.arms):
        f.problems.append(f"{label}: arms must be exactly the registered {reg.arms}")
    elif not set(arms_in) <= set(reg.arms):
        f.problems.append(
            f"{label}: arms {sorted(set(arms_in) - set(reg.arms))} are not registered"
        )
    arms: dict[str, ArmRead] = {}
    # A VOID-by-status attempt lists only the arms that ran; each is still checked.
    listed = reg.arms if facts.completed else [n for n in reg.arms if n in arms_in]
    for name in listed:
        entry = arms_in.get(name)
        path = _load_component(bundle_dir, entry, f"{label} arm {name}", f)
        if path is None or not isinstance(entry, Mapping):
            continue
        envelope = _read_json(path, f"{label} arm {name}", f)
        read = check_arm(
            name, entry, envelope, reg=reg, sha=sha, facts=facts,
            label=f"{label} arm {name}", f=f,
        )  # fmt: skip
        if read is not None:
            arms[name] = read
    check_budget(run, facts, label, f)
    check_arm_identities(arms, reg, label, f)

    if not facts.completed and not {PRIMARY, SYMMETRY, POSITIVE_CONTROL} <= set(arms):
        # VOID by status (section 7.1, "Interruptions"); the judged arms never all ran.
        stored = run.get("verdict")
        if not isinstance(stored, Mapping) or stored.get("verdict") != VOID:
            f.problems.append(
                f"{label}: run.json status is not COMPLETED, so the stored verdict must be "
                f"'VOID', not {stored.get('verdict') if isinstance(stored, Mapping) else stored!r}"
            )
        reasons = tuple(facts.interruptions)
        f.notes.append(f"{label}: verdict VOID ({'; '.join(reasons)}; listed, not judged)")
        return RunOutcome(role, run_id, VOID, facts.games, sha, reasons)
    if facts.completed and set(arms) != set(reg.arms):
        return RunOutcome(role, run_id, None, facts.games, sha)
    computed = compute_verdict(
        arms[PRIMARY].games,
        arms[SYMMETRY].games,
        arms[POSITIVE_CONTROL].games,
        interruptions=facts.interruptions,
    )
    compare_verdict(run.get("verdict"), computed.to_dict(), label, f)
    f.notes.append(f"{label}: verdict {computed.verdict} ({'; '.join(computed.reasons)})")
    return RunOutcome(role, run_id, computed.verdict, facts.games, sha, computed.reasons)


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


def check_play_path(
    git: Git, outcome: RunOutcome, targets: Sequence[str], play_path: Sequence[str], f: Findings
) -> None:
    """No play-path change from a judged attempt's commit to the rc or the release head."""
    if outcome.sha is None:
        return
    label = f"run {outcome.run_id} ({outcome.role})"
    specs = [f":(glob){p}" for p in play_path]
    for target in dict.fromkeys(targets):
        changed = git.lines(
            "diff", "--name-only", "--no-renames", outcome.sha, target, "--", *specs
        )
        if changed:
            f.problems.append(
                f"{label}: the play path changed between {outcome.sha[:12]} and {target[:12]}: "
                f"{', '.join(changed)}"
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
    from scripts.release_version import ReleaseError, parse_tag

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
    from scripts.release_checks import highest_rc
    from scripts.release_version import ReleaseError, pyproject_version

    rc_tag, rc_sha = str(bundle["rc_tag"]), str(bundle["rc_sha"])
    if not git.exists(rc_sha):
        raise MissingEvidenceError(
            f"rc_sha {rc_sha[:12]} is not in this repository (fetch-depth 0?)"
        )
    if not git.exists(f"refs/tags/{rc_tag}"):
        raise MissingEvidenceError(f"tag {rc_tag} is not in this repository (fetch tags)")
    if git.resolve(f"refs/tags/{rc_tag}") != rc_sha:
        f.problems.append(f"{rc_tag} does not point at rc_sha {rc_sha[:12]}")
    highest = highest_rc(git, str(bundle["release"]))
    if highest is not None and highest.name != rc_tag:
        f.problems.append(
            f"the bundle names {rc_tag}, but the highest rc of {bundle['release']} is "
            f"{highest.name}: regenerate the evidence on the rc that ships"
        )
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
    """The bundle's protocol hash is section 7.1's at HEAD and at every attempt's commit."""
    from alpha_zero.evaluation.a1_protocol import ProtocolMarkerError, protocol_sha256
    from scripts.check_branch_direction import GitError

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
        if digest == recorded:
            continue
        why = (
            "section 7.1 changed after the run; under section 7.1 'Amendments' that is a new "
            "registration, which retires the confirmatory seeds"
            if where == "HEAD"
            else "the run was not made under the registration the bundle records"
        )
        f.problems.append(
            f"protocol_sha256 {recorded[:12]}... != section 7.1 at {where} ({digest[:12]}...): {why}"
        )


def judged_attempts(outcomes: Sequence[RunOutcome], f: Findings) -> dict[str, RunOutcome]:
    """Exactly one non-VOID attempt per role present; that one is judged."""
    ids = [o.run_id for o in outcomes]
    if len(set(ids)) != len(ids):
        f.problems.append(f"run_id must be unique (a re-run gets a new card_id); got {ids}")
    judged: dict[str, RunOutcome] = {}
    for role in ROLES:
        attempts = [o for o in outcomes if o.role == role]
        if not attempts:
            continue
        valid = [o for o in attempts if o.verdict not in (None, VOID)]
        voids = [o for o in attempts if o.verdict == VOID]
        if len(valid) == 1:
            judged[role] = valid[0]
        elif not valid:
            why = "; ".join(r for o in voids for r in o.reasons) or "no verdict could be computed"
            f.problems.append(
                f"{role} has no non-VOID attempt ({why}). A VOID records no learning result: "
                "fix the cause and re-run under a new card_id at the same seed (section 7.1, "
                "'After a VOID')"
            )
        else:
            f.problems.append(
                f"{role} has {len(valid)} non-VOID attempts {[o.run_id for o in valid]}; "
                "exactly one may be judged"
            )
    if not any(o.role == "A1" for o in outcomes):
        f.problems.append("the bundle has no A1 run")
    return judged


def _claim_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if _LEARNS.search(_MENTION.sub("", line))]


def changelog_section(text: str, release: str) -> str:
    """The ``## [X.Y.Z]`` section of a Keep-a-Changelog file ('' when absent)."""
    out: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line.startswith(f"## [{release}]")
            continue
        if inside:
            out.append(line)
    return "\n".join(out)


def check_learning(
    bundle: Mapping[str, Any],
    judged: Mapping[str, RunOutcome],
    bundle_dir: Path,
    repo_root: Path,
    f: Findings,
) -> None:
    verdicts = {role: o.verdict for role, o in judged.items()}
    claimed = all(verdicts.get(r) == "PASS" for r in CLAIM_ROLES)
    if bundle["learning_recorded"] is not True:
        f.problems.append(
            "learning_recorded is required (true): the verdict is recorded either way"
        )
    if bundle["learning_claimed"] is not claimed:
        f.problems.append(
            f"learning_claimed is {bundle['learning_claimed']!r}, but the judged verdicts "
            f"{verdicts} {'permit' if claimed else 'do not permit'} the claim"
        )
    report = bundle_dir / REPORT_FILE
    if not report.is_file():
        f.missing.append(f"{REPORT_FILE} is missing from {bundle_dir.name}/")
        return
    if claimed:
        return
    texts = {REPORT_FILE: report.read_text(encoding="utf-8")}
    index = repo_root / EVIDENCE_INDEX
    if index.is_file():
        texts["the index row"] = "\n".join(
            line
            for line in index.read_text(encoding="utf-8").splitlines()
            if f"]({bundle_dir.name}/" in line
        )
    changelog = repo_root / CHANGELOG
    if changelog.is_file():
        texts[f"{CHANGELOG} [{bundle['release']}]"] = changelog_section(
            changelog.read_text(encoding="utf-8"), str(bundle["release"])
        )
    for where, text in texts.items():
        used = _claim_lines(text)
        if used:
            f.problems.append(f"{where} claims learning without learning_claimed: {used[0]!r}")


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
    from scripts.check_branch_direction import GitError

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
    rc_sha = str(bundle["rc_sha"])
    try:
        check_release_refs(bundle, git, head, repo_root, f)
        outcomes = [
            o for run in bundle["runs"] if (o := verify_run(run, bundle_dir, git, rc_sha, f))
        ]
        judged = judged_attempts(outcomes, f)
        for outcome in judged.values():
            check_play_path(git, outcome, [rc_sha, head], play, f)
        check_protocol(bundle, git, [o.sha for o in outcomes if o.sha], repo_root, f)
    except (MissingEvidenceError, GitError) as exc:
        f.missing.append(str(exc))
        return f
    except (ValueError, KeyError, TypeError) as exc:  # malformed input: fail closed
        f.problems.append(f"cannot verify the bundle: {type(exc).__name__}: {exc}")
        return f
    total = sum(o.games or 0 for o in outcomes)
    if bundle["games_total"] != total:
        f.problems.append(f"games_total {bundle['games_total']!r} != the runs' sum {total}")
    f.notes.append(
        f"bundle total: {total:,} games across {len(outcomes)} attempt(s) (reported, not gated)"
    )
    check_learning(bundle, judged, bundle_dir, repo_root, f)
    check_index(repo_root, bundle_dir, f)
    check_runs(bundle, api, repo, f)
    return f


def check_runs(bundle: Mapping[str, Any], api: Api | None, repo: str, f: Findings) -> None:
    """The rc commit's checks, read as ``wait-for-ci`` reads them (one pass, no polling).

    An API error is a failure, as in ``wait-for-ci`` (``ci_snapshot`` reports it);
    being unable to call the API at all (no repo, no ``gh``) is missing.
    """
    from scripts.release_checks import ci_snapshot
    from scripts.release_version import parse_tag

    if api is None or not repo:
        f.missing.append("check runs: --repo (or GITHUB_REPOSITORY) is required to read them")
        return
    tag = parse_tag(str(bundle["rc_tag"]))
    try:
        failures, waiting = ci_snapshot(api, repo, str(bundle["rc_sha"]), tag)
    except OSError as exc:
        f.missing.append(f"check runs: cannot call the GitHub API: {exc}")
        return
    f.problems.extend(f"check runs: {p}" for p in failures)
    f.problems.extend(f"check runs: not green: {w}" for w in waiting)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_bundle(repo_root: Path) -> Path:
    """``docs/evidence/release_vX_Y_Z/bundle.json`` for pyproject.toml's version."""
    from scripts.release_version import pyproject_version

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
    parser.add_argument(
        "--head", default="HEAD", help="the release head: rc_sha must be its ancestor"
    )
    parser.add_argument("--scope", action="store_true", help="CI: print n/a or applies; exit 0")
    args = parser.parse_args(argv)
    env = os.environ if env is None else env
    if args.scope:
        return _report_scope(env)

    from scripts.check_branch_direction import Git, gh_api
    from scripts.release_version import ReleaseError

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
