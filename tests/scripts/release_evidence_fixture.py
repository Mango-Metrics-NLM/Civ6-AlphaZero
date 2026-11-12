"""Build release evidence bundles for ``tests/scripts/test_verify_release_evidence.py``.

Not a test module (no ``test_`` prefix). Every bundle here is generated: the
per-arm ``result.json`` rows come from the benchmark's own ``_result_record``
(so they are exactly what ``benchmark_suite.py`` writes), the stored verdict from
``a1_protocol.compute_verdict``, and every sha256 from the files as written. A
test then breaks one thing and asserts the verifier notices.

The repository is a throwaway git repo (:class:`tests.scripts.git_fixture.GitRepo`)
holding a copy of the real ``docs/EVALUATION.md``, so the protocol hash a bundle
records is the registered one.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alpha_zero.evaluation.a1_protocol import compute_verdict, protocol_sha256
from alpha_zero.evaluation.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    SEARCH_EPOCH,
    BenchmarkProtocol,
    _result_record,
)
from alpha_zero.evaluation.benchmark_evidence import SEARCH_SEMANTICS
from scripts import verify_release_evidence as vre
from scripts.check_branch_direction import ApiError
from tests.scripts.git_fixture import GitRepo

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_TEXT = (REPO_ROOT / "docs" / "EVALUATION.md").read_text(encoding="utf-8")
REGISTERED_SHA = protocol_sha256(EVALUATION_TEXT)
REPO = "acme/widget"
RELEASE = "2.1.0"
RC_TAG = "v2.1.0-rc.1"
BUNDLE_DIR = Path("docs/evidence/release_v2_1_0")
GENERATED = "2026-12-01T00:00:00+00:00"

Pair = tuple[str, str]


def games(pairs: Sequence[Pair], n_games: int) -> list[dict[str, Any]]:
    """``n_games`` per-game records, cycling through ``pairs`` (seat 0, seat 1)."""
    out = []
    for p in range(n_games // 2):
        first, second = pairs[p % len(pairs)]
        out += [
            {"pair": p, "seat": 0, "result": first},
            {"pair": p, "seat": 1, "result": second},
        ]
    return out


WIN: list[Pair] = [("win", "win"), ("win", "draw"), ("win", "win"), ("draw", "loss")]
LOSS: list[Pair] = [("loss", "loss"), ("loss", "draw"), ("loss", "loss"), ("draw", "win")]
EVEN: list[Pair] = [("win", "loss"), ("draw", "draw"), ("loss", "win"), ("draw", "draw")]
SYMMETRIC: list[Pair] = [("win", "loss"), ("loss", "win")]
LOPSIDED: list[Pair] = [("win", "win")]
PRIMARY = {"PASS": WIN, "FAIL": LOSS, "INCONCLUSIVE": EVEN}


def ckpt(n: int, run_id: str) -> str:
    return "checkpoint:" + hashlib.sha256(f"{run_id}/{n}".encode()).hexdigest()


@dataclass
class RunSpec:
    """One component run. Defaults are a clean, registered A1 card at 980 games."""

    run_id: str = "a1-20261111"
    role: str = "A1"
    primary: str = "PASS"
    symmetry: list[Pair] = field(default_factory=lambda: list(SYMMETRIC))
    status: str = "COMPLETED"
    env_var_names: list[str] = field(default_factory=lambda: ["CIV6AZ_DATA_ROOT", "PATH"])
    approval_ref: str | None = None
    train_games: int = 480
    positive_control_games: int = 40
    resumed: bool = False

    @property
    def seed(self) -> int:
        return vre.ROLE_SEEDS[self.role]


def arm_argv(name: str, seed: int, candidate: str) -> list[str]:
    """A stand-in for the arm's registered arguments: enough to tell arms apart."""
    return ["--seed", str(seed), "--candidate", candidate, "--out", f"arena/{name}"]


def _envelope(
    candidate: str,
    opponent: str,
    records: list[dict[str, Any]],
    *,
    seed: int,
    sha: str,
    argv: list[str],
) -> dict[str, Any]:
    protocol = BenchmarkProtocol(
        board_size=8, max_turns=60, num_simulations=24, gumbel_k=8, n_games=len(records)
    )
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "protocol": protocol.to_dict(),
        "candidate": {
            "kind": "checkpoint" if candidate.startswith("checkpoint:") else candidate,
            "id": candidate,
        },
        "shard": None,
        "evidence": {
            "schema_version": 1,
            "git_sha": sha,
            "generated_utc": GENERATED,
            "search_epoch": SEARCH_EPOCH,
            "search_semantics": SEARCH_SEMANTICS,
            "seed": seed,
            # As run_card launches it: the card's argv plus --max_games <priced>.
            "command": shlex.join(
                [
                    "python",
                    "-m",
                    "alpha_zero.evaluation.benchmark",
                    *argv,
                    "--max_games",
                    str(len(records)),
                ]
            ),
            "device": "cpu",
            "search": {
                "mcts_type": "gumbel",
                "num_simulations": 24,
                "gumbel_k": 8,
                "search_semantics": SEARCH_SEMANTICS,
            },
            "hardware": {
                "cpu_model": None,
                "gpu_model": None,
                "logical_cores": 4,
                "omp_num_threads": "2",
            },
            "python": "3.11.9",
            "torch": "2.5.0",
            "numpy": "2.1.0",
            "games_total": len(records),
            "claims": [],
            "non_claims": [],
        },
        "results": [_result_record(opponent, records, GENERATED)],
    }


def _arms(spec: RunSpec) -> dict[str, tuple[str, str, list[dict[str, Any]]]]:
    c0, c20 = ckpt(0, spec.run_id), ckpt(20, spec.run_id)
    pc = spec.positive_control_games
    arms = {
        "primary": (c20, c0, games(PRIMARY[spec.primary], 200)),
        "symmetry": (c0, c0, games(spec.symmetry, 40)),
        "positive_control": ("heuristic", "random", games(WIN, pc)),
        "final_vs_random": (c20, "random", games(WIN, vre.BRANCH_FINAL_GAMES[pc])),
    }
    for n in (5, 10, 15):
        arms[f"secondary_{n}"] = (ckpt(n, spec.run_id), c0, games(EVEN, 60))
    return arms


def _write(path: Path, text: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_run(bundle_dir: Path, spec: RunSpec, sha: str) -> dict[str, Any]:
    """Write one run's components under ``bundle_dir/<run_id>/``; return its bundle entry."""
    base = Path(spec.run_id)
    arms = _arms(spec)
    invocations = [{"name": "train", "priced_games": spec.train_games, "exit_code": 0}]
    for name, (cand, _o, rec) in arms.items():
        argv = arm_argv(name, spec.seed, cand)
        invocations.append(
            {
                "name": name.replace("_", "-"),
                "priced_games": len(rec),
                "exit_code": 0,
                "argv": argv,
                "launched_argv": [*argv, "--max_games", str(len(rec))],
            }
        )
    total = sum(i["priced_games"] for i in invocations)
    run_json = {
        "schema_version": 1,
        "card_id": spec.run_id,
        "sha": sha,
        "status": spec.status,
        "env_var_names": spec.env_var_names,
        "approval_ref": spec.approval_ref,
        "total_priced_games": total,
        "invocations": invocations,
    }
    entry: dict[str, Any] = {
        "run_id": spec.run_id,
        "role": spec.role,
        "seed": spec.seed,
        "git_sha": sha,
        "games": total,
        "approval_ref": spec.approval_ref,
    }
    rel = (base / "run.json").as_posix()
    entry["run_json"] = {"path": rel, **_write(bundle_dir / rel, json.dumps(run_json, indent=2))}
    lines = [
        json.dumps(
            {
                "run_id": spec.run_id,
                "seed": spec.seed,
                "git_sha": sha,
                "iteration": i,
                "resumed_from": "ckpt/checkpoint_10.pt" if spec.resumed and i > 10 else None,
            }
        )
        for i in range(1, vre.TRAINING_ITERATIONS + 1)
    ]
    rel = (base / "iterations.jsonl").as_posix()
    entry["iterations"] = {"path": rel, **_write(bundle_dir / rel, "\n".join(lines) + "\n")}
    entry["arms"] = {}
    for name, (cand, opp, rec) in arms.items():
        envelope = _envelope(
            cand, opp, rec, seed=spec.seed, sha=sha, argv=arm_argv(name, spec.seed, cand)
        )
        rel = (base / "arena" / name / "result.json").as_posix()
        written = _write(bundle_dir / rel, json.dumps(envelope, indent=2))
        summary = vre.arm_summary(envelope["results"][0], vre.ARM_CONFIDENCE.get(name, 0.95))
        entry["arms"][name] = {"path": rel, **written, **summary}
    interruptions = []
    if spec.status != "COMPLETED":
        interruptions.append(f"run.json status is {spec.status!r}")
    if spec.resumed:
        interruptions.append("resumed")
    if any(n.startswith("MCTS_") for n in spec.env_var_names):
        interruptions.append("MCTS_")
    entry["verdict"] = compute_verdict(
        arms["primary"][2], arms["symmetry"][2], arms["positive_control"][2],
        interruptions=interruptions,
    ).to_dict()  # fmt: skip
    return entry


def write_bundle(
    bundle_dir: Path,
    specs: Sequence[RunSpec],
    *,
    sha: str,
    rc_sha: str,
    report: str | None = None,
) -> dict[str, Any]:
    runs = [write_run(bundle_dir, s, sha) for s in specs]
    verdicts = {r["role"]: r["verdict"]["verdict"] for r in runs}
    claimed = all(verdicts.get(role) == "PASS" for role in vre.CLAIM_ROLES)
    total = sum(r["games"] for r in runs)
    bundle = {
        "bundle_schema_version": vre.BUNDLE_SCHEMA_VERSION,
        "release": RELEASE,
        "rc_tag": RC_TAG,
        "rc_sha": rc_sha,
        "protocol_sha256": REGISTERED_SHA,
        "runs": runs,
        "games_total": total,
        "learning_recorded": True,
        "learning_claimed": claimed,
        "evidence": {
            "schema_version": 1,
            "git_sha": rc_sha,
            "generated_utc": GENERATED,
            "search_epoch": SEARCH_EPOCH,
            "seed": 20261111,
            "command": "python scripts/verify_release_evidence.py",
            "hardware": {
                "cpu_model": None,
                "gpu_model": None,
                "logical_cores": 4,
                "omp_num_threads": "2",
            },
            "python": "3.11.9",
            "torch": "2.5.0",
            "numpy": "2.1.0",
            "games_total": total,
            "claims": ["the A1 verdict under section 7.1"],
            "non_claims": ["strength under any other protocol"],
        },
    }
    save_bundle(bundle_dir, bundle)
    if report is None:
        report = "# Release v2.1.0 evidence\n\nVerdict recorded; the `learns` claim is withheld.\n"
    _write(bundle_dir / vre.REPORT_FILE, report)
    return bundle


def save_bundle(bundle_dir: Path, bundle: dict[str, Any]) -> None:
    _write(bundle_dir / vre.BUNDLE_FILE, json.dumps(bundle, indent=2))


def rehash(bundle_dir: Path, entry: dict[str, Any]) -> None:
    """Re-record ``entry``'s sha256 after a test edits its file (so only recomputation can tell)."""
    entry["sha256"] = hashlib.sha256((bundle_dir / entry["path"]).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The repository
# ---------------------------------------------------------------------------

PYPROJECT = f'[project]\nname = "x"\nversion = "{RELEASE}"\n'
CODE = {
    "pyproject.toml": PYPROJECT,
    "docs/EVALUATION.md": EVALUATION_TEXT,
    "alpha_zero/core/mcts/base.py": "BASE = 1\n",
    "alpha_zero/unrelated.py": "X = 1\n",
}
INDEX = (
    "# Evidence index\n\n| Artifact | Status |\n|---|---|\n"
    "| [`release_v2_1_0/`](release_v2_1_0/report.md) | current |\n"
)


@dataclass
class Release:
    repo: GitRepo
    run_sha: str
    rc_sha: str
    bundle: dict[str, Any]

    @property
    def bundle_dir(self) -> Path:
        return self.repo.path / BUNDLE_DIR

    def save(self) -> None:
        save_bundle(self.bundle_dir, self.bundle)


def make_release(
    root: Path,
    specs: Sequence[RunSpec] = (RunSpec(),),
    *,
    code: dict[str, str] | None = None,
    after_run: dict[str, str] | None = None,
    index: str = INDEX,
    report: str | None = None,
) -> Release:
    """``code`` at the run commit; ``after_run`` (if any) committed before the rc tag.

    The evidence commit (the bundle and the index row) goes on top of the rc,
    as an ``evidence/<rc>`` branch merged into qa would put it.
    """
    repo = GitRepo(root)
    run_sha = repo.commit("code", {**CODE, **(code or {})})
    rc_sha = repo.commit("after the run", after_run) if after_run else run_sha
    repo.tag(RC_TAG, rc_sha)
    bundle = write_bundle(repo.path / BUNDLE_DIR, specs, sha=run_sha, rc_sha=rc_sha, report=report)
    repo.commit("evidence", {"docs/evidence/README.md": index})
    repo.git("add", "-A")
    repo.git("commit", "-q", "-m", "bundle")
    return Release(repo, run_sha, rc_sha, bundle)


# ---------------------------------------------------------------------------
# A fake GitHub API: the rc's checks, as wait-for-ci reads them
# ---------------------------------------------------------------------------

PR_HEAD = "e" * 40


def fake_api(rc_sha: str, *, conclusion: str = "success") -> Callable[[str], Any]:
    """Every check run green (or ``conclusion``); the rc is the merge of a PR into qa."""
    pr = {
        "number": 7,
        "merged_at": "2026-12-01T00:00:00Z",
        "merge_commit_sha": rc_sha,
        "head": {"ref": "dev", "sha": PR_HEAD},
        "base": {"ref": "qa"},
    }
    run = {
        "id": 1,
        "status": "completed",
        "conclusion": conclusion,
        "app": {"slug": "github-actions"},
        "completed_at": "2026-11-30T23:00:00Z",
    }
    calls: list[str] = []

    def api(path: str) -> Any:
        calls.append(path)
        if path == f"repos/{REPO}/commits/{rc_sha}/pulls?per_page=100":
            return [pr]
        if "/check-runs?" in path:
            return {"check_runs": [run]}
        raise ApiError(f"no fake answer for {path}")

    api.calls = calls  # type: ignore[attr-defined]
    return api
