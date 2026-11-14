"""Build release evidence bundles for ``tests/scripts/test_verify_release_evidence.py``.

Not a test module (no ``test_`` prefix). Every bundle here is generated: each
run's invocations are section 7.1's registered commands
(``a1_registration.registered_invocations``), the per-arm ``result.json`` rows come
from the benchmark's own ``_result_record`` (exactly what ``benchmark_suite.py``
writes), the stored verdict from ``a1_protocol.compute_verdict``, and every
sha256 from the files as written. A test then breaks one thing and asserts the
verifier notices.

The repository is a throwaway git repo (:class:`tests.scripts.git_fixture.GitRepo`)
holding a copy of the real ``docs/EVALUATION.md`` with its "Branch in force"
slot filled (as T0b will fill it), so the commands are the registered ones.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alpha_zero.evaluation.a1_protocol import compute_verdict, protocol_sha256, registered_protocol
from alpha_zero.evaluation.a1_registration import (
    A1R_SEED,
    REGISTERED_DEVICE_ORDER,
    flag_value,
    registered_devices,
    registered_invocations,
)
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
REAL_EVALUATION = (REPO_ROOT / "docs" / "EVALUATION.md").read_text(encoding="utf-8")
PENDING = "**Branch in force: pending T0b.**"
REPO = "acme/widget"
RELEASE = "2.1.0"
RC_TAG = "v2.1.0-rc.1"
BUNDLE_DIR = Path("docs/evidence/release_v2_1_0")
GENERATED = "2026-12-01T00:00:00+00:00"
SEEDS = {"A1": 20261111, "A1-R": A1R_SEED}


def evaluation_text(branch: str = "40") -> str:
    """The real ``docs/EVALUATION.md`` with T0b's branch slot filled."""
    assert REAL_EVALUATION.count(PENDING) == 1
    return REAL_EVALUATION.replace(PENDING, f"**Branch in force: {branch}.**")


EVALUATION_TEXT = evaluation_text()
REGISTERED_SHA = protocol_sha256(EVALUATION_TEXT)

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


def ckpt(path: str) -> str:
    """A checkpoint's ``checkpoint:<sha256>`` id, stable per path."""
    return "checkpoint:" + hashlib.sha256(path.encode()).hexdigest()


@dataclass
class RunSpec:
    """One attempt. Defaults are a clean, registered A1 card at 980 games."""

    run_id: str = "a1-20261111"
    role: str = "A1"
    primary: str = "PASS"
    symmetry: list[Pair] = field(default_factory=lambda: list(SYMMETRIC))
    status: str = "COMPLETED"
    env_var_names: list[str] = field(default_factory=lambda: ["CIV6AZ_DATA_ROOT", "PATH"])
    approval_ref: str | None = None
    train_games: int = 480
    resumed: bool = False
    #: Run at the rc commit instead of the run commit (a re-run after a fix).
    at_rc: bool = False
    #: Registered invocations to edit before writing: name -> function(argv) -> argv.
    edit_argv: dict[str, Callable[[list[str]], list[str]]] = field(default_factory=dict)
    #: An interrupted attempt as run_card writes it: only the first ``recorded``
    #: invocations finished (and are in run.json), ``total_priced_games`` is still
    #: the whole card's price, and only their arms wrote result.json.
    recorded: int | None = None
    #: Training iterations that finished (an interruption inside training); 0 lists
    #: no iterations.jsonl at all.
    iterations_written: int = 20
    #: The next arm wrote result.json, but the attempt was interrupted before
    #: run_card recorded its invocation.
    unrecorded_next_arm: bool = False
    #: Edit the registered ``device_plan`` before it is written (another GPU, torch...).
    edit_device_plan: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None
    #: Write the record a ``run_card --probe`` leaves.
    probe: bool = False

    @property
    def seed(self) -> int:
        return SEEDS[self.role]


def _envelope(
    candidate: str,
    opponent: str,
    records: list[dict[str, Any]],
    *,
    seed: int,
    sha: str,
    launched: list[str],
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
            # What cli_command writes for a run_card child: the launched argv.
            "command": shlex.join(["python", "-m", "alpha_zero.evaluation.benchmark", *launched]),
            # What the arena records: the device it was launched with (section 7.1).
            "device": launched[launched.index("--device") + 1],
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


def _sides(argv: list[str]) -> tuple[str, str]:
    """The candidate and opponent ids an arm's arguments name."""
    path = flag_value(argv, "--checkpoint")
    candidate = ckpt(path) if path else str(flag_value(argv, "--candidate"))
    opp_path = flag_value(argv, "--opponent-checkpoint")
    if opp_path:
        return candidate, ckpt(opp_path)
    named = [a.split("=", 1)[1] for a in argv if a.startswith("--opponents=")]
    return candidate, [n for n in named if n][-1]


def _results(name: str, n: int, spec: RunSpec) -> list[dict[str, Any]]:
    if name == "primary":
        return games(PRIMARY[spec.primary], n)
    if name == "symmetry":
        return games(spec.symmetry, n)
    if name.startswith("secondary"):
        return games(EVEN, n)
    return games(WIN, n)


def registered_device_plan(section: str) -> dict[str, Any] | None:
    """``run.json``'s ``device_plan`` for a run on exactly the registered devices.

    ``None`` when the section registers no devices (a pre-GPU registration), which
    is what ``run_card`` wrote then.
    """
    registered = registered_devices(section)
    if registered is None:
        return None
    names = [registered.gpus[i] for i in sorted(registered.gpus)]
    return {
        "CUDA_VISIBLE_DEVICES": registered.cuda_visible_devices,
        "CUDA_DEVICE_ORDER": REGISTERED_DEVICE_ORDER,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "source": "card",
        "torch_pin": registered.torch_pin,
        "torch": {
            "version": f"{registered.torch_version}+cu128",
            "cuda": "12.8",
            "cuda_available": True,
            "device_count": len(names),
            "devices": names,
        },
    }


def _write(path: Path, text: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_run(bundle_dir: Path, spec: RunSpec, sha: str, section: str) -> dict[str, Any]:
    """Write one attempt's components under ``bundle_dir/<run_id>/``; return its entry."""
    base = Path(spec.run_id)
    registered = registered_invocations(section, card_id=spec.run_id, seed=spec.seed)
    invocations = []
    arm_records: dict[str, tuple[str, str, list[dict[str, Any]], list[str]]] = {}
    for name, argv in registered.items():
        argv = spec.edit_argv.get(name, lambda a: a)(list(argv))
        if name == "train":
            priced = spec.train_games
        else:
            priced = int(flag_value(argv, "--n-games") or 0)
        launched = [*argv, "--max_games", str(priced)]
        invocations.append(
            {
                "name": name,
                "runner": "training_civ6" if name == "train" else "benchmark",
                "priced_games": priced,
                "exit_code": 0,
                "argv": argv,
                "launched_argv": launched,
            }
        )
        reached = None if spec.recorded is None else spec.recorded + spec.unrecorded_next_arm
        if name != "train" and (reached is None or len(invocations) <= reached):
            cand, opp = _sides(argv)
            arm_records[vre.arm_name(name)] = (cand, opp, _results(name, priced, spec), launched)
    total = sum(i["priced_games"] for i in invocations)  # the whole card, as run_card prices it
    if spec.recorded is not None:
        invocations = invocations[: spec.recorded]
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
    plan = registered_device_plan(section)
    if plan is not None and spec.edit_device_plan is not None:
        plan = spec.edit_device_plan(plan)
    if plan is not None:
        run_json["device_plan"] = plan
    if spec.probe:
        run_json["probe"] = True
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
        for i in range(1, spec.iterations_written + 1)
    ]
    if lines:
        rel = (base / "iterations.jsonl").as_posix()
        entry["iterations"] = {"path": rel, **_write(bundle_dir / rel, "\n".join(lines) + "\n")}
    entry["arms"] = {}
    for name, (cand, opp, rec, launched) in arm_records.items():
        envelope = _envelope(cand, opp, rec, seed=spec.seed, sha=sha, launched=launched)
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
    if {"primary", "symmetry", "positive_control"} <= set(arm_records):
        entry["verdict"] = compute_verdict(
            arm_records["primary"][2],
            arm_records["symmetry"][2],
            arm_records["positive_control"][2],
            interruptions=interruptions,
        ).to_dict()
    else:  # the judged arms never all ran: VOID by status
        entry["verdict"] = {"verdict": "VOID", "reasons": interruptions}
    return entry


def write_bundle(
    bundle_dir: Path,
    specs: Sequence[RunSpec],
    *,
    sha: str,
    rc_sha: str,
    rc_tag: str = RC_TAG,
    document: str,
    report: str | None = None,
) -> dict[str, Any]:
    section = registered_protocol(document)
    runs = [write_run(bundle_dir, s, rc_sha if s.at_rc else sha, section) for s in specs]
    judged = {r["role"]: r["verdict"]["verdict"] for r in runs if r["verdict"]["verdict"] != "VOID"}
    claimed = all(judged.get(role) == "PASS" for role in vre.CLAIM_ROLES)
    total = sum(r["games"] for r in runs)
    bundle = {
        "bundle_schema_version": vre.BUNDLE_SCHEMA_VERSION,
        "release": RELEASE,
        "rc_tag": rc_tag,
        "rc_sha": rc_sha,
        "protocol_sha256": protocol_sha256(document),
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
    evaluation: str | None = None,
    after_run: dict[str, str] | None = None,
    later_rc: dict[str, str] | None = None,
    after_rc: dict[str, str] | None = None,
    index: str = INDEX,
    report: str | None = None,
    extra: dict[str, str] | None = None,
) -> Release:
    """A repository whose rc carries the evidence on top, as ``evidence/<rc>`` would.

    * ``code`` (and ``evaluation``, the whole ``docs/EVALUATION.md``) at the run commit;
    * ``after_run`` committed before ``v2.1.0-rc.1`` (so it is in rc.1);
    * ``later_rc`` committed and tagged ``v2.1.0-rc.2`` (the bundle still names rc.1);
    * ``after_rc`` committed after the rc(s), untagged (it reaches the release head);
    * the evidence commit (bundle, index, ``extra`` files) on top.
    """
    text = evaluation if evaluation is not None else EVALUATION_TEXT
    repo = GitRepo(root)
    run_sha = repo.commit("code", {**CODE, "docs/EVALUATION.md": text, **(code or {})})
    rc_sha = repo.commit("after the run", after_run) if after_run else run_sha
    repo.tag(RC_TAG, rc_sha)
    if later_rc:
        repo.tag("v2.1.0-rc.2", repo.commit("rcfix", later_rc))
    if after_rc:
        repo.commit("after the rc", after_rc)
    bundle = write_bundle(
        repo.path / BUNDLE_DIR, specs, sha=run_sha, rc_sha=rc_sha, document=text, report=report
    )
    repo.commit("evidence", {"docs/evidence/README.md": index, **(extra or {})})
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
