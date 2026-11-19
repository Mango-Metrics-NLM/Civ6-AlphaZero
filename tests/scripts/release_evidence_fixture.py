"""Build release evidence bundles for ``tests/scripts/test_verify_release_evidence.py``.

Not a test module (no ``test_`` prefix). Every bundle here is generated: each
run's invocations are section 7.1's registered commands
(``a1_registration.registered_invocations``), the per-arm ``result.json`` rows come
from the benchmark's own ``_result_record`` (exactly what ``benchmark_suite.py``
writes), the stored verdict from ``a1_protocol.compute_verdict``, and every
sha256 from the files as written. A test then breaks one thing and asserts the
verifier notices.

An interrupted attempt that has a re-run also gets the compare record section 7.1
("After a VOID") requires, shaped as ``python -m alpha_zero.evaluation.reproduction
compare --out`` writes it: both runs' identities and their digests, the arena and
iteration digests taken from the bundled files and the checkpoint digests stand-ins
(:func:`checkpoint_digests`; a bundle holds no checkpoints). The outcome stored
beside it and the re-run's verdict come from ``scripts/release_reproduction.py``.

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

from alpha_zero.evaluation import reproduction as rp
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
from scripts import release_reproduction as rr
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
#: The NVIDIA driver every fixture run reports unless its spec says otherwise.
DRIVER = "576.88"
#: The registered training command's ``--save_interval``: which checkpoints a run saves.
SAVE_INTERVAL = 5
JUDGED_ARMS = ("primary", "symmetry", "positive_control")


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
    #: ``run.json``'s ``launched``: the invocations run_card started. ``None`` is what
    #: run_card writes -- every recorded invocation and, for an attempt that did not
    #: record them all, the one that was in flight.
    launched: list[str] | None = None
    #: ``run.json``'s ``started_at``; ``None`` spaces the attempts a day apart, in bundle order.
    started_at: str | None = None
    #: The NVIDIA driver version ``preflight.nvidia_smi`` reports: one string for
    #: every GPU, or one per GPU; ``None`` records that nvidia-smi could not be run.
    driver: str | Sequence[str] | None = DRIVER
    #: Write ``run.json`` without ``launched`` and ``preflight``, as run_card wrote it
    #: before those keys existed (``started_at`` it has always written).
    legacy_run_json: bool = False
    #: Edit ``run.json`` just before it is written (a missing key, a ``torch_wheel``...).
    edit_run_json: Callable[[dict[str, Any]], None] | None = None
    #: Rotate the primary arm's pairs: the same counts and verdict, other games.
    primary_rotation: int = 0
    #: Salt of the stand-in checkpoint digests: another value is other trained weights.
    checkpoint_salt: str = ""
    #: An interrupted attempt with a re-run gets its compare record unless this is False.
    compare_record: bool = True
    #: Edit the compare record before it is written and its outcome derived.
    edit_record: Callable[[dict[str, Any]], None] | None = None

    @property
    def seed(self) -> int:
        return SEEDS[self.role]


@dataclass
class Written:
    """One attempt as written: its bundle entry (no verdict yet) and what judges it."""

    spec: RunSpec
    entry: dict[str, Any]
    attempt: rr.Attempt
    #: The judged arms' per-game records; ``None`` when they never all ran.
    judged: tuple[list[dict[str, Any]], ...] | None
    interruptions: list[str]


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
        pairs = PRIMARY[spec.primary]
        shift = spec.primary_rotation % len(pairs)
        return games([*pairs[shift:], *pairs[:shift]], n)
    if name == "symmetry":
        return games(spec.symmetry, n)
    if name.startswith("secondary"):
        return games(EVEN, n)
    return games(WIN, n)


def _installed_version(pinned: str) -> str:
    """``torch.__version__`` of the registered build (a tag-less pin installs a cu128 wheel)."""
    return pinned if "+" in pinned else f"{pinned}+cu128"


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
            "version": _installed_version(registered.torch_version),
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


def _launched(spec: RunSpec, names: Sequence[str]) -> list[str]:
    """The invocations run_card started: those it recorded and the one in flight."""
    if spec.launched is not None:
        return list(spec.launched)
    return list(names) if spec.recorded is None else list(names[: spec.recorded + 1])


def _nvidia_smi(spec: RunSpec, gpus: Sequence[str]) -> Any:
    """``preflight.nvidia_smi`` as run_card records it: one csv line per GPU, or an error."""
    if spec.driver is None:
        return {"error": "FileNotFoundError"}
    drivers = [spec.driver] * len(gpus) if isinstance(spec.driver, str) else list(spec.driver)
    return [
        f"{i}, {name}, 00000000:0{i + 1}:00.0, {driver}"
        for i, (name, driver) in enumerate(zip(gpus, drivers, strict=True))
    ]


def write_run(
    bundle_dir: Path, spec: RunSpec, sha: str, section: str, position: int = 0
) -> Written:
    """Write one attempt's components under ``bundle_dir/<run_id>/``.

    The entry has no verdict yet: a re-run's depends on its compare record
    (:func:`write_records`, then :func:`stored_verdict`).
    """
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
    devices = registered_devices(section)
    run_json["started_at"] = spec.started_at or f"2026-11-{11 + position:02d}T08:00:00Z"
    if not spec.legacy_run_json:
        run_json["launched"] = _launched(spec, list(registered))
        if devices is not None:
            gpus = [devices.gpus[i] for i in sorted(devices.gpus)]
            run_json["preflight"] = {"nvidia_smi": _nvidia_smi(spec, gpus)}
    if plan is not None and spec.edit_device_plan is not None:
        plan = spec.edit_device_plan(plan)
    if plan is not None:
        run_json["device_plan"] = plan
    if spec.probe:
        run_json["probe"] = True
    if spec.edit_run_json is not None:
        spec.edit_run_json(run_json)
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
    iterations: Path | None = None
    if lines:
        rel = (base / "iterations.jsonl").as_posix()
        entry["iterations"] = {"path": rel, **_write(bundle_dir / rel, "\n".join(lines) + "\n")}
        iterations = bundle_dir / rel
    entry["arms"] = {}
    arenas: dict[str, Path] = {}
    for name, (cand, opp, rec, launched) in arm_records.items():
        envelope = _envelope(cand, opp, rec, seed=spec.seed, sha=sha, launched=launched)
        rel = (base / "arena" / name / "result.json").as_posix()
        written = _write(bundle_dir / rel, json.dumps(envelope, indent=2))
        summary = vre.arm_summary(envelope["results"][0], vre.ARM_CONFIDENCE.get(name, 0.95))
        entry["arms"][name] = {"path": rel, **written, **summary}
        arenas[rr.arena_directory(str(flag_value(launched, "--out")))] = bundle_dir / rel
    interruptions = []
    if spec.status != "COMPLETED":
        interruptions.append(f"run.json status is {spec.status!r}")
    if spec.resumed:
        interruptions.append("resumed")
    if any(str(n).upper().startswith("MCTS_") for n in spec.env_var_names):
        interruptions.append("MCTS_")
    judged = None
    if set(JUDGED_ARMS) <= set(arm_records):
        judged = tuple(arm_records[name][2] for name in JUDGED_ARMS)
    attempt = rr.Attempt(
        spec.run_id,
        spec.role,
        sha,
        run_json,
        entry["run_json"]["sha256"],
        spec.status == "COMPLETED",
        tuple(interruptions),
        control_void=judged is not None and compute_verdict(*judged).verdict == "VOID",
        devices_registered=devices is not None,
        iterations=iterations,
        arenas=arenas,
    )
    return Written(spec, entry, attempt, judged, interruptions)


def checkpoint_digests(spec: RunSpec) -> dict[str, dict[str, str]]:
    """Stand-in tensor digests for the checkpoints the attempt saved.

    ``checkpoint_0`` and one every :data:`SAVE_INTERVAL` finished iterations, each
    a function of the seed, the iteration and the spec's salt alone, so an attempt
    and its faithful re-run agree. Nothing in a bundle anchors these.
    """
    saved = [0, *range(SAVE_INTERVAL, spec.iterations_written + 1, SAVE_INTERVAL)]
    salt = {n: "" if n == 0 else spec.checkpoint_salt for n in saved}  # the seed fixes number 0
    return {
        f"checkpoint_{n}.pt": {
            "conv.weight": hashlib.sha256(f"{spec.seed}:{n}:{salt[n]}".encode()).hexdigest()
        }
        for n in saved
    }


def run_digest(written: Written) -> dict[str, Any]:
    """What ``reproduction.run_digest`` gives for the attempt's run directory, as recorded."""
    unreadable: dict[str, str] = {}
    attempt = written.attempt
    digest = {
        rp.SCHEMA_KEY: rp.DIGEST_SCHEMA_VERSION,
        rp.COMPONENT_CHECKPOINTS: checkpoint_digests(written.spec),
        rp.COMPONENT_ARENAS: {name: rp.arena_digest(p) for name, p in attempt.arenas.items()},
        rp.COMPONENT_ITERATIONS: (
            rp.iteration_digests(attempt.iterations, unreadable) if attempt.iterations else {}
        ),
        rp.UNREADABLE: unreadable,
    }
    return dict(json.loads(json.dumps(digest, sort_keys=True)))


def compare_record(void: Written, rerun: Written) -> dict[str, Any]:
    """The record ``reproduction compare <void's dir> <re-run's dir> --out`` writes."""
    sides = {rp.RECORD_ATTEMPT: void, rp.RECORD_RERUN: rerun}
    digests = {side: run_digest(written) for side, written in sides.items()}
    status, differences = rp.compare_status(digests[rp.RECORD_ATTEMPT], digests[rp.RECORD_RERUN])
    record: dict[str, Any] = {
        "tool": "alpha_zero.evaluation.reproduction compare",
        rp.RECORD_SCHEMA_KEY: rp.RECORD_SCHEMA_VERSION,
        "components_compared": rp.compared_components(digests[rp.RECORD_ATTEMPT]),
        "components": rp.component_counts(digests[rp.RECORD_ATTEMPT]),
        "rerun_components": rp.component_counts(digests[rp.RECORD_RERUN]),
        rp.RECORD_DIFFERENCES: differences,
        rp.RECORD_DIGESTS: digests,
        rp.RECORD_STATUS: status,
    }
    for side, written in sides.items():
        attempt = written.attempt
        record[f"{side}_dir"] = f"E:\\civ6az-data\\runs\\{attempt.run_id}"
        record[side] = {
            "card_id": attempt.run_id,
            "sha": attempt.sha,
            "status": attempt.record["status"],
            rp.RUN_JSON_SHA256: attempt.run_json_sha256,
        }
    return record


def write_records(bundle_dir: Path, runs: Sequence[Written]) -> dict[str, list[str]]:
    """Write each interrupted attempt's compare record; ``{re-run: why it is non-reproducing}``."""
    attempts = [written.attempt for written in runs]
    non_reproducing: dict[str, list[str]] = {}
    for void in runs:
        rerun = rr.record_rerun(void.attempt, attempts)
        if rerun is None or not void.spec.compare_record:
            continue
        record = compare_record(void, next(w for w in runs if w.attempt is rerun))
        if void.spec.edit_record is not None:
            void.spec.edit_record(record)
        rel = rr.record_name(void.spec.run_id)
        saved = _write(bundle_dir / rel, json.dumps(record, indent=2, sort_keys=True))
        result = rr.outcome(record, void.attempt, rerun)
        void.entry["reproduction"] = rr.bundle_reference({"path": rel, **saved}, rerun, result)
        if result.kind == rr.OUTCOME_NON_REPRODUCING:
            non_reproducing.setdefault(rerun.run_id, []).extend(
                rr.verdict_reasons(void.attempt, result)
            )
    return non_reproducing


def stored_verdict(written: Written, non_reproducing: Sequence[str] = ()) -> dict[str, Any]:
    """The verdict the bundle stores for an attempt, the re-run rule applied."""
    if written.judged is None:  # the judged arms never all ran: VOID by status
        return {"verdict": "VOID", "reasons": written.interruptions}
    return compute_verdict(
        *written.judged, interruptions=written.interruptions, non_reproducing=non_reproducing
    ).to_dict()


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
    written = [
        write_run(bundle_dir, s, rc_sha if s.at_rc else sha, section, position)
        for position, s in enumerate(specs)
    ]
    non_reproducing = write_records(bundle_dir, written)
    for item in written:
        item.entry["verdict"] = stored_verdict(item, non_reproducing.get(item.spec.run_id, ()))
    runs = [item.entry for item in written]
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
