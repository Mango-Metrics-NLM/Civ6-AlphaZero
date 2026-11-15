"""Real producers -> ``build_release_bundle`` -> ``verify_release_evidence`` (final review I-3).

Every other release-evidence test feeds the verifier hand-built records
(``release_evidence_fixture``). This one feeds it what the producers actually
write, once, at tiny scale:

1. ``alpha_zero.local.run_card.run_card`` runs a card through its real pricing,
   data-root handling, isolation (``enter_local_mode``) and ``run.json`` writer;
2. each invocation runs the real ``alpha_zero.training_civ6`` and
   ``alpha_zero.evaluation.benchmark`` ``main()``, so ``iterations.jsonl``,
   checkpoints (with provenance) and every arm's ``result.json`` (with its
   evidence block) are the producers' own;
3. ``scripts/build_release_bundle.py`` assembles the bundle;
4. ``scripts/verify_release_evidence.py`` must exit 0 on it.

What is replaced, and why (so it is not mistaken for coverage):

* **Pinning.** ``run_card`` normally builds a venv from ``git archive`` of the
  card's SHA and launches each child as a subprocess. That takes minutes, so
  ``prepare_pinned_run``/``_preflight``/``_pip_freeze`` are stubbed and
  ``_run_invocation`` runs the same ``python -m <module> <launched argv>`` in this
  process, under the child environment run_card built. ``tests/scripts/test_run_card.py``
  covers the real pinning and subprocess path.
* **The registration.** Section 7.1's card is 980 games at board 8, so the
  repository under test holds a *test-only* section 7.1 with the same structure
  (arms ``primary``/``symmetry``/``positive-control``/``final-vs-random``, the
  registered seed, ``Branch in force: 40``) at board 4 / 8 turns, 2 iterations,
  2 games per arm and 20 for the positive control (fewer cannot clear its 0.5
  bound, and a VOID-only bundle rightly fails). Its digest is therefore not the registration log's; the digest gate is
  ``tests/regression/test_a1_registration_digest_regression.py``.
* **The data root.** ``<D>`` is ``E:\\civ6az-data\\runs\\<card>`` in the
  registration; here ``a1_registration.run_directory`` points it at the test's
  temporary root, so the paths the verifier reconstructs are the ones that ran.
* **Check runs.** The rc's CI is read through the fake GitHub API the verifier's
  own tests use.
* **The clock.** ``run.json``'s ``started_at`` has one-second resolution and the
  re-run rule orders attempts by it, strictly. Real attempts are hours apart;
  these tiny ones can start within one second, so ``run_card._now`` ticks one
  minute per call here.

So what this proves: the verifier's schema, registration, recomputation, verdict,
identity and freshness checks all accept real producer output. What it does not:
the pinned-venv launch, a real ``E:`` data root, the real 980-game registration's
digest, and the GitHub API.

``test_an_interrupted_attempt_and_its_rerun_build_and_verify`` adds the re-run
rule (section 7.1, "After a VOID") with the same producers: the card is
interrupted inside its symmetry arm (a ``KeyboardInterrupt`` from the invocation,
which is how ``run_card`` sees Ctrl+C), run again under a new ``card_id``, compared
by the real ``python -m alpha_zero.evaluation.reproduction compare``, and bundled
with ``--compare``. It then edits the interrupted attempt's iteration record, so
the record made before contradicts what is bundled and a new comparison DIFFERS.

Negative controls for the three re-run tests (2026-11-15; each planted in
``scripts/build_release_bundle.py`` unless named, the planted text read back, the
test seen to FAIL, the file restored byte-identical):

* ``attach_records`` copies nothing; ``set_verdicts`` ignores the re-run rule; a
  second record for one attempt is accepted; a record naming another re-run is
  accepted; and, in ``scripts/release_reproduction.py``, ``check_record`` does not
  hold the digests to the bundled files -> each failed
  ``test_an_interrupted_attempt_and_its_rerun_build_and_verify``;
* an attempt that launched nothing is listed ->
  ``test_an_attempt_that_launched_nothing_is_not_bundled`` failed;
* ``collect_run`` does not read ``device_plan`` (``check_registered_devices``
  removed) -> ``test_a_run_on_unregistered_devices_is_bundled_as_void`` failed.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import itertools
import json
import os
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from alpha_zero.evaluation import a1_registration
from alpha_zero.evaluation import reproduction as rp
from alpha_zero.evaluation.a1_protocol import PROTOCOL_BEGIN, PROTOCOL_END
from alpha_zero.local import run_card as rc
from alpha_zero.local.pinned_env import PinnedEnv
from scripts import build_release_bundle as builder
from scripts import verify_release_evidence as vre
from tests.scripts import release_evidence_fixture as fx
from tests.scripts.git_fixture import GitRepo

CARD_ID = "a1-tiny"
RERUN_ID = "a1-tiny-r2"
SEED = 20261111
#: Runs a card; with ``interrupt_in`` the named invocation is interrupted when it starts.
RunCard = Callable[..., Path]
#: The stand-in clock (module docstring): ``run_card._now``'s format, a minute per call.
CLOCK_START = datetime(2026, 12, 1, tzinfo=UTC)
CLOCK_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
BENCH = (
    "python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 4 "
    "--max-turns 8 --sims 2 --gumbel-k 2 --mcts-type gumbel --device cpu"
)
COMMANDS = [
    (
        "train",
        "python -m alpha_zero.training_civ6 --seed 20261111 --board_size 4 --max_turns 8 "
        "--num_iterations 2 --num_games 2 --mcts gumbel --gumbel_k 2 --gumbel_sims 2 "
        "--device cpu --bootstrap none --save_interval 1 --eval_games 0 "
        "--checkpoint_dir <D>/ckpt",
    ),
    (
        "primary",
        f"{BENCH} --checkpoint <D>/ckpt/checkpoint_2.pt "
        "--opponent-checkpoint <D>/ckpt/checkpoint_0.pt --n-games 2 --out <D>/arena/primary",
    ),
    (
        "symmetry",
        f"{BENCH} --checkpoint <D>/ckpt/checkpoint_0.pt "
        "--opponent-checkpoint <D>/ckpt/checkpoint_0.pt --n-games 2 --out <D>/arena/symmetry",
    ),
    (
        "positive-control",
        f"{BENCH} --candidate heuristic --opponents=random --n-games 20 "
        "--out <D>/arena/positive_control",
    ),
    (
        "final-vs-random",
        f"{BENCH} --checkpoint <D>/ckpt/checkpoint_2.pt --opponents=random --n-games 2 "
        "--out <D>/arena/final_vs_random",
    ),
]


def tiny_evaluation_md() -> str:
    """A ``docs/EVALUATION.md`` whose section 7.1 registers the tiny card."""
    block = "\n".join(f"# {name}\n{command}" for name, command in COMMANDS)
    return (
        "# Evaluation (test-only registration)\n\n"
        f"{PROTOCOL_BEGIN} test -->\n"
        "### 7.1 A1 (tiny, test only)\n\n"
        "<!-- a1-commands:base -->\n"
        f"```text\n{block}\n```\n\n"
        "- **Branch in force: 40.**\n"
        f"{PROTOCOL_END} -->\n"
    )


@pytest.fixture(autouse=True)
def restore_process_state() -> Iterator[None]:
    """``enter_local_mode`` and the in-process children mutate the environment and argv."""
    environ, argv = os.environ.copy(), list(sys.argv)
    yield
    os.environ.clear()
    os.environ.update(environ)
    sys.argv[:] = argv


def _in_process(cmd: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> int:
    """``run_card._run_invocation``, run in this process: ``python -m MODULE ARGV...``."""
    assert list(cmd[1:2]) == ["-m"], cmd
    module, argv = cmd[2], list(cmd[3:])
    saved = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    sys.argv[:] = [module, *argv]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            entry = importlib.import_module(module)
            try:
                code = entry.main()
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
        return int(code or 0)
    finally:
        log.write_text(out.getvalue(), encoding="utf-8")
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture
def producer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[GitRepo, RunCard]:
    """A repository at a tagged rc whose section 7.1 is the tiny card, and a way to run it."""
    root = tmp_path / "root"
    monkeypatch.setattr(
        a1_registration,
        "run_directory",
        lambda card_id, data_root="": os.path.join(str(root.resolve()), "runs", card_id),
    )
    repo = GitRepo(tmp_path / "repo")
    evaluation = tiny_evaluation_md()
    run_sha = repo.commit(
        "code",
        {
            "pyproject.toml": fx.PYPROJECT,
            "docs/EVALUATION.md": evaluation,
            "docs/evidence/README.md": (
                "# Evidence index\n\n| Artifact | Status |\n|---|---|\n"
                "| [`older/`](older/report.md) | current |\n\nProse after the table.\n"
            ),
            "CHANGELOG.md": "# Changelog\n\n## [2.1.0]\n\n- Tiny.\n",
        },
    )
    repo.tag(fx.RC_TAG, run_sha)
    section = evaluation.split("\n", 2)[2]

    venv_python = tmp_path / "venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
    pinned = PinnedEnv(python=venv_python, src=fx.REPO_ROOT, sha=run_sha, wheel_sha256="0" * 64)
    monkeypatch.setattr(rc, "prepare_pinned_run", lambda *a, **k: pinned)
    monkeypatch.setattr(rc, "_preflight", lambda env, child: {"exit_code": 0})
    monkeypatch.setattr(rc, "verify_preflight", lambda env, report: None)
    monkeypatch.setattr(rc, "_pip_freeze", lambda env, child, runner: [])
    minutes = itertools.count()
    monkeypatch.setattr(
        rc, "_now", lambda: (CLOCK_START + timedelta(minutes=next(minutes))).strftime(CLOCK_FORMAT)
    )
    monkeypatch.setattr(
        rc,
        "harden_data_root",
        lambda r, **_: [(Path(r) / s).mkdir(parents=True, exist_ok=True) for s in rc.SUBDIRS],
    )

    def run(card_id: str = CARD_ID, *, interrupt_in: str | None = None) -> Path:
        invocations = a1_registration.registered_invocations(section, card_id=card_id, seed=SEED)
        card = tmp_path / f"{card_id}.yaml"
        card.write_text(
            yaml.safe_dump(
                {
                    "card_id": card_id,
                    "sha": run_sha,
                    "max_games": 1000,
                    "invocations": [
                        {
                            "name": name,
                            "runner": "training_civ6" if name == "train" else "benchmark",
                            "argv": argv,
                        }
                        for name, argv in invocations.items()
                    ],
                }
            ),
            encoding="utf-8",
        )

        def invoke(cmd: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> int:
            if log.stem == interrupt_in:
                raise KeyboardInterrupt  # Ctrl+C, as run_card's signal handler raises it
            return _in_process(cmd, cwd=cwd, env=env, log=log)

        monkeypatch.setattr(rc, "_run_invocation", invoke)
        if interrupt_in is not None:
            with pytest.raises(KeyboardInterrupt):
                rc.run_card(card, data_root=root, repo_root=repo.path)
            return root / "runs" / card_id
        result = rc.run_card(card, data_root=root, repo_root=repo.path)
        logs = {
            p.name: p.read_text(encoding="utf-8")[-2000:]
            for p in (result.run_dir / "logs").glob("*")
        }
        assert result.status == rc.STATUS_COMPLETED, logs
        return result.run_dir

    return repo, run


@pytest.fixture
def released(producer: tuple[GitRepo, RunCard]) -> tuple[GitRepo, Path]:
    """The repository, and the tiny card run to completion."""
    repo, run = producer
    return repo, run()


def test_real_producer_output_builds_and_verifies(
    released: tuple[GitRepo, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    repo, run_dir = released
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "COMPLETED"
    assert [i["name"] for i in record["invocations"]] == [n for n, _ in COMMANDS]

    code = builder.main(
        [
            "--release", fx.RELEASE, "--rc", fx.RC_TAG, "--run", f"A1={run_dir}",
            "--repo-root", str(repo.path), "--date", "2026-12-01",
        ]
    )  # fmt: skip
    out = capsys.readouterr()
    assert code == builder.EXIT_BUILT, out.out + out.err
    bundle_dir = repo.path / fx.BUNDLE_DIR
    bundle = json.loads((bundle_dir / vre.BUNDLE_FILE).read_text(encoding="utf-8"))
    (run,) = bundle["runs"]
    assert set(run["arms"]) == {"primary", "symmetry", "positive_control", "final_vs_random"}
    assert set(run["checkpoints"]) == {"ckpt/checkpoint_0.pt", "ckpt/checkpoint_2.pt"}
    assert bundle["learning_claimed"] is False  # no A1-R: the claim cannot be made
    index = (repo.path / "docs" / "evidence" / "README.md").read_text(encoding="utf-8")
    assert "](release_v2_1_0/report.md)" in index
    assert index.index("release_v2_1_0") < index.index("Prose after the table")

    repo.git("add", "-A")
    repo.git("commit", "-q", "-m", "evidence")
    code = vre.main(
        [str(bundle_dir / vre.BUNDLE_FILE), "--repo-root", str(repo.path), "--repo", fx.REPO],
        api=fx.fake_api(bundle["rc_sha"]),
        env={},
    )
    out = capsys.readouterr()
    assert code == vre.EXIT_PASS, out.out + out.err

    # Refuses to overwrite what it wrote.
    again = builder.main(
        ["--release", fx.RELEASE, "--rc", fx.RC_TAG, "--run", f"A1={run_dir}",
         "--repo-root", str(repo.path)]
    )  # fmt: skip
    assert again == builder.EXIT_REFUSED
    assert "never overwritten" in capsys.readouterr().err

    # Negative control: one tampered arm and the build fails and publishes nothing.
    shutil.rmtree(bundle_dir)
    result = run_dir / "arena" / "symmetry" / "result.json"
    envelope = json.loads(result.read_text(encoding="utf-8"))
    envelope["results"][0]["draws"] += 1
    result.write_text(json.dumps(envelope), encoding="utf-8")
    code = builder.main(
        ["--release", fx.RELEASE, "--rc", fx.RC_TAG, "--run", f"A1={run_dir}",
         "--repo-root", str(repo.path)]
    )  # fmt: skip
    assert code == builder.EXIT_FAILED
    assert "arm symmetry" in capsys.readouterr().err
    assert not bundle_dir.exists()


def test_an_interrupted_attempt_and_its_rerun_build_and_verify(
    producer: tuple[GitRepo, RunCard], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Section 7.1, "After a VOID", end to end: interrupt, re-run, compare, bundle, verify."""
    repo, run = producer
    void_dir = run(CARD_ID, interrupt_in="symmetry")
    void = json.loads((void_dir / "run.json").read_text(encoding="utf-8"))
    assert void["status"] == rc.STATUS_VOID
    assert [i["name"] for i in void["invocations"]] == ["train", "primary"]
    assert void["launched"] == ["train", "primary", "symmetry"]
    rerun_dir = run(RERUN_ID)
    rerun_started = json.loads((rerun_dir / "run.json").read_text(encoding="utf-8"))["started_at"]
    assert void["started_at"] < rerun_started, "the re-run is the attempt that started later"

    compare = tmp_path / "compare.json"
    assert rp.main(["compare", str(void_dir), str(rerun_dir), "--out", str(compare)]) == 0
    record = json.loads(compare.read_text(encoding="utf-8"))
    assert record[rp.RECORD_STATUS] == rp.STATUS_REPRODUCED
    # checkpoint_0..2, the two iteration records and the one arm the attempt finished.
    assert record["components"] == {"checkpoints": 3, "arenas": 1, "iterations": 2}
    assert record[rp.RECORD_ATTEMPT]["card_id"] == CARD_ID
    assert record[rp.RECORD_RERUN]["card_id"] == RERUN_ID
    capsys.readouterr()

    build = [
        "--release", fx.RELEASE, "--rc", fx.RC_TAG, "--run", f"A1={void_dir}",
        "--run", f"A1={rerun_dir}", "--repo-root", str(repo.path), "--date", "2026-12-01",
    ]  # fmt: skip
    bundle_dir = repo.path / fx.BUNDLE_DIR

    # Without the record the bundle is not built: the builder verifies what it staged.
    assert builder.main(build) == builder.EXIT_FAILED
    err = capsys.readouterr().err
    assert f"run {CARD_ID} (A1) reproduction record: missing" in err
    assert not bundle_dir.exists()

    # Records the builder refuses: no runs named, the wrong pair, the same pair twice.
    nameless = tmp_path / "nameless.json"
    nameless.write_text(json.dumps({**record, rp.RECORD_ATTEMPT: None}), encoding="utf-8")
    backwards = tmp_path / "backwards.json"
    rp.main(["compare", str(rerun_dir), str(void_dir), "--out", str(backwards)])
    capsys.readouterr()
    elsewhere = tmp_path / "elsewhere.json"
    other_rerun = {**record[rp.RECORD_RERUN], "card_id": "a1-other"}
    elsewhere.write_text(json.dumps({**record, rp.RECORD_RERUN: other_rerun}), encoding="utf-8")
    for files, message in (
        ([nameless], "the record names no runs"),
        ([backwards], f"it compares '{RERUN_ID}' with '{CARD_ID}', which are not"),
        ([elsewhere], f"it compares '{CARD_ID}' with 'a1-other', which are not"),
        ([compare, compare], f"a second record for '{CARD_ID}'"),
    ):
        flags = [flag for file in files for flag in ("--compare", str(file))]
        assert builder.main([*build, *flags]) == builder.EXIT_REFUSED, message
        assert message in capsys.readouterr().err
        assert not bundle_dir.exists()

    code = builder.main([*build, "--compare", str(compare)])
    out = capsys.readouterr()
    assert code == builder.EXIT_BUILT, out.out + out.err
    bundle = json.loads((bundle_dir / vre.BUNDLE_FILE).read_text(encoding="utf-8"))
    attempt, judged = bundle["runs"]
    assert (attempt["run_id"], judged["run_id"]) == (CARD_ID, RERUN_ID)
    copied = bundle_dir / f"{CARD_ID}.reproduction.json"
    assert copied.read_bytes() == compare.read_bytes(), "beside both attempts, byte for byte"
    assert attempt["reproduction"] == {
        "path": copied.name,
        "sha256": vre.sha256_file(copied),
        "rerun": RERUN_ID,
        "outcome": "REPRODUCED",
        "reasons": [],
    }
    assert attempt["verdict"]["verdict"] == "VOID" and set(attempt["arms"]) == {"primary"}
    assert judged["verdict"]["verdict"] != "VOID" and "label" not in judged["verdict"]
    assert "reproduction" not in judged
    report = (bundle_dir / vre.REPORT_FILE).read_text(encoding="utf-8")
    assert f"Reproduction check against `{RERUN_ID}` (`{copied.name}`): REPRODUCED" in report

    def verified() -> tuple[int, str]:
        repo.git("add", "-A")
        repo.git("commit", "-q", "-m", "evidence")
        code = vre.main(
            [str(bundle_dir / vre.BUNDLE_FILE), "--repo-root", str(repo.path), "--repo", fx.REPO],
            api=fx.fake_api(bundle["rc_sha"]),
            env={},
        )
        said = capsys.readouterr()
        return code, said.out + said.err

    code, said = verified()
    assert code == vre.EXIT_PASS, said

    # The attempt's self-play went otherwise: its first iteration record counts another
    # number of examples. (The verifier does not recount them: only the comparison can tell.)
    shutil.rmtree(bundle_dir)
    iterations = void_dir / "ckpt" / "iterations.jsonl"
    lines = [json.loads(line) for line in iterations.read_text(encoding="utf-8").splitlines()]
    lines[0]["self_play"]["examples"] += 1
    iterations.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    # The record made before no longer describes the attempt that is bundled.
    assert builder.main([*build, "--compare", str(compare)]) == builder.EXIT_FAILED
    assert f"attempt: iterations ['1'] differ from the files bundled for {CARD_ID}" in (
        capsys.readouterr().err
    )
    assert not bundle_dir.exists()
    differs = tmp_path / "differs.json"
    assert rp.main(["compare", str(void_dir), str(rerun_dir), "--out", str(differs)]) == 1
    capsys.readouterr()
    code = builder.main([*build, "--compare", str(differs)])
    out = capsys.readouterr()
    assert code == builder.EXIT_BUILT, out.out + out.err
    bundle = json.loads((bundle_dir / vre.BUNDLE_FILE).read_text(encoding="utf-8"))
    attempt, judged = bundle["runs"]
    assert attempt["reproduction"]["outcome"] == "NON_REPRODUCING"
    (reason,) = attempt["reproduction"]["reasons"]
    assert reason.startswith("iteration record 1: 1 field(s) differ: self_play.examples")
    assert judged["verdict"]["label"] == "non-reproducing re-run"
    assert judged["verdict"]["verdict"] not in ("PASS", "VOID")
    assert bundle["learning_claimed"] is False
    report = (bundle_dir / vre.REPORT_FILE).read_text(encoding="utf-8")
    assert (
        f"## `{RERUN_ID}` (A1): {judged['verdict']['verdict']} (non-reproducing re-run)" in report
    )
    code, said = verified()
    assert code == vre.EXIT_PASS, said


def test_an_attempt_that_launched_nothing_is_not_bundled(
    released: tuple[GitRepo, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refused before its first invocation: not an attempt, so the builder does not list it."""
    repo, run_dir = released
    refused = tmp_path / "refused"
    refused.mkdir()
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    record.update(status=rc.STATUS_FAILED, invocations=[], launched=[], error="devices: no cuda:1")
    (refused / "run.json").write_text(json.dumps(record), encoding="utf-8")
    code = builder.main(
        ["--release", fx.RELEASE, "--rc", fx.RC_TAG, "--run", f"A1={refused}",
         "--run", f"A1={run_dir}", "--repo-root", str(repo.path)]
    )  # fmt: skip
    assert code == builder.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "run.json records no launched invocation" in err and "drop its --run" in err
    assert not (repo.path / fx.BUNDLE_DIR).exists()


def test_a_run_on_unregistered_devices_is_bundled_as_void(
    released: tuple[GitRepo, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Section 7.1 "Devices": the builder reads ``device_plan`` as the verifier does.

    The tiny registration names no devices, so it is given some here; the run's
    ``device_plan`` (a CPU card without a torch pin) is then not the registered one.
    The builder must store VOID, which is what the verifier recomputes. That the
    release still fails is the role having no non-VOID attempt, not a disagreement.
    """
    repo, run_dir = released
    read = vre.read_registration
    devices = a1_registration.RegisteredDevices(
        "torch==2.11.0+cu128", "https://download.pytorch.org/whl/cu128", "0,1", {0: "A", 1: "B"}
    )

    def with_devices(git: Any, sha: str, run_id: str, role: str) -> vre.Registration:
        registration = read(git, sha, run_id, role)
        registration.devices = devices
        return registration

    monkeypatch.setattr(vre, "read_registration", with_devices)
    code = builder.main(
        ["--release", fx.RELEASE, "--rc", fx.RC_TAG, "--run", f"A1={run_dir}",
         "--repo-root", str(repo.path)]
    )  # fmt: skip
    assert code == builder.EXIT_FAILED
    err = capsys.readouterr().err
    assert "A1 has no non-VOID attempt" in err
    assert "device_plan.torch_pin None is not the registered" in err
    assert "stored verdict" not in err, "the builder and the verifier disagree on the verdict"
