"""Tests for ``scripts/check_determinism.py``: does a seeded A1-shaped run repeat in a new process?

There is no GPU where these run, and none is needed: the harness is
device-agnostic, so the CPU exercises every line a ``cuda:0`` / ``cuda:1`` check
runs except the kernels themselves. ``real_check`` runs the whole thing once --
two real subprocesses, default network size -- and the tests around it hold the
harness to what makes its answer worth having: separate processes, an untouched
environment, both directions compared, and no pass that compared nothing.

Negative controls (2026-11-14, re-run against the final code). Each defect was
planted, the planted text was read back from disk, the listed tests were seen to
FAIL, and the file was restored and checked byte-identical (sha256) before the
next one. In ``scripts/check_determinism.py`` unless another file is named:

1.  The reverse comparison dropped from ``compare_runs`` ->
    ``test_a_component_only_one_repeat_holds_is_a_difference[run_b]`` failed
    (status ``REPRODUCED``), and
    ``test_a_planted_weight_change_in_a_real_checkpoint_is_reported`` (one
    direction reported instead of two).
2.  ``coverage_gaps`` returns ``[]`` -> all five
    ``test_no_difference_is_not_enough`` cases and both
    ``test_cuda_repeats_that_agree_without_enforcement_are_incomplete`` cases
    failed (``REPRODUCED`` for repeats with no arena, no trained checkpoint,
    untrained weights, nothing at all, or no enforcement).
3.  ``resolve_run_device`` returns the fallback device instead of refusing ->
    both ``test_an_unavailable_gpu_is_refused_before_anything_runs`` cases, both
    ``test_a_malformed_or_unknown_device_is_refused`` cases,
    ``test_the_parser_leaves_the_arena_device_unset_and_resolution_follows_training``
    and ``test_a_worker_started_by_hand_is_priced_and_resolves_its_devices`` failed.
4.  A failed worker ignored (``check`` no longer returns on a non-zero status) ->
    ``test_a_failed_worker_fails_the_check_with_the_end_of_its_log`` and
    ``test_an_approval_reaches_the_worker_and_the_arena`` failed.
5.  Worker mode not priced (``enforce_cli`` skipped when ``--worker`` is given) ->
    ``test_a_worker_started_by_hand_is_priced_and_resolves_its_devices`` failed.
6.  ``_run_worker_process`` passes ``env={"PYTHONHASHSEED": "0"}`` ->
    ``test_workers_inherit_the_environment_and_get_no_hash_seed`` failed. Nothing
    else did: a Linux child with an emptied environment still trains, so the real
    check passed and that one test is the only guard on the environment.
7.  ``MIN_REPLAY_SIZE = 1000`` (the trainer's default) ->
    ``test_two_processes_reproduce_on_cpu`` and the five other ``real_check``
    tests that read a worker's output failed (``WORKER_FAILED``: the worker
    refuses to report on a run that never trained).
8.  The worker's "did every iteration train?" guard removed, with defect 7 still
    planted -> ``test_two_processes_reproduce_on_cpu``,
    ``test_training_changed_the_weights_in_the_real_run`` and
    ``test_a_planted_weight_change_in_a_real_checkpoint_is_reported`` still failed
    (``INCOMPLETE``: ``checkpoint_1.pt equals checkpoint_0.pt``) -- the second
    line of defence doing its job.
9.  ``training_argv`` drops ``--bootstrap none`` ->
    ``test_the_pipeline_is_the_registered_commands_at_other_sizes`` and
    ``test_training_changed_the_weights_in_the_real_run`` (iteration 0 played by
    SimpleMCTS) failed.
10. The parser fills in ``--arena-device`` from ``--train-device`` ->
    ``test_the_parser_leaves_the_arena_device_unset_and_resolution_follows_training``
    and ``test_the_json_names_the_devices_and_the_torch_build`` failed.
11. The CUDA-enforcement rule removed from ``coverage_gaps`` -> both
    ``test_cuda_repeats_that_agree_without_enforcement_are_incomplete`` cases
    failed.
12. ``check`` always passes ``cuda=False`` -> the two CUDA cases of
    ``test_the_check_tells_the_comparison_when_a_phase_ran_on_cuda`` failed.
13. ``alpha_zero/utils/game_budget.py``: ``_determinism_check`` ignores
    ``repeats`` -> ``test_the_check_is_priced_at_twice_one_repeat``,
    ``test_an_over_cap_check_is_refused_before_anything_runs`` and
    ``test_the_json_names_the_devices_and_the_torch_build`` failed. Planted in
    memory (a pytest plugin replaced the registry entry) rather than by rewriting
    a file another engineer was editing at the time.
14. ``alpha_zero/evaluation/reproduction.py``:
    ``ITERATION_RECORD_EXCLUDED_FIELDS = ()`` ->
    ``test_two_processes_reproduce_on_cpu`` and
    ``test_the_real_records_differ_only_in_the_excluded_fields`` failed (and every
    test built on the synthetic repeats). On two real runs the differing fields
    were exactly ``run_id``, ``started_at`` and ``self_play.seconds``.
15. ``alpha_zero/evaluation/reproduction.py``: ``compare_digests`` skips
    checkpoints -> ``test_a_planted_weight_change_in_a_real_checkpoint_is_reported``
    and both ``test_a_component_only_one_repeat_holds_is_a_difference`` cases
    failed.
16. ``alpha_zero/evaluation/reproduction.py``: game records not compared ->
    ``test_a_planted_game_change_in_a_real_arena_is_reported`` failed.
17. The worker leaves the trainer's ``log_dir`` at its default (relative to the
    checkout) -> ``test_each_repeat_ran_in_its_own_process`` failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from alpha_zero.evaluation import a1_protocol as ap
from alpha_zero.evaluation import reproduction as rp
from alpha_zero.evaluation.a1_registration import command_block
from alpha_zero.training.run_record import ITERATIONS_FILE, read_iteration_records
from alpha_zero.utils import game_budget
from scripts import check_determinism as cd
from tests.unit import test_reproduction as fx

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The smallest pipeline that still trains and plays an arena (6 games in all).
TINY = (
    *("--board_size", "6", "--max_turns", "6", "--k", "4", "--n", "4"),
    *("--iterations", "1", "--games", "1", "--arena-games", "2"),
)
TINY_ITERATIONS = 1
#: No machine this runs on has a 64th GPU.
ABSENT_GPU = "cuda:63"


# ---------------------------------------------------------------------------
# The real thing, once: two subprocesses on CPU
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealCheck:
    code: int
    payload: dict[str, Any]
    work: Path
    commands: list[list[str]]


@pytest.fixture(scope="module")
def real_check(tmp_path_factory: pytest.TempPathFactory) -> RealCheck:
    root = tmp_path_factory.mktemp("determinism")
    work, out = root / "work", root / "determinism.json"
    commands: list[list[str]] = []
    launch = cd._run_worker_process

    def recording(command: list[str], log: Path) -> int:
        commands.append(list(command))
        return launch(command, log)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cd, "_run_worker_process", recording)
        code = cd.main([*TINY, "--work-dir", str(work), "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    return RealCheck(code, payload, work, commands)


def _log_tails(check: RealCheck) -> str:
    return "\n".join(
        f"--- {name} ---\n" + "\n".join(cd._log_tail(check.work / f"{name}.log"))
        for name in cd.REPEAT_NAMES
    )


def test_two_processes_reproduce_on_cpu(real_check: RealCheck) -> None:
    payload = real_check.payload
    assert payload["status"] == cd.STATUS_REPRODUCED, (payload, _log_tails(real_check))
    assert real_check.code == cd.EXIT_REPRODUCED == 0
    assert payload["differences"] == [] and payload["coverage_gaps"] == []
    compared = payload["components_compared"]
    assert compared["checkpoints"] == TINY_ITERATIONS + 1, "checkpoint_0 and the trained one"
    assert compared["arenas"] == 1
    assert compared["iterations"] == TINY_ITERATIONS


def test_each_repeat_ran_in_its_own_process(real_check: RealCheck) -> None:
    assert len(real_check.commands) == cd.REPEATS == 2
    run_dirs = []
    for name, command in zip(cd.REPEAT_NAMES, real_check.commands, strict=True):
        assert command[:4] == [sys.executable, "-m", cd.MODULE, "--worker"]
        # Both devices are handed to the worker resolved, the unset arena one included.
        assert command[5:9] == ["--train-device", "cpu", "--arena-device", "cpu"]
        run_dirs.append(Path(command[4]))
        # Only the child writes this file, so its content is the child's own work.
        log = (real_check.work / f"{name}.log").read_text(encoding="utf-8")
        assert "Saved checkpoint" in log and "Benchmark: candidate vs" in log, name
        # Everything a repeat writes stays under its run directory, the trainer's log dir too.
        assert (real_check.work / name / cd.TRAINER_LOG_DIR_NAME).is_dir(), name
    assert run_dirs == [real_check.work / name for name in cd.REPEAT_NAMES]


def test_the_json_names_the_devices_and_the_torch_build(real_check: RealCheck) -> None:
    payload = real_check.payload
    assert payload["harness"] == cd.HARNESS
    assert payload["train_device"] == "cpu" and payload["arena_device"] == "cpu"
    assert payload["args"]["arena_device"] is None, "args are as typed; the arena followed"
    assert payload["gpu_names"] == {"train": None, "arena": None}
    assert payload["torch"] == str(torch.__version__)
    assert payload["torch_cuda"] == torch.version.cuda
    assert "cudnn" in payload and "deterministic_algorithms" in payload
    assert payload["args"]["iterations"] == TINY_ITERATIONS
    assert payload["repeats"] == cd.REPEATS
    assert payload["games_priced"] == cd.REPEATS * (1 * 1 + 2)
    assert payload["work_dir"] == str(real_check.work), "an explicit --work-dir is kept"


def _leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_leaves(item, f"{prefix}.{key}" if prefix else key))
        return flat
    return {prefix: value}


def test_the_real_records_differ_only_in_the_excluded_fields(real_check: RealCheck) -> None:
    """The exclusion list is read off two real runs: the clocks moved, nothing else did."""
    first, second = (
        _leaves(
            read_iteration_records(
                real_check.work / name / rp.CHECKPOINT_DIR_NAME / ITERATIONS_FILE
            )[0]
        )
        for name in cd.REPEAT_NAMES
    )
    moved = {path for path in first.keys() | second.keys() if first.get(path) != second.get(path)}
    excluded = {".".join(path) for path in rp.ITERATION_RECORD_EXCLUDED_FIELDS}
    assert moved <= excluded, f"fields outside the exclusion list moved: {moved - excluded}"
    assert "started_at" in moved, "two processes started seconds apart: this field must move"


def test_training_changed_the_weights_in_the_real_run(real_check: RealCheck) -> None:
    """Otherwise equal checkpoints would only show that the seeded initialisation repeats."""
    for name in cd.REPEAT_NAMES:
        digest = rp.run_digest(real_check.work / name)
        checkpoints = digest[rp.COMPONENT_CHECKPOINTS]
        assert checkpoints["checkpoint_1.pt"] != checkpoints["checkpoint_0.pt"], name
        (record,) = digest[rp.COMPONENT_ITERATIONS].values()
        assert record["train"] is not None and record["self_play"]["simple_mcts"] is False


def _copy(check: RealCheck, tmp_path: Path) -> Path:
    return Path(shutil.copytree(check.work, tmp_path / "copy"))


def test_a_planted_weight_change_in_a_real_checkpoint_is_reported(
    real_check: RealCheck, tmp_path: Path
) -> None:
    work = _copy(real_check, tmp_path)
    assert cd.compare_runs(work, TINY_ITERATIONS)["status"] == cd.STATUS_REPRODUCED
    path = work / cd.REPEAT_NAMES[1] / rp.CHECKPOINT_DIR_NAME / "checkpoint_1.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    key = next(k for k, v in payload["state_dict"].items() if v.is_floating_point() and v.numel())
    payload["state_dict"][key].view(-1)[0] += 1e-4
    torch.save(payload, path)
    outcome = cd.compare_runs(work, TINY_ITERATIONS)
    assert outcome["status"] == cd.STATUS_DIFFERS
    assert len(outcome["differences"]) == 2, "once in each direction"
    for difference in outcome["differences"]:
        assert "checkpoint_1.pt" in difference and key in difference
        assert "1 of " in difference, "exactly one tensor moved"


def test_a_planted_game_change_in_a_real_arena_is_reported(
    real_check: RealCheck, tmp_path: Path
) -> None:
    work = _copy(real_check, tmp_path)
    result = work / cd.REPEAT_NAMES[1] / rp.ARENA_DIR_NAME / cd.ARENA_ARM / rp.ARENA_RESULT_FILE
    envelope = json.loads(result.read_text(encoding="utf-8"))
    game = envelope["results"][0]["games"][1]
    game["result"] = "draw" if game["result"] != "draw" else "win"
    result.write_text(json.dumps(envelope), encoding="utf-8")
    outcome = cd.compare_runs(work, TINY_ITERATIONS)
    assert outcome["status"] == cd.STATUS_DIFFERS
    assert all(f"arena {cd.ARENA_ARM}" in d and "game index 1" in d for d in outcome["differences"])


def test_a_real_arena_is_compared_move_for_move(real_check: RealCheck, tmp_path: Path) -> None:
    """A repeat that played other moves to the same outcome is not a reproduction.

    Negative control: with the arena writing only ``{pair, seat, result}`` the first
    assertion fails (the records carry no digest to compare).
    """
    work = _copy(real_check, tmp_path)
    result = work / cd.REPEAT_NAMES[1] / rp.ARENA_DIR_NAME / cd.ARENA_ARM / rp.ARENA_RESULT_FILE
    envelope = json.loads(result.read_text(encoding="utf-8"))
    game = envelope["results"][0]["games"][0]
    assert game["moves"] > 0 and len(game["actions_sha256"]) == 64
    game["actions_sha256"] = hashlib.sha256(b"another game").hexdigest()
    result.write_text(json.dumps(envelope), encoding="utf-8")
    outcome = cd.compare_runs(work, TINY_ITERATIONS)
    assert outcome["status"] == cd.STATUS_DIFFERS
    assert all(f"arena {cd.ARENA_ARM}" in d and "game index 0" in d for d in outcome["differences"])


# ---------------------------------------------------------------------------
# compare_runs on synthetic repeats: both directions, and no vacuous pass
# ---------------------------------------------------------------------------


@pytest.fixture
def repeats(tmp_path: Path) -> Path:
    """A work directory holding two faithful synthetic repeats (``tests/unit/test_reproduction``)."""
    work = tmp_path / "work"
    for name, stamp in zip(cd.REPEAT_NAMES, ("a", "bb"), strict=True):
        fx.build_run(work / name, stamp=stamp)
    assert fx.ARM == cd.ARENA_ARM
    return work


def test_faithful_synthetic_repeats_reproduce(repeats: Path) -> None:
    outcome = cd.compare_runs(repeats, fx.ITERATIONS)
    assert outcome["status"] == cd.STATUS_REPRODUCED
    assert outcome["differences"] == [] and outcome["coverage_gaps"] == []
    assert outcome["components_compared"] == {"checkpoints": 3, "arenas": 1, "iterations": 2}


@pytest.mark.parametrize("holder", cd.REPEAT_NAMES)
def test_a_component_only_one_repeat_holds_is_a_difference(repeats: Path, holder: str) -> None:
    """One direction ignores what only the candidate holds; the check runs both."""
    fx.write_checkpoint(fx.checkpoint_path(repeats / holder, 7), fx.state(7))
    outcome = cd.compare_runs(repeats, fx.ITERATIONS)
    assert outcome["status"] == cd.STATUS_DIFFERS
    other = next(name for name in cd.REPEAT_NAMES if name != holder)
    assert outcome["differences"] == [
        f"{holder} -> {other}: checkpoint checkpoint_7.pt: missing from the candidate"
    ]


def _drop_arena(run: Path) -> None:
    shutil.rmtree(run / rp.ARENA_DIR_NAME)


def _drop_trained_checkpoint(run: Path) -> None:
    fx.checkpoint_path(run, fx.ITERATIONS).unlink()


def _drop_initial_checkpoint(run: Path) -> None:
    fx.checkpoint_path(run, 0).unlink()


def _untrain(run: Path) -> None:
    fx.write_checkpoint(fx.checkpoint_path(run, fx.ITERATIONS), fx.state(0))


def _empty(run: Path) -> None:
    shutil.rmtree(run)
    run.mkdir()


@pytest.mark.parametrize(
    ("damage", "gap"),
    [
        (_drop_arena, f"arena {fx.ARM} was not compared"),
        (_drop_trained_checkpoint, f"checkpoint_{fx.ITERATIONS}.pt was not compared"),
        (_drop_initial_checkpoint, "checkpoint_0.pt was not compared"),
        (_untrain, "training changed no weight"),
        (_empty, "was not compared"),
    ],
)
def test_no_difference_is_not_enough(repeats: Path, damage: Any, gap: str) -> None:
    """Two repeats that agree because both lack the thing under test have shown nothing."""
    for name in cd.REPEAT_NAMES:
        damage(repeats / name)
    outcome = cd.compare_runs(repeats, fx.ITERATIONS)
    assert outcome["differences"] == []
    assert outcome["status"] == cd.STATUS_INCOMPLETE
    assert len(outcome["coverage_gaps"]) >= len(cd.REPEAT_NAMES)
    assert all(gap in reason for reason in outcome["coverage_gaps"] if cd.REPEAT_NAMES[0] in reason)


def _report(repeats: Path, enforced: bool | None) -> None:
    for name in cd.REPEAT_NAMES:
        target = repeats / name / cd.WORKER_REPORT_FILE
        if enforced is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(json.dumps({"deterministic_algorithms": enforced}), "utf-8")


@pytest.mark.parametrize("enforced", [False, None])
def test_cuda_repeats_that_agree_without_enforcement_are_incomplete(
    repeats: Path, enforced: bool | None
) -> None:
    """Section 7.1 rests the reproduction on deterministic algorithms being enforced."""
    _report(repeats, enforced)
    outcome = cd.compare_runs(repeats, fx.ITERATIONS, cuda=True)
    assert outcome["differences"] == [] and outcome["status"] == cd.STATUS_INCOMPLETE
    assert outcome["coverage_gaps"] == [
        f"{name}: ran on CUDA without deterministic algorithms enforced" for name in cd.REPEAT_NAMES
    ]
    # A CPU check enforces nothing (utils.seeding leaves CPU runs untouched) and needs nothing.
    assert cd.compare_runs(repeats, fx.ITERATIONS, cuda=False)["status"] == cd.STATUS_REPRODUCED
    _report(repeats, True)
    assert cd.compare_runs(repeats, fx.ITERATIONS, cuda=True)["status"] == cd.STATUS_REPRODUCED


@pytest.mark.parametrize(
    ("devices", "expected"),
    [(("cpu", "cpu"), False), (("cuda:0", "cuda:1"), True), (("cpu", "cuda:1"), True)],
)
def test_the_check_tells_the_comparison_when_a_phase_ran_on_cuda(
    devices: tuple[str, str], expected: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(cd, "_run_worker_process", lambda command, log: 0)
    monkeypatch.setattr(
        cd, "compare_runs", lambda work, iterations, *, cuda: seen.append(cuda) or {}
    )
    args = cd._parse_args(["--train-device", devices[0], "--arena-device", devices[1]])
    cd.check(args, tmp_path)
    assert seen == [expected]


def test_repeats_that_report_different_environments_differ(repeats: Path) -> None:
    for name, version in zip(cd.REPEAT_NAMES, ("2.11.0+cu128", "2.11.1+cu128"), strict=True):
        (repeats / name / cd.WORKER_REPORT_FILE).write_text(
            json.dumps({"torch": version}), encoding="utf-8"
        )
    outcome = cd.compare_runs(repeats, fx.ITERATIONS)
    assert outcome["status"] == cd.STATUS_DIFFERS
    assert "different environments" in outcome["differences"][0]
    assert outcome["environment"] == {"torch": "2.11.0+cu128"}


# ---------------------------------------------------------------------------
# main(): refusals, pricing, failures, the work directory
# ---------------------------------------------------------------------------


@pytest.fixture
def no_workers(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Fail the test if a worker process is started; returns the (empty) launch list."""
    launched: list[list[str]] = []

    def refuse(command: list[str], log: Path) -> int:
        launched.append(list(command))
        raise AssertionError(f"a worker was started: {command}")

    monkeypatch.setattr(cd, "_run_worker_process", refuse)
    return launched


@pytest.mark.parametrize("flag", ["--train-device", "--arena-device"])
def test_an_unavailable_gpu_is_refused_before_anything_runs(
    flag: str, tmp_path: Path, no_workers: list[list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / "work"
    code = cd.main([*TINY, flag, ABSENT_GPU, "--work-dir", str(work), "--out", str(tmp_path / "o")])
    assert code == game_budget.EXIT_CODE_REFUSED
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err and ABSENT_GPU in captured.err and flag in captured.err
    assert no_workers == []
    assert not work.exists() and not (tmp_path / "o").exists(), "nothing was created"


@pytest.mark.parametrize("device", ["cuda:x", "tpu"])
def test_a_malformed_or_unknown_device_is_refused(
    device: str, tmp_path: Path, no_workers: list[list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    """``cuda:x`` names no ordinal, so a check keyed on the ordinal alone would run it on CPU."""
    code = cd.main([*TINY, "--train-device", device, "--work-dir", str(tmp_path / "work")])
    assert code == game_budget.EXIT_CODE_REFUSED
    assert device in capsys.readouterr().err
    assert no_workers == []


def test_the_parser_leaves_the_arena_device_unset_and_resolution_follows_training() -> None:
    """A run card reads the parsed namespace, and must see "not given" as ``None``:
    on a card that pins a torch build an implicit device is refused."""
    parsed = cd._parse_args(["--train-device", "cpu"])
    assert parsed.train_device == "cpu" and parsed.arena_device is None
    resolved = cd.resolved_arguments(parsed)
    assert (resolved.train_device, resolved.arena_device) == ("cpu", "cpu")
    assert parsed.arena_device is None, "the parsed namespace is left as the parser made it"
    assert cd._parse_args(["--arena-device", "cuda:1"]).arena_device == "cuda:1"
    assert cd.resolve_run_device("auto", "--train-device") in {"cpu", "cuda", "mps"}
    with pytest.raises(cd.DeviceUnavailableError, match="--arena-device"):
        cd.resolved_arguments(cd._parse_args(["--arena-device", ABSENT_GPU]))


def test_the_check_is_priced_at_twice_one_repeat() -> None:
    assert cd.RUNNER in game_budget.priced_runners()
    defaults = cd._parse_args([])
    assert cd.budget_inputs(defaults) == {
        "repeats": 2,
        "iterations": 2,
        "games": 2,
        "arena_games": 4,
    }
    assert game_budget.worst_case_games(cd.RUNNER, cd.budget_inputs(defaults)) == 2 * (2 * 2 + 4)
    worker = cd._parse_args(["--worker", "somewhere"])
    assert game_budget.worst_case_games(cd.RUNNER, cd.budget_inputs(worker)) == 2 * 2 + 4
    big = cd._parse_args(["--iterations", "20", "--games", "24", "--arena-games", "200"])
    assert game_budget.worst_case_games(cd.RUNNER, cd.budget_inputs(big)) == 2 * (20 * 24 + 200)
    with pytest.raises(KeyError):  # the model assumes no repeat count
        game_budget.worst_case_games(cd.RUNNER, {"iterations": 1, "games": 1, "arena_games": 2})


def test_an_over_cap_check_is_refused_before_anything_runs(
    tmp_path: Path, no_workers: list[list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / "work"
    # One repeat is 502 games, under the cap; the check is two of them.
    argv = ["--iterations", "249", "--games", "2", "--arena-games", "4", "--work-dir", str(work)]
    assert game_budget.worst_case_games(cd.RUNNER, cd.budget_inputs(cd._parse_args(argv))) == 1004
    with pytest.raises(SystemExit) as excinfo:
        cd.main(argv)
    assert excinfo.value.code == game_budget.EXIT_CODE_REFUSED
    assert "1,000" in capsys.readouterr().err
    assert no_workers == [] and not work.exists()


def test_a_worker_started_by_hand_is_priced_and_resolves_its_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--worker`` is how the check starts a repeat; it must not be a way around either refusal."""
    ran: list[argparse.Namespace] = []
    monkeypatch.setattr(cd, "run_worker", lambda args: ran.append(args) or 0)
    with pytest.raises(SystemExit) as excinfo:
        cd.main(["--worker", str(tmp_path / "w"), "--iterations", "1001", "--games", "1"])
    assert excinfo.value.code == game_budget.EXIT_CODE_REFUSED
    code = cd.main(["--worker", str(tmp_path / "w"), *TINY, "--train-device", ABSENT_GPU])
    assert code == game_budget.EXIT_CODE_REFUSED
    assert ran == [], "a refused worker trained"
    assert cd.main(["--worker", str(tmp_path / "w"), *TINY]) == 0
    assert len(ran) == 1 and ran[0].worker == tmp_path / "w"
    capsys.readouterr()


def _failing_worker(lines: int, status: int, launched: list[list[str]]) -> Any:
    def launch(command: list[str], log: Path) -> int:
        launched.append(list(command))
        log.write_text("".join(f"line {i}\n" for i in range(lines)), encoding="utf-8")
        return status

    return launch


def test_a_failed_worker_fails_the_check_with_the_end_of_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(cd, "_run_worker_process", _failing_worker(100, 3, launched))
    work, out = tmp_path / "work", tmp_path / "out.json"
    code = cd.main([*TINY, "--work-dir", str(work), "--out", str(out)])
    assert code == cd.EXIT_NOT_REPRODUCED == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == json.loads(capsys.readouterr().out)
    assert payload["status"] == cd.STATUS_WORKER_FAILED
    assert payload["failed_repeat"] == cd.REPEAT_NAMES[0] and payload["worker_exit_code"] == 3
    assert payload["worker_log_tail"] == [
        f"line {i}" for i in range(100 - cd.WORKER_LOG_TAIL_LINES, 100)
    ]
    assert payload["differences"] == []
    assert payload["components_compared"] == {"checkpoints": 0, "arenas": 0, "iterations": 0}
    assert len(launched) == 1, "the second repeat is not started after the first fails"
    assert payload["work_dir"] == str(work) and work.is_dir(), "kept for inspection"
    assert payload["torch"] == str(torch.__version__), "the build is reported even so"


def test_an_approval_reaches_the_worker_and_the_arena(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The worker and the benchmark each price their own argv, so each needs the approval."""
    launched: list[list[str]] = []
    monkeypatch.setattr(cd, "_run_worker_process", _failing_worker(1, 1, launched))
    over_cap = ["--iterations", "300", "--games", "2", "--arena-games", "1200"]
    argv = [*over_cap, "--max_games", "5000", "--approval_ref", "#142"]
    assert cd.main([*argv, "--work-dir", str(tmp_path / "work")]) == cd.EXIT_NOT_REPRODUCED
    (command,) = launched
    assert command[-4:] == ["--max_games", "5000", "--approval_ref", "#142"]
    worker_args = cd._parse_args(command[3:])
    assert cd.arena_argv(worker_args, tmp_path)[-4:] == command[-4:]
    plain = cd.resolved_arguments(cd._parse_args(["--worker", "w"]))
    assert "--approval_ref" not in cd.worker_command(plain, tmp_path)
    assert "--approval_ref" not in cd.arena_argv(plain, tmp_path)
    capsys.readouterr()


def test_a_work_dir_that_holds_a_repeat_is_refused(
    tmp_path: Path, no_workers: list[list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / "work"
    (work / cd.REPEAT_NAMES[1]).mkdir(parents=True)
    assert cd.main([*TINY, "--work-dir", str(work)]) == game_budget.EXIT_CODE_REFUSED
    assert cd.REPEAT_NAMES[1] in capsys.readouterr().err
    assert no_workers == []


@pytest.mark.parametrize(
    ("status", "keep", "kept"),
    [
        (cd.STATUS_REPRODUCED, False, False),
        (cd.STATUS_REPRODUCED, True, True),
        (cd.STATUS_DIFFERS, False, True),
        (cd.STATUS_INCOMPLETE, False, True),
    ],
)
def test_the_temporary_work_dir_is_removed_only_after_a_reproduction(
    status: str,
    keep: bool,
    kept: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    seen: list[Path] = []

    def canned(args: argparse.Namespace, work_dir: Path) -> dict[str, Any]:
        seen.append(work_dir)
        return {
            "status": status,
            "components_compared": {"checkpoints": 2, "arenas": 1, "iterations": 1},
            "differences": [],
            "environment": None,
        }

    monkeypatch.setattr(cd, "check", canned)
    code = cd.main([*TINY, *(["--keep"] if keep else [])])
    assert (code == cd.EXIT_REPRODUCED) == (status == cd.STATUS_REPRODUCED)
    payload = json.loads(capsys.readouterr().out)
    (work,) = seen
    assert work.parent == tmp_path and work.name.startswith(cd.HARNESS)
    assert work.exists() == kept
    assert payload["work_dir"] == (str(work) if kept else None)
    assert payload["status"] == status


@pytest.mark.parametrize("argv", [["--arena-games", "3"], ["--iterations", "0"], ["--games", "0"]])
def test_sizes_that_cannot_run_are_rejected_by_the_parser(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cd._parse_args(argv)
    assert excinfo.value.code == 2
    assert argv[0] in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The worker: a new process, the operator's environment, the registered commands
# ---------------------------------------------------------------------------


def test_workers_inherit_the_environment_and_get_no_hash_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``CUDA_VISIBLE_DEVICES`` and friends must reach the child; ``PYTHONHASHSEED`` is not added."""
    passed = {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CIV6AZ_INFERENCE_THREADS": "2",
    }
    for name, value in passed.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    probe = (
        "import json, os; "
        f"print(json.dumps({{'env': {{k: os.environ.get(k) for k in {[*passed, 'PYTHONHASHSEED']!r}}}, "
        "'cwd': os.getcwd()}))"
    )
    log = tmp_path / "probe.log"
    assert cd._run_worker_process([sys.executable, "-c", probe], log) == 0
    seen = json.loads(log.read_text(encoding="utf-8"))
    assert seen["env"] == {**passed, "PYTHONHASHSEED": None}
    assert Path(seen["cwd"]) == REPO_ROOT, "python -m scripts.check_determinism resolves from here"
    assert cd._run_worker_process([sys.executable, "-c", "raise SystemExit(7)"], log) == 7


def _registered(name: str, skip: int) -> list[str]:
    section = ap.registered_protocol((REPO_ROOT / "docs/EVALUATION.md").read_text(encoding="utf-8"))
    return dict(command_block(section, "base"))[name][skip:]


def _pairs(tokens: list[str]) -> list[tuple[str, str | None]]:
    """``[(flag, value)]`` in order; a ``--flag=value`` token is its own pair."""
    pairs: list[tuple[str, str | None]] = []
    index = 0
    while index < len(tokens):
        if "=" in tokens[index]:
            pairs.append((tokens[index], None))
            index += 1
        else:
            pairs.append((tokens[index], tokens[index + 1]))
            index += 2
    return pairs


def test_the_pipeline_is_the_registered_commands_at_other_sizes() -> None:
    """Section 7.1's training command and primary arm, flag for flag.

    With the default sizes only the seed, the game counts, the devices, the
    checkpoint cadence and the paths differ from the registered text.
    """
    args = cd.resolved_arguments(cd._parse_args([]))
    train = _pairs(cd.training_argv(args, Path("D") / "ckpt"))
    registered_train = _pairs(_registered("train", skip=3))  # python -m alpha_zero.training_civ6
    assert [flag for flag, _ in train] == [flag for flag, _ in registered_train]
    resized = {"--seed", "--num_iterations", "--num_games", "--device", "--save_interval"}
    resized.add("--checkpoint_dir")
    for (flag, value), (_, registered) in zip(train, registered_train, strict=True):
        if flag not in resized:
            assert value == registered, flag
    assert dict(train)["--save_interval"] == str(cd.SAVE_INTERVAL)

    arena = _pairs(cd.arena_argv(args, Path("D")))
    registered_arena = _pairs(_registered("primary", skip=2))  # python scripts/benchmark_suite.py
    assert [flag for flag, _ in arena] == [flag for flag, _ in registered_arena]
    moved = {"--seed", "--device", "--checkpoint", "--opponent-checkpoint", "--n-games", "--out"}
    for (flag, value), (_, registered) in zip(arena, registered_arena, strict=True):
        if flag not in moved:
            assert value == registered, flag
    # The candidate is the last checkpoint this check trains; the opponent its initialisation.
    assert Path(dict(arena)["--checkpoint"]).name == f"checkpoint_{args.iterations}.pt"
    assert Path(dict(arena)["--opponent-checkpoint"]).name == "checkpoint_0.pt"
    assert Path(dict(arena)["--out"]) == Path("D") / rp.ARENA_DIR_NAME / cd.ARENA_ARM


def test_the_default_seed_is_not_a_registered_seed() -> None:
    """A check at a registered seed could be read as an A1-family run (section 7.1)."""
    registered = {1001, 1002, 1003, 20261111, 20261112}
    assert set(ap.T0B_SEEDS) <= registered
    assert cd.DEFAULT_SEED not in registered
    assert cd._parse_args([]).seed == cd.DEFAULT_SEED


def test_the_environment_report_without_a_gpu_probe_asks_nothing_of_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> str:
        raise AssertionError("CUDA was probed")

    monkeypatch.setattr(torch.cuda, "get_device_name", forbidden)
    report = cd.environment_report("cuda:0", "cuda:1", probe_gpus=False)
    assert report["gpu_names"] == {"train": None, "arena": None}
    assert report["torch"] == str(torch.__version__) and report["cudnn"] is None
    assert cd.environment_report("cpu", "cpu")["gpu_names"] == {"train": None, "arena": None}
