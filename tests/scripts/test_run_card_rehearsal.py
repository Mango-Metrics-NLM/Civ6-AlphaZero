"""A run card is rehearsed before it claims its single-use run directory.

``run_card`` used to claim ``runs/<card_id>`` before it built the pinned
environment, ran the preflight and checked the devices, so a refused device or a
failed preflight left a FAILED ``run.json`` there and the card needed a new
``card_id``. It now rehearses first, exactly as ``--probe`` does, in a throwaway
``runs/probe-<card_id>-<stamp>`` directory, and claims the card's directory only
when the rehearsal ends ``PROBED``.

``run.json`` also records ``launched`` (the invocations that were started, written
before each child runs) and ``rehearsal`` (the rehearsal directory that preceded
the run). ``launched`` is what lets the release verifier tell an attempt that was
interrupted in its first invocation from one that never started anything.

Negative controls (each planted in ``alpha_zero/local/run_card.py``, observed
failing, then restored):

1. ``run_card`` skips the rehearsal and goes straight to the real pass ->
   ``test_a_card_refused_at_its_rehearsal_keeps_its_run_directory``,
   ``test_a_used_run_directory_is_refused_before_any_rehearsal`` and
   ``test_the_real_run_records_its_rehearsal`` fail.
2. The early ``_refuse_used_run_dir`` call is removed ->
   ``test_a_used_run_directory_is_refused_before_any_rehearsal`` fails (a second
   rehearsal directory appears before the refusal).
3. The real pass is given ``rehearsal=None`` ->
   ``test_the_real_run_records_its_rehearsal`` fails.
4. ``launched`` is appended after the child returns ->
   ``test_an_invocation_is_recorded_as_launched_before_it_finishes`` fails.
5. The CLI's "not claimed" line is removed ->
   ``test_cli_says_when_the_directory_was_not_claimed`` fails.

6. ``torch_wheel`` dropped from the record ``run_card`` writes ->
   ``test_run_json_records_the_torch_wheel`` fails.
7. ``driver_report`` not passed to the device check ->
   ``test_a_pinned_gpu_card_is_refused_when_its_driver_cannot_be_read`` fails.

The fixtures and fakes are the ones ``test_run_card.py`` defines; that file is
near the 1,000-line ratchet, so these tests live here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.local import pinned_env as pe
from alpha_zero.local import run_card as rc
from tests.scripts import test_run_card as base

# pytest finds fixtures by module attribute, so these make the shared ones local.
restore_environ = base.restore_environ
no_inherited_cuda_environment = base.no_inherited_cuda_environment
pinned = base.pinned

CARD_ID = "t1"


def _runs(tmp_path: Path) -> Path:
    return tmp_path / "root" / rc.RUNS_DIR


def _rehearsals(tmp_path: Path) -> list[Path]:
    return sorted(_runs(tmp_path).glob(f"{rc.PROBE_PREFIX}*"))


def _record(run_dir: Path) -> dict[str, Any]:
    data = json.loads((run_dir / rc.RUN_JSON).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_a_card_refused_at_its_rehearsal_keeps_its_run_directory(
    tmp_path: Path, pinned: base.FakeRunner
) -> None:
    """The bare test venv has no torch, so ``cuda:1`` is refused -- in the rehearsal."""
    refused = base._card(tmp_path, [base._train("smoke", "--dry_run", "--device", "cuda:1")])
    result = base._run(refused, tmp_path, pinned)

    assert result.status == rc.STATUS_FAILED
    assert result.claimed is False
    assert result.error is not None and result.error.startswith("devices: ")
    assert result.run_dir.name.startswith(f"{rc.PROBE_PREFIX}{CARD_ID}-")
    assert not (_runs(tmp_path) / CARD_ID).exists(), "a refused card must not claim its directory"
    record = _record(result.run_dir)
    assert record["probe"] is True and record["launched"] == [] and record["invocations"] == []

    # The cause is fixed (a device the environment has): the SAME card_id now runs.
    fixed = base._card(tmp_path, [base._train("smoke", "--dry_run", "--device", "cpu")])
    again = base._run(fixed, tmp_path, pinned)
    assert again.status == rc.STATUS_COMPLETED and again.claimed is True
    assert again.run_dir == _runs(tmp_path) / CARD_ID
    assert _record(again.run_dir)["status"] == rc.STATUS_COMPLETED


def test_a_used_run_directory_is_refused_before_any_rehearsal(
    tmp_path: Path, pinned: base.FakeRunner
) -> None:
    card = base._card(tmp_path, [base._train("smoke", "--dry_run")])
    assert base._run(card, tmp_path, pinned).status == rc.STATUS_COMPLETED
    before = _rehearsals(tmp_path)
    assert len(before) == 1, "the completed run was rehearsed once"

    with pytest.raises(rc.CardResumedError, match="single-use"):
        base._run(card, tmp_path, pinned)

    assert _rehearsals(tmp_path) == before, "a used card must be refused without a rehearsal"


def test_the_real_run_records_its_rehearsal(tmp_path: Path, pinned: base.FakeRunner) -> None:
    card = base._card(tmp_path, [base._train("smoke", "--dry_run")])
    result = base._run(card, tmp_path, pinned)

    assert result.status == rc.STATUS_COMPLETED
    record = _record(result.run_dir)
    assert record["probe"] is False
    (rehearsal_dir,) = _rehearsals(tmp_path)
    assert record["rehearsal"] == rehearsal_dir.name
    rehearsal = _record(rehearsal_dir)
    assert rehearsal["status"] == rc.STATUS_PROBED and rehearsal["probe"] is True
    assert rehearsal["rehearsal"] is None and rehearsal["launched"] == []
    assert record["launched"] == ["smoke"]
    assert [inv["name"] for inv in record["invocations"]] == ["smoke"]


def test_the_environment_is_built_once_for_the_rehearsal_and_the_run(
    tmp_path: Path, pinned: base.FakeRunner
) -> None:
    """The venv is cached by commit and torch pin: the second pass must reuse it."""
    card = base._card(tmp_path, [base._train("smoke", "--dry_run")])
    assert base._run(card, tmp_path, pinned).status == rc.STATUS_COMPLETED
    builds = [call for call in pinned.calls if call[1:3] == ["-m", "venv"]]
    assert len(builds) == 1, builds


def test_an_invocation_is_recorded_as_launched_before_it_finishes(
    tmp_path: Path, pinned: base.FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An attempt interrupted in its first child launched something, and says so."""
    seen: dict[str, Any] = {}

    def interrupted(cmd: list[str], *, cwd: Path, env: Any, log: Path) -> int:
        seen.update(_record(log.parent.parent))  # what run.json says while the child runs
        raise KeyboardInterrupt

    monkeypatch.setattr(rc, "_run_invocation", interrupted)
    card = base._card(
        tmp_path, [base._train("smoke", "--dry_run"), base._train("next", "--dry_run")]
    )

    with pytest.raises(KeyboardInterrupt):
        base._run(card, tmp_path, pinned)

    assert seen["launched"] == ["smoke"], "the launch must be on disk before the child runs"
    assert seen["invocations"] == []
    record = _record(_runs(tmp_path) / CARD_ID)
    assert record["status"] == rc.STATUS_VOID
    assert record["launched"] == ["smoke"] and record["invocations"] == []


@pytest.mark.parametrize(
    ("claimed", "status", "probe_flag", "hinted"),
    [
        (False, rc.STATUS_FAILED, False, True),  # the automatic rehearsal failed
        (False, rc.STATUS_FAILED, True, False),  # an explicit --probe: nothing to claim
        (True, rc.STATUS_FAILED, False, False),  # the card ran and failed: it is used up
        (True, rc.STATUS_COMPLETED, False, False),
    ],
)
def test_cli_says_when_the_directory_was_not_claimed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    claimed: bool,
    status: str,
    probe_flag: bool,
    hinted: bool,
) -> None:
    result = rc.CardResult(
        CARD_ID, status, tmp_path / "run", 8, error="devices: no such GPU", claimed=claimed
    )
    monkeypatch.setattr(rc, "run_card", lambda *args, **kwargs: result)
    argv = [str(tmp_path / "card.yaml"), "--data_root", str(tmp_path / "root")]

    code = rc.main([*argv, "--probe"] if probe_flag else argv)

    assert code == (0 if status == rc.STATUS_COMPLETED else 1)
    assert (rc.NOT_CLAIMED_HINT in capsys.readouterr().err) is hinted


class _ReportingRunner(base.FakeRunner):
    """``base.FakeRunner`` whose torch install writes the report pip would."""

    def __call__(
        self, cmd: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None
    ) -> str:
        out = super().__call__(cmd, cwd=cwd, env=env)
        argv = [str(c) for c in cmd]
        if "--report" in argv:
            entry = {
                "metadata": {"name": "torch"},
                "download_info": {"url": _WHEEL_URL, "archive_info": {"hash": _WHEEL_HASH}},
            }
            report = Path(argv[argv.index("--report") + 1])
            report.write_text(json.dumps({"install": [entry]}), encoding="utf-8")
        return out


_WHEEL_URL = "https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp311-cp311-win_amd64.whl"
_WHEEL_SHA = "ef" * 32
_WHEEL_HASH = f"sha256={_WHEEL_SHA}"


def _fake_gpu_preflight(monkeypatch: pytest.MonkeyPatch, **report: Any) -> None:
    """The bare test venv has no torch: give its preflight two GPUs, plus ``report``."""
    real = rc._preflight
    extra = {"torch": base._TWO_GPUS, **report}
    monkeypatch.setattr(rc, "_preflight", lambda env, child: {**real(env, child), **extra})


def _pinned_gpu_card(tmp_path: Path) -> Path:
    return base._card(
        tmp_path,
        [base._train("smoke", "--dry_run", "--device", "cuda:0")],
        torch_spec="torch==2.11.0",
        torch_index_url=base._CU128,
    )


def test_run_json_records_the_torch_wheel(
    tmp_path: Path, pinned: base.FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wheel's hash is a record of its own, never part of the compared torch pin."""
    _fake_gpu_preflight(monkeypatch, nvidia_smi=base._SMI)
    result = base._run(_pinned_gpu_card(tmp_path), tmp_path, _ReportingRunner())

    assert result.status == rc.STATUS_COMPLETED, result.error
    record = _record(result.run_dir)
    assert record["torch_wheel"] == {"url": _WHEEL_URL, "sha256": _WHEEL_SHA}
    assert record["device_plan"]["torch_pin"] == {
        "spec": "torch==2.11.0",
        "index_url": base._CU128,
    }, "the pin is compared with the registration; the wheel hash must not be inside it"
    assert pe.TorchWheel.from_mapping(record["torch_wheel"]) is not None


def test_a_card_without_a_torch_pin_records_no_torch_wheel(
    tmp_path: Path, pinned: base.FakeRunner
) -> None:
    card = base._card(tmp_path, [base._train("smoke", "--dry_run")])
    result = base._run(card, tmp_path, pinned)
    assert result.status == rc.STATUS_COMPLETED
    assert _record(result.run_dir)["torch_wheel"] is None


@pytest.mark.parametrize(
    "nvidia_smi",
    [{"error": "FileNotFoundError"}, [], ["0, GPU A, 00000000:01:00.0"]],
    ids=["tool missing", "no lines", "no version field"],
)
def test_a_pinned_gpu_card_is_refused_when_its_driver_cannot_be_read(
    tmp_path: Path, pinned: base.FakeRunner, monkeypatch: pytest.MonkeyPatch, nvidia_smi: Any
) -> None:
    """A re-run is compared under the same driver, so a run must record the one it had."""
    _fake_gpu_preflight(monkeypatch, nvidia_smi=nvidia_smi)
    result = base._run(_pinned_gpu_card(tmp_path), tmp_path, pinned)

    assert result.status == rc.STATUS_FAILED and result.claimed is False
    assert result.error is not None and "driver version could not be read" in result.error
    assert not (_runs(tmp_path) / CARD_ID).exists()


def test_a_gpu_card_without_a_torch_pin_is_not_held_to_a_driver_record(
    tmp_path: Path, pinned: base.FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_gpu_preflight(monkeypatch, nvidia_smi={"error": "FileNotFoundError"})
    card = base._card(tmp_path, [base._train("smoke", "--dry_run", "--device", "cuda:0")])
    assert base._run(card, tmp_path, pinned).status == rc.STATUS_COMPLETED
