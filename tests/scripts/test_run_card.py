"""Run-card driver (LF-E6, LF-G11): ``alpha_zero.local.run_card`` / ``scripts/run_card.py``.

What these tests pin down:

* pricing happens before anything runs, sums every invocation, and fails closed
  on an unknown runner or an unparseable argv;
* a tiny card runs a REAL child interpreter (a ``venv --without-pip`` built by the
  fake runner seam -- no wheel, no pip install) from the pinned source, and the
  pinned code's own ``assert_running_pinned`` decides whether it may proceed;
* an interrupted or resumed card is VOID, a failing child is FAILED;
* ``run.json`` records environment variable NAMES, never values;
* only a data root the driver creates is ACL-hardened.

Negative controls (recorded when written, then reverted):
* removing the ``_preflight`` call from ``run_card`` made
  ``test_preflight_version_mismatch_fails_card`` fail (the card COMPLETED);
* summing with ``max`` instead of ``sum`` made
  ``test_two_invocations_summing_to_1040_refused`` fail.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

import alpha_zero
from alpha_zero.local import pinned_env as pe
from alpha_zero.local import run_card as rc
from alpha_zero.utils.game_budget import GameBudgetExceededError

REPO = Path(alpha_zero.__file__).resolve().parents[1]
FULL_SHA = "0123456789abcdef0123456789abcdef01234567"
SENTINEL_NAME = "RUNCARD_TEST_SENTINEL"
SENTINEL_VALUE = "s3ntinel-value-never-recorded-7f3a"


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_environ() -> Iterator[None]:
    """``enter_local_mode`` mutates os.environ; restore it exactly."""
    snapshot = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(snapshot)


class FakeRunner:
    """The ``pinned_env`` runner seam: a real bare venv, faked wheel/pip/git."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self, cmd: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None
    ) -> str:
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        if cmd[0] == "git":
            return FULL_SHA + "\n"
        if cmd[1:3] == ["-m", "venv"]:
            subprocess.run(
                [sys.executable, "-m", "venv", "--without-pip", cmd[-1]],
                check=True,
                capture_output=True,
            )
            return ""
        if "wheel" in cmd:
            out = Path(cmd[cmd.index("-w") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "alpha_zero-2.0.0-py3-none-any.whl").write_bytes(b"wheel-bytes")
            return ""
        if "freeze" in cmd:
            return "alpha_zero==2.0.0\ntorch==9.9.9\n"
        return ""  # pip install of the wheel


def _fake_src(root: Path, *, init_version: str = "2.0.0") -> Path:
    """A minimal pinned tree: the REAL pinned_env/data_root modules plus a child module."""
    src = root / "fake-src"
    pkg = src / "alpha_zero"
    (pkg / "local").mkdir(parents=True)
    (pkg / "utils").mkdir()
    (pkg / "__init__.py").write_text(f'__version__ = "{init_version}"\n')
    (pkg / "local" / "__init__.py").write_text("")
    (pkg / "utils" / "__init__.py").write_text("")
    shutil.copy(REPO / "alpha_zero" / "local" / "pinned_env.py", pkg / "local" / "pinned_env.py")
    shutil.copy(REPO / "alpha_zero" / "utils" / "data_root.py", pkg / "utils" / "data_root.py")
    (pkg / "fake_child.py").write_text(
        "import os, sys\n"
        "import alpha_zero\n"
        "print('child file', alpha_zero.__file__)\n"
        "print('child prefix', sys.prefix)\n"
        "print('child argv', sys.argv[1:])\n"
        "sys.exit(int(os.environ.get('FAKE_CHILD_EXIT', '0')))\n"
    )
    (src / "pyproject.toml").write_text('[project]\nname = "alpha_zero"\nversion = "2.0.0"\n')
    return src


@pytest.fixture
def pinned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeRunner:
    """Fake git archive + fake wheel build; children of training_civ6 run fake_child."""
    fake = _fake_src(tmp_path)
    monkeypatch.setattr(
        pe, "_extract_archive", lambda sha, repo_root, dest, runner: shutil.copytree(fake, dest)
    )
    spec = rc.RUNNERS["training_civ6"]
    monkeypatch.setitem(rc.RUNNERS, "training_civ6", replace(spec, module="alpha_zero.fake_child"))
    monkeypatch.setattr(rc, "harden_data_root", lambda root, **_: _layout(root))
    return FakeRunner()


def _layout(root: Path) -> None:
    for sub in ("envs", "runs", "logs", "tmp", "storage"):
        (Path(root) / sub).mkdir(parents=True, exist_ok=True)


def _card(tmp_path: Path, invocations: list[dict[str, Any]], **extra: Any) -> Path:
    data: dict[str, Any] = {
        "card_id": "t1",
        "sha": "0123456",
        "max_games": 1000,
        "invocations": invocations,
    }
    data.update(extra)
    path = tmp_path / "card.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _train(name: str, *argv: str) -> dict[str, Any]:
    return {"name": name, "runner": "training_civ6", "argv": list(argv)}


def _run(card: Path, tmp_path: Path, runner: FakeRunner) -> rc.CardResult:
    return rc.run_card(card, data_root=tmp_path / "root", pinned_runner=runner, repo_root=REPO)


def _run_json(tmp_path: Path, card_id: str = "t1") -> dict[str, Any]:
    path = tmp_path / "root" / "runs" / card_id / "run.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pricing (no side effects)
# ---------------------------------------------------------------------------


def test_two_invocations_summing_to_1040_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """520 + 520 games: each fits the cap alone, the card does not; nothing runs."""
    spawned: list[Any] = []
    monkeypatch.setattr(rc.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: spawned.append(a))
    leg = ("--num_games", "52", "--num_iterations", "10", "--save_interval", "1000")
    card = _card(tmp_path, [_train("a", *leg), _train("b", *leg)])

    with pytest.raises(GameBudgetExceededError, match="1,040") as refused:
        rc.price_card(rc.load_card(card))
    assert "a=520, b=520" in str(refused.value)

    with pytest.raises(GameBudgetExceededError, match="1,040"):
        rc.run_card(card, data_root=tmp_path / "root", repo_root=REPO)
    assert spawned == []
    assert not (tmp_path / "root").exists(), "a refused card must create nothing"


def test_approval_ref_allows_card_over_1000(tmp_path: Path) -> None:
    leg = ("--num_games", "52", "--num_iterations", "10", "--save_interval", "1000")
    card = _card(tmp_path, [_train("a", *leg), _train("b", *leg)], approval_ref="#142")
    assert rc.price_card(rc.load_card(card)).total == 1040


def test_max_games_above_cap_needs_approval(tmp_path: Path) -> None:
    card = _card(tmp_path, [_train("a", "--dry_run")], max_games=5000)
    with pytest.raises(GameBudgetExceededError, match="only lower"):
        rc.price_card(rc.load_card(card))


def test_single_invocation_child_would_refuse_is_refused_up_front(tmp_path: Path) -> None:
    """The card is approved, but the child's own cap (no --approval_ref in argv) is not."""
    card = _card(tmp_path, [_train("big", "--num_games", "200")], approval_ref="#142")
    with pytest.raises(GameBudgetExceededError, match="training_civ6"):
        rc.price_card(rc.load_card(card))


def test_unknown_runner_refused(tmp_path: Path) -> None:
    card = _card(tmp_path, [{"name": "x", "runner": "mystery", "argv": []}])
    with pytest.raises(rc.RunCardError, match="mystery"):
        rc.price_card(rc.load_card(card))


def test_priced_but_unsupported_runner_refused(tmp_path: Path) -> None:
    """run_sweep has a cost model but no argv resolver here: fail closed."""
    card = _card(tmp_path, [{"name": "x", "runner": "run_sweep", "argv": []}])
    with pytest.raises(rc.RunCardError, match="run_sweep"):
        rc.price_card(rc.load_card(card))


def test_bad_argv_refused(tmp_path: Path) -> None:
    card = _card(tmp_path, [_train("x", "--no_such_flag")])
    with pytest.raises(rc.RunCardError, match="argv"):
        rc.price_card(rc.load_card(card))


def test_dry_run_is_priced_at_8(tmp_path: Path) -> None:
    card = _card(tmp_path, [_train("smoke", "--dry_run")])
    assert rc.price_card(rc.load_card(card)).total == 8


def test_benchmark_priced_by_its_own_parser(tmp_path: Path) -> None:
    inv = {"name": "b", "runner": "benchmark", "argv": ["--n-games", "10"]}
    assert rc.price_card(rc.load_card(_card(tmp_path, [inv]))).total == 30


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"card_id": "../escape"}, "run id"),
        ({"sha": "HEAD"}, "sha"),
        ({"invocations": []}, "invocations"),
        ({"surprise": 1}, "surprise"),
        ({"invocations": [_train("a"), _train("a")]}, "duplicate"),
    ],
)
def test_malformed_card_refused(tmp_path: Path, override: dict[str, Any], match: str) -> None:
    base = {"card_id": "t1", "sha": "0123456", "max_games": 10, "invocations": [_train("a")]}
    base.update(override)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises((rc.RunCardError, ValueError), match=match):
        rc.load_card(path)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_tiny_card_runs_from_pinned_source(
    tmp_path: Path, pinned: FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SENTINEL_NAME, SENTINEL_VALUE)
    card = _card(tmp_path, [_train("smoke", "--dry_run")])

    result = _run(card, tmp_path, pinned)

    assert result.status == rc.STATUS_COMPLETED, result
    run_dir = tmp_path / "root" / "runs" / "t1"
    data = _run_json(tmp_path)
    assert data["status"] == "COMPLETED"
    assert data["sha"] == FULL_SHA
    assert data["wheel_sha256"]
    assert data["pip_freeze"] == ["alpha_zero==2.0.0", "torch==9.9.9"]
    assert data["total_priced_games"] == 8
    [inv] = data["invocations"]
    assert inv["priced_games"] == 8 and inv["exit_code"] == 0 and inv["duration_s"] >= 0

    # The preflight ran the PINNED code's assert_running_pinned in a real child.
    pre = data["preflight"]
    assert pre["exit_code"] == 0, pre
    venv = (tmp_path / "root" / "envs" / FULL_SHA[:12]).resolve()
    assert Path(pre["prefix"]).resolve() == venv
    assert Path(pre["alpha_zero_file"]).resolve().is_relative_to((run_dir / "src").resolve())

    # The invocation itself imported alpha_zero from the pinned src.
    log = (run_dir / "logs" / "smoke.log").read_text(encoding="utf-8")
    assert str((run_dir / "src" / "alpha_zero").resolve()) in log
    assert "child argv ['--dry_run', '--max_games', '8']" in log
    assert inv["launched_argv"] == ["--dry_run", "--max_games", "8"]

    # torch facts are recorded, never required (the bare test venv has no torch).
    assert data["device_plan"]["torch"] == {"error": "ModuleNotFoundError"}

    # Names, never values.
    assert SENTINEL_NAME in data["env_var_names"]
    assert "PYTHONNOUSERSITE" in data["env_var_names"]
    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".log"}:
            assert SENTINEL_VALUE not in path.read_text(encoding="utf-8", errors="replace"), path


def test_child_env_is_pinned_and_scoped(
    tmp_path: Path, pinned: FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "leak"))
    card = _card(tmp_path, [_train("smoke", "--dry_run")])
    seen: dict[str, str] = {}
    real = rc._run_invocation

    def spy(cmd: list[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> int:
        seen.update(env)
        return real(cmd, cwd=cwd, env=env, log=log)

    monkeypatch.setattr(rc, "_run_invocation", spy)
    assert _run(card, tmp_path, pinned).status == rc.STATUS_COMPLETED
    root = tmp_path / "root"
    assert seen["PYTHONNOUSERSITE"] == "1"
    assert seen["CIV6AZ_LOCAL_RUN"] == "1"
    root = root.resolve()
    assert Path(seen["TMP"]) == root / "tmp" and Path(seen["TEMP"]) == root / "tmp"
    assert Path(seen["CIV6AZ_DATA_ROOT"]) == root
    assert "PYTHONPATH" not in seen
    # The child runs from a `git archive` export with no .git: the pinned SHA is how
    # its checkpoints and evaluations record their commit (T10 review I1).
    assert seen["CIV6AZ_GIT_SHA"] == _run_json(tmp_path)["sha"]
    assert len(seen["CIV6AZ_GIT_SHA"]) == 40


def test_preflight_version_mismatch_fails_card(
    tmp_path: Path, pinned: FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child's alpha_zero version disagrees with pyproject: preflight refuses, nothing runs."""
    bad = _fake_src(tmp_path / "bad", init_version="1.0.0")
    monkeypatch.setattr(
        pe, "_extract_archive", lambda sha, repo_root, dest, runner: shutil.copytree(bad, dest)
    )
    card = _card(tmp_path, [_train("smoke", "--dry_run")])

    result = _run(card, tmp_path, pinned)

    assert result.status == rc.STATUS_FAILED
    data = _run_json(tmp_path)
    assert data["status"] == "FAILED"
    assert data["preflight"]["exit_code"] != 0
    assert "version" in data["preflight"]["output"]
    assert data["invocations"] == []
    assert not (tmp_path / "root" / "runs" / "t1" / "logs" / "smoke.log").exists()


def test_failing_child_marks_failed_and_stops(
    tmp_path: Path, pinned: FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CHILD_EXIT", "3")
    card = _card(tmp_path, [_train("a", "--dry_run"), _train("b", "--dry_run")])

    result = _run(card, tmp_path, pinned)

    assert result.status == rc.STATUS_FAILED
    data = _run_json(tmp_path)
    assert [i["name"] for i in data["invocations"]] == ["a"]
    assert data["invocations"][0]["exit_code"] == 3


def test_killed_card_is_void(
    tmp_path: Path, pinned: FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*_: Any, **__: Any) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(rc, "_run_invocation", interrupted)
    card = _card(tmp_path, [_train("smoke", "--dry_run")])

    with pytest.raises(KeyboardInterrupt):
        _run(card, tmp_path, pinned)

    data = _run_json(tmp_path)
    assert data["status"] == "VOID"
    assert "interrupt" in data["void_reason"].lower()


def test_resumed_card_is_void(tmp_path: Path, pinned: FakeRunner) -> None:
    """A run.json left RUNNING (hard kill) means a rerun is a resume: VOID, refuse."""
    run_dir = tmp_path / "root" / "runs" / "t1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"card_id": "t1", "status": "RUNNING"}))
    card = _card(tmp_path, [_train("smoke", "--dry_run")])

    with pytest.raises(rc.CardResumedError):
        _run(card, tmp_path, pinned)
    data = _run_json(tmp_path)
    assert data["status"] == "VOID"
    assert "resume" in data["void_reason"].lower()


def test_completed_card_is_not_rerun_or_rewritten(tmp_path: Path, pinned: FakeRunner) -> None:
    run_dir = tmp_path / "root" / "runs" / "t1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"card_id": "t1", "status": "COMPLETED"}))
    card = _card(tmp_path, [_train("smoke", "--dry_run")])
    with pytest.raises(rc.CardResumedError, match="single-use"):
        _run(card, tmp_path, pinned)
    assert _run_json(tmp_path)["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# Data-root hardening
# ---------------------------------------------------------------------------


def test_harden_only_a_root_the_driver_creates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_harden(root: Path, **_: Any) -> None:
        calls.append(Path(root))
        _layout(root)

    monkeypatch.setattr(rc, "harden_data_root", fake_harden)
    new_root = tmp_path / "new-root"
    rc.prepare_data_root(new_root, repo_root=REPO)
    assert calls == [new_root]

    existing = tmp_path / "existing-root"
    existing.mkdir()
    rc.prepare_data_root(existing, repo_root=REPO)
    assert calls == [new_root], "an existing directory must never be re-ACLed"
    assert (existing / "runs").is_dir() and (existing / "tmp").is_dir()


def test_unsafe_data_root_refused_before_creation(tmp_path: Path) -> None:
    inside_repo = REPO / "civ6az-data-should-not-exist"
    with pytest.raises(ValueError, match="inside the repo"):
        rc.prepare_data_root(inside_repo, repo_root=REPO)
    assert not inside_repo.exists()


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------


def test_cli_price_only_prints_total_and_touches_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    card = _card(tmp_path, [_train("smoke", "--dry_run")])
    code = rc.main([str(card), "--price_only", "--data_root", str(tmp_path / "root")])
    assert code == 0
    assert "8" in capsys.readouterr().out
    assert not (tmp_path / "root").exists()


def test_cli_refusal_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    leg = ("--num_games", "52", "--num_iterations", "10", "--save_interval", "1000")
    card = _card(tmp_path, [_train("a", *leg), _train("b", *leg)])
    assert rc.main([str(card), "--price_only"]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_shipped_cards_price_under_the_cap() -> None:
    cards = sorted((REPO / "configs" / "run_cards").glob("*.yaml"))
    assert cards, "configs/run_cards/ ships at least the smoke card"
    for path in cards:
        card = rc.load_card(path)
        assert rc.price_card(card).total <= card.max_games, path


def test_script_wrapper_prices_the_smoke_card() -> None:
    """scripts/run_card.py runs THIS checkout's driver (sys.path pinned to the repo)."""
    out = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_card.py"),
            "--price_only",
            str(REPO / "configs" / "run_cards" / "smoke.yaml"),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "total" in out.stdout and " 8 " in out.stdout + " "


def test_child_is_capped_at_its_priced_games(
    tmp_path: Path, pinned: FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child pricing itself higher than the driver did must refuse itself (card FAILED)."""
    cmds: list[list[str]] = []
    real = rc._run_invocation

    def spy(cmd: list[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> int:
        cmds.append(list(cmd))
        return real(cmd, cwd=cwd, env=env, log=log)

    monkeypatch.setattr(rc, "_run_invocation", spy)
    one = ("--num_games", "1", "--num_iterations", "1", "--save_interval", "1000")
    card = _card(tmp_path, [_train("dry", "--dry_run"), _train("one", *one)])

    assert _run(card, tmp_path, pinned).status == rc.STATUS_COMPLETED
    assert [c[-2:] for c in cmds] == [["--max_games", "8"], ["--max_games", "1"]]
    assert cmds[1][-8:-2] == list(one), "the card argv is passed through unchanged"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("prefix", "OTHER_VENV", "not the pinned venv"),
        ("alpha_zero_file", "OUTSIDE", "outside the pinned src and venv"),
        ("version", "9.9.9", "pyproject version"),
    ],
)
def test_parent_reverifies_preflight_report(
    tmp_path: Path,
    pinned: FakeRunner,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    reason: str,
) -> None:
    """The pinned check passed, but HEAD's re-check of its report refuses the card."""
    real = rc._preflight
    substitutes = {
        "OTHER_VENV": str(tmp_path / "other-venv"),
        "OUTSIDE": str(tmp_path / "elsewhere" / "alpha_zero" / "__init__.py"),
    }

    def tampered(env: pe.PinnedEnv, child: Mapping[str, str]) -> dict[str, Any]:
        report = real(env, child)
        assert report["exit_code"] == 0, report
        report[field] = substitutes.get(value, value)
        return report

    monkeypatch.setattr(rc, "_preflight", tampered)
    card = _card(tmp_path, [_train("smoke", "--dry_run")])

    result = _run(card, tmp_path, pinned)

    assert result.status == rc.STATUS_FAILED
    data = _run_json(tmp_path)
    assert reason in data["error"]
    assert data["invocations"] == []
