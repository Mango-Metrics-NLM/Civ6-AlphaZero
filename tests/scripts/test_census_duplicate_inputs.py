"""One input file given twice would count every turn twice (round 2, R1).

``main`` resolves every input path (following symbolic links) before reading anything and
refuses, exit 2 with a one-line reason naming the file, when two inputs are one file.
Nothing is written then: not even the out directory.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.census_states import (
    load_census,
    mixed_state,
    read_json,
    run_cli,
    state_line,
    wire,
    write_json,
    write_log,
)

SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _pinned_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_GIT_SHA", SHA)


def _log(path: Path, turns: Sequence[int]) -> Path:
    return write_log(path, [state_line(wire(mixed_state(t))) for t in turns])


def _refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], *argv: str, name: str = "Lua.log"
) -> None:
    mod: Any = load_census()
    code, target = run_cli(mod, tmp_path, *argv, "--allow-small", "--boards", "8")
    err = capsys.readouterr().err
    assert code == 2
    assert not target.exists() and not target.parent.exists()
    reason = [line for line in err.splitlines() if line.strip()]
    assert len(reason) == 1 and name in reason[0]


def test_the_same_path_twice_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = _log(tmp_path / "Lua.log", range(21, 41))
    _refused(tmp_path, capsys, "--log", str(log), "--log", str(log))


def test_a_relative_and_an_absolute_spelling_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _log(tmp_path / "Lua.log", range(21, 41))
    monkeypatch.chdir(tmp_path)
    _refused(tmp_path, capsys, "--log", "Lua.log", "--log", str(log))


def test_a_symbolic_link_and_its_target_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _log(tmp_path / "Lua.log", range(21, 41))
    link = tmp_path / "link.log"
    try:
        link.symlink_to(log)
    except OSError:
        if sys.platform != "win32":
            raise
        # Windows without symlink privilege falls back to a hard link; _given_twice checks st_dev/st_ino.
        link.hardlink_to(log)
    _refused(tmp_path, capsys, "--log", str(link), "--log", str(log))


def test_one_file_as_both_log_and_states_json_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    both = write_json(tmp_path / "both.json", [wire(mixed_state(t)) for t in range(1, 4)])
    _refused(tmp_path, capsys, "--log", str(both), "--states-json", str(both), name="both.json")


def test_the_same_states_json_twice_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    states = write_json(tmp_path / "s.json", [wire(mixed_state(t)) for t in range(1, 4)])
    _refused(
        tmp_path, capsys, "--states-json", str(states), "--states-json", str(states), name="s.json"
    )


def test_control_two_different_logs_are_accepted(tmp_path: Path) -> None:
    mod: Any = load_census()
    one = _log(tmp_path / "a.log", range(1, 4))
    two = _log(tmp_path / "b.log", range(4, 8))
    code, target = run_cli(mod, tmp_path, "--log", str(one), "--log", str(two), "--allow-small")
    assert code == 0
    assert [r["turn"] for r in read_json(target)["states"]] == [1, 2, 3, 4, 5, 6, 7]
