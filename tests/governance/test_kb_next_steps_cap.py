"""H3: ``resume_pointers.next_steps`` in a session state file holds at most five items.

Slip: ``kb/sessions/dev-sqe.state.json`` grew a 15-item ``next_steps`` list that was
really a lessons log, and went stale. K6 now caps it, on both the directory scan and
the single-file (hook) path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_kb.py"
_spec = importlib.util.spec_from_file_location("validate_kb_cap", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vk = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vk
_spec.loader.exec_module(vk)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _point_at(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(vk, "REPO_ROOT", root)
    monkeypatch.setattr(vk, "LEDGER", root / "kb" / "ledger" / "ledger.jsonl")
    monkeypatch.setattr(vk, "SESSIONS_DIR", root / "kb" / "sessions")
    ledger = root / "kb" / "ledger" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("", encoding="utf-8")


def _state(root: Path, next_steps: object, name: str = "dev-sqe.state.json") -> Path:
    sessions = root / "kb" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    path = sessions / name
    path.write_text(json.dumps({"resume_pointers": {"next_steps": next_steps}}), encoding="utf-8")
    return path


@pytest.mark.parametrize("count", [0, 1, 5])
def test_up_to_five_steps_pass(tmp_path, monkeypatch, capsys, count) -> None:
    _point_at(monkeypatch, tmp_path)
    _state(tmp_path, [f"step {i}" for i in range(count)])
    assert vk.main(["--no-git"]) == 0
    assert "K6" not in capsys.readouterr().err


def test_six_steps_fail_naming_file_and_count(tmp_path, monkeypatch, capsys) -> None:
    _point_at(monkeypatch, tmp_path)
    _state(tmp_path, [f"step {i}" for i in range(6)])
    assert vk.main(["--no-git"]) == 1
    err = capsys.readouterr().err
    assert "K6" in err
    assert "dev-sqe.state.json" in err
    assert "6" in err


def test_fifteen_steps_report_the_count(tmp_path, monkeypatch, capsys) -> None:
    _point_at(monkeypatch, tmp_path)
    _state(tmp_path, [f"lesson {i}" for i in range(15)])
    assert vk.main(["--no-git"]) == 1
    err = capsys.readouterr().err
    assert "K6" in err
    assert "15" in err


def test_single_file_hook_path_applies_the_cap(tmp_path, monkeypatch, capsys) -> None:
    _point_at(monkeypatch, tmp_path)
    path = _state(tmp_path, [f"step {i}" for i in range(6)])
    assert vk.main(["--no-git", "--path", str(path)]) == 1
    err = capsys.readouterr().err
    assert "K6" in err
    assert "6" in err


@pytest.mark.parametrize(
    "bad",
    [["ok", ""], ["ok", "   "], ["ok", 3], ["ok", None], "a string", {"a": 1}, 7],
)
def test_next_steps_must_be_a_list_of_non_empty_strings(tmp_path, monkeypatch, capsys, bad) -> None:
    _point_at(monkeypatch, tmp_path)
    _state(tmp_path, bad)
    assert vk.main(["--no-git"]) == 1
    assert "K6" in capsys.readouterr().err


def test_absent_next_steps_is_fine(tmp_path, monkeypatch) -> None:
    _point_at(monkeypatch, tmp_path)
    sessions = tmp_path / "kb" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "phase-6.state.json").write_text(json.dumps({"phase": 6}), encoding="utf-8")
    (sessions / "x.state.json").write_text(json.dumps({"resume_pointers": {}}), encoding="utf-8")
    assert vk.main(["--no-git"]) == 0


def test_committed_state_files_pass_the_cap() -> None:
    for path in sorted((REPO_ROOT / "kb" / "sessions").glob("*.state.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        steps = (data.get("resume_pointers") or {}).get("next_steps", data.get("next_steps", []))
        assert isinstance(steps, list) and len(steps) <= 5, path.name
