"""Deterministic tests for ``scripts/validate_kb.py``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_kb.py"
_spec = importlib.util.spec_from_file_location("validate_kb", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vk = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vk
_spec.loader.exec_module(vk)

CURRENT_LINE = (
    '{"event": "DEV-SQE-HANDOFF", "timestamp": "2026-08-01T00:00:00Z", "session_id": "session_abc"}'
)
PRE_CUTOFF_LEGACY = '{"event": "DEV-COMPLETE", "timestamp": "2026-02-28T12:00:00"}'


def _point_at(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(vk, "REPO_ROOT", root)
    monkeypatch.setattr(vk, "LEDGER", root / "kb" / "ledger" / "ledger.jsonl")
    monkeypatch.setattr(vk, "SESSIONS_DIR", root / "kb" / "sessions")


def _write_ledger(root: Path, *lines: str) -> None:
    ledger = root / "kb" / "ledger" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_valid_ledger_passes(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_ledger(tmp_path, CURRENT_LINE, PRE_CUTOFF_LEGACY)
    assert vk.main(["--no-git"]) == 0
    assert "OK" in capsys.readouterr().out


def test_k1_malformed_json(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_ledger(tmp_path, "{not json}")
    assert vk.main(["--no-git"]) == 1
    assert "K1" in capsys.readouterr().err


def test_k2_missing_event(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_ledger(tmp_path, '{"timestamp": "2026-08-01T00:00:00Z", "session_id": "s"}')
    assert vk.main(["--no-git"]) == 1
    assert "K2" in capsys.readouterr().err


def test_k3_post_cutoff_legacy_event_rejected(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        '{"event": "DEV-COMPLETE", "timestamp": "2026-08-01T00:00:00Z", "session_id": "s"}',
    )
    assert vk.main(["--no-git"]) == 1
    assert "K3" in capsys.readouterr().err


def test_k4_post_cutoff_missing_session_id(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_ledger(tmp_path, '{"event": "PR-READY", "timestamp": "2026-08-01T00:00:00Z"}')
    assert vk.main(["--no-git"]) == 1
    assert "K4" in capsys.readouterr().err


def test_k5_missing_referenced_path(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        '{"event": "PR-READY", "timestamp": "2026-08-01T00:00:00Z", '
        '"session_id": "s", "spec_file": "kb/specs/nope.md"}',
    )
    assert vk.main(["--no-git"]) == 1
    assert "K5" in capsys.readouterr().err


def test_k6_malformed_session_state(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_ledger(tmp_path, CURRENT_LINE)
    sessions = tmp_path / "kb" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / "planning.state.json").write_text("{bad json", encoding="utf-8")
    assert vk.main(["--no-git"]) == 1
    assert "K6" in capsys.readouterr().err


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _make_git_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "commit.gpgsign", "false")


def test_k7_modified_handoff_is_rejected(tmp_path, monkeypatch, capsys):
    _make_git_repo(tmp_path)
    _point_at(monkeypatch, tmp_path)
    handoffs = tmp_path / "kb" / "handoffs"
    handoffs.mkdir(parents=True)
    handoff = handoffs / "20260101-a-to-b.md"
    handoff.write_text("original\n", encoding="utf-8")
    _write_ledger(tmp_path, CURRENT_LINE)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "add handoff")
    handoff.write_text("tampered\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "tamper")
    assert vk.main(["--git"]) == 1
    assert "K7" in capsys.readouterr().err


def test_k7_clean_history_passes(tmp_path, monkeypatch):
    _make_git_repo(tmp_path)
    _point_at(monkeypatch, tmp_path)
    handoffs = tmp_path / "kb" / "handoffs"
    handoffs.mkdir(parents=True)
    (handoffs / "20260101-a-to-b.md").write_text("original\n", encoding="utf-8")
    _write_ledger(tmp_path, CURRENT_LINE)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "add handoff")
    assert vk.main(["--git"]) == 0


def test_no_git_skips_immutability(tmp_path, monkeypatch):
    # No git repo at all: --no-git must still pass on a valid ledger.
    _point_at(monkeypatch, tmp_path)
    _write_ledger(tmp_path, CURRENT_LINE)
    assert vk.main(["--no-git"]) == 0


def test_path_mode_validates_single_ledger(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    ledger = tmp_path / "kb" / "ledger" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("{broken\n", encoding="utf-8")
    assert vk.main(["--path", str(ledger)]) == 1
    assert "K1" in capsys.readouterr().err


def test_real_repo_ledger_is_grandfathered():
    # The 12 historical lines must pass under the frozen grandfathering rules.
    assert vk.main(["--no-git", "--quiet"]) == 0
