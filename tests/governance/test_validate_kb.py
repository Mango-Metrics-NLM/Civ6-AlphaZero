"""Deterministic tests for ``scripts/validate_kb.py``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_pre_cutoff_legacy_entry_without_session_id_passes(tmp_path, monkeypatch):
    # Grandfathering: a pre-cutoff entry may use a legacy event and omit session_id.
    # (Covers what the removed real-repo smoke test asserted, without coupling to
    # the live, growing ledger — CI's `validate_kb.py --git` covers the real repo.)
    _point_at(monkeypatch, tmp_path)
    _write_ledger(tmp_path, '{"event": "PLAN-LOCKED", "timestamp": "2026-02-08T13:10:31"}')
    assert vk.main(["--no-git"]) == 0


def test_k5_archived_openspec_change_resolves(tmp_path, monkeypatch):
    """A spec_file recorded pre-archive stays valid after the change directory
    moves to openspec/changes/archive/<date>-<id>/ (the ledger is append-only)."""
    _point_at(monkeypatch, tmp_path)
    archived = tmp_path / "openspec" / "changes" / "archive" / "2026-07-25-add-thing"
    archived.mkdir(parents=True)
    (archived / "proposal.md").write_text("# p\n", encoding="utf-8")
    _write_ledger(
        tmp_path,
        '{"event": "PR-READY", "timestamp": "2026-08-01T00:00:00Z", '
        '"session_id": "s", "spec_file": "openspec/changes/add-thing/proposal.md"}',
    )
    assert vk.main(["--no-git"]) == 0


def test_k5_unarchived_missing_openspec_change_still_fails(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    (tmp_path / "openspec" / "changes" / "archive").mkdir(parents=True)
    _write_ledger(
        tmp_path,
        '{"event": "PR-READY", "timestamp": "2026-08-01T00:00:00Z", '
        '"session_id": "s", "spec_file": "openspec/changes/never-existed/proposal.md"}',
    )
    assert vk.main(["--no-git"]) == 1
    assert "K5" in capsys.readouterr().err


def test_validation_run_is_an_accepted_event(tmp_path, monkeypatch, capsys):
    """A run that produced evidence must be recordable.

    Before this event existed the ledger could only hold process events, so
    "was this code ever run?" was unanswerable from the KB.
    """
    _point_at(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        '{"event": "VALIDATION-RUN", "timestamp": "2026-08-28T02:45:00Z", "session_id": "s"}',
    )
    assert vk.main(["--no-git"]) == 0
    assert "K3" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# K7 on a shallow clone: a note by default, a FAILURE under --git.
#
# Audit finding G7 (2026-09-04): `validate_kb.py --git` printed "shallow clone --
# skipping K7" and exited 0, so `make validate` passed with the immutability rule
# unenforced on any shallow checkout. The flag now makes K7 mandatory.
#
# Negative control (performed 2026-11-10): with the `require` branch disabled in
# scripts/validate_kb.py, test_k7_shallow_clone_fails_under_git_flag failed with
# `assert 0 == 1`; restored, it passes.
# ---------------------------------------------------------------------------


def _make_shallow_clone(tmp_path: Path) -> Path:
    """A depth-1 clone of a two-commit repo with a handoff in it."""
    src = tmp_path / "src"
    src.mkdir()
    _make_git_repo(src)
    handoffs = src / "kb" / "handoffs"
    handoffs.mkdir(parents=True)
    (handoffs / "20260101-a-to-b.md").write_text("original\n", encoding="utf-8")
    ledger = src / "kb" / "ledger" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(CURRENT_LINE + "\n", encoding="utf-8")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "add handoff")
    (src / "README.md").write_text("second commit so depth=1 cuts history\n", encoding="utf-8")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "second")

    dst = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{src}", str(dst)],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=dst,
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "true", "fixture is not shallow; the tests below are vacuous"
    return dst


def test_k7_shallow_clone_fails_under_git_flag(tmp_path, monkeypatch, capsys):
    shallow = _make_shallow_clone(tmp_path)
    _point_at(monkeypatch, shallow)
    assert vk.main(["--git"]) == 1
    err = capsys.readouterr().err
    assert "K7" in err
    assert "shallow" in err


def test_k7_shallow_clone_is_only_a_note_without_the_flag(tmp_path, monkeypatch, capsys):
    """Today's default behaviour is preserved: the pre-commit hook still runs."""
    shallow = _make_shallow_clone(tmp_path)
    _point_at(monkeypatch, shallow)
    assert vk.main([]) == 0
    captured = capsys.readouterr()
    assert "shallow clone" in captured.err
    assert "OK" in captured.out


def test_k7_full_clone_passes_under_git_flag(tmp_path, monkeypatch):
    """The flag fails only an UNVERIFIABLE history, not a verifiable clean one."""
    shallow = _make_shallow_clone(tmp_path)
    subprocess.run(
        ["git", "fetch", "-q", "--unshallow"],
        cwd=shallow,
        check=True,
        capture_output=True,
        text=True,
    )
    _point_at(monkeypatch, shallow)
    assert vk.main(["--git"]) == 0


def test_git_and_no_git_are_mutually_exclusive(tmp_path, monkeypatch):
    _point_at(monkeypatch, tmp_path)
    _write_ledger(tmp_path, CURRENT_LINE)
    with pytest.raises(SystemExit) as excinfo:
        vk.main(["--git", "--no-git"])
    assert excinfo.value.code == 2
