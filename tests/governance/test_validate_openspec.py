"""Deterministic tests for ``scripts/validate_openspec.py``.

Each error code has a fixture that provokes it. The validator's module-level
``OPENSPEC_DIR``/``REPO_ROOT`` globals are monkeypatched onto a ``tmp_path`` tree
so no test touches the real ``openspec/`` directory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_openspec.py"
_spec = importlib.util.spec_from_file_location("validate_openspec", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vo = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vo  # dataclasses resolves annotations via sys.modules
_spec.loader.exec_module(vo)


VALID_BASELINE = """\
# cap Specification

## Purpose

A baseline capability used by the tests.

## Requirements

### Requirement: Existing thing

The system SHALL do the existing thing.

#### Scenario: It happens

- **WHEN** a trigger occurs
- **THEN** the outcome is observed
"""

VALID_PROPOSAL = """\
# Proposal: demo

## Why

Because the tests need a reason.

## What Changes

- Something changes.

## Impact

- cap
"""

VALID_TASKS = "# Tasks\n\n- [ ] 1.1 do the thing\n"

VALID_DELTA = """\
# cap delta

## ADDED Requirements

### Requirement: Brand new thing

The system SHALL do a brand new thing.

#### Scenario: New thing happens

- **WHEN** the new trigger occurs
- **THEN** the new outcome is observed
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_valid_tree(root: Path) -> Path:
    """Create a minimal valid ``openspec/`` tree and return its path."""
    ospec = root / "openspec"
    _write(ospec / "project.md", "# Project Context\n\n## Purpose\n\nDemo.\n")
    _write(ospec / "AGENTS.md", "# OpenSpec Agent Instructions\n\nDemo.\n")
    _write(ospec / "specs" / "cap" / "spec.md", VALID_BASELINE)
    _write(ospec / "changes" / "demo" / "proposal.md", VALID_PROPOSAL)
    _write(ospec / "changes" / "demo" / "tasks.md", VALID_TASKS)
    _write(ospec / "changes" / "demo" / "specs" / "cap" / "spec.md", VALID_DELTA)
    return ospec


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    ospec = build_valid_tree(tmp_path)
    monkeypatch.setattr(vo, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(vo, "OPENSPEC_DIR", ospec)
    return ospec


def _run(argv=None):
    return vo.main(argv or [])


def test_valid_tree_passes(tree, capsys):
    assert _run() == 0
    assert "OK" in capsys.readouterr().out


def test_os001_missing_agents(tree, capsys):
    (tree / "AGENTS.md").unlink()
    assert _run() == 1
    assert "OS001" in capsys.readouterr().err


def test_os002_non_kebab_capability(tree, capsys):
    _write(tree / "specs" / "Bad_Cap" / "spec.md", VALID_BASELINE)
    assert _run() == 1
    assert "OS002" in capsys.readouterr().err


def test_os003_missing_purpose(tree, capsys):
    _write(tree / "specs" / "cap" / "spec.md", VALID_BASELINE.replace("## Purpose", "## Overview"))
    assert _run() == 1
    assert "OS003" in capsys.readouterr().err


def test_os004_no_normative_language(tree, capsys):
    text = VALID_BASELINE.replace("The system SHALL do the existing thing.", "It does a thing.")
    _write(tree / "specs" / "cap" / "spec.md", text)
    assert _run() == 1
    assert "OS004" in capsys.readouterr().err


def test_os005_scenario_missing_then(tree, capsys):
    text = VALID_BASELINE.replace("- **THEN** the outcome is observed\n", "")
    _write(tree / "specs" / "cap" / "spec.md", text)
    assert _run() == 1
    assert "OS005" in capsys.readouterr().err


def test_os010_proposal_missing_why(tree, capsys):
    text = VALID_PROPOSAL.replace("## Why", "## Rationale")
    _write(tree / "changes" / "demo" / "proposal.md", text)
    assert _run() == 1
    assert "OS010" in capsys.readouterr().err


def test_os011_tasks_no_checkbox(tree, capsys):
    _write(tree / "changes" / "demo" / "tasks.md", "# Tasks\n\nJust prose, no checkboxes.\n")
    assert _run() == 1
    assert "OS011" in capsys.readouterr().err


def test_os012_bad_change_id(tree, capsys):
    bad = tree / "changes" / "Bad_Id"
    _write(bad / "proposal.md", VALID_PROPOSAL)
    _write(bad / "tasks.md", VALID_TASKS)
    assert _run() == 1
    assert "OS012" in capsys.readouterr().err


def test_os020_invalid_delta_section(tree, capsys):
    text = VALID_DELTA.replace("## ADDED Requirements", "## UPDATED Requirements")
    _write(tree / "changes" / "demo" / "specs" / "cap" / "spec.md", text)
    assert _run() == 1
    assert "OS020" in capsys.readouterr().err


def test_os022_modified_absent_from_baseline(tree, capsys):
    text = "# cap delta\n\n## MODIFIED Requirements\n\n" + (
        "### Requirement: Nonexistent thing\n\n"
        "The system SHALL change a thing that has no baseline.\n\n"
        "#### Scenario: It changes\n\n- **WHEN** x\n- **THEN** y\n"
    )
    _write(tree / "changes" / "demo" / "specs" / "cap" / "spec.md", text)
    assert _run() == 1
    assert "OS022" in capsys.readouterr().err


def test_os022_added_only_new_capability_passes(tree):
    # A delta for a capability with no baseline file, ADDED-only, is valid.
    _write(tree / "changes" / "demo" / "specs" / "newcap" / "spec.md", VALID_DELTA)
    assert _run() == 0


def test_os023_added_collides_with_baseline(tree, capsys):
    text = VALID_DELTA.replace("Brand new thing", "Existing thing")
    _write(tree / "changes" / "demo" / "specs" / "cap" / "spec.md", text)
    assert _run() == 1
    assert "OS023" in capsys.readouterr().err


def test_change_scoping_isolates_named_change(tree, capsys):
    broken = tree / "changes" / "other"
    _write(broken / "proposal.md", "# Proposal\n\nNo required sections.\n")
    _write(broken / "tasks.md", "# Tasks\n\nno checkbox\n")
    # Scoped to the good change: passes despite the broken sibling.
    assert _run(["--change", "demo"]) == 0
    # Whole-tree: the broken sibling fails.
    assert _run() == 1


def test_path_mode_validates_single_file(tree, capsys):
    bad = tree / "changes" / "demo" / "tasks.md"
    bad.write_text("# Tasks\n\nno checkbox here\n", encoding="utf-8")
    assert _run(["--path", str(bad)]) == 1
    assert "OS011" in capsys.readouterr().err


def test_json_output_is_machine_readable(tree, capsys):
    (tree / "AGENTS.md").unlink()
    assert _run(["--json"]) == 1
    import json

    payload = json.loads(capsys.readouterr().out)
    assert any(item["code"] == "OS001" for item in payload)
