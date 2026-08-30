"""Deterministic tests for ``scripts/validate_agents.py``.

Mirrors the house pattern used by the other governance suites: import the
script as a module, point its path globals at a ``tmp_path`` tree, call
``main([])`` and assert on the return code plus the emitted error code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_agents.py"
_spec = importlib.util.spec_from_file_location("validate_agents", _SCRIPT)
assert _spec is not None and _spec.loader is not None
va = importlib.util.module_from_spec(_spec)
# Registered in sys.modules so dataclasses can resolve the Issue annotations.
sys.modules[_spec.name] = va
_spec.loader.exec_module(va)


VALID_AGENT = """\
# Agent.md — alpha_zero.demo

> One-line summary.

---

## Persona

You are the **Demo Engineer**. You own the demo.

---

## Module Purpose

Does demo things.

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Facade** | `thing.py` | one surface |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `thing.py` | the thing |

---

## Skills

- Demoing

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Demo** | demo | never |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest` | test |

---

## Invariants

1. It demos.

---

## Dependencies

- **Internal**: none
"""


def _point_at(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(va, "REPO_ROOT", root)
    monkeypatch.setattr(va, "PACKAGE_ROOT", root / "alpha_zero")
    pyproject = root / "pyproject.toml"
    pyproject.write_text("[tool.coverage.report]\nfail_under = 85\n", encoding="utf-8")
    monkeypatch.setattr(va, "PYPROJECT", pyproject)
    monkeypatch.setattr(va, "EXTRA_AGENT_DIRS", ())


def _write_package(root: Path, name: str, body: str | None, files: tuple[str, ...] = ()) -> Path:
    pkg = root / "alpha_zero" / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for extra in files:
        (pkg / extra).write_text("", encoding="utf-8")
    if body is not None:
        (pkg / "Agent.md").write_text(body, encoding="utf-8")
    return pkg


def test_valid_agent_passes(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_package(tmp_path, "demo", VALID_AGENT, files=("thing.py",))
    assert va.main([]) == 0
    assert "OK" in capsys.readouterr().out


def test_a1_bad_heading(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    body = VALID_AGENT.replace("# Agent.md — alpha_zero.demo", "# `alpha_zero.demo` — Subpackage")
    _write_package(tmp_path, "demo", body, files=("thing.py",))
    assert va.main([]) == 1
    assert "A1" in capsys.readouterr().err


def test_a1_heading_names_missing_directory(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    body = VALID_AGENT.replace("alpha_zero.demo", "alpha_zero.nonexistent")
    _write_package(tmp_path, "demo", body, files=("thing.py",))
    assert va.main([]) == 1
    assert "A1" in capsys.readouterr().err


def test_a2_missing_persona_statement(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    body = VALID_AGENT.replace("You are the **Demo Engineer**. You own the demo.", "Owns the demo.")
    _write_package(tmp_path, "demo", body, files=("thing.py",))
    assert va.main([]) == 1
    assert "A2" in capsys.readouterr().err


def test_a3_missing_required_section_is_error(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    body = VALID_AGENT.replace("## Design Patterns", "## Something Else")
    _write_package(tmp_path, "demo", body, files=("thing.py",))
    assert va.main([]) == 1
    assert "A3" in capsys.readouterr().err


def test_a3_missing_conventional_section_is_warning_only(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    body = VALID_AGENT.replace("## Sub-Agents", "## Something Else")
    _write_package(tmp_path, "demo", body, files=("thing.py",))
    assert va.main([]) == 0
    err = capsys.readouterr().err
    assert "WARNING" in err and "A3" in err


def test_a4_coverage_literal_must_match_gate(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    body = VALID_AGENT.replace("1. It demos.", "1. Coverage stays above 80%.")
    _write_package(tmp_path, "demo", body, files=("thing.py",))
    assert va.main([]) == 1
    assert "A4" in capsys.readouterr().err


def test_a4_ignores_decimal_tail_of_a_matching_percentage(tmp_path, monkeypatch, capsys):
    """`89.30%` must not read as a bare `30%` -- the naive regex did exactly that."""
    _point_at(monkeypatch, tmp_path)
    body = VALID_AGENT.replace("1. It demos.", "1. Coverage measured at 89.30% against the gate.")
    _write_package(tmp_path, "demo", body, files=("thing.py",))
    assert va.main([]) == 0


def test_a5_package_without_agent_md(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_package(tmp_path, "demo", VALID_AGENT, files=("thing.py",))
    _write_package(tmp_path, "undocumented", None)
    assert va.main([]) == 1
    assert "A5" in capsys.readouterr().err


def test_a6_referenced_file_must_exist(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_package(tmp_path, "demo", VALID_AGENT)  # note: no thing.py
    assert va.main([]) == 1
    assert "A6" in capsys.readouterr().err


def test_a6_allow_marker_exempts_line(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    body = VALID_AGENT.replace(
        "| `thing.py` | the thing |",
        "| `gone.py` | removed 2026-01-01 | <!-- validate-agents: allow -->",
    )
    _write_package(tmp_path, "demo", body, files=("thing.py",))
    assert va.main([]) == 0


def test_path_mode_validates_one_file(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    pkg = _write_package(tmp_path, "demo", VALID_AGENT.replace("## Persona", "## Nope"))
    assert va.main(["--path", str(pkg / "Agent.md")]) == 1
    assert "A3" in capsys.readouterr().err


def test_json_mode_emits_machine_readable_issues(tmp_path, monkeypatch, capsys):
    import json

    _point_at(monkeypatch, tmp_path)
    _write_package(tmp_path, "demo", VALID_AGENT)  # missing thing.py -> A6
    assert va.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert any(issue["code"] == "A6" for issue in payload["issues"])


def test_a0_missing_path_is_an_error(tmp_path, monkeypatch, capsys):
    """`--path` naming a missing file must fail, not silently validate nothing.

    The PostToolUse hook invokes `--path`, so exiting 0 here would make a stale
    path read as a clean validation.
    """
    _point_at(monkeypatch, tmp_path)
    assert va.main(["--path", str(tmp_path / "nope" / "Agent.md")]) == 1
    assert "A0" in capsys.readouterr().err


def test_a0_non_agent_md_target_is_an_error(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    other = tmp_path / "README.md"
    other.write_text("# not a persona doc\n", encoding="utf-8")
    assert va.main(["--path", str(other)]) == 1
    assert "A0" in capsys.readouterr().err
