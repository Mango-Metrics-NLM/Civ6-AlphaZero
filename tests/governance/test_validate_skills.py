"""Deterministic tests for ``scripts/validate_skills.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_skills.py"
_spec = importlib.util.spec_from_file_location("validate_skills", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vs
_spec.loader.exec_module(vs)

VALID_SKILL = """\
---
name: demo-skill
description: A demo skill used by the tests.
allowed-tools: Read, Write
---

# Demo Skill

## Steps

1. Do a thing.
"""


def _point_at(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(vs, "REPO_ROOT", root)
    monkeypatch.setattr(vs, "SKILLS_DIR", root / ".claude" / "skills")
    pyproject = root / "pyproject.toml"
    pyproject.write_text("[tool.coverage.report]\nfail_under = 85\n", encoding="utf-8")
    monkeypatch.setattr(vs, "PYPROJECT", pyproject)


def _write_skill(root: Path, name: str, body: str) -> Path:
    skill = root / ".claude" / "skills" / name / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(body, encoding="utf-8")
    return skill


def test_valid_skill_passes(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_skill(tmp_path, "demo-skill", VALID_SKILL)
    assert vs.main([]) == 0
    assert "OK" in capsys.readouterr().out


def test_s1_missing_frontmatter(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_skill(tmp_path, "demo-skill", "# No Frontmatter\n\n## Steps\n\n1. x\n")
    assert vs.main([]) == 1
    assert "S1" in capsys.readouterr().err


def test_s1_name_mismatch(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_skill(tmp_path, "demo-skill", VALID_SKILL.replace("name: demo-skill", "name: wrong"))
    assert vs.main([]) == 1
    assert "S1" in capsys.readouterr().err


def test_s2_no_section(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    body = (
        "---\nname: demo-skill\ndescription: x\nallowed-tools: Read\n---\n\nNo H2 section here.\n"
    )
    _write_skill(tmp_path, "demo-skill", body)
    assert vs.main([]) == 1
    assert "S2" in capsys.readouterr().err


def test_s3_missing_kb_path(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    body = VALID_SKILL + "\nUpdate kb/summaries/pre-pr.md when done.\n"
    _write_skill(tmp_path, "demo-skill", body)
    assert vs.main([]) == 1
    assert "S3" in capsys.readouterr().err


def test_s3_ignores_placeholder_paths(tmp_path, monkeypatch):
    _point_at(monkeypatch, tmp_path)
    body = VALID_SKILL + "\nCreate kb/handoffs/YYYYMMDD-<slug>.md as a template.\n"
    _write_skill(tmp_path, "demo-skill", body)
    assert vs.main([]) == 0


def test_s4_coverage_literal_mismatch(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    body = VALID_SKILL + "\nEnsure coverage stays above 80% before handoff.\n"
    _write_skill(tmp_path, "demo-skill", body)
    assert vs.main([]) == 1
    assert "S4" in capsys.readouterr().err


def test_allowed_tools_missing_is_warning_not_error(tmp_path, monkeypatch):
    _point_at(monkeypatch, tmp_path)
    body = "---\nname: demo-skill\ndescription: x\n---\n\n# Demo\n\n## Steps\n\n1. x\n"
    _write_skill(tmp_path, "demo-skill", body)
    # A missing allowed-tools is a WARNING, so the run still passes.
    assert vs.main([]) == 0
