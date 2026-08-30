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
    # Both roots, or a tmp-tree test silently scans the repo's real
    # `.agents/skills/` and asserts against whatever happens to be there.
    monkeypatch.setattr(vs, "AGENT_SKILLS_DIR", root / ".agents" / "skills")
    pyproject = root / "pyproject.toml"
    pyproject.write_text("[tool.coverage.report]\nfail_under = 85\n", encoding="utf-8")
    monkeypatch.setattr(vs, "PYPROJECT", pyproject)


def _write_skill(root: Path, name: str, body: str, tree: str = ".claude") -> Path:
    skill = root / tree / "skills" / name / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(body, encoding="utf-8")
    return skill


def _write_empty_skill_dir(root: Path, name: str, tree: str = ".claude") -> Path:
    """A skill directory with no SKILL.md -- used to be skipped in silence."""
    skill_dir = root / tree / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    return skill_dir


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


# ---------------------------------------------------------------------------
# Block-scalar frontmatter (the vacuous-validation bug)
# ---------------------------------------------------------------------------

FOLDED_SKILL = """\
---
name: demo-skill
description: >
  A folded description that spans
  more than one line.
allowed-tools: Read, Write
---

# Demo Skill

## Steps

1. Do a thing.
"""

LITERAL_SKILL = FOLDED_SKILL.replace("description: >", "description: |")

EMPTY_FOLDED_SKILL = """\
---
name: demo-skill
description: >
allowed-tools: Read, Write
---

# Demo Skill

## Steps

1. Do a thing.
"""


def test_folded_description_is_parsed_as_text(tmp_path, monkeypatch, capsys):
    """`description: >` must yield the folded text, not the literal '>'.

    The hand-rolled parser used to take everything after the first colon, so a
    folded scalar validated as the string ">" -- non-empty, therefore passing.
    That is worse than not checking: it reports a guarantee it never made.
    """
    _point_at(monkeypatch, tmp_path)
    _write_skill(tmp_path, "demo-skill", FOLDED_SKILL)
    assert vs.main([]) == 0
    front = vs._parse_frontmatter(FOLDED_SKILL.splitlines())
    assert front is not None
    assert front["description"].startswith("A folded description")
    assert front["description"] != ">"


def test_literal_description_is_parsed_as_text(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_skill(tmp_path, "demo-skill", LITERAL_SKILL)
    assert vs.main([]) == 0
    front = vs._parse_frontmatter(LITERAL_SKILL.splitlines())
    assert front is not None
    assert front["description"].startswith("A folded description")


def test_empty_block_scalar_description_fails(tmp_path, monkeypatch, capsys):
    """An empty folded scalar must fail S1 -- this is the case that used to pass."""
    _point_at(monkeypatch, tmp_path)
    _write_skill(tmp_path, "demo-skill", EMPTY_FOLDED_SKILL)
    assert vs.main([]) == 1
    assert "S1" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The second skill tree, and S5
# ---------------------------------------------------------------------------


def test_agent_skills_tree_is_scanned(tmp_path, monkeypatch, capsys):
    """`.agents/skills/` is gated too -- it was routed here by the hook but by no gate."""
    _point_at(monkeypatch, tmp_path)
    _write_skill(tmp_path, "demo-skill", VALID_SKILL)
    _write_skill(tmp_path, "other-skill", VALID_SKILL, tree=".agents")  # name != dir
    assert vs.main([]) == 1
    err = capsys.readouterr().err
    assert "S1" in err and "other-skill" in err


def test_agent_skills_tree_valid_skill_passes(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_skill(tmp_path, "demo-skill", VALID_SKILL, tree=".agents")
    assert vs.main([]) == 0


def test_s5_skill_directory_without_skill_md(tmp_path, monkeypatch, capsys):
    """A skill directory with no SKILL.md used to be skipped silently."""
    _point_at(monkeypatch, tmp_path)
    _write_skill(tmp_path, "demo-skill", VALID_SKILL)
    _write_empty_skill_dir(tmp_path, "hollow-skill")
    assert vs.main([]) == 1
    err = capsys.readouterr().err
    assert "S5" in err and "hollow-skill" in err


def test_s5_applies_to_the_agent_tree_too(tmp_path, monkeypatch, capsys):
    _point_at(monkeypatch, tmp_path)
    _write_skill(tmp_path, "demo-skill", VALID_SKILL)
    _write_empty_skill_dir(tmp_path, "hollow-skill", tree=".agents")
    assert vs.main([]) == 1
    assert "S5" in capsys.readouterr().err
