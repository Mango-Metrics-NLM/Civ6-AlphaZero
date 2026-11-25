"""Rule S6 of ``scripts/validate_skills.py``: instructions must name real things.

A skill that names a script, make target or skill that does not exist sends an
agent down a dead path. S6 covers scripts, make targets and skills.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_skills.py"
_spec = importlib.util.spec_from_file_location("validate_skills_s6", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vs
_spec.loader.exec_module(vs)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _skill(body: str, name: str = "demo-skill") -> str:
    return (
        f"---\nname: {name}\ndescription: A demo skill.\nallowed-tools: Read\n---\n\n"
        f"# Demo\n\n## Steps\n\n{body}\n"
    )


def _tree(root: Path, monkeypatch, body: str, *, make_text: str = "verify:\n\ttrue\n") -> None:
    monkeypatch.setattr(vs, "REPO_ROOT", root)
    monkeypatch.setattr(vs, "SKILLS_DIR", root / ".claude" / "skills")
    monkeypatch.setattr(vs, "AGENT_SKILLS_DIR", root / ".agents" / "skills")
    pyproject = root / "pyproject.toml"
    pyproject.write_text("[tool.coverage.report]\nfail_under = 85\n", encoding="utf-8")
    monkeypatch.setattr(vs, "PYPROJECT", pyproject)
    (root / "Makefile").write_text(make_text, encoding="utf-8")
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "real_tool.py").write_text("", encoding="utf-8")
    for name in ("demo-skill", "real-skill"):
        d = root / ".claude" / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        text = _skill(body) if name == "demo-skill" else _skill("Do it.", "real-skill")
        (d / "SKILL.md").write_text(text, encoding="utf-8")


def _run(capsys) -> tuple[int, str]:
    code = vs.main([])
    return code, capsys.readouterr().err


def test_missing_script_is_s6(tmp_path, monkeypatch, capsys):
    _tree(tmp_path, monkeypatch, "Run `python scripts/ghost_tool.py` now.")
    code, err = _run(capsys)
    assert code == 1
    assert "S6" in err and "ghost_tool.py" in err and "demo-skill" in err


def test_missing_script_in_code_block_is_s6(tmp_path, monkeypatch, capsys):
    _tree(tmp_path, monkeypatch, "```bash\npython scripts/ghost_tool.py --x\n```")
    code, err = _run(capsys)
    assert code == 1 and "S6" in err and "ghost_tool.py" in err


def test_missing_make_target_is_s6(tmp_path, monkeypatch, capsys):
    _tree(tmp_path, monkeypatch, "Run `make ghost-target`.")
    code, err = _run(capsys)
    assert code == 1 and "S6" in err and "ghost-target" in err


def test_missing_skill_name_is_s6(tmp_path, monkeypatch, capsys):
    _tree(tmp_path, monkeypatch, "Follow the `ghost-skill` skill.")
    code, err = _run(capsys)
    assert code == 1 and "S6" in err and "ghost-skill" in err


def test_missing_skill_dir_path_is_s6(tmp_path, monkeypatch, capsys):
    _tree(tmp_path, monkeypatch, "See `.claude/skills/ghost-skill/` for more.")
    code, err = _run(capsys)
    assert code == 1 and "S6" in err and "ghost-skill" in err


def test_existing_references_pass(tmp_path, monkeypatch, capsys):
    body = (
        "Run `python scripts/real_tool.py` and `make verify`, "
        "see the `real-skill` skill and `.claude/skills/real-skill/`.\n\n"
        "Also see tests/scripts/test_other.py and make the number match."
    )
    _tree(tmp_path, monkeypatch, body)
    code, err = _run(capsys)
    assert code == 0, err
    assert "S6" not in err


def test_json_includes_sorted_s6(tmp_path, monkeypatch, capsys):
    _tree(
        tmp_path,
        monkeypatch,
        "Run `make zeta-target`, `python scripts/alpha_ghost.py`, the `ghost-skill` skill.",
    )
    assert vs.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    s6 = [i for i in payload if i["code"] == "S6"]
    assert len(s6) == 3
    keys = [(i["file"], i["line"] or 0, i["code"], i["message"]) for i in payload]
    assert keys == sorted(keys)


def test_real_repo_has_no_s6(capsys):
    """Every real skill (15 today, at least 13) passes S6, incl. the built-in ``code-review``."""
    vs.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert [i for i in payload if i["code"] == "S6"] == []
    assert len(list((REPO_ROOT / ".claude" / "skills").glob("*/SKILL.md"))) >= 13


@pytest.mark.parametrize("name", ["lua-bundle", "review-threads"])
def test_new_skill_exists_and_validates(name, capsys):
    path = REPO_ROOT / ".claude" / "skills" / name / "SKILL.md"
    assert path.is_file()
    assert vs.main(["--path", str(path)]) == 0, capsys.readouterr().err


# ---- make option words, fence styles, the path guard ------------------------


@pytest.mark.parametrize(
    "body",
    [
        "Run `make -C sub verify`.",
        "Run `make -j4 verify`.",
        "Run `make -j verify`.",
        "Run `make -n -f Makefile verify`.",
    ],
)
def test_make_options_before_an_existing_target_pass(tmp_path, monkeypatch, capsys, body):
    _tree(tmp_path, monkeypatch, body)
    code, err = _run(capsys)
    assert code == 0, err


@pytest.mark.parametrize(
    "body",
    [
        "Run `make -C sub ghost`.",
        "Run `make PYTHON=py ghost`.",
        "Run `make -j 4 ghost`.",
        "Run `make -j ghost`.",
    ],
)
def test_make_options_before_a_missing_target_are_s6(tmp_path, monkeypatch, capsys, body):
    _tree(tmp_path, monkeypatch, body)
    code, err = _run(capsys)
    assert code == 1 and "S6" in err and "make target 'ghost'" in err, err


def test_tilde_fence_is_read(tmp_path, monkeypatch, capsys):
    _tree(tmp_path, monkeypatch, "~~~bash\npython scripts/ghost_tool.py\n~~~")
    code, err = _run(capsys)
    assert code == 1 and "ghost_tool.py" in err


def test_four_backtick_fence_is_read_and_closed_only_by_its_own_length(
    tmp_path, monkeypatch, capsys
):
    body = (
        "````markdown\n```bash\npython scripts/ghost_tool.py\n```\n````\n\n"
        "After the fence: make ghost-prose is prose."
    )
    _tree(tmp_path, monkeypatch, body)
    code, err = _run(capsys)
    assert code == 1 and "ghost_tool.py" in err, err
    assert "ghost-prose" not in err


def test_path_guard_control(tmp_path, monkeypatch, capsys):
    """A path that continues another path is not a ``scripts/`` reference; a bare one is."""
    _tree(tmp_path, monkeypatch, "Run `python tools/scripts/ghost_tool.py` and `x.scripts/g.py`.")
    code, err = _run(capsys)
    assert code == 0, err
    _tree(tmp_path, monkeypatch, "Run `python scripts/ghost_tool.py`.")
    code, err = _run(capsys)
    assert code == 1 and "ghost_tool.py" in err


def test_skill_reference_tolerates_line_breaks_and_plurals(tmp_path, monkeypatch, capsys):
    _tree(tmp_path, monkeypatch, "Follow the `ghost-one`\nskill, then the `ghost-two` skills.")
    code, err = _run(capsys)
    assert code == 1 and "ghost-one" in err and "ghost-two" in err, err


def test_both_entry_points_share_one_skill_set(tmp_path, monkeypatch, capsys):
    """``main`` (via ``_check_references``) and ``dangling_references`` agree.

    The skill tree is patched away from ``root``: a function that rebuilt the set from
    ``root`` would see no ``real-skill`` and report it.
    """
    _tree(tmp_path, monkeypatch, "Follow the `real-skill` skill.")
    other = tmp_path / "elsewhere"
    other.mkdir()
    skill = tmp_path / ".claude" / "skills" / "demo-skill" / "SKILL.md"
    assert vs.dangling_references(skill, other, None) == []
    code, err = _run(capsys)
    assert code == 0, err
