"""Commands a skill tells you to run must exist.

Why this gate exists
--------------------
A skill is an instruction sheet an agent follows without checking. When a make
target is renamed or a script moves, the skill keeps confidently naming the old
one, and the agent runs it, gets "No rule to make target", and improvises. The
`.claude/skills/` docs cite 47 make targets, script paths and pytest paths
between them; none was verified against the tree.

Scope, stated honestly: this checks that the *referent exists* -- the make
target is defined in the Makefile, the script/hook file is on disk, the pytest
path resolves. It does not run anything, so it cannot tell you a command still
does what the skill claims.

Extraction (``scripts/validate_skills.py``, rule S6, imported here) is
restricted to fenced code blocks and inline ``code`` spans. A
naive scan over prose reported two false positives on this very tree --
"raising a baseline to make a red build green" and "make the coverage-gate
number match" -- which is how a checker of this shape becomes noise and gets
deleted.

Negative control (recorded per the ``negative-control`` skill)
--------------------------------------------------------------
``test_negative_control_detects_dangling_references`` plants a tmp skill citing
a nonexistent make target, script, hook and pytest path, and asserts all four
are reported; ``test_negative_control_ignores_prose`` asserts the two real
prose sentences above are still not reported. Verified: 4 findings on the
planted skill, 0 on the prose, 0 on the real trees.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"

# The extraction lives in the validator (rule S6) and is imported here, so the
# gate and this test cannot drift apart.
_SCRIPT = REPO_ROOT / "scripts" / "validate_skills.py"
_spec = importlib.util.spec_from_file_location("validate_skills_resolve", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_vs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _vs
_spec.loader.exec_module(_vs)

dangling_references = _vs.dangling_references


def _make_targets(makefile: Path) -> set[str]:
    targets = _vs.make_targets(makefile)
    assert targets is not None, f"no Makefile at {makefile}"
    return targets


def _all_skills() -> list[Path]:
    files: list[Path] = []
    for tree in (REPO_ROOT / ".claude" / "skills", REPO_ROOT / ".agents" / "skills"):
        files += sorted(tree.glob("*/SKILL.md"))
    return files


def test_no_skill_cites_a_command_that_does_not_exist() -> None:
    """The gate, run across both skill trees."""
    targets = _make_targets(MAKEFILE)
    assert len(targets) > 20, f"Makefile parse looks wrong -- only {len(targets)} targets"

    problems: list[str] = []
    for skill in _all_skills():
        for finding in dangling_references(skill, REPO_ROOT, targets):
            problems.append(f"{skill.relative_to(REPO_ROOT)}: {finding}")
    assert not problems, "skills naming things that do not exist:\n  " + "\n  ".join(problems)


def test_negative_control_detects_dangling_references(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: each reference kind must be catchable.

    Four kinds are planted -- make target, script path, hook path, pytest path.
    All four must be reported; a checker that catches three of four would pass
    the gate above while a renamed script sat in a skill unnoticed.
    """
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\nname: probe\ndescription: probe\n---\n\n"
        "## Steps\n\n"
        "Run `make no-such-target` then `python scripts/no_such_script.py`.\n\n"
        "```bash\n"
        "bash .claude/hooks/no_such_hook.sh\n"
        "pytest tests/no_such_dir/test_nope.py -q\n"
        "```\n",
        encoding="utf-8",
    )
    findings = dangling_references(skill, REPO_ROOT, _make_targets(MAKEFILE))
    assert any("no-such-target" in f for f in findings), f"make target missed: {findings}"
    assert any("no_such_script.py" in f for f in findings), f"script path missed: {findings}"
    assert any("no_such_hook.sh" in f for f in findings), f"hook path missed: {findings}"
    assert any("no_such_dir" in f for f in findings), f"pytest path missed: {findings}"
    assert len(findings) == 4, f"expected exactly 4 findings, got {findings}"


def test_negative_control_ignores_prose(tmp_path: Path) -> None:
    """The other half: English must not be parsed as a command.

    Both sentences are real -- they come from `debt-audit` and `doc-sync` -- and
    a prose-wide scan reported both as dangling `make` targets.
    """
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\nname: probe\ndescription: probe\n---\n\n"
        "## Notes\n\n"
        "Raising a baseline to make a red build green is the one thing this forbids.\n"
        "D3 - make the coverage-gate number match `pyproject` `fail_under`.\n",
        encoding="utf-8",
    )
    assert dangling_references(skill, REPO_ROOT, _make_targets(MAKEFILE)) == []


def test_real_commands_are_recognised(tmp_path: Path) -> None:
    """A referent that DOES exist must not be reported -- otherwise the gate
    above passes only because nothing is ever matched."""
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\nname: probe\ndescription: probe\n---\n\n"
        "## Steps\n\n```bash\nmake verify\npython scripts/validate_skills.py\n"
        "pytest tests/governance -q\n```\n",
        encoding="utf-8",
    )
    assert dangling_references(skill, REPO_ROOT, _make_targets(MAKEFILE)) == []


def test_negative_control_prose_is_never_read(tmp_path: Path) -> None:
    """Prose that would be a finding if it were read as code is not reported.

    The control above passes even when prose is scanned, because neither of its
    sentences starts with ``make`` or names a path. These lines do: under a scan
    that reads prose, each one is a finding (a dangling script, hook and pytest
    path, a make target and a skill), so only the code-span restriction keeps
    this list empty.
    """
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\nname: probe\ndescription: probe\n---\n\n"
        "## Notes\n\n"
        "make the coverage-gate number match before you push.\n"
        "The old scripts/ghost_prose.py and .claude/hooks/ghost_prose.sh are gone;\n"
        "pytest tests/ghost_prose/ no longer exists, and neither does the ghost-prose skill.\n",
        encoding="utf-8",
    )
    assert dangling_references(skill, REPO_ROOT, _make_targets(MAKEFILE)) == []
