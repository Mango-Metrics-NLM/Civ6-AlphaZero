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

Extraction is restricted to fenced code blocks and inline ``code`` spans. A
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

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"

_FENCE_RE = re.compile(r"```[a-z]*\n(.*?)```", re.S)
_INLINE_RE = re.compile(r"`([^`\n]+)`")
_MAKE_TARGET_RE = re.compile(r"(?:^|\|\s*|&&\s*)make\s+([a-z0-9_.-]+)")
_SCRIPT_RE = re.compile(r"(?<![\w/.-])(scripts/[\w./-]+\.(?:py|sh))")
_HARNESS_RE = re.compile(r"(?<![\w/.-])(\.claude/(?:hooks|skills)/[\w./-]+)")
_TESTS_RE = re.compile(r"(?<![\w/.-])(tests/[\w./-]+)")


def _make_targets(makefile: Path) -> set[str]:
    return set(re.findall(r"^([A-Za-z0-9_.-]+):", makefile.read_text(encoding="utf-8"), re.M))


def _code_spans(text: str) -> list[str]:
    """Command-bearing spans only: fenced-block lines and inline code spans."""
    spans: list[str] = []
    for block in _FENCE_RE.findall(text):
        spans += block.splitlines()
    spans += _INLINE_RE.findall(text)
    return spans


def dangling_references(skill_md: Path, root: Path, targets: set[str]) -> list[str]:
    """Referents named in ``skill_md``'s code spans that do not exist under ``root``."""
    findings: list[str] = []
    for span in _code_spans(skill_md.read_text(encoding="utf-8")):
        for target in _MAKE_TARGET_RE.findall(span.strip()):
            if target not in targets:
                findings.append(f"make target '{target}'")
        for path in _SCRIPT_RE.findall(span) + _HARNESS_RE.findall(span):
            if not (root / path).exists():
                findings.append(f"path '{path}'")
        if "pytest" in span:
            for path in _TESTS_RE.findall(span):
                if not (root / path.split("::")[0]).exists():
                    findings.append(f"pytest path '{path}'")
    return findings


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
