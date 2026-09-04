"""Every SKILL.md carries frontmatter with a description a reader can use.

Why this gate exists
--------------------
CLAUDE.md records the original defect: the hand-rolled frontmatter parser
reduced a YAML folded scalar (``description: >``) to the literal ``">"``, which
satisfied the non-empty check **vacuously** -- worse than not checking, because
the gate reported OK over a skill with no description at all.

Block scalars were fixed. Quoting was not. Measured against the validator as it
stood before this change, six spellings still passed with no description::

    description: ""      -> parsed '""'    -> errors []
    description: ''      -> parsed "''"    -> errors []
    description: ">"     -> parsed '">"'   -> errors []
    description: "|"     -> parsed '"|"'   -> errors []
    description: -       -> parsed '-'     -> errors []
    description: "   "   -> parsed '"   "' -> errors []

The value was never unquoted, so any quoted artifact was "non-empty". S1 now
normalises the scalar (trim, unquote once) and rejects the vacuous set.

The normalisation is applied in two places -- in ``_parse_frontmatter`` (so
every key, ``name`` included, is unquoted) and again inside
``_is_vacuous_description``. That is deliberate but it does blunt a naive
negative control: reverting only the parser line leaves the description check
intact and every test below still passes. The load-bearing assertion for the
parser line is therefore ``test_quoted_name_matches_directory``, and the
load-bearing one for the check is the vacuous-description control. Measured:
reverting the S1 check alone fails 2 tests here and prints ``skills: OK`` over
a skill whose description is ``">"``.

Negative control (recorded per the ``negative-control`` skill)
--------------------------------------------------------------
``test_negative_control_vacuous_descriptions_are_rejected`` plants each of the
spellings above -- the folded scalar included -- in a tmp tree and asserts the
validator exits 1 with S1. Verified against the pre-change validator: all six
quoted forms exited 0, so this test fails on the old code and passes on the new,
which is the whole point of writing it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = REPO_ROOT / "scripts" / "validate_skills.py"
_spec = importlib.util.spec_from_file_location("validate_skills_frontmatter", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vs
_spec.loader.exec_module(vs)

# Spellings that must NOT count as a description. Each is a real YAML value a
# hand parser can mistake for content.
VACUOUS_DESCRIPTIONS = (
    'description: ""',
    "description: ''",
    'description: ">"',
    "description: '>'",
    'description: "|"',
    "description: -",
    'description: "   "',
    "description: >",  # folded scalar with no continuation -- the original bug
    "description: |",
    "description: >-",
    "description:",
)

REAL_DESCRIPTIONS = (
    "description: Does a specific thing under specific conditions.",
    'description: "Quoted but real."',
    "description: >\n  A folded description that actually has a body.",
)


def _point_at(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(vs, "REPO_ROOT", root)
    monkeypatch.setattr(vs, "SKILLS_DIR", root / ".claude" / "skills")
    monkeypatch.setattr(vs, "AGENT_SKILLS_DIR", root / ".agents" / "skills")
    pyproject = root / "pyproject.toml"
    pyproject.write_text("[tool.coverage.report]\nfail_under = 85\n", encoding="utf-8")
    monkeypatch.setattr(vs, "PYPROJECT", pyproject)


def _write_skill(root: Path, description_line: str) -> Path:
    skill = root / ".claude" / "skills" / "probe" / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(
        f"---\nname: probe\n{description_line}\nallowed-tools: Read\n---\n\n## Steps\n\n1. x\n",
        encoding="utf-8",
    )
    return skill


# ---------------------------------------------------------------------------
# The real trees
# ---------------------------------------------------------------------------


def _real_skill_files() -> list[Path]:
    files: list[Path] = []
    for tree in (REPO_ROOT / ".claude" / "skills", REPO_ROOT / ".agents" / "skills"):
        files += sorted(tree.glob("*/SKILL.md"))
    return files


def test_every_real_skill_has_a_usable_description() -> None:
    """Both trees, not just the loaded one -- .agents/skills is where the folded
    scalars that started this actually live."""
    files = _real_skill_files()
    assert len(files) >= 10, f"expected both skill trees to be populated, found {len(files)}"

    bad: list[str] = []
    for skill in files:
        front = vs._parse_frontmatter(skill.read_text(encoding="utf-8").splitlines())
        if front is None:
            bad.append(f"{skill}: no frontmatter")
            continue
        description = front.get("description", "")
        if vs._is_vacuous_description(description):
            bad.append(f"{skill}: vacuous description {description!r}")
    assert not bad, "skills with unusable descriptions: " + "; ".join(bad)


def test_real_folded_descriptions_resolve_to_prose() -> None:
    """The `.agents/skills/` folded scalars must parse to their body, not to '>'.

    This is the original bug's exact shape, asserted against the real files
    rather than a fixture, so a regression in the block-scalar branch is caught
    where it happened.
    """
    folded = []
    for skill in _real_skill_files():
        lines = skill.read_text(encoding="utf-8").splitlines()
        if any(line.strip() in {"description: >", "description: |"} for line in lines):
            front = vs._parse_frontmatter(lines)
            assert front is not None
            value = front["description"]
            assert len(value) > 20, f"{skill}: folded description collapsed to {value!r}"
            folded.append(skill)
    assert folded, "no folded-scalar description found -- this assertion has gone stale"


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def test_negative_control_vacuous_descriptions_are_rejected(tmp_path, monkeypatch) -> None:
    """NEGATIVE CONTROL: every vacuous spelling must exit 1 with S1.

    Six of these eleven passed the validator before this change (see the module
    docstring for the measured table), so this test genuinely fails against the
    prior code.
    """
    for line in VACUOUS_DESCRIPTIONS:
        _point_at(monkeypatch, tmp_path)
        _write_skill(tmp_path, line)
        issues: list[vs.Issue] = []
        vs._validate_skill(tmp_path / ".claude" / "skills" / "probe" / "SKILL.md", issues, None)
        codes = [i.code for i in issues if not i.warning]
        assert "S1" in codes, f"{line!r} was accepted as a description (errors={codes})"


def test_real_descriptions_are_not_rejected(tmp_path, monkeypatch) -> None:
    """The other half of the control: the rule must not reject real prose.

    A check that rejects everything is not a check, and a quoted description is
    legal YAML that authors do write.
    """
    for line in REAL_DESCRIPTIONS:
        _point_at(monkeypatch, tmp_path)
        _write_skill(tmp_path, line)
        issues: list[vs.Issue] = []
        vs._validate_skill(tmp_path / ".claude" / "skills" / "probe" / "SKILL.md", issues, None)
        codes = [i.code for i in issues if not i.warning]
        assert not codes, f"{line!r} is a real description but was rejected (errors={codes})"


def test_quoted_name_matches_directory(tmp_path, monkeypatch) -> None:
    """`name: "demo"` is legal YAML and must match the directory `demo`.

    NEGATIVE CONTROL for the ``_parse_frontmatter`` half of the normalisation:
    with the ``_scalar_value`` call removed from the parser, ``name`` stays
    ``'"probe"'``, the S1 name comparison fails, and this test reports it.
    """
    _point_at(monkeypatch, tmp_path)
    skill = tmp_path / ".claude" / "skills" / "probe" / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(
        '---\nname: "probe"\ndescription: Real prose.\nallowed-tools: Read\n---\n\n## Steps\n\n1. x\n',
        encoding="utf-8",
    )
    issues: list[vs.Issue] = []
    vs._validate_skill(skill, issues, None)
    codes = [i.code for i in issues if not i.warning]
    assert not codes, f"a quoted frontmatter name was not unquoted before comparison: {codes}"


def test_negative_control_end_to_end_exit_code(tmp_path, monkeypatch, capsys) -> None:
    """The whole-tree run, not just the per-file helper, must exit 1."""
    _point_at(monkeypatch, tmp_path)
    _write_skill(tmp_path, 'description: ">"')
    assert vs.main([]) == 1
    assert "S1" in capsys.readouterr().err
