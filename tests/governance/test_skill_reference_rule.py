"""One rule decides what a skill reference is, for validate_skills S6 and validate_claude_agents C6.

Both validators read "the `x` skill" as a reference to skill ``x`` and accept Claude
Code's built-in skills. They used to carry their own patterns: C6's required a single
space and the singular, so "`x`\\nskill" and "`x` skills" were a reference to S6 and not
to C6. Each sample below is fed to both, against the same skill tree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{name}_ruletest", REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vs = _load("validate_skills")
vca = _load("validate_claude_agents")

SAMPLES = {
    "Follow the `real-skill` skill.": [],
    "Follow the `ghost-skill` skill.": ["ghost-skill"],
    "Follow the `ghost-break`\nskill.": ["ghost-break"],
    "Use the `ghost-a` and `ghost-b` skills.": ["ghost-b"],
    "Run the `code-review` skill on the diff.": [],
    "The `ghost-word` skillset is prose.": [],
    "Plain ghost-plain skill, not code.": [],
}


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    skills = tmp_path / ".claude" / "skills"
    (skills / "real-skill").mkdir(parents=True)
    (skills / "real-skill" / "SKILL.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(vs, "SKILLS_DIR", skills)
    monkeypatch.setattr(vs, "AGENT_SKILLS_DIR", tmp_path / ".agents" / "skills")
    monkeypatch.setattr(vca, "SKILLS_DIR", skills)
    return tmp_path


@pytest.mark.parametrize("text", sorted(SAMPLES))
def test_both_validators_agree(tree: Path, text: str) -> None:
    s6 = vs.unknown_skill_names(text)
    c6 = vca.unknown_skill_names(text)
    assert s6 == c6 == SAMPLES[text]


def test_both_use_the_one_allowlist() -> None:
    assert vs.BUILTIN_SKILLS is vca.BUILTIN_SKILLS
    assert "code-review" in vs.BUILTIN_SKILLS
