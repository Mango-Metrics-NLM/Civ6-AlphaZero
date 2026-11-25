"""What counts as a reference to a skill: one rule for two validators.

``scripts/validate_skills.py`` (rule S6) and ``scripts/validate_claude_agents.py``
(rule C6) both check that a skill a document names exists. They used to carry their
own patterns, and C6's (one space, singular) did not see "`x`" at the end of a line
followed by "skill" on the next, or "the `x` skills". Both import this module.

A reference is a code span holding a skill-shaped name (lowercase letters, digits and
hyphens) followed by the word "skill" or "skills", with any whitespace, line breaks
included, between them. A name in ``BUILTIN_SKILLS`` always exists: those are Claude
Code's own skills, which have no directory in this repository (``pre-pr-handoff``
names ``code-review``). Extend the list deliberately; never loosen the pattern.

Stdlib only, like the validators that import it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

BUILTIN_SKILLS = frozenset(
    {
        "claude-api",
        "code-review",
        "fewer-permission-prompts",
        "init",
        "keybindings-help",
        "loop",
        "review",
        "security-review",
        "simplify",
        "update-config",
    }
)

SKILL_REF_RE = re.compile(r"`([a-z0-9][a-z0-9-]*)`\s+skills?\b")


def skill_references(text: str) -> list[tuple[int, str]]:
    """``(line, name)`` for every skill reference in ``text``, in order."""
    return [
        (text.count("\n", 0, match.start()) + 1, match.group(1))
        for match in SKILL_REF_RE.finditer(text)
    ]


def known_skills(trees: Iterable[Path]) -> set[str]:
    """The built-in skills plus every ``<tree>/<name>/SKILL.md`` under ``trees``."""
    names = set(BUILTIN_SKILLS)
    for tree in trees:
        if tree.is_dir():
            names |= {p.parent.name for p in tree.glob("*/SKILL.md")}
    return names


def unknown_skill_references(text: str, known: set[str]) -> list[tuple[int, str]]:
    """The references in ``text`` whose name is not in ``known``."""
    return [(line, name) for line, name in skill_references(text) if name not in known]
