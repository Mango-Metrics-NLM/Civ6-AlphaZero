"""Every validated skill tree is either loadable or explicitly excused.

Why this gate exists
--------------------
``validate_skills.py`` scans two trees: ``.claude/skills/`` and
``.agents/skills/``. Only the first is loaded by the harness -- CLAUDE.md
records the measurement ("this session's skill list contains all nine
``.claude/skills/`` entries and none of its three"), and nothing in
``.claude/settings.json`` references ``.agents/``.

That asymmetry is not itself a bug; deleting the tree is a maintainer call, not
a test's. What *is* a bug is that it lived only as a prose note. A validated
tree that nothing loads looks, to anyone reading the gate output, exactly like a
tree that works -- so an author adds a skill there, watches it pass validation,
and never learns it will not load.

This suite converts that from an undocumented oddity into a gated decision: a
validated tree must be loadable, or it must appear in ``NOT_LOADED`` with a
reason. Adding a third tree to the validator without deciding which it is fails
here.

Negative control (recorded per the ``negative-control`` skill)
--------------------------------------------------------------
``test_negative_control_flags_an_unaccounted_tree`` feeds the same
``_unaccounted`` helper a synthetic third tree that is neither loadable nor
allowlisted, and asserts it is reported. Verified: the synthetic tree is
flagged while the two real trees are not, so the passing test above is not
vacuous.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = REPO_ROOT / "scripts" / "validate_skills.py"
_spec = importlib.util.spec_from_file_location("validate_skills_loadability", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vs
_spec.loader.exec_module(vs)

# The one tree the Claude Code harness actually loads skills from.
LOADABLE_TREES = (REPO_ROOT / ".claude" / "skills",)

# Validated but NOT loaded. Each entry needs a reason, because the point of the
# allowlist is that the exclusion was decided rather than inherited.
NOT_LOADED: dict[Path, str] = {
    REPO_ROOT / ".agents" / "skills": (
        "Gated documentation, not runnable skills. Scanned by validate_skills.py and "
        "routed here by the PostToolUse hook, but no harness surface loads it: the "
        "session skill list contains the .claude/skills/ entries and none of these "
        "three, and .claude/settings.json never names .agents/. Kept deliberately -- "
        "removing it is a maintainer decision, so it is excused here rather than "
        "quietly tolerated."
    ),
}


def _unaccounted(trees: list[Path], allowlist: dict[Path, str]) -> list[Path]:
    """Validated trees that are neither loadable nor excused with a reason."""
    return [
        tree
        for tree in trees
        if tree not in LOADABLE_TREES and not (allowlist.get(tree) or "").strip()
    ]


def test_every_validated_tree_is_loadable_or_allowlisted() -> None:
    """The gate itself, run against the validator's real tree list."""
    trees = vs._skill_trees()
    assert trees, "validate_skills.py scanned no trees at all"
    unaccounted = _unaccounted(trees, NOT_LOADED)
    assert not unaccounted, (
        f"validate_skills.py validates {[str(p) for p in unaccounted]}, which the harness "
        f"does not load and which carries no NOT_LOADED reason. Either make it loadable "
        f"(move it under .claude/skills/) or add it to NOT_LOADED with why."
    )


def test_allowlist_matches_the_trees_that_exist() -> None:
    """A stale allowlist entry is as misleading as a missing one.

    If ``.agents/skills/`` is ever deleted or promoted, this fails and forces
    the note to be removed rather than left to describe a tree that is gone.
    """
    trees = set(vs._skill_trees())
    for excused in NOT_LOADED:
        assert excused in trees, (
            f"{excused} is allowlisted as validated-but-not-loaded, but "
            f"validate_skills.py no longer scans it. Drop the NOT_LOADED entry."
        )


def test_agents_skills_tree_is_not_referenced_by_the_harness_settings() -> None:
    """The evidence behind the allowlist reason, re-checked rather than trusted."""
    settings = REPO_ROOT / ".claude" / "settings.json"
    assert settings.is_file(), "harness settings file is missing"
    assert ".agents/skills" not in settings.read_text(encoding="utf-8"), (
        "`.claude/settings.json` now references .agents/skills -- the tree may have "
        "become loadable, so re-check the NOT_LOADED reason before it goes stale."
    )


def test_negative_control_flags_an_unaccounted_tree(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: the check must fail on a tree nobody decided about.

    A third validated tree with no allowlist entry is precisely the situation
    this suite exists to catch. It must appear in ``_unaccounted``; if it did
    not, the passing test above would hold for any tree list at all.
    """
    rogue = tmp_path / ".somewhere" / "skills"
    rogue.mkdir(parents=True)

    trees = [*vs._skill_trees(), rogue]
    assert rogue in _unaccounted(trees, NOT_LOADED), (
        "_unaccounted ignored an un-allowlisted, non-loadable tree; the gate cannot fail"
    )

    # And an empty-string reason must not count as a decision either.
    assert rogue in _unaccounted(trees, {**NOT_LOADED, rogue: "   "}), (
        "a blank reason was accepted as an excuse; the allowlist must require real text"
    )
    assert rogue not in _unaccounted(trees, {**NOT_LOADED, rogue: "excused because X"})
