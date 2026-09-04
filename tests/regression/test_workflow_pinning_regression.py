"""Workflow supply-chain invariants: pinned actions, and a PR trigger that fires.

Two defects this gates against, both found by the 2026-09-04 standards audit:

1. **Mutable action tags.** 26 of 27 ``uses:`` references were on a moving major
   tag (``actions/checkout@v4``). A tag is repointable by its owner, so a
   compromised or simply changed upstream silently alters what CI executes with
   the repository's token. Pinning to a full commit SHA makes the executed code
   immutable; the trailing ``# vX.Y.Z`` comment keeps it readable and gives
   Dependabot something to bump.

2. **A ``pull_request`` trigger naming a branch that is not the default.**
   ``ci.yml`` filtered on ``branches: [main]`` while this repository's default
   branch is ``claude/civ6-alphazero-implementation-vMnun``. Pull requests into
   the real default branch therefore never produced a ``pull_request``-event
   run. Anything ``pull_request``-only would have been silently dead.

Parsed with ``yaml``, never regex: CLAUDE.md (2026-08-30) records that a hand
parser over YAML is how ``validate_skills`` came to validate a folded block
scalar vacuously.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW_DIR = Path(".github/workflows")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# A local action (`./.github/actions/foo`) has no upstream to pin; a reusable
# workflow reference is covered by the same rule as an action.
LOCAL_PREFIXES = ("./", "../")


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))


def _iter_uses(node: Any) -> list[str]:
    """Every ``uses:`` value anywhere in a parsed workflow, at any nesting depth."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "uses" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_iter_uses(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_iter_uses(item))
    return found


def _triggers(doc: dict) -> dict:
    """Return the ``on:`` block.

    PyYAML resolves the bare key ``on`` to the boolean ``True`` (YAML 1.1), so a
    lookup for the string ``"on"`` silently misses and every assertion built on
    it would pass vacuously.
    """
    if True in doc:
        return doc[True]
    return doc.get("on", {})


def test_workflow_directory_is_not_empty() -> None:
    """Guard against the whole suite passing because it found no files."""
    assert _workflow_files(), "no workflow files found; every other test here would be vacuous"


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_sha(path: Path) -> None:
    """No ``uses:`` reference may sit on a mutable tag or branch."""
    doc = yaml.safe_load(path.read_text())
    unpinned: list[str] = []
    for ref in _iter_uses(doc):
        if ref.startswith(LOCAL_PREFIXES):
            continue
        _, _, version = ref.partition("@")
        if not SHA_RE.match(version):
            unpinned.append(ref)
    assert not unpinned, (
        f"{path.name} references actions by mutable ref: {unpinned}. "
        "Pin to a full 40-character commit SHA with a trailing '# vX.Y.Z' comment."
    )


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_pinned_actions_carry_a_version_comment(path: Path) -> None:
    """A bare SHA is unreadable and unmaintainable; require the version comment."""
    missing = [
        line.strip()
        for line in path.read_text().splitlines()
        if re.search(r"uses:\s*[^\s]+@[0-9a-f]{40}", line) and "#" not in line
    ]
    assert not missing, f"{path.name} has SHA-pinned actions with no '# vX.Y.Z' comment: {missing}"


def test_ci_pull_request_trigger_is_not_filtered_to_a_stale_branch() -> None:
    """A ``branches`` filter on ``pull_request`` must not exclude the default branch.

    The safest form is no filter at all, so every pull request runs CI. If a
    filter is reintroduced it must name the repository's real default branch,
    which is deliberately read from git rather than hardcoded here -- hardcoding
    it is what produced the original defect.
    """
    doc = yaml.safe_load((WORKFLOW_DIR / "ci.yml").read_text())
    triggers = _triggers(doc)
    assert "pull_request" in triggers, "ci.yml no longer runs on pull_request at all"

    pr = triggers["pull_request"]
    if not pr or "branches" not in pr:
        return  # unfiltered: every PR runs CI, which is what we want

    import subprocess

    head = subprocess.run(
        ["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    default_branch = head.removeprefix("origin/") if head else ""
    if not default_branch:
        pytest.skip("origin/HEAD is not resolvable in this checkout")

    patterns = pr["branches"]
    matched = any(
        default_branch == pattern
        or pattern in ("**", "*")
        or default_branch.startswith(pattern.rstrip("*/"))
        for pattern in patterns
    )
    assert matched, (
        f"ci.yml pull_request filters on {patterns}, which does not include the "
        f"repository default branch {default_branch!r}: PRs into it would run no CI"
    )


def test_the_pinning_check_rejects_a_mutable_tag(tmp_path: Path) -> None:
    """Negative control: the check must fail on a planted mutable reference."""
    planted = tmp_path / "planted.yml"
    planted.write_text(
        "name: X\non:\n  push:\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - uses: actions/checkout@v4\n"
    )
    doc = yaml.safe_load(planted.read_text())
    refs = _iter_uses(doc)
    assert refs == ["actions/checkout@v4"], "the uses-extractor missed a reference"
    assert not SHA_RE.match(refs[0].partition("@")[2]), "a mutable tag was accepted as a SHA"


def test_the_uses_extractor_finds_nested_references() -> None:
    """Negative control for the walker itself: it must not miss deep nesting."""
    doc = yaml.safe_load(
        "jobs:\n  a:\n    steps:\n      - uses: x/y@" + "0" * 40 + "\n"
        "  b:\n    steps:\n      - uses: p/q@" + "1" * 40 + "\n"
    )
    assert len(_iter_uses(doc)) == 2
