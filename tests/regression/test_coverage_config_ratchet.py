"""Coverage configuration invariants: the omit list is a ratchet with reasons.

Three ways the coverage gate has been quietly weakened in this repo's history,
each now a test:

1. **The two omit lists drifting apart.** ``[tool.coverage.run].omit`` and
   ``[tool.coverage.report].omit`` are maintained as a mirrored pair; an entry
   added to one and not the other measures a file but hides it from the report,
   or the reverse. Nothing checked that they matched.
2. **An omit entry with no reason.** Every entry is a decision to stop measuring
   a file. Without a comment saying why, the list only grows, and its rationale
   rots: the "require GCP SDK" comment on ``alpha_zero/cloud/*`` was measured
   false on 2026-08-28, two years after it was written.
3. **A stale entry.** An omit pattern matching no file is either a typo (so the
   file it meant to omit is being measured, or is not) or a leftover from a
   deletion. Either way it is not doing what it says.

Negative control (recorded 2026-09-02): adding ``"alpha_zero/utils/config.py"``
to the run list only failed :func:`test_omit_lists_are_identical`; adding it to
both without a trailing comment failed :func:`test_every_omit_entry_has_a_reason`;
adding a pattern for a deleted module failed
:func:`test_no_omit_entry_is_stale`. Removing them restored the pass.
"""

from __future__ import annotations

import fnmatch
import re
import tomllib
from pathlib import Path

import pytest

#: How many omit entries one ``#`` comment may explain. Without a cap, a single
#: comment at the top of the list "explains" all 43 entries, and a comment left
#: behind by a deleted entry silently becomes the reason for whatever follows.
MAX_ENTRIES_PER_REASON = 8

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
FLOORS = REPO_ROOT / "coverage-floors.json"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _omit(pyproject: dict, table: str) -> list[str]:
    return list(pyproject["tool"]["coverage"][table].get("omit", []))


def test_omit_lists_are_identical(pyproject: dict) -> None:
    run = _omit(pyproject, "run")
    report = _omit(pyproject, "report")
    assert run == report, (
        "[tool.coverage.run].omit and [tool.coverage.report].omit must stay identical; "
        f"only in run: {sorted(set(run) - set(report))}, only in report: {sorted(set(report) - set(run))}"
    )


def test_every_omit_entry_has_a_reason(pyproject: dict) -> None:
    """Each `alpha_zero/` entry sits under a reason.

    The file's convention is a ``#`` comment heading a group of related
    patterns, so an entry is explained by the nearest comment above it inside
    the ``omit = [`` block (or by a trailing comment of its own). An entry
    appearing before any comment in the block is unexplained.
    """
    lines = PYPROJECT.read_text(encoding="utf-8").splitlines()
    entry = re.compile(r'^\s*"(?P<pattern>[^"]+)",\s*(?P<trailing>#.*)?$')
    in_omit = False
    group_reason: str | None = None
    since_reason = 0
    unexplained: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("omit = ["):
            in_omit, group_reason, since_reason = True, None, 0
            continue
        if not in_omit:
            continue
        if stripped == "]":
            in_omit = False
            continue
        if stripped.startswith("#"):
            group_reason, since_reason = stripped, 0
            continue
        match = entry.match(line)
        if match:
            since_reason += 1
            covered = match.group("trailing") or (
                group_reason is not None and since_reason <= MAX_ENTRIES_PER_REASON
            )
            if match.group("pattern").startswith("alpha_zero/") and not covered:
                unexplained.append(match.group("pattern"))
    assert not unexplained, (
        "coverage omit entries with no reason above them (head the group with a `#` comment, "
        f"or put one on the line): {unexplained}"
    )


def test_no_omit_entry_is_stale(pyproject: dict) -> None:
    """Every `alpha_zero/` omit pattern must match at least one file on disk."""
    package_files = [
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "alpha_zero").rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    stale = [
        pattern
        for pattern in _omit(pyproject, "run")
        if pattern.startswith("alpha_zero/")
        and not any(
            fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(f, "*/" + pattern) for f in package_files
        )
    ]
    assert not stale, (
        f"coverage omit patterns matching no file (typo, or left behind by a deletion): {stale}"
    )


def test_global_gate_is_at_least_85(pyproject: dict) -> None:
    """The documented gate. It may be raised here, never lowered without review."""
    assert pyproject["tool"]["coverage"]["report"]["fail_under"] >= 85


def test_floors_file_is_well_formed_and_ratchets_only_up() -> None:
    """Each floor must be met by the measurement recorded beside it."""
    import json

    data = json.loads(FLOORS.read_text(encoding="utf-8"))
    floors = data["floors"]
    assert floors, "coverage-floors.json declares no floors"
    for package, spec in floors.items():
        assert package.startswith("alpha_zero"), package
        assert 0 < spec["minimum"] <= 100, package
        assert spec["minimum"] <= spec["measured"], (
            f"{package}: floor {spec['minimum']} exceeds the recorded measurement "
            f"{spec['measured']} — the floor would fail on the tree it was set from"
        )
        assert spec.get("measured_by"), f"{package}: record the command that produced 'measured'"
