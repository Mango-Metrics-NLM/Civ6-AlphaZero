"""``coverage-floors.json`` may be raised, never lowered.

``scripts/check_coverage_floors.py`` asserts only that the *measured* coverage
clears the recorded minimum. Nothing compared the recorded minimum against its
own previous value, so a change lowering ``"minimum": 58`` to ``40`` passed lint,
types, every test suite and the floor check itself. The ratchet was a docstring.

This is the same mechanism ``tests/regression/test_debt_ratchet.py`` already uses
for the tech-debt baseline: read the committed value at the merge base, read it
at the working tree, and refuse a decrease.

Negative control (``.claude/skills/negative-control``): lowering any ``minimum``
in the working tree must fail ``test_no_floor_was_lowered``. Verified by
temporarily setting the ``alpha_zero/cloud`` floor to 40, observing the failure,
and restoring.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

FLOORS_PATH = Path("coverage-floors.json")
# Candidate bases, most specific first. A shallow CI checkout may have none of
# them, which is a skip rather than a pass -- see test_base_resolution_is_honest.
_BASE_CANDIDATES = (
    "origin/HEAD",
    "origin/claude/civ6-alphazero-implementation-vMnun",
    "origin/main",
)


def _load(text: str) -> dict[str, Any]:
    return json.loads(text)


def _minimums(doc: dict[str, Any]) -> dict[str, float]:
    """Map package -> minimum, tolerating every shape the file has used.

    The live file nests under ``"floors"``. ``"packages"`` and a bare top-level
    mapping are accepted too, so a future rename does not silently empty this
    dict -- which would make the ratchet pass vacuously. ``test_..._is_honest``
    is what catches that if it ever happens anyway; it caught this key.
    """
    packages = doc.get("floors") or doc.get("packages") or doc
    out: dict[str, float] = {}
    if isinstance(packages, dict):
        for name, entry in packages.items():
            if isinstance(entry, dict) and "minimum" in entry:
                out[name] = float(entry["minimum"])
    elif isinstance(packages, list):
        for entry in packages:
            if isinstance(entry, dict) and "minimum" in entry:
                out[str(entry.get("package", entry.get("name")))] = float(entry["minimum"])
    return out


def _committed_at(ref: str) -> dict[str, Any] | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{FLOORS_PATH}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return _load(result.stdout)
    except json.JSONDecodeError:
        return None


def _resolve_base() -> tuple[str, dict[str, Any]] | None:
    for ref in _BASE_CANDIDATES:
        doc = _committed_at(ref)
        if doc is not None:
            return ref, doc
    return None


def test_floors_file_exists_and_parses() -> None:
    assert FLOORS_PATH.exists(), f"{FLOORS_PATH} is missing; the floor gate has nothing to read"
    assert _minimums(_load(FLOORS_PATH.read_text())), (
        f"{FLOORS_PATH} declares no package minimums, so every other assertion here is vacuous"
    )


def test_no_floor_was_lowered() -> None:
    """A recorded minimum may rise or hold. Lowering one needs a different PR."""
    base = _resolve_base()
    if base is None:
        pytest.skip("no base revision of coverage-floors.json is reachable in this checkout")
    ref, base_doc = base

    current = _minimums(_load(FLOORS_PATH.read_text()))
    previous = _minimums(base_doc)

    lowered = {
        package: (previous[package], current[package])
        for package in previous
        if package in current and current[package] < previous[package]
    }
    assert not lowered, (
        "coverage floors may only ever be raised. Lowered against "
        f"{ref}: "
        + "; ".join(f"{pkg}: {was} -> {now}" for pkg, (was, now) in sorted(lowered.items()))
        + ". If coverage genuinely dropped, fix the coverage; if the package "
        "shrank, re-measure and say so explicitly in the PR body."
    )


def test_no_package_was_silently_dropped() -> None:
    """Deleting a package's entry evades the ratchet as surely as lowering it."""
    base = _resolve_base()
    if base is None:
        pytest.skip("no base revision of coverage-floors.json is reachable in this checkout")
    ref, base_doc = base

    current = set(_minimums(_load(FLOORS_PATH.read_text())))
    previous = set(_minimums(base_doc))
    removed = previous - current
    assert not removed, (
        f"packages dropped from the floor file against {ref}: {sorted(removed)}. "
        "Removing an entry stops it being measured, which is a silent floor of zero."
    )


def test_base_resolution_is_honest() -> None:
    """A skip must mean 'no base', never 'the base happened to look fine'.

    Without this, a typo in every candidate ref would make the two ratchet tests
    skip forever and read as green -- the fail-open shape this repository has
    been bitten by three times.
    """
    base = _resolve_base()
    if base is None:
        pytest.skip("no base revision reachable; nothing to verify")
    ref, doc = base
    assert _minimums(doc), (
        f"{ref}:{FLOORS_PATH} parsed but declares no minimums, so the comparison "
        "above would pass vacuously"
    )
