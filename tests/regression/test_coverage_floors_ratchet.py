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
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

FLOORS_PATH = Path("coverage-floors.json")
# Branches the base may live on when the event gives no better answer: the
# repository's default branch today and the name it is being renamed to.
_DEFAULT_BRANCH_CANDIDATES = (
    "claude/civ6-alphazero-implementation-vMnun",
    "main",
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


def _git(*args: str) -> str | None:
    """stdout of a git command, or ``None`` when it fails or prints nothing."""
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    out = result.stdout.strip()
    return out if result.returncode == 0 and out else None


def _committed_at(ref: str) -> dict[str, Any] | None:
    shown = _git("show", f"{ref}:{FLOORS_PATH}")
    if shown is None:
        return None
    try:
        return _load(shown)
    except json.JSONDecodeError:
        return None


def _candidate_refs() -> list[str]:
    """Where the base branch may be, most specific first.

    On a ``pull_request`` event GitHub sets ``GITHUB_BASE_REF`` to the branch
    the PR targets; that is the right base by definition and comes first. Then
    ``origin/HEAD`` (set by a normal clone, not by actions/checkout), then the
    known default-branch names.
    """
    refs: list[str] = []
    base_ref = os.environ.get("GITHUB_BASE_REF", "").strip()
    if base_ref:
        refs.append(f"origin/{base_ref}")
    refs.append("origin/HEAD")
    refs.extend(f"origin/{branch}" for branch in _DEFAULT_BRANCH_CANDIDATES)
    return refs


def _comparison_point(ref: str) -> tuple[str, str] | None:
    """The commit to compare the working tree against, derived from ``ref``.

    Normally ``git merge-base HEAD ref``: the fork point of this branch, so a
    PR is judged against what it branched from rather than against a base that
    has since moved. When HEAD *is* the base (a push to the default branch, or
    a local checkout sitting on it) the merge base is HEAD itself and the
    comparison would be vacuous, so the previous commit is used instead -- for
    a push event that is exactly the pre-change state.
    """
    if _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}") is None:
        return None
    base = _git("merge-base", "HEAD", ref)
    head = _git("rev-parse", "HEAD")
    if base is None or head is None:
        return None
    if base == head:
        parent = _git("rev-parse", "--verify", "--quiet", "HEAD~1")
        if parent is None:
            return None
        return parent, f"{ref} (HEAD is the base; comparing with HEAD~1 {parent[:10]})"
    return base, f"{ref} (merge base {base[:10]})"


def _resolve_base() -> tuple[str, dict[str, Any]] | None:
    for ref in _candidate_refs():
        point = _comparison_point(ref)
        if point is None:
            continue
        sha, label = point
        doc = _committed_at(sha)
        if doc is not None:
            return label, doc
    return None


def _no_base_reason() -> str:
    """Why no base resolved -- spelled out, so a skip is never read as a pass."""
    shallow = _git("rev-parse", "--is-shallow-repository") == "true"
    tried = ", ".join(_candidate_refs())
    return (
        "coverage-floor ratchet has no base revision to compare against "
        f"(tried {tried}; GITHUB_BASE_REF={os.environ.get('GITHUB_BASE_REF') or 'unset'}; "
        f"shallow clone={shallow}). In CI this means actions/checkout ran without "
        "`fetch-depth: 0`; locally, run `git fetch origin`."
    )


def _base_or_bail() -> tuple[str, dict[str, Any]]:
    """Resolve the base, or stop with the reason.

    Inside GitHub Actions an unresolvable base is a *misconfigured checkout*
    (the ci.yml test and coverage jobs set ``fetch-depth: 0`` for this test),
    so it fails rather than skips: a ratchet that skips on every CI run has
    only ever run on developers' machines. Locally it skips, with the reason.
    """
    base = _resolve_base()
    if base is not None:
        return base
    reason = _no_base_reason()
    if os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.fail(reason)
    pytest.skip(reason)


def test_floors_file_exists_and_parses() -> None:
    assert FLOORS_PATH.exists(), f"{FLOORS_PATH} is missing; the floor gate has nothing to read"
    assert _minimums(_load(FLOORS_PATH.read_text(encoding="utf-8"))), (
        f"{FLOORS_PATH} declares no package minimums, so every other assertion here is vacuous"
    )


def test_no_floor_was_lowered() -> None:
    """A recorded minimum may rise or hold. Lowering one needs a different PR."""
    ref, base_doc = _base_or_bail()

    current = _minimums(_load(FLOORS_PATH.read_text(encoding="utf-8")))
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
    ref, base_doc = _base_or_bail()

    current = set(_minimums(_load(FLOORS_PATH.read_text(encoding="utf-8"))))
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
    ref, doc = _base_or_bail()
    assert _minimums(doc), (
        f"{ref}:{FLOORS_PATH} parsed but declares no minimums, so the comparison "
        "above would pass vacuously"
    )


def test_pull_request_base_ref_is_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a PR event the branch the PR targets is the base, before any guess.

    Positive and negative halves in one place: with ``GITHUB_BASE_REF`` set it
    must lead the candidate list; without it, it must not appear at all (an
    empty value must not produce a bogus ``origin/`` candidate).
    """
    monkeypatch.setenv("GITHUB_BASE_REF", "release/next")
    assert _candidate_refs()[0] == "origin/release/next"
    monkeypatch.setenv("GITHUB_BASE_REF", "")
    assert "origin/" not in _candidate_refs()
    assert all(not ref.endswith("origin/") for ref in _candidate_refs())
    monkeypatch.delenv("GITHUB_BASE_REF")
    assert _candidate_refs()[0] == "origin/HEAD"


def test_a_missing_base_is_a_failure_in_ci_and_a_skip_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The skip must carry its reason, and CI must not be allowed to skip.

    Simulated by making every candidate unresolvable; the real resolver is left
    untouched so the message exercised here is the one a reader would see.
    """
    monkeypatch.setattr("tests.regression.test_coverage_floors_ratchet._resolve_base", lambda: None)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with pytest.raises(pytest.fail.Exception) as failed:
        _base_or_bail()
    assert "fetch-depth: 0" in str(failed.value)
    monkeypatch.delenv("GITHUB_ACTIONS")
    with pytest.raises(pytest.skip.Exception) as skipped:
        _base_or_bail()
    assert "no base revision" in str(skipped.value)
