"""The ruff version is pinned in three places; they must agree.

- ``pyproject.toml``            -> ``[project.optional-dependencies].dev`` ``ruff==X``
- ``.github/workflows/ci.yml``  -> governance job ``pip install ruff==X pytest``
- ``.pre-commit-config.yaml``   -> ``astral-sh/ruff-pre-commit`` ``rev: vX``

A Dependabot bump that touches one of the three leaves local hooks, the lint
job and the governance job disagreeing about what "clean" means, which is how
a file can pass pre-commit and fail CI (or the reverse). Spec WS4.7 requires
the bump to land in all three at once; this test is the gate for that.

Stdlib only (``tomllib`` is 3.11+, which ``requires-python`` already demands;
the two YAML files are read with anchored regexes rather than a YAML parser
because PyYAML is not a governance-job dependency).

Negative control (recorded 2026-09-02; ``.pre-commit-config.yaml`` ``rev``
temporarily edited to ``v0.15.9``, then restored)::

    FAILED tests/governance/test_tool_pins_agree.py::test_ruff_pins_agree -
    AssertionError: ruff pins disagree: .pre-commit-config.yaml=0.15.9
    (pyproject.toml=0.15.8, .github/workflows/ci.yml=0.15.8)
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"

# `pip install ruff==0.15.8 pytest` -- the version token after `ruff==`.
_CI_PIN = re.compile(r"pip install\b[^\n]*\bruff==([0-9][0-9A-Za-z.]*)")
# The `rev:` line belonging to the ruff-pre-commit repo block specifically,
# not whichever `rev:` happens to come first in the file.
_PRE_COMMIT_PIN = re.compile(
    r"repo:\s*https://github\.com/astral-sh/ruff-pre-commit\s*\n\s*rev:\s*v?([0-9][0-9A-Za-z.]*)"
)


def _pyproject_pin() -> str:
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    dev = data["project"]["optional-dependencies"]["dev"]
    pins = [m.group(1) for req in dev if (m := re.fullmatch(r"ruff==([^\s;]+)", req.strip()))]
    assert len(pins) == 1, f"expected exactly one `ruff==` pin in [dev], found {pins}"
    return pins[0]


def _ci_pin() -> str:
    matches = _CI_PIN.findall(CI_WORKFLOW.read_text(encoding="utf-8"))
    assert len(matches) == 1, (
        f"expected exactly one `pip install ruff==` in ci.yml, found {matches}"
    )
    return matches[0]


def _pre_commit_pin() -> str:
    match = _PRE_COMMIT_PIN.search(PRE_COMMIT.read_text(encoding="utf-8"))
    assert match is not None, "no astral-sh/ruff-pre-commit `rev:` found in .pre-commit-config.yaml"
    return match.group(1)


def test_ruff_pins_agree() -> None:
    pins = {
        "pyproject.toml": _pyproject_pin(),
        ".github/workflows/ci.yml": _ci_pin(),
        ".pre-commit-config.yaml": _pre_commit_pin(),
    }
    # Majority value is the reference; name every file that departs from it so
    # the failure says which pin to fix rather than merely that they differ.
    counts = {v: sum(1 for p in pins.values() if p == v) for v in set(pins.values())}
    reference = max(counts, key=lambda v: (counts[v], v))
    outliers = {name: v for name, v in pins.items() if v != reference}
    agreeing = ", ".join(f"{n}={v}" for n, v in pins.items() if v == reference)
    assert not outliers, (
        "ruff pins disagree: "
        + ", ".join(f"{n}={v}" for n, v in outliers.items())
        + f" ({agreeing})"
    )


def test_each_pin_is_a_concrete_version() -> None:
    """A range or a bare `ruff` would 'agree' with nothing and pin nothing."""
    for name, value in {
        "pyproject.toml": _pyproject_pin(),
        ".github/workflows/ci.yml": _ci_pin(),
        ".pre-commit-config.yaml": _pre_commit_pin(),
    }.items():
        assert re.fullmatch(r"\d+\.\d+\.\d+", value), f"{name}: `{value}` is not an exact X.Y.Z"
