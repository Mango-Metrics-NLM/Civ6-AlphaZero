"""The ``Agent.md`` WARNING count may go down, never up.

Why this gate exists
--------------------
``validate_agents.py`` emits A3 warnings for the six *conventional* sections a
persona doc is expected to carry. Warnings failed nothing, so the count was free
to grow: a new package could ship a three-section stub and the validator would
say ``agents: OK``. That is a coverage gap dressed as a passing gate.

Two options were available. (a) Promote the codes to errors and write the
missing sections -- but the missing sections are ``## Tools``, ``## Invariants``
and ``## Dependencies`` tables for six packages, i.e. real per-module facts that
have to be researched, and a validator change that forces them invites invented
filler. (b) Keep them advisory and ratchet the count. This suite implements (b),
which is the option the task brief prefers when (a) is a content exercise.

The baseline lives in ``agent_warning_baseline.json`` beside this file so the
number is data with a recorded provenance, not a literal buried in an assert.

Negative control (recorded per the ``negative-control`` skill)
--------------------------------------------------------------
``test_negative_control_ratchet_catches_a_new_warning`` writes a tmp tree whose
``Agent.md`` is missing one more conventional section than the baseline allows
and asserts ``--max-warnings`` exits 1. Verified: with ``--max-warnings`` at the
tree's own count it exits 0, and at one below it exits 1 -- so the flag
discriminates rather than always failing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
BASELINE_FILE = _HERE / "agent_warning_baseline.json"

_SCRIPT = REPO_ROOT / "scripts" / "validate_agents.py"
_spec = importlib.util.spec_from_file_location("validate_agents_ratchet", _SCRIPT)
assert _spec is not None and _spec.loader is not None
va = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = va
_spec.loader.exec_module(va)


def _baseline() -> int:
    data = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    return int(data["max_warnings"])


def _current_warning_count() -> int:
    issues: list[va.Issue] = []
    va._validate_all(issues, va._coverage_gate())
    return sum(1 for i in issues if i.warning)


def test_warning_count_does_not_exceed_baseline() -> None:
    """The gate: no change may add an Agent.md warning."""
    current = _current_warning_count()
    baseline = _baseline()
    assert current <= baseline, (
        f"Agent.md warnings rose from {baseline} to {current}. Add the missing "
        f"conventional sections to the document you touched -- run "
        f"`python scripts/validate_agents.py` to see which."
    )


def test_baseline_is_tightened_when_docs_improve() -> None:
    """The ratchet: an improvement must be banked, not left as slack.

    Without this half, fixing five docs would leave five warnings of headroom
    for the next regression to hide in -- which is how a ratchet decays into a
    generous constant.
    """
    current = _current_warning_count()
    baseline = _baseline()
    assert current == baseline, (
        f"Agent.md warnings dropped to {current} but the baseline is still "
        f"{baseline}. Lower `max_warnings` in {BASELINE_FILE.name} to {current} "
        f"in this same change so the improvement is locked in."
    )


def test_baseline_file_records_its_provenance() -> None:
    """A baseline without a recorded measurement is a magic number."""
    data = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    assert isinstance(data["max_warnings"], int)
    assert data["measured_on"], "baseline must record when it was measured"
    assert data["measured_by"], "baseline must record the command that measured it"


def _write_doc(root: Path, package: str, sections: tuple[str, ...]) -> None:
    pkg = root / "alpha_zero" / package
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    body = [f"# Agent.md — alpha_zero.{package}", "", "## Persona", "", "You are the **Owner**."]
    for section in sections:
        body += ["", section, "", "text"]
    (pkg / "Agent.md").write_text("\n".join(body) + "\n", encoding="utf-8")


def _point_at(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(va, "REPO_ROOT", root)
    monkeypatch.setattr(va, "PACKAGE_ROOT", root / "alpha_zero")
    pyproject = root / "pyproject.toml"
    pyproject.write_text("[tool.coverage.report]\nfail_under = 85\n", encoding="utf-8")
    monkeypatch.setattr(va, "PYPROJECT", pyproject)
    monkeypatch.setattr(va, "EXTRA_AGENT_DIRS", ())


def test_negative_control_ratchet_catches_a_new_warning(tmp_path, monkeypatch) -> None:
    """NEGATIVE CONTROL: ``--max-warnings`` must fail when the count rises.

    The doc below is missing all six conventional sections, so it emits exactly
    six warnings. At ``--max-warnings 6`` the run passes; at 5 it must fail --
    that pair is what shows the flag reads the real count instead of always
    passing or always failing.
    """
    _point_at(monkeypatch, tmp_path)
    _write_doc(tmp_path, "demo", ("## Module Purpose", "## Design Patterns"))

    assert va.main(["--max-warnings", "6", "--quiet"]) == 0, (
        "six warnings under a budget of six should pass; the flag is over-strict"
    )
    assert va.main(["--max-warnings", "5", "--quiet"]) == 1, (
        "seven-or-more warnings under a budget of five must fail; "
        "the --max-warnings ratchet does not discriminate"
    )


def test_negative_control_ratchet_ignores_errorless_clean_tree(tmp_path, monkeypatch) -> None:
    """A fully-sectioned doc emits zero warnings, so the ratchet stays quiet."""
    _point_at(monkeypatch, tmp_path)
    _write_doc(
        tmp_path,
        "demo",
        ("## Module Purpose", "## Design Patterns", *va.CONVENTIONAL_SECTIONS),
    )
    assert va.main(["--max-warnings", "0", "--quiet"]) == 0
