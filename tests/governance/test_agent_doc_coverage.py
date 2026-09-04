"""Every ``Agent.md`` that exists is actually read by the validator.

Why this gate exists
--------------------
A5 requires an ``Agent.md`` for every **top-level** package under
``alpha_zero/``, and the whole-tree scan was written to match: it iterated
``PACKAGE_ROOT.iterdir()``. ``alpha_zero/core/mcts/Agent.md`` is one level
deeper, so it was validated by nothing. Measured before the fix, by spying on
``_validate_agent_doc`` during a full ``_validate_all`` run::

    scanned: 31 on disk: 32
    NOT SCANNED:
       alpha_zero/core/mcts/Agent.md

An unvalidated governance document is worse than a missing one. A missing doc
fails A5 loudly; an unscanned doc reports ``agents: OK`` while its heading, its
persona line and its file references drift unchecked -- and it is the document
for the package that was split precisely because it had grown unreviewable.

The requirement and the coverage are deliberately different rules: A5 still only
*requires* a document at the top level (nested subpackages are covered by their
parent, per the validation-gates spec), but any document that exists anywhere
must be correct.

Negative control (recorded per the ``negative-control`` skill)
--------------------------------------------------------------
``test_negative_control_nested_doc_is_not_skipped`` builds a tmp tree whose only
defective document is nested one level down, and asserts the whole-tree run
fails on it. Verified against the pre-change validator: that run exited 0
because the nested document was never opened, so this test fails on the old code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = REPO_ROOT / "scripts" / "validate_agents.py"
_spec = importlib.util.spec_from_file_location("validate_agents_coverage", _SCRIPT)
assert _spec is not None and _spec.loader is not None
va = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = va
_spec.loader.exec_module(va)


def _scanned_docs(monkeypatch) -> set[Path]:
    """Run the whole-tree scan and record which documents were opened."""
    seen: list[Path] = []
    original = va._validate_agent_doc

    def spy(path: Path, issues: list, gate: int | None) -> None:
        seen.append(path)
        return original(path, issues, gate)

    monkeypatch.setattr(va, "_validate_agent_doc", spy)
    issues: list = []
    va._validate_all(issues, va._coverage_gate())
    return {p.resolve() for p in seen}


def _docs_on_disk() -> set[Path]:
    package_docs = set(va.PACKAGE_ROOT.rglob("Agent.md"))
    extras = {va.REPO_ROOT / extra / "Agent.md" for extra in va.EXTRA_AGENT_DIRS}
    return {p.resolve() for p in package_docs | extras if p.is_file()}


def test_every_agent_md_on_disk_is_scanned(monkeypatch) -> None:
    """The gate: no Agent.md may exist outside the validator's reach."""
    scanned = _scanned_docs(monkeypatch)
    on_disk = _docs_on_disk()
    missed = sorted(on_disk - scanned)
    assert not missed, "Agent.md documents that exist but no gate reads: " + ", ".join(
        str(p.relative_to(REPO_ROOT)) for p in missed
    )


def test_the_nested_mcts_doc_is_covered(monkeypatch) -> None:
    """Name the specific document the gap was found in.

    A set-difference assertion goes quiet if the file is deleted; this one says
    which document made the rule necessary, so its removal is a deliberate edit
    rather than a silently satisfied test.
    """
    nested = (REPO_ROOT / "alpha_zero" / "core" / "mcts" / "Agent.md").resolve()
    assert nested.is_file(), "alpha_zero/core/mcts/Agent.md is gone -- update this gate"
    assert nested in _scanned_docs(monkeypatch)


def test_scan_order_is_sorted(monkeypatch) -> None:
    """`rglob` returns filesystem order; the report must not.

    This is the assertion that keeps the ``sorted()`` around the recursive walk
    from being 'tidied away' as redundant.
    """
    seen: list[Path] = []
    original = va._validate_agent_doc
    monkeypatch.setattr(
        va,
        "_validate_agent_doc",
        lambda p, i, g: (seen.append(p), original(p, i, g))[1],
    )
    issues: list = []
    va._validate_all(issues, va._coverage_gate())

    # The scan runs in two phases -- top-level packages (A5), then the recursive
    # sweep for anything deeper -- so the combined list is not globally sorted
    # and asserting that it is would be asserting something the code never
    # promised. Each phase must be sorted internally; the second one is the
    # `rglob` whose order is filesystem-dependent.
    top_level = [p for p in seen if p.parent.parent == va.PACKAGE_ROOT]
    nested = [p for p in seen if va.PACKAGE_ROOT in p.parents and p not in top_level]

    assert top_level == sorted(top_level), (
        f"top-level documents scanned unsorted: {[str(p) for p in top_level]}"
    )
    assert nested == sorted(nested), (
        f"nested documents scanned unsorted -- the sorted() around rglob is gone: "
        f"{[str(p) for p in nested]}"
    )
    # Stated plainly: the real tree holds exactly one nested document, and a
    # one-element list is sorted no matter what the code does. The assertion
    # above therefore cannot fail here today -- `test_scan_order_is_sorted_
    # with_several_nested_docs` is the one that discriminates.
    assert nested, "no nested Agent.md was scanned, so this assertion proves nothing"


def test_scan_order_is_sorted_with_several_nested_docs(tmp_path, monkeypatch) -> None:
    """The discriminating order test: many nested docs, created out of order.

    ``rglob`` walks in filesystem order, which on this repo is demonstrably not
    alphabetical::

        >>> r = list(Path('alpha_zero').rglob('Agent.md'))
        >>> r == sorted(r)
        False

    The directories below are created in reverse-alphabetical order so the
    walk is unlikely to hand them back sorted by accident.

    NEGATIVE CONTROL: with ``sorted()`` removed from the recursive sweep in
    ``_validate_all``, this test fails and reports the unsorted sequence.
    """
    _point_at(monkeypatch, tmp_path)
    core = tmp_path / "alpha_zero" / "core"
    core.mkdir(parents=True)
    (core / "__init__.py").write_text("", encoding="utf-8")
    (core / "Agent.md").write_text(_COMPLETE_DOC.format(dotted="alpha_zero.core"), encoding="utf-8")

    for name in ("zeta", "yankee", "xray", "whiskey", "victor", "uniform"):
        sub = core / name
        sub.mkdir()
        (sub / "__init__.py").write_text("", encoding="utf-8")
        (sub / "Agent.md").write_text(
            _COMPLETE_DOC.format(dotted=f"alpha_zero.core.{name}"), encoding="utf-8"
        )

    seen: list[Path] = []
    original = va._validate_agent_doc
    monkeypatch.setattr(
        va, "_validate_agent_doc", lambda p, i, g: (seen.append(p), original(p, i, g))[1]
    )
    va._validate_all([], va._coverage_gate())

    nested = [p for p in seen if p.parent.parent != va.PACKAGE_ROOT]
    assert len(nested) == 6, f"expected six nested documents, scanned {len(nested)}"
    assert nested == sorted(nested), (
        f"nested documents scanned in filesystem order rather than sorted order: "
        f"{[p.parent.name for p in nested]}"
    )


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------


def _point_at(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(va, "REPO_ROOT", root)
    monkeypatch.setattr(va, "PACKAGE_ROOT", root / "alpha_zero")
    pyproject = root / "pyproject.toml"
    pyproject.write_text("[tool.coverage.report]\nfail_under = 85\n", encoding="utf-8")
    monkeypatch.setattr(va, "PYPROJECT", pyproject)
    monkeypatch.setattr(va, "EXTRA_AGENT_DIRS", ())


_COMPLETE_DOC = """\
# Agent.md — {dotted}

## Persona

You are the **Owner**.

## Module Purpose

Purpose.

## Design Patterns

Patterns.

## Key Files

## Skills

## Sub-Agents

## Tools

## Invariants

## Dependencies
"""


def test_negative_control_nested_doc_is_not_skipped(tmp_path, monkeypatch, capsys) -> None:
    """NEGATIVE CONTROL: a defect in a NESTED doc must fail the whole-tree run.

    The top-level document is valid, so the only thing that can fail the run is
    the nested one -- whose H1 names a module directory that does not exist (A1).
    Against the pre-change validator this run exited 0.
    """
    _point_at(monkeypatch, tmp_path)
    pkg = tmp_path / "alpha_zero" / "core"
    nested = pkg / "mcts"
    nested.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (nested / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "Agent.md").write_text(_COMPLETE_DOC.format(dotted="alpha_zero.core"), encoding="utf-8")
    (nested / "Agent.md").write_text(
        _COMPLETE_DOC.format(dotted="alpha_zero.core.does_not_exist"), encoding="utf-8"
    )

    assert va.main(["--quiet"]) == 1, (
        "a broken nested Agent.md did not fail the run -- the nested scan is not reaching it"
    )
    assert "A1" in capsys.readouterr().err


def test_negative_control_valid_nested_doc_still_passes(tmp_path, monkeypatch) -> None:
    """The other half: scanning nested docs must not fail correct ones."""
    _point_at(monkeypatch, tmp_path)
    pkg = tmp_path / "alpha_zero" / "core"
    nested = pkg / "mcts"
    nested.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (nested / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "Agent.md").write_text(_COMPLETE_DOC.format(dotted="alpha_zero.core"), encoding="utf-8")
    (nested / "Agent.md").write_text(
        _COMPLETE_DOC.format(dotted="alpha_zero.core.mcts"), encoding="utf-8"
    )
    assert va.main(["--quiet"]) == 0
