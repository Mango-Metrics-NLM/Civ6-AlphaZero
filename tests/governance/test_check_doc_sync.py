"""Deterministic tests for ``scripts/check_doc_sync.py`` using fixture repos."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_doc_sync.py"
_spec = importlib.util.spec_from_file_location("check_doc_sync", _SCRIPT)
assert _spec is not None and _spec.loader is not None
cds = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cds
_spec.loader.exec_module(cds)

PYPROJECT = """\
[project]
name = "demo"
version = "1.0.0"

[project.optional-dependencies]
dev = ["pytest"]
web = ["fastapi"]

[tool.coverage.report]
fail_under = 85
"""

MODULE_REFERENCE = """\
# Module Reference

- **Submodule count:** 2
- **Counting method:** recursive `*.py`

| Module | Files | Responsibility |
|--------|------:|----------------|
| `core` | 2 | core stuff |
| `utils` | 1 | util stuff |
"""

README = """\
# Demo

Install with `pip install -e '.[dev]'`.

See the [license](LICENSE).
"""

CHANGELOG = "# Changelog\n\n## [Unreleased]\n\n## [1.0.0] - 2026-01-01\n\n- initial\n"


def _codes(root: Path) -> set[str]:
    return {issue.code for issue in cds.run(root)}


def build_valid_repo(root: Path) -> Path:
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (root / "alpha_zero" / "core").mkdir(parents=True)
    (root / "alpha_zero" / "core" / "__init__.py").write_text("", encoding="utf-8")
    (root / "alpha_zero" / "core" / "net.py").write_text("", encoding="utf-8")
    (root / "alpha_zero" / "utils").mkdir(parents=True)
    (root / "alpha_zero" / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (root / "alpha_zero" / "__init__.py").write_text('__version__ = "1.0.0"\n', encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "MODULE_REFERENCE.md").write_text(MODULE_REFERENCE, encoding="utf-8")
    (root / "README.md").write_text(README, encoding="utf-8")
    (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (root / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    return root


def test_valid_repo_passes(tmp_path):
    build_valid_repo(tmp_path)
    assert cds.run(tmp_path) == []


def test_d1_version_drift(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "alpha_zero" / "__init__.py").write_text(
        '__version__ = "0.9.0"\n', encoding="utf-8"
    )
    assert "D1" in _codes(tmp_path)


def test_d2_module_count_drift(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "alpha_zero" / "core" / "extra.py").write_text("", encoding="utf-8")
    assert "D2" in _codes(tmp_path)


def test_d2_undocumented_module(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "alpha_zero" / "newmod").mkdir()
    (tmp_path / "alpha_zero" / "newmod" / "__init__.py").write_text("", encoding="utf-8")
    assert "D2" in _codes(tmp_path)


def test_d3_coverage_gate_drift(tmp_path):
    build_valid_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(README + "\nThe gate (80%) applies.\n", encoding="utf-8")
    assert "D3" in _codes(tmp_path)


def test_d3_historical_prose_not_flagged(tmp_path):
    build_valid_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        README + "\nWe re-tightened from 80% to 85% coverage last year.\n", encoding="utf-8"
    )
    assert "D3" not in _codes(tmp_path)


def test_d4_forbidden_readme_pattern(tmp_path):
    build_valid_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(README + "\nRun `pip install wandb` to track runs.\n", encoding="utf-8")
    assert "D4" in _codes(tmp_path)


def test_d4_allow_marker_exempts_line(tmp_path):
    build_valid_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(README + "\nLegacy wandb note. <!-- doc-sync: allow -->\n", encoding="utf-8")
    assert "D4" not in _codes(tmp_path)


def test_d5_unknown_extra(tmp_path):
    build_valid_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(README + "\nAlso `pip install -e '.[rag]'`.\n", encoding="utf-8")
    assert "D5" in _codes(tmp_path)


def test_d5_ignores_archived_docs(tmp_path):
    build_valid_repo(tmp_path)
    archive = tmp_path / "docs" / "archive"
    archive.mkdir()
    (archive / "old.md").write_text("`pip install -e '.[nonexistent]'`\n", encoding="utf-8")
    assert "D5" not in _codes(tmp_path)


def test_d6_broken_link(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "LICENSE").unlink()
    assert "D6" in _codes(tmp_path)


def test_d7_changelog_version_mismatch(tmp_path):
    build_valid_repo(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG.replace("## [1.0.0]", "## [0.9.0]"), encoding="utf-8")
    assert "D7" in _codes(tmp_path)


# ---------------------------------------------------------------------------
# D4b / D6' / D8 / D9 / D1 (CITATION.cff) — added for the public-release gate.
# ---------------------------------------------------------------------------

DOCS_INDEX = """\
# Documentation Index

| Doc | Purpose |
|-----|---------|
| [`GUIDE.md`](GUIDE.md) | A guide |
| [`MODULE_REFERENCE.md`](MODULE_REFERENCE.md) | Module inventory |
| [`sub/NOTES.md`](sub/NOTES.md) | Notes |
| [`archive/README.md`](archive/README.md) | Archive |
"""

ROADMAP = """\
# Roadmap

## Now

- Ship the index gate (#12)
- Rebuild the docs index (`openspec/changes/public-release-readiness`)

## Next

- A bullet with no reference is fine here
"""


def build_indexed_docs(root: Path) -> Path:
    """Add a docs tree whose index links every non-archive doc exactly once."""
    docs = root / "docs"
    (docs / "sub").mkdir(parents=True, exist_ok=True)
    (docs / "archive").mkdir(parents=True, exist_ok=True)
    (docs / "GUIDE.md").write_text("# Guide\n", encoding="utf-8")
    (docs / "sub" / "NOTES.md").write_text("# Notes\n", encoding="utf-8")
    (docs / "archive" / "README.md").write_text("# Archive\n", encoding="utf-8")
    (docs / "archive" / "OLD.md").write_text("# Old, unindexed on purpose\n", encoding="utf-8")
    (docs / "README.md").write_text(DOCS_INDEX, encoding="utf-8")
    return docs


def test_full_fixture_passes(tmp_path):
    build_valid_repo(tmp_path)
    build_indexed_docs(tmp_path)
    (tmp_path / "ROADMAP.md").write_text(ROADMAP, encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text('version: "1.0.0"\n', encoding="utf-8")
    (tmp_path / "PROJECT_CHARTER.md").write_text("See [the guide](docs/GUIDE.md).\n")
    assert cds.run(tmp_path) == []


# D4b ----------------------------------------------------------------------


def test_d4b_forbidden_claim_in_readme(tmp_path):
    build_valid_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(README + "\nThe pipeline is production-ready.\n", encoding="utf-8")
    assert "D4b" in _codes(tmp_path)


def test_d4b_forbidden_claim_in_roadmap(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "ROADMAP.md").write_text(
        "# Roadmap\n\n- Training convergence validated after the fix.\n", encoding="utf-8"
    )
    assert "D4b" in _codes(tmp_path)


def test_d4b_speedup_figures_both_spellings(tmp_path):
    build_valid_repo(tmp_path)
    readme = tmp_path / "README.md"
    for claim in ("13x faster", "a 5\u201310\u00d7 speed-up", "5-10x throughput"):
        readme.write_text(README + f"\nWe measured {claim}.\n", encoding="utf-8")
        assert "D4b" in _codes(tmp_path), claim


def test_d4b_allow_marker_exempts_line(tmp_path):
    build_valid_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(README + "\nproduction-ready <!-- doc-sync: allow -->\n", encoding="utf-8")
    assert "D4b" not in _codes(tmp_path)


def test_d4b_does_not_scan_other_docs(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "CONTRIBUTING.md").write_text("production-ready prose here\n", encoding="utf-8")
    assert "D4b" not in _codes(tmp_path)


# D6' ---------------------------------------------------------------------


def test_d6_dead_link_in_charter(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "PROJECT_CHARTER.md").write_text(
        "See [next steps](NEXT_STEPS.md).\n", encoding="utf-8"
    )
    assert "D6" in _codes(tmp_path)


def test_d6_governance_files_are_checked(tmp_path):
    build_valid_repo(tmp_path)
    for name in ("ROADMAP.md", "CONTRIBUTING.md", "SUPPORT.md", "SECURITY.md", "GOVERNANCE.md"):
        (tmp_path / name).write_text(f"[dead]({name}.missing.md)\n", encoding="utf-8")
        assert "D6" in _codes(tmp_path), name
        (tmp_path / name).unlink()


def test_d6_skips_anchors_http_and_mailto(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "SECURITY.md").write_text(
        "[a](#reporting) [b](https://example.org/x) [c](mailto:x@example.org)\n",
        encoding="utf-8",
    )
    assert "D6" not in _codes(tmp_path)


# D8 ----------------------------------------------------------------------


def test_d8_unindexed_doc(tmp_path):
    build_valid_repo(tmp_path)
    build_indexed_docs(tmp_path)
    (tmp_path / "docs" / "sub" / "ORPHAN.md").write_text("# Orphan\n", encoding="utf-8")
    issues = [i for i in cds.run(tmp_path) if i.code == "D8"]
    assert len(issues) == 1 and "ORPHAN.md" in issues[0].message


def test_d8_doc_indexed_twice(tmp_path):
    build_valid_repo(tmp_path)
    docs = build_indexed_docs(tmp_path)
    (docs / "README.md").write_text(
        DOCS_INDEX + "| [`GUIDE.md`](GUIDE.md#usage) | Linked again |\n", encoding="utf-8"
    )
    issues = [i for i in cds.run(tmp_path) if i.code == "D8"]
    assert len(issues) == 1 and "2 times" in issues[0].message


def test_d8_archive_and_index_itself_are_exempt(tmp_path):
    build_valid_repo(tmp_path)
    build_indexed_docs(tmp_path)
    # archive/OLD.md is unindexed by construction and docs/README.md links itself nowhere.
    assert "D8" not in _codes(tmp_path)


def test_d8_uses_git_tracking_when_available(tmp_path):
    import shutil
    import subprocess

    if shutil.which("git") is None:  # pragma: no cover - environment-dependent
        import pytest

        pytest.skip("git not on PATH")
    build_valid_repo(tmp_path)
    build_indexed_docs(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "docs"], check=True)
    # An untracked scratch file must not count ...
    (tmp_path / "docs" / "SCRATCH.md").write_text("# scratch\n", encoding="utf-8")
    assert "D8" not in _codes(tmp_path)
    # ... until it is tracked.
    subprocess.run(["git", "-C", str(tmp_path), "add", "docs/SCRATCH.md"], check=True)
    assert "D8" in _codes(tmp_path)


# D9 ----------------------------------------------------------------------


def test_d9_now_bullet_without_reference(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "ROADMAP.md").write_text(
        ROADMAP.replace("- Ship the index gate (#12)", "- Ship the index gate"), encoding="utf-8"
    )
    issues = [i for i in cds.run(tmp_path) if i.code == "D9"]
    assert len(issues) == 1 and issues[0].line == 5


def test_d9_accepts_issue_or_openspec_reference(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "ROADMAP.md").write_text(ROADMAP, encoding="utf-8")
    assert "D9" not in _codes(tmp_path)


def test_d9_only_scans_the_now_section(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "ROADMAP.md").write_text(
        "# Roadmap\n\n## Next\n\n- no reference\n\n## Now\n\n- ok (#1)\n\n## Later\n\n- none\n",
        encoding="utf-8",
    )
    assert "D9" not in _codes(tmp_path)


# D1 (CITATION.cff) -------------------------------------------------------


def test_d1_citation_cff_version_drift(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "CITATION.cff").write_text("cff-version: 1.2.0\nversion: 0.9.0\n", encoding="utf-8")
    issues = [i for i in cds.run(tmp_path) if i.code == "D1"]
    assert len(issues) == 1 and "CITATION.cff" in issues[0].message


def test_d1_citation_cff_version_match(tmp_path):
    build_valid_repo(tmp_path)
    (tmp_path / "CITATION.cff").write_text("cff-version: 1.2.0\nversion: 1.0.0\n", encoding="utf-8")
    assert "D1" not in _codes(tmp_path)
