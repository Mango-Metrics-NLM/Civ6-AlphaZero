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
