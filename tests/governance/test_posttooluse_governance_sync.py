"""Ensure ``posttooluse_governance.sh`` stays aligned with doc-sync triggers."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / ".claude" / "hooks" / "posttooluse_governance.sh"
DOC_SYNC_ARM_RE = re.compile(
    r'^\s*([^\n]+)\)\s*\n(?:[ \t]*#[^\n]*\n)*[ \t]*"\$PY" scripts/check_doc_sync\.py',
    re.MULTILINE,
)


def _doc_sync_patterns() -> list[str]:
    text = HOOK_PATH.read_text(encoding="utf-8")
    matches = DOC_SYNC_ARM_RE.findall(text)
    assert len(matches) == 1, (
        f"expected 1 doc-sync case arm in posttooluse_governance.sh, got {len(matches)}"
    )
    return [part.strip() for part in matches[0].split("|") if part.strip()]


def _matches_any(rel_path: str, patterns: list[str]) -> bool:
    posix = PurePosixPath(rel_path)
    for pat in patterns:
        if pat.endswith("/*") and rel_path.startswith(pat[:-1]):
            return True
        if posix.match(pat):
            return True
    return False


def test_posttooluse_hook_drops_deleted_setup_py() -> None:
    patterns = _doc_sync_patterns()
    assert "setup.py" not in patterns


def test_posttooluse_hook_covers_all_doc_sync_and_release_files() -> None:
    patterns = _doc_sync_patterns()
    expected = (
        "CLAUDE.md",
        "README.md",
        "ROADMAP.md",
        "PROJECT_CHARTER.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "SUPPORT.md",
        "SECURITY.md",
        "GOVERNANCE.md",
        "CITATION.cff",
        "docs/README.md",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "alpha_zero/__init__.py",
        "alpha_zero/cloud/__init__.py",
    )
    missing = [path for path in expected if not _matches_any(path, patterns)]
    assert not missing, f"posttooluse_governance.sh missing doc-sync paths: {missing}"
