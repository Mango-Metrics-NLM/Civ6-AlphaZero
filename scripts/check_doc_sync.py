#!/usr/bin/env python3
"""Deterministic documentation-consistency gate.

Prevents the documentation drift this repository has suffered from recurring.
Checks (error codes are stable and referenced by ``tests/governance/``):

  D1  version equality: ``pyproject.toml`` ``version`` == ``alpha_zero/__init__.py``
      ``__version__`` (and ``setup.py`` if it declares a version)
  D2  ``docs/MODULE_REFERENCE.md``: the module set and per-module ``*.py`` counts
      match the filesystem, and the declared submodule count matches the table
  D3  coverage-gate consistency: every current-gate declaration (``fail_under``,
      ``--cov-fail-under``, ``gate (N%)``, ``gate: N%``) equals ``pyproject``
      ``fail_under``; historical prose is not matched
  D4  the README contains no forbidden pattern (retired ``wandb`` tooling,
      ``requirements-dev.txt``); a ``<!-- doc-sync: allow -->`` line is exempt
  D4b ``README.md`` and ``ROADMAP.md`` carry no forbidden *claim* pattern
      (``FORBIDDEN_CLAIM_PATTERNS``: unmeasured speed-ups, "production-ready",
      "convergence validated", ...); the allow marker exempts a line
  D5  every ``pip install -e '.[X]'`` names a real ``pyproject`` extra
  D6  every relative Markdown link resolves in the README, ``docs/README.md`` and
      the governance files (``PROJECT_CHARTER``, ``ROADMAP``, ``CONTRIBUTING``,
      ``SUPPORT``, ``SECURITY``, ``GOVERNANCE``); anchors, http(s) and mailto skipped
  D7  the first released ``## [X.Y.Z]`` changelog heading equals ``pyproject`` version
  D8  every tracked ``docs/**/*.md`` outside ``docs/archive/`` (and other than the
      index itself) is linked from ``docs/README.md`` exactly once
  D9  every bullet under the ROADMAP "Now" heading names an issue (``#123``) or an
      OpenSpec change (``openspec/changes/<id>``)
  D1  (CFF) if ``CITATION.cff`` exists, its ``version:`` equals ``pyproject`` version

Exit codes: 0 = pass, 1 = violations, 2 = usage / internal error.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_README_PATTERNS = (
    "wandb",
    "WANDB_",
    "wandb.ai",
    "wandb_logger.py",
    "requirements-dev.txt",
)
_ALLOW_MARKER = "<!-- doc-sync: allow -->"

# D4b: claims the public documents may not make until a committed measurement
# supports them. Matched case-insensitively, line by line, in README.md and
# ROADMAP.md. The en dash (U+2013) and multiplication sign (U+00D7) are accepted
# alongside the ASCII forms because the repository's own prose writes them.
FORBIDDEN_CLAIM_PATTERNS = (
    r"\b1[0-9](?:x\b|\u00d7)",
    r"5[\u2013-]10[x\u00d7]",
    r"production[- ]ready",
    r"LLM eval gate",
    r"convergence (?:validated|proven)",
    r"Hub integration complete",
)
_CLAIM_RES = tuple(re.compile(p, re.IGNORECASE) for p in FORBIDDEN_CLAIM_PATTERNS)
_CLAIM_TARGETS = ("README.md", "ROADMAP.md")

# D6: files whose relative links must resolve.
_LINK_TARGETS = (
    "README.md",
    "docs/README.md",
    "PROJECT_CHARTER.md",
    "ROADMAP.md",
    "CONTRIBUTING.md",
    "SUPPORT.md",
    "SECURITY.md",
    "GOVERNANCE.md",
)

# D9: what a "Now" bullet must carry to count as actionable.
_NOW_REF_RE = re.compile(r"#\d+|openspec/changes/[A-Za-z0-9][A-Za-z0-9_.-]*")
_NOW_HEADING_RE = re.compile(r"^(#{1,6})\s*Now\b")
_ANY_HEADING_RE = re.compile(r"^(#{1,6})\s")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+")
_CFF_VERSION_RE = re.compile(r"^version:\s*[\"']?([^\"'\s]+)", re.MULTILINE)

_INIT_VERSION_RE = re.compile(r"__version__\s*=\s*[\"']([^\"']+)[\"']")
_SETUP_VERSION_RE = re.compile(r"version\s*=\s*[\"']([^\"']+)[\"']")
_MODULE_ROW_RE = re.compile(r"^\|\s*`([a-z0-9_]+)`\s*\|\s*(\d+)\s*\|")
_DECLARED_COUNT_RE = re.compile(r"Submodule count:\*\*\s*(\d+)")
_GATE_DECL_RES = (
    re.compile(r"fail_under\s*=\s*(\d+)"),
    re.compile(r"cov-fail-under[=\s]+(\d+)"),
    re.compile(r"gate\s*\((\d+)%\)"),
    re.compile(r"gate:\s*(\d+)%"),
)
_EXTRA_RE = re.compile(r"pip install -e ['\"]?\.\[([a-zA-Z0-9,_-]+)\]")
_LINK_RE = re.compile(r"\[[^\]]*\]\((?!https?:|#|mailto:)([^)]+)\)")
_RELEASE_HEADING_RE = re.compile(r"^##\s*\[(\d+\.\d+\.\d+)\]")


@dataclass
class Issue:
    code: str
    message: str
    file: Path
    line: int | None = None

    def render(self, root: Path) -> str:
        try:
            rel: Path | str = self.file.relative_to(root)
        except ValueError:
            rel = self.file
        loc = f"{rel}:{self.line}" if self.line is not None else str(rel)
        return f"ERROR {loc}: {self.code} {self.message}"


def _load_pyproject(root: Path) -> dict:
    with (root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _check_versions(root: Path, pyproject: dict, issues: list[Issue]) -> None:
    canonical = pyproject.get("project", {}).get("version")
    init_path = root / "alpha_zero" / "__init__.py"
    if init_path.is_file():
        match = _INIT_VERSION_RE.search(init_path.read_text(encoding="utf-8"))
        if match and match.group(1) != canonical:
            issues.append(
                Issue("D1", f"__version__ '{match.group(1)}' != pyproject '{canonical}'", init_path)
            )
    setup_path = root / "setup.py"
    if setup_path.is_file():
        match = _SETUP_VERSION_RE.search(setup_path.read_text(encoding="utf-8"))
        if match and match.group(1) != canonical:
            issues.append(
                Issue(
                    "D1",
                    f"setup.py version '{match.group(1)}' != pyproject '{canonical}'",
                    setup_path,
                )
            )
    cff_path = root / "CITATION.cff"
    if cff_path.is_file():
        match = _CFF_VERSION_RE.search(cff_path.read_text(encoding="utf-8"))
        if match is None:
            issues.append(Issue("D1", "CITATION.cff declares no 'version:'", cff_path))
        elif match.group(1) != canonical:
            issues.append(
                Issue(
                    "D1",
                    f"CITATION.cff version '{match.group(1)}' != pyproject '{canonical}'",
                    cff_path,
                )
            )


def _submodule_counts(root: Path) -> dict[str, int]:
    pkg = root / "alpha_zero"
    counts: dict[str, int] = {}
    if not pkg.is_dir():
        return counts
    for sub in sorted(pkg.iterdir()):
        if sub.is_dir() and (sub / "__init__.py").is_file():
            counts[sub.name] = sum(1 for p in sub.rglob("*.py") if "__pycache__" not in p.parts)
    return counts


def _check_module_reference(root: Path, issues: list[Issue]) -> None:
    ref = root / "docs" / "MODULE_REFERENCE.md"
    if not ref.is_file():
        return
    lines = ref.read_text(encoding="utf-8").splitlines()
    documented: dict[str, int] = {}
    declared_count: int | None = None
    for raw in lines:
        row = _MODULE_ROW_RE.match(raw)
        if row:
            documented[row.group(1)] = int(row.group(2))
        count_match = _DECLARED_COUNT_RE.search(raw)
        if count_match:
            declared_count = int(count_match.group(1))
    actual = _submodule_counts(root)
    for name, count in sorted(documented.items()):
        if name not in actual:
            issues.append(Issue("D2", f"module '{name}' is documented but not on disk", ref))
        elif actual[name] != count:
            issues.append(
                Issue("D2", f"module '{name}' count {count} != filesystem {actual[name]}", ref)
            )
    for name in sorted(actual):
        if name not in documented:
            issues.append(Issue("D2", f"module '{name}' exists but is undocumented", ref))
    if declared_count is not None and declared_count != len(documented):
        issues.append(
            Issue("D2", f"declared count {declared_count} != {len(documented)} table rows", ref)
        )


def _check_coverage_gate(root: Path, pyproject: dict, issues: list[Issue]) -> None:
    gate = pyproject.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under")
    if gate is None:
        return
    targets = [
        root / ".github" / "workflows" / "ci.yml",
        root / "README.md",
        root / "CONTRIBUTING.md",
        root / "CLAUDE.md",
        root / "docs" / "MODULE_REFERENCE.md",
    ]
    targets += sorted((root / ".claude" / "skills").glob("*/SKILL.md"))
    for target in targets:
        if not target.is_file():
            continue
        for idx, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
            if _ALLOW_MARKER in raw:
                continue
            for pattern in _GATE_DECL_RES:
                for found in pattern.findall(raw):
                    if int(found) != int(gate):
                        issues.append(
                            Issue("D3", f"coverage gate {found}% != canonical {gate}%", target, idx)
                        )


def _check_forbidden_readme(root: Path, issues: list[Issue]) -> None:
    readme = root / "README.md"
    if not readme.is_file():
        return
    for idx, raw in enumerate(readme.read_text(encoding="utf-8").splitlines(), start=1):
        if _ALLOW_MARKER in raw:
            continue
        for pattern in FORBIDDEN_README_PATTERNS:
            if pattern in raw:
                issues.append(Issue("D4", f"forbidden pattern '{pattern}' in README", readme, idx))


def _check_forbidden_claims(root: Path, issues: list[Issue]) -> None:
    for rel in _CLAIM_TARGETS:
        doc = root / rel
        if not doc.is_file():
            continue
        for idx, raw in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            if _ALLOW_MARKER in raw:
                continue
            for pattern in _CLAIM_RES:
                found = pattern.search(raw)
                if found:
                    issues.append(
                        Issue("D4b", f"forbidden claim '{found.group(0)}' in {rel}", doc, idx)
                    )


def _check_extras(root: Path, pyproject: dict, issues: list[Issue]) -> None:
    extras = set(pyproject.get("project", {}).get("optional-dependencies", {}))
    targets = [root / "README.md", root / "CONTRIBUTING.md", root / "CLAUDE.md"]
    if (root / "docs").is_dir():
        # Archived docs are frozen historical artifacts, not held to the sync gate.
        targets += sorted(p for p in (root / "docs").rglob("*.md") if "archive" not in p.parts)
    for target in targets:
        if not target.is_file():
            continue
        for idx, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
            for group in _EXTRA_RE.findall(raw):
                for extra in group.split(","):
                    if extra and extra not in extras:
                        issues.append(Issue("D5", f"unknown extra '[{extra}]'", target, idx))


def _check_links(root: Path, issues: list[Issue]) -> None:
    for rel in _LINK_TARGETS:
        doc = root / rel
        if not doc.is_file():
            continue
        base = doc.parent
        for idx, raw in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            for target in _LINK_RE.findall(raw):
                clean = target.split("#", 1)[0].strip()
                if not clean:
                    continue
                if not (base / clean).exists():
                    issues.append(Issue("D6", f"broken link '{target}'", doc, idx))


def _tracked_docs(root: Path) -> list[Path]:
    """Markdown files under ``docs/`` that D8 holds to the index.

    Uses ``git ls-files`` when ``root`` is a git checkout so that scratch files
    are not counted; falls back to a filesystem walk otherwise (fixtures).
    """
    docs = root / "docs"
    if not docs.is_dir():
        return []
    paths: list[Path] = []
    if (root / ".git").exists():
        import subprocess

        try:
            out = subprocess.run(
                ["git", "-C", str(root), "ls-files", "--", "docs"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            paths = [root / line for line in out.splitlines() if line.endswith(".md")]
        except (OSError, subprocess.CalledProcessError):
            paths = []
    if not paths:
        paths = sorted(docs.rglob("*.md"))
    index = docs / "README.md"
    return [
        p
        for p in paths
        if p.is_file() and p != index and "archive" not in p.relative_to(docs).parts
    ]


def _check_docs_index(root: Path, issues: list[Issue]) -> None:
    index = root / "docs" / "README.md"
    if not index.is_file():
        return
    counts: dict[Path, int] = {}
    base = index.parent
    for raw in index.read_text(encoding="utf-8").splitlines():
        for target in _LINK_RE.findall(raw):
            clean = target.split("#", 1)[0].strip()
            if not clean:
                continue
            try:
                resolved = (base / clean).resolve()
            except OSError:
                continue
            counts[resolved] = counts.get(resolved, 0) + 1
    for doc in _tracked_docs(root):
        n = counts.get(doc.resolve(), 0)
        rel = doc.relative_to(root)
        if n == 0:
            issues.append(Issue("D8", f"'{rel}' is not linked from docs/README.md", index))
        elif n > 1:
            issues.append(Issue("D8", f"'{rel}' is linked {n} times from docs/README.md", index))


def _check_roadmap_now(root: Path, issues: list[Issue]) -> None:
    roadmap = root / "ROADMAP.md"
    if not roadmap.is_file():
        return
    level: int | None = None
    for idx, raw in enumerate(roadmap.read_text(encoding="utf-8").splitlines(), start=1):
        heading = _ANY_HEADING_RE.match(raw)
        if heading:
            now = _NOW_HEADING_RE.match(raw)
            if now:
                level = len(now.group(1))
            elif level is not None and len(heading.group(1)) <= level:
                level = None
            continue
        if level is None or not _BULLET_RE.match(raw):
            continue
        if not _NOW_REF_RE.search(raw):
            issues.append(
                Issue(
                    "D9", "'Now' bullet names no issue (#N) or openspec/changes/<id>", roadmap, idx
                )
            )


def _check_changelog(root: Path, pyproject: dict, issues: list[Issue]) -> None:
    changelog = root / "CHANGELOG.md"
    if not changelog.is_file():
        return
    canonical = pyproject.get("project", {}).get("version")
    for raw in changelog.read_text(encoding="utf-8").splitlines():
        match = _RELEASE_HEADING_RE.match(raw)
        if match:
            if match.group(1) != canonical:
                issues.append(
                    Issue(
                        "D7",
                        f"first released heading [{match.group(1)}] != pyproject '{canonical}'",
                        changelog,
                    )
                )
            return


def run(root: Path) -> list[Issue]:
    issues: list[Issue] = []
    pyproject = _load_pyproject(root)
    _check_versions(root, pyproject, issues)
    _check_module_reference(root, issues)
    _check_coverage_gate(root, pyproject, issues)
    _check_forbidden_readme(root, issues)
    _check_forbidden_claims(root, issues)
    _check_extras(root, pyproject, issues)
    _check_links(root, issues)
    _check_docs_index(root, issues)
    _check_roadmap_now(root, issues)
    _check_changelog(root, pyproject, issues)
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check documentation consistency.")
    parser.add_argument("--root", default=str(REPO_ROOT), help="repository root to check")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    try:
        issues = run(root)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        import json

        payload = [
            {"code": i.code, "message": i.message, "file": str(i.file), "line": i.line}
            for i in issues
        ]
        print(json.dumps(payload, indent=2))
    else:
        for issue in issues:
            print(issue.render(root), file=sys.stderr)

    if issues:
        return 1
    if not args.quiet and not args.json:
        print("doc-sync: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
