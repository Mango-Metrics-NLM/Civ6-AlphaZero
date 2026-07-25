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
  D5  every ``pip install -e '.[X]'`` names a real ``pyproject`` extra
  D6  every relative Markdown link in the README and ``docs/README.md`` resolves
  D7  the first released ``## [X.Y.Z]`` changelog heading equals ``pyproject`` version

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
    for rel in ("README.md", "docs/README.md"):
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
    _check_extras(root, pyproject, issues)
    _check_links(root, issues)
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
