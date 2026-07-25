#!/usr/bin/env python3
"""Deterministic validator for Claude Code skills under ``.claude/skills/``.

Checks (error codes are stable and referenced by ``tests/governance/``):

  S1  every ``SKILL.md`` opens with YAML frontmatter whose ``name`` equals the
      skill's directory name and whose ``description`` is non-empty
      (a missing ``allowed-tools`` key is a WARNING, not an error)
  S2  the body has at least one ``## `` section
  S3  every concrete ``kb/...`` path literal referenced in the body exists
  S4  every coverage-percentage literal equals the canonical gate from
      ``pyproject.toml`` (``fail_under``)

A "concrete" kb path is lowercase with a file extension and no placeholder
characters, so example templates like ``kb/handoffs/YYYYMMDD-<slug>.md`` are
skipped.

Modes: whole tree (default) or ``--path FILE`` (validate one ``SKILL.md``).
Exit codes: 0 = pass, 1 = validation errors, 2 = usage / internal error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
PYPROJECT = REPO_ROOT / "pyproject.toml"

_KB_PATH_RE = re.compile(r"kb/[\w./-]+\.\w+")
_COVERAGE_LINE_RE = re.compile(r"coverage|cov-fail-under", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(\d{2,3})\s*%")
_COV_FLAG_RE = re.compile(r"cov-fail-under[=\s]+(\d{2,3})")
_FAIL_UNDER_RE = re.compile(r"fail_under\s*=\s*(\d+)")


@dataclass
class Issue:
    code: str
    message: str
    file: Path
    line: int | None = None
    warning: bool = False

    def render(self) -> str:
        rel = _relpath(self.file)
        loc = f"{rel}:{self.line}" if self.line is not None else str(rel)
        label = "WARNING" if self.warning else "ERROR"
        return f"{label} {loc}: {self.code} {self.message}"


def _relpath(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def _coverage_gate() -> int | None:
    if not PYPROJECT.is_file():
        return None
    match = _FAIL_UNDER_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def _parse_frontmatter(lines: list[str]) -> dict[str, str] | None:
    if not lines or lines[0].strip() != "---":
        return None
    data: dict[str, str] = {}
    for raw in lines[1:]:
        if raw.strip() == "---":
            return data
        if ":" in raw and not raw.startswith(" "):
            key, _, value = raw.partition(":")
            data[key.strip()] = value.strip()
    return None  # unterminated frontmatter


def _is_concrete_kb_path(candidate: str) -> bool:
    if any(ch in candidate for ch in "<>{}*"):
        return False
    return candidate == candidate.lower()  # placeholders like YYYYMMDD are uppercase


def _validate_skill(skill_md: Path, issues: list[Issue], gate: int | None) -> None:
    dir_name = skill_md.parent.name
    lines = skill_md.read_text(encoding="utf-8").splitlines()
    front = _parse_frontmatter(lines)
    if front is None:
        issues.append(Issue("S1", "SKILL.md has no YAML frontmatter block", skill_md, 1))
    else:
        name = front.get("name", "")
        if name != dir_name:
            issues.append(
                Issue("S1", f"frontmatter name '{name}' != directory '{dir_name}'", skill_md, 1)
            )
        if not front.get("description"):
            issues.append(Issue("S1", "frontmatter has no non-empty 'description'", skill_md, 1))
        if "allowed-tools" not in front:
            issues.append(
                Issue("S1", "frontmatter has no 'allowed-tools'", skill_md, 1, warning=True)
            )

    if not any(line.startswith("## ") for line in lines):
        issues.append(Issue("S2", "SKILL.md has no '## ' section", skill_md, 1))

    for idx, raw in enumerate(lines, start=1):
        for candidate in _KB_PATH_RE.findall(raw):
            if _is_concrete_kb_path(candidate) and not (REPO_ROOT / candidate).exists():
                issues.append(
                    Issue("S3", f"referenced kb path '{candidate}' is missing", skill_md, idx)
                )
        if gate is not None and _COVERAGE_LINE_RE.search(raw):
            for found in _PERCENT_RE.findall(raw) + _COV_FLAG_RE.findall(raw):
                if int(found) != gate:
                    issues.append(
                        Issue("S4", f"coverage literal {found}% != gate {gate}%", skill_md, idx)
                    )


def _validate_all(issues: list[Issue], gate: int | None) -> None:
    if not SKILLS_DIR.is_dir():
        return
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        _validate_skill(skill_md, issues, gate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate .claude/skills/ metadata.")
    parser.add_argument("--path", help="validate a single SKILL.md")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    gate = _coverage_gate()
    issues: list[Issue] = []
    try:
        if args.path:
            target = Path(args.path).resolve()
            if target.name == "SKILL.md" and target.is_file():
                _validate_skill(target, issues, gate)
            elif not target.is_file():
                issues.append(Issue("S1", "path does not exist", target))
        else:
            _validate_all(issues, gate)
    except OSError as exc:  # pragma: no cover - defensive I/O guard
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    errors = [i for i in issues if not i.warning]
    if args.json:
        payload = [
            {
                "code": i.code,
                "message": i.message,
                "file": str(_relpath(i.file)),
                "line": i.line,
                "warning": i.warning,
            }
            for i in issues
        ]
        print(json.dumps(payload, indent=2))
    else:
        for issue in issues:
            print(issue.render(), file=sys.stderr)

    if errors:
        return 1
    if not args.quiet and not args.json:
        print("skills: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
