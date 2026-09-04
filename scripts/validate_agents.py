#!/usr/bin/env python3
"""Deterministic validator for ``Agent.md`` persona documents.

Every package carries an ``Agent.md`` describing the role that owns it -- the
"Core Algorithm Engineer" for ``alpha_zero/core``, and so on. Thirty of them
exist and, until now, **nothing validated any of them**: no gate, no hook, no
test. They drifted accordingly (a persona doc still describing a file that had
become a re-export shim is what prompted this).

Checks (error codes are stable and referenced by ``tests/governance/``):

  A0  a ``--path`` target exists and is named ``Agent.md`` (a missing or
      misnamed target is an error, not a silent pass -- the PostToolUse hook
      calls ``--path``, so exiting 0 would make a stale path look validated)
  A1  the H1 is ``# Agent.md — <dotted.module.path>`` and that path resolves to
      a real directory
  A2  a ``## Persona`` section exists and names a role as ``You are the **X**``
  A3  the required sections are present (``## Persona``, ``## Module Purpose``,
      ``## Design Patterns``); the six conventional remainder sections are
      reported as WARNINGs so the truncated docs are visible without blocking
  A4  every coverage-percentage literal equals the canonical gate from
      ``pyproject.toml`` -- the same rule ``validate_skills.py`` S4 applies to
      skills, which caught a stale 80% in ``tests/Agent.md``
  A5  every **top-level** package under ``alpha_zero/`` has an ``Agent.md``
      (nested subpackages are covered by their parent's document; see the
      validation-gates spec for why)
  A6  every backticked ``*.py`` literal resolves, relative to the documented
      module directory or the repo root. A line carrying
      ``<!-- validate-agents: allow -->`` is exempt -- the same escape hatch
      ``check_doc_sync.py`` gives with ``<!-- doc-sync: allow -->``, for prose
      that deliberately names a file that no longer exists.

Modes: whole tree (default) or ``--path FILE`` (validate one ``Agent.md``).
Add ``--verbose`` for a trace of what was scanned and which gate was resolved --
useful when the answer is "OK" and you want to know whether it actually looked.
Exit codes: 0 = pass, 1 = validation errors, 2 = usage / internal error.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "alpha_zero"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Directories that carry an Agent.md but are not importable packages.
EXTRA_AGENT_DIRS = ("tests", "infra")

REQUIRED_SECTIONS = ("## Persona", "## Module Purpose", "## Design Patterns")
CONVENTIONAL_SECTIONS = (
    "## Key Files",
    "## Skills",
    "## Sub-Agents",
    "## Tools",
    "## Invariants",
    "## Dependencies",
)

_H1_RE = re.compile(r"^#\s+Agent\.md\s+[—-]\s+(?P<path>[\w.]+)\s*$")
_PERSONA_RE = re.compile(r"^You are the \*\*(?P<role>[^*]+)\*\*")
_BACKTICK_PY_RE = re.compile(r"`([\w./-]+\.py)`")
_COVERAGE_LINE_RE = re.compile(r"coverage|cov-fail-under", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(?<![\d.])(\d{2,3})\s*%")
_COV_FLAG_RE = re.compile(r"cov-fail-under[=\s]+(\d{2,3})")
_FAIL_UNDER_RE = re.compile(r"fail_under\s*=\s*(\d+)")
_ALLOW_MARKER = "<!-- validate-agents: allow -->"

logger = logging.getLogger("validate_agents")


@dataclass
class Issue:
    code: str
    message: str
    file: Path
    line: int | None = None
    warning: bool = False

    def render(self) -> str:
        # `relative_to` raises for an absolute path outside the repo, which is
        # exactly the shape of the A0 case this renderer has to report:
        # `--path /nonexistent/...` crashed with a ValueError traceback instead
        # of printing its own diagnostic. Fall back to the absolute path.
        rel: Path | str = self.file
        if self.file.is_absolute():
            try:
                rel = self.file.relative_to(REPO_ROOT)
            except ValueError:
                rel = self.file
        where = f"{rel}:{self.line}" if self.line else str(rel)
        level = "WARNING" if self.warning else "ERROR"
        return f"{level} {where}: {self.code} {self.message}"


def _coverage_gate() -> int | None:
    if not PYPROJECT.is_file():
        return None
    match = _FAIL_UNDER_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def _documented_dir(dotted: str) -> Path:
    """Map ``alpha_zero.core`` -> ``<repo>/alpha_zero/core``."""
    return REPO_ROOT.joinpath(*dotted.split("."))


def _validate_agent_doc(agent_md: Path, issues: list[Issue], gate: int | None) -> None:
    lines = agent_md.read_text(encoding="utf-8").splitlines()
    logger.debug("scanning %s (%d lines)", agent_md, len(lines))
    text = "\n".join(lines)

    # A1 -- heading form and that it points at something real.
    module_dir: Path | None = None
    heading = lines[0] if lines else ""
    match = _H1_RE.match(heading.strip())
    if match is None:
        issues.append(
            Issue("A1", f"H1 must be '# Agent.md — <module.path>', got {heading!r}", agent_md, 1)
        )
    else:
        module_dir = _documented_dir(match.group("path"))
        if not module_dir.is_dir():
            issues.append(
                Issue(
                    "A1",
                    f"H1 names '{match.group('path')}' but {module_dir.name}/ does not exist",
                    agent_md,
                    1,
                )
            )
            module_dir = None

    # A2 -- a named role, so the doc says who owns the module.
    if not any(_PERSONA_RE.match(line.strip()) for line in lines):
        issues.append(Issue("A2", "no 'You are the **<Role>**' persona statement", agent_md, None))

    # A3 -- structure. Required blocks are errors; the conventional remainder warns.
    for section in REQUIRED_SECTIONS:
        if not any(line.strip() == section for line in lines):
            issues.append(Issue("A3", f"missing required section '{section}'", agent_md, None))
    for section in CONVENTIONAL_SECTIONS:
        if not any(line.strip() == section for line in lines):
            issues.append(
                Issue("A3", f"missing conventional section '{section}'", agent_md, None, True)
            )

    for idx, raw in enumerate(lines, start=1):
        if _ALLOW_MARKER in raw:
            continue
        # A4 -- coverage literals must track the canonical gate.
        if gate is not None and _COVERAGE_LINE_RE.search(raw):
            for found in _PERCENT_RE.findall(raw) + _COV_FLAG_RE.findall(raw):
                if int(found) != gate:
                    issues.append(
                        Issue("A4", f"coverage literal {found}% != gate {gate}%", agent_md, idx)
                    )
        # A6 -- backticked python paths must resolve.
        for candidate in _BACKTICK_PY_RE.findall(raw):
            if "*" in candidate or "<" in candidate:
                continue
            roots = [module_dir, REPO_ROOT] if module_dir else [REPO_ROOT]
            if not any((root / candidate).exists() for root in roots if root):
                issues.append(
                    Issue("A6", f"referenced file '{candidate}' does not exist", agent_md, idx)
                )

    del text  # kept for readability of the checks above; not otherwise needed


def _validate_all(issues: list[Issue], gate: int | None) -> None:
    seen: set[Path] = set()
    logger.debug("package root: %s | coverage gate: %s", PACKAGE_ROOT, gate)

    # A5 -- every importable top-level package must document its owner.
    if PACKAGE_ROOT.is_dir():
        for package in sorted(p for p in PACKAGE_ROOT.iterdir() if p.is_dir()):
            if not (package / "__init__.py").is_file():
                continue
            agent_md = package / "Agent.md"
            if not agent_md.is_file():
                issues.append(Issue("A5", "package has no Agent.md", package, None))
                continue
            seen.add(agent_md)
            _validate_agent_doc(agent_md, issues, gate)

    # A5 only *requires* a document at the top level, but a document that exists
    # anywhere must still be correct. `alpha_zero/core/mcts/Agent.md` was written
    # for a nested subpackage and was therefore scanned by nothing at all -- an
    # unvalidated governance document is worse than a missing one, because it
    # reads as covered. `rglob` order is filesystem-dependent, so it is sorted:
    # unsorted validator output makes a CI failure unreproducible.
    if PACKAGE_ROOT.is_dir():
        for agent_md in sorted(PACKAGE_ROOT.rglob("Agent.md")):
            if agent_md.is_file() and agent_md not in seen:
                seen.add(agent_md)
                _validate_agent_doc(agent_md, issues, gate)

    for extra in EXTRA_AGENT_DIRS:
        agent_md = REPO_ROOT / extra / "Agent.md"
        if agent_md.is_file() and agent_md not in seen:
            seen.add(agent_md)
            _validate_agent_doc(agent_md, issues, gate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Agent.md persona documents.")
    parser.add_argument("--path", help="validate a single Agent.md")
    parser.add_argument("--quiet", action="store_true", help="suppress the OK line")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log every document scanned and the resolved coverage gate (stderr)",
    )
    parser.add_argument(
        "--max-warnings",
        type=int,
        default=None,
        dest="max_warnings",
        help=(
            "fail (exit 1) when the WARNING count exceeds N. The A3 conventional-section "
            "warnings are advisory by design -- filling them in is a content-writing "
            "exercise, not a validation one -- so the ratchet keeps the count from growing "
            "instead of pretending the docs are complete."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    issues: list[Issue] = []
    gate = _coverage_gate()

    try:
        if args.path:
            target = Path(args.path)
            if not target.is_absolute():
                target = REPO_ROOT / target
            # A missing or misnamed target is an error, never a silent pass. The
            # PostToolUse hook invokes --path, so falling through to exit 0 would
            # make a stale path read as a clean validation -- the failure mode
            # that makes a gate worthless.
            if not target.is_file():
                issues.append(Issue("A0", "path does not exist", target))
            elif target.name != "Agent.md":
                issues.append(Issue("A0", f"not an Agent.md file: {target.name}", target))
            else:
                _validate_agent_doc(target, issues, gate)
        else:
            _validate_all(issues, gate)
    except OSError as exc:  # pragma: no cover - filesystem failure
        print(f"agents: internal error: {exc}", file=sys.stderr)
        return 2

    errors = [i for i in issues if not i.warning]
    warnings = [i for i in issues if i.warning]
    logger.debug(
        "finished: %d issue(s) -- %d error(s), %d warning(s)",
        len(issues),
        len(errors),
        len(warnings),
    )
    over_budget = args.max_warnings is not None and len(warnings) > args.max_warnings
    if args.as_json:
        print(
            json.dumps(
                {
                    "ok": not errors and not over_budget,
                    "warning_count": len(warnings),
                    "issues": [
                        {
                            "code": i.code,
                            "message": i.message,
                            "file": str(i.file),
                            "line": i.line,
                            "warning": i.warning,
                        }
                        for i in issues
                    ],
                },
                indent=2,
            )
        )
        return 1 if (errors or over_budget) else 0

    for issue in issues:
        print(issue.render(), file=sys.stderr)
    if errors:
        return 1
    if over_budget:
        print(
            f"agents: {len(warnings)} warning(s) exceeds the ratchet of {args.max_warnings}. "
            "Add the missing sections, or raise the baseline only with a recorded reason.",
            file=sys.stderr,
        )
        return 1
    if not args.quiet:
        print(f"agents: OK ({len(warnings)} warning(s))" if warnings else "agents: OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
