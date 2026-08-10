#!/usr/bin/env python3
"""Deterministic validator for the ``openspec/`` spec-driven change directory.

This is a vendored, stdlib-only re-implementation of the structural checks the
`Fission-AI/OpenSpec <https://github.com/Fission-AI/OpenSpec>`_ CLI performs.
It is the *binding* governance gate for this repo (the npm CLI is optional; an
``npx ...@latest`` network fetch is unfit for a deterministic gate). The
directory layout it validates stays compatible with the upstream CLI.

Checks (error codes are stable and referenced by ``tests/governance/``):

Structure / baseline specs
  OS001  ``openspec/`` contains ``project.md``, ``AGENTS.md``, ``specs/``, ``changes/``
  OS002  every ``specs/<cap>/`` capability directory name is kebab-case
  OS003  every ``spec.md`` has a ``## Purpose`` heading and >= 1 ``### Requirement:``
  OS004  every requirement body states a normative SHALL or MUST
  OS005  every requirement has >= 1 ``#### Scenario:`` and each scenario has WHEN + THEN

Change directories
  OS010  every change dir has ``proposal.md`` with ``## Why`` and ``## What Changes``
  OS011  every change dir has ``tasks.md`` with >= 1 well-formed ``- [ ]`` / ``- [x]`` item
  OS012  change-id (directory name) matches ``^[a-z0-9][a-z0-9-]*$``

Spec deltas (``changes/<id>/specs/<cap>/spec.md``)
  OS020  delta H2 sections are a subset of ``ADDED|MODIFIED|REMOVED Requirements``
  OS021  every ADDED / MODIFIED requirement carries scenarios (OS004 + OS005 rules)
  OS022  every MODIFIED / REMOVED requirement name exists verbatim in the baseline
         spec for that capability (a *missing* baseline file is an empty set, so
         ADDED-only deltas for a brand-new capability are valid)
  OS023  every ADDED requirement name does not collide with an existing baseline name

Modes
  (default)        validate the whole ``openspec/`` tree
  --change <id>    validate structure + only the named change directory
  --path <file>    validate a single file in isolation (syntax/schema of that file
                   only; no cross-file checks) -- used by the PostToolUse hook

Exit codes: 0 = pass, 1 = validation errors, 2 = usage / internal error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENSPEC_DIR = REPO_ROOT / "openspec"

VALID_DELTA_SECTIONS = frozenset({"ADDED", "MODIFIED", "REMOVED"})
_KEBAB_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_CHANGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_REQUIREMENT_RE = re.compile(r"^###\s+Requirement:\s*(.+?)\s*$")
_SCENARIO_RE = re.compile(r"^####\s+Scenario:\s*(.+?)\s*$")
_DELTA_H2_RE = re.compile(r"^##\s+(ADDED|MODIFIED|REMOVED)\s+Requirements\s*$")
_CHECKBOX_RE = re.compile(r"^\s*-\s+\[[ xX]\]\s+\S")
_WHEN_RE = re.compile(r"\*\*WHEN\*\*", re.IGNORECASE)
_THEN_RE = re.compile(r"\*\*THEN\*\*", re.IGNORECASE)
_NORMATIVE_RE = re.compile(r"\b(SHALL|MUST)\b")


@dataclass
class Issue:
    """A single validation failure, rendered as ``ERROR <file>[:<line>]: <CODE> <msg>``."""

    code: str
    message: str
    file: Path
    line: int | None = None

    def render(self) -> str:
        rel = _relpath(self.file)
        loc = f"{rel}:{self.line}" if self.line is not None else str(rel)
        return f"ERROR {loc}: {self.code} {self.message}"


@dataclass
class Scenario:
    name: str
    line: int
    body: list[str] = field(default_factory=list)


@dataclass
class Requirement:
    name: str
    line: int
    body: list[str] = field(default_factory=list)
    scenarios: list[Scenario] = field(default_factory=list)


def _relpath(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _parse_requirements(lines: list[str]) -> list[Requirement]:
    """Parse ``### Requirement:`` blocks and their ``#### Scenario:`` children.

    A requirement body runs until the next ``## ``/``### `` heading; scenarios
    (``#### ``) live inside a requirement and run until the next ``#### `` or the
    requirement boundary.
    """
    requirements: list[Requirement] = []
    current: Requirement | None = None
    scenario: Scenario | None = None
    for idx, raw in enumerate(lines, start=1):
        req_match = _REQUIREMENT_RE.match(raw)
        if req_match:
            current = Requirement(name=req_match.group(1), line=idx)
            requirements.append(current)
            scenario = None
            continue
        # A new top-level (##) or requirement-level (###) heading closes the block.
        if raw.startswith("## ") or (raw.startswith("### ") and not req_match):
            current = None
            scenario = None
            continue
        if current is None:
            continue
        scen_match = _SCENARIO_RE.match(raw)
        if scen_match:
            scenario = Scenario(name=scen_match.group(1), line=idx)
            current.scenarios.append(scenario)
            continue
        if scenario is not None:
            scenario.body.append(raw)
        else:
            current.body.append(raw)
    return requirements


def _check_requirement_semantics(
    req: Requirement, path: Path, issues: list[Issue], *, require_scenarios: bool
) -> None:
    """OS004 + OS005: normative language and well-formed scenarios."""
    body_text = "\n".join(req.body)
    if not _NORMATIVE_RE.search(body_text):
        issues.append(
            Issue("OS004", f"requirement '{req.name}' states no SHALL/MUST", path, req.line)
        )
    if not require_scenarios:
        return
    if not req.scenarios:
        issues.append(Issue("OS005", f"requirement '{req.name}' has no scenario", path, req.line))
        return
    for scen in req.scenarios:
        scen_text = "\n".join(scen.body)
        if not _WHEN_RE.search(scen_text):
            issues.append(
                Issue("OS005", f"scenario '{scen.name}' has no WHEN line", path, scen.line)
            )
        if not _THEN_RE.search(scen_text):
            issues.append(
                Issue("OS005", f"scenario '{scen.name}' has no THEN line", path, scen.line)
            )


def _validate_baseline_spec(path: Path, issues: list[Issue]) -> None:
    lines = _read_lines(path)
    if not any(line.strip().lower() == "## purpose" for line in lines):
        issues.append(Issue("OS003", "spec has no '## Purpose' section", path, 1))
    requirements = _parse_requirements(lines)
    if not requirements:
        issues.append(Issue("OS003", "spec has no '### Requirement:' heading", path, 1))
    for req in requirements:
        _check_requirement_semantics(req, path, issues, require_scenarios=True)


def _baseline_requirement_names(capability: str) -> set[str]:
    """Requirement names in the deployed baseline for a capability (empty if none)."""
    baseline = OPENSPEC_DIR / "specs" / capability / "spec.md"
    if not baseline.is_file():
        return set()
    return {req.name for req in _parse_requirements(_read_lines(baseline))}


def _validate_delta_spec(path: Path, issues: list[Issue], *, cross_file: bool) -> None:
    lines = _read_lines(path)
    capability = path.parent.name
    # OS020: only ADDED/MODIFIED/REMOVED Requirements H2s are allowed.
    section: str | None = None
    section_lines: dict[str, int] = {}
    for idx, raw in enumerate(lines, start=1):
        if raw.startswith("## "):
            delta_match = _DELTA_H2_RE.match(raw)
            if not delta_match:
                issues.append(
                    Issue("OS020", f"unexpected delta section heading '{raw.strip()}'", path, idx)
                )
                section = None
                continue
            section = delta_match.group(1)
            section_lines.setdefault(section, idx)

    requirements = _parse_requirements(lines)
    # Attribute each requirement to the delta section it falls under.
    added: list[Requirement] = []
    modified: list[Requirement] = []
    removed: list[Requirement] = []
    active: str | None = None
    section_by_line = sorted(
        ((ln, name) for name, ln in _iter_delta_sections(lines)), key=lambda t: t[0]
    )
    for req in requirements:
        active = _section_for_line(section_by_line, req.line)
        if active == "ADDED":
            added.append(req)
        elif active == "MODIFIED":
            modified.append(req)
        elif active == "REMOVED":
            removed.append(req)

    # OS021: ADDED/MODIFIED carry scenarios; REMOVED needs only a name.
    for req in added + modified:
        _check_requirement_semantics(req, path, issues, require_scenarios=True)

    if not cross_file:
        return

    baseline_names = _baseline_requirement_names(capability)
    # OS022: MODIFIED/REMOVED must name an existing baseline requirement.
    for req in modified + removed:
        if req.name not in baseline_names:
            issues.append(
                Issue(
                    "OS022",
                    f"MODIFIED/REMOVED requirement '{req.name}' is absent from baseline "
                    f"capability '{capability}'",
                    path,
                    req.line,
                )
            )
    # OS023: ADDED must not collide with an existing baseline requirement.
    for req in added:
        if req.name in baseline_names:
            issues.append(
                Issue(
                    "OS023",
                    f"ADDED requirement '{req.name}' already exists in baseline "
                    f"capability '{capability}'",
                    path,
                    req.line,
                )
            )


def _iter_delta_sections(lines: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for idx, raw in enumerate(lines, start=1):
        delta_match = _DELTA_H2_RE.match(raw)
        if delta_match:
            out.append((delta_match.group(1), idx))
    return out


def _section_for_line(section_by_line: list[tuple[int, str]], line: int) -> str | None:
    active: str | None = None
    for start, name in section_by_line:
        if start <= line:
            active = name
        else:
            break
    return active


def _validate_proposal(path: Path, issues: list[Issue]) -> None:
    text = path.read_text(encoding="utf-8")
    if "## Why" not in text:
        issues.append(Issue("OS010", "proposal.md has no '## Why' section", path, 1))
    if "## What Changes" not in text:
        issues.append(Issue("OS010", "proposal.md has no '## What Changes' section", path, 1))


def _validate_tasks(path: Path, issues: list[Issue]) -> None:
    lines = _read_lines(path)
    if not any(_CHECKBOX_RE.match(line) for line in lines):
        issues.append(Issue("OS011", "tasks.md has no '- [ ]' checkbox item", path, 1))


def _validate_change_dir(change_dir: Path, issues: list[Issue], *, cross_file: bool) -> None:
    change_id = change_dir.name
    if not _CHANGE_ID_RE.match(change_id):
        issues.append(Issue("OS012", f"change-id '{change_id}' is not kebab-case", change_dir))
    proposal = change_dir / "proposal.md"
    if proposal.is_file():
        _validate_proposal(proposal, issues)
    else:
        issues.append(Issue("OS010", "change dir has no proposal.md", change_dir))
    tasks = change_dir / "tasks.md"
    if tasks.is_file():
        _validate_tasks(tasks, issues)
    else:
        issues.append(Issue("OS011", "change dir has no tasks.md", change_dir))
    for delta in sorted((change_dir / "specs").glob("*/spec.md")):
        _validate_delta_spec(delta, issues, cross_file=cross_file)


def _validate_structure(issues: list[Issue]) -> bool:
    """OS001/OS002. Returns False (with an issue) if the tree is fundamentally absent."""
    if not OPENSPEC_DIR.is_dir():
        issues.append(Issue("OS001", "openspec/ directory does not exist", OPENSPEC_DIR))
        return False
    for required in ("project.md", "AGENTS.md"):
        if not (OPENSPEC_DIR / required).is_file():
            issues.append(Issue("OS001", f"openspec/{required} is missing", OPENSPEC_DIR))
    for required_dir in ("specs", "changes"):
        if not (OPENSPEC_DIR / required_dir).is_dir():
            issues.append(Issue("OS001", f"openspec/{required_dir}/ is missing", OPENSPEC_DIR))
    specs_dir = OPENSPEC_DIR / "specs"
    if specs_dir.is_dir():
        for cap_dir in sorted(p for p in specs_dir.iterdir() if p.is_dir()):
            if not _KEBAB_RE.match(cap_dir.name):
                issues.append(
                    Issue("OS002", f"capability dir '{cap_dir.name}' is not kebab-case", cap_dir)
                )
    return True


def _validate_all(issues: list[Issue], *, only_change: str | None) -> None:
    if not _validate_structure(issues):
        return
    for spec in sorted((OPENSPEC_DIR / "specs").glob("*/spec.md")):
        _validate_baseline_spec(spec, issues)
    changes_dir = OPENSPEC_DIR / "changes"
    if not changes_dir.is_dir():
        return
    for change_dir in sorted(p for p in changes_dir.iterdir() if p.is_dir()):
        if change_dir.name == "archive":
            continue
        if only_change is not None and change_dir.name != only_change:
            continue
        _validate_change_dir(change_dir, issues, cross_file=True)
    if only_change is not None and not (changes_dir / only_change).is_dir():
        issues.append(Issue("OS010", f"change '{only_change}' does not exist", changes_dir))


def _validate_path(path: Path, issues: list[Issue]) -> None:
    """Single-file mode for the hook: only checks intrinsic to that file."""
    if not path.is_file():
        issues.append(Issue("OS001", "path does not exist", path))
        return
    name = path.name
    parts = path.parts
    if name == "proposal.md":
        _validate_proposal(path, issues)
    elif name == "tasks.md":
        _validate_tasks(path, issues)
    elif name == "spec.md" and "changes" in parts:
        _validate_delta_spec(path, issues, cross_file=False)
    elif name == "spec.md":
        _validate_baseline_spec(path, issues)
    # project.md / AGENTS.md have no intra-file schema beyond being present.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the openspec/ directory.")
    parser.add_argument("--change", help="validate only the named change directory")
    parser.add_argument("--path", help="validate a single file in isolation (hook mode)")
    parser.add_argument("--quiet", action="store_true", help="suppress the pass message")
    parser.add_argument("--json", action="store_true", help="emit issues as JSON")
    args = parser.parse_args(argv)

    issues: list[Issue] = []
    try:
        if args.path:
            _validate_path(Path(args.path).resolve(), issues)
        else:
            _validate_all(issues, only_change=args.change)
    except OSError as exc:  # pragma: no cover - defensive I/O guard
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = [
            {"code": i.code, "message": i.message, "file": str(_relpath(i.file)), "line": i.line}
            for i in issues
        ]
        print(json.dumps(payload, indent=2))
    else:
        for issue in issues:
            print(issue.render(), file=sys.stderr)

    if issues:
        return 1
    if not args.quiet and not args.json:
        print("openspec: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
