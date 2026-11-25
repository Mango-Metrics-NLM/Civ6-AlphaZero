#!/usr/bin/env python3
"""Deterministic validator for Claude Code skills.

Scans **both** skill trees: ``.claude/skills/`` (project skills) and
``.agents/skills/`` (agent-authored skills). The second tree was previously
unscanned by every gate -- the PostToolUse hook routed it here, but neither
pre-commit nor CI did -- so violations in it accumulated silently.

Checks (error codes are stable and referenced by ``tests/governance/``):

  S1  every ``SKILL.md`` opens with YAML frontmatter whose ``name`` equals the
      skill's directory name and whose ``description`` is non-vacuous -- not
      empty, not a quoted-empty string, and not a bare YAML block-scalar
      indicator (a missing ``allowed-tools`` key is a WARNING, not an error)
  S2  the body has at least one ``## `` section
  S3  every concrete ``kb/...`` path literal referenced in the body exists
  S4  every coverage-percentage literal equals the canonical gate from
      ``pyproject.toml`` (``fail_under``)
  S5  every skill directory actually contains a ``SKILL.md`` (a directory
      without one used to be skipped in silence)
  S6  every script, hook, make target, pytest path and skill that the
      instructions name exists: a ``scripts/*.py|sh`` or
      ``.claude/hooks|skills/...`` path is on disk, a ``make`` target is
      defined in ``REPO_ROOT / "Makefile"`` (read at call time), a path after
      ``pytest`` resolves, and a skill named as one is a skill directory or one
      of Claude Code's built-in skills. What counts as a skill reference (a code
      span followed by "skill" or "skills", line breaks allowed) and the
      built-in allowlist (``BUILTIN_SKILLS``) live in ``scripts/skill_refs.py``,
      which ``validate_claude_agents.py`` (C6) uses too

S6 reads only fenced code blocks (three or more backticks or tildes; a fence
closes on a line of its own character at least as long) and inline code spans,
never prose: a scan over prose reported "make a red build green" and "make the
coverage-gate number match" as make targets. In a ``make`` command the option
words (``-C DIR``, ``-f FILE``, ``-j4``, ``-n``) and ``VAR=value`` assignments
are skipped before the targets. A path must not continue another path (no word
character, ``/``, ``.`` or ``-`` before it), so ``tests/scripts/x.py`` is not
read as ``scripts/x.py``. ``tests/governance/test_skill_commands_resolve.py``
imports the same extraction (``dangling_references``) rather than keeping a
copy.

Known limits of S6, not checked: paths under ``.agents/skills/``; the module form
of a command (``python -m scripts.x``, which names no ``scripts/`` path);
a glob in a pytest path, which reads as its prefix and is reported, so a skill
names a make target or a file instead.

A "concrete" kb path is lowercase with a file extension and no placeholder
characters, so example templates like ``kb/handoffs/YYYYMMDD-<slug>.md`` are
skipped.

Modes: whole tree (default) or ``--path FILE`` (validate one ``SKILL.md``).
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

# The skill-reference rule is shared with validate_claude_agents.py (C6). The checkout
# heads sys.path whether this runs as a script, as a module or is loaded by a test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.skill_refs import (
    BUILTIN_SKILLS,  # noqa: F401 -- re-exported: the allowlist the rule uses
    known_skills,
    unknown_skill_references,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
# Agent-authored skills. Kept as a separate constant rather than a list so the
# existing ``monkeypatch.setattr(vs, "SKILLS_DIR", ...)`` idiom in
# tests/governance/ keeps working unchanged.
AGENT_SKILLS_DIR = REPO_ROOT / ".agents" / "skills"
PYPROJECT = REPO_ROOT / "pyproject.toml"
logger = logging.getLogger("validate_skills")

_KB_PATH_RE = re.compile(r"kb/[\w./-]+\.\w+")
# A description that reduces to one of these is the signature of a parser that
# did not read the value, not of an authored description. ``description: ">"``
# and ``description: ""`` both used to satisfy the non-empty check *vacuously*,
# which is the exact failure mode this validator was written to end.
_VACUOUS_DESCRIPTIONS = frozenset({"", ">", "|", ">-", "|-", ">+", "|+", "-"})
_COVERAGE_LINE_RE = re.compile(r"coverage|cov-fail-under", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(?<![\d.])(\d{2,3})\s*%")
_COV_FLAG_RE = re.compile(r"cov-fail-under[=\s]+(\d{2,3})")
_FAIL_UNDER_RE = re.compile(r"fail_under\s*=\s*(\d+)")

# S6. A fence opens with three or more backticks or tildes and closes with a line of
# the same character, at least as long.
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_RE = re.compile(r"`([^`\n]+)`")
# A ``make`` command: at the start of the span or after a shell operator.
_MAKE_RE = re.compile(r"(?:^|[|&;]\s*)make((?:\s+[^\s|&;#<>]+)*)")
_TARGET_RE = re.compile(r"[a-z0-9_.-]+")
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
# Options of GNU make that take the next word as their argument.
_MAKE_ARG_OPTIONS = frozenset({"-C", "-f", "-I", "-o", "-W", "--directory", "--file"})
# ``-j`` and ``-l`` take a number only when one follows (``make -j verify`` is legal).
_MAKE_NUMBER_OPTIONS = frozenset({"-j", "-l"})
_SCRIPT_RE = re.compile(r"(?<![\w/.-])(scripts/[\w./-]+\.(?:py|sh))")
_HARNESS_RE = re.compile(r"(?<![\w/.-])(\.claude/(?:hooks|skills)/[\w./-]+)")
_TESTS_RE = re.compile(r"(?<![\w/.-])(tests/[\w./-]+)")


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
    """Parse the frontmatter block into a flat ``key -> value`` mapping.

    Not a general YAML parser, deliberately -- the governance validators are
    stdlib-only. It does understand block scalars (``key: >`` and ``key: |``)
    because three skills in ``.agents/skills/`` use them: treating the value as
    the literal ``">"`` made a folded ``description`` pass the non-empty check
    vacuously, which is worse than not checking at all.
    """
    if not lines or lines[0].strip() != "---":
        return None
    data: dict[str, str] = {}
    pending_key: str | None = None
    pending: list[str] = []

    def _flush() -> None:
        nonlocal pending_key
        if pending_key is not None:
            data[pending_key] = " ".join(pending).strip()
            pending_key = None
            pending.clear()

    for raw in lines[1:]:
        if raw.strip() == "---":
            _flush()
            return data
        if raw.startswith((" ", "\t")):
            # Continuation of a block scalar; anything else indented is ignored.
            if pending_key is not None:
                pending.append(raw.strip())
            continue
        if ":" in raw:
            _flush()
            key, _, value = raw.partition(":")
            value = value.strip()
            if value in {">", "|", ">-", "|-", ">+", "|+"}:
                pending_key = key.strip()
            else:
                data[key.strip()] = _scalar_value(value)
    return None  # unterminated frontmatter


def _scalar_value(raw: str) -> str:
    """Normalise one inline frontmatter value: trim, then unquote once.

    Quotes were previously left in place, so ``description: ""`` parsed as the
    two-character string ``""`` -- non-empty, and therefore passing.
    """
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def _is_vacuous_description(value: str) -> bool:
    """True when the description carries no content a reader could use."""
    return _scalar_value(value) in _VACUOUS_DESCRIPTIONS


def _is_concrete_kb_path(candidate: str) -> bool:
    if any(ch in candidate for ch in "<>{}*"):
        return False
    return candidate == candidate.lower()  # placeholders like YYYYMMDD are uppercase


def make_targets(makefile: Path) -> set[str] | None:
    """Targets defined in ``makefile``, or ``None`` when there is no Makefile."""
    if not makefile.is_file():
        return None
    return set(re.findall(r"^([A-Za-z0-9_.-]+):", makefile.read_text(encoding="utf-8"), re.M))


def code_spans(text: str) -> list[tuple[int, str]]:
    """``(line, span)`` for every fenced-block line and every inline code span."""
    spans: list[tuple[int, str]] = []
    fence: str | None = None
    for idx, line in enumerate(text.splitlines(), start=1):
        opener = _FENCE_RE.match(line)
        if fence is None and opener:
            fence = opener.group(1)
            continue
        if fence is not None:
            if opener and opener.group(1)[0] == fence[0] and len(opener.group(1)) >= len(fence):
                if not line.strip().lstrip(fence[0]):
                    fence = None
                    continue
            spans.append((idx, line))
            continue
        spans += [(idx, m.group(1)) for m in _INLINE_RE.finditer(line)]
    return spans


def make_invocation_targets(span: str) -> list[str]:
    """The targets of every ``make`` command in ``span``, option words skipped.

    ``-C DIR``, ``-f FILE``, ``-j N`` and the other options that take an argument skip
    it; any other ``-x`` word and a ``VAR=value`` assignment are skipped alone.
    """
    targets: list[str] = []
    for match in _MAKE_RE.finditer(span.strip()):
        words = match.group(1).split()
        i = 0
        while i < len(words):
            word = words[i]
            if word in _MAKE_ARG_OPTIONS:
                i += 2
                continue
            if word in _MAKE_NUMBER_OPTIONS and i + 1 < len(words) and words[i + 1].isdigit():
                i += 2
                continue
            if word.startswith("-") or _ASSIGNMENT_RE.match(word):
                i += 1
                continue
            if not _TARGET_RE.fullmatch(word):
                break
            targets.append(word)
            i += 1
    return targets


def _known_skills() -> set[str]:
    """The one skill set S6 checks against: both skill trees plus the built-ins."""
    return known_skills(_skill_trees())


def unknown_skill_names(text: str) -> list[str]:
    """Names ``text`` refers to as skills that do not exist (the rule C6 shares)."""
    return [name for _, name in unknown_skill_references(text, _known_skills())]


def dangling_line_references(
    text: str, root: Path, targets: set[str] | None, skills: set[str]
) -> list[tuple[int, str]]:
    """``(line, finding)`` for each referent in ``text`` that does not exist."""
    findings: list[tuple[int, str]] = []
    for idx, span in code_spans(text):
        if targets is not None:
            for target in make_invocation_targets(span):
                if target not in targets:
                    findings.append((idx, f"make target '{target}'"))
        for path in _SCRIPT_RE.findall(span) + _HARNESS_RE.findall(span):
            if not (root / path).exists():
                findings.append((idx, f"path '{path}'"))
        if "pytest" in span:
            for path in _TESTS_RE.findall(span):
                if not (root / path.split("::")[0]).exists():
                    findings.append((idx, f"pytest path '{path}'"))
    findings += [(idx, f"skill '{name}'") for idx, name in unknown_skill_references(text, skills)]
    return sorted(findings)


def dangling_references(
    skill_md: Path, root: Path, targets: set[str] | None, skills: set[str] | None = None
) -> list[str]:
    """Referents named in ``skill_md``'s code spans that do not exist under ``root``."""
    known = _known_skills() if skills is None else skills
    text = skill_md.read_text(encoding="utf-8")
    return [finding for _, finding in dangling_line_references(text, root, targets, known)]


def _check_references(skill_md: Path, lines: list[str], issues: list[Issue]) -> None:
    """S6, against ``REPO_ROOT``, its Makefile and the skill trees at call time."""
    targets = make_targets(REPO_ROOT / "Makefile")
    known = _known_skills()
    for idx, finding in dangling_line_references("\n".join(lines), REPO_ROOT, targets, known):
        issues.append(Issue("S6", f"{finding} does not exist", skill_md, idx))


def _validate_skill(skill_md: Path, issues: list[Issue], gate: int | None) -> None:
    dir_name = skill_md.parent.name
    lines = skill_md.read_text(encoding="utf-8").splitlines()
    logger.debug("scanning %s (%d lines)", skill_md, len(lines))
    front = _parse_frontmatter(lines)
    if front is None:
        issues.append(Issue("S1", "SKILL.md has no YAML frontmatter block", skill_md, 1))
    else:
        name = front.get("name", "")
        if name != dir_name:
            issues.append(
                Issue("S1", f"frontmatter name '{name}' != directory '{dir_name}'", skill_md, 1)
            )
        description = front.get("description", "")
        if _is_vacuous_description(description):
            issues.append(
                Issue(
                    "S1",
                    f"frontmatter 'description' is vacuous ({description!r}) -- a block-scalar "
                    "indicator or an empty/quoted-empty value is not a description",
                    skill_md,
                    1,
                )
            )
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

    _check_references(skill_md, lines, issues)


def _skill_trees() -> list[Path]:
    """Skill roots to scan, read at call time so tests can monkeypatch either."""
    return [d for d in (SKILLS_DIR, AGENT_SKILLS_DIR) if d.is_dir()]


def _validate_all(issues: list[Issue], gate: int | None) -> None:
    trees = _skill_trees()
    logger.debug("skill trees: %s | coverage gate: %s", [str(t) for t in trees], gate)
    for tree in trees:
        for skill_dir in sorted(p for p in tree.iterdir() if p.is_dir()):
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.is_file():
                issues.append(Issue("S5", "skill directory has no SKILL.md", skill_dir, None))
                continue
            _validate_skill(skill_md, issues, gate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate .claude/skills/ metadata.")
    parser.add_argument("--path", help="validate a single SKILL.md")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log every scanned skill document and resolved gate (stderr)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    gate = _coverage_gate()
    issues: list[Issue] = []
    try:
        if args.path:
            target = Path(args.path).resolve()
            # A missing OR misnamed target is an error, never a silent pass. The
            # if/elif here had no else, so an existing file that is not a
            # SKILL.md recorded nothing and exited 0 -- the PostToolUse hook
            # calls --path, so a misnamed skill file read as validated without
            # ever being opened. Same hole that validate_agents.py had.
            if not target.is_file():
                issues.append(Issue("S1", "path does not exist", target))
            elif target.name != "SKILL.md":
                issues.append(Issue("S1", f"not a SKILL.md file: {target.name}", target))
            else:
                _validate_skill(target, issues, gate)
        else:
            _validate_all(issues, gate)
    except OSError as exc:  # pragma: no cover - defensive I/O guard
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    issues.sort(key=lambda i: (_relpath(i.file).as_posix(), i.line or 0, i.code, i.message))
    errors = [i for i in issues if not i.warning]
    warnings = [i for i in issues if i.warning]
    logger.debug(
        "finished: %d issue(s) -- %d error(s), %d warning(s)",
        len(issues),
        len(errors),
        len(warnings),
    )
    if args.json:
        payload = [
            {
                "code": i.code,
                "message": i.message,
                "file": _relpath(i.file).as_posix(),
                "line": i.line,
                "warning": i.warning,
            }
            for i in issues
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
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
