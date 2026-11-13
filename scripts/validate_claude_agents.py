#!/usr/bin/env python3
"""Deterministic validator for the SDLC role subagents in ``.claude/agents/``.

Eight persisted Claude Code subagents (``sdlc-product-owner`` ... ``sdlc-release-manager``,
local-first plan LF-G13) carry the SDLC roles. They are *not* the ``Agent.md`` persona
docs (``validate_agents.py``) and not skills (``validate_skills.py``); this validator
is the third gate of the three things this repo calls "agents".

Checks (codes are stable and referenced by ``tests/governance/``):

  C0  a ``--path`` target exists and is a ``.claude/agents/*.md`` file (a missing
      or misnamed target is an error, never a silent pass -- the PostToolUse hook
      calls ``--path``)
  C1  frontmatter parses; ``name`` equals the file stem, starts with ``sdlc-`` and
      has no ``:``; nothing but ``sdlc-*.md`` lives in the directory
  C2  ``description`` is not vacuous (length, word count, no placeholders)
  C3  tools: every tool is known and the list EXACTLY matches the role table;
      read-only roles carry no Edit/Write/NotebookEdit; the only frontmatter
      keys are the allowed ones (no ``permissionMode``, no ``mcpServers``); the
      only hooks are the approved ones -- ``readonly_shell_guard.sh`` on every
      role whose Bash is read-only, ``tests_only_write_guard.sh`` on qa-sdet, and
      nothing anywhere else
  C4  ``model`` is valid (``sonnet|opus|haiku|fable|inherit|claude-...``) and equals
      the role table's
  C5  the body has the required sections, a ``**KB role:**`` line matching the role
      table, the hard constraints (no commit to dev/qa/main or tags; the 1,000-game
      cap; maintainer-only actions; no GitHub side effects for non-developer
      roles) and the owner-vs-executor rule (``authored-by: <name>``)
  C6  every skill the file names (frontmatter ``skills`` or "`x` skill") exists
  C7  all eight roles exist, and no other ``sdlc-*`` role does
  C8  each role is dispatched by the skill(s) the role table names
  C9  coverage-percentage literals equal the canonical gate in ``pyproject.toml``,
      and any game-cap literal equals 1,000

Modes: whole directory (default) or ``--path FILE`` (C0-C6 and C9 for one file).
Exit codes: 0 = pass, 1 = validation errors, 2 = usage / internal error.
Stdlib only: the CI governance job installs nothing else.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"
PYPROJECT = REPO_ROOT / "pyproject.toml"

logger = logging.getLogger("validate_claude_agents")

KNOWN_TOOLS = frozenset(
    {
        "Read",
        "Grep",
        "Glob",
        "Bash",
        "PowerShell",
        "Edit",
        "Write",
        "NotebookEdit",
        "WebFetch",
        "WebSearch",
        "Agent",
        "Task",
        "TodoWrite",
        "LSP",
    }
)
WRITE_TOOLS = frozenset({"Edit", "Write", "NotebookEdit"})
NAMED_MODELS = frozenset({"sonnet", "opus", "haiku", "fable", "inherit"})
_FULL_MODEL_RE = re.compile(r"^claude-[a-z0-9][a-z0-9.\-]*$")

# Frontmatter keys a role file may carry. permissionMode / mcpServers /
# disallowedTools / background / omitClaudeMd / memory are deliberately absent: a
# role's authority comes from `tools` and the approved hook, nothing else.
ALLOWED_KEYS = frozenset({"name", "description", "tools", "model", "hooks", "skills", "maxTurns"})
FORBIDDEN_KEYS = frozenset({"permissionMode", "mcpServers"})

# Every hook command ends in `|| exit 2`: if the script cannot start (a path that does not
# resolve, a rename), bash exits 127, which Claude Code treats as a NON-blocking error and
# the command would run unguarded. Exit 2 blocks. A guard that starts and denies exits 0
# with a decision, so the suffix never changes a normal decision.
EXIT_SUFFIX = " || exit 2"
READONLY_GUARD_CMD = (
    'bash "$CLAUDE_PROJECT_DIR"/.claude/hooks/readonly_shell_guard.sh' + EXIT_SUFFIX
)
QA_SHELL_GUARD_CMD = (
    'bash "$CLAUDE_PROJECT_DIR"/.claude/hooks/readonly_shell_guard.sh --qa' + EXIT_SUFFIX
)
TESTS_GUARD_CMD = 'bash "$CLAUDE_PROJECT_DIR"/.claude/hooks/tests_only_write_guard.sh' + EXIT_SUFFIX
# (event, matcher, command) triples a role file may declare.
READONLY_HOOK = ("PreToolUse", "Bash|PowerShell", READONLY_GUARD_CMD)
QA_SHELL_HOOK = ("PreToolUse", "Bash|PowerShell", QA_SHELL_GUARD_CMD)
TESTS_HOOK = ("PreToolUse", "Edit|Write", TESTS_GUARD_CMD)
APPROVED_HOOKS = frozenset({READONLY_HOOK, QA_SHELL_HOOK, TESTS_HOOK})
MIN_HOOK_TIMEOUT = 10  # a hook timeout is non-blocking, so a short one fails open
_HOOK_ENTRY_KEYS = frozenset({"type", "command", "timeout", "statusMessage"})


@dataclass(frozen=True)
class Role:
    name: str
    model: str
    tools: tuple[str, ...] | None  # None = inherit all tools (key omitted)
    kb_role: str
    hooks: tuple[tuple[str, str, str], ...]
    dispatched_by: tuple[str, ...]
    owner_note: bool = True  # read-only roles must name `authored-by: <name>`

    @property
    def read_only(self) -> bool:
        return self.tools is not None and not (set(self.tools) & WRITE_TOOLS)


_RO = ("Read", "Grep", "Glob")
ROLES: dict[str, Role] = {
    r.name: r
    for r in (
        Role("sdlc-product-owner", "sonnet", _RO, "Planning", (), ("planning-handoff",)),
        Role(
            "sdlc-scrum-master",
            "sonnet",
            (*_RO, "Bash"),
            "Planning",
            (READONLY_HOOK,),
            ("planning-handoff",),
        ),
        Role(
            "sdlc-architect",
            "opus",
            (*_RO, "Bash"),
            "Planning",
            (READONLY_HOOK,),
            ("planning-handoff",),
        ),
        Role(
            "sdlc-ml-scientist",
            "opus",
            (*_RO, "Bash"),
            "Planning",
            (READONLY_HOOK,),
            ("planning-handoff",),
        ),
        Role(
            "sdlc-ai-engineer",
            "opus",
            (*_RO, "Bash"),
            "Planning",
            (READONLY_HOOK,),
            ("planning-handoff",),
        ),
        Role(
            "sdlc-developer",
            "inherit",
            None,
            "DEV-SQE",
            (),
            ("dev-sqe-handoff",),
            owner_note=False,
        ),
        Role(
            "sdlc-qa-sdet",
            "sonnet",
            (*_RO, "Bash", "Edit", "Write"),
            "DEV-SQE + Pre-PR",
            (QA_SHELL_HOOK, TESTS_HOOK),
            ("dev-sqe-handoff", "pre-pr-handoff"),
            owner_note=False,
        ),
        Role(
            "sdlc-release-manager",
            "sonnet",
            (*_RO, "Bash"),
            "Pre-PR",
            (READONLY_HOOK,),
            ("pre-pr-handoff",),
        ),
    )
}

REQUIRED_SECTIONS = (
    "## Role",
    "## Accountable for",
    "## Reviews",
    "## Inputs and outputs",
    "## Constraints",
    "## Hand-off",
)

_PLACEHOLDER_RE = re.compile(r"\b(todo|tbd|fixme|lorem|placeholder|xxx|description here)\b", re.I)
_SKILL_REF_RE = re.compile(r"`([a-z0-9][a-z0-9-]*)` skill\b")
_KB_ROLE_RE = re.compile(r"^\*\*KB role:\*\*\s*(?P<role>.+?)\s*$", re.M)
_COVERAGE_LINE_RE = re.compile(r"coverage|cov-fail-under", re.I)
_PERCENT_RE = re.compile(r"(?<![\d.])(\d{2,3})\s*%")
_COV_FLAG_RE = re.compile(r"cov-fail-under[=\s]+(\d{2,3})")
_FAIL_UNDER_RE = re.compile(r"fail_under\s*=\s*(\d+)")
_CAP_LINE_RE = re.compile(r"\bgames?\b.*\bcap\b|\bcap\b.*\bgames?\b", re.I)
_CAP_NUM_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d{3,})(?![\d,])")
_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*):(?:\s+(.*))?$")


@dataclass
class Issue:
    code: str
    message: str
    file: Path
    line: int | None = None
    warning: bool = False

    def render(self) -> str:
        rel: Path | str = self.file
        if self.file.is_absolute():
            try:
                rel = self.file.relative_to(REPO_ROOT)
            except ValueError:
                rel = self.file
        where = f"{rel}:{self.line}" if self.line else str(rel)
        level = "WARNING" if self.warning else "ERROR"
        return f"{level} {where}: {self.code} {self.message}"


# --- a small YAML-subset reader (stdlib only; the governance job has no PyYAML) -----


_PLAIN_FORBIDDEN = re.compile(r"[{}\[\]*&#!|>%@`]|: |:$| #")
# A plain (unquoted) scalar is accepted ONLY if it matches this whitelist. A blacklist missed
# `description: - text` and a colon followed by a TAB, both of which Claude Code fails to
# parse (twice: plain, then its quoting retry) and so loads the role with a name only.
_PLAIN_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._,/()+=-]*$")


def _scalar(raw: str) -> str | list[str]:
    """Read one scalar STRICTLY: anything Claude Code's loader could read differently raises.

    Claude Code parses frontmatter with a YAML loader and, if that fails, retries once after
    quoting values that contain one of ``{}[]*&#!|>%@`` or a backtick or ``: ``. If the retry
    fails too, the file loads with every field but the name silently dropped -- no ``tools``
    (so every tool is inherited) and no ``hooks``. So this reader accepts only values that are
    unambiguous under both: balanced quotes, no flow mapping, no comment, and no plain scalar
    carrying a character the loader treats specially.
    """
    s = raw.strip(" ")
    if s.startswith('"'):
        if len(s) < 2 or not s.endswith('"'):
            raise ValueError(f"unbalanced double quote in value {raw!r}")
        out: list[str] = []
        k = 1
        while k < len(s) - 1:
            ch = s[k]
            if ch == "\\":
                if k + 1 >= len(s) - 1 or s[k + 1] not in '"\\':
                    raise ValueError(f"unsupported escape in value {raw!r}")
                out.append(s[k + 1])
                k += 2
                continue
            if ch == '"':
                raise ValueError(f"unescaped double quote inside value {raw!r}")
            out.append(ch)
            k += 1
        return "".join(out)
    if s.startswith("'"):
        if len(s) < 2 or not s.endswith("'"):
            raise ValueError(f"unbalanced single quote in value {raw!r}")
        inner = s[1:-1]
        if "'" in inner.replace("''", ""):
            raise ValueError(f"unescaped single quote inside value {raw!r}")
        return inner.replace("''", "'")
    if s.startswith("["):
        if not s.endswith("]"):
            raise ValueError(f"unbalanced flow sequence {raw!r}")
        inner = s[1:-1].strip(" ")
        items = [p.strip(" ") for p in inner.split(",")] if inner else []
        for item in items:
            if not item or not _PLAIN_OK.match(item):
                raise ValueError(f"flow sequence item {item!r} is not a plain word")
        return items
    if not s:
        return s
    if not _PLAIN_OK.match(s) or _PLAIN_FORBIDDEN.search(s):
        raise ValueError(
            f"plain value {raw!r} is outside the conservative whitelist "
            "([A-Za-z0-9][A-Za-z0-9 ._,/()+=-]*, no colon, no leading - ? [ { & * ! | > % @); quote it"
        )
    return s


def _parse_node(items: list[tuple[int, str]], pos: int, indent: int) -> tuple[object, int]:
    if items[pos][1].startswith("-"):
        out_list: list[object] = []
        while pos < len(items) and items[pos][0] == indent and items[pos][1].startswith("-"):
            text = items[pos][1]
            rest = text[1:].strip(" ")
            if not rest:
                if pos + 1 >= len(items) or items[pos + 1][0] <= indent:
                    raise ValueError("empty list item")
                value, pos = _parse_node(items, pos + 1, items[pos + 1][0])
                out_list.append(value)
            elif _KEY_RE.match(rest):
                inner = indent + (len(text) - len(rest))
                items[pos] = (inner, rest)
                value, pos = _parse_node(items, pos, inner)
                out_list.append(value)
            else:
                out_list.append(_scalar(rest))
                pos += 1
        return out_list, pos
    out: dict[str, object] = {}
    while pos < len(items) and items[pos][0] == indent and not items[pos][1].startswith("-"):
        match = _KEY_RE.match(items[pos][1])
        if match is None:
            raise ValueError(f"cannot parse line {items[pos][1]!r}")
        key, rest = match.group(1), match.group(2)
        if key == "description" and not (rest or "").strip(" ").startswith('"'):
            raise ValueError("description must be a double-quoted string (quote it)")
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        if rest is None or rest.strip(" ") == "":
            nxt = pos + 1
            if nxt < len(items) and (
                items[nxt][0] > indent
                or (items[nxt][0] == indent and items[nxt][1].startswith("-"))
            ):
                out[key], pos = _parse_node(items, nxt, items[nxt][0])
            else:
                out[key] = ""
                pos += 1
        else:
            out[key] = _scalar(rest)
            pos += 1
    return out, pos


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Return ``(frontmatter, body)``; raise ``ValueError`` when malformed or ambiguous.

    Strict on purpose. Claude Code ends the block at the FIRST ``---`` anywhere after the opening
    fence (a lazy regex, roughly dashes-newline then anything up to the first three dashes), so a ``---`` inside a description or a comment
    truncates it; this reader refuses any ``---`` other than the closing line. It also refuses
    comment lines, which the YAML loader and the fallback treat differently.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip(" ") != "---":
        raise ValueError("file does not start with a '---' frontmatter fence")
    rest = "\n".join(lines[1:])
    idx = rest.find("---")
    if idx == -1:
        raise ValueError("frontmatter is never closed with '---'")
    line_end = rest.find("\n", idx)
    closing = rest[idx : line_end if line_end >= 0 else None]
    if (idx != 0 and rest[idx - 1] != "\n") or closing.strip(" ") != "---":
        raise ValueError(
            "'---' appears inside the frontmatter; Claude Code ends the block at the first '---' anywhere"
        )
    block = rest[:idx]
    if "\t" in block:
        raise ValueError("a TAB in the frontmatter (YAML and Claude Code read it differently)")
    # Printable ASCII and newlines only. Python's str.strip() removes NBSP, VT, FF, NEL and U+3000,
    # but Bun.YAML keeps them as part of the key (the role then loads with every tool and no guard),
    # and a NUL makes the loader reject the whole block. CRLF was normalised above, so a remaining
    # CR is a stray one. Nothing legitimate is lost: every role file's frontmatter is pure ASCII.
    for ch in block:
        if ch != "\n" and not 0x20 <= ord(ch) <= 0x7E:
            raise ValueError(
                f"character U+{ord(ch):04X} in the frontmatter; only printable ASCII is allowed "
                "(the YAML loader treats it differently from this reader)"
            )
    body = rest[line_end + 1 :] if line_end >= 0 else ""
    items: list[tuple[int, str]] = []
    for raw in block.split("\n"):
        if not raw.strip(" "):
            continue
        if raw.lstrip(" ").startswith("#"):
            raise ValueError(f"comment line {raw.strip(' ')!r} is not allowed in frontmatter")
        items.append((len(raw) - len(raw.lstrip(" ")), raw.strip(" ")))
    if not items:
        raise ValueError("frontmatter is empty")
    value, pos = _parse_node(items, 0, items[0][0])
    if pos != len(items) or not isinstance(value, dict):
        raise ValueError("frontmatter is not a single mapping")
    return value, body


# --- helpers -------------------------------------------------------------------


def _coverage_gate() -> int | None:
    if not PYPROJECT.is_file():
        return None
    match = _FAIL_UNDER_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def _tool_list(value: object) -> tuple[str, ...] | None:
    if isinstance(value, str):
        return tuple(p.strip() for p in value.split(",") if p.strip())
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return tuple(v.strip() for v in value)
    return None


def _skill_exists(name: str) -> bool:
    return (SKILLS_DIR / name / "SKILL.md").is_file()


def _section(body: str, heading: str) -> str:
    """Text under ``heading`` up to the next ``## `` heading ('' when absent)."""
    match = re.search(rf"^{re.escape(heading)}\s*$", body, re.M)
    if match is None:
        return ""
    rest = body[match.end() :]
    nxt = re.search(r"^## ", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


def _declared_hooks(front: dict[str, object]) -> tuple[set[tuple[str, str, str]], list[str]]:
    """Flatten ``hooks`` into (event, matcher, command) triples plus any shape errors."""
    hooks = front.get("hooks")
    problems: list[str] = []
    found: set[tuple[str, str, str]] = set()
    if hooks in (None, ""):
        return found, problems
    if not isinstance(hooks, dict):
        return found, ["'hooks' must be a mapping of event -> list"]
    for event, entries in hooks.items():
        if not isinstance(entries, list):
            problems.append(f"hooks.{event} must be a list")
            continue
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - {"matcher", "hooks"}:
                problems.append(f"hooks.{event} entry must have only 'matcher' and 'hooks'")
                continue
            matcher = entry.get("matcher")
            inner = entry.get("hooks")
            if not isinstance(matcher, str) or not isinstance(inner, list):
                problems.append(f"hooks.{event} entry needs a string matcher and a hooks list")
                continue
            for hook in inner:
                if (
                    not isinstance(hook, dict)
                    or hook.get("type") != "command"
                    or not isinstance(hook.get("command"), str)
                    or set(hook) - _HOOK_ENTRY_KEYS
                ):
                    problems.append(f"hooks.{event} contains a non-'command' or malformed hook")
                    continue
                timeout = hook.get("timeout")
                if timeout is not None and not (
                    isinstance(timeout, str)
                    and timeout.isdigit()
                    and int(timeout) >= MIN_HOOK_TIMEOUT
                ):
                    problems.append(
                        f"hook timeout {timeout!r} must be >= {MIN_HOOK_TIMEOUT} seconds (a hook timeout is non-blocking)"
                    )
                found.add((event, matcher, str(hook["command"]).strip()))
    return found, problems


# --- per-file checks -----------------------------------------------------------


@dataclass
class _Ctx:
    """What every per-file check needs."""

    path: Path
    front: dict[str, object]
    body: str
    gate: int | None
    issues: list[Issue]
    name: str = ""
    role: Role | None = None

    def add(self, code: str, message: str) -> None:
        self.issues.append(Issue(code, message, self.path))


def _check_identity(c: _Ctx) -> None:
    """C1: name, stem and prefix; resolves the role table entry."""
    name = c.front.get("name")
    stem = c.path.stem
    if not isinstance(name, str) or not name:
        c.add("C1", "frontmatter has no 'name'")
        name = ""
    else:
        if ":" in name:
            c.add("C1", f"name {name!r} contains ':'")
        if name != stem:
            c.add("C1", f"name {name!r} must equal the file stem {stem!r}")
        if not name.startswith("sdlc-"):
            c.add("C1", f"name {name!r} must start with 'sdlc-'")
    c.name = name
    c.role = ROLES.get(name) or ROLES.get(stem)


def _check_description(c: _Ctx) -> None:
    """C2: the description says what the role is for and when to use it."""
    front, role, add = c.front, c.role, c.add
    desc = front.get("description")
    if not isinstance(desc, str):
        add("C2", "frontmatter has no 'description'")
    else:
        words = desc.split()
        if len(desc) < 40 or len(words) < 8:
            add(
                "C2",
                f"description is vacuous ({len(words)} words, {len(desc)} chars; need >= 8 / 40)",
            )
        if _PLACEHOLDER_RE.search(desc):
            add("C2", "description contains a placeholder word")
        if role is not None and not any(w in desc.lower() for w in ("use ", "when ")):
            add("C2", "description must say when to use the role ('Use ...' / 'when ...')")


def _check_tools_and_hooks(c: _Ctx) -> None:
    """C3: keys, tools, hooks."""
    front, role, add = c.front, c.role, c.add
    for key in sorted(front):
        if key in FORBIDDEN_KEYS:
            add(
                "C3",
                f"'{key}' is forbidden: a role's authority comes from tools and the approved hook",
            )
        elif key not in ALLOWED_KEYS:
            add("C3", f"unknown frontmatter key '{key}'")
    tools: tuple[str, ...] | None = None
    if "tools" in front:
        tools = _tool_list(front["tools"])
        if tools is None:
            add("C3", "'tools' must be a comma-separated string or a list of names")
        else:
            for tool in tools:
                if tool not in KNOWN_TOOLS:
                    add("C3", f"unknown tool {tool!r}")
    if role is not None:
        if role.tools is None:
            if "tools" in front:
                add("C3", "sdlc-developer inherits all tools: omit 'tools'")
        elif tools is not None and tuple(tools) != role.tools:
            add("C3", f"tools {', '.join(tools)} != role table {', '.join(role.tools)}")
        elif tools is None and "tools" not in front:
            add("C3", f"missing 'tools' (role table: {', '.join(role.tools)})")
        if role.read_only and tools is not None and set(tools) & WRITE_TOOLS:
            add("C3", "a read-only role must not have Edit/Write/NotebookEdit")
    declared, problems = _declared_hooks(front)
    for problem in problems:
        add("C3", problem)
    for hook in sorted(declared):
        if hook not in APPROVED_HOOKS:
            add(
                "C3",
                (
                    f"hook command {hook[2]!r} must end with '{EXIT_SUFFIX.strip()}' so a guard that cannot start blocks"
                    if any(
                        hook[2] + EXIT_SUFFIX == a[2] and hook[:2] == a[:2] for a in APPROVED_HOOKS
                    )
                    else f"unapproved hook {hook!r}; approved: the read-only shell guard (plain or --qa) and the tests-only write guard"
                ),
            )
    if role is not None:
        wanted = set(role.hooks)
        if not problems and declared != wanted:
            missing = sorted(wanted - declared)
            extra = sorted(declared - wanted)
            if missing:
                add("C3", f"required hook missing: {missing}")
            if extra:
                add("C3", f"hook not allowed for this role: {extra}")
    for script in ("readonly_shell_guard.sh", "tests_only_write_guard.sh"):
        if any(script in h[2] for h in declared) and not (HOOKS_DIR / script).is_file():
            add("C3", f"hook script .claude/hooks/{script} does not exist")


def _check_model(c: _Ctx) -> None:
    """C4: the model is valid and equals the role table."""
    front, role, add = c.front, c.role, c.add
    model = front.get("model")
    if not isinstance(model, str) or not model:
        add("C4", "frontmatter has no 'model'")
    else:
        if model not in NAMED_MODELS and not _FULL_MODEL_RE.match(model):
            add("C4", f"model {model!r} is not sonnet|opus|haiku|fable|inherit|claude-<id>")
        if role is not None and model != role.model:
            add("C4", f"model {model!r} != role table {role.model!r}")


def _check_body(c: _Ctx) -> None:
    """C5: required sections, KB role, hard constraints, owner-vs-executor."""
    body, role, name, add = c.body, c.role, c.name, c.add
    for heading in REQUIRED_SECTIONS:
        if not re.search(rf"^{re.escape(heading)}\s*$", body, re.M):
            add("C5", f"missing required section '{heading}'")
    kb = _KB_ROLE_RE.search(body)
    if kb is None:
        add("C5", "missing a '**KB role:** <role>' line")
    elif role is not None and kb.group("role") != role.kb_role:
        add("C5", f"KB role {kb.group('role')!r} != role table {role.kb_role!r}")
    constraints = _section(body, "## Constraints")
    if constraints:
        commit_lines = [ln for ln in constraints.splitlines() if re.search(r"\bcommit", ln, re.I)]
        if not any(
            re.search(r"`dev`", ln)
            and re.search(r"`qa`", ln)
            and re.search(r"`main`", ln)
            and "tag" in ln
            for ln in commit_lines
        ):
            add("C5", "Constraints must forbid commits to `dev`, `qa`, `main` and tags (one line)")
        if "1,000" not in constraints:
            add("C5", "Constraints must state the 1,000-game cap")
        if "maintainer" not in constraints.lower():
            add(
                "C5", "Constraints must state that merges into qa/main and tags are maintainer-only"
            )
        if name != "sdlc-developer" and not re.search(r"no github side effects", constraints, re.I):
            add("C5", "Constraints must say 'No GitHub side effects' for a non-developer role")
    if name == "sdlc-qa-sdet" and not (
        "not a sandbox" in body and "pytest executes test code" in body
    ):
        add(
            "C5",
            "sdlc-qa-sdet must state the residual risk: pytest executes test code, so its guard is not a sandbox",
        )
    if role is not None and role.owner_note and f"authored-by: {role.name}" not in body:
        add("C5", f"a read-only role must name its commit trailer 'authored-by: {role.name}'")
    if name == "sdlc-developer" and "authored-by: sdlc-" not in body:
        add("C5", "sdlc-developer must state the 'authored-by: sdlc-<role>' commit trailer rule")


def _check_skills(c: _Ctx) -> None:
    """C6: every skill the file names exists."""
    front, body, add = c.front, c.body, c.add
    named: list[str] = []
    fm_skills = front.get("skills")
    if isinstance(fm_skills, str) and fm_skills:
        named += [s.strip() for s in fm_skills.split(",") if s.strip()]
    elif isinstance(fm_skills, list):
        named += [str(s) for s in fm_skills]
    named += _SKILL_REF_RE.findall(body)
    for skill in sorted(set(named)):
        if not _skill_exists(skill):
            add("C6", f"skill {skill!r} does not exist under .claude/skills/")


def _check_literals(c: _Ctx) -> None:
    """C9: coverage and game-cap literals."""
    path, body, gate, issues = c.path, c.body, c.gate, c.issues
    for idx, raw in enumerate(body.splitlines(), start=1):
        if gate is not None and _COVERAGE_LINE_RE.search(raw):
            for found in _PERCENT_RE.findall(raw) + _COV_FLAG_RE.findall(raw):
                if int(found) != gate:
                    issues.append(
                        Issue("C9", f"coverage literal {found}% != gate {gate}%", path, idx)
                    )
        if _CAP_LINE_RE.search(raw):
            for found in _CAP_NUM_RE.findall(raw):
                if int(found.replace(",", "")) != 1000:
                    issues.append(Issue("C9", f"game-cap literal {found} != 1,000", path, idx))


def validate_agent_file(path: Path, issues: list[Issue], gate: int | None) -> None:
    try:
        front, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        issues.append(Issue("C1", f"frontmatter: {exc}", path))
        return
    logger.debug("validating %s", path)
    c = _Ctx(path, front, body, gate, issues)
    _check_identity(c)
    for check in (
        _check_description,
        _check_tools_and_hooks,
        _check_model,
        _check_body,
        _check_skills,
        _check_literals,
    ):
        check(c)


# --- whole-directory checks ----------------------------------------------------


@dataclass
class _Context:
    issues: list[Issue] = field(default_factory=list)


def _skill_texts() -> dict[str, str]:
    out: dict[str, str] = {}
    if SKILLS_DIR.is_dir():
        for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
            out[skill_md.parent.name] = skill_md.read_text(encoding="utf-8")
    return out


def validate_all(issues: list[Issue], gate: int | None) -> None:
    files = sorted(AGENTS_DIR.glob("*")) if AGENTS_DIR.is_dir() else []
    present: set[str] = set()
    for path in files:
        if not path.is_file():
            continue
        if path.suffix != ".md" or not path.stem.startswith("sdlc-"):
            issues.append(Issue("C1", "only sdlc-*.md role files belong in .claude/agents/", path))
            continue
        present.add(path.stem)
        validate_agent_file(path, issues, gate)

    # C7 -- the role set is closed.
    for name in sorted(ROLES):
        if name not in present:
            issues.append(Issue("C7", f"role {name} has no .claude/agents/{name}.md", AGENTS_DIR))
    for name in sorted(present - set(ROLES)):
        issues.append(
            Issue("C7", f"{name} is not one of the eight roles", AGENTS_DIR / f"{name}.md")
        )

    # C8 -- each role is dispatched by the skills the table names.
    skills = _skill_texts()
    for role in ROLES.values():
        for skill in role.dispatched_by:
            text = skills.get(skill)
            if text is None:
                issues.append(
                    Issue("C8", f"dispatching skill {skill!r} does not exist", SKILLS_DIR)
                )
            elif role.name not in text:
                issues.append(
                    Issue(
                        "C8",
                        f"skill {skill} never dispatches {role.name}",
                        SKILLS_DIR / skill / "SKILL.md",
                    )
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate .claude/agents SDLC role subagents.")
    parser.add_argument("--path", help="validate a single .claude/agents/*.md file")
    parser.add_argument("--quiet", action="store_true", help="suppress the OK line")
    parser.add_argument("--verbose", action="store_true", help="trace what was scanned (stderr)")
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
            if not target.is_file():
                issues.append(Issue("C0", "path does not exist", target))
            elif target.suffix != ".md" or target.parent.resolve() != AGENTS_DIR.resolve():
                issues.append(
                    Issue("C0", f"not a .claude/agents/*.md role file: {target.name}", target)
                )
            else:
                validate_agent_file(target, issues, gate)
        else:
            validate_all(issues, gate)
    except OSError as exc:  # pragma: no cover - filesystem failure
        print(f"claude-agents: internal error: {exc}", file=sys.stderr)
        return 2

    errors = [i for i in issues if not i.warning]
    for issue in issues:
        print(issue.render(), file=sys.stderr)
    if errors:
        return 1
    if not args.quiet:
        print("claude-agents: OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
