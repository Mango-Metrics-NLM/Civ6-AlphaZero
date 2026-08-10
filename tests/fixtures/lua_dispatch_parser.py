"""Parser for the Lua ``ACTION_HANDLERS`` dispatch table.

v1.8.0 W2.1 step 6 — used by the action-mask invariant test to assert
that every ``ActionType`` enum value the Python side advertises has a
matching entry in the Lua dispatcher AND that the entry's handler
function is NOT ``executeNotImplemented`` (i.e. the action is actually
implemented, not just present as a stub).

Why a parser instead of executing the Lua: the Civ6 Lua runtime is
sandboxed (no ``lupa`` / ``lua-c-api`` available in CI). The parser
reads ``mod/AlphaZeroAI/Scripts/ActionExecutor.lua`` as text and
extracts the two structures the invariant cares about:

1. ``ACTION_TYPES``: the string-keyed table of action-type identifiers
   (the Lua mirror of Python ``ActionType``).
2. ``ACTION_HANDLERS``: the dispatch table that maps each action type
   to a handler function name.

Handles: line comments (``-- ...``), inline comments after entries,
trailing commas, multi-line tables, and whitespace variation.

Does NOT handle:
- Conditional ``if FEATURE_X then handler = ...`` branches.
- Runtime mutation of the table (Civ6 mod tables are static at load
  time in this codebase).
- Lua long-string literals containing ``--`` (e.g. ``[[option--value]]``).
  The comment-stripper would corrupt the affected line, but the failure
  mode is *loud, not silent*: the truncated line will not match the
  ``KEY = "value"`` regex, so the entry is dropped and the resulting
  count mismatch surfaces as either ``ACTION_TYPES block not found``,
  ``contained no entries``, or (downstream) the action-mask invariant
  test's enum-mismatch assertion. No silent corruption — just a less
  helpful first-line error message than ideal. Documented per
  gap-analysis I4 on commit 3572832; a future fix would two-pass strip
  (long-strings first, then line comments) if any string value ever
  legitimately contains ``--``.

Public API:

    parse_action_types(lua_source) -> dict[str, str]
    parse_action_handlers(lua_source) -> dict[str, str]
    load_dispatch_from_file(path) -> tuple[dict[str, str], dict[str, str]]

Raises :class:`LuaParseError` on malformed input rather than returning
partial dicts that could silently mask the very gaps the invariant test
is meant to catch.
"""

from __future__ import annotations

import re
from pathlib import Path


class LuaParseError(ValueError):
    """Raised when the Lua dispatch tables cannot be parsed.

    Used in place of returning a partial dict so the invariant test
    fails loud rather than green-with-missing-entries.
    """


# Sentinel name the Lua side uses for the stub handler. The invariant
# test compares against this so adding a real handler == removing the
# stub assignment.
NOT_IMPLEMENTED_HANDLER = "executeNotImplemented"

# Strip Lua line comments while preserving line numbers (we don't need them
# for the regex but keeping the line structure makes the parser's regex
# patterns predictable).
_LINE_COMMENT = re.compile(r"--[^\n]*")

# ACTION_TYPES = { KEY = "value", KEY = "value", ... }
_ACTION_TYPES_BLOCK = re.compile(
    r"local\s+ACTION_TYPES\s*=\s*\{(?P<body>.*?)\}",
    re.DOTALL,
)
_ACTION_TYPES_ENTRY = re.compile(
    r"(?P<key>[A-Z_][A-Z0-9_]*)\s*=\s*\"(?P<value>[^\"]+)\"",
)

# ACTION_HANDLERS = { [ACTION_TYPES.KEY] = funcName, ... }
_ACTION_HANDLERS_BLOCK = re.compile(
    r"local\s+ACTION_HANDLERS\s*=\s*\{(?P<body>.*?)\}",
    re.DOTALL,
)
_ACTION_HANDLERS_ENTRY = re.compile(
    r"\[\s*ACTION_TYPES\.(?P<key>[A-Z_][A-Z0-9_]*)\s*\]\s*=\s*(?P<func>[A-Za-z_][A-Za-z0-9_]*)",
)


def _strip_comments(lua_source: str) -> str:
    """Remove Lua line comments. Preserves newlines so blocks stay intact."""
    return _LINE_COMMENT.sub("", lua_source)


def parse_action_types(lua_source: str) -> dict[str, str]:
    """Parse the ``ACTION_TYPES = { ... }`` block.

    Args:
        lua_source: Full text of ``ActionExecutor.lua``.

    Returns:
        Dict mapping the Lua identifier (e.g. ``"UNIT_MOVE"``) to its
        string value (e.g. ``"unit_move"``).

    Raises:
        LuaParseError: If the block is missing or contains no entries.
    """
    source = _strip_comments(lua_source)
    match = _ACTION_TYPES_BLOCK.search(source)
    if match is None:
        raise LuaParseError(
            "ACTION_TYPES block not found in Lua source. "
            "Did the table definition change shape (e.g. moved into a module return)?"
        )

    entries: dict[str, str] = {}
    for entry in _ACTION_TYPES_ENTRY.finditer(match.group("body")):
        key = entry.group("key")
        value = entry.group("value")
        if key in entries:
            raise LuaParseError(f"Duplicate ACTION_TYPES key: {key!r}")
        entries[key] = value

    if not entries:
        raise LuaParseError("ACTION_TYPES block parsed but contained no entries.")

    return entries


def parse_action_handlers(lua_source: str) -> dict[str, str]:
    """Parse the ``ACTION_HANDLERS = { ... }`` dispatch table.

    Args:
        lua_source: Full text of ``ActionExecutor.lua``.

    Returns:
        Dict mapping the Lua ``ACTION_TYPES`` enum identifier
        (e.g. ``"UNIT_MOVE"``) to the handler function name
        (e.g. ``"executeUnitMove"`` or :data:`NOT_IMPLEMENTED_HANDLER`).

    Raises:
        LuaParseError: If the block is missing or has zero entries.
    """
    source = _strip_comments(lua_source)
    match = _ACTION_HANDLERS_BLOCK.search(source)
    if match is None:
        raise LuaParseError(
            "ACTION_HANDLERS block not found. The invariant test cannot "
            "verify Python/Lua dispatch sync without it."
        )

    entries: dict[str, str] = {}
    for entry in _ACTION_HANDLERS_ENTRY.finditer(match.group("body")):
        key = entry.group("key")
        func = entry.group("func")
        if key in entries:
            raise LuaParseError(f"Duplicate ACTION_HANDLERS key: {key!r}")
        entries[key] = func

    if not entries:
        raise LuaParseError("ACTION_HANDLERS block parsed but contained no entries.")

    return entries


def load_dispatch_from_file(
    path: Path | str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Read a Lua file and parse both tables.

    Convenience wrapper used by :mod:`tests.unit.test_bridge_action_invariant`.

    Args:
        path: Path to ``ActionExecutor.lua`` (or a test-fixture snippet).

    Returns:
        ``(action_types, action_handlers)`` dicts.
    """
    text = Path(path).read_text(encoding="utf-8")
    return parse_action_types(text), parse_action_handlers(text)


def implemented_action_values(
    action_types: dict[str, str],
    action_handlers: dict[str, str],
) -> set[str]:
    """Return the set of action-value strings whose Lua handler is real.

    "Real" means: the dispatch table entry exists AND the mapped handler
    name is not :data:`NOT_IMPLEMENTED_HANDLER`.

    Args:
        action_types: Output of :func:`parse_action_types`.
        action_handlers: Output of :func:`parse_action_handlers`.

    Returns:
        Set of ``ActionType`` value strings (the lowercase form, e.g.
        ``"unit_move"``) the Lua dispatcher can actually execute.
    """
    implemented: set[str] = set()
    for key, handler_name in action_handlers.items():
        if handler_name == NOT_IMPLEMENTED_HANDLER:
            continue
        value = action_types.get(key)
        if value is not None:
            implemented.add(value)
    return implemented
