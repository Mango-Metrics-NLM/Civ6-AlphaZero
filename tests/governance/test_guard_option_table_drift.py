"""The read-only guard's per-script option table must not drift from the scripts.

The table in ``readonly_shell_guard.sh`` (``script_check``) is hand-written: it lists, per
vetted script, the exact long options the guard will let a role pass. argparse accepts any
unambiguous abbreviation, so only exact members are safe, and a script that gains a writing
option must not silently inherit shell access. This test AST-scans each script's
``add_argument`` calls and asserts ``table == scan - EXCLUDED``.

It also fails closed on anything the scan cannot see: a non-literal option name, ``parents=``,
subparsers, argument groups or ``allow_abbrev`` -- those would make the scan an
under-approximation, which is the failure this test exists to prevent.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / ".claude" / "hooks" / "readonly_shell_guard.sh"

# Options the scripts define but the guard deliberately does NOT allow, with the reason.
EXCLUDED: dict[str, dict[str, str]] = {
    "scripts/check_learning_trend.py": {"--out": "writes a report file"},
}
# Short options the guard allows beyond the long ones (argparse prefix matching only
# applies to long options).
SHORT_ALLOWED: dict[str, set[str]] = {"scripts/check_coverage_floors.py": {"-v"}}

_ENTRY = re.compile(r'^\s+(scripts/[A-Za-z0-9_]+\.py\)) opts="([^"]*)"', re.M)


def _guard_table() -> dict[str, set[str]]:
    text = GUARD.read_text(encoding="utf-8")
    table = {m.group(1)[:-1]: set(m.group(2).split()) for m in _ENTRY.finditer(text)}
    assert table, "could not parse the option table out of readonly_shell_guard.sh"
    return table


def _scan(script: Path) -> set[str]:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in {"add_subparsers", "add_argument_group", "add_mutually_exclusive_group"}:
            pytest.fail(f"{script.name}: {name} is not understood by the drift scan")
        if name == "ArgumentParser":
            for kw in node.keywords:
                if kw.arg in {"parents", "allow_abbrev", "prefix_chars", "fromfile_prefix_chars"}:
                    pytest.fail(
                        f"{script.name}: ArgumentParser({kw.arg}=...) changes the option surface"
                    )
        if name == "add_argument":
            for arg in node.args:
                if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                    pytest.fail(f"{script.name}: non-literal option name in add_argument")
                if arg.value.startswith("-"):
                    found.add(arg.value)
    return found


def test_the_table_covers_exactly_the_ten_vetted_scripts() -> None:
    table = _guard_table()
    assert len(table) == 10, sorted(table)
    for rel in table:
        assert (REPO_ROOT / rel).is_file(), f"{rel} is in the guard table but does not exist"


@pytest.mark.parametrize("rel", sorted(_guard_table()))
def test_guard_options_equal_the_argparse_scan(rel: str) -> None:
    scanned = _scan(REPO_ROOT / rel)
    long_scanned = {o for o in scanned if o.startswith("--")}
    short_scanned = {o for o in scanned if not o.startswith("--")}
    excluded = set(EXCLUDED.get(rel, {}))
    expected_long = long_scanned - excluded
    table = _guard_table()[rel]
    table_long = {o for o in table if o.startswith("--")}
    table_short = {o for o in table if not o.startswith("--")}
    assert table_long == expected_long, (
        f"{rel}: guard table {sorted(table_long)} != argparse scan minus exclusions "
        f"{sorted(expected_long)}. A new option must be reviewed and added (or excluded with a reason); "
        "a removed one must leave the table."
    )
    assert table_short == SHORT_ALLOWED.get(rel, set()), (
        f"{rel}: short options in the table {sorted(table_short)} are not the reviewed set"
    )
    assert short_scanned >= table_short
    assert excluded <= long_scanned, f"{rel}: EXCLUDED names an option the script no longer has"


def test_every_exclusion_has_a_reason() -> None:
    for rel, options in EXCLUDED.items():
        for option, reason in options.items():
            assert reason.strip(), f"{rel} {option}: exclusion without a reason"
