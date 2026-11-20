"""The rule-set log in ``docs/GAME_RULES.md`` section 2 equals the frozen pins (GM-K1).

Same shape as ``test_a1_registration_digest_regression``: the log and the code each
state a hash, and this test holds them equal, so a frozen rule set cannot change
or appear in one place only. Drafts are never logged. The comparison is a function
exercised on injected inputs, so it is not vacuous while both sides are empty.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from alpha_zero.rules.builtin import BUILTIN, STATUS_DRAFT, frozen_pins

REPO_ROOT = Path(__file__).resolve().parents[2]
GAME_RULES_MD = REPO_ROOT / "docs" / "GAME_RULES.md"
LOG_HEADING = "## 2. Rule-set log"
HEX64 = re.compile(r"[0-9a-f]{64}")


def log_rows(text: str) -> list[tuple[str, str]]:
    """``(id, sha256)`` of each data row of the section 2 table; ``[]`` if it is empty."""
    start = text.index(LOG_HEADING)
    rest = text[start + len(LOG_HEADING) :]
    nxt = re.search(r"^## ", rest, re.MULTILINE)
    section = rest[: nxt.start()] if nxt else rest
    rows: list[tuple[str, str]] = []
    header_seen = False
    for line in section.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if not header_seen:
            header_seen = True
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append((cells[0], cells[1] if len(cells) > 1 else ""))
    return rows


def log_problems(
    rows: Iterable[tuple[str, str]], pins: Mapping[str, str], draft_ids: Iterable[str]
) -> list[str]:
    """Every way the log disagrees with the frozen pins; each names the id."""
    problems: list[str] = []
    drafts = set(draft_ids)
    seen: dict[str, str] = {}
    for rule_id, digest in rows:
        if rule_id in drafts or rule_id.endswith("-draft"):
            problems.append(f"{rule_id}: a draft is never logged")
        elif rule_id in seen:
            problems.append(f"{rule_id}: logged more than once")
        elif rule_id not in pins:
            problems.append(f"{rule_id}: logged but not a frozen built-in (extra row)")
        elif digest != pins[rule_id]:
            problems.append(f"{rule_id}: log says {digest}, builtin pins {pins[rule_id]}")
        seen.setdefault(rule_id, digest)
    for rule_id in pins:
        if rule_id not in seen:
            problems.append(f"{rule_id}: frozen in builtin but missing from the log")
    return problems


def _table(*rows: str) -> str:
    head = (
        "| Id | Content sha256 | Date | What changed | OpenSpec change |\n|---|---|---|---|---|\n"
    )
    return f"intro\n\n{LOG_HEADING}\n\nAppend-only.\n\n{head}{''.join(r + chr(10) for r in rows)}\n## 3. Next\n| x | y |\n"


A, B = "a" * 64, "b" * 64


def test_the_log_equals_the_frozen_pins() -> None:
    text = GAME_RULES_MD.read_text(encoding="utf-8")
    drafts = [name for name, e in BUILTIN.items() if e.status == STATUS_DRAFT]
    problems = log_problems(log_rows(text), frozen_pins(), drafts)
    assert not problems, "docs/GAME_RULES.md section 2 and builtin.py disagree: " + "; ".join(
        problems
    )


def test_the_real_table_is_found_and_parses() -> None:
    text = GAME_RULES_MD.read_text(encoding="utf-8")
    assert LOG_HEADING in text
    for rule_id, digest in log_rows(text):
        assert HEX64.fullmatch(digest), (rule_id, digest)


def test_empty_log_and_no_pins_agree() -> None:
    assert log_problems([], {}, ["x-draft"]) == []
    assert log_rows(_table()) == []


def test_parser_reads_rows_and_stops_at_the_next_section() -> None:
    text = _table(f"| `a-v1` | {A} | 2026-11-19 | first | add-x |", f"| b-v1 | {B} | d | c | o |")
    assert log_rows(text) == [("a-v1", A), ("b-v1", B)]


def test_matching_rows_pass() -> None:
    assert log_problems([("a-v1", A)], {"a-v1": A}, []) == []


def test_missing_row_fails_and_names_the_id() -> None:
    (problem,) = log_problems([], {"a-v1": A}, [])
    assert "a-v1" in problem and "missing" in problem


def test_extra_row_fails_and_names_the_id() -> None:
    (problem,) = log_problems([("z-v1", A)], {}, [])
    assert "z-v1" in problem and "extra" in problem


def test_hash_mismatch_fails_and_names_the_id() -> None:
    (problem,) = log_problems([("a-v1", B)], {"a-v1": A}, [])
    assert "a-v1" in problem and A in problem and B in problem


def test_a_draft_row_fails_and_names_the_id() -> None:
    (problem,) = log_problems([("s-v1-draft", A)], {}, ["s-v1-draft"])
    assert "s-v1-draft" in problem and "draft" in problem
    # a draft is rejected by name even if the caller forgot to list it
    assert log_problems([("t-draft", A)], {}, []) != []


def test_a_duplicate_row_fails() -> None:
    problems = log_problems([("a-v1", A), ("a-v1", A)], {"a-v1": A}, [])
    assert problems and "a-v1" in problems[0]
