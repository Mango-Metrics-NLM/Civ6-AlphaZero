"""An edit to the registered A1 protocol must be deliberate (T13 review I3).

``docs/EVALUATION.md`` section 7.1 is frozen by its sha256. The digest is
recomputed from the text, so on its own nothing would notice an edit -- a
formatter rewriting a fenced block, or a substantive change slipped into a docs
PR. The "Registration log" table **after** the end marker (outside the hashed
region) records every registered digest; the last row must equal the digest of
the text as it stands. Changing the region therefore needs a new log row in the
same change, which a reviewer sees.

Negative control (in the file): one character changed inside the region moves
the digest away from the recorded one, and the same change outside the region
does not.
"""

from __future__ import annotations

import re
from pathlib import Path

from alpha_zero.evaluation.a1_protocol import (
    PROTOCOL_END,
    protocol_sha256,
    registered_protocol,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_MD = REPO_ROOT / "docs" / "EVALUATION.md"

_LOG_HEADING = "**Registration log.**"
_ROW_RE = re.compile(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*`([0-9a-f]{64})`\s*\|(.+)\|\s*$")


def registration_log(text: str) -> list[tuple[str, str, str]]:
    """``[(date, digest, change)]`` from the log table after the end marker."""
    text = text.replace("\r\n", "\n")
    tail = text[text.index(PROTOCOL_END) :]
    assert _LOG_HEADING in tail, "the registration log must follow the end marker"
    rows = []
    for line in tail[tail.index(_LOG_HEADING) :].splitlines():
        match = _ROW_RE.match(line)
        if match:
            rows.append((match.group(1), match.group(2), match.group(3).strip()))
    return rows


def _text() -> str:
    return EVALUATION_MD.read_text(encoding="utf-8")


def test_the_last_logged_digest_is_the_current_one() -> None:
    text = _text()
    rows = registration_log(text)
    assert rows, "no registered digest is recorded"
    current = protocol_sha256(text)
    assert rows[-1][1] == current, (
        f"section 7.1 now hashes to {current}, but the registration log's last row "
        f"records {rows[-1][1]}. Edit the region only deliberately, and add a log row "
        "(date, new digest, what changed and why) in the same change."
    )


def test_every_logged_change_says_what_changed() -> None:
    for date, _digest, change in registration_log(_text()):
        assert len(change) >= 10, f"log row {date} does not say what changed"


def test_the_log_is_outside_the_hashed_region() -> None:
    text = _text()
    region = registered_protocol(text)
    assert _LOG_HEADING not in region
    for _date, digest, _change in registration_log(text):
        assert digest not in region


def test_negative_control_an_edit_inside_the_region_is_caught() -> None:
    text = _text()
    recorded = registration_log(text)[-1][1]
    region = registered_protocol(text)
    edited_inside = text.replace(region, region.replace("VOID", "VOlD", 1), 1)
    assert protocol_sha256(edited_inside) != recorded
    edited_outside = text.replace("## 8.", "## 8 .", 1)
    assert protocol_sha256(edited_outside) == recorded


def test_the_region_holds_no_python_fence() -> None:
    """``ruff format`` (>= 0.16) rewrites fenced Python in Markdown; none may sit in the region."""
    region = registered_protocol(_text())
    fences = re.findall(r"^\s*```(\w*)", region, re.MULTILINE)
    assert all(lang in ("", "text") for lang in fences), fences


def test_the_registration_constants_were_confirmed_against_the_last_row() -> None:
    """``a1_registration.REGISTRATION_DIGEST`` pins the constants read from prose (T14 M-3).

    A1-R's seed, the data root and the verdict table's thresholds are code, not
    parsed. A re-registration must re-confirm them: update REGISTRATION_DIGEST
    in the same change, after checking each one against the new text.
    """
    from alpha_zero.evaluation.a1_registration import REGISTRATION_DIGEST

    assert registration_log(_text())[-1][1] == REGISTRATION_DIGEST
