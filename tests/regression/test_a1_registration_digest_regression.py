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

import pytest

from alpha_zero.evaluation.a1_protocol import (
    REGISTRATION_LOG_HEADING,
    digest_report,
    protocol_sha256,
    registered_protocol,
    registration_log,
)
from alpha_zero.evaluation.a1_protocol import main as a1_protocol_main

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_MD = REPO_ROOT / "docs" / "EVALUATION.md"


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
    assert REGISTRATION_LOG_HEADING not in region
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


# ---------------------------------------------------------------------------
# ``python -m alpha_zero.evaluation.a1_protocol digest`` -- the re-registration gate
#
# Negative control (recorded, then reverted): making ``digest_report`` return
# ``agree=True`` unconditionally made the two "disagree" tests below fail.
# ---------------------------------------------------------------------------


def test_the_digest_command_agrees_on_the_committed_registration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert a1_protocol_main(["digest"]) == 0
    out = capsys.readouterr().out
    assert protocol_sha256(_text()) in out and "AGREE" in out


def test_the_digest_command_fails_when_the_region_changed_without_a_log_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    edited = tmp_path / "EVALUATION.md"
    text = _text()
    region = registered_protocol(text)
    edited.write_text(text.replace(region, region + "An unlogged edit.\n", 1), encoding="utf-8")
    assert a1_protocol_main(["digest", "--file", str(edited)]) == 1
    out = capsys.readouterr().out
    assert "add a registration-log row" in out and "REGISTRATION_DIGEST" in out


def test_digest_report_names_a_stale_constant() -> None:
    lines, agree = digest_report(_text(), "0" * 64)
    assert not agree
    assert any("re-confirm the constants" in line for line in lines)
    assert not any("add a registration-log row" in line for line in lines)


def test_the_digest_command_reports_an_unreadable_file(tmp_path: Path) -> None:
    assert a1_protocol_main(["digest", "--file", str(tmp_path / "missing.md")]) == 2
