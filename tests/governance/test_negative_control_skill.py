"""H4: the negative-control skill never restores a planted defect with git.

Slip: the agent undid a planted defect with ``git checkout <file>`` and lost
uncommitted work in the same file. The skill must forbid that, say why, and give a
backup-and-copy-back procedure.
"""

from __future__ import annotations

import re
from pathlib import Path

SKILL = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "negative-control" / "SKILL.md"
_GIT_RESTORE = re.compile(r"git\s+(checkout|restore|stash)\b", re.I)
_PROHIBITION = re.compile(r"never[^\n]*(git checkout|git restore|git stash)", re.I)
_NEGATION = re.compile(r"\b(never|not|do not|don't|must not)\b", re.I)


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_skill_prohibits_restoring_with_git() -> None:
    text = _text()
    assert _PROHIBITION.search(text), "no 'never git checkout/restore/stash' sentence"
    for verb in ("git checkout", "git restore", "git stash"):
        assert verb in text, verb


def test_no_step_tells_the_reader_to_restore_with_git() -> None:
    text = _text()
    # Every mention of a git restoring verb must sit on a line that forbids it.
    for line in text.splitlines():
        if _GIT_RESTORE.search(line):
            assert _NEGATION.search(line), line
    # Fenced command blocks must not contain such a command at all.
    for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.S):
        assert not _GIT_RESTORE.search(block), block


def test_skill_states_why_in_a_sentence() -> None:
    text = _text().lower()
    assert re.search(r"uncommitted[^.\n]*(lose|lost|discard|overwrit|destroy)", text) or (
        re.search(r"(lose|lost|discard|overwrit|destroy)[^.\n]*uncommitted", text)
    )


def test_skill_requires_clean_file_or_backup_of_the_file_as_it_is() -> None:
    text = _text().lower()
    assert "no uncommitted changes" in text
    assert "as it is" in text


def test_backup_goes_to_scratchpad_or_temp_by_full_path_never_beside_the_file() -> None:
    text = _text().lower()
    assert "scratchpad" in text
    assert "temporary directory" in text
    assert "full path" in text
    assert "beside the file" in text


def test_restore_copies_back_and_verifies_with_checksum_or_diff() -> None:
    text = _text()
    assert re.search(r"sha256sum|checksum|\bcmp\b|\bdiff\b", text)
    assert re.search(r"equals? the backup|identical to the backup|matches the backup", text, re.I)


def test_generated_files_are_backed_up_and_restored_too() -> None:
    text = _text().lower()
    assert "generated file" in text
    assert "lua bundle" in text


def test_restore_does_not_rely_on_a_shell_variable_surviving() -> None:
    # Each tool call is a fresh shell: a $BACKUP_DIR set in step 2 is empty in step 4.
    blocks = [b for b in re.findall(r"```[a-z]*\n(.*?)```", _text(), re.S) if "cmp" in b]
    assert blocks, "no restore block with a cmp check"
    for block in blocks:
        assert "$BACKUP_DIR" not in block and "${BACKUP_DIR}" not in block, block


def test_skill_says_to_print_the_backup_directory_and_use_its_literal_path() -> None:
    text = _text().lower()
    assert re.search(r"\bprint\w*\b[^.\n]*(backup|director)", text)
    assert "literal" in text and "absolute path" in text
    assert re.search(r"(between|across)[^.\n]*(calls|commands)", text)


def test_backups_are_named_by_repository_path_not_base_name() -> None:
    text = _text().lower()
    assert "repository path" in text
    assert "base name" in text or "basename" in text


def test_skill_says_to_stop_when_cmp_reports_a_difference() -> None:
    assert re.search(r"stop[^.\n]*(cmp|differ)", _text(), re.I)
