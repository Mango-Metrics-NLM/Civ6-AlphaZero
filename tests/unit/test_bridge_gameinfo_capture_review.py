"""GM-B4 review round: refusals and bookkeeping of ``alpha_zero.bridge.gameinfo_capture``.

Symlinks into a checkout (D1), I/O failures (D2), complete captures only (D3), which log
message was used (D4), chunk bookkeeping (D5), no-overwrite without hard links (D6),
file names (D7) and the repository's own tree (D8). No Lua runtime is needed.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.fixtures import gameinfo_snapshot as g

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_ENV = "CIV6AZ_DATA_ROOT"
DOC = g.MINIMAL_DOCUMENT
OLDER: dict[str, Any] = {**DOC, "Technologies": DOC["Technologies"][:1], "TechnologyPrereqs": []}


@pytest.fixture
def gc() -> ModuleType:
    return importlib.import_module("alpha_zero.bridge.gameinfo_capture")


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv(DATA_ROOT_ENV, str(data))
    monkeypatch.chdir(tmp_path)
    return data


def _files(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())


def _checkout(tmp_path: Path) -> Path:
    """A fake checkout (a ``.git`` directory) with a ``docs`` subdirectory."""
    checkout = tmp_path / "work"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "docs").mkdir()
    return checkout


def _run(gc: ModuleType, *argv: str) -> int:
    try:
        return int(gc.main(list(argv)))
    except SystemExit as exit_:
        return int(exit_.code or 0)


def _fail_one_line(capsys: pytest.CaptureFixture[str]) -> str:
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1 and lines[0]
    return lines[0]


def _source(tmp_path: Path, document: dict[str, Any] = DOC) -> Path:
    path = tmp_path / "gameinfo.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture
def repo_probe() -> Iterator[Path]:
    """A directory name inside the real checkout, removed afterwards whatever happens."""
    probe = REPO_ROOT / "gameinfo_capture_test_probe_dir"
    assert not probe.exists()
    yield probe
    shutil.rmtree(probe, ignore_errors=True)


# ---------------------------------------------------------------------------
# D1: where the file would really land
# ---------------------------------------------------------------------------
class TestSymlinksIntoACheckout:
    def test_gameinfo_directory_linked_to_a_checkout_subdirectory(
        self, gc: ModuleType, root: Path, tmp_path: Path
    ) -> None:
        checkout = _checkout(tmp_path)
        (root / "gameinfo").symlink_to(checkout / "docs", target_is_directory=True)
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(DOC, name="probe")
        assert _files(checkout) == []

    def test_data_root_linked_to_a_checkout_subdirectory(
        self, gc: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkout = _checkout(tmp_path)
        (tmp_path / "link").symlink_to(checkout / "docs", target_is_directory=True)
        monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "link"))
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(DOC, name="probe")
        assert _files(checkout) == []

    def test_the_command_line_refuses_the_same_way(
        self,
        gc: ModuleType,
        root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        checkout = _checkout(tmp_path)
        (root / "gameinfo").symlink_to(checkout / "docs", target_is_directory=True)
        assert _run(gc, "--from-file", str(_source(tmp_path)), "--name", "probe") == 1
        _fail_one_line(capsys)
        assert _files(checkout) == []


# ---------------------------------------------------------------------------
# D2: an I/O failure is a refusal, not a traceback
# ---------------------------------------------------------------------------
def _boom(*_args: Any, **_kwargs: Any) -> Any:
    raise OSError("simulated I/O failure")


class TestIoFailures:
    def _file_as_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setenv(DATA_ROOT_ENV, str(blocker))
        return blocker

    def _file_as_subdirectory(self, root: Path) -> Path:
        blocker = root / "gameinfo"
        blocker.write_text("x", encoding="utf-8")
        return blocker

    def _mkstemp_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tempfile, "mkstemp", _boom)

    def _dump_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(json, "dump", _boom)

    @pytest.mark.parametrize("how", ["root-is-a-file", "gameinfo-is-a-file", "mkstemp", "dump"])
    def test_write_snapshot_raises_a_capture_error_and_leaves_nothing(
        self, gc: ModuleType, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
    ) -> None:
        if how == "root-is-a-file":
            self._file_as_root(tmp_path, monkeypatch)
        elif how == "gameinfo-is-a-file":
            self._file_as_subdirectory(root)
        elif how == "mkstemp":
            self._mkstemp_fails(monkeypatch)
        else:
            self._dump_fails(monkeypatch)
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(DOC, name="probe")
        assert [f for f in _files(tmp_path) if ".json" in f or ".tmp" in f] == []

    @pytest.mark.parametrize("how", ["root-is-a-file", "gameinfo-is-a-file", "mkstemp", "dump"])
    def test_the_command_line_exits_one_with_one_line(
        self,
        gc: ModuleType,
        root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        how: str,
    ) -> None:
        source = _source(tmp_path)
        if how == "root-is-a-file":
            self._file_as_root(tmp_path, monkeypatch)
        elif how == "gameinfo-is-a-file":
            self._file_as_subdirectory(root)
        elif how == "mkstemp":
            self._mkstemp_fails(monkeypatch)
        else:
            self._dump_fails(monkeypatch)
        assert _run(gc, "--from-file", str(source), "--name", "probe") == 1
        _fail_one_line(capsys)

    def test_a_failed_dump_leaves_no_temporary_file(
        self, gc: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._dump_fails(monkeypatch)
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(DOC, name="probe")
        assert _files(root) == []


# ---------------------------------------------------------------------------
# D3: a capture is complete
# ---------------------------------------------------------------------------
class TestCompleteCapturesOnly:
    @pytest.mark.parametrize("missing", g.TABLES)
    def test_an_absent_table_is_refused_by_name(self, gc: ModuleType, missing: str) -> None:
        document = {k: v for k, v in g.STUB_DOCUMENT.items() if k != missing}
        with pytest.raises(gc.GameInfoCaptureError) as caught:
            gc.check_snapshot(document)
        assert missing in str(caught.value)

    def test_every_absent_table_is_named(self, gc: ModuleType) -> None:
        gone = ("Buildings", "Civics", "Units")
        document = {k: v for k, v in g.STUB_DOCUMENT.items() if k not in gone}
        with pytest.raises(gc.GameInfoCaptureError) as caught:
            gc.check_snapshot(document)
        assert all(name in str(caught.value) for name in gone)

    def test_present_and_empty_tables_are_fine(self, gc: ModuleType) -> None:
        document = {**g.STUB_DOCUMENT, "Units": [], "Buildings": [], "CivicPrereqs": []}
        summary = gc.check_snapshot(document)
        assert (summary.units, summary.buildings) == (0, 0)

    def test_the_full_stub_document_is_accepted(self, gc: ModuleType) -> None:
        assert gc.check_snapshot(g.STUB_DOCUMENT).techs == 4

    @pytest.mark.parametrize("missing", ["Units", "Buildings", "Civics", "CivicPrereqs"])
    def test_write_snapshot_and_the_command_line_refuse_an_incomplete_capture(
        self,
        gc: ModuleType,
        root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        missing: str,
    ) -> None:
        document = {k: v for k, v in g.STUB_DOCUMENT.items() if k != missing}
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(document, name="probe")
        assert _run(gc, "--from-file", str(_source(tmp_path, document))) == 1
        assert missing in _fail_one_line(capsys)
        assert _files(root) == []


# ---------------------------------------------------------------------------
# D4: which message was used
# ---------------------------------------------------------------------------
def _broken_tail(kind: str) -> str:
    lines = g.chunk_lines(DOC)
    if kind == "no-end":
        lines = lines[:-1]
    elif kind == "missing-piece":
        del lines[1]
    else:
        lines[-1] = g.end_line(len(lines), g.adler32(g.compact_text(DOC)))
    return "".join(lines)


TAILS = ["no-end", "missing-piece", "end-too-large"]


class TestScanLog:
    def test_an_empty_log(self, gc: ModuleType) -> None:
        scan = gc.scan_log("nothing\n")
        assert (scan.document, scan.complete, scan.later_incomplete) == (None, 0, False)

    def test_one_single_line_message(self, gc: ModuleType) -> None:
        scan = gc.scan_log(g.single_line(DOC, "AlphaZeroMain: "))
        assert (scan.document, scan.complete, scan.later_incomplete) == (DOC, 1, False)

    def test_one_chunked_message(self, gc: ModuleType) -> None:
        scan = gc.scan_log("".join(g.chunk_lines(DOC)))
        assert (scan.document, scan.complete, scan.later_incomplete) == (DOC, 1, False)

    def test_several_messages_count_and_the_last_is_the_document(self, gc: ModuleType) -> None:
        text = g.single_line(OLDER) + "".join(g.chunk_lines(OLDER)) + g.single_line(DOC)
        scan = gc.scan_log(text)
        assert (scan.document, scan.complete, scan.later_incomplete) == (DOC, 3, False)

    @pytest.mark.parametrize("kind", TAILS)
    def test_an_unfinished_message_after_the_last_complete_one(
        self, gc: ModuleType, kind: str
    ) -> None:
        scan = gc.scan_log(g.single_line(OLDER) + _broken_tail(kind))
        assert (scan.document, scan.complete, scan.later_incomplete) == (OLDER, 1, True)

    @pytest.mark.parametrize("kind", TAILS)
    def test_an_unfinished_message_before_a_complete_one_is_not_later(
        self, gc: ModuleType, kind: str
    ) -> None:
        scan = gc.scan_log(_broken_tail(kind) + g.single_line(DOC))
        assert (scan.document, scan.complete, scan.later_incomplete) == (DOC, 1, False)

    def test_snapshot_from_log_is_the_scans_document_or_an_error(self, gc: ModuleType) -> None:
        assert gc.snapshot_from_log(g.single_line(OLDER) + _broken_tail("no-end")) == OLDER
        with pytest.raises(gc.GameInfoCaptureError):
            gc.snapshot_from_log(_broken_tail("no-end"))
        with pytest.raises(gc.GameInfoCaptureError):
            gc.snapshot_from_log("AZ_GAMEINFO:[]\n")


class TestCommandLineWhichMessage:
    def _log(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "Lua.log"
        path.write_text(text, encoding="utf-8")
        return path

    @pytest.mark.parametrize("kind", TAILS)
    def test_a_newer_incomplete_message_is_refused_unless_allowed(
        self,
        gc: ModuleType,
        root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        kind: str,
    ) -> None:
        log = self._log(tmp_path, g.single_line(OLDER) + _broken_tail(kind))
        assert _run(gc, "--from-log", str(log), "--name", "n") == 1
        message = _fail_one_line(capsys)
        assert "incomplete" in message and "--allow-earlier" in message
        assert _files(root) == []

        assert _run(gc, "--from-log", str(log), "--name", "n", "--allow-earlier") == 0
        assert "earlier" in capsys.readouterr().err.lower()
        assert json.loads((root / "gameinfo" / "n.json").read_text(encoding="utf-8")) == OLDER

    def test_several_complete_messages_file_the_last_and_say_how_many(
        self, gc: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log = self._log(tmp_path, g.single_line(OLDER) + g.single_line(OLDER) + g.single_line(DOC))
        assert _run(gc, "--from-log", str(log), "--name", "n") == 0
        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        assert any(
            "message" in line and re.search(r"\b3\b", line) for line in combined.splitlines()
        ), combined
        assert json.loads((root / "gameinfo" / "n.json").read_text(encoding="utf-8")) == DOC

    def test_allow_earlier_is_harmless_with_a_file(
        self, gc: ModuleType, root: Path, tmp_path: Path
    ) -> None:
        argv = ["--from-file", str(_source(tmp_path)), "--name", "n", "--allow-earlier"]
        assert _run(gc, *argv) == 0
        assert json.loads((root / "gameinfo" / "n.json").read_text(encoding="utf-8")) == DOC


# ---------------------------------------------------------------------------
# D5: chunk bookkeeping
# ---------------------------------------------------------------------------
class TestChunkBookkeeping:
    def test_headers_disagreeing_with_the_end_count_make_the_set_incomplete(
        self, gc: ModuleType
    ) -> None:
        mismatched = "".join(g.chunk_lines(DOC, 3, header_total=4))
        scan = gc.scan_log(g.single_line(OLDER) + mismatched)
        assert (scan.document, scan.complete, scan.later_incomplete) == (OLDER, 1, True)

    def test_an_end_count_disagreeing_with_the_headers_makes_the_set_incomplete(
        self, gc: ModuleType
    ) -> None:
        lines = g.chunk_lines(DOC, 3)
        lines[-1] = g.end_line(2, g.adler32(g.compact_text(DOC)))
        scan = gc.scan_log(g.single_line(OLDER) + "".join(lines))
        assert (scan.document, scan.complete, scan.later_incomplete) == (OLDER, 1, True)

    def test_chunks_that_disagree_with_each_other_make_the_set_incomplete(
        self, gc: ModuleType
    ) -> None:
        lines = g.chunk_lines(DOC, 2)
        lines[1] = lines[1].replace("2/2:", "2/3:")
        scan = gc.scan_log(g.single_line(OLDER) + "".join(lines))
        assert (scan.document, scan.complete, scan.later_incomplete) == (OLDER, 1, True)

    def test_a_lone_inconsistent_set_is_not_a_snapshot(self, gc: ModuleType) -> None:
        with pytest.raises(gc.GameInfoCaptureError):
            gc.snapshot_from_log("".join(g.chunk_lines(DOC, 3, header_total=4)))

    def test_a_consistent_set_is_still_complete(self, gc: ModuleType) -> None:
        scan = gc.scan_log("".join(g.chunk_lines(DOC, 3)))
        assert (scan.document, scan.complete) == (DOC, 1)


# ---------------------------------------------------------------------------
# D6: no overwrite without hard links
# ---------------------------------------------------------------------------
class TestNoHardLinks:
    def test_the_file_is_written_by_a_rename_when_links_fail(
        self, gc: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "link", _boom)
        path = gc.write_snapshot(DOC, name="probe")
        assert path == root / "gameinfo" / "probe.json"
        assert json.loads(path.read_text(encoding="utf-8")) == DOC
        assert _files(root) == ["gameinfo/probe.json"]

    def test_an_existing_destination_is_still_refused(
        self, gc: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = gc.write_snapshot(DOC, name="probe")
        before = first.read_bytes()
        monkeypatch.setattr(os, "link", _boom)
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(OLDER, name="probe")
        assert first.read_bytes() == before
        assert _files(root) == ["gameinfo/probe.json"]

    def test_links_and_renames_both_failing_leave_nothing(
        self, gc: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("link", "replace", "rename"):
            monkeypatch.setattr(os, name, _boom)
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(DOC, name="probe")
        assert _files(root) == []


# ---------------------------------------------------------------------------
# D7: names
# ---------------------------------------------------------------------------
DEVICE_NAMES = [
    "CON",
    "con",
    "nul",
    "NUL.backup",
    "PRN",
    "Aux",
    "AUX.txt",
    "COM1",
    "com9",
    "Com3.json",
    "LPT1",
    "lpt9",
    "LPT5.x.y",
]


class TestNames:
    @pytest.mark.parametrize("name", [*DEVICE_NAMES, "a" * 101])
    def test_a_device_name_or_a_long_name_is_refused(
        self, gc: ModuleType, root: Path, name: str
    ) -> None:
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(DOC, name=name)
        assert _files(root.parent) == []

    @pytest.mark.parametrize("name", ["console", "nullable", "com", "lpt", "a" * 100])
    def test_a_name_that_only_starts_like_a_device_or_is_at_the_limit_is_accepted(
        self, gc: ModuleType, root: Path, name: str
    ) -> None:
        assert gc.write_snapshot(DOC, name=name).name == f"{name}.json"

    def test_the_command_line_refuses_a_device_name(
        self, gc: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(gc, "--from-file", str(_source(tmp_path)), "--name", "NUL.backup") == 1
        _fail_one_line(capsys)
        assert _files(root) == []


# ---------------------------------------------------------------------------
# D8: the repository's own tree, with no .git to find
# ---------------------------------------------------------------------------
class TestRepositoryTree:
    @pytest.mark.parametrize("how", ["direct", "via-symlink"])
    def test_a_destination_inside_the_checkout_root_is_refused_without_a_git_directory(
        self,
        gc: ModuleType,
        repo_probe: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        how: str,
    ) -> None:
        rules_gameinfo = importlib.import_module("alpha_zero.rules.gameinfo")
        monkeypatch.setattr(rules_gameinfo, "enclosing_checkout", lambda _path: None)
        assert rules_gameinfo.checkout_root() == REPO_ROOT, "premise: this is the checkout"
        if how == "direct":
            data = repo_probe
        else:
            repo_probe.mkdir()
            data = tmp_path / "link"
            data.symlink_to(repo_probe, target_is_directory=True)
        monkeypatch.setenv(DATA_ROOT_ENV, str(data))
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(DOC, name="probe")
        assert not repo_probe.exists() or _files(repo_probe) == []
        if how == "direct":
            assert not repo_probe.exists(), "a directory was created inside the checkout"
