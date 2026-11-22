"""GM-B4 second review round: three findings on ``alpha_zero.bridge.gameinfo_capture``.

1. Pieces of two exports must not be joined into one snapshot: the END line carries the
   Adler-32 of the whole JSON text (``AZ_GAMEINFO_END:<n>:<adler32>``) and a chunk set is
   complete only when the joined text matches it.
2. Without hard links, publishing claims the name with an exclusive create, so a file that
   appears meanwhile is never replaced.
3. ``AZ_GAMEINFO_BEGIN`` announces an export attempt: a newer attempt that left no complete
   message makes the earlier one "not the newest".

No Lua runtime is needed. Tests marked "control" pass on the pre-fix tree on purpose; they
pin behaviour the fix must keep, or reject what the fix must keep rejecting.
"""

from __future__ import annotations

import builtins
import errno
import importlib
import io
import json
import os
import pathlib
import tracemalloc
import zlib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from alpha_zero.bridge.log_tailer import LogTailer
from tests.fixtures import gameinfo_snapshot as g

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
    return sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())


def _run(gc: ModuleType, *argv: str) -> int:
    try:
        return int(gc.main(list(argv)))
    except SystemExit as exit_:
        return int(exit_.code or 0)


def _log(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "Lua.log"
    path.write_text(text, encoding="utf-8")
    return path


def _filed(root: Path, name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(
        (root / "gameinfo" / f"{name}.json").read_text(encoding="utf-8")
    )
    return loaded


def _cut(text: str, pieces: int) -> list[str]:
    size = -(-len(text) // pieces)
    return [text[i : i + size] for i in range(0, len(text), size)]


def _chunks(parts: list[str], first: int, total: int, prefix: str = "") -> list[str]:
    """Chunk lines for ``parts`` numbered from ``first`` of ``total``."""
    return [
        f"{prefix}AZ_GAMEINFO_CHUNK:{i}/{total}:{p}\n" for i, p in enumerate(parts, start=first)
    ]


def _scan(gc: ModuleType, text: str) -> tuple[Any, int, bool]:
    scan = gc.scan_log(text)
    return scan.document, scan.complete, scan.later_incomplete


# ---------------------------------------------------------------------------
# Finding 1: the checksum
# ---------------------------------------------------------------------------
def _doc(cost: int, unit: str) -> dict[str, Any]:
    """Documents that differ in the FIRST piece (a cost) and in the LAST (a unit name)."""
    return {
        **DOC,
        "Technologies": [
            {"TechnologyType": "TECH_A", "Cost": cost, "EraType": "ERA_ANCIENT"},
            {"TechnologyType": "TECH_B", "Cost": 20, "EraType": "ERA_ANCIENT"},
        ],
        "Units": [{"UnitType": unit, "Cost": 50, "Combat": 10, "BaseMoves": 2}],
    }


DOC_A = _doc(10, "UNIT_AAAAAAAA")
DOC_B = _doc(20, "UNIT_BBBBBBBB")
TEXT_A = g.compact_text(DOC_A)
TEXT_B = g.compact_text(DOC_B)
PIECES = 3


def _mixture_text() -> str:
    a, b = _cut(TEXT_A, PIECES), _cut(TEXT_B, PIECES)
    return a[0] + "".join(b[1:])


class TestTheMixturePremises:
    """The scenario is only a scenario if these hold; they do not depend on the fix."""

    def test_the_two_exports_have_equal_length_and_piece_count(self) -> None:
        assert len(TEXT_A) == len(TEXT_B)
        assert len(_cut(TEXT_A, PIECES)) == len(_cut(TEXT_B, PIECES)) == PIECES

    def test_they_differ_in_the_first_and_in_a_later_piece(self) -> None:
        a, b = _cut(TEXT_A, PIECES), _cut(TEXT_B, PIECES)
        assert a[0] != b[0]
        assert a[-1] != b[-1]

    def test_the_mixture_is_valid_json_and_neither_export(self) -> None:
        mixture = json.loads(_mixture_text())
        assert isinstance(mixture, dict)
        assert mixture != DOC_A and mixture != DOC_B
        assert g.adler32(_mixture_text()) not in (g.adler32(TEXT_A), g.adler32(TEXT_B))


def _lost_end_then_lost_chunk_one(end: str, prefix: str = "") -> str:
    """Export A loses its END line; export B loses its chunk 1; B's END arrives.

    ``end`` is how B's END line is written: ``"old"`` (no checksum, the pre-fix format),
    ``"new"`` (B's own, correct Adler-32 of B's whole text).
    """
    a, b = _cut(TEXT_A, PIECES), _cut(TEXT_B, PIECES)
    checksum = None if end == "old" else g.adler32(TEXT_B)
    return (
        "".join(_chunks(a, 1, PIECES, prefix))
        + "".join(_chunks(b[1:], 2, PIECES, prefix))
        + g.end_line(PIECES, checksum, prefix)
    )


END_STYLES = ["old", "new"]


class TestPiecesOfTwoExportsAreNotJoined:
    @pytest.mark.parametrize("end", END_STYLES)
    def test_nothing_is_accepted_on_its_own(self, gc: ModuleType, end: str) -> None:
        assert _scan(gc, _lost_end_then_lost_chunk_one(end)) == (None, 0, False)

    @pytest.mark.parametrize("end", END_STYLES)
    def test_an_earlier_message_stays_the_document_and_the_newer_is_unfinished(
        self, gc: ModuleType, end: str
    ) -> None:
        text = g.single_line(OLDER, "AlphaZeroMain: ") + _lost_end_then_lost_chunk_one(
            end, "AlphaZeroMain: "
        )
        assert _scan(gc, text) == (OLDER, 1, True)

    @pytest.mark.parametrize("end", END_STYLES)
    def test_the_command_line_refuses_and_files_nothing(
        self,
        gc: ModuleType,
        root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        end: str,
    ) -> None:
        only = _log(tmp_path, _lost_end_then_lost_chunk_one(end))
        assert _run(gc, "--from-log", str(only), "--name", "n") == 1
        assert "no complete GameInfo message" in capsys.readouterr().err
        assert _files(root) == []

        earlier = _log(tmp_path, g.single_line(OLDER) + _lost_end_then_lost_chunk_one(end))
        assert _run(gc, "--from-log", str(earlier), "--name", "n") == 1
        message = capsys.readouterr().err
        assert "incomplete" in message and "--allow-earlier" in message
        assert _files(root) == []

    @pytest.mark.parametrize("end", END_STYLES)
    def test_allow_earlier_files_the_earlier_one_never_the_mixture(
        self, gc: ModuleType, root: Path, tmp_path: Path, end: str
    ) -> None:
        log = _log(tmp_path, g.single_line(OLDER) + _lost_end_then_lost_chunk_one(end))
        assert _run(gc, "--from-log", str(log), "--name", "n", "--allow-earlier") == 0
        assert _filed(root, "n") == OLDER

    def test_a_mixture_whose_checksum_matches_what_was_joined_is_a_message(
        self, gc: ModuleType
    ) -> None:
        # A checksum that matches the joined text IS the proof of integrity; this pins that
        # the scanner checks the text it joined, not "the first export's" or "the last's".
        a, b = _cut(TEXT_A, PIECES), _cut(TEXT_B, PIECES)
        mixture = _mixture_text()
        text = (
            "".join(_chunks([a[0]], 1, PIECES))
            + "".join(_chunks(b[1:], 2, PIECES))
            + g.end_line(PIECES, g.adler32(mixture))
        )
        document, complete, later = _scan(gc, text)
        assert document == json.loads(mixture) and (complete, later) == (1, False)


class TestTheChecksum:
    def _set(self, document: dict[str, Any], pieces: int = 3, **kw: Any) -> str:
        return "".join(g.chunk_lines(document, pieces, **kw))

    def test_a_correct_chunk_set_is_accepted(self, gc: ModuleType) -> None:
        for pieces in (2, 3, 7):
            assert _scan(gc, self._set(DOC, pieces)) == (DOC, 1, False)

    def test_a_correct_chunk_set_with_the_games_prefix_is_accepted(self, gc: ModuleType) -> None:
        lines = [f"AlphaZeroMain: {line}" for line in g.chunk_lines(DOC, 3)]
        assert _scan(gc, "".join(lines)) == (DOC, 1, False)

    def test_the_checksum_is_over_utf8_bytes_not_characters(self, gc: ModuleType) -> None:
        # A JSON string holding a non-ASCII letter: bytes and code points disagree.
        document = {**DOC, "Note": "café"}
        text = json.dumps(document, separators=(",", ":"), ensure_ascii=False)
        assert g.adler32(text) != zlib.adler32(text.encode("latin-1"))
        parts = _cut(text, 3)
        log = "".join(_chunks(parts, 1, 3)) + g.end_line(3, g.adler32(text))
        assert _scan(gc, log) == (document, 1, False)

    def test_one_flipped_character_in_a_piece_is_not_accepted(self, gc: ModuleType) -> None:
        lines = g.chunk_lines(DOC, 3)
        damaged = lines[1].replace("TECH", "TEXH", 1) if "TECH" in lines[1] else None
        if damaged is None:  # flip a digit instead: still valid JSON, so only the checksum sees it
            damaged = lines[1].replace("0", "1", 1)
        assert damaged != lines[1], "premise: a character was changed"
        lines[1] = damaged
        premise = "".join(line.split(":", 2)[2] for line in lines[:-1])
        assert premise != g.compact_text(DOC)
        assert _scan(gc, g.single_line(OLDER) + "".join(lines)) == (OLDER, 1, True)
        assert _scan(gc, "".join(lines)) == (None, 0, False)

    def test_a_digit_flip_that_leaves_valid_json_is_not_accepted(self, gc: ModuleType) -> None:
        text = g.compact_text(DOC)
        flipped = text.replace('"Cost":10', '"Cost":11', 1)
        assert flipped != text and json.loads(flipped)
        log = "".join(_chunks(_cut(flipped, 3), 1, 3)) + g.end_line(3, g.adler32(text))
        assert _scan(gc, g.single_line(OLDER) + log) == (OLDER, 1, True)

    def test_a_missing_checksum_completes_nothing(self, gc: ModuleType) -> None:
        text = self._set(DOC, 3, checksum=None)
        assert text.rstrip().endswith("AZ_GAMEINFO_END:3")
        assert _scan(gc, g.single_line(OLDER) + text) == (OLDER, 1, True)
        assert _scan(gc, text) == (None, 0, False)

    @pytest.mark.parametrize(
        "bad",
        ["", "abc", "12x", "x12", "0x1f", "1.5", "1e3", "-5", "-0", "1 2", "12:34", "+5", "٣"],
        ids=repr,
    )
    def test_a_checksum_that_is_not_a_decimal_number_completes_nothing(
        self, gc: ModuleType, bad: str
    ) -> None:
        text = self._set(DOC, 3, checksum=bad)
        assert _scan(gc, g.single_line(OLDER) + text) == (OLDER, 1, True)
        assert _scan(gc, text) == (None, 0, False)

    def test_a_wrong_number_completes_nothing(self, gc: ModuleType) -> None:
        right = g.adler32(g.compact_text(DOC))
        for wrong in (right + 1, right - 1, 0, 1, 2**32, right + 2**32):
            text = self._set(DOC, 3, checksum=wrong)
            assert _scan(gc, g.single_line(OLDER) + text) == (OLDER, 1, True), wrong

    def test_the_checksum_alone_is_not_enough(self, gc: ModuleType) -> None:
        # Right checksum, but the END count disagrees with the headers: still unfinished.
        text = self._set(DOC, 3, end_total=2)
        assert _scan(gc, g.single_line(OLDER) + text) == (OLDER, 1, True)
        text = self._set(DOC, 3, header_total=4)
        assert _scan(gc, g.single_line(OLDER) + text) == (OLDER, 1, True)

    @pytest.mark.parametrize("pad", [(" ", ""), ("", "  "), ("  ", "\t "), (" ", " ")])
    def test_whitespace_around_the_checksum_is_accepted(
        self, gc: ModuleType, pad: tuple[str, str]
    ) -> None:
        # Decision: the mod prints none, but a trailing space or a CR is harmless noise.
        lines = g.chunk_lines(DOC, 3)
        c = g.adler32(g.compact_text(DOC))
        lines[-1] = g.end_line(3, f"{pad[0]}{c}{pad[1]}")
        assert _scan(gc, "".join(lines)) == (DOC, 1, False)

    def test_windows_line_endings_are_accepted(self, gc: ModuleType) -> None:
        text = self._set(DOC, 3).replace("\n", "\r\n")
        assert _scan(gc, text) == (DOC, 1, False)

    def test_single_line_messages_need_no_checksum(self, gc: ModuleType) -> None:
        assert _scan(gc, g.single_line(DOC)) == (DOC, 1, False)

    def test_the_helper_renders_the_new_end_line(self) -> None:
        lines = g.chunk_lines(DOC, 3)
        assert lines[-1] == f"AZ_GAMEINFO_END:3:{g.adler32(g.compact_text(DOC))}\n"
        assert g.chunk_lines(DOC, 3, checksum=None)[-1] == "AZ_GAMEINFO_END:3\n"


# ---------------------------------------------------------------------------
# Finding 2: publishing without hard links
# ---------------------------------------------------------------------------
def _no_links(*_args: Any, **_kwargs: Any) -> Any:
    raise OSError(errno.EPERM, "hard links are not supported here")


def _leftovers(root: Path) -> list[str]:
    return [name for name in _files(root) if name.endswith(".tmp")]


class TestPublishWithoutHardLinks:
    def test_control_the_snapshot_is_written(
        self, gc: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "link", _no_links)
        path = gc.write_snapshot(DOC, name="probe")
        assert path == root / "gameinfo" / "probe.json"
        assert json.loads(path.read_text(encoding="utf-8")) == DOC
        assert _files(root) == ["gameinfo/probe.json"]

    def test_control_an_existing_destination_is_refused_and_untouched(
        self, gc: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = gc.write_snapshot(DOC, name="probe")
        before = first.read_bytes()
        monkeypatch.setattr(os, "link", _no_links)
        with pytest.raises(gc.GameInfoCaptureError, match="never overwritten"):
            gc.write_snapshot(OLDER, name="probe")
        assert first.read_bytes() == before
        assert _files(root) == ["gameinfo/probe.json"]

    def test_control_the_command_line_refuses_an_existing_destination(
        self,
        gc: ModuleType,
        root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        source = tmp_path / "gameinfo.json"
        source.write_text(json.dumps(OLDER), encoding="utf-8")
        first = gc.write_snapshot(DOC, name="probe")
        monkeypatch.setattr(os, "link", _no_links)
        assert _run(gc, "--from-file", str(source), "--name", "probe") == 1
        assert "already exists" in capsys.readouterr().err
        assert json.loads(first.read_text(encoding="utf-8")) == DOC

    def _race(self, monkeypatch: pytest.MonkeyPatch, destination: Path) -> dict[str, bool]:
        """A competitor writes ``destination`` the moment the fallback first touches it.

        Armed when ``os.link`` fails. Queries (``exists``, ``stat``) return the truth and
        THEN the file appears (the gap between a check and the rename); a mutating call
        (``os.open``, ``open``, ``os.replace``, ``os.rename``) sees the file already there.
        Nothing assumes which of these the fix uses.
        """
        state = {"armed": False, "appeared": False}
        real_open = os.open
        target = str(destination)

        def appear() -> None:
            if state["armed"] and not state["appeared"]:
                state["appeared"] = True
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
                fd = real_open(target, flags)
                os.write(fd, b"COMPETITOR\n")
                os.close(fd)

        def is_target(path: Any) -> bool:
            try:
                return os.fspath(path) == target
            except TypeError:
                return False

        def after(original: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(path: Any, *args: Any, **kwargs: Any) -> Any:
                result = original(path, *args, **kwargs)
                if is_target(path):
                    appear()
                return result

            return wrapper

        def before(original: Callable[..., Any], *, destination_at: int) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if len(args) > destination_at and is_target(args[destination_at]):
                    appear()
                return original(*args, **kwargs)

            return wrapper

        def link(*_args: Any, **_kwargs: Any) -> Any:
            state["armed"] = True
            raise OSError(errno.EPERM, "hard links are not supported here")

        monkeypatch.setattr(os, "link", link)
        monkeypatch.setattr(pathlib.Path, "exists", _path_exists(after, is_target, appear))
        monkeypatch.setattr(os.path, "exists", after(os.path.exists))
        monkeypatch.setattr(os.path, "lexists", after(os.path.lexists))
        monkeypatch.setattr(os, "stat", after(os.stat))
        monkeypatch.setattr(os, "lstat", after(os.lstat))
        monkeypatch.setattr(os, "open", before(os.open, destination_at=0))
        monkeypatch.setattr(builtins, "open", before(builtins.open, destination_at=0))
        monkeypatch.setattr(io, "open", before(io.open, destination_at=0))
        monkeypatch.setattr(os, "replace", before(os.replace, destination_at=1))
        monkeypatch.setattr(os, "rename", before(os.rename, destination_at=1))
        return state

    def test_a_file_that_appears_meanwhile_is_refused_and_untouched(
        self, gc: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        destination = (root / "gameinfo" / "probe.json").resolve()
        (root / "gameinfo").mkdir()
        state = self._race(monkeypatch, destination)
        with pytest.raises(gc.GameInfoCaptureError, match=r"already exists.*never overwritten"):
            gc.write_snapshot(DOC, name="probe")
        assert state["appeared"], "premise: the competitor appeared while publishing"
        assert destination.read_bytes() == b"COMPETITOR\n"
        assert _files(root) == ["gameinfo/probe.json"]
        assert _leftovers(root) == []

    def test_the_command_line_refuses_the_same_race(
        self,
        gc: ModuleType,
        root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        source = tmp_path / "gameinfo.json"
        source.write_text(json.dumps(DOC), encoding="utf-8")
        destination = (root / "gameinfo" / "probe.json").resolve()
        (root / "gameinfo").mkdir()
        state = self._race(monkeypatch, destination)
        assert _run(gc, "--from-file", str(source), "--name", "probe") == 1
        err = capsys.readouterr().err
        assert state["appeared"]
        assert "already exists" in err and "Traceback" not in err
        assert destination.read_bytes() == b"COMPETITOR\n"
        assert _leftovers(root) == []

    def test_no_temporary_file_is_left_after_a_failed_publish(
        self, gc: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "link", _no_links)
        monkeypatch.setattr(os, "replace", _no_links)
        monkeypatch.setattr(os, "rename", _no_links)
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(DOC, name="probe")
        assert _files(root) == []


def _path_exists(after: Any, is_target: Any, appear: Any) -> Callable[..., bool]:
    original = pathlib.Path.exists

    def exists(self: pathlib.Path, *args: Any, **kwargs: Any) -> bool:
        result = bool(original(self, *args, **kwargs))
        if is_target(self):
            appear()
        return result

    return exists


# ---------------------------------------------------------------------------
# Finding 3: AZ_GAMEINFO_BEGIN
# ---------------------------------------------------------------------------
BEGIN = g.begin_line()
NEWER: dict[str, Any] = {**DOC, "Civics": []}


class TestBegin:
    @pytest.mark.parametrize("prefix", ["", "AlphaZeroMain: "])
    def test_a_begin_after_a_complete_message_makes_it_not_the_newest(
        self, gc: ModuleType, prefix: str
    ) -> None:
        for message in (g.single_line(DOC, prefix), "".join(g.chunk_lines(DOC))):
            assert _scan(gc, message + g.begin_line(prefix)) == (DOC, 1, True)

    @pytest.mark.parametrize("prefix", ["", "AlphaZeroMain: "])
    def test_a_complete_message_after_a_begin_is_the_newest(
        self, gc: ModuleType, prefix: str
    ) -> None:
        for message in (g.single_line(DOC, prefix), "".join(g.chunk_lines(DOC))):
            assert _scan(gc, g.begin_line(prefix) + message) == (DOC, 1, False)

    def test_complete_begin_complete_gives_the_newest_and_counts_both(self, gc: ModuleType) -> None:
        text = g.single_line(OLDER) + BEGIN + "".join(g.chunk_lines(NEWER))
        assert _scan(gc, text) == (NEWER, 2, False)

    def test_the_usual_log_of_a_launch_is_clean(self, gc: ModuleType) -> None:
        # Every export announces itself and then completes: no begin is "unfinished".
        text = BEGIN + g.single_line(OLDER) + BEGIN + g.single_line(DOC)
        assert _scan(gc, text) == (DOC, 2, False)

    def test_a_lone_begin_is_no_document(self, gc: ModuleType) -> None:
        assert _scan(gc, BEGIN) == (None, 0, False)
        assert _scan(gc, BEGIN + BEGIN) == (None, 0, False)

    def test_several_begins_after_a_complete_message_still_leave_it_unfinished(
        self, gc: ModuleType
    ) -> None:
        assert _scan(gc, g.single_line(DOC) + BEGIN + BEGIN) == (DOC, 1, True)

    def test_a_begin_does_not_count_as_a_message(self, gc: ModuleType) -> None:
        scan = gc.scan_log(BEGIN + g.single_line(DOC) + BEGIN)
        assert scan.complete == 1

    def test_the_command_line_refuses_unless_allowed(
        self,
        gc: ModuleType,
        root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        log = _log(tmp_path, g.single_line(DOC) + BEGIN)
        assert _run(gc, "--from-log", str(log), "--name", "n") == 1
        message = capsys.readouterr().err
        assert "incomplete" in message and "--allow-earlier" in message
        assert _files(root) == []

        assert _run(gc, "--from-log", str(log), "--name", "n", "--allow-earlier") == 0
        assert "earlier" in capsys.readouterr().err.lower()
        assert _filed(root, "n") == DOC

    def test_the_command_line_files_the_newest_when_a_complete_one_follows(
        self, gc: ModuleType, root: Path, tmp_path: Path
    ) -> None:
        log = _log(tmp_path, g.single_line(OLDER) + BEGIN + g.single_line(DOC))
        assert _run(gc, "--from-log", str(log), "--name", "n") == 0
        assert _filed(root, "n") == DOC

    def test_a_lone_begin_is_the_no_complete_message_exit(
        self,
        gc: ModuleType,
        root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        log = _log(tmp_path, BEGIN)
        assert _run(gc, "--from-log", str(log), "--name", "n") == 1
        assert "no complete GameInfo message" in capsys.readouterr().err
        assert _files(root) == []
        with pytest.raises(gc.GameInfoCaptureError, match="no complete GameInfo message"):
            gc.snapshot_from_log(BEGIN)

    def test_a_begin_between_the_chunks_drops_the_earlier_chunks(self, gc: ModuleType) -> None:
        # Control for the fixed scanner (it passes on the old one too, for another reason):
        # X's first piece, BEGIN, then the rest of Y with a checksum that matches the
        # JOIN of X's piece 1 and Y's pieces 2..n. Only dropping X's piece at the BEGIN
        # keeps that join from being accepted.
        a, b = _cut(TEXT_A, PIECES), _cut(TEXT_B, PIECES)
        joined = a[0] + "".join(b[1:])
        assert json.loads(joined) not in (DOC_A, DOC_B), "premise: a valid third document"
        text = (
            "".join(_chunks([a[0]], 1, PIECES))
            + BEGIN
            + "".join(_chunks(b[1:], 2, PIECES))
            + g.end_line(PIECES, g.adler32(joined))
        )
        assert _scan(gc, text) == (None, 0, False)
        assert _scan(gc, g.single_line(OLDER) + text) == (OLDER, 1, True)

    def test_a_chunk_set_after_a_begin_in_the_middle_is_taken_whole(self, gc: ModuleType) -> None:
        text = "".join(g.chunk_lines(OLDER)[:2]) + BEGIN + "".join(g.chunk_lines(DOC))
        assert _scan(gc, text) == (DOC, 1, False)

    def test_a_begin_after_a_partial_set_keeps_the_earlier_complete_one_current_only_as_earlier(
        self, gc: ModuleType
    ) -> None:
        text = g.single_line(OLDER) + "".join(g.chunk_lines(DOC)[:2]) + BEGIN
        assert _scan(gc, text) == (OLDER, 1, True)


class TestTheStateReaderIgnoresTheNewLines:
    LINES = [
        'AlphaZeroMain: AZ_STATE:{"turn_id": 1}',
        "AlphaZeroMain: AZ_GAMEINFO_BEGIN",
        "AlphaZeroMain: AZ_GAMEINFO_CHUNK:1/2:{}",
        "AlphaZeroMain: AZ_GAMEINFO_CHUNK:2/2:{}",
        "AlphaZeroMain: AZ_GAMEINFO_END:2:123456789",
        "AZ_GAMEINFO_BEGIN",
        "AZ_GAMEINFO_END:2:4294967295",
        'AlphaZeroMain: AZ_STATE:{"turn_id": 2}',
    ]

    @staticmethod
    def _read(tmp_path: Path, lines: list[str]) -> tuple[Any, dict[str, int], dict[int, str]]:
        tailer = LogTailer(tmp_path / "Lua.log")
        seen: list[Any] = []
        for line in lines:
            tailer._parse_line(line)
            seen.append(tailer.latest_state)
        stats = tailer.stats
        counters = {k: stats[k] for k in ("states_received", "chunks_received", "parse_errors")}
        return seen, counters, dict(tailer._chunk_buffer)

    def test_the_same_states_with_and_without_them(self, tmp_path: Path) -> None:
        states = [line for line in self.LINES if "AZ_STATE:" in line]
        with_lines = self._read(tmp_path, self.LINES)
        without = self._read(tmp_path, states)
        assert with_lines[0][-1] == without[0][-1] == {"turn_id": 2}
        assert (
            with_lines[1]
            == without[1]
            == {
                "states_received": 2,
                "chunks_received": 0,
                "parse_errors": 0,
            }
        )
        assert with_lines[2] == without[2] == {}


class TestCountsFromTheLogCostNothing:
    """A count in a log line is text someone wrote; it must not size an allocation."""

    #: Large enough that walking 1..n would show (hundreds of megabytes), small
    #: enough that the test only wastes time, not the machine, if it regresses.
    HUGE = 5_000_000

    def test_a_huge_claimed_total_is_unfinished_and_allocates_nothing(self, gc: ModuleType) -> None:
        text = (
            g.single_line(OLDER)
            + f"AZ_GAMEINFO_CHUNK:1/{self.HUGE}:x\n"
            + g.end_line(self.HUGE, zlib.adler32(b"x"))
        )
        tracemalloc.start()
        try:
            result = _scan(gc, text)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert result == (OLDER, 1, True)
        assert peak < 2_000_000, f"scanning three short lines peaked at {peak} bytes"

    @pytest.mark.parametrize(
        "line",
        [
            "AZ_GAMEINFO_CHUNK:" + "9" * 5000 + "/3:x",
            "AZ_GAMEINFO_CHUNK:1/" + "9" * 5000 + ":x",
            "AZ_GAMEINFO_END:" + "9" * 5000 + ":1",
            "AZ_GAMEINFO_END:1:" + "9" * 5000,
        ],
        ids=["piece-number", "piece-total", "end-total", "end-checksum"],
    )
    def test_a_run_of_thousands_of_digits_is_not_a_number(self, gc: ModuleType, line: str) -> None:
        # Python raises ValueError converting a decimal string longer than 4300 digits.
        assert _scan(gc, g.single_line(OLDER) + line + "\n")[:2] == (OLDER, 1)

    def test_a_ten_digit_checksum_is_still_read(self, gc: ModuleType) -> None:
        # Adler-32 goes up to 4294967295: ten digits must fit.
        document = None
        for pad in range(400):
            candidate = {**DOC, "Units": [{"UnitType": "UNIT_" + "Z" * (pad + 1), "Cost": 1}]}
            if g.adler32(g.compact_text(candidate)) >= 1_000_000_000:
                document = candidate
                break
        assert document is not None, "no ten-digit checksum found among the candidates"
        assert _scan(gc, "".join(g.chunk_lines(document, 3))) == (document, 1, False)
