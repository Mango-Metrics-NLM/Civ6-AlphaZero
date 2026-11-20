"""GM-B4 second review round, the mod's side (lupa, Lua 5.1).

* The END line the mod prints is ``AZ_GAMEINFO_END:<n>:<adler32>``.
* Every export attempt announces itself with ``AZ_GAMEINFO_BEGIN`` (print mode) or by
  deleting the previous load's file (file mode), and clears ``ExposedMembers.AlphaZeroGameInfo``.

These need the regenerated bundle (``python scripts/build_lua_bundle.py``); until it is
regenerated they fail. The file-access branch runs against an in-memory stand-in for ``io``
and ``os.remove`` written in Lua and installed by the test: the harness's lockdown (no
real ``io`` or ``os``) stays in force, so nothing here can touch the file system.
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.bridge import gameinfo_capture as gc
from alpha_zero.bridge.log_tailer import LogTailer
from tests.fixtures import gameinfo_snapshot as g

BEGIN = "AZ_GAMEINFO_BEGIN"
FAILED = "no GameInfo table could be read"
EMPTY_GAME = "GameInfo = {}"
CHUNKED_DOCUMENT = {
    **g.STUB_DOCUMENT,
    "Units": g.STUB_DOCUMENT["Units"] + g.extra_unit_documents(200),
}

#: In-memory ``io.open`` and ``os.remove``: the mod then sees file access (``HAS_IO``).
FAKE_FILES = """
FakeFS = {}
io = {
  open = function(path, mode)
    mode = mode or "r"
    if mode == "r" then
      if FakeFS[path] == nil then return nil, "no such file" end
      local content, done = FakeFS[path], false
      return {
        read = function(_, n)
          if done then return nil end
          done = true
          return content
        end,
        seek = function() return #content end,
        close = function() end,
      }
    end
    local buffer = (mode == "a" and FakeFS[path]) or ""
    return {
      write = function(self, text) buffer = buffer .. text return self end,
      flush = function() end,
      seek = function() return #buffer end,
      close = function() FakeFS[path] = buffer end,
    }
  end,
}
os.remove = function(path)
  if FakeFS[path] == nil then return nil, path .. ": No such file" end
  FakeFS[path] = nil
  return true
end
"""


def _gameinfo_lines(runtime: Any) -> list[str]:
    return [line for line in g.printed(runtime) if line.startswith(g.MARKER)]


def _outcome(result: Any) -> tuple[Any, Any]:
    if isinstance(result, tuple):
        return (*result, None)[:2]
    return result, None


def _attempt(runtime: Any) -> tuple[Any, Any, list[str]]:
    """One ``ExportAndWrite()`` after clearing the log: ``(success, message, gameinfo lines)``."""
    g.clear_printed(runtime)
    success, message = _outcome(runtime.eval("AlphaZeroGameInfo.ExportAndWrite()"))
    return success, message, _gameinfo_lines(runtime)


# ---------------------------------------------------------------------------
# Finding 1: the END line
# ---------------------------------------------------------------------------
class TestTheEndLine:
    def test_it_carries_the_count_and_the_adler32_of_the_joined_text(self) -> None:
        runtime = g.load_printing(g.extra_units(200))
        lines = _gameinfo_lines(runtime)
        chunks = [line for line in lines if line.startswith("AZ_GAMEINFO_CHUNK:")]
        assert len(chunks) >= 2
        end = lines[-1]
        match = re.fullmatch(r"AZ_GAMEINFO_END:(\d+):(\d+)", end)
        assert match, end
        assert int(match[1]) == len(chunks)
        ((text, _),) = g.messages(lines)
        assert int(match[2]) == zlib.adler32(text.encode("utf-8"))
        assert json.loads(text) == CHUNKED_DOCUMENT

    def test_there_is_one_end_line_and_it_closes_the_set(self) -> None:
        runtime = g.load_printing(g.extra_units(200))
        lines = _gameinfo_lines(runtime)
        assert sum(line.startswith("AZ_GAMEINFO_END:") for line in lines) == 1
        assert lines[-1].startswith("AZ_GAMEINFO_END:")

    def test_a_checksum_above_two_to_the_31_is_a_plain_decimal_integer(self) -> None:
        # Search padded documents until an Adler-32 above 2**31 turns up (about half do), so
        # the format is asserted on a real large value: no exponent, no point, no sign.
        for pad in range(1, 60):
            runtime = g.load_printing(g.extra_units(200) + g.padded_units(pad))
            lines = _gameinfo_lines(runtime)
            match = re.fullmatch(r"AZ_GAMEINFO_END:(\d+):(\d+)", lines[-1])
            assert match, lines[-1]
            ((text, _),) = g.messages(lines)
            expected = zlib.adler32(text.encode("utf-8"))
            assert match[2] == str(expected)
            if expected > 2**31:
                assert expected < 2**32
                return
        pytest.fail("no document with an Adler-32 above 2**31 in 59 tries")

    def test_a_single_line_message_has_no_end_line(self) -> None:
        runtime = g.load_printing(g.SMALL_SETUP)
        lines = _gameinfo_lines(runtime)
        assert len(lines) == 2 and lines[0] == BEGIN and lines[1].startswith("AZ_GAMEINFO:")


class TestWhatTheModPrintsIsWhatTheScannerAccepts:
    @pytest.mark.parametrize("prefix", ["", "AlphaZeroMain: "])
    def test_a_chunked_snapshot(self, prefix: str) -> None:
        runtime = g.load_printing(g.extra_units(200))
        scan = gc.scan_log(g.log_text(g.printed(runtime), prefix))
        assert (scan.document, scan.complete, scan.later_incomplete) == (
            CHUNKED_DOCUMENT,
            1,
            False,
        )

    def test_the_whole_stub_snapshot(self) -> None:
        runtime = g.load_printing()
        scan = gc.scan_log(g.log_text(g.printed(runtime)))
        assert (scan.document, scan.complete, scan.later_incomplete) == (g.STUB_DOCUMENT, 1, False)

    def test_a_single_line_snapshot(self) -> None:
        runtime = g.load_printing(g.SMALL_SETUP)
        scan = gc.scan_log(g.log_text(g.printed(runtime)))
        assert (scan.document, scan.complete, scan.later_incomplete) == (g.SMALL_DOCUMENT, 1, False)

    def test_two_loads_are_two_messages_and_the_second_wins(self) -> None:
        first = g.load_printing(g.SMALL_SETUP)
        second = g.load_printing()
        text = g.log_text(g.printed(first)) + g.log_text(g.printed(second))
        scan = gc.scan_log(text)
        assert (scan.document, scan.complete, scan.later_incomplete) == (g.STUB_DOCUMENT, 2, False)


# ---------------------------------------------------------------------------
# Finding 3: BEGIN
# ---------------------------------------------------------------------------
SETUPS = {
    "single-line": g.SMALL_SETUP,
    "chunked-stub": "",
    "chunked-long": g.extra_units(200),
}


class TestBegin:
    @pytest.mark.parametrize("setup", SETUPS.values(), ids=SETUPS.keys())
    def test_it_is_printed_once_per_attempt_and_before_any_data_line(self, setup: str) -> None:
        runtime = g.load_printing(setup)
        at_load = _gameinfo_lines(runtime)
        assert at_load[0] == BEGIN and at_load.count(BEGIN) == 1

        success, _, lines = _attempt(runtime)
        assert success is True
        assert lines[0] == BEGIN and lines.count(BEGIN) == 1
        assert len(lines) > 1 and not any(line == BEGIN for line in lines[1:])
        assert len(g.messages(lines)) == 1

        _, _, again = _attempt(runtime)
        assert again.count(BEGIN) == 1 and again[0] == BEGIN

    def test_it_is_not_taken_for_a_data_line(self) -> None:
        runtime = g.load_printing(g.extra_units(200))
        lines = _gameinfo_lines(runtime)
        assert BEGIN in lines
        assert len(g.messages(lines)) == 1

    def test_unreadable_tables_return_false_but_still_announce_the_attempt(self) -> None:
        runtime = g.load_printing()
        runtime.execute(EMPTY_GAME)
        success, message, lines = _attempt(runtime)
        assert success is False
        assert message == FAILED
        assert lines == [BEGIN]

    def test_a_value_already_exposed_is_nil_after_a_failed_attempt(self) -> None:
        runtime = g.load_printing()
        assert isinstance(runtime.eval("ExposedMembers.AlphaZeroGameInfo"), str), "premise"
        runtime.execute(EMPTY_GAME)
        _attempt(runtime)
        assert runtime.eval("ExposedMembers.AlphaZeroGameInfo") is None
        runtime.execute("ExposedMembers.AlphaZeroGameInfo = 'stale'")
        _attempt(runtime)
        assert runtime.eval("ExposedMembers.AlphaZeroGameInfo") is None

    def test_a_good_attempt_after_a_failed_one_exposes_the_new_text(self) -> None:
        runtime = g.load_printing(g.SMALL_SETUP)
        runtime.execute(EMPTY_GAME)
        _attempt(runtime)
        assert runtime.eval("ExposedMembers.AlphaZeroGameInfo") is None
        runtime.execute(g.SMALL_SETUP)
        success, _, lines = _attempt(runtime)
        assert success is True
        ((text, _),) = g.messages(lines)
        assert runtime.eval("ExposedMembers.AlphaZeroGameInfo") == text

    def test_begin_is_not_printed_when_the_flag_is_off(self) -> None:
        flag_off = (
            "local init = AlphaZeroConfig.Init\n"
            "AlphaZeroConfig.Init = function() local c = init() c.ENABLE_GAMEINFO_EXPORT = false "
            "return c end"
        )
        runtime = g.load_printing()
        runtime.execute(flag_off)
        g.clear_printed(runtime)
        runtime.execute("AlphaZeroBridge.Reset()")
        assert runtime.eval("AlphaZeroBridge.Initialize()") is True
        assert _gameinfo_lines(runtime) == []

    def test_end_to_end_a_load_that_exports_then_a_load_that_fails(self) -> None:
        load_one = g.load_printing()
        load_two = g.load_printing(EMPTY_GAME)
        assert _gameinfo_lines(load_two) == [BEGIN], "premise: load 2 announced and printed no data"
        scan = gc.scan_log(
            g.log_text(g.printed(load_one), "AlphaZeroMain: ")
            + g.log_text(g.printed(load_two), "AlphaZeroMain: ")
        )
        assert (scan.document, scan.complete, scan.later_incomplete) == (g.STUB_DOCUMENT, 1, True)

    def test_end_to_end_a_failed_load_then_a_good_one(self) -> None:
        load_one = g.load_printing(EMPTY_GAME)
        load_two = g.load_printing()
        scan = gc.scan_log(g.log_text(g.printed(load_one)) + g.log_text(g.printed(load_two)))
        assert (scan.document, scan.complete, scan.later_incomplete) == (g.STUB_DOCUMENT, 1, False)

    def test_end_to_end_the_command_line_refuses_the_older_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data = tmp_path / "data"
        data.mkdir()
        monkeypatch.setenv("CIV6AZ_DATA_ROOT", str(data))
        monkeypatch.chdir(tmp_path)
        text = g.log_text(g.printed(g.load_printing())) + g.log_text(
            g.printed(g.load_printing(EMPTY_GAME))
        )
        log = tmp_path / "Lua.log"
        log.write_text(text, encoding="utf-8")
        assert gc.main(["--from-log", str(log), "--name", "n"]) == 1
        assert "--allow-earlier" in capsys.readouterr().err
        assert not list(data.rglob("*.json"))
        assert gc.main(["--from-log", str(log), "--name", "n", "--allow-earlier"]) == 0
        filed = json.loads((data / "gameinfo" / "n.json").read_text(encoding="utf-8"))
        assert filed == g.STUB_DOCUMENT


class TestBeginWithFileAccess:
    """File mode, against an in-memory ``io``: the previous load's file must not survive."""

    def _load(self, extra: str = "") -> Any:
        return g.load_printing(FAKE_FILES + extra)

    def _path(self, runtime: Any) -> str:
        path = runtime.eval("AlphaZeroConfig.Get('GAMEINFO_PATH')")
        assert isinstance(path, str) and path
        return str(path)

    def _has_file(self, runtime: Any, path: str) -> bool:
        return bool(runtime.eval("FakeFS[AlphaZeroConfig.Get('GAMEINFO_PATH')] ~= nil"))

    def test_control_file_mode_writes_the_file_and_prints_no_gameinfo(self) -> None:
        runtime = self._load()
        path = self._path(runtime)
        assert self._has_file(runtime, path), "premise: with file access the mod wrote the file"
        assert (
            json.loads(runtime.eval("FakeFS[AlphaZeroConfig.Get('GAMEINFO_PATH')]"))
            == g.STUB_DOCUMENT
        )
        assert not any(line.startswith("AZ_GAMEINFO:") for line in _gameinfo_lines(runtime))
        assert not any(line.startswith("AZ_GAMEINFO_CHUNK:") for line in _gameinfo_lines(runtime))

    def test_a_failed_attempt_deletes_the_previous_loads_file(self) -> None:
        runtime = self._load()
        path = self._path(runtime)
        assert self._has_file(runtime, path), "premise"
        runtime.execute(EMPTY_GAME)
        success, message, _ = _attempt(runtime)
        assert (success, message) == (False, FAILED)
        assert not self._has_file(runtime, path)
        assert runtime.eval("ExposedMembers.AlphaZeroGameInfo") is None

    def test_a_good_attempt_replaces_the_file_with_the_new_document(self) -> None:
        runtime = self._load()
        runtime.execute(g.SMALL_SETUP)
        success, _, _ = _attempt(runtime)
        assert success is True
        assert (
            json.loads(runtime.eval("FakeFS[AlphaZeroConfig.Get('GAMEINFO_PATH')]"))
            == g.SMALL_DOCUMENT
        )

    def test_a_failed_load_in_a_fresh_game_leaves_no_file_behind(self) -> None:
        runtime = self._load()
        path = self._path(runtime)
        runtime.execute("FakeFS[AlphaZeroConfig.Get('GAMEINFO_PATH')] = '{}'")
        runtime.execute(EMPTY_GAME)
        runtime.execute("AlphaZeroBridge.Reset()")
        assert runtime.eval("AlphaZeroBridge.Initialize()") is True
        assert not self._has_file(runtime, path)

    def test_the_fake_file_system_is_what_the_mod_really_uses(self) -> None:
        # Guards the stand-in itself: other files the mod writes land in it as well.
        runtime = self._load()
        count = runtime.eval(
            "(function() local n = 0 for _ in pairs(FakeFS) do n = n + 1 end return n end)()"
        )
        assert count >= 2, "the state or heartbeat files are written too"


class TestTheStateReaderIgnoresTheNewLines:
    def test_the_log_tailer_reads_the_same_states_with_begin_and_the_new_end(
        self, tmp_path: Path
    ) -> None:
        runtime = g.load_printing(g.extra_units(200))
        at_load = _gameinfo_lines(runtime)
        runtime.execute(EMPTY_GAME)
        lines = [*at_load, *_attempt(runtime)[2]]
        assert lines.count(BEGIN) >= 1
        assert any(re.fullmatch(r"AZ_GAMEINFO_END:\d+:\d+", line) for line in lines)

        def read(feed: list[str]) -> tuple[Any, tuple[int, int, int], dict[int, str]]:
            tailer = LogTailer(tmp_path / "Lua.log")
            for line in feed:
                tailer._parse_line(f"AlphaZeroMain: {line}")
            stats = tailer.stats
            return (
                tailer.latest_state,
                (stats["states_received"], stats["chunks_received"], stats["parse_errors"]),
                dict(tailer._chunk_buffer),
            )

        state = 'AZ_STATE:{"turn_id": 7}'
        assert read([state, *lines, *lines]) == read([state])
        assert read([state])[0] == {"turn_id": 7}, "control: the reader does read a state"
