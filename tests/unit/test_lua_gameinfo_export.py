"""GM-B4: the mod's GameInfo snapshot, executed against the invented game (lupa, Lua 5.1).

``mod/AlphaZeroAI/Scripts/GameInfoExporter.lua`` exports the game's own rule tables so
the Python side can build a rule set from them. These tests run it on the stub and read
what it prints, as the Python log reader would. The expected document is written out by
hand in ``tests/fixtures/gameinfo_snapshot.py``; it is not derived from the exporter.

The modinfo test needs no Lua runtime; every other test loads the mod.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.bridge.log_tailer import LogTailer
from alpha_zero.rules.gameinfo import convert
from tests.fixtures import gameinfo_snapshot as g
from tests.fixtures import lua_harness as h

SCRIPTS = h.REPO_ROOT / "mod" / "AlphaZeroAI" / "Scripts"
FLAG = "ENABLE_GAMEINFO_EXPORT"
FLAG_OFF = (
    "local init = AlphaZeroConfig.Init\n"
    "AlphaZeroConfig.Init = function() local c = init() c.ENABLE_GAMEINFO_EXPORT = false return c end"
)


def _without_tables(*names: str) -> dict[str, Any]:
    return {k: v for k, v in g.STUB_DOCUMENT.items() if k not in names}


def _gameinfo_lines(runtime: Any) -> list[str]:
    return [line for line in g.printed(runtime) if line.startswith(g.MARKER)]


def _message_documents(runtime: Any) -> list[Any]:
    return [json.loads(text) for text, _ in g.messages(g.printed(runtime))]


def _outcome(result: Any) -> tuple[Any, Any]:
    """``(success, message)`` from a Lua call that returns ``true`` or ``false, message``."""
    if isinstance(result, tuple):
        return (*result, None)[:2]
    return result, None


def _reinitialise(runtime: Any) -> bool:
    g.clear_printed(runtime)
    runtime.execute("AlphaZeroBridge.Reset()")
    return bool(runtime.eval("AlphaZeroBridge.Initialize()"))


class TestModuleIsShipped:
    def test_modinfo_lists_the_exporter_after_the_state_exporter(self) -> None:
        modinfo = next((SCRIPTS.parent).glob("*.modinfo")).read_text(encoding="utf-8")
        listed = re.findall(r"<File>\s*Scripts/([^<\s]+\.lua)\s*</File>", modinfo)
        modules = [name for name in listed if name != "AlphaZeroMain.lua"]
        assert len(modules) == 8
        assert modules.index("GameInfoExporter.lua") == modules.index("StateExporter.lua") + 1

    def test_the_module_file_exists_and_is_in_the_bundle(self) -> None:
        assert (SCRIPTS / "GameInfoExporter.lua").is_file()
        bundle = (SCRIPTS / "AlphaZeroMain.lua").read_text(encoding="utf-8")
        assert "-- ========== BEGIN GameInfoExporter.lua ==========" in bundle

    def test_the_global_exists_with_both_entry_points(self) -> None:
        runtime = h.load_mod()
        assert runtime.eval("type(AlphaZeroGameInfo.Export)") == "function"
        assert runtime.eval("type(AlphaZeroGameInfo.ExportAndWrite)") == "function"


class TestExport:
    def test_the_stub_exports_exactly_the_six_tables_and_their_listed_columns(self) -> None:
        document, failed, _ = g.lua_export(h.load_mod())
        assert failed == []
        assert document == g.STUB_DOCUMENT

    def test_zero_is_kept_and_nil_columns_are_absent(self) -> None:
        document, _, _ = g.lua_export(h.load_mod())
        settler = document["Units"][0]
        assert settler["Combat"] == 0 and type(settler["Combat"]) is int
        monument = document["Buildings"][0]
        assert "PrereqTech" not in monument
        assert document["Buildings"][1]["PrereqTech"] == "TECH_POTTERY"

    def test_unlisted_columns_are_not_copied(self) -> None:
        document, _, _ = g.lua_export(h.load_mod())
        columns = {c for rows in document.values() for row in rows for c in row}
        assert not columns & {"Index", "Hash", "FormationClass"}

    def test_rows_follow_the_games_iteration_order(self) -> None:
        reverse = g.replace_table(
            "Technologies",
            "{ { TechnologyType = 'TECH_B', Cost = 2, EraType = 'ERA_ANCIENT' },"
            " { TechnologyType = 'TECH_A', Cost = 1, EraType = 'ERA_ANCIENT' } }",
        )
        document, failed, _ = g.lua_export(h.load_mod(setup=reverse))
        assert failed == []
        assert [r["TechnologyType"] for r in document["Technologies"]] == ["TECH_B", "TECH_A"]

    def test_a_table_object_that_is_called_as_in_the_game(self) -> None:
        setup = (
            "do local original = GameInfo.Units\n"
            "GameInfo.Units = setmetatable({}, { __call = function() return original() end })\n"
            "end"
        )
        document, failed, _ = g.lua_export(h.load_mod(setup=setup))
        assert failed == []
        assert document["Units"] == g.STUB_DOCUMENT["Units"]

    def test_an_empty_table_is_present_and_encodes_as_an_empty_array(self) -> None:
        runtime = h.load_mod(setup=g.replace_table("Units", "{}"))
        document, failed, _ = g.lua_export(runtime)
        assert failed == []
        # An empty JSON array, not an empty object (``{}`` would not equal ``[]``).
        assert document["Units"] == []
        assert document["Technologies"] == g.STUB_DOCUMENT["Technologies"]


class TestSanitising:
    GOOD = "{ UnitType = 'UNIT_OK', Cost = 3.0, Combat = -1, BaseMoves = 0, PrereqTech = 'a_1' }"

    def _units(self, *rows: str) -> list[dict[str, Any]]:
        setup = g.replace_table("Units", "{ " + ", ".join(rows) + " }")
        document, failed, _ = g.lua_export(h.load_mod(setup=setup))
        assert failed == []
        units: list[dict[str, Any]] = document["Units"]
        return units

    def test_identifiers_and_whole_numbers_survive(self) -> None:
        (unit,) = self._units(self.GOOD)
        assert unit == {
            "UnitType": "UNIT_OK",
            "Cost": 3,
            "Combat": -1,
            "BaseMoves": 0,
            "PrereqTech": "a_1",
        }
        assert all(type(unit[c]) is int for c in ("Cost", "Combat", "BaseMoves"))

    @pytest.mark.parametrize(
        "bad",
        [
            "'TECH WITH SPACE'",
            "'9LIVES'",
            "''",
            "'TECH-DASH'",
            "'TECH.DOT'",
            "'_LEADING'",
            "'TECH_\\195\\156'",
            "'TECH_A\\n'",
            "'12'",
        ],
    )
    def test_a_string_that_is_not_an_identifier_is_left_out(self, bad: str) -> None:
        (unit,) = self._units(f"{{ UnitType = 'UNIT_OK', Cost = 1, PrereqTech = {bad} }}")
        assert unit == {"UnitType": "UNIT_OK", "Cost": 1}

    @pytest.mark.parametrize("bad", ["2.5", "0.5", "true", "false", "{ 1 }", "{}"])
    def test_a_number_that_is_not_whole_or_not_a_number_is_left_out(self, bad: str) -> None:
        (unit,) = self._units(f"{{ UnitType = 'UNIT_OK', Cost = 1, Combat = {bad} }}")
        assert unit == {"UnitType": "UNIT_OK", "Cost": 1}

    @pytest.mark.parametrize("value", ["2^60", "1e20", "-(2^60)", "math.huge", "-math.huge", "0/0"])
    def test_encoded_numbers_are_json_integers_or_absent(self, value: str) -> None:
        # Above 2^53 the mod's encoder writes a float, which the importer refuses.
        (unit,) = self._units(f"{{ UnitType = 'UNIT_OK', Cost = 1, Combat = {value} }}")
        assert unit["UnitType"] == "UNIT_OK"
        assert "Combat" not in unit or type(unit["Combat"]) is int

    def test_the_sanitised_stub_document_is_accepted_by_the_importer(self) -> None:
        document, _, _ = g.lua_export(h.load_mod())
        summary = convert(document)
        assert (summary.techs, summary.units, summary.buildings) == (4, 3, 2)
        assert summary.civics_not_imported == 4


class TestTablesThatCannotBeRead:
    @pytest.mark.parametrize("name", g.TABLES)
    def test_a_missing_table_is_absent_and_named(self, name: str) -> None:
        document, failed, _ = g.lua_export(h.load_mod(setup=f"GameInfo.{name} = nil"))
        assert failed == [name]
        assert document == _without_tables(name)

    @pytest.mark.parametrize(
        "raiser",
        [
            "function() error('boom at call') end",
            "function() local i = 0 return function() i = i + 1 if i == 2 then error('mid') end"
            " return { TechnologyType = 'TECH_A', Cost = 1, EraType = 'ERA_ANCIENT' } end end",
        ],
        ids=["raises-when-called", "raises-mid-iteration"],
    )
    def test_a_table_whose_iteration_raises_is_absent_not_partial(self, raiser: str) -> None:
        document, failed, _ = g.lua_export(h.load_mod(setup=f"GameInfo.Technologies = {raiser}"))
        assert failed == ["Technologies"]
        assert document == _without_tables("Technologies")

    def test_several_failures_are_all_named_and_the_rest_exported(self) -> None:
        setup = "GameInfo.Units = nil\nGameInfo.Civics = function() error('x') end"
        document, failed, _ = g.lua_export(h.load_mod(setup=setup))
        assert sorted(failed) == ["Civics", "Units"]
        assert document == _without_tables("Civics", "Units")

    def test_a_missing_gameinfo_is_an_empty_document_and_all_six_names(self) -> None:
        runtime = h.load_mod(setup="GameInfo = nil")
        has_keys, failed = runtime.execute(
            "local d, f = AlphaZeroGameInfo.Export() return next(d) ~= nil, table.concat(f, ',')"
        )
        assert has_keys is False
        assert sorted(str(failed).split(",")) == sorted(g.TABLES)


class TestExportAndWrite:
    def test_a_short_snapshot_is_one_line_and_is_exposed(self) -> None:
        runtime = g.load_printing(g.SMALL_SETUP)
        g.clear_printed(runtime)
        assert _outcome(runtime.eval("AlphaZeroGameInfo.ExportAndWrite()"))[0] is True
        lines = _gameinfo_lines(runtime)
        assert len(lines) == 1 and lines[0].startswith("AZ_GAMEINFO:")
        text = lines[0][len("AZ_GAMEINFO:") :]
        assert len(text) <= g.MAX_PIECE
        assert json.loads(text) == g.SMALL_DOCUMENT
        assert runtime.eval("ExposedMembers.AlphaZeroGameInfo") == text

    def test_the_whole_stub_snapshot_is_exposed_whichever_way_it_is_printed(self) -> None:
        runtime = g.load_printing()
        g.clear_printed(runtime)
        assert _outcome(runtime.eval("AlphaZeroGameInfo.ExportAndWrite()"))[0] is True
        ((text, _),) = g.messages(_gameinfo_lines(runtime))
        assert json.loads(text) == g.STUB_DOCUMENT
        assert runtime.eval("ExposedMembers.AlphaZeroGameInfo") == text

    def test_a_long_snapshot_is_chunked_and_reassembles(self) -> None:
        runtime = g.load_printing(g.extra_units(200))
        g.clear_printed(runtime)
        assert _outcome(runtime.eval("AlphaZeroGameInfo.ExportAndWrite()"))[0] is True
        lines = _gameinfo_lines(runtime)
        assert not any(line.startswith("AZ_GAMEINFO:") for line in lines)
        chunks = [line for line in lines if line.startswith("AZ_GAMEINFO_CHUNK:")]
        assert len(chunks) >= 2 and lines[-1] == f"AZ_GAMEINFO_END:{len(chunks)}"
        assert all(
            line.startswith(f"AZ_GAMEINFO_CHUNK:{i}/{len(chunks)}:")
            for i, line in enumerate(chunks, 1)
        )
        ((text, sizes),) = g.messages(lines)
        assert all(size == g.MAX_PIECE for size in sizes[:-1]) and 0 < sizes[-1] <= g.MAX_PIECE
        assert len(sizes) == -(-len(text) // g.MAX_PIECE)
        expected = {
            **g.STUB_DOCUMENT,
            "Units": g.STUB_DOCUMENT["Units"] + g.extra_unit_documents(200),
        }
        assert json.loads(text) == expected
        assert runtime.eval("ExposedMembers.AlphaZeroGameInfo") == text

    @pytest.mark.parametrize(("extra", "pieces"), [(0, 1), (1, 2)])
    def test_the_line_limit_is_exactly_1800_characters(self, extra: int, pieces: int) -> None:
        probe = g.load_printing(g.SMALL_SETUP + g.padded_units(1))
        base = len(g.lua_export(probe)[2])
        assert base < g.MAX_PIECE, "premise: the small snapshot plus one row fits a line"
        runtime = g.load_printing(g.SMALL_SETUP + g.padded_units(g.MAX_PIECE - base + 1 + extra))
        g.clear_printed(runtime)
        assert _outcome(runtime.eval("AlphaZeroGameInfo.ExportAndWrite()"))[0] is True
        ((text, sizes),) = g.messages(_gameinfo_lines(runtime))
        assert len(text) == g.MAX_PIECE + extra
        assert len(sizes) == pieces

    def test_a_failing_encoder_returns_false_and_a_message_and_prints_nothing(self) -> None:
        runtime = g.load_printing()
        g.clear_printed(runtime)
        runtime.execute("AlphaZeroJson.Encode = function() return nil, 'encoder boom' end")
        success, message = _outcome(runtime.eval("AlphaZeroGameInfo.ExportAndWrite()"))
        assert success is False
        assert isinstance(message, str) and message
        assert _gameinfo_lines(runtime) == []


class TestInitialisation:
    def test_the_flag_defaults_to_on(self) -> None:
        runtime = h.load_mod()
        assert runtime.eval(f"AlphaZeroConfig.Get('{FLAG}')") is True
        assert runtime.eval("AlphaZeroConfig.IsEnabled('GAMEINFO_EXPORT')") is True

    def test_loading_the_bundle_prints_one_complete_snapshot(self) -> None:
        runtime = g.load_printing()
        ((text, _),) = g.messages(_gameinfo_lines(runtime))
        assert json.loads(text) == g.STUB_DOCUMENT
        assert runtime.eval("ExposedMembers.AlphaZeroGameInfo") == text

    def test_a_short_snapshot_is_printed_as_a_single_line_at_load(self) -> None:
        runtime = g.load_printing(g.SMALL_SETUP)
        lines = _gameinfo_lines(runtime)
        assert len(lines) == 1 and lines[0].startswith("AZ_GAMEINFO:")
        assert json.loads(lines[0][len("AZ_GAMEINFO:") :]) == g.SMALL_DOCUMENT

    def test_a_long_snapshot_is_printed_as_one_complete_chunk_set_at_load(self) -> None:
        runtime = g.load_printing(g.extra_units(200))
        lines = _gameinfo_lines(runtime)
        assert sum(line.startswith("AZ_GAMEINFO_END:") for line in lines) == 1
        (document,) = _message_documents(runtime)
        assert len(document["Units"]) == 203

    def test_initialising_twice_without_a_reset_does_not_print_again(self) -> None:
        runtime = g.load_printing()
        assert runtime.eval("AlphaZeroBridge.Initialize()") is True
        assert len(g.messages(_gameinfo_lines(runtime))) == 1

    def test_a_reinitialisation_prints_again_when_the_flag_is_on(self) -> None:
        runtime = g.load_printing()
        assert _reinitialise(runtime) is True
        assert _message_documents(runtime) == [g.STUB_DOCUMENT]

    def test_the_flag_off_prints_nothing_for_gameinfo_but_still_initialises(self) -> None:
        # Initialize() re-reads the defaults (Config.Init), so a Set() beforehand is lost:
        # the flag is turned off from inside Init, as an override file would.
        runtime = g.load_printing()
        assert runtime.eval("AlphaZeroGameInfo ~= nil") is True, "no exporter; silence is vacuous"
        runtime.execute(FLAG_OFF)
        assert _reinitialise(runtime) is True
        assert runtime.eval(f"AlphaZeroConfig.Get('{FLAG}')") is False
        assert _gameinfo_lines(runtime) == []
        assert runtime.eval("AlphaZeroBridge.GetStatus().initialized") is True

    @pytest.mark.parametrize(
        "broken",
        ["function() error('export boom') end", "function() return false, 'no luck' end"],
        ids=["raises", "returns-false"],
    )
    def test_a_failing_export_does_not_stop_the_mod_initialising(self, broken: str) -> None:
        runtime = g.load_printing()
        runtime.execute(f"AlphaZeroGameInfo.ExportAndWrite = {broken}")
        assert _reinitialise(runtime) is True
        assert runtime.eval("AlphaZeroBridge.GetStatus().initialized") is True
        assert _outcome(runtime.eval("AlphaZeroBridge.ExportState()"))[0] is True
        states = g.messages(g.printed(runtime), "AZ_STATE")
        assert json.loads(states[-1][0]) == h.export_state(runtime)

    def test_a_game_without_the_tables_still_initialises_and_prints_what_it_has(self) -> None:
        runtime = g.load_printing("GameInfo.Units = nil\nGameInfo.Civics = nil")
        assert runtime.eval("AlphaZeroBridge.GetStatus().initialized") is True
        assert _message_documents(runtime) == [_without_tables("Civics", "Units")]


class TestExistingPrintsAndTheLogReader:
    def test_state_and_heartbeat_prints_are_unchanged(self) -> None:
        runtime = g.load_printing()
        assert any(line.startswith("AZ_HB:") for line in g.printed(runtime))
        g.clear_printed(runtime)
        assert _outcome(runtime.eval("AlphaZeroBridge.ExportState()"))[0] is True
        lines = g.printed(runtime)
        assert _gameinfo_lines(runtime) == []
        ((text, _),) = g.messages(lines, "AZ_STATE")
        assert json.loads(text) == h.export_state(runtime)

    def test_the_log_tailer_ignores_every_gameinfo_line(self, tmp_path: Path) -> None:
        runtime = g.load_printing(g.extra_units(200))
        lines = _gameinfo_lines(runtime)
        assert any(line.startswith("AZ_GAMEINFO_CHUNK:") for line in lines)
        tailer = LogTailer(tmp_path / "Lua.log")
        for line in lines:
            tailer._parse_line(f"AlphaZeroMain: {line}")
        assert tailer.latest_state is None
        stats = tailer.stats
        assert (stats["states_received"], stats["chunks_received"], stats["parse_errors"]) == (
            0,
            0,
            0,
        )
        assert tailer._chunk_buffer == {}
        # Control: the same reader does take a state line, so the zeros above are not vacuous.
        tailer._parse_line('AlphaZeroMain: AZ_STATE:{"turn_id": 1}')
        assert tailer.stats["states_received"] == 1
