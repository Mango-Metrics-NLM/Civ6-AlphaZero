"""GM-B4 third review round, the mod's side (lupa, Lua 5.1).

L1  a game table the exporter cannot read whole is left out whole.
L2  a failed attempt in file mode cannot leave the previous snapshot looking current:
    ``BeginGameInfo`` overwrites the file with the BEGIN marker, ``Delete`` tells the truth.
L3  overriding a file name (or the directory) moves its path: ``AlphaZeroConfig.ApplyOverrides``.

These need the regenerated bundle (``python scripts/build_lua_bundle.py``). The file-access
branch runs against the in-memory ``io`` of ``test_lua_gameinfo_export_review2`` (the harness's
lockdown stays in force). Tests named ``control`` pass on the pre-fix tree on purpose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.bridge import gameinfo_capture as gc
from tests.fixtures import gameinfo_snapshot as g
from tests.unit.test_lua_gameinfo_export_review2 import EMPTY_GAME, FAILED, FAKE_FILES

BEGIN = "AZ_GAMEINFO_BEGIN"


def _lua(value: Any) -> str:
    """A Lua literal for a JSON-like value (ASCII strings, numbers, booleans, lists, dicts)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "{" + ", ".join(_lua(v) for v in value) + "}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"[{json.dumps(k)}] = {_lua(v)}" for k, v in value.items()) + "}"
    raise TypeError(value)


def _tuple(result: Any) -> tuple[Any, ...]:
    return result if isinstance(result, tuple) else (result,)


# ---------------------------------------------------------------------------
# L1: a table is exported whole or not at all
# ---------------------------------------------------------------------------
GOOD_TECH = "{ TechnologyType = 'TECH_A', Cost = 1, EraType = 'ERA_ANCIENT' }"

#: Rows (Lua source) that make the Technologies table unreadable.
BAD_TECH_ROWS = {
    "fractional-number": "{ TechnologyType = 'TECH_B', Cost = 1.5, EraType = 'ERA_ANCIENT' }",
    "number-beyond-2-to-53": "{ TechnologyType = 'TECH_B', Cost = 2^53, EraType = 'ERA_ANCIENT' }",
    "infinity": "{ TechnologyType = 'TECH_B', Cost = 1/0, EraType = 'ERA_ANCIENT' }",
    "text-with-a-space": "{ TechnologyType = 'TECH B', Cost = 1, EraType = 'ERA_ANCIENT' }",
    "text-with-a-hyphen": "{ TechnologyType = 'TECH-B', Cost = 1, EraType = 'ERA_ANCIENT' }",
    "text-starting-with-a-digit": "{ TechnologyType = '1TECH', Cost = 1 }",
    "empty-text": "{ TechnologyType = 'TECH_B', Cost = 1, EraType = '' }",
    "boolean-true": "{ TechnologyType = 'TECH_B', Cost = true }",
    "boolean-false": "{ TechnologyType = 'TECH_B', Cost = false }",
    "a-table-value": "{ TechnologyType = 'TECH_B', Cost = { 1 } }",
    "a-function-value": "{ TechnologyType = 'TECH_B', Cost = function() end }",
    "no-requested-column-at-all": "{ Other = 'TECH_B', Description = 'LOC_X' }",
    "an-empty-row": "{ }",
}


def _tech_setup(*rows: str) -> str:
    return g.replace_table("Technologies", "{ " + ", ".join(rows) + " }")


def _export(setup: str) -> tuple[dict[str, Any], list[str]]:
    runtime = g.h.load_mod(setup=g.PRINT_CAPTURE + setup)
    document, failed, _ = g.lua_export(runtime)
    return document, failed


class TestATableIsExportedWholeOrNotAtAll:
    @pytest.mark.parametrize("case", BAD_TECH_ROWS)
    def test_one_unreadable_row_leaves_the_whole_table_out(self, case: str) -> None:
        document, failed = _export(_tech_setup(GOOD_TECH, BAD_TECH_ROWS[case]))
        assert "Technologies" not in document, document.get("Technologies")
        assert failed == ["Technologies"]

    @pytest.mark.parametrize("case", ["fractional-number", "no-requested-column-at-all"])
    def test_wherever_the_bad_row_stands(self, case: str) -> None:
        for rows in (
            (BAD_TECH_ROWS[case], GOOD_TECH),
            (GOOD_TECH, BAD_TECH_ROWS[case], GOOD_TECH.replace("TECH_A", "TECH_C")),
        ):
            document, failed = _export(_tech_setup(*rows))
            assert "Technologies" not in document
            assert failed == ["Technologies"]

    def test_the_other_tables_are_still_exported(self) -> None:
        document, failed = _export(_tech_setup(GOOD_TECH, BAD_TECH_ROWS["fractional-number"]))
        expected = {k: v for k, v in g.STUB_DOCUMENT.items() if k != "Technologies"}
        assert document == expected
        assert failed == ["Technologies"]

    @pytest.mark.parametrize("table", ["Civics", "TechnologyPrereqs", "Units", "Buildings"])
    def test_it_holds_for_every_table(self, table: str) -> None:
        bad = {
            "Civics": "{ CivicType = 'CIVIC_B', Cost = 0.5 }",
            "TechnologyPrereqs": "{ Technology = 'TECH_A', PrereqTech = 'TECH A' }",
            "Units": "{ UnitType = 'UNIT_B', Cost = 1, Combat = true }",
            "Buildings": "{ Nothing = 1 }",
        }[table]
        document, failed = _export(g.replace_table(table, "{ " + bad + " }"))
        assert table not in document
        assert failed == [table]

    def test_control_an_absent_column_is_left_out_of_that_row_only(self) -> None:
        document, failed = _export(
            _tech_setup(
                GOOD_TECH,
                "{ TechnologyType = 'TECH_B', Cost = 2 }",  # no EraType
                "{ TechnologyType = 'TECH_C', EraType = 'ERA_ANCIENT' }",  # no Cost
            )
        )
        assert failed == []
        assert document["Technologies"] == [
            {"TechnologyType": "TECH_A", "Cost": 1, "EraType": "ERA_ANCIENT"},
            {"TechnologyType": "TECH_B", "Cost": 2},
            {"TechnologyType": "TECH_C", "EraType": "ERA_ANCIENT"},
        ]

    def test_control_an_empty_table_is_exported_empty(self) -> None:
        document, failed = _export(_tech_setup())
        assert document["Technologies"] == []
        assert failed == []

    def test_control_negative_whole_numbers_and_the_largest_safe_one_are_readable(self) -> None:
        document, failed = _export(
            g.replace_table(
                "Units",
                "{ { UnitType = 'UNIT_A', Cost = 9007199254740991, Combat = -1, BaseMoves = 0 } }",
            )
        )
        assert failed == []
        assert document["Units"] == [
            {"UnitType": "UNIT_A", "Cost": 9007199254740991, "Combat": -1, "BaseMoves": 0}
        ]

    def test_control_the_stub_game_still_exports_everything(self) -> None:
        document, failed = _export("")
        assert (document, failed) == (g.STUB_DOCUMENT, [])

    def test_end_to_end_the_capture_refuses_a_snapshot_that_lacks_the_table(self) -> None:
        runtime = g.load_printing(_tech_setup(GOOD_TECH, BAD_TECH_ROWS["fractional-number"]))
        scan = gc.scan_log(g.log_text(g.printed(runtime)))
        assert scan.document is not None and "Technologies" not in scan.document
        with pytest.raises(gc.GameInfoCaptureError) as caught:
            gc.check_snapshot(scan.document)
        assert "Technologies" in str(caught.value)


# ---------------------------------------------------------------------------
# L2: BeginGameInfo and Delete
# ---------------------------------------------------------------------------
PATH_EXPR = "AlphaZeroConfig.Get('GAMEINFO_PATH')"


def _file_mode(extra: str = "") -> Any:
    return g.load_printing(FAKE_FILES + extra)


def _content(runtime: Any) -> Any:
    return runtime.eval(f"FakeFS[{PATH_EXPR}]")


class TestBeginGameInfoInFileMode:
    def test_it_overwrites_the_file_with_the_marker_and_returns_true(self) -> None:
        runtime = _file_mode()
        assert _content(runtime) is not None, "premise: load 1 wrote the snapshot"
        assert runtime.eval("AlphaZeroFileIO.BeginGameInfo()") is True
        assert str(_content(runtime)).strip() == BEGIN
        assert len(str(_content(runtime)).strip().splitlines()) == 1

    def test_it_creates_the_marker_when_there_was_no_file(self) -> None:
        runtime = _file_mode()
        runtime.execute(f"FakeFS[{PATH_EXPR}] = nil")
        assert runtime.eval("AlphaZeroFileIO.BeginGameInfo()") is True
        assert str(_content(runtime)).strip() == BEGIN

    def test_it_does_not_depend_on_being_able_to_delete(self) -> None:
        runtime = _file_mode("os.remove = nil")
        assert runtime.eval("AlphaZeroFileIO.BeginGameInfo()") is True
        assert str(_content(runtime)).strip() == BEGIN

    def test_it_returns_false_when_the_write_fails_and_prints_no_begin_line(self) -> None:
        runtime = _file_mode()
        runtime.execute(
            "local real = io.open\nio.open = function(p, m) if m == 'w' then return nil, 'denied' end return real(p, m) end"
        )
        g.clear_printed(runtime)
        assert runtime.eval("AlphaZeroFileIO.BeginGameInfo()") is False
        assert BEGIN not in g.printed(runtime)

    def test_it_clears_the_exposed_value_whether_or_not_the_write_works(self) -> None:
        runtime = _file_mode()
        runtime.execute("ExposedMembers.AlphaZeroGameInfo = 'stale'")
        runtime.execute("io.open = function() return nil, 'denied' end")
        runtime.eval("AlphaZeroFileIO.BeginGameInfo()")
        assert runtime.eval("ExposedMembers.AlphaZeroGameInfo") is None

    def test_a_failed_export_attempt_leaves_the_marker_not_the_old_snapshot(self) -> None:
        runtime = _file_mode("os.remove = nil")
        runtime.execute(EMPTY_GAME)
        g.clear_printed(runtime)
        result = _tuple(runtime.eval("AlphaZeroGameInfo.ExportAndWrite()"))
        assert result[:2] == (False, FAILED)
        assert str(_content(runtime)).strip() == BEGIN

    def test_a_good_attempt_after_the_marker_replaces_it_with_the_document(self) -> None:
        runtime = _file_mode()
        runtime.eval("AlphaZeroFileIO.BeginGameInfo()")
        assert _tuple(runtime.eval("AlphaZeroGameInfo.ExportAndWrite()"))[0] is True
        assert json.loads(_content(runtime)) == g.STUB_DOCUMENT


class TestDeleteTellsTheTruth:
    def _fs(self, extra: str = "", *, present: bool = True) -> Any:
        runtime = _file_mode(extra)
        runtime.execute(f"FakeFS['victim'] = {'nil' if not present else repr('x')}")
        return runtime

    @staticmethod
    def _delete(runtime: Any) -> tuple[Any, ...]:
        return _tuple(runtime.eval("AlphaZeroFileIO.Delete('victim')"))

    def test_control_removed_is_true_and_the_file_is_gone(self) -> None:
        runtime = self._fs()
        assert self._delete(runtime)[0] is True
        assert runtime.eval("FakeFS['victim']") is None

    def test_a_file_that_was_not_there_is_true(self) -> None:
        # The fake os.remove answers nil for a missing file: Delete must not call that a failure.
        runtime = self._fs(present=False)
        assert self._delete(runtime)[0] is True

    def test_remove_answering_nil_while_the_file_stays_is_false_with_a_reason(self) -> None:
        runtime = self._fs("os.remove = function() return nil, 'locked' end")
        result = self._delete(runtime)
        assert result[0] is False
        assert isinstance(result[1], str) and result[1]
        assert runtime.eval("FakeFS['victim']") == "x", "premise: the file really is still there"

    def test_remove_answering_nil_with_no_reason_is_still_false(self) -> None:
        runtime = self._fs("os.remove = function() return nil end")
        assert self._delete(runtime)[0] is False

    def test_no_os_remove_and_the_file_is_there_is_false_with_a_reason(self) -> None:
        runtime = self._fs("os.remove = nil")
        result = self._delete(runtime)
        assert result[0] is False
        assert isinstance(result[1], str) and result[1]

    def test_no_os_remove_and_no_file_is_true(self) -> None:
        runtime = self._fs("os.remove = nil", present=False)
        assert self._delete(runtime)[0] is True

    def test_control_remove_raising_while_the_file_stays_is_false(self) -> None:
        runtime = self._fs("os.remove = function() error('boom') end")
        assert self._delete(runtime)[0] is False

    def test_remove_that_raises_after_removing_is_true(self) -> None:
        runtime = self._fs("os.remove = function(p) FakeFS[p] = nil error('late') end")
        assert self._delete(runtime)[0] is True

    def test_control_clear_all_still_removes_the_state_file(self) -> None:
        runtime = _file_mode()
        runtime.execute("FakeFS[AlphaZeroConfig.Get('STATE_PATH')] = 'x'")
        runtime.execute("AlphaZeroFileIO.ClearAll()")
        assert runtime.eval("FakeFS[AlphaZeroConfig.Get('STATE_PATH')]") is None


class TestEndToEndAFailedSecondLoad:
    @pytest.mark.parametrize(
        "remove",
        ["os.remove = nil", "os.remove = function() return nil, 'locked' end"],
        ids=["no-os-remove", "remove-fails"],
    )
    def test_load_2_reads_no_table_and_from_file_exits_1(
        self,
        remove: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        data = tmp_path / "data"
        data.mkdir()
        monkeypatch.setenv("CIV6AZ_DATA_ROOT", str(data))
        monkeypatch.chdir(tmp_path)

        load_one = _file_mode()
        path = load_one.eval(PATH_EXPR)
        snapshot = load_one.eval(f"FakeFS[{PATH_EXPR}]")
        assert json.loads(snapshot) == g.STUB_DOCUMENT, "premise: load 1 exported"

        # Load 2 is another game load on the same file system: the previous file is there,
        # the game can read no table, and the mod cannot delete.
        carry = f"FakeFS[{json.dumps(path)}] = {json.dumps(snapshot)}\n"
        load_two = g.load_printing(FAKE_FILES + carry + remove + "\n" + EMPTY_GAME)
        assert str(load_two.eval(f"FakeFS[{PATH_EXPR}]")).strip() == BEGIN

        shared = tmp_path / "gameinfo.json"
        shared.write_text(load_two.eval(f"FakeFS[{PATH_EXPR}]"), encoding="utf-8")
        assert gc.main(["--from-file", str(shared), "--name", "n"]) == 1
        err = capsys.readouterr().err
        assert "Traceback" not in err and "began" in err.lower()
        assert not list(data.rglob("*.json"))

    def test_control_load_1_alone_is_filed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = tmp_path / "data"
        data.mkdir()
        monkeypatch.setenv("CIV6AZ_DATA_ROOT", str(data))
        monkeypatch.chdir(tmp_path)
        runtime = _file_mode()
        shared = tmp_path / "gameinfo.json"
        shared.write_text(runtime.eval(f"FakeFS[{PATH_EXPR}]"), encoding="utf-8")
        assert gc.main(["--from-file", str(shared), "--name", "n"]) == 0


# ---------------------------------------------------------------------------
# L3: ApplyOverrides
# ---------------------------------------------------------------------------
FILES = ("STATE", "COMMANDS", "ACK", "HEARTBEAT", "ERROR", "GAMEINFO")


def _runtime() -> Any:
    runtime = g.h.load_mod()
    assert runtime.eval("type(AlphaZeroConfig.ApplyOverrides)") == "function", (
        "AlphaZeroConfig.ApplyOverrides is missing"
    )
    return runtime


def _paths(runtime: Any) -> dict[str, str]:
    return {n: str(runtime.eval(f"AlphaZeroConfig.Get('{n}_PATH')")) for n in FILES}


def _apply(runtime: Any, overrides: dict[str, Any]) -> None:
    runtime.execute(f"AlphaZeroConfig.ApplyOverrides({_lua(overrides)})")


def _sep(runtime: Any) -> str:
    return str(runtime.eval("AlphaZeroConfig.Get('PATH_SEP')"))


def _comm_dir(runtime: Any) -> str:
    return str(runtime.eval("AlphaZeroConfig.Get('COMM_DIR')"))


class TestApplyOverrides:
    @pytest.mark.parametrize("name", FILES)
    def test_a_file_name_override_moves_that_path_only(self, name: str) -> None:
        runtime = _runtime()
        before = _paths(runtime)
        _apply(runtime, {f"{name}_FILE": "renamed.dat"})
        after = _paths(runtime)
        assert after[name] == _comm_dir(runtime) + _sep(runtime) + "renamed.dat"
        assert {k: v for k, v in after.items() if k != name} == {
            k: v for k, v in before.items() if k != name
        }

    @pytest.mark.parametrize("name", FILES)
    def test_an_explicit_path_override_wins_over_the_file_name(self, name: str) -> None:
        runtime = _runtime()
        _apply(runtime, {f"{name}_FILE": "renamed.dat", f"{name}_PATH": "/explicit/p.dat"})
        assert _paths(runtime)[name] == "/explicit/p.dat"

    @pytest.mark.parametrize("name", FILES)
    def test_an_explicit_path_alone_is_kept_as_given(self, name: str) -> None:
        runtime = _runtime()
        _apply(runtime, {f"{name}_PATH": "/explicit/p.dat"})
        assert _paths(runtime)[name] == "/explicit/p.dat"

    def test_a_directory_override_moves_all_six_paths(self) -> None:
        runtime = _runtime()
        sep = _sep(runtime)
        new_dir = "X" + sep + "custom"
        before_files = {n: str(runtime.eval(f"AlphaZeroConfig.Get('{n}_FILE')")) for n in FILES}
        _apply(runtime, {"COMM_DIR": new_dir})
        assert _paths(runtime) == {n: new_dir + sep + before_files[n] for n in FILES}

    @pytest.mark.parametrize("name", FILES)
    def test_a_directory_override_leaves_an_explicit_path_alone(self, name: str) -> None:
        runtime = _runtime()
        sep = _sep(runtime)
        new_dir = "X" + sep + "custom"
        _apply(runtime, {"COMM_DIR": new_dir, f"{name}_PATH": "/explicit/p.dat"})
        paths = _paths(runtime)
        assert paths[name] == "/explicit/p.dat"
        for other in FILES:
            if other != name:
                assert paths[other].startswith(new_dir + sep), (other, paths[other])

    def test_a_directory_and_a_file_name_together(self) -> None:
        runtime = _runtime()
        sep = _sep(runtime)
        new_dir = "X" + sep + "custom"
        _apply(runtime, {"COMM_DIR": new_dir, "GAMEINFO_FILE": "snap.json"})
        paths = _paths(runtime)
        assert paths["GAMEINFO"] == new_dir + sep + "snap.json"
        assert paths["STATE"] == new_dir + sep + "state.json"

    def test_control_unrelated_overrides_move_no_path(self) -> None:
        runtime = _runtime()
        before = _paths(runtime)
        _apply(runtime, {"LOG_LEVEL": "DEBUG", "MAX_ACTIONS_PER_TURN": 7})
        assert _paths(runtime) == before
        assert runtime.eval("AlphaZeroConfig.Get('MAX_ACTIONS_PER_TURN')") == 7

    def test_the_file_the_mod_writes_follows_the_renamed_path(self) -> None:
        runtime = _file_mode()
        _apply(runtime, {"GAMEINFO_FILE": "renamed.json"})
        assert _tuple(runtime.eval("AlphaZeroGameInfo.ExportAndWrite()"))[0] is True
        assert json.loads(_content(runtime)) == g.STUB_DOCUMENT
        assert str(runtime.eval(PATH_EXPR)).endswith("renamed.json")
        # Only the renamed file's path moved.
        assert str(runtime.eval("AlphaZeroConfig.Get('STATE_PATH')")).endswith("state.json")


class TestTheOverrideFileGoesThroughIt:
    """``alphazero_config.json`` in the communication directory, read through the fake ``io``."""

    def _load_with_override_file(self, overrides: dict[str, Any]) -> Any:
        runtime = _file_mode()
        config_path = _comm_dir(runtime) + _sep(runtime) + "alphazero_config.json"
        runtime.execute(f"FakeFS[{json.dumps(config_path)}] = {json.dumps(json.dumps(overrides))}")
        runtime.execute("AlphaZeroConfig.LoadOverrides()")
        return runtime

    @pytest.mark.parametrize("name", FILES)
    def test_a_file_name_in_the_override_file_moves_its_path(self, name: str) -> None:
        runtime = self._load_with_override_file({f"{name}_FILE": "renamed.dat"})
        assert _paths(runtime)[name] == _comm_dir(runtime) + _sep(runtime) + "renamed.dat"

    def test_control_a_directory_in_the_override_file_moves_all_six_paths(self) -> None:
        sep = _sep(_file_mode())
        new_dir = "X" + sep + "custom"
        runtime = self._load_with_override_file({"COMM_DIR": new_dir})
        assert all(p.startswith(new_dir + sep) for p in _paths(runtime).values())
