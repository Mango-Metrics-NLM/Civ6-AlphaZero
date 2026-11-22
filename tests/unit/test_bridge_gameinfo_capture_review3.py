"""GM-B4 third review round: the capture, the strict reader and the civic check (Python side).

P1  strict JSON: ``alpha_zero.rules.gameinfo.loads`` and every place the capture parses.
P2  civic names are checked before filing, by the research registry's own rule.
P3  a half-written last log line is "unfinished", not an error.
P4  the printed follow-up commands survive a path with spaces.
L2  (Python half) a file holding only the BEGIN marker is refused with its own message.

No Lua runtime is needed here. Tests named ``control`` pass on the pre-fix tree on purpose:
they pin behaviour the fix must keep, or reject what the fix must keep rejecting.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from alpha_zero.bridge import gameinfo_capture as gc
from alpha_zero.bridge import tech_registry as tr
from alpha_zero.rules import gameinfo as rg
from tests.fixtures import gameinfo_snapshot as g

DATA_ROOT_ENV = "CIV6AZ_DATA_ROOT"
DOC = g.MINIMAL_DOCUMENT
TEXT = g.compact_text(DOC)
OLDER: dict[str, Any] = {**DOC, "Technologies": DOC["Technologies"][:1], "TechnologyPrereqs": []}
HUGE = "9" * 5000  # more digits than Python converts (limit 4300)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv(DATA_ROOT_ENV, str(data))
    monkeypatch.chdir(tmp_path)
    return data


def _filed(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file())


def _chunked(text: str, pieces: int = 3) -> str:
    size = -(-len(text) // pieces)
    parts = [text[i : i + size] for i in range(0, len(text), size)]
    lines = [f"AZ_GAMEINFO_CHUNK:{i}/{len(parts)}:{p}\n" for i, p in enumerate(parts, 1)]
    return "".join(lines) + g.end_line(len(parts), g.adler32(text))


def _single(text: str) -> str:
    return f"AZ_GAMEINFO:{text}\n"


def _one_line_error(capsys: pytest.CaptureFixture[str]) -> str:
    err = capsys.readouterr().err
    assert "Traceback" not in err
    lines = err.strip().splitlines()
    assert len(lines) == 1, err
    assert lines[0].startswith("gameinfo_capture:")
    return lines[0]


# ---------------------------------------------------------------------------
# P1: strict JSON
# ---------------------------------------------------------------------------
BAD_TEXTS: dict[str, tuple[str, str | None]] = {
    # name -> (text, key the message must name)
    "huge-number": (TEXT.replace('"Cost":10', f'"Cost":{HUGE}', 1), None),
    "two-Technologies-tables": ('{"Technologies":[],' + TEXT[1:], "Technologies"),
    "Cost-twice-in-a-row": (TEXT.replace('"Cost":10', '"Cost":10,"Cost":11', 1), "Cost"),
    "same-key-same-value-in-another-table": (
        TEXT.replace('"PrereqTech":"TECH_A"', '"PrereqTech":"TECH_A","PrereqTech":"TECH_A"', 1),
        "PrereqTech",
    ),
}


class TestStrictLoads:
    def test_it_returns_what_json_returns_for_valid_text(self) -> None:
        assert rg.loads(TEXT) == DOC
        assert rg.loads("[1, 2]") == [1, 2]
        assert rg.loads('{"a": {"b": [1, null, "x"]}}') == {"a": {"b": [1, None, "x"]}}

    @pytest.mark.parametrize(
        ("text", "key"),
        [
            ('{"a":1,"a":2}', "a"),
            ('{"a":{"b":1,"b":2}}', "b"),
            ('{"x":[{"Cost":1,"Cost":2}]}', "Cost"),
            ('[{"k":1},{"z":1,"z":1}]', "z"),
        ],
    )
    def test_a_duplicate_key_at_any_depth_names_the_key(self, text: str, key: str) -> None:
        with pytest.raises(rg.GameInfoError) as caught:
            rg.loads(text)
        assert key in str(caught.value)

    def test_an_oversized_number_is_a_game_info_error_not_a_bare_value_error(self) -> None:
        with pytest.raises(rg.GameInfoError):
            rg.loads('{"Cost":' + HUGE + "}")

    @pytest.mark.parametrize("text", ["", "{", '{"a":}', "not json", "{'a': 1}", '{"a":1} x'])
    def test_anything_that_is_not_json_is_a_game_info_error(self, text: str) -> None:
        with pytest.raises(rg.GameInfoError):
            rg.loads(text)

    def test_import_gameinfo_reads_through_the_same_strict_reader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Control for the existing behaviour: both refusals already happen in the importer.
        monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "data"))
        for name in ("two-Technologies-tables", "huge-number"):
            source = tmp_path / f"{name}.json"
            source.write_text(BAD_TEXTS[name][0], encoding="utf-8")
            with pytest.raises(rg.GameInfoError):
                rg.import_gameinfo(source, str(tmp_path / "out" / f"{name}.yaml"))


class TestTheCaptureReadsStrictly:
    @pytest.mark.parametrize("case", BAD_TEXTS)
    def test_a_file(self, case: str, tmp_path: Path) -> None:
        text, key = BAD_TEXTS[case]
        path = tmp_path / "gameinfo.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(gc.GameInfoCaptureError) as caught:
            gc.snapshot_from_file(path)
        if key:
            assert key in str(caught.value)

    @pytest.mark.parametrize("case", BAD_TEXTS)
    def test_a_single_line_message(self, case: str) -> None:
        text, key = BAD_TEXTS[case]
        with pytest.raises(gc.GameInfoCaptureError) as caught:
            gc.scan_log(_single(text))
        if key:
            assert key in str(caught.value)

    @pytest.mark.parametrize("case", BAD_TEXTS)
    def test_a_joined_chunk_set(self, case: str) -> None:
        text, key = BAD_TEXTS[case]
        with pytest.raises(gc.GameInfoCaptureError) as caught:
            gc.scan_log(_chunked(text))
        if key:
            assert key in str(caught.value)

    @pytest.mark.parametrize("case", BAD_TEXTS)
    @pytest.mark.parametrize("source", ["file", "single", "chunked"])
    def test_the_command_line_exits_1_with_one_line_and_files_nothing(
        self,
        case: str,
        source: str,
        root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        text, key = BAD_TEXTS[case]
        path = tmp_path / "input"
        if source == "file":
            path.write_text(text, encoding="utf-8")
            argv = ["--from-file", str(path)]
        else:
            path.write_text(_single(text) if source == "single" else _chunked(text), "utf-8")
            argv = ["--from-log", str(path)]
        assert gc.main([*argv, "--name", "n"]) == 1
        message = _one_line_error(capsys)
        if key:
            assert key in message
        assert _filed(root) == []

    def test_control_valid_text_is_read_by_all_three(self, tmp_path: Path) -> None:
        path = tmp_path / "gameinfo.json"
        path.write_text(TEXT, encoding="utf-8")
        assert gc.snapshot_from_file(path) == DOC
        assert gc.scan_log(_single(TEXT)).document == DOC
        assert gc.scan_log(_chunked(TEXT)).document == DOC


# ---------------------------------------------------------------------------
# P2: civic names
# ---------------------------------------------------------------------------
def _with_civics(rows: Any) -> dict[str, Any]:
    return {**DOC, "Civics": rows}


def _civic(name: str) -> dict[str, Any]:
    return {"CivicType": name, "Cost": 10, "EraType": "ERA_ANCIENT"}


#: Civics tables the registry cannot number. ``now`` says whether the pre-fix capture already
#: refuses it (those cases are controls).
REFUSED_CIVICS: dict[str, tuple[Any, bool]] = {
    "row-not-an-object": (["CIVIC_X"], True),
    "row-is-a-number": ([7], True),
    "row-without-CivicType": ([{"Cost": 10, "EraType": "ERA_ANCIENT"}], False),
    "CivicType-null": ([{"CivicType": None, "Cost": 10}], False),
    "CivicType-a-number": ([{"CivicType": 5, "Cost": 10}], False),
    "CivicType-empty": ([{"CivicType": "", "Cost": 10}], True),
    "same-name-twice": ([_civic("CIVIC_X"), _civic("CIVIC_X")], False),
    "same-name-other-case": ([_civic("CIVIC_X"), _civic("civic_x")], False),
    "same-civic-with-and-without-prefix": (
        [_civic("CODE_OF_LAWS"), _civic("CIVIC_CODE_OF_LAWS")],
        False,
    ),
    "prefix-form-first": ([_civic("CIVIC_CODE_OF_LAWS"), _civic("code_of_laws")], False),
}


class TestCivicNamesAreCheckedBeforeFiling:
    @pytest.mark.parametrize("case", REFUSED_CIVICS)
    def test_check_snapshot_refuses(self, case: str) -> None:
        with pytest.raises(gc.GameInfoCaptureError):
            gc.check_snapshot(_with_civics(REFUSED_CIVICS[case][0]))

    @pytest.mark.parametrize("case", REFUSED_CIVICS)
    def test_nothing_is_filed_and_the_command_line_exits_1(
        self, case: str, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "gameinfo.json"
        path.write_text(json.dumps(_with_civics(REFUSED_CIVICS[case][0])), encoding="utf-8")
        assert gc.main(["--from-file", str(path), "--name", "n"]) == 1
        _one_line_error(capsys)
        assert _filed(root) == []

    def test_a_duplicate_names_the_civic(self) -> None:
        with pytest.raises(gc.GameInfoCaptureError) as caught:
            gc.check_snapshot(_with_civics([_civic("CIVIC_X"), _civic("civic_x")]))
        assert "civic_x" in str(caught.value).lower()

    def test_control_distinct_civics_are_accepted(self) -> None:
        gc.check_snapshot(
            _with_civics([_civic("CIVIC_CODE_OF_LAWS"), _civic("CIVIC_FOREIGN_TRADE")])
        )
        gc.check_snapshot(_with_civics([]))
        gc.check_snapshot(g.STUB_DOCUMENT)

    @pytest.mark.parametrize("case", [*REFUSED_CIVICS, "ok", "ok-empty", "ok-prefix-mix"])
    def test_property_whatever_check_snapshot_accepts_the_registry_loads(
        self, case: str, tmp_path: Path
    ) -> None:
        rows = {
            "ok": [_civic("CIVIC_A"), _civic("CIVIC_B")],
            "ok-empty": [],
            "ok-prefix-mix": [_civic("CODE_OF_LAWS"), _civic("CIVIC_FOREIGN_TRADE")],
        }.get(case) or REFUSED_CIVICS.get(case, ([], True))[0]
        document = _with_civics(rows)
        try:
            gc.check_snapshot(document)
        except gc.GameInfoCaptureError:
            return
        path = tmp_path / "accepted.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        tr.load_registry("science-v1-draft", path)  # must not raise

    def test_a_filed_snapshot_loads_as_a_registry(self, root: Path) -> None:
        written = gc.write_snapshot(_with_civics([_civic("CIVIC_A"), _civic("CIVIC_B")]), name="s")
        registry = tr.load_registry("science-v1-draft", written)
        assert registry.civic_ids == ("civic_a", "civic_b")


class TestCivicIds:
    def test_lower_case_ids_in_order(self) -> None:
        assert tr.civic_ids(["CIVIC_B", "CIVIC_A", "Civic_C"]) == ("civic_b", "civic_a", "civic_c")
        assert tr.civic_ids(iter(["X", "Y"])) == ("x", "y")
        assert tr.civic_ids([]) == ()

    @pytest.mark.parametrize("bad", [None, 5, True, "", b"x", ["x"], {"a": 1}])
    def test_a_name_that_is_not_a_non_empty_string_is_refused(self, bad: Any) -> None:
        with pytest.raises(tr.ResearchRegistryError):
            tr.civic_ids(["CIVIC_A", bad])

    @pytest.mark.parametrize(
        "names",
        [
            ["CIVIC_X", "CIVIC_X"],
            ["CIVIC_X", "civic_x"],
            ["CODE_OF_LAWS", "CIVIC_CODE_OF_LAWS"],
            ["civic_code_of_laws", "CODE_OF_LAWS"],
        ],
    )
    def test_a_repeat_or_the_same_game_type_is_refused(self, names: list[str]) -> None:
        with pytest.raises(tr.ResearchRegistryError):
            tr.civic_ids(names)

    def test_the_same_game_type_error_names_both(self) -> None:
        with pytest.raises(tr.ResearchRegistryError) as caught:
            tr.civic_ids(["CODE_OF_LAWS", "CIVIC_CODE_OF_LAWS"])
        message = str(caught.value).lower()
        assert "code_of_laws" in message

    def test_from_ruleset_uses_the_same_rule_and_keeps_its_behaviour(self) -> None:
        spec = tr._load_spec("science-v1-draft")
        assert tr.from_ruleset(spec, ["CIVIC_B", "CIVIC_A"]).civic_ids == ("civic_b", "civic_a")
        for names in (["CIVIC_A", "civic_a"], ["CODE_OF_LAWS", "CIVIC_CODE_OF_LAWS"], [""], [3]):
            with pytest.raises(tr.ResearchRegistryError):
                tr.from_ruleset(spec, names)  # type: ignore[arg-type]


class TestCivicTypesOf:
    def test_the_civic_types_in_row_order(self) -> None:
        assert tr.civic_types_of(g.STUB_DOCUMENT, "here") == (
            "CIVIC_CODE_OF_LAWS",
            "CIVIC_CRAFTSMANSHIP",
            "CIVIC_FOREIGN_TRADE",
            "CIVIC_EARLY_EMPIRE",
        )
        assert tr.civic_types_of({"Civics": []}, "here") == ()

    def test_it_does_not_apply_the_uniqueness_rule(self) -> None:
        # Uniqueness is civic_ids' job; this only reads the column.
        assert tr.civic_types_of({"Civics": [_civic("A"), _civic("A")]}, "here") == ("A", "A")

    @pytest.mark.parametrize(
        "document",
        [
            [],
            "Civics",
            None,
            {},
            {"Civics": None},
            {"Civics": {"CivicType": "A"}},
            {"Civics": "A"},
            {"Civics": ["A"]},
            {"Civics": [{"Cost": 1}]},
            {"Civics": [_civic("A"), {"CivicType": 4}]},
            {"Civics": [{"CivicType": ""}]},
            {"Civics": [{"CivicType": None}]},
        ],
    )
    def test_a_document_it_cannot_read_is_refused_and_names_where(self, document: Any) -> None:
        with pytest.raises(tr.ResearchRegistryError) as caught:
            tr.civic_types_of(document, "THE-PLACE")
        assert "THE-PLACE" in str(caught.value)

    def test_control_civic_types_from_gameinfo_behaves_as_before(self, tmp_path: Path) -> None:
        path = tmp_path / "gi.json"
        path.write_text(json.dumps(g.STUB_DOCUMENT), encoding="utf-8")
        assert tr.civic_types_from_gameinfo(path)[:2] == (
            "CIVIC_CODE_OF_LAWS",
            "CIVIC_CRAFTSMANSHIP",
        )
        path.write_text(json.dumps({"Civics": [{"Cost": 1}]}), encoding="utf-8")
        with pytest.raises(tr.ResearchRegistryError) as caught:
            tr.civic_types_from_gameinfo(path)
        assert str(path) in str(caught.value)
        path.write_text("{", encoding="utf-8")
        with pytest.raises(tr.ResearchRegistryError):
            tr.civic_types_from_gameinfo(path)


# ---------------------------------------------------------------------------
# P3: a half-written last line
# ---------------------------------------------------------------------------
HALF = '{"Technologies":[{"TechnologyType":"TECH_A","Co'


class TestAHalfWrittenLastLine:
    @pytest.mark.parametrize("prefix", ["", "AlphaZeroMain: "])
    def test_it_does_not_replace_the_earlier_complete_message(self, prefix: str) -> None:
        text = g.single_line(OLDER, prefix) + f"{prefix}AZ_GAMEINFO:{HALF}"
        scan = gc.scan_log(text)
        assert (scan.document, scan.complete, scan.later_incomplete) == (OLDER, 1, True)

    def test_with_no_earlier_message_there_is_nothing(self) -> None:
        scan = gc.scan_log(f"AZ_GAMEINFO:{HALF}")
        assert (scan.document, scan.complete, scan.later_incomplete) == (None, 0, False)

    def test_after_an_earlier_chunked_message(self) -> None:
        text = _chunked(TEXT) + f"AZ_GAMEINFO:{HALF}"
        scan = gc.scan_log(text)
        assert (scan.document, scan.complete, scan.later_incomplete) == (DOC, 1, True)

    def test_a_last_line_that_parses_but_is_not_an_object_is_unfinished_too(self) -> None:
        scan = gc.scan_log(g.single_line(OLDER) + "AZ_GAMEINFO:[1]")
        assert (scan.document, scan.complete, scan.later_incomplete) == (OLDER, 1, True)

    def test_a_message_that_is_not_the_last_line_is_still_an_error(self) -> None:
        # Control: the same broken line followed by another line is a real error.
        with pytest.raises(gc.GameInfoCaptureError):
            gc.scan_log(g.single_line(OLDER) + f"AZ_GAMEINFO:{HALF}\nsomething else\n")

    @pytest.mark.parametrize("ending", ["\n", "\r\n"])
    def test_control_the_same_line_with_its_newline_is_still_an_error(self, ending: str) -> None:
        with pytest.raises(gc.GameInfoCaptureError):
            gc.scan_log(g.single_line(OLDER) + f"AZ_GAMEINFO:{HALF}{ending}")
        with pytest.raises(gc.GameInfoCaptureError):
            gc.scan_log(f"AZ_GAMEINFO:{HALF}{ending}")

    def test_control_an_unterminated_last_line_that_parses_is_a_complete_message(self) -> None:
        text = g.single_line(OLDER) + f"AZ_GAMEINFO:{TEXT}"
        scan = gc.scan_log(text)
        assert (scan.document, scan.complete, scan.later_incomplete) == (DOC, 2, False)

    def test_the_command_line_refuses_it_unless_allow_earlier(
        self, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log = tmp_path / "Lua.log"
        log.write_text(g.single_line(OLDER) + f"AZ_GAMEINFO:{HALF}", encoding="utf-8")
        assert gc.main(["--from-log", str(log), "--name", "n"]) == 1
        assert "--allow-earlier" in _one_line_error(capsys)
        assert _filed(root) == []

        assert gc.main(["--from-log", str(log), "--name", "n", "--allow-earlier"]) == 0
        (filed,) = _filed(root)
        assert json.loads(filed.read_text(encoding="utf-8")) == OLDER

    def test_with_no_earlier_message_the_command_line_says_there_is_none(
        self, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log = tmp_path / "Lua.log"
        log.write_text(f"AZ_GAMEINFO:{HALF}", encoding="utf-8")
        for extra in ([], ["--allow-earlier"]):
            assert gc.main(["--from-log", str(log), "--name", "n", *extra]) == 1
            assert "no complete GameInfo message" in _one_line_error(capsys)
        assert _filed(root) == []


# ---------------------------------------------------------------------------
# P4: printed commands and a path with spaces
# ---------------------------------------------------------------------------
class TestPrintedCommandsQuoteTheirPath:
    def _capture(self, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> tuple[str, str]:
        base.mkdir(parents=True)
        monkeypatch.setenv(DATA_ROOT_ENV, str(base))
        monkeypatch.chdir(base.parent)
        log = base.parent / "Lua.log"
        log.write_text(g.single_line(DOC), encoding="utf-8")
        assert gc.main(["--from-log", str(log), "--name", "n"]) == 0
        (filed,) = _filed(base)
        return str(filed.resolve()), capsys.readouterr().out

    # Only names every file system accepts: Windows refuses a tab in a file name. The tab
    # is covered below, on a path that is never created.
    @pytest.mark.parametrize("directory", ["my data", "two  spaces"])
    def test_a_path_with_whitespace_is_in_double_quotes(
        self,
        directory: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path, out = self._capture(tmp_path / directory, monkeypatch, capsys)
        lines = out.splitlines()
        assert f'  python -m alpha_zero.rules import-gameinfo "{path}"' in lines
        assert f'  CIV6AZ_BRIDGE_GAMEINFO="{path}"' in lines

    def test_control_a_path_without_whitespace_is_unchanged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The temporary directory's own spelling may hold a space (a Windows user profile),
        # so this checks the wiring: both lines print the path as the helper spells it.
        path, out = self._capture(tmp_path / "data", monkeypatch, capsys)
        shown = gc._for_a_shell(Path(path))
        lines = out.splitlines()
        assert f"  python -m alpha_zero.rules import-gameinfo {shown}" in lines
        assert f"  CIV6AZ_BRIDGE_GAMEINFO={shown}" in lines


class TestForAShell:
    """The spelling of a path in a printed command, on paths that are never created."""

    @pytest.mark.parametrize("windows", [False, True])
    def test_a_plain_path_is_unchanged(self, windows: bool) -> None:
        for text in ["/data/gameinfo/g-1.json", "C:/data/g.json", "rel/a_b-c.json"]:
            assert gc._for_a_shell(PurePosixPath(text), windows=windows) == text

    @pytest.mark.parametrize("windows", [False, True])
    @pytest.mark.parametrize("text", ["/my data/g.json", "/a\tb/g.json", "/two  spaces/g.json"])
    def test_whitespace_alone_gets_double_quotes(self, text: str, windows: bool) -> None:
        assert gc._for_a_shell(PurePosixPath(text), windows=windows) == f'"{text}"'

    @pytest.mark.parametrize(
        "text",
        [
            "/tmp/$HOME/game data/g.json",
            "/a`b`/g.json",
            '/say "hi"/g.json',
            "/it's/g.json",
            "/a;b/g.json",
            "/a%TEMP%/g.json",
        ],
    )
    def test_a_posix_shell_gets_the_literal_path_back(self, text: str) -> None:
        import shlex

        shown = gc._for_a_shell(PurePosixPath(text), windows=False)
        # What a POSIX shell would hand the program is the path itself, not an expansion.
        assert shlex.split(f"cmd {shown}") == ["cmd", text]
        assert shlex.split(f"CIV6AZ_BRIDGE_GAMEINFO={shown}") == [f"CIV6AZ_BRIDGE_GAMEINFO={text}"]
        assert not shown.startswith('"'), "double quotes would let the shell expand it"

    def test_on_windows_a_special_character_gets_double_quotes(self) -> None:
        assert gc._for_a_shell(PurePosixPath("C:/a&b/g.json"), windows=True) == '"C:/a&b/g.json"'

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("C:/$env:TEMP/g.json", "'C:/$env:TEMP/g.json'"),
            ("C:/%TEMP%/g.json", "'C:/%TEMP%/g.json'"),
            ("C:/a`b/g.json", "'C:/a`b/g.json'"),
            ("C:/it's $100/g.json", "'C:/it''s $100/g.json'"),
        ],
    )
    def test_on_windows_expanding_characters_use_single_quotes(
        self, text: str, expected: str
    ) -> None:
        assert gc._for_a_shell(PurePosixPath(text), windows=True) == expected


class TestACompleteButInvalidLastLineIsAnError:
    """A last line that is whole JSON and breaks a rule is refused, not called unfinished."""

    BAD = (
        '{"Technologies": [], "Technologies": []}',
        '{"Units": [{"UnitType": "UNIT_A", "Cost": ' + "9" * 5000 + "}]}",
    )

    @pytest.mark.parametrize("bad", BAD, ids=["duplicate key", "oversized number"])
    def test_scan_log_raises_even_with_an_earlier_complete_message(self, bad: str) -> None:
        text = g.single_line(DOC) + "\n" + gc.LINE_PREFIX + bad
        assert not text.endswith("\n"), "premise: the last line is unterminated"
        with pytest.raises(gc.GameInfoCaptureError):
            gc.scan_log(text)

    @pytest.mark.parametrize("bad", BAD, ids=["duplicate key", "oversized number"])
    def test_allow_earlier_does_not_file_the_older_snapshot(
        self,
        bad: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        base = tmp_path / "data"
        base.mkdir()
        monkeypatch.setenv(DATA_ROOT_ENV, str(base))
        log = tmp_path / "Lua.log"
        log.write_text(g.single_line(DOC) + "\n" + gc.LINE_PREFIX + bad, encoding="utf-8")
        assert gc.main(["--from-log", str(log), "--allow-earlier"]) == 1
        assert _filed(base) == []
        assert "Traceback" not in capsys.readouterr().err

    def test_control_half_a_line_is_still_unfinished(self) -> None:
        text = g.single_line(DOC) + "\n" + gc.LINE_PREFIX + '{"Technologies": [{"Technolo'
        scan = gc.scan_log(text)
        assert scan.document == DOC
        assert scan.later_incomplete is True


# ---------------------------------------------------------------------------
# L2 (Python half): the BEGIN marker as a file
# ---------------------------------------------------------------------------
class TestAFileThatHoldsOnlyTheBeginMarker:
    @pytest.mark.parametrize(
        "content", ["AZ_GAMEINFO_BEGIN", "AZ_GAMEINFO_BEGIN\n", "AZ_GAMEINFO_BEGIN\r\n"]
    )
    def test_snapshot_from_file_says_the_mod_began_and_did_not_finish(
        self, content: str, tmp_path: Path
    ) -> None:
        path = tmp_path / "gameinfo.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(gc.GameInfoCaptureError) as caught:
            gc.snapshot_from_file(path)
        message = str(caught.value).lower()
        assert "began" in message and "finish" in message
        assert "not valid json" not in message

    def test_the_command_line_exits_1_and_files_nothing(
        self, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "gameinfo.json"
        path.write_text("AZ_GAMEINFO_BEGIN\n", encoding="utf-8")
        assert gc.main(["--from-file", str(path), "--name", "n"]) == 1
        assert "began" in _one_line_error(capsys).lower()
        assert _filed(root) == []

    def test_control_other_garbage_still_says_not_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "gameinfo.json"
        path.write_text("AZ_GAMEINFO_BEGIN and more", encoding="utf-8")
        with pytest.raises(gc.GameInfoCaptureError) as caught:
            gc.snapshot_from_file(path)
        assert "began" not in str(caught.value).lower()
