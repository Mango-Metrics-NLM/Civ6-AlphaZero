"""GM-B4: ``alpha_zero.bridge.gameinfo_capture`` and the snapshot path end to end.

The module is imported inside the ``gc`` fixture, so while it does not exist only the
tests that use it fail; nothing here is a collection error. Only ``TestEndToEnd`` needs
a Lua runtime. Every file is built under ``tmp_path``; no game data is tracked.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from alpha_zero.bridge.action_converter import ActionConverter
from alpha_zero.bridge.protocol import GameStateData
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.bridge.tech_registry import load_registry
from alpha_zero.rules.gameinfo import GameInfoError, convert, import_gameinfo
from tests.fixtures import gameinfo_snapshot as g
from tests.fixtures import lua_harness as h
from tests.fixtures.research_wiring import CIVIC_RANGE, TECH_RANGE, encode, research_bits

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_ENV = "CIV6AZ_DATA_ROOT"

DOCUMENT: dict[str, Any] = {
    "Technologies": [
        {"TechnologyType": "TECH_A", "Cost": 10, "EraType": "ERA_ANCIENT"},
        {"TechnologyType": "TECH_B", "Cost": 20, "EraType": "ERA_ANCIENT"},
    ],
    "TechnologyPrereqs": [{"Technology": "TECH_B", "PrereqTech": "TECH_A"}],
    "Civics": [{"CivicType": "CIVIC_X", "Cost": 10, "EraType": "ERA_ANCIENT"}],
    "CivicPrereqs": [],
    "Buildings": [],
    "Units": [],
}


@pytest.fixture
def gc() -> ModuleType:
    return importlib.import_module("alpha_zero.bridge.gameinfo_capture")


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data root outside the repository; also the working directory, to catch strays."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv(DATA_ROOT_ENV, str(data))
    monkeypatch.chdir(tmp_path)
    return data


def _files(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())


def _single(document: dict[str, Any], prefix: str = "") -> str:
    return f"{prefix}AZ_GAMEINFO:{json.dumps(document, separators=(',', ':'))}\n"


def _chunked(document: dict[str, Any], size: int = 40, prefix: str = "") -> list[str]:
    text = json.dumps(document, separators=(",", ":"))
    pieces = [text[i : i + size] for i in range(0, len(text), size)]
    lines = [f"{prefix}AZ_GAMEINFO_CHUNK:{i}/{len(pieces)}:{p}\n" for i, p in enumerate(pieces, 1)]
    return [*lines, g.end_line(len(pieces), g.adler32(text), prefix)]


def _run(gc: ModuleType, *argv: str) -> int:
    try:
        return int(gc.main(list(argv)))
    except SystemExit as exit_:  # argparse reports a usage error this way
        return int(exit_.code or 0)


class TestSnapshotFromLog:
    def test_single_line(self, gc: ModuleType) -> None:
        assert gc.snapshot_from_log(_single(DOCUMENT)) == DOCUMENT

    def test_game_context_prefix_is_accepted(self, gc: ModuleType) -> None:
        assert gc.snapshot_from_log(_single(DOCUMENT, "AlphaZeroMain: ")) == DOCUMENT

    def test_chunked_with_and_without_prefix(self, gc: ModuleType) -> None:
        for prefix in ("", "AlphaZeroMain: "):
            lines = _chunked(DOCUMENT, prefix=prefix)
            assert len(lines) > 3, "premise: a real chunk set"
            assert gc.snapshot_from_log("".join(lines)) == DOCUMENT

    def test_the_last_complete_message_wins(self, gc: ModuleType) -> None:
        older = {**DOCUMENT, "Technologies": DOCUMENT["Technologies"][:1], "TechnologyPrereqs": []}
        text = _single(older) + "".join(_chunked(DOCUMENT)) + _single(older)
        assert gc.snapshot_from_log(text) == older
        text = _single(older) + "".join(_chunked(DOCUMENT))
        assert gc.snapshot_from_log(text) == DOCUMENT
        text = "".join(_chunked(older)) + _single(DOCUMENT)
        assert gc.snapshot_from_log(text) == DOCUMENT

    def test_other_log_lines_between_are_ignored(self, gc: ModuleType) -> None:
        noise = 'AlphaZeroMain: AZ_STATE:{"turn_id": 1}\n[AlphaZero] note\nAZ_HB:{"a": 1}\n\n'
        text = noise + _single(DOCUMENT, "AlphaZeroMain: ") + noise
        assert gc.snapshot_from_log(text) == DOCUMENT

    def test_windows_line_endings(self, gc: ModuleType) -> None:
        single = _single(DOCUMENT).replace("\n", "\r\n")
        assert gc.snapshot_from_log(single) == DOCUMENT
        chunked = "".join(_chunked(DOCUMENT)).replace("\n", "\r\n")
        assert gc.snapshot_from_log(chunked) == DOCUMENT

    @pytest.mark.parametrize("damage", ["no-end", "missing-piece", "end-too-large"])
    def test_an_incomplete_trailing_chunk_set_yields_to_an_earlier_message(
        self, gc: ModuleType, damage: str
    ) -> None:
        older = {**DOCUMENT, "Civics": []}
        lines = _chunked(DOCUMENT)
        if damage == "no-end":
            lines = lines[:-1]
        elif damage == "missing-piece":
            del lines[1]
        else:
            lines[-1] = g.end_line(len(lines), g.adler32(g.compact_text(DOCUMENT)))
        for earlier in (_single(older), "".join(_chunked(older))):
            assert gc.snapshot_from_log(earlier + "".join(lines)) == older

    def test_an_interrupted_chunk_set_followed_by_a_fresh_complete_one(
        self, gc: ModuleType
    ) -> None:
        interrupted = "".join(_chunked({**DOCUMENT, "Civics": []})[:2])
        assert gc.snapshot_from_log(interrupted + "".join(_chunked(DOCUMENT))) == DOCUMENT

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "nothing here\nAZ_STATE:{}\n",
            "".join(_chunked(DOCUMENT)[:-1]),
            "".join(_chunked(DOCUMENT)[1:]),
            "AZ_GAMEINFO:{not json\n",
            "AZ_GAMEINFO:\n",
        ],
        ids=["empty", "no-message", "no-end", "no-first-piece", "bad-json", "empty-payload"],
    )
    def test_no_complete_message_or_bad_json_is_an_error(self, gc: ModuleType, text: str) -> None:
        with pytest.raises(gc.GameInfoCaptureError):
            gc.snapshot_from_log(text)

    def test_a_later_bad_message_is_an_error_not_a_fallback(self, gc: ModuleType) -> None:
        with pytest.raises(gc.GameInfoCaptureError):
            gc.snapshot_from_log(_single(DOCUMENT) + "AZ_GAMEINFO:{broken\n")

    def test_the_error_is_a_value_error(self, gc: ModuleType) -> None:
        assert issubclass(gc.GameInfoCaptureError, ValueError)


class TestSnapshotFromFile:
    def test_reads_the_document(self, gc: ModuleType, tmp_path: Path) -> None:
        path = tmp_path / "gameinfo.json"
        path.write_text(json.dumps(DOCUMENT), encoding="utf-8")
        assert gc.snapshot_from_file(path) == DOCUMENT
        assert gc.snapshot_from_file(str(path)) == DOCUMENT

    @pytest.mark.parametrize("content", [None, "{not json", "[1, 2]", '"text"', ""])
    def test_a_bad_file_is_an_error_naming_the_path(
        self, gc: ModuleType, tmp_path: Path, content: str | None
    ) -> None:
        path = tmp_path / "gameinfo.json"
        if content is not None:
            path.write_text(content, encoding="utf-8")
        with pytest.raises(gc.GameInfoCaptureError) as caught:
            gc.snapshot_from_file(path)
        assert str(path) in str(caught.value)


class TestCheckSnapshot:
    def test_returns_the_importers_summary(self, gc: ModuleType) -> None:
        summary = gc.check_snapshot(DOCUMENT)
        expected = convert(DOCUMENT)
        assert summary == expected
        assert (summary.techs, summary.civics_not_imported) == (2, 1)

    @pytest.mark.parametrize(
        "document",
        [
            {**DOCUMENT, "Wonders": []},
            {**DOCUMENT, "Technologies": [{"TechnologyType": "TECH WITH SPACE", "Cost": 1}]},
            {**DOCUMENT, "TechnologyPrereqs": [{"Technology": "TECH_B", "PrereqTech": "TECH_Z"}]},
            [],
        ],
        ids=["unknown-table", "non-identifier", "unknown-prereq", "list"],
    )
    def test_an_unimportable_document_raises_with_the_importers_message(
        self, gc: ModuleType, document: Any
    ) -> None:
        try:
            convert(document)
        except Exception as exc:
            importer_message = str(exc)
        else:
            pytest.fail("premise: the importer accepts this document")
        with pytest.raises(gc.GameInfoCaptureError) as caught:
            gc.check_snapshot(document)
        assert importer_message in str(caught.value)
        assert not isinstance(caught.value, GameInfoError)


class TestWriteSnapshot:
    def test_writes_under_the_data_root_and_round_trips(self, gc: ModuleType, root: Path) -> None:
        path = gc.write_snapshot(DOCUMENT, name="probe")
        assert path == root / "gameinfo" / "probe.json"
        assert json.loads(path.read_text(encoding="utf-8")) == DOCUMENT
        assert _files(root) == ["gameinfo/probe.json"]

    def test_the_default_name_is_a_utc_timestamp(self, gc: ModuleType, root: Path) -> None:
        before = datetime.now(UTC).replace(microsecond=0)
        path = gc.write_snapshot(DOCUMENT)
        after = datetime.now(UTC)
        match = re.fullmatch(r"gameinfo-(\d{8}T\d{6}Z)\.json", path.name)
        assert match, path.name
        stamp = datetime.strptime(match[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        assert before <= stamp <= after
        assert path.parent == root / "gameinfo"

    @pytest.mark.parametrize("name", ["a.b-c_d", "A1", "x", "gameinfo-2026"])
    def test_a_plain_stem_is_accepted(self, gc: ModuleType, root: Path, name: str) -> None:
        assert gc.write_snapshot(DOCUMENT, name=name).name == f"{name}.json"

    @pytest.mark.parametrize(
        "name",
        ["", "../x", "a/b", "a\\b", "a b", "x:y", "/abs", "a\nb", "n;m", "ü"],
    )
    def test_a_bad_name_is_refused_and_nothing_is_written(
        self, gc: ModuleType, root: Path, name: str
    ) -> None:
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(DOCUMENT, name=name)
        assert _files(root.parent) == []

    def test_an_unset_data_root_is_refused(
        self, gc: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(DATA_ROOT_ENV)
        with pytest.raises(gc.GameInfoCaptureError) as caught:
            gc.write_snapshot(DOCUMENT, name="probe")
        assert DATA_ROOT_ENV in str(caught.value)
        assert _files(root.parent) == []

    def test_a_data_root_inside_the_repository_is_refused(
        self, gc: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inside = REPO_ROOT / "gameinfo_capture_test_probe_dir"
        assert not inside.exists()
        monkeypatch.setenv(DATA_ROOT_ENV, str(inside))
        try:
            with pytest.raises(gc.GameInfoCaptureError):
                gc.write_snapshot(DOCUMENT, name="probe")
            assert not inside.exists(), "something was created inside the checkout"
        finally:
            if inside.exists():
                for leftover in sorted(inside.rglob("*"), reverse=True):
                    leftover.unlink() if leftover.is_file() else leftover.rmdir()
                inside.rmdir()

    def test_a_data_root_inside_any_git_checkout_is_refused(
        self, gc: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkout = tmp_path / "somewhere" / "work"
        (checkout / ".git").mkdir(parents=True)
        monkeypatch.setenv(DATA_ROOT_ENV, str(checkout / "data"))
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(DOCUMENT, name="probe")
        assert _files(checkout) == []

    @pytest.mark.parametrize("where", ["root", "gameinfo-subdirectory"])
    def test_a_destination_that_resolves_into_a_checkout_is_refused(
        self, gc: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
    ) -> None:
        checkout = tmp_path / "work"
        (checkout / ".git").mkdir(parents=True)
        data = tmp_path / "data"
        data.mkdir()
        if where == "root":
            (tmp_path / "link").symlink_to(checkout, target_is_directory=True)
            monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "link"))
        else:
            (data / "gameinfo").symlink_to(checkout, target_is_directory=True)
            monkeypatch.setenv(DATA_ROOT_ENV, str(data))
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(DOCUMENT, name="probe")
        assert _files(checkout) == []

    def test_an_existing_destination_is_never_overwritten(self, gc: ModuleType, root: Path) -> None:
        first = gc.write_snapshot(DOCUMENT, name="probe")
        before = first.read_bytes()
        other = {**DOCUMENT, "Civics": []}
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(other, name="probe")
        assert first.read_bytes() == before
        assert _files(root) == ["gameinfo/probe.json"]

    def test_an_unimportable_snapshot_is_refused_before_anything_is_written(
        self, gc: ModuleType, root: Path
    ) -> None:
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot({"Wonders": []}, name="probe")
        assert _files(root) == []

    def test_a_failed_final_rename_leaves_no_partial_file(
        self, gc: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("simulated publish failure")

        # Both ways of giving the file its name fail: a hard link and a rename.
        monkeypatch.setattr(os, "link", refuse)
        monkeypatch.setattr(os, "replace", refuse)
        monkeypatch.setattr(os, "rename", refuse)
        with pytest.raises(gc.GameInfoCaptureError):
            gc.write_snapshot(DOCUMENT, name="probe")
        assert _files(root) == []


class TestCommandLine:
    def _source(self, tmp_path: Path, *, chunked: bool = False) -> Path:
        path = tmp_path / "Lua.log"
        text = (
            "".join(_chunked(DOCUMENT, prefix="AlphaZeroMain: ")) if chunked else _single(DOCUMENT)
        )
        path.write_text("AlphaZeroMain: noise\n" + text, encoding="utf-8")
        return path

    @pytest.mark.parametrize("chunked", [False, True])
    def test_from_log_writes_and_reports(
        self,
        gc: ModuleType,
        root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        chunked: bool,
    ) -> None:
        assert (
            _run(gc, "--from-log", str(self._source(tmp_path, chunked=chunked)), "--name", "n1")
            == 0
        )
        out = capsys.readouterr().out
        written = root / "gameinfo" / "n1.json"
        assert json.loads(written.read_text(encoding="utf-8")) == DOCUMENT
        assert str(written) in out
        for table, rows in DOCUMENT.items():
            assert any(
                table in line and re.search(rf"\b{len(rows)}\b", line) for line in out.splitlines()
            ), table
        assert f"python -m alpha_zero.rules import-gameinfo {written}" in out
        assert f"CIV6AZ_BRIDGE_GAMEINFO={written}" in out

    def test_from_file_with_the_default_name(
        self, gc: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "gameinfo.json"
        source.write_text(json.dumps(DOCUMENT), encoding="utf-8")
        assert _run(gc, "--from-file", str(source)) == 0
        (written,) = (root / "gameinfo").glob("gameinfo-*.json")
        assert str(written) in capsys.readouterr().out

    def _fails(self, gc: ModuleType, capsys: pytest.CaptureFixture[str], *argv: str) -> str:
        assert _run(gc, *argv) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Traceback" not in captured.err
        lines = captured.err.strip().splitlines()
        assert len(lines) == 1 and lines[0]
        return lines[0]

    def test_a_missing_source_is_exit_one_with_one_line(
        self, gc: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "nope.json"
        assert str(missing) in self._fails(gc, capsys, "--from-file", str(missing))
        assert _files(root) == []

    def test_a_log_without_a_snapshot_is_exit_one(
        self, gc: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log = tmp_path / "Lua.log"
        log.write_text("AlphaZeroMain: nothing\n", encoding="utf-8")
        self._fails(gc, capsys, "--from-log", str(log))
        assert _files(root) == []

    def test_an_unimportable_snapshot_is_exit_one_and_names_the_problem(
        self, gc: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({**DOCUMENT, "Wonders": []}), encoding="utf-8")
        assert "Wonders" in self._fails(gc, capsys, "--from-file", str(bad))
        assert _files(root) == []

    def test_an_existing_destination_is_exit_one(
        self, gc: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "gameinfo.json"
        source.write_text(json.dumps(DOCUMENT), encoding="utf-8")
        assert _run(gc, "--from-file", str(source), "--name", "twice") == 0
        capsys.readouterr()
        self._fails(gc, capsys, "--from-file", str(source), "--name", "twice")

    def test_an_unset_data_root_is_exit_one(
        self,
        gc: ModuleType,
        root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "gameinfo.json"
        source.write_text(json.dumps(DOCUMENT), encoding="utf-8")
        monkeypatch.delenv(DATA_ROOT_ENV)
        assert DATA_ROOT_ENV in self._fails(gc, capsys, "--from-file", str(source))

    def test_a_bad_name_is_exit_one(
        self, gc: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "gameinfo.json"
        source.write_text(json.dumps(DOCUMENT), encoding="utf-8")
        self._fails(gc, capsys, "--from-file", str(source), "--name", "../escape")
        assert _files(root.parent) == ["gameinfo.json"]

    @pytest.mark.parametrize(
        "argv",
        [[], ["--name", "x"], ["--from-log", "a", "--from-file", "b"], ["--bogus"]],
        ids=["neither", "name-only", "both", "unknown-option"],
    )
    def test_a_usage_error_is_exit_two(
        self, gc: ModuleType, root: Path, argv: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(gc, *argv) == 2
        capsys.readouterr()
        assert _files(root) == []

    def test_the_module_runs_as_a_script(self, root: Path, tmp_path: Path) -> None:
        source = tmp_path / "gameinfo.json"
        source.write_text(json.dumps(DOCUMENT), encoding="utf-8")
        env = {**os.environ, DATA_ROOT_ENV: str(root), "PYTHONPATH": str(REPO_ROOT)}
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "alpha_zero.bridge.gameinfo_capture",
                "--from-file",
                str(source),
                "--name",
                "viaso",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        assert str(root / "gameinfo" / "viaso.json") in run.stdout


class TestImportsStayLight:
    def test_source_imports_neither_torch_nor_numpy(self) -> None:
        source = REPO_ROOT / "alpha_zero" / "bridge" / "gameinfo_capture.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        assert "alpha_zero" in roots, "no alpha_zero import read; vacuous"
        assert not roots & {"torch", "numpy"}

    def test_importing_it_loads_neither_torch_nor_numpy(self) -> None:
        code = (
            "import sys\n"
            "import alpha_zero.bridge\n"
            "before = set(sys.modules)\n"
            "import alpha_zero.bridge.gameinfo_capture\n"
            "new = set(sys.modules) - before\n"
            "bad = sorted(m for m in new if m.split('.')[0] in ('torch', 'numpy'))\n"
            "print('LOADED' if 'alpha_zero.bridge.gameinfo_capture' in sys.modules else 'NOT', bad)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "LOADED []"


# ---------------------------------------------------------------------------
# The path after the snapshot, on a hand-written copy of what the stub must export.
# These pass today: they pin what the end-to-end tests below assume downstream.
# ---------------------------------------------------------------------------
TECHS = ("tech_pottery", "tech_mining", "tech_writing", "tech_bronze_working")
CIVICS = ("civic_code_of_laws", "civic_craftsmanship", "civic_foreign_trade", "civic_early_empire")


def _registry(tmp_path: Path, document: dict[str, Any]) -> Any:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(document), encoding="utf-8")
    ruleset, summary = import_gameinfo(snapshot, str(tmp_path / "rules" / "ruleset.yaml"))
    assert (summary.techs, summary.units, summary.buildings) == (4, 3, 2)
    return load_registry(ruleset, gameinfo=snapshot)


class TestDownstreamPremise:
    def test_the_stub_document_gives_slots_in_the_stubs_order(self, tmp_path: Path) -> None:
        registry = _registry(tmp_path, g.STUB_DOCUMENT)
        assert registry.tech_ids == TECHS
        assert registry.civic_ids == CIVICS


class TestEndToEnd:
    @pytest.mark.parametrize("prefix", ["", "AlphaZeroMain: "])
    def test_the_printed_snapshot_becomes_a_registry(
        self, gc: ModuleType, root: Path, tmp_path: Path, prefix: str
    ) -> None:
        runtime = g.load_printing()
        text = g.log_text(g.printed(runtime), prefix)
        document = gc.snapshot_from_log(text)
        assert document == g.STUB_DOCUMENT
        summary = gc.check_snapshot(document)
        assert (summary.techs, summary.units, summary.buildings) == (4, 3, 2)
        assert summary.civics_not_imported == 4
        path = gc.write_snapshot(document, name="stubgame")
        assert path == root / "gameinfo" / "stubgame.json"
        ruleset, _ = import_gameinfo(path, str(tmp_path / "rules" / "ruleset.yaml"))
        registry = load_registry(ruleset, gameinfo=path)
        assert registry.tech_ids == TECHS
        assert registry.civic_ids == CIVICS

    def test_the_command_line_path_from_a_log_to_a_registry(
        self, gc: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        runtime = g.load_printing()
        log = tmp_path / "Lua.log"
        log.write_text(g.log_text(g.printed(runtime), "AlphaZeroMain: "), encoding="utf-8")
        assert _run(gc, "--from-log", str(log), "--name", "cli") == 0
        capsys.readouterr()
        path = root / "gameinfo" / "cli.json"
        ruleset, _ = import_gameinfo(path, str(tmp_path / "rules" / "ruleset.yaml"))
        assert load_registry(ruleset, gameinfo=path).tech_ids == TECHS

    def test_a_chunked_snapshot_reassembles_to_the_same_document(self, gc: ModuleType) -> None:
        runtime = g.load_printing(g.extra_units(200))
        lines = g.printed(runtime)
        assert any(line.startswith("AZ_GAMEINFO_CHUNK:") for line in lines)
        assert not any(line.startswith("AZ_GAMEINFO:") for line in lines)
        document = gc.snapshot_from_log(g.log_text(lines, "AlphaZeroMain: "))
        exported, failed, _ = g.lua_export(runtime)
        assert failed == [] and document == exported
        assert document["Units"] == g.STUB_DOCUMENT["Units"] + g.extra_unit_documents(200)
        summary = gc.check_snapshot(document)
        assert summary.units == 203

    def test_the_registry_drives_the_encoder_and_the_mask(
        self, gc: ModuleType, tmp_path: Path
    ) -> None:
        runtime = g.load_printing()
        document = gc.snapshot_from_log(g.log_text(g.printed(runtime)))
        registry = _registry(tmp_path, document)
        parsed = GameStateData.from_dict(h.export_state(runtime))
        state = StateConverter(research_registry=registry)
        vector = encode(state, parsed)
        # TECH_POTTERY and CIVIC_CODE_OF_LAWS are researched; both sit at slot 0.
        assert vector[70 + registry.tech_slot("TECH_POTTERY")] == 1.0
        assert vector[180 + registry.civic_slot("CIVIC_CODE_OF_LAWS")] == 1.0
        assert float(vector[70:170].sum()) == 1.0 and float(vector[180:280].sum()) == 1.0
        assert state.research_stats["unmapped_techs"] == 0
        assert state.research_stats["unmapped_civics"] == 0
        actions = ActionConverter(research_registry=registry)
        converted = state.convert(parsed)
        mask = actions.generate_action_mask(parsed.legal_actions, converted)
        assert research_bits(actions, mask, TECH_RANGE) == {registry.tech_slot("TECH_WRITING")}
        assert research_bits(actions, mask, CIVIC_RANGE) == {
            registry.civic_slot("CIVIC_FOREIGN_TRADE")
        }
        assert (registry.tech_slot("TECH_WRITING"), registry.civic_slot("CIVIC_FOREIGN_TRADE")) == (
            2,
            2,
        )
