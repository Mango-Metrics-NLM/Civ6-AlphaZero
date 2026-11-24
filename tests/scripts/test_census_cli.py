"""The census command line (GM-B11): exit codes, refusals, what may be called a result.

Spec: ``kb/specs/20261123-expressivity-census-preregistration.md``, "Input required for a
real answer" and "What a run on synthetic stand-in states does and does not show".
Exit 0 computed, 1 malformed input, 2 refused or usage, 3 below the registered minimum.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.bridge.search_translator import SearchTranslator
from tests.fixtures.census_states import (
    chunk_lines,
    end_turn_only,
    load_census,
    mixed_state,
    state_line,
    wire,
    write_json,
    write_log,
)
from tests.regression.test_evidence_contract_regression import block_problems

SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def mod() -> Any:
    return load_census()


@pytest.fixture(autouse=True)
def _pinned_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_GIT_SHA", SHA)


def _states(turns: Sequence[int], *, schema: int = 2, observes: bool | None = True) -> list[Any]:
    return [wire(mixed_state(t), schema=schema, observes=observes) for t in turns]


def _log(path: Path, turns: Sequence[int], **kw: Any) -> Path:
    return write_log(path, [state_line(s) for s in _states(turns, **kw)])


def _run(mod: Any, tmp: Path, *argv: str) -> tuple[int, Path]:
    out = tmp / "out"
    return mod.main([*argv, "--out", str(out)]), out / "census.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# -- the registered minimum, on a real log ---------------------------------------------------


@pytest.mark.parametrize(
    ("turns", "why"),
    [
        (range(1, 40), "39 states, 9 after turn 30"),
        (range(2, 41), "39 states, 10 after turn 30"),
        (range(0, 40), "40 states, 9 after turn 30 (turns 0..39)"),
    ],
)
def test_one_short_of_the_minimum_exits_3_and_writes_nothing(
    mod: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], turns: range, why: str
) -> None:
    log = _log(tmp_path / "Lua.log", turns)
    code, target = _run(mod, tmp_path, "--log", str(log), "--boards", "8")
    assert code == 3, why
    assert not target.exists() and not target.parent.exists()
    assert "BELOW THE REGISTERED MINIMUM" in capsys.readouterr().err


def test_allow_small_computes_but_says_it_is_not_a_result_and_not_decided(
    mod: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _log(tmp_path / "Lua.log", [1, 2, 3])
    code, target = _run(mod, tmp_path, "--log", str(log), "--allow-small", "--boards", "8")
    assert code == 0
    out = capsys.readouterr().out
    assert "NOT met" in out and "NOT a census result" in out
    assert "stop rule is NOT decided" in out
    data = _read(target)
    assert data["input"]["meets_registered_minimum"] is False
    assert data["input"]["is_census_result"] is False
    assert data["stop_rule"]["decided_by_this_input"] is False
    assert any("NOT a census result" in line for line in data["evidence"]["non_claims"])
    assert data["evidence"]["claims"] == ["the four computations ran on this input, nothing more"]


def test_a_states_json_is_never_a_census_result_even_with_forty_states(
    mod: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    states = write_json(tmp_path / "states.json", _states(range(1, 46)))
    code, target = _run(mod, tmp_path, "--states-json", str(states), "--boards", "8")
    assert code == 0  # it meets the minimum, so it is computed
    data = _read(target)
    assert data["input"]["meets_registered_minimum"] is True
    assert data["input"]["input_kind"] == "states_json"
    assert data["input"]["is_census_result"] is False
    assert data["stop_rule"]["decided_by_this_input"] is False
    out = capsys.readouterr().out
    assert "NOT a census result" in out and "not only a real game's Lua.log" in out


def test_a_log_mixed_with_a_states_json_is_not_a_census_result(mod: Any, tmp_path: Path) -> None:
    log = _log(tmp_path / "Lua.log", range(1, 41))
    states = write_json(tmp_path / "s.json", _states(range(1, 4)))
    code, target = _run(
        mod, tmp_path, "--log", str(log), "--states-json", str(states), "--boards", "8"
    )
    assert code == 0
    data = _read(target)
    assert data["input"]["input_kind"] == "mixed" and data["input"]["is_census_result"] is False


@pytest.mark.parametrize(("schema", "observes"), [(1, True), (2, None)])
def test_states_from_an_older_mod_are_not_a_census_result(
    mod: Any, tmp_path: Path, schema: int, observes: bool | None
) -> None:
    log = _log(tmp_path / "Lua.log", range(1, 41), schema=schema, observes=observes)
    code, target = _run(mod, tmp_path, "--log", str(log), "--boards", "8")
    assert code == 0
    info = _read(target)["input"]
    assert info["states_not_from_the_current_mod"] == 40 and info["is_census_result"] is False


def test_two_logs_are_two_sources_with_their_own_medians(mod: Any, tmp_path: Path) -> None:
    first = _log(tmp_path / "a.log", range(21, 41))
    second = _log(tmp_path / "b.log", range(21, 41))  # the same turns again: its own game
    code, target = _run(mod, tmp_path, "--log", str(first), "--log", str(second), "--boards", "8")
    assert code == 0
    data = _read(target)
    assert data["input"]["states_measured"] == 40 and len(data["sources"]) == 2
    per = data["aggregates"]["8"]["E1"]["per_source_median"]
    assert set(per) == {"0", "1"} and per["0"] == pytest.approx(8 / 12)


def test_what_was_recorded_is_in_the_output(mod: Any, tmp_path: Path) -> None:
    log = _log(tmp_path / "Lua.log", [1, 2])
    code, target = _run(
        mod, tmp_path, "--log", str(log), "--allow-small", "--boards", "8",
        "--mod-version", "0.9.1", "--played-by", "greedy policy",
    )  # fmt: skip
    assert code == 0
    recorded = _read(target)["recorded"]
    assert recorded["mod_version"] == "0.9.1" and recorded["played_by"] == "greedy policy"
    assert recorded["map_sizes"] == [[8, 8]]
    assert len(recorded["civilizations"]) == 1


# -- void states ------------------------------------------------------------------------------


def test_a_state_whose_translation_is_not_offered_is_void_and_never_counted(
    mod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SearchTranslator.root_mask

    def corrupted(self: Any, valid: Any, decode: Any) -> Any:
        mask, commands, reasons = original(self, valid, decode)
        commands[9999] = {"action_type": "unit_skip", "unit_id": 777}  # never offered
        return mask, commands, reasons

    monkeypatch.setattr(SearchTranslator, "root_mask", corrupted)
    states = write_json(tmp_path / "s.json", _states(range(1, 4)))
    code, target = _run(
        mod, tmp_path, "--states-json", str(states), "--allow-small", "--boards", "8"
    )
    assert code == 0
    data = _read(target)
    assert data["input"]["states_void"] == 3 and data["input"]["states_measured"] == 0
    assert data["aggregates"]["8"]["E1"]["all"]["median"] is None
    assert data["stop_rule"]["verdict"] not in ("stop", "no_stop")


def test_void_states_do_not_count_towards_the_registered_minimum(
    mod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = mod.measure_state

    def void_late(state: Any, board: int) -> Any:
        if state.turn_id > 30:
            return {"void": True, "void_reason": "planted"}
        return real(state, board)

    monkeypatch.setattr(mod, "measure_state", void_late)
    log = _log(tmp_path / "Lua.log", range(1, 41))
    code, target = _run(mod, tmp_path, "--log", str(log), "--boards", "8")
    assert code == 3 and not target.exists()  # 30 measured, 0 after turn 30


# -- refusals, malformed input, usage -----------------------------------------------------------


def test_an_existing_census_json_is_never_overwritten(mod: Any, tmp_path: Path) -> None:
    log = _log(tmp_path / "Lua.log", [1, 2])
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "census.json"
    existing.write_text('{"keep": "me"}', encoding="utf-8")
    code = mod.main(["--log", str(log), "--allow-small", "--boards", "8", "--out", str(out)])
    assert code == 2
    assert existing.read_text(encoding="utf-8") == '{"keep": "me"}'


def test_a_second_identical_run_is_refused_and_leaves_the_first_intact(
    mod: Any, tmp_path: Path
) -> None:
    log = _log(tmp_path / "Lua.log", [1, 2])
    argv = ["--log", str(log), "--allow-small", "--boards", "8"]
    code, target = _run(mod, tmp_path, *argv)
    first = target.read_bytes()
    assert code == 0 and _run(mod, tmp_path, *argv)[0] == 2
    assert target.read_bytes() == first


def test_no_source_is_a_refusal(mod: Any, tmp_path: Path) -> None:
    assert _run(mod, tmp_path)[0] == 2


def test_a_missing_input_file_is_a_refusal(mod: Any, tmp_path: Path) -> None:
    code, target = _run(mod, tmp_path, "--log", str(tmp_path / "nope.log"))
    assert code == 2 and not target.exists()


def test_a_board_size_that_is_not_positive_is_a_refusal(mod: Any, tmp_path: Path) -> None:
    states = write_json(tmp_path / "s.json", _states([1]))
    assert _run(mod, tmp_path, "--states-json", str(states), "--boards", "0", "8")[0] == 2


def test_a_missing_out_is_a_usage_error_exit_2(mod: Any, tmp_path: Path) -> None:
    states = write_json(tmp_path / "s.json", _states([1]))
    with pytest.raises(SystemExit) as raised:
        mod.main(["--states-json", str(states)])
    assert raised.value.code == 2


@pytest.mark.parametrize(
    "text",
    ["{not json", '{"a": 1}', "[]", "[1, 2, 3]", '[{"no_turn": true}]', ""],
)
def test_malformed_states_json_exits_1(mod: Any, tmp_path: Path, text: str) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(text, encoding="utf-8")
    code, target = _run(mod, tmp_path, "--states-json", str(bad), "--allow-small")
    assert code == 1 and not target.exists()


def test_a_log_with_no_complete_state_exits_1(mod: Any, tmp_path: Path) -> None:
    log = write_log(tmp_path / "Lua.log", ["AZ_HB:{}", "noise", "AZ_STATE:{broken"])
    code, target = _run(mod, tmp_path, "--log", str(log), "--allow-small")
    assert code == 1 and not target.exists()


def test_a_state_that_cannot_be_read_is_void_not_a_crash(mod: Any, tmp_path: Path) -> None:
    states = write_json(tmp_path / "s.json", [{"turn_id": 1}, {"turn_id": 2}])
    code, target = _run(mod, tmp_path, "--states-json", str(states), "--allow-small")
    assert code == 0
    assert _read(target)["input"]["states_void"] == 2


# -- the evidence block and the price --------------------------------------------------------


def test_the_evidence_block_has_no_games_and_passes_the_repository_contract(
    mod: Any, tmp_path: Path
) -> None:
    log = _log(tmp_path / "Lua.log", [1, 2])
    code, target = _run(mod, tmp_path, "--log", str(log), "--allow-small", "--boards", "8")
    assert code == 0
    block = _read(target)["evidence"]
    assert block["games_total"] == 0 and block["seed"] == 0
    assert block["search_epoch"] == "post-2026-09-04" and block["git_sha"] == SHA
    assert (
        "census_search_expressivity.py" in block["command"] and "--allow-small" in block["command"]
    )
    assert block_problems(block) == []


def test_an_unresolved_sha_is_refused_before_anything_is_read_or_written(
    mod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import alpha_zero.utils.checkpoint as checkpoint

    monkeypatch.delenv("CIV6AZ_GIT_SHA", raising=False)
    monkeypatch.setattr(checkpoint, "read_git_sha", lambda *a, **k: None)
    log = _log(tmp_path / "Lua.log", [1, 2])
    code, target = _run(mod, tmp_path, "--log", str(log), "--allow-small", "--boards", "8")
    assert code == 2
    assert not target.exists() and not target.parent.exists()
    assert "CIV6AZ_GIT_SHA" in capsys.readouterr().err


def test_a_resolved_sha_is_recorded_and_the_block_passes_the_contract(
    mod: Any, tmp_path: Path
) -> None:
    log = _log(tmp_path / "Lua.log", [1, 2])
    code, target = _run(mod, tmp_path, "--log", str(log), "--allow-small", "--boards", "8")
    assert code == 0
    block = _read(target)["evidence"]
    assert block["git_sha"] == SHA and block_problems(block) == []


def test_a_state_that_lost_a_piece_is_dropped_and_counted_through_the_cli(
    mod: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pieces = chunk_lines({"turn_id": 2, "pad": "x" * 900}, pieces=4)  # the rest still parses
    lines = [
        state_line(_states([1])[0]),
        *[pieces[0], pieces[2], pieces[3], pieces[4]],  # piece 2 of 4 left out
        state_line(_states([3])[0]),
    ]
    log = write_log(tmp_path / "Lua.log", lines)
    code, target = _run(mod, tmp_path, "--log", str(log), "--allow-small", "--boards", "8")
    assert code == 0
    data = _read(target)
    assert data["sources"][0]["states_dropped_missing_piece"] == 1
    assert data["input"]["states_dropped_missing_piece"] == 1
    assert [r["turn"] for r in data["states"]] == [1, 3]
    assert "1 dropped for a missing piece" in capsys.readouterr().out


def test_the_script_is_priced_at_zero_games(mod: Any) -> None:
    from alpha_zero.utils.game_budget import priced_runners, worst_case_games

    assert mod.RUNNER in priced_runners()
    assert worst_case_games(mod.RUNNER, {"states": 0}) == 0
    assert worst_case_games(mod.RUNNER, {"states": 1_000_000}) == 0


def test_the_end_turn_only_state_is_a_value_free_state_not_a_failure(
    mod: Any, tmp_path: Path
) -> None:
    states = write_json(tmp_path / "s.json", [wire(end_turn_only(t)) for t in (1, 2, 3)])
    code, target = _run(
        mod, tmp_path, "--states-json", str(states), "--allow-small", "--boards", "8"
    )
    assert code == 0
    e1 = _read(target)["aggregates"]["8"]["E1"]["all"]
    assert e1["median"] is None and e1["states_excluded"] == 3 and e1["states_with_value"] == 0
