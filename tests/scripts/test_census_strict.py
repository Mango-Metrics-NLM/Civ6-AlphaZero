"""The census voids a state the parser had to repair (review of #169).

``GameStateData.from_dict`` is tolerant on purpose: it drops an invalid tile, unit, city,
player, legal action, target or item and returns the rest. A damaged but JSON-valid state
must not be measured as if whole. Contract: ``parser_losses(raw, state)`` returns only the
kinds with a loss; ``_record`` voids such a state on every board; ``census.json`` counts
them under ``input.states_void_parser_dropped``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.bridge.protocol import GameStateData
from tests.fixtures.census_damage import DAMAGE, damaged, expected_loss, plain
from tests.fixtures.census_states import (
    ATTEST,
    load_census,
    read_json,
    run_cli,
    state_line,
    write_json,
    write_log,
)

SHA = "0123456789abcdef0123456789abcdef01234567"
KINDS = list(DAMAGE)


@pytest.fixture(scope="module")
def mod() -> Any:
    return load_census()


@pytest.fixture(autouse=True)
def _pinned_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_GIT_SHA", SHA)


def _losses(mod: Any, raw: dict[str, Any]) -> dict[str, int]:
    return mod.parser_losses(raw, GameStateData.from_dict(copy.deepcopy(raw)))


# -- the live parser is unchanged (pin) ------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_the_live_parser_still_drops_and_does_not_raise(kind: str) -> None:
    whole = GameStateData.from_dict(plain())
    state = GameStateData.from_dict(damaged(kind))
    counts = {
        "tile": (len(whole.map_data.tiles), len(state.map_data.tiles)),
        "unit": (len(whole.units), len(state.units)),
        "city": (len(whole.cities), len(state.cities)),
        "player": (len(whole.players), len(state.players)),
        "legal_action": (len(whole.legal_actions), len(state.legal_actions)),
    }
    if kind in counts:  # nothing is kept of the damaged entries: the count is unchanged
        assert counts[kind][0] == counts[kind][1]
    elif kind == "target":
        before = [a.targets for a in whole.legal_actions]
        assert [a.targets for a in state.legal_actions] == before
    else:
        assert [a.items for a in state.legal_actions] == [a.items for a in whole.legal_actions]


# -- parser_losses, directly -----------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_of_loss_is_reported_with_its_count(mod: Any, kind: str) -> None:
    assert _losses(mod, damaged(kind)) == expected_loss(kind)


def test_two_kinds_lost_in_one_state_are_both_reported(mod: Any) -> None:
    assert _losses(mod, damaged("unit", "item")) == expected_loss("unit", "item")


def test_all_seven_kinds_at_once(mod: Any) -> None:
    assert _losses(mod, damaged(*KINDS)) == expected_loss(*KINDS)


def test_a_legal_action_dropped_whole_is_not_also_counted_as_lost_targets(mod: Any) -> None:
    raw = damaged()
    # dropped whole for its unit_id; its own bad target and bad item are never parsed
    raw["legal_actions"].append(
        {"action_type": "move", "unit_id": "q", "targets": [[1, 2], [4], "bad"], "items": [None]}
    )
    raw["legal_actions"].append({"action_type": "move", "unit_id": 1, "targets": "abc"})
    assert _losses(mod, raw) == {"legal_actions": 2}


def test_the_measures_of_a_damaged_state_would_otherwise_be_valid(mod: Any) -> None:
    # the control for every void test below: the damage, not the rest, is what voids
    for kind in KINDS:
        state = GameStateData.from_dict(damaged(kind))
        assert not any(mod.measure_state(state, b).get("void") for b in (8, 16)), kind


# -- a valid state loses nothing, whatever the parser defaults or reshapes -------------------


def _valid_variants() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {"mixed": plain()}
    empty = plain()
    empty["units"], empty["cities"], empty["legal_actions"] = [], [], []
    out["empty_lists"] = empty
    absent = plain()
    for key in ("units", "cities", "legal_actions"):
        del absent[key]
    del absent["map"]["tiles"]
    out["absent_lists"] = absent
    summary = plain()  # no players list: the parser builds one player from the summary
    summary["players"] = []
    out["players_from_summary"] = summary
    no_players = plain()
    del no_players["players"]
    out["players_key_absent"] = no_players
    bare = plain()  # actions with no targets or items, in every spelling the parser accepts
    bare["legal_actions"] = [
        {"action_type": "end_turn"},
        {"action_type": "skip", "unit_id": 1, "targets": None, "items": None},
        {"action_type": "skip", "unit_id": 2, "targets": [], "items": []},
    ]
    out["no_targets_no_items"] = bare
    shapes = plain()  # legitimate spellings: dict and pair targets, one item, numeric strings
    shapes["legal_actions"] = [
        {"action_type": "move", "unit_id": 1, "targets": [{"x": 4, "y": 3}, [2, 3]]},
        {"action_type": "research_tech", "item": "TECH_POTTERY"},
        {"action_type": "produce", "city_id": 10, "items": ["12", "UNIT_WARRIOR", 5, True]},
        {"action_type": "research_tech", "items": "TECH_MINING"},
    ]
    out["legitimate_spellings"] = shapes
    dup = plain()  # duplicates are kept, not merged, by the parser
    dup["units"] += copy.deepcopy(dup["units"][:1])
    dup["map"]["tiles"] += [{"x": 1, "y": 1, "terrain": "TERRAIN_GRASS"}] * 2
    out["duplicates"] = dup
    return out


@pytest.mark.parametrize("name", list(_valid_variants()))
def test_a_valid_state_reports_no_loss(mod: Any, name: str) -> None:
    raw = _valid_variants()[name]
    assert _losses(mod, raw) == {}


def test_duplicates_really_are_kept_by_the_parser() -> None:
    raw = _valid_variants()["duplicates"]
    state = GameStateData.from_dict(copy.deepcopy(raw))
    assert len(state.units) == len(raw["units"])
    assert len(state.map_data.tiles) == len(raw["map"]["tiles"])


def test_a_summary_only_player_is_one_player_not_a_loss(mod: Any) -> None:
    raw = _valid_variants()["players_from_summary"]
    assert len(GameStateData.from_dict(copy.deepcopy(raw)).players) == 1
    assert mod.parser_losses(raw, GameStateData.from_dict(copy.deepcopy(raw))) == {}


def test_the_losses_are_only_the_kinds_with_a_loss(mod: Any) -> None:
    losses = _losses(mod, damaged("city"))
    assert set(losses) == {"cities"} and all(v > 0 for v in losses.values())


# -- _record: void on every board, with the reason named ------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_a_repaired_state_is_void_on_every_board(mod: Any, kind: str) -> None:
    rec = mod._record(0, damaged(kind), [8, 16])
    assert set(rec["boards"]) == {"8", "16"}
    for board in ("8", "16"):
        cell = rec["boards"][board]
        assert cell["void"] is True
        assert "dropped" in cell["void_reason"]


@pytest.mark.parametrize(
    ("kind", "named"),
    [
        ("tile", "tiles: 2"),
        ("unit", "units: 1"),
        ("city", "cities: 1"),
        ("player", "players: 1"),
        ("legal_action", "legal_actions: 1"),
        ("target", "legal_action_targets: 2"),
        ("item", "legal_action_items: 2"),
    ],
)
def test_the_reason_names_what_was_dropped_and_how_many(mod: Any, kind: str, named: str) -> None:
    rec = mod._record(0, damaged(kind), [8])
    assert named in rec["boards"]["8"]["void_reason"]


def test_two_kinds_lost_are_both_named_in_the_reason(mod: Any) -> None:
    reason = mod._record(0, damaged("unit", "city"), [8])["boards"]["8"]["void_reason"]
    assert "units: 1" in reason and "cities: 1" in reason


def test_a_whole_state_is_not_void(mod: Any) -> None:
    rec = mod._record(0, plain(), [8, 16])
    assert not any(c["void"] for c in rec["boards"].values())


# -- through the command line ----------------------------------------------------------------


def _log(tmp: Path, states: list[dict[str, Any]]) -> Path:
    return write_log(tmp / "Lua.log", [state_line(s) for s in states])


def _turns(turns: range, bad: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [bad.get(t, plain(t)) for t in turns]


@pytest.mark.parametrize("kind", KINDS)
def test_one_repaired_state_is_counted_and_voided_through_the_cli(
    mod: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    states = write_json(tmp_path / "s.json", _turns(range(1, 4), {2: damaged(kind, turn=2)}))
    code, target = run_cli(mod, tmp_path, "--states-json", str(states), "--allow-small")
    assert code == 0
    info = read_json(target)["input"]
    assert info["states_void_parser_dropped"] == 1
    assert info["states_void"] == 1 and info["states_measured"] == 2
    shown = [ln for ln in capsys.readouterr().out.splitlines() if "repaired by the parser" in ln]
    assert shown and "1" in shown[0]


def test_a_whole_run_reports_zero_repaired_states(mod: Any, tmp_path: Path) -> None:
    states = write_json(tmp_path / "s.json", _turns(range(1, 4), {}))
    code, target = run_cli(mod, tmp_path, "--states-json", str(states), "--allow-small")
    assert code == 0
    info = read_json(target)["input"]
    assert info["states_void_parser_dropped"] == 0 and info["states_void"] == 0


def test_the_census_json_names_the_loss_in_the_state_record(mod: Any, tmp_path: Path) -> None:
    states = write_json(tmp_path / "s.json", _turns(range(1, 3), {1: damaged("unit", turn=1)}))
    _, target = run_cli(mod, tmp_path, "--states-json", str(states), "--allow-small")
    rec = read_json(target)["states"][0]
    assert all(c["void"] and "units: 1" in c["void_reason"] for c in rec["boards"].values())


def test_a_repaired_state_does_not_count_towards_the_minimum(mod: Any, tmp_path: Path) -> None:
    # 40 states, 10 after turn 30; the one repaired state is the 40th: 39 measured
    log = _log(tmp_path, _turns(range(1, 41), {40: damaged("unit", turn=40)}))
    code, target = run_cli(mod, tmp_path, "--log", str(log), "--boards", "8", *ATTEST)
    assert code == 3 and not target.exists()


def test_a_repaired_late_state_is_not_replaced_by_a_later_state_of_its_turn(
    mod: Any, tmp_path: Path
) -> None:
    # turns 21..40 give exactly 10 states after turn 30; a whole copy of turn 31 follows the
    # repaired one in the log and must not stand in for it
    first = _turns(range(1, 41), {31: damaged("city", turn=31)})
    log = _log(tmp_path, [*first, plain(31)])
    code, target = run_cli(mod, tmp_path, "--log", str(log), "--boards", "8", *ATTEST)
    assert code == 3 and not target.exists()


def test_forty_whole_states_are_still_a_census_result(mod: Any, tmp_path: Path) -> None:
    log = _log(tmp_path, _turns(range(1, 41), {}))
    code, target = run_cli(mod, tmp_path, "--log", str(log), *ATTEST)  # default boards 8 and 16
    assert code == 0
    info = read_json(target)["input"]
    assert info["is_census_result"] is True and info["states_void_parser_dropped"] == 0


def test_a_repaired_state_leaves_every_aggregate(mod: Any, tmp_path: Path) -> None:
    states = write_json(tmp_path / "s.json", _turns(range(1, 4), {2: damaged("unit", turn=2)}))
    _, target = run_cli(mod, tmp_path, "--states-json", str(states), "--allow-small")
    agg = read_json(target)["aggregates"]["8"]["E2"]["all"]
    assert agg["states_with_value"] + agg["states_excluded"] == 2  # three sampled, one void
