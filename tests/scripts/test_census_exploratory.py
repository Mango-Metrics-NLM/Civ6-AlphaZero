"""The exploratory addendum and "void on one board is void on all" (GM-B11, notes 4).

Addendum: per census state at board 8, can the merged rule's losing ending fire at all (the
player has no city and no own unit is left out of the stand-in)? Counted in three turn bands;
there is no verdict. Not part of the registered census.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.census_states import (
    load_census,
    read_json,
    run_cli,
    settler_state,
    wire,
    write_json,
)


@pytest.fixture(scope="module")
def mod() -> Any:
    return load_census()


@pytest.fixture(autouse=True)
def _pinned_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_GIT_SHA", "0123456789abcdef0123456789abcdef01234567")


@pytest.mark.parametrize(
    ("kwargs", "fires"),
    [
        ({}, True),  # one settler, no city, nothing left out
        ({"builder": True}, False),  # the stand-in has no builder: one unit is left out
        ({"own_city": True}, False),  # the player has a city
        ({"enemy_city": True}, True),  # only the other player's city: still no own city
        ({"builder": True, "own_city": True}, False),
    ],
)
def test_loss_can_fire_on_hand_built_states(mod: Any, kwargs: dict[str, bool], fires: bool) -> None:
    result = mod.measure_state(settler_state(**kwargs), 8)
    assert result["void"] is False and result["loss_can_fire"] is fires


def test_left_out_count_of_the_builder_state_is_what_stops_it_firing(mod: Any) -> None:
    assert mod.measure_state(settler_state(builder=True), 8)["E4"]["numerator"] == 1
    assert mod.measure_state(settler_state(), 8)["E4"]["numerator"] == 0


def _rec(turn: int, fires: bool | None) -> dict[str, Any]:
    cell: dict[str, Any] = {"void": True} if fires is None else {"void": False}
    if fires is not None:
        cell["loss_can_fire"] = fires
    return {"turn": turn, "boards": {"8": cell}}


def test_bands_and_their_edges_by_hand(mod: Any) -> None:
    records = [
        _rec(0, True),
        _rec(5, False),
        _rec(10, True),  # last turn of the first band
        _rec(11, True),  # first turn of the second
        _rec(12, None),  # void: left out of its band
        _rec(30, False),  # last turn of the second
        _rec(31, True),  # first turn after turn 30
        _rec(60, True),
    ]
    bands = mod.loss_can_fire_by_band(records)
    # up to turn 10: turns 0, 5, 10 -> 2 of 3; turns 11-30: 11, 30 -> 1 of 2 (12 is void);
    # after turn 30: 31, 60 -> 2 of 2.
    got = {k: (v["numerator"], v["denominator"]) for k, v in bands.items()}
    assert got == {"up_to_turn_10": (2, 3), "turn_11_to_30": (1, 2), "after_turn_30": (2, 2)}
    assert bands["up_to_turn_10"]["value"] == pytest.approx(2 / 3)


def test_a_band_with_no_state_has_no_share(mod: Any) -> None:
    bands = mod.loss_can_fire_by_band([_rec(3, True)])
    assert bands["turn_11_to_30"]["value"] is None and bands["after_turn_30"]["denominator"] == 0


def test_through_the_cli_one_summary_line_and_no_verdict(
    mod: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    states = [
        wire(settler_state(5)),  # fires, band 1
        wire(settler_state(11, builder=True)),  # a unit left out, band 2
        wire(settler_state(30, own_city=True)),  # has a city, band 2
        wire(settler_state(31)),  # fires, band 3
    ]
    path = write_json(tmp_path / "s.json", states)
    code, target = run_cli(mod, tmp_path, "--states-json", str(path), "--allow-small")
    assert code == 0
    data = read_json(target)
    exploratory = data["exploratory"]
    assert exploratory["not_part_of_the_registered_census"] is True
    assert exploratory["board"] == 8 and "verdict" not in exploratory
    got = {k: (v["numerator"], v["denominator"]) for k, v in exploratory["loss_can_fire"].items()}
    assert got == {"up_to_turn_10": (1, 1), "turn_11_to_30": (0, 2), "after_turn_30": (1, 1)}
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("Exploratory")]
    assert len(lines) == 1
    line = lines[0]
    assert "1/1 up to turn 10" in line and "0/2 turn 11 to 30" in line
    assert "1/1 after turn 30" in line
    assert "no verdict" in line and "no_stop" not in line and "undetermined" not in line


# -- a state void on any board is void on all -----------------------------------------------


def _states_json(tmp: Path) -> Path:
    return write_json(tmp / "s.json", [wire(settler_state(t)) for t in (1, 2, 3)])


@pytest.mark.parametrize("how", ["returns_void", "raises"])
def test_a_state_void_on_board_16_only_is_void_on_board_8_too(
    mod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    real = mod.measure_state

    def measure(state: Any, board: int) -> Any:
        if board == 16 and state.turn_id == 2:
            if how == "raises":
                raise RuntimeError("planted")
            return {"void": True, "void_reason": "planted"}
        return real(state, board)

    monkeypatch.setattr(mod, "measure_state", measure)
    code, target = run_cli(
        mod, tmp_path, "--states-json", str(_states_json(tmp_path)), "--allow-small"
    )
    assert code == 0
    data = read_json(target)
    by_turn = {r["turn"]: r for r in data["states"]}
    assert by_turn[2]["boards"]["8"]["void"] is True and by_turn[2]["boards"]["16"]["void"] is True
    assert "planted" in by_turn[2]["boards"]["8"]["void_reason"]
    assert by_turn[1]["boards"]["8"]["void"] is False
    # Absent from the board-8 aggregates and counted void once, not once per board.
    for board in ("8", "16"):
        assert data["aggregates"][board]["states"] == 2
        assert data["aggregates"][board]["states_void"] == 1
        assert data["aggregates"][board]["E4"]["all"]["states_with_value"] == 2
    assert data["input"]["states_void"] == 1 and data["input"]["states_measured"] == 2
    assert data["exploratory"]["loss_can_fire"]["up_to_turn_10"]["denominator"] == 2
