"""A state whose list or object field has the wrong type is void on every board (round 2).

Present and not a list (null included) for ``units``, ``cities``, ``players``,
``legal_actions`` and ``map.tiles``; ``map`` present and not an object. The reason says
"malformed" and names the field. An absent key is not malformed: the parser defaults it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.census_states import (
    load_census,
    mixed_state,
    read_json,
    run_cli,
    state_line,
    wire,
    write_json,
    write_log,
)

SHA = "0123456789abcdef0123456789abcdef01234567"

Damage = Callable[[dict[str, Any]], None]


def _set(key: str, value: Any) -> Damage:
    def apply(raw: dict[str, Any]) -> None:
        raw[key] = value

    return apply


def _tiles_null(raw: dict[str, Any]) -> None:
    raw["map"] = {**raw["map"], "tiles": None}


CASES: list[tuple[str, Damage, str]] = [
    ("units null", _set("units", None), "units"),
    ("cities string", _set("cities", "x"), "cities"),
    ("legal_actions int", _set("legal_actions", 5), "legal_actions"),
    ("players dict", _set("players", {}), "players"),
    ("map list", _set("map", []), "map"),
    ("map.tiles null", _tiles_null, "tiles"),
]
IDS = [c[0] for c in CASES]


@pytest.fixture(autouse=True)
def _pinned_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_GIT_SHA", SHA)


def _damaged(turn: int, damage: Damage) -> dict[str, Any]:
    raw = wire(mixed_state(turn))
    damage(raw)
    return raw


@pytest.mark.parametrize(("damage", "field"), [c[1:] for c in CASES], ids=IDS)
def test_record_voids_the_state_on_every_board_and_names_the_field(
    damage: Damage, field: str
) -> None:
    mod: Any = load_census()
    rec = mod._record(0, _damaged(3, damage), [8, 16])
    assert set(rec["boards"]) == {"8", "16"}
    for cell in rec["boards"].values():
        assert cell["void"] is True
        assert "malformed" in cell["void_reason"] and field in cell["void_reason"]
    assert "map_size" not in rec


@pytest.mark.parametrize("key", ["units", "cities", "players", "legal_actions", "map"])
def test_control_an_absent_key_is_not_malformed(key: str) -> None:
    mod: Any = load_census()
    raw = wire(mixed_state(3))
    raw.pop(key, None)
    rec = mod._record(0, raw, [8])
    reason = rec["boards"]["8"].get("void_reason") or ""
    assert "malformed" not in reason


@pytest.mark.parametrize(("damage", "field"), [c[1:] for c in CASES], ids=IDS)
def test_through_the_cli_it_is_void_counted_not_in_the_minimum_and_a_later_state_is_measured(
    tmp_path: Path, damage: Damage, field: str
) -> None:
    mod: Any = load_census()
    states = [_damaged(2, damage), wire(mixed_state(3))]
    if field in ("map", "tiles"):
        # The live reader itself raises on a bad map, so a log never delivers it (counted
        # unreadable there); the census meets it only through --states-json.
        source = ["--states-json", str(write_json(tmp_path / "s.json", states))]
    else:
        source = ["--log", str(write_log(tmp_path / "Lua.log", [state_line(x) for x in states]))]
    code, target = run_cli(mod, tmp_path, *source, "--allow-small", "--boards", "8")
    assert code == 0
    data = read_json(target)
    info = data["input"]
    assert info["states_sampled"] == 2
    assert info["states_void"] == 1 and info["states_measured"] == 1
    assert info["meets_registered_minimum"] is False
    bad, good = data["states"]
    assert bad["turn"] == 2 and good["turn"] == 3 and "map_size" in good
    reason = bad["boards"]["8"]["void_reason"]
    assert "malformed" in reason and field in reason
