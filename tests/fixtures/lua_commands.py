"""Helpers for the GM-B7a tests: drive the mod against the stand-in game's frames."""

from __future__ import annotations

import json
from typing import Any

from tests.fixtures import gameinfo_snapshot as g
from tests.fixtures import lua_harness as h

#: Run before the bundle: a clock the test sets, and a log of every strict check asked.
CLOCK = "Civ6Stub.now = 1000\nos.time = function() return Civ6Stub.now end\n"
CHECK_LOG = """
Civ6Stub.checks = {}
Civ6Stub.commandChecks = {}
Civ6Stub.warChecks = {}
Civ6Stub.produceChecks = {}
do
  local op = UnitManager.CanStartOperation
  UnitManager.CanStartOperation = function(unit, hash, plot, fourth, fifth)
    table.insert(Civ6Stub.checks, { hash = hash, plot = plot, fourth = fourth, fifth = fifth })
    return op(unit, hash, plot, fourth, fifth)
  end
  local cmd = UnitManager.CanStartCommand
  UnitManager.CanStartCommand = function(unit, hash, loose, want)
    table.insert(Civ6Stub.commandChecks, { hash = hash, loose = loose, want = want })
    return cmd(unit, hash, loose, want)
  end
  local war = CombatManager.IsAttackChangeWarState
  CombatManager.IsAttackChangeWarState = function(id, x, y)
    table.insert(Civ6Stub.warChecks, { player = id.player, id = id.id, x = x, y = y })
    return war(id, x, y)
  end
  local queue = Players[0]:GetCities():FindID(301):GetBuildQueue()
  local can = queue.CanProduce
  queue.CanProduce = function(self, ref, loose, strict)
    table.insert(Civ6Stub.produceChecks, { ref = ref, loose = loose, strict = strict })
    return can(self, ref, loose, strict)
  end
end
"""
#: A ranged unit 102 (range 2) beside the enemy warrior 201 at (2, 0).
ARCHER = "Civ6Stub.unit(102).ranged = 30\nCiv6Stub.unit(102).range = 2\n"


def load(setup: str = "") -> Any:
    return h.load_mod(setup=setup)


def frame(runtime: Any) -> None:
    """The game's next frame: carry out what was requested (no event published)."""
    runtime.execute("Civ6Stub.applyRequests()")


def num(runtime: Any, expression: str) -> Any:
    return runtime.eval(expression)


def requests(runtime: Any, name: str) -> list[tuple[int, dict[Any, Any]]]:
    """Recorded game-changing calls called ``name``: ``(operation, params)`` each."""
    found = []
    for i in range(1, runtime.eval("#Civ6Stub.calls") + 1):
        call = runtime.eval(f"Civ6Stub.calls[{i}]")
        if call["name"] == name:
            args = call["args"]
            found.append((args[2], dict(args[3].items()) if args[3] is not None else {}))
    return found


def queued(runtime: Any) -> int:
    """Requests recorded for the game and not yet carried out."""
    return int(runtime.eval("#Civ6Stub.unitRequests"))


def dropped(runtime: Any) -> int:
    return int(runtime.eval("#Civ6Stub.dropped"))


def const(runtime: Any, expression: str) -> Any:
    return runtime.eval(expression)


def row_hash(runtime: Any, table: str, column: str, type_name: str) -> int:
    """The ``Hash`` of the ``GameInfo.<table>`` row whose ``column`` is ``type_name``."""
    return int(
        runtime.eval(
            f"(function() for r in GameInfo.{table}() do "
            f"if r.{column} == '{type_name}' then return r.Hash end end end)()"
        )
    )


# ---- the main loop ---------------------------------------------------------
def loaded_turn(setup: str = "", *, extra: str = "") -> Any:
    """The mod on the AI player's turn, clock at 1000, ``print`` captured."""
    runtime = g.load_printing(CLOCK + setup)
    runtime.execute(extra)
    runtime.execute("AlphaZeroBridge.Initialize()\nAlphaZeroBridge.OnTurnStart(0)")
    return runtime


def send(runtime: Any, commands: list[dict[str, Any]], msg_id: int = 1) -> None:
    text = json.dumps({"msg_id": msg_id, "commands": commands})
    runtime.eval("function(t) ExposedMembers.AlphaZeroCommand = t end")(text)


def pending(runtime: Any) -> bool:
    return bool(runtime.eval("ExposedMembers.AlphaZeroCommand ~= nil"))


def tick(runtime: Any, count: int = 1) -> None:
    for _ in range(count):
        runtime.execute("Civ6Stub.tick()")


def at(runtime: Any, seconds: int) -> None:
    runtime.execute(f"Civ6Stub.now = {1000 + int(seconds)}")


def states(runtime: Any) -> list[dict[str, Any]]:
    return [json.loads(text) for text, _ in g.messages(g.printed(runtime), "AZ_STATE")]


def acks(runtime: Any) -> list[dict[str, Any]]:
    return [json.loads(text) for text, _ in g.messages(g.printed(runtime), "AZ_ACK")]


def unit_in(state: dict[str, Any], unit_id: int) -> dict[str, Any]:
    (found,) = [u for u in state["units"] if u["id"] == unit_id]
    return found


def unit_move(unit_id: int, x: int, y: int) -> dict[str, Any]:
    return {"action_type": "unit_move", "unit_id": unit_id, "target": {"x": x, "y": y}}
