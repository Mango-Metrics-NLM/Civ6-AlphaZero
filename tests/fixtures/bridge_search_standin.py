"""The real search driving the real loop in the stand-in game (GM-B6b), for tests C and D.

``LuaStandInBridge`` runs the mod's own Lua; ``StateConverter``, ``ActionConverter``, the
translator and ``Civ6MctsPolicy`` (a real ``MicroCivNetwork`` and a real ``GumbelMCTS`` at a
small budget) are the real ones. Nothing here mocks any of them.

``Restricted`` is the one liberty taken: it hands the real policy an ``allowed`` filter that is
the ledger's AND a test's own rule, so a scenario is decided by what may be named and not by an
untrained network's luck. The search still runs on the restricted root, and the game still
decides what happens.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import torch

from alpha_zero.bridge.action_converter import ActionConverter
from alpha_zero.bridge.agent_loop import AgentLoopConfig, BridgeAgentLoop
from alpha_zero.bridge.mcts_policy import Civ6MctsPolicy
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.core.civ6_mcts import GumbelMCTSConfig
from alpha_zero.core.civ6_network import MicroCivNetwork
from tests.fixtures import lua_commands as c
from tests.fixtures.lua_bridge import LuaStandInBridge

BOARD = 16
SIM_ACTIONS = 376
END = {"action_type": "end_turn"}
RESEARCH_BLOCKER = (
    "Civ6Stub.notifications[0] = { { id = 5, blocking = EndTurnBlockingTypes."
    "ENDTURN_BLOCKING_RESEARCH, message = 'Choose a technology',"
    " typeName = 'NOTIFICATION_CHOOSE_TECH', dismissed = false } }"
)
#: Nothing to do but research and end the turn: no moves, nothing to produce.
NO_UNITS_NO_BUILDS = (
    "Civ6Stub.unit(101).moves = 0\nCiv6Stub.unit(102).moves = 0\n"
    "Civ6Stub.cities[301].canProduce = {}\n"
)
NOTHING_RESEARCHABLE = "Civ6Stub.players[0].tech.can = {}\nCiv6Stub.players[0].civic.can = {}\n"
#: The game accepts every unit and city request and does nothing with it.
IGNORE_UNIT_REQUESTS = (
    "do local apply = Civ6Stub.applyRequests\n"
    "function Civ6Stub.applyRequests() Civ6Stub.unitRequests = {} return apply() end end\n"
)
SAFETY_TURNS = 40


def small_network() -> MicroCivNetwork:
    torch.manual_seed(0)
    network = MicroCivNetwork(board_size=BOARD, action_space_size=SIM_ACTIONS)
    network.eval()
    return network


def search_policy(seed: int | None = 5, network: Any = None) -> Civ6MctsPolicy:
    config = GumbelMCTSConfig(num_simulations=8, num_sampled_actions=4, seed=seed)
    return Civ6MctsPolicy(network or small_network(), config, device="cpu", max_turns=30)


class Restricted:
    """The real policy, asked to name only commands that ``keep`` accepts as well."""

    def __init__(self, inner: Civ6MctsPolicy, keep: Callable[[dict[str, Any]], bool]) -> None:
        self.inner = inner
        self.keep = keep
        self.config = inner.config
        self.returned: list[dict[str, Any]] = []

    async def select_command(
        self,
        converted_state: Any,
        *,
        allowed: Callable[[dict[str, Any]], bool] | None = None,
        decision_seed: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        def both(command: dict[str, Any]) -> bool:
            return bool(self.keep(command)) and (allowed is None or bool(allowed(command)))

        command, record = await self.inner.select_command(
            converted_state, allowed=both, decision_seed=decision_seed
        )
        self.returned.append(command)
        return command, record


class SpoilingBridge(LuaStandInBridge):
    """Runs Lua just before the mod reads one command, and Lua that undoes it after."""

    def __init__(self, runtime: Any, spoil: dict[str, tuple[str, str]] | None = None) -> None:
        super().__init__(runtime)
        self._spoil = spoil or {}
        self.turns: list[int] = []

    async def send_command(self, command: Any) -> bool:
        (single,) = json.loads(command.to_json())["commands"]
        self.turns.append(int(self.runtime.eval("Civ6Stub.turn")))
        before, after = self._spoil.get(json.dumps(single, sort_keys=True), ("", ""))
        if before:
            self.runtime.execute(before)
        sent = await super().send_command(command)
        if after:
            self.runtime.execute(after)
        return sent


def play(
    extra: str = "",
    *,
    policy: Any = None,
    keep: Callable[[dict[str, Any]], bool] | None = None,
    seed: int | None = 5,
    spoil: dict[str, tuple[str, str]] | None = None,
    **config: Any,
) -> tuple[Any, SpoilingBridge, BridgeAgentLoop, Any, int]:
    """Play the loop over a fresh stand-in game; returns runtime, bridge, loop, policy, exit code."""
    runtime = c.loaded_turn(extra=extra)
    if policy is None:
        policy = search_policy(seed)
        if keep is not None:
            policy = Restricted(policy, keep)
    bridge = SpoilingBridge(runtime, spoil)
    config.setdefault("max_turns", SAFETY_TURNS)
    loop = BridgeAgentLoop(
        bridge=bridge,
        policy=policy,
        state_converter=StateConverter(board_size=BOARD),
        action_converter=ActionConverter(board_size=BOARD),
        config=AgentLoopConfig(per_turn_timeout_s=5.0, **config),
    )
    code = asyncio.run(loop.run())
    return runtime, bridge, loop, policy, code


def only(*kinds: str) -> Callable[[dict[str, Any]], bool]:
    return lambda command: command["action_type"] in kinds


def failed_acks(runtime: Any) -> list[str]:
    """The per-command error code of every acknowledgement that was a failure."""
    return [a["executed"][0]["error_code"] for a in c.acks(runtime) if not a["success"]]


def unit_xy(runtime: Any, unit_id: int) -> tuple[int, int] | None:
    """Where the stand-in game itself has a unit; None once it is gone."""
    found = runtime.eval(
        f"(function() local u = Civ6Stub.unit({unit_id}) if not u then return nil end"
        " return { x = u.x, y = u.y } end)()"
    )
    return None if found is None else (int(found["x"]), int(found["y"]))


def own_city_count(runtime: Any) -> int:
    return int(runtime.eval("#Civ6Stub.cityRecords[0]"))
