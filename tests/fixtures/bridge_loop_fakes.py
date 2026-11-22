"""Small fakes for driving the real ``BridgeAgentLoop`` without a game (GM-B6b).

Classes, not mocks: a test reads what the bridge was really given. The state is a
``SimpleNamespace`` the real ``TurnLedger`` can read (it cannot summarise one, so a withheld
``end_turn`` stays withheld until the turn changes, as in the reselect tests).

``Table`` maps an action index to the command it means; the mask offered is whatever the test
says. Nothing here knows about command policies: they are the tests' own fakes.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np

from alpha_zero.bridge.action_converter import InvalidActionError
from alpha_zero.bridge.agent_loop import AgentLoopConfig, BridgeAgentLoop
from alpha_zero.bridge.protocol import AckMessage, LuaCommand

SIZE = 8
END = {"action_type": "end_turn"}
SKIP_1 = {"action_type": "unit_skip", "unit_id": 1}
SKIP_2 = {"action_type": "unit_skip", "unit_id": 2}
MOVE_1 = {"action_type": "unit_move", "unit_id": 1, "target": {"x": 0, "y": 1}}
RESEARCH = {"action_type": "research_tech", "item_id": "TECH_WRITING"}
TABLE: dict[int, dict[str, Any]] = {0: END, 1: SKIP_1, 2: SKIP_2, 3: MOVE_1, 4: RESEARCH}
ALL = tuple(TABLE)


def game_state(turn: int = 5, *, over: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        turn_id=turn,
        player_id=0,
        is_game_over=over,
        winner_id=None,
        legal_actions=[],
        end_turn_blocking=None,
    )


def ok(_command: dict[str, Any]) -> AckMessage:
    return AckMessage(msg_id=1, success=True)


def refusing(*targets: dict[str, Any]) -> Callable[[dict[str, Any]], AckMessage]:
    def ack_for(command: dict[str, Any]) -> AckMessage:
        if command in targets:
            executed = [{"success": False, "error_code": "OPERATION_REFUSED", "message": "no"}]
            return AckMessage(
                msg_id=1, success=False, error_code="COMMAND_FAILED", executed=executed
            )
        return AckMessage(msg_id=1, success=True)

    return ack_for


def blocking_end_turn(command: dict[str, Any]) -> AckMessage:
    """The game blocks every end_turn and accepts everything else."""
    if command == END:
        executed = [{"success": False, "error_code": "END_TURN_BLOCKED", "message": "B"}]
        return AckMessage(msg_id=1, success=False, error_code="COMMAND_FAILED", executed=executed)
    return AckMessage(msg_id=1, success=True)


class FakeBridge:
    """Hands out scripted states, records every command, answers by ``ack_for``."""

    def __init__(
        self,
        states: list[SimpleNamespace],
        ack_for: Callable[[dict[str, Any]], AckMessage] = ok,
    ) -> None:
        self._states = list(states)
        self._ack_for = ack_for
        self._id = 0
        self.sent: list[dict[str, Any]] = []

    def next_msg_id(self) -> int:
        self._id += 1
        return self._id

    def check_heartbeat(self) -> bool:
        return True

    async def wait_for_state(self, timeout: float | None = None, require_new: bool = True) -> Any:
        return self._states.pop(0)

    def _record(self, command: Any) -> dict[str, Any]:
        (single,) = json.loads(command.to_json())["commands"]
        self.sent.append(single)
        return single

    async def send_command(self, command: Any) -> bool:
        self._record(command)
        return True

    async def send_and_wait_with_retry(self, command: Any, *_: Any, **__: Any) -> AckMessage:
        return self._ack_for(self._record(command))


class FakeStateConverter:
    def __init__(self) -> None:
        self.converted: list[dict[str, Any]] = []

    def convert(self, state: Any) -> dict[str, Any]:
        converted = {
            "_legal_actions": [],
            "_obs": {
                "spatial": np.zeros((13, SIZE, SIZE), dtype=np.float32),
                "vector": np.zeros(64, dtype=np.float32),
            },
        }
        self.converted.append(converted)
        return converted


class FakeActionConverter:
    """Index -> ``TABLE``; the mask offers ``offered`` (default: every index in the table)."""

    def __init__(self, offered: tuple[int, ...] = ALL) -> None:
        self.offered = offered

    def generate_action_mask(self, legal_actions: Any, converted_state: Any) -> np.ndarray:
        mask = np.zeros(SIZE, dtype=np.float32)
        mask[list(self.offered)] = 1.0
        return mask

    def convert(
        self, idx: int, state: Any, legal_actions: Any = None, validate: bool = True
    ) -> Any:
        if idx not in TABLE:
            raise InvalidActionError(f"action {idx} is not offered")
        return LuaCommand.from_dict(TABLE[idx])


class Harness:
    """An eval harness that remembers what it was told."""

    def __init__(self) -> None:
        self.turn_ends: list[tuple[Any, dict[str, Any]]] = []

    def on_start(self, *_: Any, **__: Any) -> None:
        pass

    def on_stop(self, *_: Any, **__: Any) -> None:
        pass

    def on_turn_end(self, turn_id: Any, diagnostics: dict[str, Any]) -> None:
        self.turn_ends.append((turn_id, dict(diagnostics)))


def play(
    states: list[SimpleNamespace],
    policy: Any,
    *,
    ack_for: Callable[[dict[str, Any]], AckMessage] = ok,
    offered: tuple[int, ...] = ALL,
    harness: Any = None,
    **config: Any,
) -> tuple[BridgeAgentLoop, FakeBridge, FakeStateConverter, int]:
    bridge = FakeBridge(states, ack_for)
    converter = FakeStateConverter()
    config.setdefault("max_turns", 0)
    loop = BridgeAgentLoop(
        bridge=bridge,  # type: ignore[arg-type]
        policy=policy,
        state_converter=converter,  # type: ignore[arg-type]
        action_converter=FakeActionConverter(offered),  # type: ignore[arg-type]
        config=AgentLoopConfig(per_turn_timeout_s=0.5, **config),
        eval_harness=harness,
    )
    code = asyncio.run(loop.run())
    return loop, bridge, converter, code
