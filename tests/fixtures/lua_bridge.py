"""The mod's real Lua, in the stand-in game, as a ``GameBridge`` for the agent loop.

Lets a test run the real ``BridgeAgentLoop`` against the real mod code: a
command goes into the mod's command slot, frames are played until the mod has
answered, and states are read from what the mod printed. The stand-in game
advances a turn when an end of turn was carried out (``Civ6Stub.nextTurn``).

What this still does not show: that the real game behaves like the stand-in,
the log file and its timing (``LogTailer`` tests cover those), or how many
frames a real request takes.
"""

from __future__ import annotations

from typing import Any

from alpha_zero.bridge.file_bridge import StateTimeoutError
from alpha_zero.bridge.protocol import AckMessage, CommandMessage, GameStateData, StateMessage
from tests.fixtures import lua_commands as c

__all__ = ["LuaStandInBridge"]


class LuaStandInBridge:
    """A ``GameBridge`` over a ``lua_commands.loaded_turn()`` runtime."""

    def __init__(self, runtime: Any, *, max_frames: int = 12) -> None:
        self.runtime = runtime
        self._max_frames = max_frames
        self._states_seen = 0
        self._msg_id = 0
        #: Every command message handed to the mod, as sent.
        self.sent: list[dict[str, Any]] = []

    # --- GameBridge -----------------------------------------------------
    def next_msg_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def check_heartbeat(self) -> bool:
        return True

    async def wait_for_state(
        self, timeout: float | None = None, require_new: bool = True
    ) -> GameStateData:
        for _ in range(self._max_frames):
            exported = c.states(self.runtime)
            if len(exported) > self._states_seen:
                self._states_seen = len(exported)
                return StateMessage.from_dict(exported[-1]).state
            if bool(self.runtime.eval("Civ6Stub.turnEnded")):
                self.runtime.execute(
                    "Civ6Stub.nextTurn()\n"
                    "AlphaZeroBridge.OnTurnEnd(0)\n"
                    "AlphaZeroBridge.OnTurnStart(0)"
                )
            else:
                c.tick(self.runtime)
        raise StateTimeoutError(timeout or 0.0)

    async def send_command(self, command: CommandMessage) -> bool:
        message = command.to_dict() if hasattr(command, "to_dict") else None
        if message is None:
            import json

            message = json.loads(command.to_json())
        self.sent.append(message)
        c.send(self.runtime, message["commands"], msg_id=message["msg_id"])
        for _ in range(self._max_frames):
            if not c.pending(self.runtime):
                break
            c.tick(self.runtime)
        return True

    async def send_and_wait_with_retry(self, command: CommandMessage, *_: Any, **__: Any) -> Any:
        await self.send_command(command)
        for ack in reversed(c.acks(self.runtime)):
            if ack.get("msg_id") == command.msg_id:
                return AckMessage.from_dict(ack)
        raise TimeoutError(f"the mod did not answer message {command.msg_id}")

    # --- for assertions -------------------------------------------------
    def commands_sent(self) -> list[dict[str, Any]]:
        """Every single command sent, in order."""
        return [command for message in self.sent for command in message["commands"]]
