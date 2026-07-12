"""File-based command bridge for Python-to-Lua command delivery.

Writes command JSON to a file that the FireTuner panel relay reads and
injects into ExposedMembers.AlphaZeroCommand for the mod to consume.

Architecture:
    Python (TunerBridge) --writes--> commands.json
        --read by--> FireTuner Panel Lua (AlphaZeroBridge.ltp)
        --sets--> ExposedMembers.AlphaZeroCommand
        --polled by--> Mod Main.lua -> ActionExecutor

The FireTuner panel must be loaded in the FireTuner debug tool for
commands to be relayed. Without it, commands.json will accumulate
but not be delivered to the game.

Usage:
    bridge = TunerBridge(config)
    async with bridge:
        msg_id = await bridge.send_end_turn()
"""

from __future__ import annotations

import logging
from typing import Any

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.protocol import (
    ActionType,
    CommandMessage,
    LuaCommand,
    create_end_turn_command,
    create_unit_move_command,
)
from alpha_zero.bridge.utils import next_monotonic_msg_id, seed_msg_counter

logger = logging.getLogger(__name__)

__all__ = [
    "TunerBridge",
    "TunerBridgeError",
]


class TunerBridgeError(Exception):
    """Exception for TunerBridge errors."""


class TunerBridge:
    """File-based command bridge for Python→Lua communication.

    Writes commands to ``commands.json`` in the bridge comm directory.
    A FireTuner panel (``AlphaZeroBridge.ltp``) running inside Civ6
    periodically reads this file and injects the command JSON into
    ``ExposedMembers.AlphaZeroCommand`` for the mod to consume.

    Args:
        config: Bridge configuration with comm_dir and other settings.
    """

    def __init__(self, config: BridgeConfig) -> None:
        self._config = config
        self._msg_counter = seed_msg_counter()
        self._running = False
        self._commands_sent = 0

    # --- Context manager ---

    async def __aenter__(self) -> TunerBridge:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    # --- Lifecycle ---

    async def start(self) -> None:
        """Ensure the comm directory exists and clear stale commands."""
        self._config.ensure_comm_dir()

        # Clear any stale commands file
        commands_path = self._config.commands_file
        if commands_path.exists():
            try:
                commands_path.write_text("")
            except OSError as e:
                logger.warning("Could not clear stale commands: %s", e)

        self._running = True
        logger.info(
            "TunerBridge started. Commands file: %s",
            self._config.commands_file,
        )

    async def stop(self) -> None:
        """Stop the bridge."""
        self._running = False
        logger.info("TunerBridge stopped. Commands sent: %d", self._commands_sent)

    # --- Command delivery ---

    def next_msg_id(self) -> int:
        """Generate a monotonically increasing message ID."""
        self._msg_counter = next_monotonic_msg_id(self._msg_counter)
        return self._msg_counter

    async def send_command(self, command: CommandMessage) -> bool:
        """Write a command message to commands.json.

        Uses atomic write (write to .tmp then rename) to prevent
        partial reads by the FireTuner panel.

        Args:
            command: The command message to send.

        Returns:
            True if the command was written successfully.

        Raises:
            TunerBridgeError: If writing fails.
        """
        if not self._running:
            raise TunerBridgeError("TunerBridge is not running")

        commands_path = self._config.commands_file
        tmp_path = commands_path.with_suffix(".tmp")

        try:
            json_str = command.to_json()

            # Atomic write: write to temp, then rename
            tmp_path.write_text(json_str, encoding="utf-8")
            # On Windows, need to remove target first if it exists
            if commands_path.exists():
                commands_path.unlink()
            tmp_path.rename(commands_path)

            self._commands_sent += 1
            logger.info(
                "Command sent (msg_id=%d, %d commands): %s",
                command.msg_id,
                len(command.commands),
                [c.action_type for c in command.commands],
            )
            return True

        except Exception as e:
            # Clean up temp file
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            logger.error("Failed to write command: %s", e)
            raise TunerBridgeError(f"Failed to write command: {e}") from e

    # --- Convenience methods ---

    async def send_end_turn(self) -> int:
        """Send an end_turn command.

        Returns:
            The message ID used for this command.
        """
        msg_id = self.next_msg_id()
        cmd = create_end_turn_command(msg_id)
        await self.send_command(cmd)
        return msg_id

    async def send_unit_move(self, unit_id: int, x: int, y: int) -> int:
        """Send a unit_move command.

        Args:
            unit_id: The unit to move.
            x: Target x coordinate.
            y: Target y coordinate.

        Returns:
            The message ID used for this command.
        """
        msg_id = self.next_msg_id()
        cmd = create_unit_move_command(msg_id, unit_id, x, y)
        await self.send_command(cmd)
        return msg_id

    async def send_city_produce(
        self, city_id: int, item_id: int, item_type: str = "unit"
    ) -> int:
        """Send a city_produce command.

        Args:
            city_id: The city ID.
            item_id: The item to produce.
            item_type: Type of item (unit/building/district).

        Returns:
            The message ID used for this command.
        """
        msg_id = self.next_msg_id()
        command = LuaCommand(
            action_type=ActionType.CITY_PRODUCE,
            city_id=city_id,
            item_id=item_id,
            item_type=item_type,
        )
        cmd = CommandMessage.create_single(msg_id, command)
        await self.send_command(cmd)
        return msg_id

    # --- Properties ---

    @property
    def is_running(self) -> bool:
        """Whether the bridge is currently running."""
        return self._running

    @property
    def commands_sent(self) -> int:
        """Total number of commands sent."""
        return self._commands_sent

    @property
    def stats(self) -> dict[str, Any]:
        """Return bridge statistics."""
        return {
            "commands_sent": self._commands_sent,
            "running": self._running,
        }
