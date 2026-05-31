"""Unified live bridge composing LogTailer and TunerBridge.

Provides the same interface as FileBridge but uses:
- LogTailer: Reads game state, heartbeats, and acks from Lua.log
- TunerBridge: Writes commands to commands.json (relayed by FireTuner panel)

This is a drop-in replacement for FileBridge when running against a live
Civ6 Steam instance with print-based IPC (io sandbox mode).

Usage:
    config = BridgeConfig(comm_dir=Path("..."))
    log_path = Path(r"C:/Users/.../Logs/Lua.log")
    bridge = LiveBridge(config, log_path)

    async with bridge:
        state = await bridge.wait_for_state()
        msg_id = bridge.next_msg_id()
        cmd = create_end_turn_command(msg_id)
        await bridge.send_command(cmd)
        ack = await bridge.wait_for_ack(msg_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.log_tailer import (
    AckTimeoutError,
    LogTailer,
    StateTimeoutError,
)
from alpha_zero.bridge.protocol import CommandMessage, GameStateData, StateMessage
from alpha_zero.bridge.tuner_bridge import TunerBridge

logger = logging.getLogger(__name__)

__all__ = [
    "LiveBridge",
    "LiveBridgeError",
]


class LiveBridgeError(Exception):
    """Exception for LiveBridge errors."""


class LiveBridge:
    """Unified bridge composing LogTailer + TunerBridge.

    Provides an interface compatible with FileBridge for the agent loop,
    using log-tailing for reads and file-based command delivery for writes.

    Args:
        config: Bridge configuration.
        log_path: Path to Civ6 Lua.log file.
        poll_interval: Log polling interval in seconds.
    """

    def __init__(
        self,
        config: BridgeConfig,
        log_path: Path,
        poll_interval: float = 0.1,
    ) -> None:
        self._config = config
        self._tailer = LogTailer(log_path, poll_interval=poll_interval)
        self._tuner = TunerBridge(config)
        self._running = False
        self._msg_counter = int(time.time() * 1000)

    # --- Context manager ---

    async def __aenter__(self) -> LiveBridge:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    # --- Lifecycle ---

    async def start(self) -> None:
        """Start both the log tailer and command bridge."""
        await self._tuner.start()
        await self._tailer.start()
        self._running = True
        logger.info("LiveBridge started")

    async def stop(self) -> None:
        """Stop both components."""
        self._running = False
        await self._tailer.stop()
        await self._tuner.stop()
        logger.info("LiveBridge stopped")

    # --- State reading (delegates to LogTailer) ---

    async def poll_state(self) -> GameStateData | None:
        """Poll for new game state (non-blocking).

        Returns:
            Game state if a new state is available, None otherwise.
        """
        raw = self._tailer.latest_state
        if raw is None:
            return None
        return StateMessage.from_dict(raw).state

    async def wait_for_state(
        self, timeout: float | None = None, require_new: bool = True
    ) -> GameStateData:
        """Wait for game state to arrive.

        Args:
            timeout: Maximum seconds to wait. Defaults to config timeout.
            require_new: If True, waits for a new state even if one exists.

        Returns:
            Game state object.

        Raises:
            StateTimeoutError: If timeout is exceeded.
        """
        if timeout is None:
            timeout = self._config.command_timeout_ms / 1000.0

        if not require_new and self._tailer.latest_state is not None:
            return StateMessage.from_dict(self._tailer.latest_state).state

        raw = await self._tailer.wait_for_state(timeout=timeout)
        return StateMessage.from_dict(raw).state

    def check_heartbeat(self, max_age: float | None = None) -> bool:
        """Check if a recent heartbeat was received.

        Args:
            max_age: Maximum age in seconds. Defaults to config timeout.

        Returns:
            True if heartbeat is fresh.
        """
        if max_age is None:
            max_age = self._config.heartbeat_timeout_s
        return self._tailer.check_heartbeat(max_age=max_age)

    # --- Command sending (delegates to TunerBridge) ---

    def next_msg_id(self) -> int:
        """Generate a monotonically increasing message ID."""
        self._msg_counter += 1
        return self._msg_counter

    async def send_command(self, command: CommandMessage) -> bool:
        """Send a command to the game.

        Args:
            command: The command message to send.

        Returns:
            True if the command was written successfully.
        """
        return await self._tuner.send_command(command)

    async def wait_for_ack(
        self, msg_id: int, timeout: float | None = None
    ) -> dict[str, Any]:
        """Wait for command acknowledgment.

        Args:
            msg_id: The command message ID to wait for.
            timeout: Maximum seconds to wait.

        Returns:
            Ack dictionary.

        Raises:
            AckTimeoutError: If timeout is exceeded.
        """
        if timeout is None:
            timeout = self._config.command_timeout_ms / 1000.0
        return await self._tailer.wait_for_ack(msg_id, timeout=timeout)

    async def send_and_wait(
        self, command: CommandMessage, timeout: float | None = None
    ) -> dict[str, Any]:
        """Send a command and wait for its acknowledgment.

        Args:
            command: The command message to send.
            timeout: Maximum seconds to wait for ack.

        Returns:
            Ack dictionary.
        """
        await self.send_command(command)
        return await self.wait_for_ack(command.msg_id, timeout=timeout)

    # --- Properties ---

    @property
    def is_connected(self) -> bool:
        """Whether the bridge appears connected (heartbeat is fresh)."""
        return self._running and self.check_heartbeat()

    @property
    def stats(self) -> dict[str, Any]:
        """Combined statistics from both components."""
        tailer_stats = self._tailer.stats
        tuner_stats = self._tuner.stats
        return {
            **tailer_stats,
            **tuner_stats,
            "running": self._running,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics (compatibility with FileBridge)."""
        return self.stats

    def clear_files(self) -> None:
        """Clear communication files."""
        commands_path = self._config.commands_file
        if commands_path.exists():
            try:
                commands_path.write_text("")
            except OSError:
                pass
