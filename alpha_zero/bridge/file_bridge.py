"""File-based IPC bridge for Civ6 communication.

This module implements file-based inter-process communication between
the Python AlphaZero agent and the Civ6 Lua mod. It uses file polling
because Civ6's Lua runtime is sandboxed and cannot use sockets.

Communication Protocol:
    1. Lua writes game state to state.json
    2. Python polls and reads state.json
    3. Python writes action command to commands.json
    4. Lua reads commands.json and executes
    5. Lua writes acknowledgment to ack.json
    6. Heartbeat: Python writes initial heartbeat at start; Lua writes
       heartbeat periodically. Python checks Lua's heartbeat for liveness.

Usage:
    from alpha_zero.bridge import FileBridge, BridgeConfig

    config = BridgeConfig()
    bridge = FileBridge(config)

    async with bridge:
        # Wait for initial state
        state = await bridge.wait_for_state()

        # Send command
        await bridge.send_command(command)

        # Wait for acknowledgment
        ack = await bridge.wait_for_ack(command.msg_id)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.protocol import (
    AckMessage,
    CommandMessage,
    ErrorCode,
    ErrorMessage,
    GameStateData,
    StateMessage,
)

logger = logging.getLogger(__name__)


@dataclass
class BridgeStats:
    """Statistics for bridge operation."""

    states_received: int = 0
    commands_sent: int = 0
    acks_received: int = 0
    errors: int = 0
    reconnects: int = 0
    last_state_time: float = 0.0
    last_command_time: float = 0.0
    last_ack_time: float = 0.0
    last_heartbeat_time: float = 0.0


class FileBridgeError(Exception):
    """Base exception for file bridge errors."""

    def __init__(self, message: str, code: str = ErrorCode.UNKNOWN):
        super().__init__(message)
        self.code = code


class StateTimeoutError(FileBridgeError):
    """Timeout waiting for state."""

    def __init__(self, timeout: float):
        super().__init__(f"Timeout waiting for state after {timeout}s", ErrorCode.TIMEOUT)


class AckTimeoutError(FileBridgeError):
    """Timeout waiting for acknowledgment."""

    def __init__(self, msg_id: int, timeout: float):
        super().__init__(
            f"Timeout waiting for ack (msg_id={msg_id}) after {timeout}s",
            ErrorCode.TIMEOUT,
        )


class ConnectionLostError(FileBridgeError):
    """Connection to game lost (heartbeat timeout)."""

    def __init__(self):
        super().__init__("Connection to game lost (heartbeat timeout)", ErrorCode.CONNECTION_LOST)


class FileBridge:
    """File-based IPC bridge for Civ6 communication.

    This class manages the file-based communication channel between
    Python and the Civ6 Lua mod.

    Attributes:
        config: Bridge configuration.
        stats: Operation statistics.
    """

    def __init__(self, config: BridgeConfig | None = None):
        """Initialize the file bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or BridgeConfig()
        self.stats = BridgeStats()
        self._msg_counter = 0
        self._last_state_mtime = 0.0
        self._last_state: GameStateData | None = None
        self._running = False
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> FileBridge:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Start the bridge.

        Creates the communication directory and initializes state.
        """
        logger.info(f"Starting file bridge with comm_dir: {self.config.comm_dir}")
        self.config.ensure_comm_dir()
        self._running = True
        self._msg_counter = int(time.time() * 1000)
        self._write_heartbeat()
        logger.info("File bridge started")

    async def stop(self) -> None:
        """Stop the bridge.

        Cleans up resources.
        """
        logger.info("Stopping file bridge")
        self._running = False
        logger.info("File bridge stopped")

    def _next_msg_id(self) -> int:
        """Generate next message ID."""
        self._msg_counter += 1
        return self._msg_counter

    def _write_heartbeat(self) -> None:
        """Write heartbeat file to indicate Python is alive."""
        try:
            heartbeat = {
                "timestamp": time.time(),
                "python_alive": True,
            }
            self.config.heartbeat_file.write_text(json.dumps(heartbeat))
            self.stats.last_heartbeat_time = time.time()
        except Exception as e:
            logger.warning(f"Failed to write heartbeat: {e}")

    def check_heartbeat(self) -> bool:
        """Check if the game is still alive (recent heartbeat).

        Returns:
            True if game heartbeat is recent, False otherwise.
        """
        try:
            if not self.config.heartbeat_file.exists():
                return False

            heartbeat_text = self.config.heartbeat_file.read_text()
            heartbeat = json.loads(heartbeat_text)
            heartbeat_time = heartbeat.get("timestamp", 0)

            age = time.time() - heartbeat_time
            return age < self.config.heartbeat_timeout_s

        except Exception as e:
            logger.debug(f"Heartbeat check failed: {e}")
            return False

    async def poll_state(self) -> GameStateData | None:
        """Poll for new state from the game.

        Returns:
            GameStateData if new state available, None otherwise.
        """
        try:
            state_file = self.config.state_file

            if not state_file.exists():
                return None

            # Check if file has been modified
            mtime = state_file.stat().st_mtime
            if mtime <= self._last_state_mtime:
                return None

            # Read and parse state
            async with self._lock:
                try:
                    content = state_file.read_text(encoding="utf-8")
                    data = json.loads(content)

                    state_msg = StateMessage.from_dict(data)
                    self._last_state = state_msg.state
                    self._last_state_mtime = mtime
                    self.stats.states_received += 1
                    self.stats.last_state_time = time.time()

                    logger.debug(
                        f"Received state: turn={state_msg.state.turn_id}, "
                        f"units={len(state_msg.state.units)}, "
                        f"cities={len(state_msg.state.cities)}"
                    )

                    return state_msg.state

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in state file: {e}")
                    return None

        except Exception as e:
            logger.error(f"Error polling state: {e}")
            self.stats.errors += 1
            return None

    async def wait_for_state(
        self,
        timeout: float | None = None,
        require_new: bool = True,
    ) -> GameStateData:
        """Wait for a state update from the game.

        Args:
            timeout: Maximum time to wait (seconds). Uses config default if None.
            require_new: If True, wait for a new state (different mtime).

        Returns:
            The received game state.

        Raises:
            StateTimeoutError: If timeout expires without receiving state.
            ConnectionLostError: If heartbeat times out.
        """
        timeout = timeout or (self.config.command_timeout_ms / 1000.0)
        poll_interval = self.config.poll_interval_ms / 1000.0
        start_time = time.time()

        logger.debug(f"Waiting for state (timeout={timeout}s, require_new={require_new})")

        # Track heartbeat check timing separately
        last_heartbeat_check = start_time

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise StateTimeoutError(timeout)

            # Check heartbeat after grace period (allow initial startup time)
            # Only check periodically to avoid excessive file I/O
            now = time.time()
            if elapsed > self.config.heartbeat_timeout_s and now - last_heartbeat_check > 0.5:
                last_heartbeat_check = now
                if not self.check_heartbeat():
                    raise ConnectionLostError()

            # Poll for state
            state = await self.poll_state()

            if state is not None:
                if not require_new or self.stats.states_received > 1:
                    return state
                # First state is always considered new
                if self.stats.states_received == 1:
                    return state

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    def get_last_state(self) -> GameStateData | None:
        """Get the last received state.

        Returns:
            Last received GameStateData, or None if no state received.
        """
        return self._last_state

    async def send_command(self, command: CommandMessage) -> bool:
        """Send a command to the game.

        Args:
            command: CommandMessage to send.

        Returns:
            True if command was written successfully.
        """
        try:
            async with self._lock:
                json_str = command.to_json()
                self.config.commands_file.write_text(json_str, encoding="utf-8")

                self.stats.commands_sent += 1
                self.stats.last_command_time = time.time()

                logger.debug(
                    f"Sent command: msg_id={command.msg_id}, actions={len(command.commands)}"
                )

                return True

        except Exception as e:
            logger.error(f"Error sending command: {e}")
            self.stats.errors += 1
            return False

    async def send_command_with_retry(
        self,
        command: CommandMessage,
        max_retries: int | None = None,
    ) -> bool:
        """Send a command to the game with retry logic.

        Implements exponential backoff retry for transient I/O failures.

        Args:
            command: CommandMessage to send.
            max_retries: Maximum retry attempts. Uses config.max_retries if None.

        Returns:
            True if command was written successfully after retries.
        """
        retries = max_retries if max_retries is not None else self.config.max_retries

        for attempt in range(retries):
            success = await self.send_command(command)
            if success:
                return True

            if attempt < retries - 1:
                delay = self._get_retry_delay(attempt)
                logger.warning(
                    "Send command failed (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    retries,
                    delay,
                )
                await asyncio.sleep(delay)
                self.stats.reconnects += 1

        logger.error(
            "All %d send attempts failed for msg_id=%d",
            retries,
            command.msg_id,
        )
        return False

    async def wait_for_ack(
        self,
        msg_id: int,
        timeout: float | None = None,
    ) -> AckMessage:
        """Wait for acknowledgment of a command.

        Args:
            msg_id: Message ID to wait for.
            timeout: Maximum time to wait (seconds).

        Returns:
            The acknowledgment message.

        Raises:
            AckTimeoutError: If timeout expires without receiving ack.
        """
        timeout = timeout or (self.config.command_timeout_ms / 1000.0)
        poll_interval = self.config.poll_interval_ms / 1000.0
        start_time = time.time()

        logger.debug(f"Waiting for ack (msg_id={msg_id}, timeout={timeout}s)")

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise AckTimeoutError(msg_id, timeout)

            # Try to read ack
            ack = await self._poll_ack(msg_id)
            if ack is not None:
                return ack

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    async def _poll_ack(self, expected_msg_id: int) -> AckMessage | None:
        """Poll for acknowledgment file.

        Args:
            expected_msg_id: Expected message ID.

        Returns:
            AckMessage if found and matches, None otherwise.
        """
        try:
            ack_file = self.config.ack_file

            if not ack_file.exists():
                return None

            content = ack_file.read_text(encoding="utf-8")
            data = json.loads(content)

            ack = AckMessage.from_dict(data)

            # Check if this is the ack we're waiting for
            if ack.msg_id != expected_msg_id:
                return None

            self.stats.acks_received += 1
            self.stats.last_ack_time = time.time()

            logger.debug(f"Received ack: msg_id={ack.msg_id}, success={ack.success}")

            return ack

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in ack file: {e}")
            return None
        except Exception as e:
            logger.error(f"Error polling ack: {e}")
            return None

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff.

        Args:
            attempt: Current retry attempt (0-indexed).

        Returns:
            Delay in seconds for this attempt.
        """
        base_delay = self.config.retry_delay_s
        # Exponential backoff with cap at 60 seconds
        delay = base_delay * (2 ** attempt)
        return min(delay, 60.0)

    async def send_and_wait(
        self,
        command: CommandMessage,
        timeout: float | None = None,
    ) -> AckMessage:
        """Send a command and wait for acknowledgment.

        Args:
            command: CommandMessage to send.
            timeout: Maximum time to wait for ack.

        Returns:
            The acknowledgment message.

        Raises:
            FileBridgeError: If send fails.
            AckTimeoutError: If timeout expires.
        """
        success = await self.send_command(command)
        if not success:
            raise FileBridgeError("Failed to send command", ErrorCode.FILE_IO_ERROR)

        return await self.wait_for_ack(command.msg_id, timeout)

    async def send_and_wait_with_retry(
        self,
        command: CommandMessage,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> AckMessage:
        """Send a command and wait for acknowledgment with retry logic.

        Implements exponential backoff retry strategy for transient failures.
        Uses configuration values for max_retries and retry_delay_s if not
        explicitly provided.

        Args:
            command: CommandMessage to send.
            timeout: Maximum time to wait for ack per attempt.
            max_retries: Maximum retry attempts. Uses config.max_retries if None.

        Returns:
            The acknowledgment message.

        Raises:
            FileBridgeError: If all retries exhausted.
            AckTimeoutError: If final attempt times out.
        """
        retries = max_retries if max_retries is not None else self.config.max_retries
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                return await self.send_and_wait(command, timeout)

            except AckTimeoutError as e:
                last_error = e
                self.stats.errors += 1

                if attempt < retries - 1:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        "Command timed out (attempt %d/%d), retrying in %.1fs: msg_id=%d",
                        attempt + 1,
                        retries,
                        delay,
                        command.msg_id,
                    )
                    await asyncio.sleep(delay)
                    self.stats.reconnects += 1

            except FileBridgeError as e:
                last_error = e
                self.stats.errors += 1

                if attempt < retries - 1:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        "Command failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        retries,
                        delay,
                        str(e),
                    )
                    await asyncio.sleep(delay)
                    self.stats.reconnects += 1

        # All retries exhausted
        logger.error(
            "All %d retry attempts exhausted for msg_id=%d",
            retries,
            command.msg_id,
        )
        if last_error is not None:
            raise last_error
        raise FileBridgeError(
            f"All {retries} retry attempts failed",
            ErrorCode.TIMEOUT,
        )

    def read_errors(self) -> list[ErrorMessage]:
        """Read error messages from the errors file.

        Returns:
            List of error messages.
        """
        try:
            errors_file = self.config.errors_file

            if not errors_file.exists():
                return []

            content = errors_file.read_text(encoding="utf-8")

            # Try to parse as JSON array
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return [ErrorMessage.from_json(json.dumps(e)) for e in data]
                else:
                    return [ErrorMessage.from_json(content)]
            except Exception:
                # Try line-by-line
                errors = []
                for line in content.strip().split("\n"):
                    if line:
                        try:
                            errors.append(ErrorMessage.from_json(line))
                        except Exception:
                            pass
                return errors

        except Exception as e:
            logger.error(f"Error reading errors file: {e}")
            return []

    def clear_files(self) -> None:
        """Clear all communication files.

        Useful for starting fresh or resetting state.
        """
        for file_path in [
            self.config.state_file,
            self.config.commands_file,
            self.config.ack_file,
            self.config.errors_file,
        ]:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to clear {file_path}: {e}")

        self._last_state_mtime = 0.0
        self._last_state = None
        logger.info("Cleared communication files")

    def get_stats(self) -> dict[str, Any]:
        """Get bridge statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "states_received": self.stats.states_received,
            "commands_sent": self.stats.commands_sent,
            "acks_received": self.stats.acks_received,
            "errors": self.stats.errors,
            "reconnects": self.stats.reconnects,
            "last_state_time": self.stats.last_state_time,
            "last_command_time": self.stats.last_command_time,
            "last_ack_time": self.stats.last_ack_time,
            "is_connected": self.check_heartbeat(),
            "running": self._running,
        }

    @property
    def is_connected(self) -> bool:
        """Check if bridge is connected to game."""
        return self._running and self.check_heartbeat()


class MockFileBridge(FileBridge):
    """Mock file bridge for testing.

    Uses in-memory state instead of actual files. No file I/O is performed
    by this class - all operations are simulated in memory.
    """

    def __init__(self, config: BridgeConfig | None = None):
        """Initialize mock bridge."""
        super().__init__(config)
        self._mock_state: dict[str, Any] | None = None
        self._mock_ack: dict[str, Any] | None = None
        self._commands_sent: list[CommandMessage] = []

    async def start(self) -> None:
        """Start the mock bridge (no file I/O).

        Unlike the real FileBridge, this does not create directories
        or write heartbeat files, but still updates stats for consistency.
        """
        logger.debug("Starting mock file bridge (no file I/O)")
        self._running = True
        self._msg_counter = int(time.time() * 1000)
        # Update heartbeat stats without file I/O
        self._write_heartbeat()
        logger.debug("Mock file bridge started")

    async def stop(self) -> None:
        """Stop the mock bridge (no file I/O)."""
        logger.debug("Stopping mock file bridge")
        self._running = False
        logger.debug("Mock file bridge stopped")

    def _write_heartbeat(self) -> None:
        """Mock heartbeat write (no-op).

        Overrides parent to avoid file I/O.
        """
        self.stats.last_heartbeat_time = time.time()

    def set_mock_state(self, state: dict[str, Any]) -> None:
        """Set mock state data."""
        self._mock_state = state
        self._last_state_mtime = 0.0  # Force read

    def set_mock_ack(self, ack: dict[str, Any]) -> None:
        """Set mock ack data."""
        self._mock_ack = ack

    def get_sent_commands(self) -> list[CommandMessage]:
        """Get list of commands sent."""
        return self._commands_sent

    async def poll_state(self) -> GameStateData | None:
        """Poll mock state."""
        if self._mock_state is None:
            return None

        state_msg = StateMessage.from_dict(self._mock_state)
        self._last_state = state_msg.state
        self.stats.states_received += 1
        self.stats.last_state_time = time.time()

        # Clear after reading to simulate one-time delivery
        result = state_msg.state
        self._mock_state = None
        return result

    async def send_command(self, command: CommandMessage) -> bool:
        """Record mock command."""
        self._commands_sent.append(command)
        self.stats.commands_sent += 1
        self.stats.last_command_time = time.time()
        return True

    async def _poll_ack(self, expected_msg_id: int) -> AckMessage | None:
        """Poll mock ack."""
        if self._mock_ack is None:
            return None

        ack = AckMessage.from_dict(self._mock_ack)
        if ack.msg_id == expected_msg_id:
            self.stats.acks_received += 1
            self.stats.last_ack_time = time.time()
            self._mock_ack = None
            return ack

        return None

    def check_heartbeat(self) -> bool:
        """Mock always returns connected."""
        return True
