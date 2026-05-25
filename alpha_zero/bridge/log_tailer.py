"""Log-tailing IPC bridge for reading Civ6 Lua.log output.

Tails the Civ6 Lua.log file in real-time, parsing AZ_STATE:, AZ_HB:, and AZ_ACK:
prefixed lines emitted by the AlphaZero Lua mod's print-based IPC fallback.

The Lua mod prints game state, heartbeats, and acknowledgments to stdout which
Civ6 redirects to Lua.log. This module tails that file asynchronously and
provides structured access to the parsed data.

Usage:
    tailer = LogTailer(Path(r'C:/Users/.../Logs/Lua.log'))
    async with tailer:
        state = await tailer.wait_for_state(timeout=30.0)
        ack = await tailer.wait_for_ack(msg_id=1, timeout=10.0)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "LogTailer",
    "LogTailerError",
    "StateTimeoutError",
    "AckTimeoutError",
]

# --- Line prefix patterns ---
# Civ6 prepends the context name, e.g. "AlphaZeroMain: AZ_STATE:{json}"
_PREFIX_RE = re.compile(r"^(?:AlphaZeroMain:\s*)?")
_STATE_PREFIX = "AZ_STATE:"
_HB_PREFIX = "AZ_HB:"
_ACK_PREFIX = "AZ_ACK:"
_CHUNK_PREFIX = "AZ_STATE_CHUNK:"
_CHUNK_END_PREFIX = "AZ_STATE_END:"

# Unicode escape pattern for Lua JSON encoder output
_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")


# --- Exceptions ---


class LogTailerError(Exception):
    """Base exception for log tailer errors."""


class StateTimeoutError(LogTailerError):
    """Raised when waiting for state times out."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(f"Timed out waiting for game state after {timeout:.1f}s")


class AckTimeoutError(LogTailerError):
    """Raised when waiting for ack times out."""

    def __init__(self, msg_id: int, timeout: float) -> None:
        self.msg_id = msg_id
        self.timeout = timeout
        super().__init__(
            f"Timed out waiting for ack (msg_id={msg_id}) after {timeout:.1f}s"
        )


# --- Stats ---


@dataclass
class TailerStats:
    """Statistics for log tailer operation."""

    states_received: int = 0
    acks_received: int = 0
    heartbeats_received: int = 0
    chunks_received: int = 0
    parse_errors: int = 0
    lines_processed: int = 0


# --- Main class ---


class LogTailer:
    """Asynchronous Lua.log tailer for Civ6 print-based IPC.

    Tails Lua.log seeking to end on start, parses AZ_STATE/AZ_HB/AZ_ACK
    prefixed lines, and provides async wait methods for state and ack data.

    Supports chunked state output (AZ_STATE_CHUNK:i/n:data) for large
    game states that exceed Lua.log's ~4KB line limit.

    Args:
        log_path: Absolute path to Civ6 Lua.log file.
        poll_interval: Seconds between file poll attempts.
    """

    def __init__(self, log_path: Path, poll_interval: float = 0.1) -> None:
        self._log_path = Path(log_path)
        self._poll_interval = poll_interval
        self._running = False
        self._poll_task: asyncio.Task[None] | None = None
        self._file: Any = None
        self._file_pos: int = 0

        # Latest parsed data
        self._latest_state: dict[str, Any] | None = None
        self._latest_heartbeat: dict[str, Any] | None = None
        self._heartbeat_time: float = 0.0

        # Event for state arrival notification
        self._state_event = asyncio.Event()
        # Ack storage keyed by msg_id
        self._ack_queue: dict[int, dict[str, Any]] = {}
        self._ack_events: dict[int, asyncio.Event] = {}

        # Chunk reassembly buffer
        self._chunk_buffer: dict[int, str] = {}  # seq -> data
        self._chunk_total: int = 0

        # Stats
        self._stats = TailerStats()

    # --- Context manager ---

    async def __aenter__(self) -> LogTailer:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    # --- Lifecycle ---

    async def start(self) -> None:
        """Open log file and start polling for new lines.

        If the file doesn't exist yet, waits up to 30s for it to appear.
        Seeks to end of file so only new lines are processed.
        """
        if self._running:
            return

        # Wait for file to exist
        wait_start = time.monotonic()
        while not self._log_path.exists():
            if time.monotonic() - wait_start > 30.0:
                raise LogTailerError(
                    f"Log file not found after 30s: {self._log_path}"
                )
            logger.debug("Waiting for log file: %s", self._log_path)
            await asyncio.sleep(1.0)

        # Open and seek to end
        self._file = open(self._log_path, "r", encoding="utf-8", errors="replace")
        self._file.seek(0, os.SEEK_END)
        self._file_pos = self._file.tell()
        self._running = True

        # Start poll loop
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name="log_tailer_poll"
        )
        logger.info("LogTailer started on %s (pos=%d)", self._log_path, self._file_pos)

    async def stop(self) -> None:
        """Stop polling and close the log file."""
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._file:
            self._file.close()
            self._file = None
        logger.info(
            "LogTailer stopped. Stats: %d states, %d acks, %d HBs, %d errors",
            self._stats.states_received,
            self._stats.acks_received,
            self._stats.heartbeats_received,
            self._stats.parse_errors,
        )

    # --- Polling ---

    async def _poll_loop(self) -> None:
        """Main polling loop: read new lines and parse them."""
        while self._running:
            try:
                self._read_new_lines()
            except Exception:
                logger.exception("Error reading log lines")
            await asyncio.sleep(self._poll_interval)

    def _read_new_lines(self) -> None:
        """Read any new lines appended to the log file."""
        if not self._file:
            return

        # Detect file truncation (e.g. game restart)
        try:
            file_size = os.path.getsize(self._log_path)
        except OSError:
            return

        if file_size < self._file_pos:
            logger.info(
                "Log file truncated (size=%d < pos=%d), seeking to start",
                file_size,
                self._file_pos,
            )
            self._file.seek(0)
            self._file_pos = 0

        # Read new lines
        for line in self._file:
            line = line.rstrip("\n\r")
            if line:
                self._stats.lines_processed += 1
                self._parse_line(line)

        self._file_pos = self._file.tell()

    # --- Line parsing ---

    def _parse_line(self, line: str) -> None:
        """Parse a single log line for AZ_ prefixed messages."""
        # Strip Civ6 context prefix (e.g. "AlphaZeroMain: ")
        content = _PREFIX_RE.sub("", line).strip()

        if content.startswith(_STATE_PREFIX):
            self._handle_state(content[len(_STATE_PREFIX) :])
        elif content.startswith(_CHUNK_PREFIX):
            self._handle_chunk(content[len(_CHUNK_PREFIX) :])
        elif content.startswith(_CHUNK_END_PREFIX):
            self._handle_chunk_end(content[len(_CHUNK_END_PREFIX) :])
        elif content.startswith(_HB_PREFIX):
            self._handle_heartbeat(content[len(_HB_PREFIX) :])
        elif content.startswith(_ACK_PREFIX):
            self._handle_ack(content[len(_ACK_PREFIX) :])

    def _handle_state(self, json_str: str) -> None:
        """Parse a complete (non-chunked) state JSON."""
        data = self._parse_json(json_str)
        if data is not None:
            self._latest_state = data
            self._stats.states_received += 1
            self._state_event.set()
            logger.debug(
                "State received: turn=%s, tiles=%s",
                data.get("turn_id", "?"),
                len(data.get("map", {}).get("tiles", [])),
            )

    def _handle_chunk(self, chunk_str: str) -> None:
        """Parse a state chunk: seq/total:data."""
        # Format: "1/5:json_data_here"
        match = re.match(r"(\d+)/(\d+):(.*)", chunk_str, re.DOTALL)
        if not match:
            logger.warning("Malformed chunk header: %.100s", chunk_str)
            self._stats.parse_errors += 1
            return

        seq = int(match.group(1))
        total = int(match.group(2))
        data = match.group(3)

        self._chunk_total = total
        self._chunk_buffer[seq] = data
        self._stats.chunks_received += 1

    def _handle_chunk_end(self, total_str: str) -> None:
        """Reassemble chunks on receiving AZ_STATE_END."""
        try:
            expected_total = int(total_str.strip())
        except ValueError:
            logger.warning("Invalid chunk end total: %s", total_str)
            self._stats.parse_errors += 1
            return

        if len(self._chunk_buffer) < expected_total:
            logger.warning(
                "Incomplete chunks: got %d/%d",
                len(self._chunk_buffer),
                expected_total,
            )

        # Reassemble in order
        parts = []
        for i in range(1, expected_total + 1):
            if i in self._chunk_buffer:
                parts.append(self._chunk_buffer[i])
            else:
                logger.warning("Missing chunk %d/%d", i, expected_total)

        full_json = "".join(parts)
        self._chunk_buffer.clear()
        self._chunk_total = 0

        data = self._parse_json(full_json)
        if data is not None:
            self._latest_state = data
            self._stats.states_received += 1
            self._state_event.set()
            logger.info(
                "Chunked state reassembled (%d chunks, %d bytes): turn=%s",
                expected_total,
                len(full_json),
                data.get("turn_id", "?"),
            )

    def _handle_heartbeat(self, json_str: str) -> None:
        """Parse a heartbeat JSON."""
        data = self._parse_json(json_str)
        if data is not None:
            self._latest_heartbeat = data
            self._heartbeat_time = time.monotonic()
            self._stats.heartbeats_received += 1

    def _handle_ack(self, json_str: str) -> None:
        """Parse an ack JSON and notify waiters."""
        data = self._parse_json(json_str)
        if data is not None:
            msg_id = data.get("msg_id")
            self._ack_queue[msg_id] = data
            self._stats.acks_received += 1

            # Notify any waiter for this msg_id
            if msg_id in self._ack_events:
                self._ack_events[msg_id].set()

            logger.debug("Ack received: msg_id=%s success=%s", msg_id, data.get("success"))

    def _parse_json(self, json_str: str) -> dict[str, Any] | None:
        """Parse JSON string with unicode unescape handling.

        The Lua JSON encoder produces \\uXXXX escapes for all string
        characters. Python's json.loads handles these natively.
        """
        if not json_str:
            return None
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to handle truncated JSON gracefully
            logger.warning("JSON parse error (len=%d): %s", len(json_str), e)
            self._stats.parse_errors += 1
            return None

    # --- Public query methods ---

    async def wait_for_state(self, timeout: float = 30.0) -> dict[str, Any]:
        """Wait for the next game state to arrive.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            Parsed game state dictionary.

        Raises:
            StateTimeoutError: If no state arrives within timeout.
        """
        self._state_event.clear()
        try:
            await asyncio.wait_for(self._state_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise StateTimeoutError(timeout) from None

        if self._latest_state is None:
            raise StateTimeoutError(timeout)
        return self._latest_state

    async def wait_for_ack(
        self, msg_id: int, timeout: float = 10.0
    ) -> dict[str, Any]:
        """Wait for an acknowledgment with a specific message ID.

        Args:
            msg_id: The command message ID to wait for.
            timeout: Maximum seconds to wait.

        Returns:
            Parsed ack dictionary.

        Raises:
            AckTimeoutError: If no matching ack arrives within timeout.
        """
        # Check if already received
        if msg_id in self._ack_queue:
            return self._ack_queue.pop(msg_id)

        # Create event and wait
        event = asyncio.Event()
        self._ack_events[msg_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise AckTimeoutError(msg_id, timeout) from None
        finally:
            self._ack_events.pop(msg_id, None)

        return self._ack_queue.pop(msg_id, {"msg_id": msg_id, "success": False})

    def check_heartbeat(self, max_age: float = 5.0) -> bool:
        """Check if a recent heartbeat has been received.

        Args:
            max_age: Maximum age in seconds for a heartbeat to be valid.

        Returns:
            True if a heartbeat was received within max_age seconds.
        """
        if self._heartbeat_time == 0.0:
            return False
        return (time.monotonic() - self._heartbeat_time) < max_age

    @property
    def latest_state(self) -> dict[str, Any] | None:
        """The most recently received game state, or None."""
        return self._latest_state

    @property
    def latest_heartbeat(self) -> dict[str, Any] | None:
        """The most recently received heartbeat, or None."""
        return self._latest_heartbeat

    @property
    def stats(self) -> dict[str, int]:
        """Return tailer statistics as a dictionary."""
        return {
            "states_received": self._stats.states_received,
            "acks_received": self._stats.acks_received,
            "heartbeats_received": self._stats.heartbeats_received,
            "chunks_received": self._stats.chunks_received,
            "parse_errors": self._stats.parse_errors,
            "lines_processed": self._stats.lines_processed,
        }

    @property
    def is_running(self) -> bool:
        """Whether the tailer is currently running."""
        return self._running
