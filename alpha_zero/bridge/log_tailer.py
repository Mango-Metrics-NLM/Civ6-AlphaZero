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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

__all__ = [
    "DROPPED_MISSING_PIECE",
    "UNREADABLE_STATE",
    "AckTimeoutError",
    "LogTailer",
    "LogTailerError",
    "StateTimeoutError",
    "iter_log_states",
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
        super().__init__(f"Timed out waiting for ack (msg_id={msg_id}) after {timeout:.1f}s")


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
    #: States that arrived and were replaced by a newer one before anyone
    #: asked for them.
    states_skipped: int = 0
    #: Acks nobody waited for, dropped to keep the unread set bounded.
    acks_dropped: int = 0
    #: States taken from the log as it stood at start (a start in mid-turn).
    backlog_states_used: int = 0


def _turn_of(state: dict[str, Any]) -> Any:
    """The game turn of a state message, wrapped (``{"state": {...}}``) or bare."""
    inner = state.get("state")
    if isinstance(inner, dict) and "turn_id" in inner:
        return inner["turn_id"]
    return state.get("turn_id")


# --- Main class ---


class LogTailer:
    """Asynchronous Lua.log tailer for Civ6 print-based IPC.

    Tails Lua.log seeking to end on start, parses AZ_STATE/AZ_HB/AZ_ACK
    prefixed lines, and provides async wait methods for state and ack data.

    Supports chunked state output (AZ_STATE_CHUNK:i/n:data) for large
    game states that exceed Lua.log's ~4KB line limit.

    Every complete state gets a sequence number. :meth:`wait_for_state` hands
    out the newest state that has not been handed out yet, at once when one is
    already there. (It used to clear its event and then wait, so a state read
    in the same poll batch as the ack before it was thrown away and the caller
    waited for a state that had already come.)

    Args:
        log_path: Absolute path to Civ6 Lua.log file.
        poll_interval: Seconds between file poll attempts.
        backlog_bytes: How much of the end of the existing log ``start`` scans
            for the last complete state, so a start in the middle of a turn
            can see the state the mod already wrote. ``0`` turns it off.
    """

    #: Unread acks kept; fire-and-forget commands are never waited for.
    MAX_UNREAD_ACKS = 256

    def __init__(
        self,
        log_path: Path,
        poll_interval: float = 0.1,
        backlog_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self._log_path = Path(log_path)
        self._poll_interval = poll_interval
        self._backlog_bytes = max(0, int(backlog_bytes))
        self._running = False
        self._poll_task: asyncio.Task[None] | None = None
        self._file: Any = None
        self._file_pos: int = 0

        # Latest parsed data
        self._latest_state: dict[str, Any] | None = None
        # States made available so far, and the number of the last one handed
        # to a caller. A state is new to the caller while the first is larger.
        self._state_seq: int = 0
        self._handed_seq: int = 0
        # The last complete state in the log as it stood at start, until a
        # live heartbeat shows the mod is listening (then it is made
        # available) or a live state replaces it.
        self._backlog_state: dict[str, Any] | None = None
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
                raise LogTailerError(f"Log file not found after 30s: {self._log_path}")
            logger.debug("Waiting for log file: %s", self._log_path)
            await asyncio.sleep(1.0)

        # Open and seek to end
        # ``surrogateescape`` keeps the bytes of a character the mod split
        # across two chunk lines, so the pieces can be joined and decoded as
        # one (see ``_parse_json``); ``replace`` destroyed both halves.
        self._file = open(self._log_path, encoding="utf-8", errors="surrogateescape")
        # A restart is a new session: nothing read before it is current.
        self._forget_session()
        # The live tail starts exactly where the backlog scan stopped, so a
        # line written in between is in one of the two, never in neither.
        self._backlog_state, scanned_to = self._scan_backlog()
        self._file.seek(scanned_to)
        self._file_pos = self._file.tell()
        self._running = True

        # Start poll loop
        self._poll_task = asyncio.create_task(self._poll_loop(), name="log_tailer_poll")
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
            # A new log is a new game session: nothing read from the old one
            # is current. That covers the state found at start and half-read
            # pieces, and also a state already published (handed out or not)
            # and the old game's heartbeat.
            self._backlog_state = None
            self._forget_session()

        # Read new lines with readline() in a loop rather than
        # ``for line in self._file``: the file-object iterator keeps an
        # internal read-ahead buffer and caches EOF, so once it first hits the
        # end of the log it can stop yielding lines appended on later polls —
        # silently stalling the live tail. readline() re-checks EOF each call.
        while True:
            line = self._file.readline()
            if not line:
                break
            stripped = line.rstrip("\n\r")
            if stripped:
                self._stats.lines_processed += 1
                self._parse_line(stripped)

        self._file_pos = self._file.tell()

    # --- Line parsing ---

    def _parse_line(self, line: str) -> None:
        """Parse a single log line for AZ_ prefixed messages."""
        # Strip Civ6 context prefix (e.g. "AlphaZeroMain: ")
        unstripped = _PREFIX_RE.sub("", line.lstrip())
        content = unstripped.strip()

        if content.startswith(_STATE_PREFIX):
            self._handle_state(content[len(_STATE_PREFIX) :])
        elif content.startswith(_CHUNK_PREFIX):
            # A piece is cut anywhere, also next to a space inside a string:
            # its own leading and trailing spaces are data.
            self._handle_chunk(unstripped[len(_CHUNK_PREFIX) :])
        elif content.startswith(_CHUNK_END_PREFIX):
            self._handle_chunk_end(content[len(_CHUNK_END_PREFIX) :])
        elif content.startswith(_HB_PREFIX):
            self._handle_heartbeat(content[len(_HB_PREFIX) :])
        elif content.startswith(_ACK_PREFIX):
            self._handle_ack(content[len(_ACK_PREFIX) :])

    def _handle_state(self, json_str: str) -> None:
        """Parse a complete (non-chunked) state JSON."""
        self._backlog_state = None  # a newer state, readable or not
        data = self._parse_json(json_str)
        if data is not None:
            self._stats.states_received += 1
            self._publish_state(data)
            logger.debug(
                "State received: turn=%s, tiles=%s",
                data.get("turn_id", "?"),
                len(data.get("map", {}).get("tiles", [])),
            )

    def _forget_session(self) -> None:
        """Drop everything read so far: published state, cursors, pieces, heartbeat."""
        self._latest_state = None
        self._state_seq = self._handed_seq = 0
        self._latest_heartbeat = None
        self._heartbeat_time = 0.0
        self._chunk_buffer.clear()
        self._chunk_total = 0
        self._state_event.clear()

    def _publish_state(self, data: dict[str, Any]) -> None:
        """Make ``data`` the newest state and wake whoever waits for one."""
        self._latest_state = data
        self._state_seq += 1
        # A state read from the live log is newer than anything in the backlog.
        self._backlog_state = None
        self._state_event.set()

    def _scan_backlog(self) -> tuple[dict[str, Any] | None, int]:
        """The last complete state in the existing log, and where the scan ended.

        Only states are read: an old ack answers nothing and an old heartbeat
        must not make the mod look alive. The state is not used when the log
        ends inside a state sent in pieces (a newer one is being written) or
        when the newest state in it could not be read.

        Returns:
            ``(state or None, byte offset the live tail must start from)``.
        """
        try:
            size = os.path.getsize(self._log_path)
        except OSError:
            return None, 0
        if not self._backlog_bytes:
            return None, size
        try:
            with open(self._log_path, "rb") as handle:
                start = max(0, size - self._backlog_bytes)
                handle.seek(start)
                raw = handle.read(size - start)
        except OSError:
            return None, size
        # Only complete lines: a last line still being written belongs to the
        # live tail. ``split("\n")``, as the live path splits; ``splitlines``
        # would also cut at U+2028 inside a name.
        end = raw.rfind(b"\n") + 1
        scanned_to = start + end
        lines = raw[:end].decode("utf-8", errors="surrogateescape").split("\n")
        if start > 0 and lines:
            lines = lines[1:]  # the scan began inside a line
        reader = LogTailer(self._log_path, backlog_bytes=0)
        newest_unreadable = False
        for line in lines:
            line = line.rstrip("\r")
            content = _PREFIX_RE.sub("", line.strip()).strip()
            if not content.startswith((_STATE_PREFIX, _CHUNK_PREFIX, _CHUNK_END_PREFIX)):
                continue
            before = reader._state_seq
            try:
                reader._parse_line(line)
            except Exception:  # a damaged line is not worth failing start for
                logger.debug("Unreadable backlog line skipped", exc_info=True)
            if not content.startswith(_CHUNK_PREFIX):
                newest_unreadable = reader._state_seq == before
        if reader._chunk_buffer:
            # The log ends inside a state sent in pieces. Its first pieces are
            # behind the point the live tail starts from, so they are handed
            # over: the rest then completes the state instead of being joined
            # with nothing. The older state is not the current one.
            self._chunk_buffer = dict(reader._chunk_buffer)
            self._chunk_total = reader._chunk_total
            return None, scanned_to
        if newest_unreadable or reader._latest_state is None:
            return None, scanned_to
        logger.info(
            "A state from before start is in the log (turn=%s); it is used once a "
            "heartbeat for that turn shows the mod is listening",
            _turn_of(reader._latest_state),
        )
        return reader._latest_state, scanned_to

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
        # A newer state was written, readable or not: the one from before
        # start is no longer the current one.
        self._backlog_state = None

        data = self._parse_json(full_json)
        if data is not None:
            self._stats.states_received += 1
            self._publish_state(data)
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
            backlog = self._backlog_state
            if backlog is not None:
                backlog_turn = _turn_of(backlog)
                if backlog_turn is not None and data.get("turn") == backlog_turn:
                    # The mod is alive and on the turn of the state it wrote
                    # before we started. (A heartbeat alone is not enough: the mod
                    # also writes one when a game loads.) A reload of the same
                    # save at the same turn cannot be told apart from here.
                    self._stats.backlog_states_used += 1
                    self._publish_state(backlog)

    def _handle_ack(self, json_str: str) -> None:
        """Parse an ack JSON and notify waiters."""
        data = self._parse_json(json_str)
        if data is None:
            return

        msg_id = data.get("msg_id")
        # A malformed/truncated ack (e.g. from the same live-write race that
        # produces partial lines) can be missing msg_id or carry a non-int
        # value. Filing it under that key would be unreachable to any
        # waiter, which is keyed by the real integer msg_id in
        # wait_for_ack(), silently starving the in-flight command until it
        # times out. Log and drop instead of queuing an unmatchable ack.
        if not isinstance(msg_id, int) or isinstance(msg_id, bool):
            logger.warning("Dropping ack with invalid msg_id=%r: %s", msg_id, data)
            return

        self._ack_queue[msg_id] = data
        self._stats.acks_received += 1
        self._drop_old_unread_acks()

        # Notify any waiter for this msg_id
        if msg_id in self._ack_events:
            self._ack_events[msg_id].set()

        logger.debug("Ack received: msg_id=%s success=%s", msg_id, data.get("success"))

    def _drop_old_unread_acks(self) -> None:
        """Keep at most ``MAX_UNREAD_ACKS`` acks nobody is waiting for."""
        unread = [key for key in self._ack_queue if key not in self._ack_events]
        for key in unread[: max(0, len(unread) - self.MAX_UNREAD_ACKS)]:
            del self._ack_queue[key]
            self._stats.acks_dropped += 1

    def _parse_json(self, json_str: str) -> dict[str, Any] | None:
        """Parse JSON string with unicode unescape handling.

        The Lua JSON encoder produces \\uXXXX escapes for all string
        characters. Python's json.loads handles these natively.
        """
        if not json_str:
            return None
        if not json_str.isascii():
            # Bytes that did not decode alone (half of a character split across
            # two chunk lines) were kept as surrogates; now that the pieces are
            # joined, decode them together. What is still not UTF-8 is replaced.
            json_str = json_str.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
        try:
            # json.loads returns Any; bridge messages are always JSON objects.
            return cast("dict[str, Any]", json.loads(json_str))
        except json.JSONDecodeError as e:
            # Try to handle truncated JSON gracefully
            logger.warning("JSON parse error (len=%d): %s", len(json_str), e)
            self._stats.parse_errors += 1
            return None

    # --- Public query methods ---

    async def wait_for_state(self, timeout: float = 30.0) -> dict[str, Any]:
        """The newest state not handed out yet; waits for one if there is none.

        Returns at once when a state has arrived since the last hand-out.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            Parsed game state dictionary.

        Raises:
            StateTimeoutError: If no state arrives within timeout.
        """
        deadline = time.monotonic() + timeout
        while self._state_seq <= self._handed_seq:
            # Nothing unseen. No await sits between this check and the clear,
            # so a state cannot slip in between and be lost. A loop, because
            # another caller woken by the same state may have taken it.
            self._state_event.clear()
            try:
                await asyncio.wait_for(
                    self._state_event.wait(), timeout=max(0.0, deadline - time.monotonic())
                )
            except TimeoutError:
                raise StateTimeoutError(timeout) from None
            if self._latest_state is None:
                raise StateTimeoutError(timeout)

        state = self._hand_out()
        if state is None:
            raise StateTimeoutError(timeout)
        return state

    def current_state(self) -> dict[str, Any] | None:
        """The latest state, marked as handed out; None when there is none."""
        return self._hand_out()

    def _hand_out(self) -> dict[str, Any] | None:
        if self._latest_state is None:
            return None
        skipped = self._state_seq - self._handed_seq - 1
        if skipped > 0:
            self._stats.states_skipped += skipped
            logger.warning("%d state(s) were replaced by a newer one before being read", skipped)
        self._handed_seq = self._state_seq
        return self._latest_state

    async def wait_for_ack(self, msg_id: int, timeout: float = 10.0) -> dict[str, Any]:
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
        except TimeoutError:
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
    def state_seq(self) -> int:
        """How many states have been made available so far."""
        return self._state_seq

    @property
    def backlog_state(self) -> dict[str, Any] | None:
        """The state found in the log at start, while it is still pending."""
        return self._backlog_state

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
            "states_skipped": self._stats.states_skipped,
            "acks_dropped": self._stats.acks_dropped,
            "backlog_states_used": self._stats.backlog_states_used,
        }

    @property
    def is_running(self) -> bool:
        """Whether the tailer is currently running."""
        return self._running


#: Keys of the counts :func:`iter_log_states` keeps of states it did not yield.
DROPPED_MISSING_PIECE = "states_dropped_missing_piece"
UNREADABLE_STATE = "states_unreadable"
#: Piece numbers and totals in the offline reader: ASCII, at most nine digits.
_PIECE_HEADER_RE = re.compile(r"([0-9]{1,9})/([0-9]{1,9}):")
_PIECE_NUMBER_RE = re.compile(r"[0-9]{1,9}")


def iter_log_states(
    log_path: Path | str, counts: dict[str, int] | None = None
) -> Iterator[dict[str, Any]]:
    """Every complete state in a ``Lua.log``, oldest first.

    Offline reading for analysis: nothing is tailed and nothing waits. Lines go
    through :class:`LogTailer`'s own parser, but a state sent in pieces is held
    to a stricter rule than the live transport's, which joins whatever it has.
    For a measurement an altered state is worse than a missing one, so a pieced
    state is yielded only when its pieces are exactly ``1..n`` of its end line.
    An export is given up, and counted, when:

    * its end line arrives and a piece is missing, or the totals disagree;
    * a piece arrives whose number is not above the previous piece's, or whose
      total differs (the next export has begun: the old one lost its end line);
    * a whole ``AZ_STATE`` line arrives, or the log ends, in the middle of it.

    What this cannot see, because pieces carry no export id: an export that
    loses its last pieces and its end line, followed by one with the same piece
    count that loses exactly its first pieces up to the same number. Their
    pieces still number ``1..n`` in order; the joined text is yielded only if
    it also happens to parse.

    An end line's number is compared with the count of pieces held before
    anything of its size is built, and piece numbers longer than nine digits are
    not stored: a number in the log cannot make the reader allocate.

    A state whose text is not a JSON object (``null``, ``[]``, ``3``, broken
    JSON), or on which the live handler raises (``AttributeError`` or
    ``TypeError``: it reads ``map`` and ``map.tiles`` without checking their
    types), is not yielded, is counted, and the reading goes on. Acks and
    heartbeats are not read.

    Args:
        log_path: The log file.
        counts: Optional dict, raised by one under ``DROPPED_MISSING_PIECE``
            for each export given up and under ``UNREADABLE_STATE`` for each
            state whose text was not a JSON object the live reader takes.

    Yields:
        Each state as the mod wrote it (wrapped as ``{"state": {...}}`` or bare).

    Raises:
        OSError: If the file cannot be opened or read.
    """
    reader = LogTailer(Path(log_path), backlog_bytes=0)
    last_piece: tuple[int, int] | None = None  # (number, total) of the piece before

    def give_up(key: str = DROPPED_MISSING_PIECE) -> None:
        reader._chunk_buffer.clear()
        reader._chunk_total = 0
        if counts is not None:
            counts[key] = counts.get(key, 0) + 1

    def is_object(text: str) -> bool:
        # The live reader's own decoder (it re-joins a character split across pieces).
        try:
            return isinstance(reader._parse_json(text), dict)
        except RecursionError:  # nested deeper than the decoder goes
            return False

    with open(log_path, encoding="utf-8", errors="surrogateescape") as handle:
        for line in handle:
            stripped = line.rstrip("\n\r")
            content = _PREFIX_RE.sub("", stripped.lstrip())
            if content.startswith(_CHUNK_PREFIX):
                header = _PIECE_HEADER_RE.match(content[len(_CHUNK_PREFIX) :])
                piece = (int(header.group(1)), int(header.group(2))) if header else None
                if last_piece and (
                    not piece or piece[0] <= last_piece[0] or piece[1:] != last_piece[1:]
                ):
                    give_up()
                last_piece = piece
                if piece is not None:  # a header the bounded pattern refuses is not stored
                    reader._parse_line(stripped)
                continue
            if not content.startswith((_STATE_PREFIX, _CHUNK_END_PREFIX)):
                continue
            if content.startswith(_CHUNK_END_PREFIX):
                total = content[len(_CHUNK_END_PREFIX) :].strip()
                held = reader._chunk_buffer
                # The end line's number is checked against what is held before anything
                # of its size is built: a number above the pieces held is a missing piece.
                n = int(total) if _PIECE_NUMBER_RE.fullmatch(total) else -1
                whole = n == len(held) == reader._chunk_total and set(held) == set(range(1, n + 1))
                last_piece = None
                if not whole:
                    give_up()
                    continue
                payload = "".join(held[i] for i in range(1, n + 1))
            else:
                if last_piece is not None:  # a whole state in the middle of a pieced one
                    give_up()
                    last_piece = None
                payload = content.strip()[len(_STATE_PREFIX) :]
            # Offline only: a payload that is not a JSON object, or that the live
            # handler raises on (it reads ``map`` and ``map.tiles`` without checking
            # their types), is counted and the reading goes on. A whole line and a
            # joined one go through the same handler, so they are judged alike.
            reader._chunk_buffer.clear()
            reader._chunk_total = 0
            if not is_object(payload):
                give_up(UNREADABLE_STATE)
                continue
            before = reader.state_seq
            try:
                reader._handle_state(payload)
            except (AttributeError, TypeError):
                give_up(UNREADABLE_STATE)
                continue
            if reader.state_seq > before and reader.latest_state is not None:
                yield reader.latest_state
            else:
                give_up(UNREADABLE_STATE)
    if last_piece is not None:  # the log ends inside a pieced state
        give_up()
