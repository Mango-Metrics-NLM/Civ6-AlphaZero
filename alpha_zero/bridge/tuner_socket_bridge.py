"""Headless FireTuner TCP client — direct Python→Civ6 command delivery.

The Civ6 Lua sandbox exposes no ``io`` library, so the AlphaZeroAI mod
cannot read ``commands.json`` itself. The original design bridged that gap
with the FireTuner GUI plus an ``.ltp`` panel that polled the file and set
``ExposedMembers.AlphaZeroCommand``. This module removes both the file hop
and the GUI: it speaks the FireTuner wire protocol directly to the running
game (TCP server on ``127.0.0.1:4318``) and executes, in the
``GameCore_Tuner`` Lua context::

    ExposedMembers.AlphaZeroCommand = <command-json-as-lua-long-string>

``ExposedMembers`` is shared across every Civ6 Lua state, so the mod's
gameplay-context poller reads it on its next cycle. Acknowledgements still
travel out through ``Lua.log`` (``AZ_ACK:``) and are read by ``LogTailer``,
so this transport only replaces the *command* direction.

Wire protocol (reverse-engineered; verified live against Civ6 build):
    * Each message is an 8-byte header + NUL-terminated UTF-8 payload.
      ``header = struct.pack("<Ii", payload_len_incl_nul, tag)``.
    * ``tag=3`` executes ``CMD:<state_index>:<lua>``; ``tag=4`` is a query
      (``APP:`` identity, ``LSQ:`` state list).
    * ``LSQ:`` replies with NUL-separated ``(index, name)`` pairs — the CMD
      state index is the *explicit* number, not the list position.
    * Output frames come back with ``tag=-1`` as ``O\\0<ctx>: <text>``; a
      command completes with a ``tag=3`` empty (NUL-only) frame.

``SocketTunerBridge`` mirrors :class:`~alpha_zero.bridge.tuner_bridge.TunerBridge`'s
async surface (``start``/``stop``/``send_command``/``stats``) so
:class:`~alpha_zero.bridge.live_bridge.LiveBridge` can select it by config.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import socket
import struct
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from alpha_zero.bridge.config import BridgeConfig
    from alpha_zero.bridge.protocol import CommandMessage

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_APP_MAX_DURATION_S",
    "DEFAULT_APP_QUIET_AFTER_S",
    "DEFAULT_APP_WAIT_FIRST_S",
    "DEFAULT_HANDSHAKE_QUIET_S",
    "DEFAULT_INITIAL_DRAIN_CAP_S",
    "DEFAULT_INITIAL_DRAIN_QUIET_S",
    "DEFAULT_LSQ_MAX_DURATION_S",
    "DEFAULT_LSQ_QUIET_AFTER_S",
    "DEFAULT_LSQ_WAIT_FIRST_S",
    "DEFAULT_TUNER_HOST",
    "DEFAULT_TUNER_PORT",
    "ENV_TUNER_HANDSHAKE_QUIET_S",
    "FireTunerClient",
    "FireTunerError",
    "FireTunerTimingConfig",
    "SocketTunerBridge",
    "lua_longstring",
]

# --- Wire-protocol constants -------------------------------------------------

#: ``struct`` format for the frame header: uint32 LE payload length (including
#: the trailing NUL) followed by an int32 LE tag.
HEADER_FMT = "<Ii"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 8

TAG_HELP = 1
TAG_COMMAND = 3  # execute code / command-complete sentinel
TAG_QUERY = 4  # APP: / LSQ: handshake queries
TAG_OUTPUT = -1  # server → client print() output frames

#: Prefix (with an internal NUL) marking a ``tag=-1`` print() output frame:
#: ``O\x00<ctx>: <text>``. The trailing NUL is stripped before parsing, but
#: this leading one survives, so it distinguishes wrapped output from a bare
#: error frame.
OUTPUT_FRAME_PREFIX = "O\x00"

#: Marker beginning a Lua execution-error message (bare ``ERR:...`` frame or
#: wrapped as ``O\x00<ctx>: ERR:...``). Detected after prefix normalisation.
ERROR_PREFIX = "ERR:"

#: Default FireTuner endpoint (the game is the listener).
DEFAULT_TUNER_HOST = "127.0.0.1"
DEFAULT_TUNER_PORT = 4318

#: Named defaults for FireTuner handshake windows (seconds).
DEFAULT_HANDSHAKE_QUIET_S = 1.0
DEFAULT_INITIAL_DRAIN_QUIET_S = 0.2
DEFAULT_INITIAL_DRAIN_CAP_S = 0.25
DEFAULT_APP_WAIT_FIRST_S = 2.0
DEFAULT_APP_QUIET_AFTER_S = 0.15
DEFAULT_APP_MAX_DURATION_S = 3.0
DEFAULT_LSQ_WAIT_FIRST_S = 2.5
DEFAULT_LSQ_QUIET_AFTER_S = 0.25
DEFAULT_LSQ_MAX_DURATION_S = 5.0
ENV_TUNER_HANDSHAKE_QUIET_S = "CIV6_BRIDGE_TUNER_HANDSHAKE_QUIET_S"

#: Lua contexts to target for the command global, most-preferred first. The
#: proven ``.ltp`` panel used ``GameCore_Tuner``; ``InGame`` is the fallback.
PREFERRED_STATES = ("GameCore_Tuner", "InGame")


def _validate_positive_finite(name: str, value: float) -> float:
    """Validate that ``value`` is strictly positive and finite."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


@dataclass(frozen=True)
class FireTunerTimingConfig:
    """Validated monotonic timing windows for the FireTuner TCP handshake.

    All durations are in seconds and must be strictly positive and finite. Allows
    overriding quiet/cap windows dynamically without hardcoded inline magic constants.
    """

    initial_drain_quiet_s: float = DEFAULT_INITIAL_DRAIN_QUIET_S
    initial_drain_cap_s: float = DEFAULT_INITIAL_DRAIN_CAP_S
    app_wait_first_s: float = DEFAULT_APP_WAIT_FIRST_S
    app_quiet_after_s: float = DEFAULT_APP_QUIET_AFTER_S
    app_max_duration_s: float = DEFAULT_APP_MAX_DURATION_S
    lsq_wait_first_s: float = DEFAULT_LSQ_WAIT_FIRST_S
    lsq_quiet_after_s: float = DEFAULT_LSQ_QUIET_AFTER_S
    lsq_max_duration_s: float = DEFAULT_LSQ_MAX_DURATION_S

    def __post_init__(self) -> None:
        for field_name in (
            "initial_drain_quiet_s",
            "initial_drain_cap_s",
            "app_wait_first_s",
            "app_quiet_after_s",
            "app_max_duration_s",
            "lsq_wait_first_s",
            "lsq_quiet_after_s",
            "lsq_max_duration_s",
        ):
            _validate_positive_finite(field_name, getattr(self, field_name))

    @classmethod
    def from_handshake_quiet(cls, handshake_quiet: float, timeout: float) -> FireTunerTimingConfig:
        """Derive consistent handshake windows from a base quiet window and socket timeout."""
        _validate_positive_finite("handshake_quiet", handshake_quiet)
        _validate_positive_finite("timeout", timeout)
        return cls(
            initial_drain_quiet_s=min(handshake_quiet, DEFAULT_INITIAL_DRAIN_QUIET_S),
            initial_drain_cap_s=min(handshake_quiet, DEFAULT_INITIAL_DRAIN_CAP_S),
            app_wait_first_s=max(handshake_quiet, min(timeout, DEFAULT_APP_WAIT_FIRST_S)),
            app_quiet_after_s=min(handshake_quiet, DEFAULT_APP_QUIET_AFTER_S),
            app_max_duration_s=max(handshake_quiet * 2.0, min(timeout, DEFAULT_APP_MAX_DURATION_S)),
            lsq_wait_first_s=max(handshake_quiet * 1.5, min(timeout, DEFAULT_LSQ_WAIT_FIRST_S)),
            lsq_quiet_after_s=min(handshake_quiet, DEFAULT_LSQ_QUIET_AFTER_S),
            lsq_max_duration_s=max(handshake_quiet * 3.0, min(timeout, DEFAULT_LSQ_MAX_DURATION_S)),
        )

    @classmethod
    def from_env(
        cls,
        *,
        handshake_quiet: float | None = None,
        timeout: float = 5.0,
    ) -> FireTunerTimingConfig:
        """Build timing configuration from ``CIV6_BRIDGE_TUNER_HANDSHAKE_QUIET_S`` when set."""
        resolved_quiet = handshake_quiet
        if resolved_quiet is None:
            raw = os.environ.get(ENV_TUNER_HANDSHAKE_QUIET_S, "").strip()
            if raw:
                try:
                    resolved_quiet = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"{ENV_TUNER_HANDSHAKE_QUIET_S} must be a positive number, got {raw!r}"
                    ) from exc
            else:
                resolved_quiet = DEFAULT_HANDSHAKE_QUIET_S
        return cls.from_handshake_quiet(resolved_quiet, timeout)


class FireTunerError(Exception):
    """Raised on FireTuner connection, protocol, or Lua execution errors."""


def lua_longstring(text: str) -> str:
    """Wrap ``text`` in a Lua long-bracket literal that cannot collide.

    Chooses a ``=`` level whose opening/closing brackets do not appear in
    ``text``, so arbitrary JSON (quotes, braces, brackets) needs no escaping.
    The leading newline is ignored by Lua and guards a leading ``]``.

    Args:
        text: Arbitrary string to embed as a Lua string literal.

    Returns:
        A Lua ``[==[ ... ]==]``-style long-string literal.
    """
    level = 0
    while ("]" + "=" * level + "]") in text or ("[" + "=" * level + "[") in text:
        level += 1
    eq = "=" * level
    return "[" + eq + "[\n" + text + "]" + eq + "]"


class FireTunerClient:
    """Synchronous FireTuner TCP client for executing Lua in a live Civ6 game.

    Not thread-safe; drive from a single caller (``SocketTunerBridge`` funnels
    all access through ``asyncio.to_thread`` so calls are serialized).

    Args:
        host: Tuner host (the game listens locally).
        port: Tuner port (default 4318).
        timeout: Socket timeout in seconds for connect and per-frame reads.
        preferred_states: Ordered Lua context names to target for commands.
    """

    def __init__(
        self,
        host: str = DEFAULT_TUNER_HOST,
        port: int = DEFAULT_TUNER_PORT,
        timeout: float = 5.0,
        preferred_states: tuple[str, ...] = PREFERRED_STATES,
        handshake_quiet: float | None = None,
        timing: FireTunerTimingConfig | None = None,
    ) -> None:
        _validate_positive_finite("timeout", timeout)
        if handshake_quiet is not None:
            _validate_positive_finite("handshake_quiet", handshake_quiet)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.preferred_states = preferred_states
        # Quiet-timeout override used to bound handshake waits when provided.
        # ``None`` defers to ``CIV6_BRIDGE_TUNER_HANDSHAKE_QUIET_S`` or default.
        self._handshake_quiet: float | None = handshake_quiet
        self._timing = timing
        self._sock: socket.socket | None = None
        self.states: dict[str, int] = {}

    def _require_socket(self) -> socket.socket:
        """Return the connected socket or raise ``FireTunerError`` (never ``assert``)."""
        if self._sock is None:
            raise FireTunerError("FireTunerClient is not connected")
        return self._sock

    # --- lifecycle -----------------------------------------------------------

    def connect(self) -> FireTunerClient:
        """Open the socket and discover Lua state indices.

        Raises:
            FireTunerError: If the connection fails.
        """
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as exc:
            raise FireTunerError(
                f"Cannot connect to FireTuner at {self.host}:{self.port}: {exc}. "
                "Is Civ6 running with EnableTuner=1 and no other tuner client "
                "(e.g. the FireTuner GUI) holding the single connection slot?"
            ) from exc
        self._sock.settimeout(self.timeout)
        try:
            self._handshake()
        except BaseException:
            # A failed handshake (e.g. the single tuner slot is already held,
            # or the game drops us mid-LSQ) must not leak the fd — a leaked
            # socket can occupy the slot until GC and block a legitimate
            # reconnect. Close before propagating.
            self.close()
            raise
        return self

    def close(self) -> None:
        """Close the socket if open."""
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> FireTunerClient:
        return self.connect()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def connected(self) -> bool:
        """Whether the socket is currently open."""
        return self._sock is not None

    # --- low-level framing ---------------------------------------------------

    def _send(self, tag: int, text: str) -> None:
        sock = self._require_socket()
        payload = text.encode("utf-8") + b"\x00"  # NUL is counted in the length
        sock.sendall(struct.pack(HEADER_FMT, len(payload), tag) + payload)

    def _recv_exact(self, n: int) -> bytes:
        sock = self._require_socket()
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise FireTunerError("FireTuner closed the connection")
            buf += chunk
        return bytes(buf)

    def _recv_frame(self) -> tuple[int, bytes]:
        length, tag = struct.unpack(HEADER_FMT, self._recv_exact(HEADER_SIZE))
        return tag, self._recv_exact(length)

    @staticmethod
    def _frame_payload(text: str) -> str:
        """Extract the human-readable payload from a decoded reply frame.

        Output frames are shaped ``O\\x00<ctx>: <text>`` (the internal NUL
        survives the trailing-NUL strip); error and other frames arrive as
        plain text (e.g. ``ERR:Runtime Error: ...``). This normalises both to
        just ``<text>`` so error detection (an ``ERR:`` prefix) works whether
        the game emits a bare ``ERR:`` frame or wraps it in an output frame,
        without false-positiving on normal output that merely contains "ERR".

        Args:
            text: The decoded frame body (trailing NULs already stripped).

        Returns:
            The payload message with any ``O\\x00<ctx>: `` prefix removed.
        """
        if not text.startswith(OUTPUT_FRAME_PREFIX):
            return text
        after_nul = text.split("\x00", 1)[1] if "\x00" in text else text[2:]
        ctx_split = after_nul.split(": ", 1)
        return ctx_split[1] if len(ctx_split) > 1 else after_nul

    def _drain(
        self, quiet: float = 0.4, max_duration: float | None = None
    ) -> list[tuple[int, bytes]]:
        """Read and return any pending frames until a short quiet timeout or cap."""
        _validate_positive_finite("quiet", quiet)
        if max_duration is not None:
            _validate_positive_finite("max_duration", max_duration)
        sock = self._require_socket()
        prev = sock.gettimeout()
        hard_deadline = (time.monotonic() + max_duration) if max_duration is not None else None
        sock.settimeout(quiet)
        frames: list[tuple[int, bytes]] = []
        try:
            while True:
                if hard_deadline is not None:
                    remaining = hard_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    sock.settimeout(min(quiet, remaining))
                frames.append(self._recv_frame())
        except (TimeoutError, FireTunerError, OSError) as exc:
            logger.debug("FireTuner drain stopped on %s: %s", type(exc).__name__, exc)
        finally:
            sock.settimeout(prev)
        return frames

    def _recv_query_frames(
        self,
        *,
        wait_first: float,
        quiet_after: float,
        max_duration: float,
    ) -> list[bytes]:
        """Collect ``TAG_QUERY`` reply frames without being starved by output chatter.

        Live Civ6 games emit 1 Hz ``AZ_HB:`` heartbeats (and Lua ``print()`` lines)
        as ``TAG_OUTPUT`` (``tag=-1``) frames on FireTuner. Only ``TAG_QUERY`` frames
        are collected and allowed to advance the post-reply quiet window; non-query
        frames are skipped without extending any deadline, and a hard monotonic cap
        bounds total wait time. Broken connections propagate rather than returning
        empty states silently.
        """
        _validate_positive_finite("wait_first", wait_first)
        _validate_positive_finite("quiet_after", quiet_after)
        _validate_positive_finite("max_duration", max_duration)
        sock = self._require_socket()
        prev = sock.gettimeout()
        start = time.monotonic()
        hard_deadline = start + max_duration
        idle_deadline = start + wait_first
        replies: list[bytes] = []
        skipped = 0
        try:
            while True:
                remaining = min(hard_deadline, idle_deadline) - time.monotonic()
                if remaining <= 0:
                    break
                sock.settimeout(remaining)
                try:
                    tag, frame = self._recv_frame()
                except TimeoutError:
                    break
                if tag == TAG_QUERY:
                    replies.append(frame)
                    idle_deadline = time.monotonic() + quiet_after
                else:
                    skipped += 1
        finally:
            sock.settimeout(prev)
            if skipped > 0:
                logger.debug("Ignored %d non-query FireTuner frame(s) during handshake", skipped)
        return replies

    # --- handshake / discovery ----------------------------------------------

    def _handshake(self) -> None:
        """Identify to the game and discover Lua state indices via ``LSQ:``."""
        started = time.monotonic()
        timing = self._timing or FireTunerTimingConfig.from_env(
            handshake_quiet=self._handshake_quiet,
            timeout=self.timeout,
        )
        self._drain(
            quiet=timing.initial_drain_quiet_s,
            max_duration=timing.initial_drain_cap_s,
        )
        self._send(TAG_QUERY, "APP:alphazero")
        self._recv_query_frames(
            wait_first=timing.app_wait_first_s,
            quiet_after=timing.app_quiet_after_s,
            max_duration=timing.app_max_duration_s,
        )

        self._send(TAG_QUERY, "LSQ:")
        frames = self._recv_query_frames(
            wait_first=timing.lsq_wait_first_s,
            quiet_after=timing.lsq_quiet_after_s,
            max_duration=timing.lsq_max_duration_s,
        )
        raw = b"".join(frames)
        self.states = self._parse_states(raw)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if self.states:
            logger.info(
                "FireTuner connected in %.0fms; %d Lua states (GameCore_Tuner=%s, InGame=%s)",
                elapsed_ms,
                len(self.states),
                self.states.get("GameCore_Tuner"),
                self.states.get("InGame"),
            )
        else:
            logger.warning("FireTuner LSQ returned no parseable states after %.0fms", elapsed_ms)

    @staticmethod
    def _parse_states(raw: bytes) -> dict[str, int]:
        """Parse an ``LSQ:`` reply of NUL-separated ``(index, name)`` pairs.

        Returns:
            Mapping of Lua state name → explicit integer index. The index is
            the number the game pairs with each name, NOT the list position.
        """
        parts = [p for p in raw.rstrip(b"\x00").decode("utf-8", "replace").split("\x00") if p]
        states: dict[str, int] = {}
        for i in range(0, len(parts) - 1, 2):
            num, name = parts[i], parts[i + 1]
            if num.isdigit():
                # Last write wins for duplicate names (e.g. two AlphaZeroMain);
                # command-target names (GameCore_Tuner/InGame) are unique.
                states[name] = int(num)
        return states

    def resolve_index(self, context: str | None = None) -> int:
        """Resolve a Lua context name to its CMD state index.

        Args:
            context: Explicit context name, or None to use ``preferred_states``.

        Returns:
            The integer state index.

        Raises:
            FireTunerError: If no candidate context is available.
        """
        if context is not None and context in self.states:
            return self.states[context]
        for name in self.preferred_states:
            if name in self.states:
                return self.states[name]
        raise FireTunerError(
            f"No usable Lua context found. Wanted {context or self.preferred_states}; "
            f"discovered {sorted(self.states)[:8]}..."
        )

    # --- execution -----------------------------------------------------------

    def run_lua(self, code: str, context: str | None = None, timeout: float = 5.0) -> str:
        """Execute Lua ``code`` in ``context`` and return collected output.

        Reads reply frames until the ``tag=3`` completion frame (or timeout).
        Output frames (``tag=-1``) are collected; an ``ERR:``-prefixed frame
        raises.

        Args:
            code: Lua source to execute.
            context: Target Lua state name (defaults to preferred states).
            timeout: Total deadline in seconds (must be positive).

        Returns:
            Newline-joined output text from ``print()`` calls.

        Raises:
            FireTunerError: On execution error or lost connection.
            ValueError: If ``timeout`` is not positive.
        """
        _validate_positive_finite("timeout", timeout)
        sock = self._require_socket()
        idx = self.resolve_index(context)
        prev = sock.gettimeout()
        deadline = time.monotonic() + timeout
        outputs: list[str] = []
        try:
            self._send(TAG_COMMAND, f"CMD:{idx}:{code}")
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                sock.settimeout(remaining)
                try:
                    tag, frame = self._recv_frame()
                except TimeoutError:
                    break
                body = frame.rstrip(b"\x00")
                if tag == TAG_COMMAND and body == b"":
                    break  # command-complete sentinel
                text = body.decode("utf-8", "replace")
                payload = self._frame_payload(text)
                if payload.startswith(ERROR_PREFIX):
                    raise FireTunerError(f"Lua error in {context or idx}: {payload}")
                outputs.append(payload)
        finally:
            sock.settimeout(prev)
        return "\n".join(outputs)

    def set_command(self, command_json: str, context: str | None = None) -> None:
        """Set ``ExposedMembers.AlphaZeroCommand`` to ``command_json``.

        Args:
            command_json: The serialized CommandMessage JSON string.
            context: Target Lua state (defaults to GameCore_Tuner).
        """
        lua = (
            "if not ExposedMembers then ExposedMembers = {} end; "
            "ExposedMembers.AlphaZeroCommand = " + lua_longstring(command_json)
        )
        # Honor the configured per-command timeout (self.timeout, derived from
        # BridgeConfig.command_timeout_ms) instead of run_lua's 5s default.
        self.run_lua(lua, context=context, timeout=self.timeout)

    def clear_command(self, context: str | None = None) -> None:
        """Clear ``ExposedMembers.AlphaZeroCommand`` (best effort)."""
        self.run_lua(
            "if ExposedMembers then ExposedMembers.AlphaZeroCommand = nil end",
            context=context,
        )


class SocketTunerBridge:
    """Async command bridge that injects commands over the FireTuner socket.

    Drop-in for :class:`~alpha_zero.bridge.tuner_bridge.TunerBridge`: exposes
    ``start``/``stop``/``send_command``/``stats`` so
    :class:`~alpha_zero.bridge.live_bridge.LiveBridge` can select it via
    ``BridgeConfig.command_transport == "socket"``. Blocking socket work runs
    in a worker thread so the event loop is never stalled.

    Args:
        config: Bridge configuration (supplies host/port/timeout when present).
        host: Override tuner host; defaults to ``config.tuner_host`` or local.
        port: Override tuner port; defaults to ``config.tuner_port`` or 4318.
    """

    def __init__(
        self,
        config: BridgeConfig,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self._config = config
        resolved_host = host or getattr(config, "tuner_host", None) or DEFAULT_TUNER_HOST
        resolved_port = port or getattr(config, "tuner_port", None) or DEFAULT_TUNER_PORT
        timeout = max(1.0, config.command_timeout_ms / 1000.0)
        self._client = FireTunerClient(host=resolved_host, port=resolved_port, timeout=timeout)
        self._running = False
        self._commands_sent = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Connect to the FireTuner socket (in a worker thread)."""
        await asyncio.to_thread(self._client.connect)
        self._running = True
        logger.info(
            "SocketTunerBridge started (%s:%d)",
            self._client.host,
            self._client.port,
        )

    async def stop(self) -> None:
        """Close the FireTuner socket.

        Acquires ``self._lock`` before closing so a close cannot interleave
        with an in-flight ``send_command``/``kick_end_turn`` worker thread
        operating on the same (non-thread-safe) socket. All socket ops are
        bounded by the client timeout, so the lock frees promptly; if a stray
        worker is somehow still holding it, we time out and close anyway
        rather than hang shutdown.
        """
        self._running = False
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self._client.timeout + 1.0)
            locked = True
        except TimeoutError:
            locked = False
            logger.warning("SocketTunerBridge.stop() closing without lock (worker stuck)")
        try:
            await asyncio.to_thread(self._client.close)
        finally:
            if locked:
                self._lock.release()
        logger.info("SocketTunerBridge stopped. Commands sent: %d", self._commands_sent)

    async def __aenter__(self) -> SocketTunerBridge:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def send_command(self, command: CommandMessage) -> bool:
        """Inject a command into the running game via the tuner socket.

        Args:
            command: The command message to deliver.

        Returns:
            True if the command was injected without a protocol error.
        """
        if not self._running:
            raise FireTunerError("SocketTunerBridge is not running")
        command_json = command.to_json()
        async with self._lock:
            await asyncio.to_thread(self._client.set_command, command_json)
        self._commands_sent += 1
        return True

    async def kick_end_turn(self, context: str = "InGame") -> bool:
        """Force an end-of-turn via the tuner to un-stall a waiting game.

        The mod only exports state and polls for commands on
        ``LocalPlayerTurnBegin``. If the previous turn's command window has
        already expired (the game is sitting idle on the local player's turn),
        no fresh state will ever arrive and the agent waits forever. Executing
        ``UI.RequestAction(ACTION_ENDTURN)`` directly — the same call the mod's
        ``executeEndTurn`` uses — advances the turn so a new turn-begin export
        is emitted, which the already-running LogTailer then catches.

        Args:
            context: Lua UI context to run the action in (``InGame`` default).

        Returns:
            True if the kick Lua executed without a protocol error.
        """
        if not self._running:
            raise FireTunerError("SocketTunerBridge is not running")
        lua = (
            "if UI and UI.RequestAction and ActionTypes then "
            "UI.RequestAction(ActionTypes.ACTION_ENDTURN) end; "
            'print("AZ_KICK_SENT")'
        )
        async with self._lock:
            out = await asyncio.to_thread(self._client.run_lua, lua, context)
        logger.info("Kick end-turn sent via tuner (%s): %s", context, out.strip())
        return True

    @property
    def stats(self) -> dict[str, Any]:
        """Return bridge statistics (mirrors TunerBridge.stats)."""
        return {
            "commands_sent": self._commands_sent,
            "running": self._running,
        }
