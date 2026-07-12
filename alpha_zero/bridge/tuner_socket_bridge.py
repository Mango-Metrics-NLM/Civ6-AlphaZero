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
import socket
import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from alpha_zero.bridge.config import BridgeConfig
    from alpha_zero.bridge.protocol import CommandMessage

logger = logging.getLogger(__name__)

__all__ = [
    "FireTunerClient",
    "FireTunerError",
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

#: Default FireTuner endpoint (the game is the listener).
DEFAULT_TUNER_HOST = "127.0.0.1"
DEFAULT_TUNER_PORT = 4318

#: Lua contexts to target for the command global, most-preferred first. The
#: proven ``.ltp`` panel used ``GameCore_Tuner``; ``InGame`` is the fallback.
PREFERRED_STATES = ("GameCore_Tuner", "InGame")


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
        handshake_quiet: float = 1.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.preferred_states = preferred_states
        # Quiet-timeout used to bound the handshake drains (which have no
        # completion sentinel). Lowered in tests to keep them fast.
        self._handshake_quiet = handshake_quiet
        self._sock: socket.socket | None = None
        self.states: dict[str, int] = {}

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
        self._handshake()
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
        if self._sock is None:
            raise FireTunerError("FireTunerClient is not connected")
        payload = text.encode("utf-8") + b"\x00"  # NUL is counted in the length
        self._sock.sendall(struct.pack(HEADER_FMT, len(payload), tag) + payload)

    def _recv_exact(self, n: int) -> bytes:
        assert self._sock is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise FireTunerError("FireTuner closed the connection")
            buf += chunk
        return bytes(buf)

    def _recv_frame(self) -> tuple[int, bytes]:
        length, tag = struct.unpack(HEADER_FMT, self._recv_exact(HEADER_SIZE))
        return tag, self._recv_exact(length)

    def _drain(self, quiet: float = 0.4) -> list[tuple[int, bytes]]:
        """Read and return any pending frames until a short quiet timeout."""
        assert self._sock is not None
        prev = self._sock.gettimeout()
        self._sock.settimeout(quiet)
        frames: list[tuple[int, bytes]] = []
        try:
            while True:
                frames.append(self._recv_frame())
        except (TimeoutError, FireTunerError, OSError):
            pass
        finally:
            self._sock.settimeout(prev)
        return frames

    # --- handshake / discovery ----------------------------------------------

    def _handshake(self) -> None:
        """Identify to the game and discover Lua state indices via ``LSQ:``."""
        self._drain(quiet=self._handshake_quiet)
        self._send(TAG_QUERY, "APP:alphazero")
        self._drain(quiet=self._handshake_quiet)  # identity string; unused

        self._send(TAG_QUERY, "LSQ:")
        raw = b""
        for _tag, frame in self._drain(quiet=self._handshake_quiet * 1.5):
            raw += frame
        self.states = self._parse_states(raw)
        if self.states:
            logger.info(
                "FireTuner connected; %d Lua states (GameCore_Tuner=%s, InGame=%s)",
                len(self.states),
                self.states.get("GameCore_Tuner"),
                self.states.get("InGame"),
            )
        else:
            logger.warning("FireTuner LSQ returned no parseable states")

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
            timeout: Per-read timeout in seconds.

        Returns:
            Newline-joined output text from ``print()`` calls.

        Raises:
            FireTunerError: On execution error or lost connection.
        """
        if self._sock is None:
            raise FireTunerError("FireTunerClient is not connected")
        idx = self.resolve_index(context)
        prev = self._sock.gettimeout()
        self._sock.settimeout(timeout)
        outputs: list[str] = []
        try:
            self._send(TAG_COMMAND, f"CMD:{idx}:{code}")
            while True:
                try:
                    tag, frame = self._recv_frame()
                except TimeoutError:
                    break
                body = frame.rstrip(b"\x00")
                if tag == TAG_COMMAND and body == b"":
                    break  # command-complete sentinel
                text = body.decode("utf-8", "replace")
                if text.startswith("ERR:") or "] ERR:" in text:
                    raise FireTunerError(f"Lua error in {context or idx}: {text}")
                outputs.append(text)
        finally:
            self._sock.settimeout(prev)
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
        self.run_lua(lua, context=context)

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
        """Close the FireTuner socket."""
        self._running = False
        await asyncio.to_thread(self._client.close)
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
