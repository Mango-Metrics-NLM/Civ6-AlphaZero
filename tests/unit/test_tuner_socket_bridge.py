"""Unit tests for the headless FireTuner socket transport.

Framing/protocol behavior is exercised against an in-process mock tuner
server (a real localhost TCP socket) so the pack/parse round-trip, the
``LSQ:`` pair-index discovery, ``run_lua`` completion-frame handling, and
error detection are all verified without the live game.
"""

from __future__ import annotations

import socket
import struct
import threading

import pytest

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.tuner_socket_bridge import (
    HEADER_FMT,
    HEADER_SIZE,
    TAG_COMMAND,
    TAG_OUTPUT,
    TAG_QUERY,
    FireTunerClient,
    FireTunerError,
    SocketTunerBridge,
    lua_longstring,
)

# --- helpers -----------------------------------------------------------------


def _frame(tag: int, text: str) -> bytes:
    payload = text.encode("utf-8") + b"\x00"
    return struct.pack(HEADER_FMT, len(payload), tag) + payload


def _frame_raw(tag: int, raw: bytes) -> bytes:
    return struct.pack(HEADER_FMT, len(raw), tag) + raw


# Canonical LSQ reply: NUL-separated (index, name) pairs, exactly as the live
# Civ6 build returns (GameCore_Tuner=2, InGame=96 in the real dump).
_LSQ_PAIRS = _frame_raw(
    TAG_QUERY,
    b"\x00".join(
        [
            b"0",
            b"Main State",
            b"1",
            b"DebugHotloadCache",
            b"2",
            b"GameCore_Tuner",
            b"3",
            b"WorldInput",
            b"96",
            b"InGame",
            b"97",
            b"AdvisorPopup",
        ]
    )
    + b"\x00",
)


class MockTunerServer:
    """Minimal in-process FireTuner server for exercising the client.

    Replies to ``APP:`` with an identity frame, to ``LSQ:`` with the pair
    list, and to a ``CMD:<idx>:<code>`` frame with an echo output frame plus
    a completion frame — unless ``error_on_cmd`` is set, in which case it
    returns an ``ERR:`` frame. Records every command payload it receives.
    """

    def __init__(self, error_on_cmd: bool = False) -> None:
        self._error_on_cmd = error_on_cmd
        self.received_cmds: list[str] = []
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self.host, self.port = self._srv.getsockname()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _recv_frame(self, conn: socket.socket) -> tuple[int, bytes] | None:
        hdr = self._recv_exact(conn, HEADER_SIZE)
        if hdr is None:
            return None
        length, tag = struct.unpack(HEADER_FMT, hdr)
        body = self._recv_exact(conn, length)
        if body is None:
            return None
        return tag, body

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = conn.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def _serve(self) -> None:
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        with conn:
            while True:
                frame = self._recv_frame(conn)
                if frame is None:
                    return
                _tag, body = frame
                text = body.rstrip(b"\x00").decode("utf-8", "replace")
                if text.startswith("APP:"):
                    conn.sendall(_frame(TAG_QUERY, "Civ6"))
                elif text.startswith("LSQ:"):
                    conn.sendall(_LSQ_PAIRS)
                elif text.startswith("CMD:"):
                    self.received_cmds.append(text)
                    if self._error_on_cmd:
                        conn.sendall(_frame(TAG_OUTPUT, "ERR: boom"))
                    else:
                        conn.sendall(_frame(TAG_OUTPUT, "O\x00GameCore_Tuner: ok"))
                    conn.sendall(_frame_raw(TAG_COMMAND, b"\x00"))  # completion

    def close(self) -> None:
        try:
            self._srv.close()
        except OSError:
            pass


@pytest.fixture
def mock_server():
    srv = MockTunerServer()
    yield srv
    srv.close()


@pytest.fixture
def error_server():
    srv = MockTunerServer(error_on_cmd=True)
    yield srv
    srv.close()


def _client_for(srv: MockTunerServer) -> FireTunerClient:
    return FireTunerClient(host=srv.host, port=srv.port, timeout=2.0, handshake_quiet=0.15)


# --- lua_longstring ----------------------------------------------------------


class TestLuaLongstring:
    def test_wraps_plain_json(self) -> None:
        out = lua_longstring('{"a":1}')
        assert out == '[[\n{"a":1}]]'

    def test_escalates_level_on_collision(self) -> None:
        # Text containing ]] must force a higher = level so it can't close early.
        out = lua_longstring("x]]y")
        assert out.startswith("[=[")
        assert out.endswith("]=]")
        assert "x]]y" in out

    def test_roundtrip_marker_present(self) -> None:
        payload = '{"msg":"has ]] and ]=] inside"}'
        out = lua_longstring(payload)
        # The chosen level's closer must not appear inside the payload.
        level = out.count("=", 0, out.index("[", 1) + 1)
        closer = "]" + "=" * level + "]"
        assert payload not in ("",)  # sanity
        assert out.count(closer) == 1
        assert out.endswith(closer)


# --- state parsing -----------------------------------------------------------


class TestParseStates:
    def test_pairs_use_explicit_index_not_position(self) -> None:
        states = FireTunerClient._parse_states(_LSQ_PAIRS[HEADER_SIZE:])
        # GameCore_Tuner is the 3rd pair but its explicit index is 2.
        assert states["GameCore_Tuner"] == 2
        assert states["InGame"] == 96
        assert states["Main State"] == 0

    def test_ignores_malformed_trailing(self) -> None:
        raw = b"\x00".join([b"0", b"Alpha", b"1", b"Beta", b"notanumber"]) + b"\x00"
        states = FireTunerClient._parse_states(raw)
        assert states == {"Alpha": 0, "Beta": 1}

    def test_empty_reply(self) -> None:
        assert FireTunerClient._parse_states(b"") == {}


# --- handshake + run_lua against the mock server -----------------------------


class TestFireTunerClientLive:
    def test_connect_discovers_states(self, mock_server) -> None:
        with _client_for(mock_server) as client:
            assert client.states["GameCore_Tuner"] == 2
            assert client.states["InGame"] == 96
            assert client.connected

    def test_resolve_index_prefers_game_core_tuner(self, mock_server) -> None:
        with _client_for(mock_server) as client:
            assert client.resolve_index() == 2
            assert client.resolve_index("InGame") == 96

    def test_run_lua_targets_resolved_index(self, mock_server) -> None:
        with _client_for(mock_server) as client:
            out = client.run_lua('print("hi")', context="GameCore_Tuner")
            assert "ok" in out
            assert mock_server.received_cmds[-1].startswith("CMD:2:")

    def test_set_command_sends_longstring_json(self, mock_server) -> None:
        with _client_for(mock_server) as client:
            client.set_command('{"msg_id":7,"commands":[]}')
        cmd = mock_server.received_cmds[-1]
        assert cmd.startswith("CMD:2:")
        assert "ExposedMembers.AlphaZeroCommand =" in cmd
        assert '{"msg_id":7,"commands":[]}' in cmd

    def test_run_lua_raises_on_error_frame(self, error_server) -> None:
        with _client_for(error_server) as client:
            with pytest.raises(FireTunerError, match="boom"):
                client.run_lua('error("x")')

    def test_connect_failure_is_wrapped(self) -> None:
        # Nothing listening on this port.
        client = FireTunerClient(host="127.0.0.1", port=1, timeout=0.5)
        with pytest.raises(FireTunerError, match="Cannot connect"):
            client.connect()

    def test_run_lua_without_connection_raises(self) -> None:
        client = FireTunerClient()
        with pytest.raises(FireTunerError, match="not connected"):
            client.run_lua('print("x")')

    def test_connect_closes_socket_when_handshake_fails(self, monkeypatch) -> None:
        # A handshake failure must not leak the fd (it could hold the single
        # tuner slot). connect() should close and re-raise, leaving connected=False.
        client = FireTunerClient(host="127.0.0.1", port=1, timeout=0.5, handshake_quiet=0.05)

        def boom(self_) -> None:
            raise FireTunerError("handshake exploded")

        # Let the socket "connect" succeed but the handshake blow up.
        opened = {}

        def fake_create_connection(addr, timeout=None):
            import socket as _s

            a, b = _s.socketpair()
            opened["sock"] = a
            b.close()
            return a

        monkeypatch.setattr(
            "alpha_zero.bridge.tuner_socket_bridge.socket.create_connection",
            fake_create_connection,
        )
        monkeypatch.setattr(FireTunerClient, "_handshake", boom)
        with pytest.raises(FireTunerError, match="handshake exploded"):
            client.connect()
        assert client.connected is False  # socket was closed, not leaked


# --- SocketTunerBridge async surface -----------------------------------------


class _FakeClient:
    """Records set_command calls without touching a socket."""

    def __init__(self, **_kw) -> None:
        self.host = "127.0.0.1"
        self.port = 4318
        self.timeout = 5.0
        self.connected = False
        self.commands: list[str] = []
        self.closed = False

    def connect(self):
        self.connected = True
        return self

    def close(self):
        self.closed = True

    def set_command(self, command_json: str, context=None) -> None:
        self.commands.append(command_json)


class _FakeCommand:
    def __init__(self, msg_id: int) -> None:
        self.msg_id = msg_id

    def to_json(self) -> str:
        return f'{{"msg_id":{self.msg_id},"type":"command","commands":[]}}'


@pytest.fixture
def socket_bridge(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(
        "alpha_zero.bridge.tuner_socket_bridge.FireTunerClient",
        lambda **kw: fake,
    )
    bridge = SocketTunerBridge(BridgeConfig(command_transport="socket"))
    return bridge, fake


class TestSocketTunerBridge:
    async def test_start_connects_and_stop_closes(self, socket_bridge) -> None:
        bridge, fake = socket_bridge
        await bridge.start()
        assert fake.connected
        assert bridge.stats["running"] is True
        await bridge.stop()
        assert fake.closed
        assert bridge.stats["running"] is False

    async def test_send_command_injects_json_and_counts(self, socket_bridge) -> None:
        bridge, fake = socket_bridge
        await bridge.start()
        assert await bridge.send_command(_FakeCommand(42)) is True
        assert fake.commands == ['{"msg_id":42,"type":"command","commands":[]}']
        assert bridge.stats["commands_sent"] == 1

    async def test_send_command_before_start_raises(self, socket_bridge) -> None:
        bridge, _fake = socket_bridge
        with pytest.raises(FireTunerError, match="not running"):
            await bridge.send_command(_FakeCommand(1))

    async def test_kick_end_turn_runs_action_lua(self, mock_server) -> None:
        bridge = SocketTunerBridge(
            BridgeConfig(command_transport="socket"),
            host=mock_server.host,
            port=mock_server.port,
        )
        # Shrink handshake drains for the real socket client used here.
        bridge._client._handshake_quiet = 0.15
        await bridge.start()
        try:
            assert await bridge.kick_end_turn() is True
        finally:
            await bridge.stop()
        cmd = mock_server.received_cmds[-1]
        assert "ACTION_ENDTURN" in cmd
        # InGame index is 96 in the canonical LSQ dump.
        assert cmd.startswith("CMD:96:")

    async def test_kick_before_start_raises(self, socket_bridge) -> None:
        bridge, _fake = socket_bridge
        with pytest.raises(FireTunerError, match="not running"):
            await bridge.kick_end_turn()


# --- config wiring -----------------------------------------------------------


class TestTransportConfig:
    def test_default_transport_is_file(self) -> None:
        assert BridgeConfig().command_transport == "file"

    def test_socket_transport_valid(self) -> None:
        cfg = BridgeConfig(command_transport="socket", tuner_port=4318)
        assert cfg.command_transport == "socket"
        assert cfg.tuner_port == 4318

    def test_invalid_transport_rejected(self) -> None:
        with pytest.raises(ValueError, match="command_transport"):
            BridgeConfig(command_transport="carrier-pigeon")

    def test_invalid_tuner_port_rejected(self) -> None:
        with pytest.raises(ValueError, match="tuner_port"):
            BridgeConfig(tuner_port=0)

    def test_from_env_reads_transport(self, monkeypatch) -> None:
        monkeypatch.setenv("CIV6_BRIDGE_TRANSPORT", "socket")
        monkeypatch.setenv("CIV6_BRIDGE_TUNER_PORT", "5000")
        cfg = BridgeConfig.from_env()
        assert cfg.command_transport == "socket"
        assert cfg.tuner_port == 5000

    def test_live_bridge_selects_socket_transport(self, tmp_path) -> None:
        from alpha_zero.bridge.live_bridge import LiveBridge
        from alpha_zero.bridge.tuner_socket_bridge import SocketTunerBridge

        log = tmp_path / "Lua.log"
        log.write_text("")
        cfg = BridgeConfig(command_transport="socket", comm_dir=tmp_path)
        bridge = LiveBridge(cfg, log)
        assert isinstance(bridge._tuner, SocketTunerBridge)

    def test_live_bridge_defaults_to_file_transport(self, tmp_path) -> None:
        from alpha_zero.bridge.live_bridge import LiveBridge
        from alpha_zero.bridge.tuner_bridge import TunerBridge

        log = tmp_path / "Lua.log"
        log.write_text("")
        cfg = BridgeConfig(comm_dir=tmp_path)
        bridge = LiveBridge(cfg, log)
        assert isinstance(bridge._tuner, TunerBridge)
