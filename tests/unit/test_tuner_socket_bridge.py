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
import time

import pytest

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.tuner_socket_bridge import (
    DEFAULT_HANDSHAKE_QUIET_S,
    ENV_TUNER_HANDSHAKE_QUIET_S,
    HEADER_FMT,
    HEADER_SIZE,
    TAG_COMMAND,
    TAG_OUTPUT,
    TAG_QUERY,
    FireTunerClient,
    FireTunerError,
    FireTunerTimingConfig,
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

    def __init__(
        self,
        error_on_cmd: bool = False,
        *,
        chatter_interval_s: float | None = None,
        split_lsq: bool = False,
        omit_cmd_completion: bool = False,
        empty_lsq: bool = False,
        drop_on_lsq: bool = False,
    ) -> None:
        self._error_on_cmd = error_on_cmd
        self._chatter_interval_s = chatter_interval_s
        self._split_lsq = split_lsq
        self._omit_cmd_completion = omit_cmd_completion
        self._empty_lsq = empty_lsq
        self._drop_on_lsq = drop_on_lsq
        self._stop = threading.Event()
        self._send_lock = threading.Lock()
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

    def _send_bytes(self, conn: socket.socket, data: bytes) -> None:
        with self._send_lock:
            conn.sendall(data)

    def _emit_chatter(self, conn: socket.socket) -> None:
        assert self._chatter_interval_s is not None
        toggle = False
        while not self._stop.is_set():
            try:
                msg = (
                    'O\x00AlphaZeroMain: AZ_HB:{"lua_alive":true,"turn":1}'
                    if not toggle
                    else "O\x00AlphaZeroMain: spurious\x00part"
                )
                toggle = not toggle
                self._send_bytes(conn, _frame(TAG_OUTPUT, msg))
            except OSError:
                return
            self._stop.wait(self._chatter_interval_s)

    def _serve(self) -> None:
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        chatter_thread: threading.Thread | None = None
        if self._chatter_interval_s is not None:
            chatter_thread = threading.Thread(target=self._emit_chatter, args=(conn,), daemon=True)
            chatter_thread.start()
        with conn:
            while not self._stop.is_set():
                frame = self._recv_frame(conn)
                if frame is None:
                    self._stop.set()
                    return
                _tag, body = frame
                text = body.rstrip(b"\x00").decode("utf-8", "replace")
                if text.startswith("APP:"):
                    self._send_bytes(conn, _frame(TAG_QUERY, "Civ6"))
                elif text.startswith("LSQ:"):
                    if self._drop_on_lsq:
                        conn.close()
                        self._stop.set()
                        return
                    if self._empty_lsq:
                        self._send_bytes(conn, _frame_raw(TAG_QUERY, b"\x00"))
                    elif self._split_lsq:
                        # Split LSQ payload across two TAG_QUERY frames with
                        # non-query frames interleaved to simulate slow LSQ.
                        p1 = b"0\x00Main State\x002\x00GameCore_Tuner\x00"
                        p2 = b"96\x00InGame\x0097\x00AdvisorPopup\x00"
                        self._send_bytes(conn, _frame_raw(TAG_QUERY, p1))
                        self._send_bytes(
                            conn, _frame(TAG_OUTPUT, "O\x00AlphaZeroMain: interleaved")
                        )
                        self._send_bytes(conn, _frame_raw(TAG_QUERY, p2))
                    else:
                        self._send_bytes(conn, _LSQ_PAIRS)
                elif text.startswith("CMD:"):
                    self.received_cmds.append(text)
                    if self._error_on_cmd:
                        self._send_bytes(conn, _frame(TAG_OUTPUT, "ERR: boom"))
                    else:
                        self._send_bytes(conn, _frame(TAG_OUTPUT, "O\x00GameCore_Tuner: ok"))
                    if not self._omit_cmd_completion:
                        self._send_bytes(conn, _frame_raw(TAG_COMMAND, b"\x00"))  # completion

    def close(self) -> None:
        self._stop.set()
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


class TestFramePayload:
    """_frame_payload normalises reply frames so ERR: detection is robust."""

    def test_strips_output_frame_prefix(self) -> None:
        assert FireTunerClient._frame_payload("O\x00GameCore_Tuner: SET_OK") == "SET_OK"

    def test_bare_error_frame_passes_through(self) -> None:
        assert FireTunerClient._frame_payload("ERR:Runtime Error: x") == "ERR:Runtime Error: x"

    def test_error_wrapped_in_output_frame_is_extracted(self) -> None:
        # The reviewer's case: a Lua error surfaced as an output frame must
        # still expose the ERR: prefix after the context is stripped.
        assert FireTunerClient._frame_payload("O\x00GameCore_Tuner: ERR: boom").startswith("ERR:")

    def test_normal_output_containing_err_word_is_not_flagged(self) -> None:
        # Must not false-positive: 'ERROR_CODE' is not an 'ERR:' error marker.
        assert not FireTunerClient._frame_payload("O\x00ctx: ERROR_CODE").startswith("ERR:")


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

    def test_connect_not_starved_by_continuous_output_chatter(self) -> None:
        # Output frames faster than handshake_quiet (0.03s < 0.15s) used to reset
        # _drain() indefinitely and corrupt LSQ parsing; handshake must finish
        # quickly and resolve exact states.
        srv = MockTunerServer(chatter_interval_s=0.03)
        try:
            started = time.monotonic()
            with _client_for(srv) as client:
                elapsed = time.monotonic() - started
                assert elapsed < 1.5
                assert client.states["GameCore_Tuner"] == 2
                assert client.states["InGame"] == 96
                out = client.run_lua('print("ok")', context="GameCore_Tuner")
                assert "ok" in out
        finally:
            srv.close()

    def test_split_lsq_ignores_interleaved_output_frames(self) -> None:
        srv = MockTunerServer(split_lsq=True)
        try:
            with _client_for(srv) as client:
                assert client.states == {
                    "Main State": 0,
                    "GameCore_Tuner": 2,
                    "InGame": 96,
                    "AdvisorPopup": 97,
                }
        finally:
            srv.close()

    def test_drain_respects_max_duration_under_chatter(self) -> None:
        srv = MockTunerServer(chatter_interval_s=0.03)
        try:
            with _client_for(srv) as client:
                started = time.monotonic()
                frames = client._drain(quiet=0.5, max_duration=0.15)
                elapsed = time.monotonic() - started
                assert elapsed < 0.6
                assert len(frames) >= 1
        finally:
            srv.close()

    def test_run_lua_respects_total_timeout_under_chatter(self) -> None:
        srv = MockTunerServer(chatter_interval_s=0.03, omit_cmd_completion=True)
        try:
            with _client_for(srv) as client:
                started = time.monotonic()
                out = client.run_lua("local x = 1", context="GameCore_Tuner", timeout=0.25)
                elapsed = time.monotonic() - started
                assert elapsed < 0.8
                assert "ok" in out
        finally:
            srv.close()

    def test_clear_command_clears_exposed_members(self, mock_server) -> None:
        with _client_for(mock_server) as client:
            client.clear_command()
        assert "ExposedMembers.AlphaZeroCommand = nil" in mock_server.received_cmds[-1]

    def test_resolve_index_raises_when_missing(self, mock_server) -> None:
        with _client_for(mock_server) as client:
            client.states = {"UnrelatedState": 1}
            with pytest.raises(FireTunerError, match="No usable Lua context"):
                client.resolve_index("MissingState")

    def test_empty_lsq_reply_leaves_states_empty(self, caplog) -> None:
        srv = MockTunerServer(empty_lsq=True)
        try:
            with caplog.at_level("WARNING"):
                with _client_for(srv) as client:
                    assert client.states == {}
            assert "no parseable states" in caplog.text
        finally:
            srv.close()

    def test_dropped_connection_during_lsq_closes_socket_and_raises(self) -> None:
        srv = MockTunerServer(drop_on_lsq=True)
        try:
            client = _client_for(srv)
            with pytest.raises(FireTunerError):
                client.connect()
            assert client.connected is False
        finally:
            srv.close()

    def test_disconnected_guards_and_invalid_timeouts(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            FireTunerClient(timeout=0)
        with pytest.raises(ValueError, match="handshake_quiet"):
            FireTunerClient(handshake_quiet=0)
        client = FireTunerClient()
        with pytest.raises(FireTunerError, match="not connected"):
            client._send(TAG_QUERY, "APP:x")
        with pytest.raises(FireTunerError, match="not connected"):
            client._recv_exact(4)
        with pytest.raises(FireTunerError, match="not connected"):
            client._drain()
        with pytest.raises(FireTunerError, match="not connected"):
            client._recv_query_frames(wait_first=0.1, quiet_after=0.1, max_duration=0.2)
        with pytest.raises(ValueError, match="must be positive"):
            client._recv_query_frames(wait_first=0.0, quiet_after=0.1, max_duration=0.2)
        with pytest.raises(ValueError, match="timeout must be positive"):
            client.run_lua("print(1)", timeout=0)


class TestFireTunerTimingConfig:
    def test_defaults_and_validation(self) -> None:
        cfg = FireTunerTimingConfig()
        assert cfg.app_wait_first_s > 0
        with pytest.raises(ValueError, match="must be positive"):
            FireTunerTimingConfig(app_max_duration_s=0)
        with pytest.raises(ValueError, match="handshake_quiet"):
            FireTunerTimingConfig.from_handshake_quiet(0, 5.0)
        with pytest.raises(ValueError, match="timeout"):
            FireTunerTimingConfig.from_handshake_quiet(1.0, 0)

    def test_from_env_uses_override_and_rejects_bad_values(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_TUNER_HANDSHAKE_QUIET_S, raising=False)
        default_cfg = FireTunerTimingConfig.from_env(timeout=5.0)
        assert default_cfg == FireTunerTimingConfig.from_handshake_quiet(
            DEFAULT_HANDSHAKE_QUIET_S, 5.0
        )

        monkeypatch.setenv(ENV_TUNER_HANDSHAKE_QUIET_S, "0.25")
        custom_cfg = FireTunerTimingConfig.from_env(timeout=5.0)
        assert custom_cfg == FireTunerTimingConfig.from_handshake_quiet(0.25, 5.0)

        monkeypatch.setenv(ENV_TUNER_HANDSHAKE_QUIET_S, "not-a-float")
        with pytest.raises(ValueError, match=ENV_TUNER_HANDSHAKE_QUIET_S):
            FireTunerTimingConfig.from_env(timeout=5.0)

    def test_client_accepts_explicit_timing(self, mock_server) -> None:
        timing = FireTunerTimingConfig.from_handshake_quiet(0.15, 2.0)
        with FireTunerClient(
            host=mock_server.host, port=mock_server.port, timeout=2.0, timing=timing
        ) as client:
            assert client.states["GameCore_Tuner"] == 2

    def test_client_default_handshake_quiet_honors_env(self, mock_server, monkeypatch) -> None:
        monkeypatch.setenv(ENV_TUNER_HANDSHAKE_QUIET_S, "0.15")
        with FireTunerClient(host=mock_server.host, port=mock_server.port, timeout=2.0) as client:
            assert client._handshake_quiet is None
            assert client.states["GameCore_Tuner"] == 2

        # Explicit handshake_quiet takes precedence even when env is invalid.
        # MockTunerServer accepts one client per instance, so create fresh servers.
        monkeypatch.setenv(ENV_TUNER_HANDSHAKE_QUIET_S, "bad-float")
        srv2 = MockTunerServer()
        try:
            with FireTunerClient(
                host=srv2.host,
                port=srv2.port,
                timeout=2.0,
                handshake_quiet=0.15,
            ) as client:
                assert client.states["GameCore_Tuner"] == 2
        finally:
            srv2.close()

        srv3 = MockTunerServer()
        try:
            with pytest.raises(ValueError, match=ENV_TUNER_HANDSHAKE_QUIET_S):
                FireTunerClient(host=srv3.host, port=srv3.port, timeout=2.0).connect()
        finally:
            srv3.close()

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0])
    def test_non_finite_or_non_positive_values_rejected(
        self, bad_value: float, monkeypatch
    ) -> None:
        with pytest.raises(ValueError, match="must be positive and finite"):
            FireTunerTimingConfig(app_wait_first_s=bad_value)
        with pytest.raises(ValueError, match="must be positive and finite"):
            FireTunerClient(timeout=bad_value)
        client = FireTunerClient()
        with pytest.raises(ValueError, match="must be positive and finite"):
            client.run_lua("print(1)", timeout=bad_value)
        monkeypatch.setenv(ENV_TUNER_HANDSHAKE_QUIET_S, str(bad_value))
        with pytest.raises(ValueError, match="must be positive and finite"):
            FireTunerTimingConfig.from_env(timeout=5.0)

    async def test_socket_bridge_async_context_and_lock_timeout_stop(
        self, mock_server, caplog
    ) -> None:
        bridge = SocketTunerBridge(
            BridgeConfig(command_transport="socket", command_timeout_ms=100),
            host=mock_server.host,
            port=mock_server.port,
        )
        bridge._client._handshake_quiet = 0.15
        async with bridge:
            assert bridge.stats["running"] is True
        assert bridge.stats["running"] is False

        # Simulate stuck worker holding self._lock when stop() runs
        bridge._client.timeout = 0.05
        await bridge._lock.acquire()
        with caplog.at_level("WARNING"):
            await bridge.stop()
        assert "closing without lock" in caplog.text


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
