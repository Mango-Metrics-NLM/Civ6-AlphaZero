"""The kick re-arms the mod and never ends the turn: the Python side (contract PR 1)."""

from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.live_bridge import LiveBridge
from alpha_zero.bridge.tuner_socket_bridge import FireTunerError, SocketTunerBridge
from alpha_zero.civ6_local import Civ6LocalConfig, run_civ6_bridge
from tests.unit.test_tuner_socket_bridge import MockTunerServer


@pytest.fixture
def server():
    srv = MockTunerServer()
    yield srv
    srv.close()


async def started(srv: MockTunerServer) -> SocketTunerBridge:
    bridge = SocketTunerBridge(
        BridgeConfig(command_transport="socket"), host=srv.host, port=srv.port
    )
    bridge._client._handshake_quiet = 0.15
    await bridge.start()
    return bridge


async def rearm(bridge: SocketTunerBridge, **kw: Any) -> Any:
    result = bridge.request_rearm(**kw)
    return await result if inspect.isawaitable(result) else result


class TestRequestRearm:
    async def test_frame_targets_gamecore_and_sets_the_flag(self, server) -> None:
        bridge = await started(server)
        try:
            nonce = await rearm(bridge)
        finally:
            await bridge.stop()
        assert isinstance(nonce, str) and nonce
        cmd = server.received_cmds[-1]
        assert cmd.startswith("CMD:2:")  # GameCore_Tuner, as set_command
        assert "ExposedMembers.AlphaZeroRearm" in cmd
        assert nonce in cmd
        assert "ACTION_ENDTURN" not in cmd
        assert "RequestAction" not in cmd

    async def test_each_request_has_a_fresh_nonce(self, server) -> None:
        bridge = await started(server)
        try:
            first, second = await rearm(bridge), await rearm(bridge)
        finally:
            await bridge.stop()
        assert first != second

    async def test_an_explicit_context_is_honoured(self, server) -> None:
        bridge = await started(server)
        try:
            await rearm(bridge, context="InGame")
        finally:
            await bridge.stop()
        assert server.received_cmds[-1].startswith("CMD:96:")

    async def test_before_start_raises(self) -> None:
        bridge = SocketTunerBridge(BridgeConfig(command_transport="socket"))
        with pytest.raises(FireTunerError, match="not running"):
            await rearm(bridge)

    async def test_the_deprecated_kick_warns_and_sends_the_same_flag(self, server) -> None:
        bridge = await started(server)
        try:
            with pytest.warns(DeprecationWarning):
                assert await bridge.kick_end_turn() is True
        finally:
            await bridge.stop()
        cmd = server.received_cmds[-1]
        assert cmd.startswith("CMD:2:")
        assert "ExposedMembers.AlphaZeroRearm" in cmd
        assert "ACTION_ENDTURN" not in cmd

    async def test_the_deprecated_kick_calls_request_rearm(self, monkeypatch) -> None:
        bridge = SocketTunerBridge(BridgeConfig(command_transport="socket"))
        bridge._running = True
        calls: list[Any] = []

        async def fake(*a: Any, **k: Any) -> str:
            calls.append((a, k))
            return "n"

        monkeypatch.setattr(bridge, "request_rearm", fake)
        with pytest.warns(DeprecationWarning):
            assert await bridge.kick_end_turn() is True
        assert len(calls) == 1


class TestLiveBridgeKick:
    @staticmethod
    def bridge(tmp_path: Path, tuner: Any) -> LiveBridge:
        live = LiveBridge(BridgeConfig(comm_dir=tmp_path), log_path=tmp_path / "Lua.log")
        live._tuner = tuner
        return live

    async def test_uses_request_rearm_when_the_transport_has_it(self, tmp_path) -> None:
        tuner = MagicMock(spec=["request_rearm"])
        tuner.request_rearm = AsyncMock(return_value="nonce")
        assert await self.bridge(tmp_path, tuner).kick_turn() is True
        tuner.request_rearm.assert_awaited_once()

    async def test_does_not_call_kick_end_turn(self, tmp_path) -> None:
        tuner = MagicMock(spec=["request_rearm", "kick_end_turn"])
        tuner.request_rearm = AsyncMock(return_value="nonce")
        tuner.kick_end_turn = AsyncMock(return_value=True)
        await self.bridge(tmp_path, tuner).kick_turn()
        tuner.kick_end_turn.assert_not_awaited()

    async def test_returns_false_without_request_rearm(self, tmp_path) -> None:
        tuner = MagicMock(spec=["send_command"])
        assert await self.bridge(tmp_path, tuner).kick_turn() is False


def run_bridge(monkeypatch, kick_env: str | None, kick: AsyncMock) -> tuple[int, MagicMock]:
    if kick_env is None:
        monkeypatch.delenv("CIV6_BRIDGE_KICK_START", raising=False)
    else:
        monkeypatch.setenv("CIV6_BRIDGE_KICK_START", kick_env)
    bridge = MagicMock()
    bridge.start, bridge.stop, bridge.kick_turn = AsyncMock(), AsyncMock(), kick
    loop = MagicMock()
    loop.run = AsyncMock(return_value=0)
    with (
        patch("alpha_zero.civ6_local._build_network_for_bridge", return_value=MagicMock()),
        patch.dict(
            sys.modules,
            {
                "alpha_zero.bridge.config": MagicMock(BridgeConfig=MagicMock()),
                "alpha_zero.bridge.live_bridge": MagicMock(
                    LiveBridge=MagicMock(return_value=bridge)
                ),
                "alpha_zero.bridge.state_converter": MagicMock(),
                "alpha_zero.bridge.action_converter": MagicMock(),
                "alpha_zero.bridge.agent_loop": MagicMock(
                    BridgeAgentLoop=MagicMock(return_value=loop), GreedyNetworkPolicy=MagicMock()
                ),
            },
        ),
    ):
        rc = run_civ6_bridge(Civ6LocalConfig(mode="civ6"))
    loop.run.assert_awaited_once()
    return rc, bridge


class TestCiv6LocalKickStart:
    def test_on_calls_the_kick_once(self, monkeypatch) -> None:
        kick = AsyncMock(return_value=True)
        rc, _ = run_bridge(monkeypatch, "1", kick)
        assert rc == 0
        kick.assert_awaited_once()

    def test_on_logs_what_it_sent_at_info(self, monkeypatch, caplog) -> None:
        # Review L6: the operator is told a re-arm was requested and where its answer is.
        with caplog.at_level(logging.INFO):
            run_bridge(monkeypatch, "1", AsyncMock(return_value=True))
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("re-arm" in m.lower() and "AZ_REARM" in m for m in infos), infos

    @pytest.mark.parametrize("value", [None, "0", ""])
    def test_off_never_calls_it(self, monkeypatch, value) -> None:
        kick = AsyncMock(return_value=True)
        rc, _ = run_bridge(monkeypatch, value, kick)
        assert rc == 0
        kick.assert_not_awaited()

    @pytest.mark.parametrize("error", [FireTunerError("tuner said no"), RuntimeError("odd")])
    def test_a_failing_kick_is_logged_and_the_loop_still_runs(
        self, monkeypatch, caplog, error
    ) -> None:
        kick = AsyncMock(side_effect=error)
        with caplog.at_level(logging.DEBUG):
            rc, bridge = run_bridge(monkeypatch, "true", kick)
        assert rc == 0
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(str(error) in r.getMessage() for r in warnings)
        bridge.stop.assert_awaited_once()
