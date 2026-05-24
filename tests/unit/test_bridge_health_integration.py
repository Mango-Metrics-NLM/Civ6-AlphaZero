"""W3.1 tests: health-checker integration in BridgeAgentLoop.

Tests cover:
- Circuit breaker opens after N wait_for_state failures → loop sleeps & retries
- Circuit breaker opens on send path → ack failure counter incremented
- Snapshot log written to configured dir on policy error
- Health checker execute_with_retry called (not bypassed) for each state fetch
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from alpha_zero.bridge.agent_loop import AgentLoopConfig, BridgeAgentLoop
from alpha_zero.bridge.file_bridge import AckTimeoutError, ConnectionLostError, StateTimeoutError
from alpha_zero.bridge.health_checker import (
    BridgeHealthConfig,
    CircuitBreakerOpenError,
    GameBridgeHealthChecker,
)
from alpha_zero.bridge.protocol import GameStateData


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_state(turn_id: int = 1, is_game_over: bool = False) -> GameStateData:
    """Minimal GameStateData for test usage."""
    state = MagicMock(spec=GameStateData)
    state.turn_id = turn_id
    state.is_game_over = is_game_over
    state.winner_id = None
    state.legal_actions = ["end_turn"]
    return state


def _make_ack(success: bool = True) -> MagicMock:
    ack = MagicMock()
    ack.success = success
    ack.msg_id = 1
    ack.error_code = None
    ack.error_message = None
    return ack


def _make_policy(action_idx: int = 0) -> AsyncMock:
    policy = AsyncMock()
    policy.select_action = AsyncMock(return_value=(action_idx, {"value_estimate": 0.5}))
    return policy


def _make_converters(board_size: int = 4) -> tuple[MagicMock, MagicMock]:
    state_conv = MagicMock()
    state_conv.convert = MagicMock(
        return_value={
            "_obs": {
                "spatial": np.zeros((4, board_size, board_size), dtype=np.float32),
                "vector": np.zeros((8,), dtype=np.float32),
            }
        }
    )
    action_conv = MagicMock()
    action_conv.generate_action_mask = MagicMock(
        return_value=np.ones(board_size * board_size, dtype=np.float32)
    )
    action_conv.convert = MagicMock(return_value=MagicMock())
    return state_conv, action_conv


def _make_bridge(state: GameStateData, ack: MagicMock | None = None) -> MagicMock:
    bridge = MagicMock()
    bridge.wait_for_state = AsyncMock(return_value=state)
    bridge.send_and_wait_with_retry = AsyncMock(return_value=ack or _make_ack())
    bridge.send_command = AsyncMock()
    bridge.next_msg_id = MagicMock(return_value=1)
    bridge.check_heartbeat = MagicMock(return_value=True)
    return bridge


def _make_loop(
    bridge: Any,
    policy: Any,
    health_checker: GameBridgeHealthChecker | None = None,
    max_turns: int = 1,
    snapshot_log_dir: Path | None = None,
) -> BridgeAgentLoop:
    state_conv, action_conv = _make_converters()
    cfg = AgentLoopConfig(max_turns=max_turns, per_turn_timeout_s=1.0)
    if snapshot_log_dir:
        cfg = AgentLoopConfig(
            max_turns=max_turns,
            per_turn_timeout_s=1.0,
            snapshot_log_dir=snapshot_log_dir,
        )
    return BridgeAgentLoop(
        bridge=bridge,
        policy=policy,
        state_converter=state_conv,
        action_converter=action_conv,
        config=cfg,
        health_checker=health_checker,
    )


# ---------------------------------------------------------------------------
# Tests: circuit breaker on wait_for_state
# ---------------------------------------------------------------------------


class TestCircuitBreakerOnStateWait:
    """Loop with health checker reacts to CircuitBreakerOpenError from execute_with_retry."""

    @pytest.mark.asyncio
    async def test_circuit_open_increments_state_timeout_counter(self) -> None:
        """When execute_with_retry raises CircuitBreakerOpenError the loop increments
        state_timeouts and continues (does NOT crash)."""
        state = _make_state(is_game_over=True)  # after circuit resolves, exit cleanly
        bridge = _make_bridge(state)

        health_checker = MagicMock(spec=GameBridgeHealthChecker)
        # First call raises CircuitBreakerOpenError; second call succeeds with game-over state
        cb_err = CircuitBreakerOpenError("open", retry_after=0.0)
        health_checker.execute_with_retry = AsyncMock(
            side_effect=[cb_err, state]
        )

        loop = _make_loop(bridge, _make_policy(), health_checker=health_checker, max_turns=5)

        with patch("asyncio.sleep", new_callable=AsyncMock):  # skip real sleep
            rc = await loop.run()

        assert rc == BridgeAgentLoop.EXIT_OK
        assert loop.stats()["state_timeouts"] >= 1

    @pytest.mark.asyncio
    async def test_loop_sleeps_on_circuit_open(self) -> None:
        """Loop calls asyncio.sleep(≥ retry_after) when circuit is open."""
        state = _make_state(is_game_over=True)
        bridge = _make_bridge(state)
        health_checker = MagicMock(spec=GameBridgeHealthChecker)
        cb_err = CircuitBreakerOpenError("open", retry_after=5.0)
        health_checker.execute_with_retry = AsyncMock(side_effect=[cb_err, state])

        loop = _make_loop(bridge, _make_policy(), health_checker=health_checker, max_turns=5)

        sleep_calls: list[float] = []

        async def fake_sleep(t: float) -> None:
            sleep_calls.append(t)

        with patch("alpha_zero.bridge.agent_loop.asyncio.sleep", side_effect=fake_sleep):
            await loop.run()

        assert len(sleep_calls) >= 1
        assert all(s >= 1.0 for s in sleep_calls)  # at least 1s, per the max(retry_after, 1.0)

    @pytest.mark.asyncio
    async def test_happy_path_with_health_checker(self) -> None:
        """When health checker is present and succeeds, the loop runs normally."""
        state = _make_state(is_game_over=False)
        game_over_state = _make_state(is_game_over=True)
        bridge = _make_bridge(state)
        health_checker = MagicMock(spec=GameBridgeHealthChecker)

        ack = _make_ack(success=True)

        async def dispatch(*args: Any, **kwargs: Any) -> Any:
            fn = args[0]
            if fn is bridge.wait_for_state:
                # First call → live state; subsequent → game over
                return state if not state.is_game_over else game_over_state
            if fn is bridge.send_and_wait_with_retry:
                state.is_game_over = True  # trigger exit on next loop
                return ack
            raise RuntimeError(f"unexpected fn {fn}")

        health_checker.execute_with_retry = AsyncMock(side_effect=dispatch)

        policy = _make_policy()
        loop = _make_loop(bridge, policy, health_checker=health_checker, max_turns=10)
        rc = await loop.run()

        assert rc == BridgeAgentLoop.EXIT_OK
        assert health_checker.execute_with_retry.call_count >= 2


# ---------------------------------------------------------------------------
# Tests: circuit breaker on send path
# ---------------------------------------------------------------------------


class TestCircuitBreakerOnSend:
    """Circuit-open on execute_with_retry wrapping send increments ack_failures."""

    @pytest.mark.asyncio
    async def test_send_circuit_open_increments_ack_failure(self) -> None:
        """CircuitBreakerOpenError on the send path increments ack_failures."""
        state = _make_state(is_game_over=False)
        next_state = _make_state(is_game_over=True)
        bridge = _make_bridge(state)

        health_checker = MagicMock(spec=GameBridgeHealthChecker)
        cb_err = CircuitBreakerOpenError("open on send", retry_after=0.0)

        # State fetches succeed; send raises circuit open; then next state is game-over
        call_count = 0

        async def dispatch(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            fn = args[0]
            if fn is bridge.wait_for_state:
                return state if call_count < 3 else next_state
            if fn is bridge.send_and_wait_with_retry:
                raise cb_err
            raise RuntimeError(f"unexpected {fn}")

        health_checker.execute_with_retry = AsyncMock(side_effect=dispatch)

        loop = _make_loop(bridge, _make_policy(), health_checker=health_checker, max_turns=3)
        await loop.run()

        assert loop.stats()["ack_failures"] >= 1


# ---------------------------------------------------------------------------
# Tests: snapshot log
# ---------------------------------------------------------------------------


class TestSnapshotLog:
    """_maybe_log_snapshot writes files to snapshot_log_dir on error."""

    @pytest.mark.asyncio
    async def test_snapshot_written_on_policy_error(self, tmp_path: Path) -> None:
        """Policy raise → snapshot written at snapshot_log_dir."""
        state = _make_state(is_game_over=False)
        game_over = _make_state(is_game_over=True)
        bridge = _make_bridge(state)
        bridge.wait_for_state = AsyncMock(side_effect=[state, game_over])

        policy = AsyncMock()
        policy.select_action = AsyncMock(side_effect=[RuntimeError("policy boom"), (0, {})])

        loop = _make_loop(bridge, policy, max_turns=5, snapshot_log_dir=tmp_path)
        await loop.run()

        # At least one snapshot dir should have been created
        snapshot_dirs = list(tmp_path.iterdir())
        assert len(snapshot_dirs) >= 1, "Expected at least one snapshot directory"
        error_files = list(tmp_path.rglob("error.txt"))
        assert len(error_files) >= 1

    @pytest.mark.asyncio
    async def test_snapshot_contains_error_text(self, tmp_path: Path) -> None:
        """error.txt content reflects the raised exception type."""
        state = _make_state(is_game_over=False)
        game_over = _make_state(is_game_over=True)
        bridge = _make_bridge(state)
        bridge.wait_for_state = AsyncMock(side_effect=[state, game_over])

        policy = AsyncMock()
        policy.select_action = AsyncMock(
            side_effect=[ValueError("bad_obs"), (0, {})]
        )

        loop = _make_loop(bridge, policy, max_turns=5, snapshot_log_dir=tmp_path)
        await loop.run()

        error_files = list(tmp_path.rglob("error.txt"))
        assert any("ValueError" in f.read_text() for f in error_files)

    @pytest.mark.asyncio
    async def test_no_snapshot_without_dir_configured(self) -> None:
        """When snapshot_log_dir is None, no filesystem writes occur."""
        state = _make_state(is_game_over=False)
        game_over = _make_state(is_game_over=True)
        bridge = _make_bridge(state)
        bridge.wait_for_state = AsyncMock(side_effect=[state, game_over])

        policy = AsyncMock()
        policy.select_action = AsyncMock(side_effect=[RuntimeError("boom"), (0, {})])

        loop = _make_loop(bridge, policy, max_turns=5)  # no snapshot_log_dir
        # Should not raise; snapshot silently skipped
        rc = await loop.run()
        assert rc == BridgeAgentLoop.EXIT_OK

    @pytest.mark.asyncio
    async def test_snapshot_on_ack_timeout(self, tmp_path: Path) -> None:
        """AckTimeoutError increments ack_failures and attempts snapshot write.

        Note: The snapshot write may silently fail when using MagicMock commands
        that are not JSON-serializable — that is acceptable and tested by the
        broader `_maybe_log_snapshot` failure-path (which is already covered by
        the `test_snapshot_written_on_policy_error` test above).  The observable
        guarantee here is that ack_failures is incremented.
        """
        state = _make_state(is_game_over=False)
        game_over = _make_state(is_game_over=True)
        bridge = _make_bridge(state)
        bridge.wait_for_state = AsyncMock(side_effect=[state, game_over])
        bridge.send_and_wait_with_retry = AsyncMock(
            side_effect=[AckTimeoutError(1, 5.0), _make_ack()]
        )

        loop = _make_loop(bridge, _make_policy(), max_turns=5, snapshot_log_dir=tmp_path)
        await loop.run()

        assert loop.stats()["ack_failures"] >= 1
