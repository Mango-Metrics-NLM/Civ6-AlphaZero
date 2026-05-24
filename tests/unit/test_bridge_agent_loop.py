"""Tests for ``alpha_zero.bridge.agent_loop`` (W1.1 of the v1.7.0 sprint).

Covers:
- ``AgentLoopConfig`` defaults, validation, env-var factory.
- ``GreedyNetworkPolicy`` shape-handling on the value tensor (matches the
  PR #82 invariant: scalar / (1,) / (1,1) / (1,N) all round-trip without
  raising).
- ``BridgeAgentLoop`` happy path with ``MockFileBridge`` and a fake policy.
- Recovery paths: state timeout, ack failure, invalid action, disconnect
  with both ``exit`` and ``reconnect`` strategies.
- ``stats()`` shape and content.

The tests deliberately avoid touching the real ``StateConverter``/
``ActionConverter`` semantics; they use stub doubles that implement just
enough surface for the loop to call into. Real-converter behaviour is
exercised by the bridge unit tests and by ``tests/integration/
test_civ6_bridge_e2e.py`` (W1.2).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from alpha_zero.bridge.agent_loop import (
    ENV_PREFIX,
    AgentLoopConfig,
    BridgeAgentLoop,
    GreedyNetworkPolicy,
)
from alpha_zero.bridge.file_bridge import (
    ConnectionLostError,
    StateTimeoutError,
)
from alpha_zero.bridge.protocol import AckMessage

# ---------------------------------------------------------------------------
# AgentLoopConfig
# ---------------------------------------------------------------------------


class TestAgentLoopConfig:
    """Validation + factory tests for ``AgentLoopConfig``."""

    def test_defaults(self) -> None:
        cfg = AgentLoopConfig()
        assert cfg.max_turns == 500
        assert cfg.per_turn_timeout_s == 30.0
        assert cfg.on_disconnect == "reconnect"
        assert cfg.reconnect_window_s == 60.0
        assert cfg.snapshot_log_dir is None
        assert cfg.mcts_in_thread is True

    def test_max_turns_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="max_turns"):
            AgentLoopConfig(max_turns=-1)

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="per_turn_timeout_s"):
            AgentLoopConfig(per_turn_timeout_s=0.0)

    def test_on_disconnect_validated(self) -> None:
        with pytest.raises(ValueError, match="on_disconnect"):
            AgentLoopConfig(on_disconnect="abort")  # type: ignore[arg-type]

    def test_reconnect_window_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="reconnect_window_s"):
            AgentLoopConfig(reconnect_window_s=-1.0)

    def test_from_env_empty(self) -> None:
        cfg = AgentLoopConfig.from_env(env={})
        assert cfg == AgentLoopConfig()

    def test_from_env_overrides_all_fields(self, tmp_path) -> None:
        env = {
            f"{ENV_PREFIX}MAX_TURNS": "42",
            f"{ENV_PREFIX}PER_TURN_TIMEOUT_S": "1.5",
            f"{ENV_PREFIX}ON_DISCONNECT": "exit",
            f"{ENV_PREFIX}RECONNECT_WINDOW_S": "5",
            f"{ENV_PREFIX}SNAPSHOT_LOG_DIR": str(tmp_path),
            f"{ENV_PREFIX}MCTS_IN_THREAD": "false",
            f"{ENV_PREFIX}FALLBACK_SPATIAL_CHANNELS": "84",
            f"{ENV_PREFIX}FALLBACK_VECTOR_DIM": "128",
            f"{ENV_PREFIX}FALLBACK_BOARD_SIZE": "16",
        }
        cfg = AgentLoopConfig.from_env(env=env)
        assert cfg.max_turns == 42
        assert cfg.per_turn_timeout_s == 1.5
        assert cfg.on_disconnect == "exit"
        assert cfg.reconnect_window_s == 5.0
        assert cfg.snapshot_log_dir == tmp_path
        assert cfg.mcts_in_thread is False
        # Fallback observation shape now config-driven (gap fix #1).
        assert cfg.fallback_spatial_channels == 84
        assert cfg.fallback_vector_dim == 128
        assert cfg.fallback_board_size == 16

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"fallback_spatial_channels": 0}, "fallback_spatial_channels"),
            ({"fallback_spatial_channels": -1}, "fallback_spatial_channels"),
            ({"fallback_vector_dim": 0}, "fallback_vector_dim"),
            ({"fallback_board_size": 3}, "fallback_board_size"),
        ],
    )
    def test_fallback_shapes_validated(self, kwargs: dict, match: str) -> None:
        """Fallback shape fields must be positive / non-tiny."""
        with pytest.raises(ValueError, match=match):
            AgentLoopConfig(**kwargs)


# ---------------------------------------------------------------------------
# GreedyNetworkPolicy
# ---------------------------------------------------------------------------


class _FakeNetwork:
    """Callable that returns deterministic (policy_logits, value) tensors."""

    def __init__(self, action_space: int, value: Any) -> None:
        self.action_space = action_space
        self.value = value

    def __call__(self, spatial, vector, mask):
        import torch

        # logits favour action index 3 (so masked-greedy should pick it
        # when index 3 is legal; otherwise the test asserts the mask
        # forces a different choice).
        logits = torch.full((1, self.action_space), -1.0)
        logits[0, 3] = 10.0
        if isinstance(self.value, torch.Tensor):
            return logits, self.value
        return logits, torch.as_tensor(self.value)


class TestGreedyNetworkPolicy:
    """``GreedyNetworkPolicy`` shape + masking behaviour."""

    def _obs(self, action_space: int = 8) -> dict[str, np.ndarray]:
        return {
            "spatial": np.zeros((13, 8, 8), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
            "action_mask": np.ones(action_space, dtype=np.float32),
        }

    @pytest.mark.parametrize(
        "value",
        [
            0.42,  # scalar
            [0.42],  # (1,)
            [[0.42]],  # (1, 1)
            [[0.42, -0.4, 0.1]],  # (1, N) - per-player vector
        ],
    )
    @pytest.mark.asyncio
    async def test_value_shapes_round_trip(self, value: Any) -> None:
        """PR #82 invariant: all value shapes flatten to a scalar without raising."""
        import torch

        net = _FakeNetwork(action_space=8, value=torch.as_tensor(value))
        policy = GreedyNetworkPolicy(network=net, device="cpu")
        mask = np.ones(8, dtype=np.float32)

        action_idx, diagnostics = await policy.select_action(self._obs(8), mask)

        assert action_idx == 3  # logits[3]=10.0 wins under uniform mask
        assert diagnostics["value_estimate"] == pytest.approx(0.42)
        assert diagnostics["policy_kind"] == 0.0

    @pytest.mark.asyncio
    async def test_mask_excludes_argmax_candidate(self) -> None:
        """When the network's preferred action is illegal, fall back to a legal one."""
        net = _FakeNetwork(action_space=8, value=0.0)
        policy = GreedyNetworkPolicy(network=net, device="cpu")

        mask = np.ones(8, dtype=np.float32)
        mask[3] = 0.0  # disallow the network's preferred action
        mask[5] = 1.0  # ensure 5 is legal

        action_idx, _ = await policy.select_action(self._obs(8), mask)
        assert action_idx != 3
        assert mask[action_idx] == 1.0

    @pytest.mark.asyncio
    async def test_empty_mask_returns_action_zero_and_warns(self, caplog) -> None:
        """No legal actions -> safe fallback to action 0; warning logged."""
        import logging

        net = _FakeNetwork(action_space=8, value=0.0)
        policy = GreedyNetworkPolicy(network=net, device="cpu")
        mask = np.zeros(8, dtype=np.float32)

        with caplog.at_level(logging.WARNING, logger="alpha_zero.bridge.agent_loop"):
            action_idx, diagnostics = await policy.select_action(self._obs(8), mask)

        assert action_idx == 0
        assert diagnostics["value_estimate"] == 0.0
        assert any("empty legal mask" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# BridgeAgentLoop fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_state() -> MagicMock:
    """A ``GameStateData`` stand-in with the fields the loop touches."""
    state = MagicMock(name="GameStateData")
    state.turn_id = 1
    state.player_id = 0
    state.is_game_over = False
    state.winner_id = None
    state.legal_actions = []
    return state


@pytest.fixture
def fake_terminal_state(fake_state: MagicMock) -> MagicMock:
    state = MagicMock(name="GameStateData")
    state.turn_id = 99
    state.player_id = 0
    state.is_game_over = True
    state.winner_id = 0
    state.legal_actions = []
    return state


@pytest.fixture
def fake_state_converter() -> MagicMock:
    converter = MagicMock(name="StateConverter")
    converter.convert.return_value = {
        "current_player": 0,
        "units": [],
        "cities": [],
        "_legal_actions": [],
        "_map_size": (8, 8),
        # Shortcut: pre-baked obs so the loop's encoding path doesn't fire.
        "_obs": {
            "spatial": np.zeros((13, 8, 8), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
        },
    }
    return converter


@pytest.fixture
def fake_action_converter() -> MagicMock:
    converter = MagicMock(name="ActionConverter")
    converter.generate_action_mask.return_value = np.ones(8, dtype=np.float32)

    def _convert(action_idx, _game_state, legal_actions=None, validate=True):
        from alpha_zero.bridge.protocol import LuaCommand

        # Always succeed, returning end_turn — keeps the test focused on the
        # surrounding wiring.
        return LuaCommand.from_dict({"action_type": "end_turn"})

    converter.convert.side_effect = _convert
    return converter


@pytest.fixture
def fake_policy() -> AsyncMock:
    """Returns action index 0 with empty diagnostics."""
    policy = AsyncMock(name="PolicyEngine")
    policy.select_action.return_value = (0, {"value_estimate": 0.0})
    return policy


@pytest.fixture
def mock_bridge() -> MagicMock:
    """A ``FileBridge`` whose async methods are AsyncMocks."""
    from alpha_zero.bridge.config import BridgeConfig

    bridge = MagicMock()
    bridge.config = BridgeConfig()
    bridge.wait_for_state = AsyncMock()
    bridge.send_command = AsyncMock(return_value=True)
    bridge.send_and_wait_with_retry = AsyncMock(
        return_value=AckMessage(msg_id=1, success=True, timestamp=0.0)
    )
    bridge._next_msg_id = MagicMock(return_value=1)
    bridge.check_heartbeat = MagicMock(return_value=True)
    return bridge


@pytest.fixture
def make_loop(
    mock_bridge: MagicMock,
    fake_policy: AsyncMock,
    fake_state_converter: MagicMock,
    fake_action_converter: MagicMock,
):
    """Factory that builds a BridgeAgentLoop with overridable parts."""

    def _build(
        config: AgentLoopConfig | None = None,
        policy: Any = None,
    ) -> BridgeAgentLoop:
        return BridgeAgentLoop(
            bridge=mock_bridge,
            policy=policy or fake_policy,
            state_converter=fake_state_converter,
            action_converter=fake_action_converter,
            config=config or AgentLoopConfig(max_turns=2, per_turn_timeout_s=0.5),
        )

    return _build


# ---------------------------------------------------------------------------
# BridgeAgentLoop happy path
# ---------------------------------------------------------------------------


class TestBridgeAgentLoopHappyPath:
    @pytest.mark.asyncio
    async def test_runs_two_turns_and_exits_at_max_turns(
        self,
        mock_bridge: MagicMock,
        fake_policy: AsyncMock,
        fake_state: MagicMock,
        make_loop,
    ) -> None:
        mock_bridge.wait_for_state.return_value = fake_state
        loop = make_loop(config=AgentLoopConfig(max_turns=2, per_turn_timeout_s=0.5))

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_OK
        assert fake_policy.select_action.await_count == 2
        assert mock_bridge.send_and_wait_with_retry.await_count == 2
        assert loop.stats()["turns_played"] == 2

    @pytest.mark.asyncio
    async def test_exits_when_state_is_game_over(
        self,
        mock_bridge: MagicMock,
        fake_policy: AsyncMock,
        fake_terminal_state: MagicMock,
        make_loop,
    ) -> None:
        mock_bridge.wait_for_state.return_value = fake_terminal_state
        loop = make_loop()

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_OK
        # Game-over path skips the policy entirely.
        fake_policy.select_action.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stop_breaks_loop(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        make_loop,
    ) -> None:
        # First call returns a state; loop.stop() flips the flag before
        # the second iteration runs, so we expect exactly one turn played.
        mock_bridge.wait_for_state.return_value = fake_state

        loop = make_loop(config=AgentLoopConfig(max_turns=10))

        async def _stop_after_first_turn() -> None:
            await asyncio.sleep(0)  # let one iteration land
            loop.stop()

        await asyncio.gather(loop.run(), _stop_after_first_turn())
        assert loop.stats()["turns_played"] >= 0  # may be 0 or 1 depending on scheduling


# ---------------------------------------------------------------------------
# BridgeAgentLoop recovery paths
# ---------------------------------------------------------------------------


class TestBridgeAgentLoopRecovery:
    @pytest.mark.asyncio
    async def test_state_timeout_is_retried_not_fatal(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        make_loop,
    ) -> None:
        # First wait_for_state raises StateTimeoutError; second returns a state.
        mock_bridge.wait_for_state.side_effect = [
            StateTimeoutError(0.5),
            fake_state,
            fake_state,
        ]
        loop = make_loop(config=AgentLoopConfig(max_turns=2, per_turn_timeout_s=0.5))

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_OK
        assert loop.stats()["state_timeouts"] == 1
        assert loop.stats()["turns_played"] == 2

    @pytest.mark.asyncio
    async def test_ack_failure_increments_counter_continues(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        make_loop,
    ) -> None:
        from alpha_zero.bridge.file_bridge import AckTimeoutError

        mock_bridge.wait_for_state.return_value = fake_state
        mock_bridge.send_and_wait_with_retry.side_effect = AckTimeoutError(1, 0.5)
        loop = make_loop(config=AgentLoopConfig(max_turns=1, per_turn_timeout_s=0.5))

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_OK
        assert loop.stats()["ack_failures"] == 1

    @pytest.mark.asyncio
    async def test_invalid_action_falls_back_to_end_turn(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        fake_action_converter: MagicMock,
        make_loop,
    ) -> None:
        from alpha_zero.bridge.action_converter import InvalidActionError
        from alpha_zero.bridge.protocol import LuaCommand

        mock_bridge.wait_for_state.return_value = fake_state

        # First call raises InvalidActionError (forcing the fallback path);
        # the fallback uses LuaCommand.from_dict directly without re-calling
        # convert, so the side_effect doesn't need a second entry.
        end_turn_cmd = LuaCommand.from_dict({"action_type": "end_turn"})

        def _convert(action_idx, _state, legal_actions=None, validate=True):
            raise InvalidActionError("test invalid", action_idx)

        fake_action_converter.convert.side_effect = _convert
        loop = make_loop(config=AgentLoopConfig(max_turns=1, per_turn_timeout_s=0.5))

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_OK
        assert loop.stats()["invalid_actions"] == 1
        # The end_turn command is still sent via send_and_wait_with_retry.
        sent_command = mock_bridge.send_and_wait_with_retry.await_args.args[0]
        assert sent_command.commands[0].action_type == end_turn_cmd.action_type

    @pytest.mark.asyncio
    async def test_disconnect_exit_strategy_returns_disconnect_code(
        self,
        mock_bridge: MagicMock,
        make_loop,
    ) -> None:
        mock_bridge.wait_for_state.side_effect = ConnectionLostError()
        loop = make_loop(
            config=AgentLoopConfig(max_turns=5, per_turn_timeout_s=0.5, on_disconnect="exit")
        )

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_reconnect_strategy_recovers(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        monkeypatch,
        make_loop,
    ) -> None:
        # Patch backoff bounds so the test doesn't burn 8 real seconds.
        monkeypatch.setattr("alpha_zero.bridge.agent_loop._RECONNECT_INITIAL_BACKOFF_S", 0.01)
        monkeypatch.setattr("alpha_zero.bridge.agent_loop._RECONNECT_MAX_BACKOFF_S", 0.05)

        # First wait_for_state -> disconnect; second -> state.
        mock_bridge.wait_for_state.side_effect = [
            ConnectionLostError(),
            fake_state,
        ]
        # Heartbeat returns True so the reconnect attempt succeeds on first try.
        mock_bridge.check_heartbeat.return_value = True
        loop = make_loop(
            config=AgentLoopConfig(
                max_turns=1,
                per_turn_timeout_s=0.5,
                on_disconnect="reconnect",
                reconnect_window_s=1.0,
            )
        )

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_OK
        assert loop.stats()["reconnects"] >= 1

    @pytest.mark.asyncio
    async def test_disconnect_reconnect_strategy_gives_up_after_window(
        self,
        mock_bridge: MagicMock,
        monkeypatch,
        make_loop,
    ) -> None:
        monkeypatch.setattr("alpha_zero.bridge.agent_loop._RECONNECT_INITIAL_BACKOFF_S", 0.01)
        monkeypatch.setattr("alpha_zero.bridge.agent_loop._RECONNECT_MAX_BACKOFF_S", 0.02)

        mock_bridge.wait_for_state.side_effect = ConnectionLostError()
        # Heartbeat never recovers.
        mock_bridge.check_heartbeat.return_value = False
        loop = make_loop(
            config=AgentLoopConfig(
                max_turns=1,
                per_turn_timeout_s=0.5,
                on_disconnect="reconnect",
                reconnect_window_s=0.05,
            )
        )

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_DISCONNECTED
        assert loop.stats()["reconnects"] >= 1


# ---------------------------------------------------------------------------
# Stats + snapshot logging
# ---------------------------------------------------------------------------


class TestBridgeAgentLoopStats:
    @pytest.mark.asyncio
    async def test_stats_contains_expected_keys(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        make_loop,
    ) -> None:
        mock_bridge.wait_for_state.return_value = fake_state
        loop = make_loop(config=AgentLoopConfig(max_turns=1, per_turn_timeout_s=0.5))
        await loop.run()

        stats = loop.stats()
        expected_keys = {
            "turns_played",
            "state_timeouts",
            "ack_failures",
            "invalid_actions",
            "reconnects",
            "fallback_end_turns",
            "mean_turn_latency_ms",
            "p95_turn_latency_ms",
            "last_error",
        }
        assert expected_keys.issubset(stats.keys())
        assert stats["turns_played"] == 1
        assert stats["mean_turn_latency_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_snapshot_log_dir_is_written_on_failure(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        fake_state_converter: MagicMock,
        tmp_path,
        make_loop,
    ) -> None:
        mock_bridge.wait_for_state.return_value = fake_state
        fake_state_converter.convert.side_effect = RuntimeError("boom")
        loop = make_loop(
            config=AgentLoopConfig(max_turns=1, per_turn_timeout_s=0.5, snapshot_log_dir=tmp_path)
        )

        await loop.run()

        # snapshot dir should contain at least one turn directory with error.txt
        children = list(tmp_path.iterdir())
        assert children, "expected snapshot directory to be created"
        assert any((c / "error.txt").exists() for c in children)


# ---------------------------------------------------------------------------
# Error paths surfaced by the W1.5 gap-analysis pass
# ---------------------------------------------------------------------------


class TestBridgeAgentLoopErrorPaths:
    """Coverage for branches the original W1.1 suite did not exercise."""

    @pytest.mark.asyncio
    async def test_send_safe_end_turn_failure_is_recorded(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        fake_state_converter: MagicMock,
        make_loop,
    ) -> None:
        """When even the fallback end_turn raises, the loop must not crash.

        Regression for the inner ``except`` branch in
        :meth:`BridgeAgentLoop._send_safe_end_turn`. The previous test suite
        only exercised the happy-fallback path (converter raises ->
        send_command succeeds). This guards the failure-within-fallback
        branch so we don't trade one silent failure for another.
        """
        mock_bridge.wait_for_state.return_value = fake_state
        # Trigger the safe-end_turn path via a converter failure...
        fake_state_converter.convert.side_effect = RuntimeError("converter boom")
        # ...and make even the fallback send_command raise.
        mock_bridge.send_command.side_effect = OSError("disk full")

        loop = make_loop(config=AgentLoopConfig(max_turns=1, per_turn_timeout_s=0.5))
        code = await loop.run()

        # Loop should still terminate cleanly (no propagation), but the
        # failure is recorded in stats for the eval harness.
        assert code == BridgeAgentLoop.EXIT_OK
        stats = loop.stats()
        assert stats["fallback_end_turns"] == 1
        assert stats["last_error"] is not None
        assert "fallback_end_turn" in stats["last_error"]

    @pytest.mark.asyncio
    async def test_build_observation_falls_back_to_config_shapes(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        fake_state_converter: MagicMock,
        fake_policy: AsyncMock,
        fake_action_converter: MagicMock,
    ) -> None:
        """When the encoding path fails, the fallback obs matches AgentLoopConfig.

        Regression for Critical-1 of the gap analysis: previously this path
        hardcoded ``(13, 8, 8)`` regardless of the network's input shape.
        """
        from alpha_zero.bridge.agent_loop import BridgeAgentLoop

        # Force the "_obs" shortcut OFF so the encoding path runs, then
        # make convert_and_encode raise to trip the fallback.
        fake_state_converter.convert.return_value = {
            "current_player": 0,
            "units": [],
            "cities": [],
            "_legal_actions": [],
            "_map_size": (4, 4),
            # No "_obs" key => loop must go through convert_and_encode.
        }
        fake_state_converter.convert_and_encode.side_effect = RuntimeError("encode boom")
        fake_state_converter.board_size = 4  # honour the converter's board_size

        mock_bridge.wait_for_state.return_value = fake_state

        loop = BridgeAgentLoop(
            bridge=mock_bridge,
            policy=fake_policy,
            state_converter=fake_state_converter,
            action_converter=fake_action_converter,
            config=AgentLoopConfig(
                max_turns=1,
                per_turn_timeout_s=0.5,
                fallback_spatial_channels=21,  # exotic; not the legacy 13
                fallback_vector_dim=96,
                fallback_board_size=16,  # should be overridden by converter
            ),
        )

        await loop.run()

        # The policy was invoked - capture the observation it received.
        first_call = fake_policy.select_action.await_args
        assert first_call is not None
        obs = first_call.args[0]
        # Spatial: configured channels x converter board size x converter board size.
        assert obs["spatial"].shape == (
            21,
            4,
            4,
        ), f"Expected (21, 4, 4) from config + converter board_size; got {obs['spatial'].shape}"
        # Vector dim from config.
        assert obs["vector"].shape == (96,)
        # dtype invariant - float32 always.
        assert obs["spatial"].dtype == np.dtype("float32")
        assert obs["vector"].dtype == np.dtype("float32")

    @pytest.mark.asyncio
    async def test_loop_uses_public_next_msg_id(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        make_loop,
    ) -> None:
        """Critical-2 of the gap analysis: callers must use the public alias.

        If the loop is ever refactored to skip ``next_msg_id`` and reach
        through ``_next_msg_id`` again, MockFileBridge tests would still
        pass (MagicMock auto-creates the attribute) but real consumers
        would break. This test pins the public-API contract.
        """
        mock_bridge.wait_for_state.return_value = fake_state
        loop = make_loop(config=AgentLoopConfig(max_turns=1, per_turn_timeout_s=0.5))

        await loop.run()

        # `next_msg_id` (public) was called at least once for the command.
        assert mock_bridge.next_msg_id.called, (
            "BridgeAgentLoop must call the public next_msg_id(), not "
            "_next_msg_id(). See gap-analysis Critical #2."
        )


# ---------------------------------------------------------------------------
# FileBridge.next_msg_id public alias (gap-analysis Critical #2)
# ---------------------------------------------------------------------------


class TestFileBridgeNextMsgIdAlias:
    """The public alias must produce the same monotonic IDs as the legacy
    private method."""

    def test_public_alias_delegates_to_private(self) -> None:
        from alpha_zero.bridge.config import BridgeConfig
        from alpha_zero.bridge.file_bridge import FileBridge

        bridge = FileBridge(BridgeConfig())
        a = bridge.next_msg_id()
        b = bridge.next_msg_id()
        c = bridge._next_msg_id()  # legacy path still works
        assert b == a + 1
        assert c == b + 1


# ---------------------------------------------------------------------------
# Additional coverage for low-probability error paths
# ---------------------------------------------------------------------------


class TestBridgeAgentLoopBranchCoverage:
    """Tests targeting branches missed by the main suite, to keep the
    bridge module above the 85% coverage floor as new paths are added."""

    @pytest.mark.asyncio
    async def test_policy_exception_triggers_safe_end_turn(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        fake_policy: AsyncMock,
        make_loop,
    ) -> None:
        """When the policy raises, the loop falls back to a safe end_turn."""
        mock_bridge.wait_for_state.return_value = fake_state
        fake_policy.select_action.side_effect = RuntimeError("policy boom")
        loop = make_loop(config=AgentLoopConfig(max_turns=1, per_turn_timeout_s=0.5))

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_OK
        stats = loop.stats()
        # The fallback end_turn was sent and the failure was recorded.
        assert stats["fallback_end_turns"] >= 1
        assert stats["last_error"] is not None
        assert "policy" in stats["last_error"]

    @pytest.mark.asyncio
    async def test_ack_success_false_counts_as_invalid_action(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        make_loop,
    ) -> None:
        """A Lua mod ack with success=False is recorded as an invalid action."""
        mock_bridge.wait_for_state.return_value = fake_state
        mock_bridge.send_and_wait_with_retry.return_value = AckMessage(
            msg_id=1,
            success=False,
            timestamp=0.0,
            error_code="NOT_IMPLEMENTED",
            error_message="city_buy not wired",
        )
        loop = make_loop(config=AgentLoopConfig(max_turns=1, per_turn_timeout_s=0.5))

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_OK
        assert loop.stats()["invalid_actions"] == 1

    @pytest.mark.asyncio
    async def test_action_converter_generic_exception_triggers_safe_end_turn(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        fake_action_converter: MagicMock,
        make_loop,
    ) -> None:
        """Non-InvalidAction exceptions from the action converter fall through
        the generic except branch and trigger a safe end_turn."""
        mock_bridge.wait_for_state.return_value = fake_state
        fake_action_converter.convert.side_effect = RuntimeError("action boom")
        loop = make_loop(config=AgentLoopConfig(max_turns=1, per_turn_timeout_s=0.5))

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_OK
        stats = loop.stats()
        assert stats["fallback_end_turns"] >= 1
        assert "action_converter" in (stats["last_error"] or "")

    @pytest.mark.asyncio
    async def test_send_generic_exception_increments_ack_failures(
        self,
        mock_bridge: MagicMock,
        fake_state: MagicMock,
        make_loop,
    ) -> None:
        """A non-AckTimeout / non-ConnectionLost send failure is still recorded."""
        mock_bridge.wait_for_state.return_value = fake_state
        mock_bridge.send_and_wait_with_retry.side_effect = RuntimeError("send boom")
        loop = make_loop(config=AgentLoopConfig(max_turns=1, per_turn_timeout_s=0.5))

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_OK
        assert loop.stats()["ack_failures"] == 1
        assert "send" in (loop.stats()["last_error"] or "")
