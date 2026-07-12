"""Agent loop that drives a FileBridge with a PolicyEngine.

W1.1 of the v1.7.0 local-gameplay sprint. This module composes existing
bridge components into a runnable turn-by-turn agent:

    Civ6 Lua mod  ──state.json──>  FileBridge  ──>  BridgeAgentLoop
                                                    │ (StateConverter)
                                                    │
                                                    v
                                                PolicyEngine.select_action
                                                    │
                                                    v
                                                ActionConverter -> LuaCommand
                                                    │
                                  <──commands.json── FileBridge

The loop is async and runs inside the same event loop as the bridge.
MCTS-style policies offload tree expansion to ``asyncio.to_thread`` so
the heartbeat / I/O coroutines stay responsive. Bridge I/O is *always*
on the event-loop thread; policy compute is *always* in the worker
thread - never crossed (see Risk 2 in the sprint plan).

The ``PolicyEngine`` Protocol means MicroCiv, Civ6 (Lua mod), and
Unciv can reuse this loop by swapping the policy + converter pair.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from alpha_zero.bridge.action_converter import ActionConverter, InvalidActionError, NoLegalActionsError
from alpha_zero.bridge.file_bridge import (
    AckTimeoutError,
    ConnectionLostError,
    FileBridge,
    StateTimeoutError,
)
from alpha_zero.bridge.health_checker import (
    CircuitBreakerOpenError,
    GameBridgeHealthChecker,
)
from alpha_zero.bridge.protocol import (
    CommandMessage,
    GameStateData,
    LuaCommand,
    create_end_turn_command,
)
from alpha_zero.bridge.state_converter import GameNotStartedError, StateConverter
from alpha_zero.bridge.utils import extract_scalar_value, parse_bool_env

logger = logging.getLogger(__name__)

# Env-var prefix for AgentLoopConfig overrides. Matches the Civ6LocalConfig
# CIV6AZ_LOCAL_ pattern - see alpha_zero/civ6_local.py:48.
ENV_PREFIX = "CIV6AZ_AGENT_"

_DISCONNECT_LITERAL = ("exit", "reconnect")
DisconnectStrategy = Literal["exit", "reconnect"]

# Reconnect backoff bounds (seconds). Kept as module constants instead of
# magic numbers per the "no hardcoded values" plan constraint - tests can
# patch these if needed.
_RECONNECT_INITIAL_BACKOFF_S = 0.5
_RECONNECT_MAX_BACKOFF_S = 8.0


class PolicyEngine(Protocol):
    """Async policy producer for ``BridgeAgentLoop``.

    Implementations live in ``alpha_zero.bridge.mcts_policy`` (W2.2):
    - ``GreedyNetworkPolicy`` - raw network argmax, no tree search.
    - ``Civ6MctsPolicy`` - GumbelMCTS over a MicroCiv-backed env adapter.
    Future: ``UncivMctsPolicy`` (separate sprint).

    Implementations MUST be async even when the underlying compute is
    synchronous (use ``asyncio.to_thread``) so the loop's event-loop
    thread is never blocked.
    """

    async def select_action(
        self,
        obs: dict[str, np.ndarray],
        legal_mask: np.ndarray,
        converted_state: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, float]]:
        """Choose an action index given an observation and legal mask.

        Args:
            obs: Encoder observation dict (``spatial``, ``vector``, ...).
            legal_mask: ``(action_space_size,)`` float32 mask;
                ``1.0`` = legal, ``0.0`` = illegal.
            converted_state: Full converted state dict from the bridge,
                required by MCTS-based policies to seed their simulation env.

        Returns:
            ``(action_idx, diagnostics)``. The diagnostics dict is logged
            and forwarded to the eval harness (W2.3a) but is not
            interpreted by the loop itself.
        """
        ...


@dataclass
class GreedyNetworkPolicy:
    """Network-only policy that picks the argmax over masked logits.

    Used in W1.1 / W1.2 to prove the loop wiring before W2.2 lands
    ``Civ6MctsPolicy``. The implementation is intentionally minimal -
    actual inference happens in ``asyncio.to_thread`` to maintain the
    "policy compute never on the event loop" invariant.

    Attributes:
        network: Any callable producing ``(policy_logits, value)`` tensors.
            Must expose ``__call__(spatial, vector, action_mask) -> tuple``.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    network: Any
    device: str = "cpu"

    async def select_action(
        self,
        obs: dict[str, np.ndarray],
        legal_mask: np.ndarray,
        converted_state: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, float]]:
        """Run network inference in a worker thread and return masked argmax."""
        # Import torch lazily - module-level import would force the heavy
        # dep on consumers that only need the protocol.
        import torch

        action_idx, value = await asyncio.to_thread(self._infer_sync, obs, legal_mask, torch)
        return action_idx, {
            "value_estimate": float(value),
            "policy_kind": 0.0,  # 0.0 = greedy, 1.0 = mcts (for harness filtering)
        }

    def _infer_sync(
        self,
        obs: dict[str, np.ndarray],
        legal_mask: np.ndarray,
        torch_mod: Any,
    ) -> tuple[int, float]:
        """Synchronous inference body executed inside ``asyncio.to_thread``."""
        spatial = torch_mod.from_numpy(obs["spatial"]).float().unsqueeze(0).to(self.device)
        vector = torch_mod.from_numpy(obs["vector"]).float().unsqueeze(0).to(self.device)
        mask_tensor = torch_mod.from_numpy(legal_mask).float().unsqueeze(0).to(self.device)

        with torch_mod.no_grad():
            policy_logits, value = self.network(spatial, vector, mask_tensor)

        logits = policy_logits.squeeze(0).cpu().numpy().astype(np.float32)
        # Mask via additive -inf so softmax/argmax both ignore illegal moves.
        masked = np.where(legal_mask > 0, logits, -np.inf)
        if not np.isfinite(masked).any():
            # Pathological: no legal action. Fall back to safe end_turn.
            raise NoLegalActionsError("No legal actions available from the mask")

        action_idx = int(np.argmax(masked))
        # Value tensor may be ()/(1,)/(1,1)/(1,N) per the PR #82 invariant;
        # delegate to the shared shape-normaliser so this stays in sync with
        # alpha_zero.inference.inference_engine._apply_policy_postprocessing.
        value_scalar = extract_scalar_value(value)
        return action_idx, value_scalar


@dataclass
class AgentLoopConfig:
    """Runtime knobs for :class:`BridgeAgentLoop`.

    All fields are overridable via ``CIV6AZ_AGENT_*`` env vars so the loop
    can be tuned without code changes - see :meth:`from_env`.

    Attributes:
        max_turns: Hard cap on turns played in a single session. ``0`` means
            unlimited. Default 500 matches the v2.0 spec turn budget.
        per_turn_timeout_s: Maximum time to wait for a single state message
            before raising ``StateTimeoutError``.
        on_disconnect: ``"exit"`` returns from :meth:`BridgeAgentLoop.run`
            with code 2 on a heartbeat loss. ``"reconnect"`` polls
            ``bridge.check_heartbeat`` with exponential backoff up to
            :attr:`reconnect_window_s` before giving up.
        reconnect_window_s: Total time budget for reconnect attempts.
        snapshot_log_dir: When set, dump the offending state / commands /
            ack triplet to this directory on any unhandled exception. The
            loop *continues* to the next turn after dumping.
        mcts_in_thread: Reserved for W2.2 - whether the MCTS policy
            offloads ``search`` to ``asyncio.to_thread``. Always ``True``
            in this release; kept as a config knob for future profiling.
    """

    max_turns: int = 500
    per_turn_timeout_s: float = 30.0
    on_disconnect: DisconnectStrategy = "reconnect"
    reconnect_window_s: float = 60.0
    snapshot_log_dir: Path | None = None
    mcts_in_thread: bool = True
    # Observation-fallback shape used when the StateConverter encoding path
    # raises. Defaults match the 2-player MicroCiv encoder convention; bump
    # ``fallback_spatial_channels`` to 84 when targeting Civ6StateEncoder
    # and ``fallback_board_size`` to whatever the trained network expects.
    fallback_spatial_channels: int = 13
    fallback_vector_dim: int = 64
    fallback_board_size: int = 8

    def __post_init__(self) -> None:
        if self.max_turns < 0:
            raise ValueError(f"max_turns must be >= 0, got {self.max_turns}")
        if self.per_turn_timeout_s <= 0:
            raise ValueError(f"per_turn_timeout_s must be > 0, got {self.per_turn_timeout_s}")
        if self.on_disconnect not in _DISCONNECT_LITERAL:
            raise ValueError(
                f"on_disconnect must be one of {_DISCONNECT_LITERAL}, got {self.on_disconnect!r}"
            )
        if self.reconnect_window_s < 0:
            raise ValueError(f"reconnect_window_s must be >= 0, got {self.reconnect_window_s}")
        if self.fallback_spatial_channels <= 0:
            raise ValueError(
                f"fallback_spatial_channels must be > 0, got {self.fallback_spatial_channels}"
            )
        if self.fallback_vector_dim <= 0:
            raise ValueError(f"fallback_vector_dim must be > 0, got {self.fallback_vector_dim}")
        if self.fallback_board_size < 4:
            raise ValueError(f"fallback_board_size must be >= 4, got {self.fallback_board_size}")
        if self.snapshot_log_dir is not None:
            self.snapshot_log_dir = Path(self.snapshot_log_dir)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> AgentLoopConfig:
        """Build a config from environment variables.

        Recognised keys (all optional):
            CIV6AZ_AGENT_MAX_TURNS
            CIV6AZ_AGENT_PER_TURN_TIMEOUT_S
            CIV6AZ_AGENT_ON_DISCONNECT
            CIV6AZ_AGENT_RECONNECT_WINDOW_S
            CIV6AZ_AGENT_SNAPSHOT_LOG_DIR
            CIV6AZ_AGENT_MCTS_IN_THREAD  ("1"/"0"/"true"/"false")
            CIV6AZ_AGENT_FALLBACK_SPATIAL_CHANNELS
            CIV6AZ_AGENT_FALLBACK_VECTOR_DIM
            CIV6AZ_AGENT_FALLBACK_BOARD_SIZE
        """
        source = env if env is not None else os.environ
        kwargs: dict[str, Any] = {}
        if (raw := source.get(ENV_PREFIX + "MAX_TURNS")) is not None:
            kwargs["max_turns"] = int(raw)
        if (raw := source.get(ENV_PREFIX + "PER_TURN_TIMEOUT_S")) is not None:
            kwargs["per_turn_timeout_s"] = float(raw)
        if (raw := source.get(ENV_PREFIX + "ON_DISCONNECT")) is not None:
            kwargs["on_disconnect"] = raw.lower()
        if (raw := source.get(ENV_PREFIX + "RECONNECT_WINDOW_S")) is not None:
            kwargs["reconnect_window_s"] = float(raw)
        if (raw := source.get(ENV_PREFIX + "SNAPSHOT_LOG_DIR")) is not None:
            kwargs["snapshot_log_dir"] = Path(raw)
        if (raw := source.get(ENV_PREFIX + "MCTS_IN_THREAD")) is not None:
            # Shares vocabulary with BridgeConfig.from_env / Civ6LocalConfig.from_env
            # (1/true/yes -> True, 0/false/no -> False) via parse_bool_env, so
            # CIV6_BRIDGE_*, CIV6AZ_LOCAL_*, and CIV6AZ_AGENT_* env vars all agree
            # on truthy/falsy spelling. Falls back to False for anything else
            # (matches this flag's prior behaviour for unrecognised strings).
            kwargs["mcts_in_thread"] = parse_bool_env(raw, default=False)
        if (raw := source.get(ENV_PREFIX + "FALLBACK_SPATIAL_CHANNELS")) is not None:
            kwargs["fallback_spatial_channels"] = int(raw)
        if (raw := source.get(ENV_PREFIX + "FALLBACK_VECTOR_DIM")) is not None:
            kwargs["fallback_vector_dim"] = int(raw)
        if (raw := source.get(ENV_PREFIX + "FALLBACK_BOARD_SIZE")) is not None:
            kwargs["fallback_board_size"] = int(raw)
        return cls(**kwargs)


@dataclass
class _SessionStats:
    """Internal per-session counters surfaced via :meth:`BridgeAgentLoop.stats`."""

    turns_played: int = 0
    state_timeouts: int = 0
    ack_failures: int = 0
    invalid_actions: int = 0
    reconnects: int = 0
    fallback_end_turns: int = 0
    fallback_zero_observations: int = 0
    turn_latencies_ms: list[float] = field(default_factory=list)
    last_error: str | None = None

    def record_turn(self, latency_ms: float) -> None:
        self.turns_played += 1
        self.turn_latencies_ms.append(latency_ms)

    def as_dict(self) -> dict[str, Any]:
        n = len(self.turn_latencies_ms)
        mean = float(np.mean(self.turn_latencies_ms)) if n else 0.0
        p95 = float(np.percentile(self.turn_latencies_ms, 95)) if n else 0.0
        return {
            "turns_played": self.turns_played,
            "state_timeouts": self.state_timeouts,
            "ack_failures": self.ack_failures,
            "invalid_actions": self.invalid_actions,
            "reconnects": self.reconnects,
            "fallback_end_turns": self.fallback_end_turns,
            "fallback_zero_observations": self.fallback_zero_observations,
            "mean_turn_latency_ms": mean,
            "p95_turn_latency_ms": p95,
            "last_error": self.last_error,
        }


class BridgeAgentLoop:
    """Composes FileBridge + PolicyEngine into a turn-by-turn agent driver.

    See module docstring for the data-flow diagram.

    Lifecycle:
        - Caller is responsible for ``await bridge.start()`` / ``stop()``.
          Typical pattern (see ``run_civ6_bridge`` in W1.2):

              async def _session(bridge, loop):
                  await bridge.start()
                  try:
                      return await loop.run()
                  finally:
                      await bridge.stop()
              asyncio.run(_session(...))

        - This avoids the double-``asyncio.run`` bug fixed in W1.2; the
          loop never opens its own event loop.

    Exit codes returned by :meth:`run`:
        - ``0`` - game ended cleanly (state.is_game_over or max_turns).
        - ``2`` - heartbeat lost and on_disconnect == "exit".
        - ``130`` - cancelled (e.g. KeyboardInterrupt propagated).
    """

    EXIT_OK = 0
    EXIT_DISCONNECTED = 2
    EXIT_CANCELLED = 130

    def __init__(
        self,
        bridge: FileBridge,
        policy: PolicyEngine,
        state_converter: StateConverter,
        action_converter: ActionConverter,
        config: AgentLoopConfig,
        health_checker: GameBridgeHealthChecker | None = None,
        eval_harness: Any | None = None,
    ) -> None:
        self.bridge = bridge
        self.policy = policy
        self.state_converter = state_converter
        self.action_converter = action_converter
        self.config = config
        self.health_checker = health_checker
        self.eval_harness = eval_harness
        self._stats = _SessionStats()
        self._stopped = False

        logger.debug(
            "BridgeAgentLoop initialised: max_turns=%d per_turn_timeout_s=%.1f on_disconnect=%s",
            config.max_turns,
            config.per_turn_timeout_s,
            config.on_disconnect,
        )

    def stop(self) -> None:
        """Request a graceful stop after the current turn completes."""
        self._stopped = True

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of per-session statistics for the eval harness."""
        return self._stats.as_dict()

    async def run(self) -> int:
        """Drive turn-by-turn play until the game ends or a stop is requested.

        Returns:
            Exit code. ``0`` is normal completion; see class-level constants
            for the full list.
        """
        logger.info("BridgeAgentLoop.run() starting")
        if self.eval_harness:
            self.eval_harness.on_start(
                config_dict={
                    "max_turns": self.config.max_turns,
                    "per_turn_timeout_s": self.config.per_turn_timeout_s,
                    "mcts_in_thread": self.config.mcts_in_thread,
                }
            )

        try:
            while not self._stopped:
                if self.config.max_turns and self._stats.turns_played >= self.config.max_turns:
                    logger.info("Reached max_turns=%d; stopping", self.config.max_turns)
                    return self.EXIT_OK

                try:
                    if self.health_checker is not None:
                        state = await self.health_checker.execute_with_retry(
                            self.bridge.wait_for_state,
                            timeout=self.config.per_turn_timeout_s,
                            require_new=True,
                        )
                    else:
                        state = await self.bridge.wait_for_state(
                            timeout=self.config.per_turn_timeout_s,
                            require_new=True,
                        )
                except StateTimeoutError as exc:
                    self._stats.state_timeouts += 1
                    self._stats.last_error = str(exc)
                    logger.warning(
                        "State timeout (#%d) after %.1fs; will retry next iteration",
                        self._stats.state_timeouts,
                        self.config.per_turn_timeout_s,
                    )
                    # Retry on next loop iteration; do NOT abort the whole session
                    # for a single missed state - Civ6 can pause for animations.
                    continue
                except CircuitBreakerOpenError as exc:
                    self._stats.state_timeouts += 1
                    self._stats.last_error = str(exc)
                    logger.warning(
                        "Circuit breaker open; sleeping %.1fs before retry",
                        exc.retry_after,
                    )
                    await asyncio.sleep(max(exc.retry_after, 1.0))
                    continue
                except ConnectionLostError as exc:
                    recovered = await self._handle_disconnect(exc)
                    if not recovered:
                        return self.EXIT_DISCONNECTED
                    continue

                if state.is_game_over:
                    logger.info(
                        "Game reported is_game_over=True (winner_id=%s); stopping",
                        state.winner_id,
                    )
                    return self.EXIT_OK
                await self._step_once(state)

        except asyncio.CancelledError:
            logger.info("BridgeAgentLoop cancelled")
            return self.EXIT_CANCELLED
        finally:
            if self.eval_harness:
                self.eval_harness.on_stop(self.stats())
            logger.info(
                "BridgeAgentLoop finished: turns=%d ack_failures=%d reconnects=%d",
                self._stats.turns_played,
                self._stats.ack_failures,
                self._stats.reconnects,
            )

        return self.EXIT_OK

    async def _step_once(self, state: GameStateData) -> None:
        """Convert -> select -> send -> ack for a single turn.

        Failures are caught and downgraded to a safe ``end_turn`` so a
        bad policy choice or transient bridge error does not crash an
        in-progress game. The turn counter is always incremented in a
        ``finally`` block so a persistent error path (e.g. converter
        always raising) still advances toward ``max_turns`` rather than
        spinning forever.
        """
        turn_start = time.monotonic()
        try:
            await self._step_once_inner(state)
        except Exception as exc:
            # Global safety net: any failure not already handled inside
            # ``_step_once_inner`` (e.g. in mask generation or observation
            # building) is downgraded to a safe ``end_turn`` so a single bad
            # turn cannot crash the whole game loop. ``Exception`` (not
            # ``BaseException``) deliberately lets ``asyncio.CancelledError``
            # propagate so cancellation still works.
            self._stats.last_error = f"step_once: {exc}"
            logger.error("Unhandled exception in turn step: %s", exc, exc_info=True)
            await self._send_safe_end_turn()
        finally:
            latency_ms = (time.monotonic() - turn_start) * 1000.0
            self._stats.record_turn(latency_ms)
            logger.debug(
                "Turn %d completed in %.1fms (total=%d)",
                state.turn_id,
                latency_ms,
                self._stats.turns_played,
            )

    async def _step_once_inner(self, state: GameStateData) -> None:
        """Inner body of :meth:`_step_once`; see that method for semantics."""
        try:
            converted_state = self.state_converter.convert(state)
        except GameNotStartedError as exc:
            # The game isn't fully started or map is completely empty.
            # Do not log a traceback and do NOT send end_turn, just wait.
            logger.debug("Skipping state because game not fully initialized: %s", exc)
            return
        except Exception as exc:
            self._stats.last_error = f"state_converter: {exc}"
            logger.error("State conversion failed: %s", exc, exc_info=True)
            self._maybe_log_snapshot(state, None, None, exc)
            await self._send_safe_end_turn()
            return

        legal_mask = self.action_converter.generate_action_mask(
            state.legal_actions,
            converted_state,
        )

        # Build the observation in the encoder's format. Networks consume
        # spatial + vector + action_mask; the converter has already produced
        # the structural inputs.
        obs = self._build_observation(converted_state, legal_mask)

        try:
            action_idx, diagnostics = await self.policy.select_action(
                obs, legal_mask, converted_state
            )
        except NoLegalActionsError as exc:
            self._stats.invalid_actions += 1
            logger.warning("No legal actions available for turn %d (%s); safely ending turn", state.turn_id, exc)
            await self._send_safe_end_turn()
            return
        except Exception as exc:
            self._stats.last_error = f"policy: {exc}"
            logger.error("Policy selection failed: %s", exc, exc_info=True)
            self._maybe_log_snapshot(state, None, None, exc)
            await self._send_safe_end_turn()
            return

        logger.debug(
            "Turn %d: policy chose action_idx=%d diagnostics=%s",
            state.turn_id,
            action_idx,
            diagnostics,
        )
        if self.eval_harness:
            self.eval_harness.on_turn_end(state.turn_id, diagnostics)

        # Convert action -> Lua command, falling back to end_turn on invalid.
        try:
            lua_command = self.action_converter.convert(
                action_idx,
                converted_state,
                legal_actions=state.legal_actions,
                validate=True,
            )
        except InvalidActionError as exc:
            self._stats.invalid_actions += 1
            logger.warning(
                "Policy chose invalid action %d (%s); falling back to end_turn",
                action_idx,
                exc,
            )
            lua_command = LuaCommand.from_dict({"action_type": "end_turn"})
        except Exception as exc:
            self._stats.last_error = f"action_converter: {exc}"
            logger.error("Action conversion failed: %s", exc, exc_info=True)
            self._maybe_log_snapshot(state, None, None, exc)
            await self._send_safe_end_turn()
            return

        command = CommandMessage.create_single(
            msg_id=self.bridge.next_msg_id(),
            command=lua_command,
        )

        try:
            if self.health_checker is not None:
                ack = await self.health_checker.execute_with_retry(
                    self.bridge.send_and_wait_with_retry,
                    command,
                )
            else:
                ack = await self.bridge.send_and_wait_with_retry(command)
        except CircuitBreakerOpenError as exc:
            self._stats.ack_failures += 1
            self._stats.last_error = f"circuit_open: {exc}"
            logger.warning("Circuit breaker open on send (%s); skipping ack", exc)
            self._maybe_log_snapshot(state, command, None, exc)
        except (AckTimeoutError, ConnectionLostError) as exc:
            self._stats.ack_failures += 1
            self._stats.last_error = f"ack: {exc}"
            logger.warning(
                "Ack failure (#%d): %s; recovering on next turn",
                self._stats.ack_failures,
                exc,
            )
            self._maybe_log_snapshot(state, command, None, exc)
        except Exception as exc:
            self._stats.ack_failures += 1
            self._stats.last_error = f"send: {exc}"
            logger.error("send_and_wait_with_retry raised: %s", exc, exc_info=True)
            self._maybe_log_snapshot(state, command, None, exc)
        else:
            if not ack.success:
                self._stats.invalid_actions += 1
                logger.warning(
                    "Lua mod returned ack.success=False (msg_id=%d, code=%s): %s",
                    ack.msg_id,
                    ack.error_code,
                    ack.error_message,
                )

    async def _handle_disconnect(self, exc: ConnectionLostError) -> bool:
        """Apply the configured disconnect strategy. Returns True if recovered."""
        self._stats.last_error = str(exc)
        if self.config.on_disconnect == "exit":
            logger.error("Heartbeat lost and on_disconnect='exit'; exiting")
            return False

        deadline = time.monotonic() + self.config.reconnect_window_s
        backoff = _RECONNECT_INITIAL_BACKOFF_S
        attempt = 0
        while time.monotonic() < deadline and not self._stopped:
            attempt += 1
            self._stats.reconnects += 1
            logger.warning("Heartbeat lost; reconnect attempt %d (backoff=%.1fs)", attempt, backoff)
            await asyncio.sleep(backoff)
            if self.bridge.check_heartbeat():
                logger.info("Heartbeat recovered after %d attempt(s)", attempt)
                return True
            backoff = min(backoff * 2, _RECONNECT_MAX_BACKOFF_S)

        logger.error(
            "Heartbeat did not recover within %.1fs (%d attempts); giving up",
            self.config.reconnect_window_s,
            attempt,
        )
        return False

    async def _send_safe_end_turn(self) -> None:
        """Send an end_turn fallback. Best-effort; never raises."""
        self._stats.fallback_end_turns += 1
        try:
            cmd = create_end_turn_command(self.bridge.next_msg_id())
            await self.bridge.send_command(cmd)
            logger.debug("Sent safe end_turn fallback (count=%d)", self._stats.fallback_end_turns)
        except Exception as exc:
            logger.error("Even end_turn fallback failed: %s", exc, exc_info=True)
            self._stats.last_error = f"fallback_end_turn: {exc}"

    def _build_observation(
        self,
        converted_state: dict[str, Any],
        legal_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Encode the converted state into the network input format.

        Delegates to ``StateConverter.convert_and_encode`` when available
        (real bridge path) or constructs an inline ``{spatial, vector,
        action_mask}`` dict from the converted state's structural fields
        (used by the unit tests that pre-encode their observations).

        Returns:
            Dict with keys ``spatial`` (CxHxW), ``vector`` (D,) and
            ``action_mask`` (action_space_size,).
        """
        # When tests inject a pre-encoded observation via the converted_state's
        # ``_obs`` shortcut, honour it - lets us exercise the loop without
        # spinning up the heavy encoder.
        if "_obs" in converted_state:
            obs = dict(converted_state["_obs"])
            obs.setdefault("action_mask", legal_mask)
            return obs

        # Real path: re-encode through Civ6StateEncoder via the converter.
        # convert_and_encode returns *unbatched* (spatial, vector) torch
        # tensors - Civ6StateEncoder.encode()/StateConverter.convert_and_encode()
        # never add a leading batch dim, so there is nothing to strip here.
        # (Policy engines such as GreedyNetworkPolicy/Civ6MctsPolicy add the
        # batch dim themselves via unsqueeze(0) before calling the network.)
        try:
            spatial_t, vector_t = self.state_converter.convert_and_encode(
                # `_legal_actions` is the only field convert_and_encode needs
                # beyond what's in converted_state; reuse the dict we built.
                converted_state.get("_raw_state", converted_state)
            )
            return {
                "spatial": spatial_t.cpu().numpy().astype(np.float32),
                "vector": vector_t.cpu().numpy().astype(np.float32),
                "action_mask": legal_mask,
            }
        except Exception as exc:
            # Fallback: zeros with the configured fallback shapes if the
            # converter cannot re-encode. The policy will still produce a
            # (likely random) action, which the loop then masks/validates.
            # Shapes come from AgentLoopConfig.fallback_* (not hardcoded), and
            # board_size additionally honours the live StateConverter when
            # available so it stays correct after a reconfigure.
            #
            # Recorded in _stats (like every other degraded-path in this
            # class) so a persistently failing StateConverter shows up in
            # BridgeAgentLoop.stats() / the eval harness instead of reporting
            # a healthy session while the agent plays blind.
            self._stats.fallback_zero_observations += 1
            self._stats.last_error = f"state_encoding: {exc}"
            shapes = self._resolve_fallback_shapes()
            logger.warning(
                "State encoding failed (%s); using zero observation "
                "(spatial_channels=%d, board_size=%d, vector_dim=%d)",
                exc,
                shapes["spatial_channels"],
                shapes["board_size"],
                shapes["vector_dim"],
            )
            return {
                "spatial": np.zeros(
                    (
                        shapes["spatial_channels"],
                        shapes["board_size"],
                        shapes["board_size"],
                    ),
                    dtype=np.float32,
                ),
                "vector": np.zeros(shapes["vector_dim"], dtype=np.float32),
                "action_mask": legal_mask,
            }

    def _resolve_fallback_shapes(self) -> dict[str, int]:
        """Resolve fallback-observation shapes from config and converter.

        Order of precedence (most specific wins):
          1. ``state_converter.board_size`` if the live converter exposes
             one (covers per-instance overrides without a config reload).
          2. ``AgentLoopConfig.fallback_*`` (env-overridable via
             ``CIV6AZ_AGENT_FALLBACK_*``).
        """
        board_size = int(
            getattr(self.state_converter, "board_size", self.config.fallback_board_size)
            or self.config.fallback_board_size
        )
        return {
            "spatial_channels": self.config.fallback_spatial_channels,
            "vector_dim": self.config.fallback_vector_dim,
            "board_size": board_size,
        }

    def _maybe_log_snapshot(
        self,
        state: GameStateData | None,
        command: CommandMessage | None,
        ack: Any,
        exc: BaseException,
    ) -> None:
        """Dump state/commands/ack triplet for offline debugging."""
        if self.config.snapshot_log_dir is None:
            return
        try:
            ts = int(time.time() * 1000)
            target = self.config.snapshot_log_dir / f"turn{self._stats.turns_played}_{ts}"
            target.mkdir(parents=True, exist_ok=True)

            # error.txt first: it only depends on ``exc`` and must always be
            # recorded, even if a payload below fails to serialise.
            (target / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

            def _dump(name: str, obj: Any) -> None:
                """Write ``obj`` as valid JSON, isolated so one failure can't
                skip the others. Prefers ``to_json``, then ``asdict`` (dataclass),
                then a plain ``json.dumps``; skips the payload if none apply (e.g.
                a test Mock) rather than raising."""
                text: str | None = None
                to_json = getattr(obj, "to_json", None)
                if callable(to_json):
                    try:
                        candidate = to_json()
                        if isinstance(candidate, str):
                            text = candidate
                    except Exception:
                        text = None
                if text is None:
                    for attempt in (
                        lambda: json.dumps(asdict(obj), indent=2, default=str),
                        lambda: json.dumps(obj, indent=2, default=str),
                    ):
                        try:
                            text = attempt()
                            break
                        except TypeError:
                            continue
                if text is None:
                    logger.debug("Snapshot %s skipped (not serialisable)", name)
                    return
                (target / name).write_text(text, encoding="utf-8")

            if state is not None:
                _dump("state.json", state)
            if command is not None:
                _dump("commands.json", command)
            if ack is not None:
                _dump("ack.json", ack)
            logger.debug("Snapshot written to %s", target)
        except Exception as snap_exc:
            logger.warning("Failed to write debug snapshot: %s", snap_exc)
