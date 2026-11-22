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
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, cast

import numpy as np

from alpha_zero.bridge.action_converter import (
    ActionConverter,
    InvalidActionError,
    NoLegalActionsError,
)
from alpha_zero.bridge.agent_loop_config import ENV_PREFIX as ENV_PREFIX
from alpha_zero.bridge.agent_loop_config import AgentLoopConfig as AgentLoopConfig
from alpha_zero.bridge.agent_loop_config import DisconnectStrategy as DisconnectStrategy
from alpha_zero.bridge.command_policy import choose, diagnostics_of, is_command_policy
from alpha_zero.bridge.file_bridge import (
    AckTimeoutError,
    ConnectionLostError,
    StateTimeoutError,
)
from alpha_zero.bridge.health_checker import (
    CircuitBreakerOpenError,
    GameBridgeHealthChecker,
)
from alpha_zero.bridge.offer_filter import first_to_unblock, sendable, vetoed
from alpha_zero.bridge.protocol import (
    END_TURN_BLOCKED,
    AckMessage,
    CommandMessage,
    GameStateData,
    LuaCommand,
    create_end_turn_command,
)
from alpha_zero.bridge.state_converter import GameNotStartedError, StateConverter
from alpha_zero.bridge.turn_ledger import LedgerLimits, TurnLedger
from alpha_zero.bridge.utils import extract_scalar_value

logger = logging.getLogger(__name__)

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


class GameBridge(Protocol):
    """Transport interface required by :class:`BridgeAgentLoop`.

    Structural protocol capturing exactly the surface the loop uses, so any
    transport (``FileBridge`` file IPC, ``LiveBridge`` FireTuner socket, or a
    test double) can drive the loop without inheriting from ``FileBridge``.
    New transports implement this protocol; the loop is never edited per
    transport (charter Invariant 1).
    """

    def next_msg_id(self) -> int:
        """Return the next monotonically increasing command message id."""
        ...

    def check_heartbeat(self) -> bool:
        """Return True while the game-side mod is alive."""
        ...

    async def wait_for_state(
        self,
        timeout: float | None = None,
        require_new: bool = True,
    ) -> GameStateData:
        """Block until a (new) game state arrives or *timeout* elapses."""
        ...

    async def send_command(self, command: CommandMessage) -> bool:
        """Fire-and-forget command send; True if written."""
        ...

    async def send_and_wait_with_retry(
        self,
        command: CommandMessage,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> Any:
        """Send a command and await its acknowledgment with retry/backoff."""
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


def _others_left(mask: Any, action_idx: Any) -> bool:
    """Whether the mask offers anything besides ``action_idx``."""
    try:
        offered = int(np.count_nonzero(mask))
        return offered - (1 if mask[action_idx] else 0) > 0
    except Exception:  # a mask or an index that cannot be read offers nothing else
        return False


def _is_offered(mask: Any, action_idx: Any) -> bool:
    """Whether ``action_idx`` is in the mask (a policy may answer outside it)."""
    try:
        return bool(mask[action_idx])
    except Exception:
        return False


def _as_ack_message(ack: Any) -> AckMessage:
    """The transport's acknowledgement as an :class:`AckMessage`.

    ``FileBridge`` returns an ``AckMessage``; ``LiveBridge`` returns the parsed
    ``AZ_ACK`` JSON as a dict. Reading ``.success`` on the dict raised, and the
    step's catch-all then sent a second ``end_turn`` after every acknowledged
    command on the socket transport.

    Raises:
        TypeError: If the transport returned neither shape.
    """
    if isinstance(ack, AckMessage):
        return ack
    if isinstance(ack, dict):
        return AckMessage.from_dict(ack)
    if hasattr(ack, "success"):
        # A test double or a transport's own ack type that reads like one.
        return cast(AckMessage, ack)
    raise TypeError(f"unusable acknowledgement of type {type(ack).__name__}")


def _blocked_end_turn_message(ack: Any) -> str | None:
    """The mod's message when ``ack`` reports a blocked end of turn, else None.

    The code arrives per command, in ``ack.executed``; the ack's own
    ``error_code`` is a generic one. A blocked report without a message still
    counts, so it yields a placeholder.
    """
    executed = getattr(ack, "executed", None)
    if not isinstance(executed, list):
        return None
    for entry in executed:
        if isinstance(entry, dict) and entry.get("error_code") == END_TURN_BLOCKED:
            message = entry.get("message")
            return message if isinstance(message, str) and message else END_TURN_BLOCKED
    return None


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
    # end_turn commands the game refused because a pending choice blocks the
    # turn (the mod's END_TURN_BLOCKED). Not an invalid action: the command was
    # well-formed and the mod keeps listening.
    blocked_end_turns: int = 0
    last_end_turn_blocker: str | None = None
    # Commands sent (several per game turn), choices the policy was asked to
    # make again, and game turns given up as stalled (GM-B7c-2).
    commands_sent: int = 0
    reselections: int = 0
    stalled_turns: int = 0
    # A command policy (GM-B6b): times asked, named nothing, raised.
    search_decisions: int = 0
    search_fallbacks: int = 0
    search_failures: int = 0
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
            "blocked_end_turns": self.blocked_end_turns,
            "last_end_turn_blocker": self.last_end_turn_blocker,
            "commands_sent": self.commands_sent,
            "reselections": self.reselections,
            "stalled_turns": self.stalled_turns,
            "search_decisions": self.search_decisions,
            "search_fallbacks": self.search_fallbacks,
            "search_failures": self.search_failures,
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
        - ``3`` - a game turn could neither be ended nor unblocked (a stall).
        - ``130`` - cancelled (e.g. KeyboardInterrupt propagated).
    """

    EXIT_OK = 0
    EXIT_DISCONNECTED = 2
    EXIT_STALLED = 3
    EXIT_CANCELLED = 130

    def __init__(
        self,
        bridge: GameBridge,
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
        # What the game did with this game turn's commands (GM-B7c-2).
        self._ledger = TurnLedger(LedgerLimits(max_commands_per_turn=config.max_commands_per_turn))
        self._current_state: GameStateData | None = None
        self._stall_exit = False
        self._stalled_turn: Any = None
        self._stall_deadline = 0.0

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
        stats = self._stats.as_dict()
        ledger = self._ledger.stats()
        for key in ("game_turns_played", "refused_commands", "no_effect_commands"):
            stats[key] = ledger[key]
        # The ledger also sees a fallback end_turn blocked (from the next state).
        stats["blocked_end_turns"] = max(stats["blocked_end_turns"], ledger["blocked_end_turns"])
        return stats

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
                if self._stalled_turn is not None and time.monotonic() > self._stall_deadline:
                    logger.error("Stalled turn was not ended by hand in time; stopping")
                    return self.EXIT_STALLED

                try:
                    if self.health_checker is not None:
                        state: GameStateData = await self.health_checker.execute_with_retry(
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

                self._ledger.observe(state)
                finished = self._ledger.game_turns_finished
                if self.config.max_game_turns and finished >= self.config.max_game_turns:
                    logger.info("Reached max_game_turns=%d; stopping", self.config.max_game_turns)
                    return self.EXIT_OK
                if self._stalled_turn is not None:
                    if state.turn_id == self._stalled_turn:
                        continue  # on_stall="wait": nothing more is sent this turn
                    self._stalled_turn = None
                self._current_state = state
                await self._step_once(state)
                self._check_stall(state)
                if self._stall_exit:
                    return self.EXIT_STALLED

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

        lua_command = await self._choose_command(state, converted_state, legal_mask)
        if lua_command is not None:
            await self._send_and_account(state, lua_command)

    async def _choose_command(
        self,
        state: GameStateData,
        converted_state: dict[str, Any],
        legal_mask: np.ndarray,
    ) -> LuaCommand | None:
        """The command to send for ``state``; None when a failure was already handled.

        The policy's choice is not sent, and the policy is asked again with
        that action masked out, when the converter rejects it, when the game
        refused it (or ignored it) earlier this game turn, or when it is
        ``end_turn`` while the game is blocking the end of turn and something
        else is offered. Past the turn's command cap only ``end_turn`` is sent.
        """
        end_turn = LuaCommand.from_dict({"action_type": "end_turn"})
        unblocking_only = False
        if self._ledger.over_cap():
            if not self._ledger.may_unblock():
                logger.warning(
                    "Turn %s: %d commands sent, the cap; only end_turn from here",
                    state.turn_id,
                    self._ledger.commands_this_turn,
                )
                return end_turn
            # The game is blocking end_turn: what could unblock it still goes.
            unblocking_only = True

        def convert(index: int) -> LuaCommand:
            return self.action_converter.convert(
                index, converted_state, legal_actions=state.legal_actions, validate=True
            )

        mask, ready = sendable(self._ledger, convert, legal_mask, unblocking_only)
        if not np.any(mask):
            return end_turn
        if is_command_policy(self.policy):
            chosen = await self._ask_command_policy(state, converted_state, unblocking_only)
            return chosen if chosen is not None else self._unanswered(ready, end_turn)

        for attempt in range(self.config.max_reselections + 1):
            # Networks consume spatial + vector + action_mask; the converter
            # has already produced the structural inputs.
            obs = self._build_observation(converted_state, mask)
            try:
                action_idx, diagnostics = await self.policy.select_action(
                    obs, mask, converted_state
                )
            except NoLegalActionsError as exc:
                self._stats.invalid_actions += 1
                logger.warning(
                    "No legal actions available for turn %d (%s); safely ending turn",
                    state.turn_id,
                    exc,
                )
                await self._send_safe_end_turn()
                return None
            except Exception as exc:
                self._stats.last_error = f"policy: {exc}"
                logger.error("Policy selection failed: %s", exc, exc_info=True)
                self._maybe_log_snapshot(state, None, None, exc)
                await self._send_safe_end_turn()
                return None

            logger.debug(
                "Turn %d: policy chose action_idx=%d diagnostics=%s",
                state.turn_id,
                action_idx,
                diagnostics,
            )
            if self.eval_harness and attempt == 0:
                self.eval_harness.on_turn_end(state.turn_id, diagnostics)

            if action_idx in ready:
                return cast(LuaCommand, ready[action_idx])
            # Convert action -> Lua command.
            try:
                lua_command = self.action_converter.convert(
                    action_idx,
                    converted_state,
                    legal_actions=state.legal_actions,
                    validate=True,
                )
            except InvalidActionError as exc:
                self._stats.invalid_actions += 1
                why = f"invalid ({exc})"
            except Exception as exc:
                self._stats.last_error = f"action_converter: {exc}"
                logger.error("Action conversion failed: %s", exc, exc_info=True)
                self._maybe_log_snapshot(state, None, None, exc)
                await self._send_safe_end_turn()
                return None
            else:
                others = _others_left(mask, action_idx)
                if not vetoed(self._ledger, lua_command, unblocking_only, others):
                    return lua_command
                why = "not to be sent again this turn"

            if (
                attempt >= self.config.max_reselections
                or not _is_offered(mask, action_idx)
                or not _others_left(mask, action_idx)
            ):
                logger.warning("Action %d not sent (%s); no further choice", action_idx, why)
                break
            logger.info("Action %d not sent (%s); choosing again", action_idx, why)
            self._stats.reselections += 1
            mask = mask.copy()
            mask[action_idx] = 0

        return self._unanswered(ready, end_turn)

    def _unanswered(self, ready: dict[int, Any], end_turn: LuaCommand) -> LuaCommand:
        """What goes out when the policy named nothing sendable."""
        # While the game blocks the end of turn, anything that may still be
        # sent beats an end_turn that will be refused (and answers a choice the
        # search's simulator cannot express, such as research).
        command = first_to_unblock(ready.values()) if self._ledger.end_turn_withheld() else None
        return end_turn if command is None else cast(LuaCommand, command)

    async def _ask_command_policy(
        self, state: GameStateData, converted_state: dict[str, Any], unblocking_only: bool
    ) -> LuaCommand | None:
        """Ask a policy that answers with a command; None when it named nothing."""
        choice = await choose(
            self.policy,
            self._ledger,
            converted_state,
            unblocking_only=unblocking_only,
            turn_id=state.turn_id,
        )
        self._stats.search_decisions += 1
        if choice.failed:
            self._stats.search_failures += 1
            self._stats.last_error = choice.reason
        elif choice.command is None:
            self._stats.search_fallbacks += 1
        logger.debug(
            "Turn %s: command policy: %s (%s)", state.turn_id, choice.record, choice.reason
        )
        if self.eval_harness:
            self.eval_harness.on_turn_end(state.turn_id, diagnostics_of(choice))
        return choice.command

    async def _send_and_account(self, state: GameStateData, lua_command: LuaCommand) -> None:
        """Send one command, read the mod's answer and record what it was."""
        command = CommandMessage.create_single(
            msg_id=self.bridge.next_msg_id(),
            command=lua_command,
        )
        self._ledger.sent(lua_command, state)
        self._stats.commands_sent += 1

        try:
            if self.health_checker is not None:
                ack: Any = await self.health_checker.execute_with_retry(
                    self.bridge.send_and_wait_with_retry,
                    command,
                )
            else:
                ack = await self.bridge.send_and_wait_with_retry(command)
            ack = _as_ack_message(ack)
        except CircuitBreakerOpenError as exc:
            self._ledger.send_failed()
            self._stats.ack_failures += 1
            self._stats.last_error = f"circuit_open: {exc}"
            logger.warning("Circuit breaker open on send (%s); skipping ack", exc)
            self._maybe_log_snapshot(state, command, None, exc)
        except (AckTimeoutError, ConnectionLostError) as exc:
            self._ledger.send_failed()
            self._stats.ack_failures += 1
            self._stats.last_error = f"ack: {exc}"
            logger.warning(
                "Ack failure (#%d): %s; recovering on next turn",
                self._stats.ack_failures,
                exc,
            )
            self._maybe_log_snapshot(state, command, None, exc)
        except Exception as exc:
            self._ledger.send_failed()
            self._stats.ack_failures += 1
            self._stats.last_error = f"send: {exc}"
            logger.error("send_and_wait_with_retry raised: %s", exc, exc_info=True)
            self._maybe_log_snapshot(state, command, None, exc)
        else:
            self._ledger.answered(lua_command, ack)
            if not ack.success:
                blocker = _blocked_end_turn_message(ack)
                if blocker is not None:
                    # The game will not end the turn until a pending choice is
                    # made. The mod sent nothing to the game, exports the state
                    # again (with end_turn_blocking set) and keeps listening,
                    # so the next step sees the blocker; until the state
                    # changes, end_turn is not chosen while anything else is.
                    self._stats.blocked_end_turns += 1
                    self._stats.last_end_turn_blocker = blocker
                    logger.warning(
                        "End turn blocked by the game (#%d, msg_id=%s): %s",
                        self._stats.blocked_end_turns,
                        getattr(ack, "msg_id", None),
                        blocker,
                    )
                else:
                    self._stats.invalid_actions += 1
                    logger.warning(
                        "Lua mod returned ack.success=False (msg_id=%d, code=%s): %s",
                        ack.msg_id,
                        ack.error_code,
                        ack.error_message,
                    )

    def _check_stall(self, state: GameStateData) -> None:
        """Give a game turn up when it can neither be ended nor unblocked."""
        if not self._ledger.stalled() or self._stalled_turn == state.turn_id or self._stall_exit:
            return
        self._stats.stalled_turns += 1
        self._stats.last_error = f"stalled: {self._stats.last_end_turn_blocker}"
        logger.error(
            "Turn %s is stalled: the game keeps blocking the end of turn (%s) and nothing "
            "sent has changed that. on_stall=%s",
            state.turn_id,
            self._stats.last_end_turn_blocker or getattr(state, "end_turn_blocking", None),
            self.config.on_stall,
        )
        if self.config.on_stall == "exit":
            self._stall_exit = True
        else:
            self._stalled_turn = state.turn_id
            self._stall_deadline = time.monotonic() + self.config.stall_wait_s

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
        # Counted against the turn; its answer is not read, so a blocked one
        # shows only in the next state (the ledger reads it there).
        self._ledger.sent({"action_type": "end_turn"}, self._current_state)
        self._stats.commands_sent += 1
        try:
            cmd = create_end_turn_command(self.bridge.next_msg_id())
            await self.bridge.send_command(cmd)
            logger.debug("Sent safe end_turn fallback (count=%d)", self._stats.fallback_end_turns)
        except Exception as exc:
            # It never went out: a blocker in the next state is not its answer.
            self._ledger.send_failed()
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
