"""Runtime settings of :class:`~alpha_zero.bridge.agent_loop.BridgeAgentLoop`.

Split from ``agent_loop.py`` (GM-B7c-2) so that file stays under the size
ratchet; ``agent_loop`` re-exports everything here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from alpha_zero.bridge.utils import parse_bool_env

__all__ = ["ENV_PREFIX", "AgentLoopConfig", "DisconnectStrategy", "StallStrategy"]

# Env-var prefix for AgentLoopConfig overrides. Matches the Civ6LocalConfig
# CIV6AZ_LOCAL_ pattern - see alpha_zero/civ6_local.py:48.
ENV_PREFIX = "CIV6AZ_AGENT_"

_DISCONNECT_LITERAL = ("exit", "reconnect")
DisconnectStrategy = Literal["exit", "reconnect"]

_STALL_LITERAL = ("exit", "wait")
StallStrategy = Literal["exit", "wait"]


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
        max_game_turns: Stop after this many finished GAME turns (``0`` = no
            limit). ``max_turns`` above counts loop steps, one per command.
        max_commands_per_turn: Past this many commands in one game turn only
            ``end_turn`` is sent. Kept under the mod's own cap.
        max_reselections: How often per step the policy is asked again after
            a choice that cannot be sent (invalid, refused earlier this turn,
            or an ``end_turn`` the game is blocking).
        on_stall: ``"exit"`` returns exit code 3 when a turn can neither be
            ended nor unblocked; ``"wait"`` stops sending for that turn and
            waits for the next one (someone ends the turn by hand).
        stall_wait_s: How long ``"wait"`` waits before it exits too.
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
    max_game_turns: int = 0
    max_commands_per_turn: int = 100
    max_reselections: int = 8
    on_stall: StallStrategy = "exit"
    stall_wait_s: float = 300.0

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
        if self.max_game_turns < 0:
            raise ValueError(f"max_game_turns must be >= 0, got {self.max_game_turns}")
        if self.max_commands_per_turn < 1:
            raise ValueError(
                f"max_commands_per_turn must be >= 1, got {self.max_commands_per_turn}"
            )
        if self.max_reselections < 0:
            raise ValueError(f"max_reselections must be >= 0, got {self.max_reselections}")
        if self.on_stall not in _STALL_LITERAL:
            raise ValueError(f"on_stall must be one of {_STALL_LITERAL}, got {self.on_stall!r}")
        if self.stall_wait_s <= 0:
            raise ValueError(f"stall_wait_s must be > 0, got {self.stall_wait_s}")
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
            CIV6AZ_AGENT_MAX_GAME_TURNS
            CIV6AZ_AGENT_MAX_COMMANDS_PER_TURN
            CIV6AZ_AGENT_MAX_RESELECTIONS
            CIV6AZ_AGENT_ON_STALL  ("exit"/"wait")
            CIV6AZ_AGENT_STALL_WAIT_S
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
        for key, name in (
            ("MAX_GAME_TURNS", "max_game_turns"),
            ("MAX_COMMANDS_PER_TURN", "max_commands_per_turn"),
            ("MAX_RESELECTIONS", "max_reselections"),
        ):
            if (raw := source.get(ENV_PREFIX + key)) is not None:
                kwargs[name] = int(raw)
        if (raw := source.get(ENV_PREFIX + "ON_STALL")) is not None:
            kwargs["on_stall"] = raw.lower()
        if (raw := source.get(ENV_PREFIX + "STALL_WAIT_S")) is not None:
            kwargs["stall_wait_s"] = float(raw)
        return cls(**kwargs)
