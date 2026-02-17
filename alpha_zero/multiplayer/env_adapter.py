"""Environment adapter for N-player games.

Wraps existing 2-player environments (like MicroCivEnv) to provide
the MultiplayerEnvironment protocol. Handles player rotation, state
encoding adjustments, and info dict enrichment.

For environments that natively support N players, this adapter is
a thin passthrough. For 2-player environments used in multiplayer
contexts, it provides the necessary protocol bridge.

Example:
    >>> from alpha_zero.envs.microciv import MicroCivEnv
    >>> env = MicroCivEnv(board_size=16)
    >>> adapter = NPlayerEnvAdapter(env, num_players=2)
    >>> obs, info = adapter.reset()
    >>> isinstance(adapter, MultiplayerEnvironment)
    True
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class NPlayerEnvAdapter:
    """Adapter that wraps any environment to satisfy MultiplayerEnvironment.

    Provides:
    - num_players property
    - Enriched info dicts with rankings and scores
    - Proper player rotation tracking
    - State observation augmentation with player context

    Attributes:
        env: The wrapped environment.
        num_players: Number of players.
        action_space: Action space from wrapped environment.
        current_player: Currently active player ID.
    """

    def __init__(
        self,
        env: Any,
        num_players: int = 2,
        track_scores: bool = True,
    ) -> None:
        """Initialize the adapter.

        Args:
            env: Base environment to wrap. Must have action_space, current_player,
                step(), reset(), get_observation(), get_info(), get_valid_actions().
            num_players: Number of players.
            track_scores: Whether to track per-player cumulative scores.
        """
        self.env = env
        self._num_players = num_players
        self._track_scores = track_scores
        self._player_scores: dict[int, float] = dict.fromkeys(range(num_players), 0.0)
        self._eliminated: set[int] = set()
        self._turn_count = 0
        self._move_history: list[dict[str, Any]] = []

        logger.debug(
            "NPlayerEnvAdapter created",
            extra={"num_players": num_players, "track_scores": track_scores},
        )

    @property
    def num_players(self) -> int:
        """Number of players in the game."""
        return self._num_players

    @property
    def action_space(self) -> Any:
        """Action space from the wrapped environment."""
        return self.env.action_space

    @property
    def current_player(self) -> int:
        """Currently active player ID."""
        return int(self.env.current_player)

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Optional random seed.

        Returns:
            Tuple of (observation, info).
        """
        self._player_scores = dict.fromkeys(range(self._num_players), 0.0)
        self._eliminated = set()
        self._turn_count = 0
        self._move_history = []

        result = self.env.reset(seed=seed) if seed is not None else self.env.reset()

        # Handle different reset return signatures
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        info = self._enrich_info(info)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action to execute.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        current = self.current_player
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track scores
        if self._track_scores:
            self._player_scores[current] += float(reward)

        # Track move
        self._move_history.append({
            "player": current,
            "action": action,
            "reward": float(reward),
        })

        # Track turns
        if self.current_player == 0:
            self._turn_count += 1

        # Track eliminations from info
        if "eliminated_player" in info:
            self._eliminated.add(info["eliminated_player"])

        # Enrich info dict
        info = self._enrich_info(info, terminated=terminated or truncated)

        return obs, reward, terminated, truncated, info

    def clone(self) -> NPlayerEnvAdapter:
        """Create a deep copy for MCTS simulation.

        Returns:
            Deep copy of this adapter and its wrapped environment.
        """
        cloned_env = self.env.clone() if hasattr(self.env, "clone") else copy.deepcopy(self.env)
        cloned = NPlayerEnvAdapter(
            env=cloned_env,
            num_players=self._num_players,
            track_scores=self._track_scores,
        )
        cloned._player_scores = self._player_scores.copy()
        cloned._eliminated = self._eliminated.copy()
        cloned._turn_count = self._turn_count
        # Don't deep copy move history for performance in MCTS
        cloned._move_history = []
        return cloned

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation."""
        return dict(self.env.get_observation())

    def get_info(self) -> dict[str, Any]:
        """Get current environment info, enriched with multiplayer data."""
        base_info = self.env.get_info() if hasattr(self.env, "get_info") else {}
        return self._enrich_info(base_info)

    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions."""
        return self.env.get_valid_actions()

    def _enrich_info(
        self,
        info: dict[str, Any],
        terminated: bool = False,
    ) -> dict[str, Any]:
        """Add multiplayer metadata to info dict.

        Args:
            info: Base info dict from wrapped environment.
            terminated: Whether the game has ended.

        Returns:
            Enriched info dict.
        """
        info["num_players"] = self._num_players
        info["turn_count"] = self._turn_count
        info["player_scores"] = self._player_scores.copy()
        info["eliminated"] = self._eliminated.copy()

        # Compute rankings if game is over
        if terminated and "rankings" not in info:
            winner = info.get("winner")
            if winner is not None or self._player_scores:
                info["rankings"] = self._compute_rankings(winner)

        return info

    def _compute_rankings(self, winner: int | None) -> list[int]:
        """Compute final rankings as ordered list of player IDs.

        Args:
            winner: Winning player ID, or None for draw.

        Returns:
            List of player IDs ordered by rank (best first).
        """
        players = list(range(self._num_players))

        if winner is not None:
            # Winner first, then others sorted by score
            others = [p for p in players if p != winner]
            others.sort(key=lambda p: self._player_scores.get(p, 0), reverse=True)
            return [winner, *others]

        # No winner: sort by score
        players.sort(key=lambda p: self._player_scores.get(p, 0), reverse=True)
        return players

    @property
    def move_history(self) -> list[dict[str, Any]]:
        """Get history of all moves made."""
        return self._move_history

    def __repr__(self) -> str:
        return (
            f"NPlayerEnvAdapter(env={type(self.env).__name__}, "
            f"num_players={self._num_players}, "
            f"current_player={self.current_player})"
        )
