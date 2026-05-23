"""Opponent modeling for multiplayer MCTS.

Tracks opponent action history and predicts future behavior using
frequency-based analysis with exponential decay. The predicted
distributions can be used to improve MCTS search by biasing
opponent move selection toward likely actions.

Design:
    Uses a lightweight, non-neural approach (action frequency counting
    with exponential decay) to avoid adding training complexity.
    Can be extended to neural opponent modeling in the future.

Example:
    >>> model = OpponentModel(OpponentModelConfig(enabled=True), num_players=4, action_space_size=356)
    >>> model.update(player_id=1, game_state={}, action=42)
    >>> probs = model.predict_action_distribution(player_id=1, game_state={}, valid_actions=mask)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from alpha_zero.multiplayer.config import OpponentModelConfig

logger = logging.getLogger(__name__)

# Numerical stability threshold for division and comparison to zero.
# Used across prediction, entropy, and blending computations.
_EPSILON = 1e-8


class OpponentModel:
    """Frequency-based opponent action modeling.

    Tracks per-opponent action frequencies with exponential decay,
    providing predicted action distributions for MCTS guidance.

    Attributes:
        config: Model configuration.
        num_players: Number of players in the game.
        action_space_size: Total action space size.
    """

    def __init__(
        self,
        config: OpponentModelConfig,
        num_players: int,
        action_space_size: int,
    ) -> None:
        """Initialize opponent model.

        Args:
            config: Opponent model configuration.
            num_players: Number of players.
            action_space_size: Size of the action space.
        """
        self.config = config
        self.num_players = num_players
        self.action_space_size = action_space_size

        # Per-player action frequency counts (with decay)
        self._action_counts: dict[int, np.ndarray] = {}
        # Per-player total observation count
        self._observation_counts: dict[int, int] = defaultdict(int)
        # Per-player recent action buffer (for windowed analysis)
        self._recent_actions: dict[int, list[int]] = defaultdict(list)

        self._initialize_counts()

        logger.debug(
            "OpponentModel initialized",
            extra={
                "enabled": config.enabled,
                "num_players": num_players,
                "action_space_size": action_space_size,
                "history_window": config.history_window,
                "decay_rate": config.decay_rate,
            },
        )

    def _initialize_counts(self) -> None:
        """Initialize action count arrays for all players."""
        for player_id in range(self.num_players):
            self._action_counts[player_id] = np.zeros(self.action_space_size, dtype=np.float64)

    def update(
        self,
        player_id: int,
        game_state: dict[str, Any],
        action: int,
    ) -> None:
        """Record an opponent's action.

        Applies exponential decay to existing counts before adding
        the new observation. This gives more weight to recent actions.

        Args:
            player_id: ID of the player who acted.
            game_state: State in which the action was taken (for future use).
            action: Action index chosen.
        """
        if not self.config.enabled:
            return

        if player_id < 0 or player_id >= self.num_players:
            logger.warning("Invalid player_id %d (num_players=%d)", player_id, self.num_players)
            return

        if action < 0 or action >= self.action_space_size:
            logger.warning(
                "Invalid action %d (action_space_size=%d)", action, self.action_space_size
            )
            return

        # Apply decay to existing counts
        self._action_counts[player_id] *= self.config.decay_rate

        # Increment count for the observed action
        self._action_counts[player_id][action] += 1.0

        # Track total observations
        self._observation_counts[player_id] += 1

        logger.debug(
            "Opponent model updated",
            extra={
                "player_id": player_id,
                "action": action,
                "total_observations": self._observation_counts[player_id],
            },
        )

        # Maintain recent action window
        recent = self._recent_actions[player_id]
        recent.append(action)
        if len(recent) > self.config.history_window:
            recent.pop(0)

    def predict_action_distribution(
        self,
        player_id: int,
        game_state: dict[str, Any],
        valid_actions: np.ndarray,
    ) -> np.ndarray:
        """Predict action probability distribution for an opponent.

        Combines decayed frequency counts with uniform prior, masked
        to valid actions only.

        Args:
            player_id: ID of the opponent to predict.
            game_state: Current game state (for future context use).
            valid_actions: Mask of valid actions (1=valid, 0=invalid).

        Returns:
            Probability distribution over actions, shape (action_space_size,).
        """
        if not self.config.enabled:
            return self._uniform_distribution(valid_actions)

        if player_id not in self._action_counts:
            return self._uniform_distribution(valid_actions)

        counts = self._action_counts[player_id].copy()

        # Mask to valid actions
        counts = counts * valid_actions

        total = counts.sum()
        if total < _EPSILON:
            return self._uniform_distribution(valid_actions)

        # Blend with uniform prior (smoothing)
        uniform = self._uniform_distribution(valid_actions)
        obs_count = self._observation_counts.get(player_id, 0)

        # Confidence grows with observations (Bayesian-like blending)
        if obs_count == 0:
            return uniform

        confidence = min(1.0, obs_count / self.config.history_window)
        predicted = counts / total
        blended = confidence * predicted + (1.0 - confidence) * uniform

        # Renormalize
        blend_sum = blended.sum()
        if blend_sum > _EPSILON:
            blended = blended / blend_sum

        return blended

    def get_most_likely_action(
        self,
        player_id: int,
        game_state: dict[str, Any],
        valid_actions: np.ndarray,
    ) -> int:
        """Get the most likely action for an opponent.

        Args:
            player_id: Opponent player ID.
            game_state: Current game state.
            valid_actions: Mask of valid actions.

        Returns:
            Most likely action index.
        """
        probs = self.predict_action_distribution(player_id, game_state, valid_actions)
        return int(np.argmax(probs))

    def get_action_entropy(self, player_id: int) -> float:
        """Compute entropy of opponent's action distribution.

        Higher entropy means more unpredictable opponent.

        Args:
            player_id: Opponent player ID.

        Returns:
            Shannon entropy of the action distribution.
        """
        if player_id not in self._action_counts:
            return float(np.log(self.action_space_size))

        counts = self._action_counts[player_id]
        total = counts.sum()
        if total < _EPSILON:
            return float(np.log(self.action_space_size))

        probs = counts / total
        probs = probs[probs > _EPSILON]  # Filter zeros
        return float(-np.sum(probs * np.log(probs)))

    def get_player_summary(self, player_id: int) -> dict[str, Any]:
        """Get summary statistics for an opponent.

        Args:
            player_id: Opponent player ID.

        Returns:
            Dict with model statistics.
        """
        obs_count = self._observation_counts.get(player_id, 0)
        entropy = self.get_action_entropy(player_id)
        recent = self._recent_actions.get(player_id, [])

        # Most common recent actions
        top_actions: list[tuple[int, int]] = []
        if recent:
            counter = Counter(recent)
            top_actions = counter.most_common(5)

        return {
            "player_id": player_id,
            "total_observations": obs_count,
            "recent_action_count": len(recent),
            "action_entropy": entropy,
            "top_recent_actions": top_actions,
            "confidence": min(1.0, obs_count / self.config.history_window)
            if obs_count > 0
            else 0.0,
        }

    def reset(self) -> None:
        """Reset model state for a new game."""
        self._initialize_counts()
        self._observation_counts.clear()
        self._recent_actions.clear()
        logger.debug("OpponentModel reset")

    def _uniform_distribution(self, valid_actions: np.ndarray) -> np.ndarray:
        """Create uniform distribution over valid actions.

        Args:
            valid_actions: Mask of valid actions.

        Returns:
            Uniform probability distribution.
        """
        probs = valid_actions.astype(np.float64)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return probs
