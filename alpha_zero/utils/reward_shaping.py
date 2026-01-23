"""Reward shaping utilities for improved learning.

This module provides reward shaping functions to provide intermediate
learning signals during training. Reward shaping helps the agent learn
faster by providing more frequent feedback than sparse terminal rewards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StateFeatures:
    """Features extracted from game state for reward shaping."""

    own_units: int = 0
    enemy_units: int = 0
    own_cities: int = 0
    enemy_cities: int = 0
    own_avg_health: float = 100.0
    enemy_avg_health: float = 100.0
    own_territory: int = 0
    enemy_territory: int = 0
    total_tiles: int = 64  # 8x8 board
    turn: int = 0
    max_turns: int = 100


@dataclass
class RewardShapingConfig:
    """Configuration for reward shaping."""

    enabled: bool = True
    # Shaping coefficients
    unit_advantage_weight: float = 0.05
    city_advantage_weight: float = 0.1
    health_advantage_weight: float = 0.02
    territory_weight: float = 0.01
    # Potential-based shaping coefficient
    gamma: float = 0.99  # Discount factor for potential shaping
    # Normalization
    max_units: int = 10
    max_cities: int = 5


def extract_features_from_obs(obs: dict[str, np.ndarray], board_size: int = 8) -> StateFeatures:
    """Extract state features from observation dictionary.

    Args:
        obs: Observation dictionary with 'spatial' and 'vector' keys.
        board_size: Size of the game board.

    Returns:
        StateFeatures extracted from the observation.
    """
    spatial = obs["spatial"]  # Shape: (13, board_size, board_size)
    vector = obs["vector"]  # Shape: (64,)

    # Extract from vector encoding (based on MicroCivEncoder)
    # Index 0: Turn number (normalized by 200)
    # Index 2: Own unit count / 20.0
    # Index 3: Enemy unit count / 20.0
    # Index 4: Own cities / 5.0
    # Index 5: Enemy cities / 5.0
    # Index 6-7: Average unit health

    turn = int(vector[0] * 200)
    own_units = int(vector[2] * 20)
    enemy_units = int(vector[3] * 20)
    own_cities = int(vector[4] * 5)
    enemy_cities = int(vector[5] * 5)

    # Health from vector if available
    own_avg_health = vector[6] * 100 if len(vector) > 6 else 100.0
    enemy_avg_health = vector[7] * 100 if len(vector) > 7 else 100.0

    # Calculate territory from spatial (approximate based on unit positions)
    # Channels 3-5: Own units, Channels 6-8: Enemy units
    own_unit_mask = np.sum(spatial[3:6], axis=0) > 0
    enemy_unit_mask = np.sum(spatial[6:9], axis=0) > 0

    # Territory approximation: tiles near own units vs enemy units
    own_territory = int(np.sum(own_unit_mask))
    enemy_territory = int(np.sum(enemy_unit_mask))
    total_tiles = board_size * board_size

    return StateFeatures(
        own_units=own_units,
        enemy_units=enemy_units,
        own_cities=own_cities,
        enemy_cities=enemy_cities,
        own_avg_health=own_avg_health,
        enemy_avg_health=enemy_avg_health,
        own_territory=own_territory,
        enemy_territory=enemy_territory,
        total_tiles=total_tiles,
        turn=turn,
    )


def compute_state_potential(
    features: StateFeatures,
    config: RewardShapingConfig,
) -> float:
    """Compute potential function value for a state.

    This is used for potential-based reward shaping, which preserves
    optimal policies while providing better learning signals.

    Args:
        features: State features extracted from observation.
        config: Reward shaping configuration.

    Returns:
        Potential value for the state.
    """
    # Unit advantage potential
    unit_diff = features.own_units - features.enemy_units
    unit_potential = unit_diff / config.max_units * config.unit_advantage_weight

    # City advantage potential
    city_diff = features.own_cities - features.enemy_cities
    city_potential = city_diff / config.max_cities * config.city_advantage_weight

    # Health advantage potential
    health_diff = features.own_avg_health - features.enemy_avg_health
    health_potential = health_diff / 100.0 * config.health_advantage_weight

    # Territory potential
    if features.total_tiles > 0:
        territory_ratio = (features.own_territory - features.enemy_territory) / features.total_tiles
        territory_potential = territory_ratio * config.territory_weight
    else:
        territory_potential = 0.0

    return unit_potential + city_potential + health_potential + territory_potential


def compute_shaped_reward(
    base_reward: float,
    prev_features: StateFeatures | None,
    curr_features: StateFeatures,
    config: RewardShapingConfig,
    terminated: bool = False,
) -> float:
    """Compute shaped reward using potential-based shaping.

    Uses the potential-based reward shaping formula:
    F(s, a, s') = γ * Φ(s') - Φ(s)

    This guarantees that the optimal policy is preserved while
    providing more frequent learning signals.

    Args:
        base_reward: Original reward from environment.
        prev_features: Features from previous state (None for first step).
        curr_features: Features from current state.
        config: Reward shaping configuration.
        terminated: Whether the episode has terminated.

    Returns:
        Shaped reward value.
    """
    if not config.enabled:
        return base_reward

    if prev_features is None:
        # First step, no shaping
        return base_reward

    # Compute potentials
    prev_potential = compute_state_potential(prev_features, config)
    curr_potential = compute_state_potential(curr_features, config)

    # Potential-based shaping: γ * Φ(s') - Φ(s)
    # When terminated, curr_potential is 0 (terminal states have 0 potential)
    if terminated:
        shaping_bonus = -prev_potential
    else:
        shaping_bonus = config.gamma * curr_potential - prev_potential

    return base_reward + shaping_bonus


class RewardShaper:
    """Stateful reward shaper that tracks previous state for shaping calculations."""

    def __init__(self, config: RewardShapingConfig | None = None, board_size: int = 8):
        """Initialize the reward shaper.

        Args:
            config: Reward shaping configuration.
            board_size: Size of the game board.
        """
        self.config = config or RewardShapingConfig()
        self.board_size = board_size
        self._prev_features: StateFeatures | None = None
        self._episode_rewards: list[float] = []
        self._episode_shaped_rewards: list[float] = []

    def reset(self) -> None:
        """Reset the shaper for a new episode."""
        self._prev_features = None
        self._episode_rewards = []
        self._episode_shaped_rewards = []

    def shape_reward(
        self,
        obs: dict[str, np.ndarray],
        reward: float,
        terminated: bool = False,
    ) -> float:
        """Shape a reward based on current observation.

        Args:
            obs: Current observation dictionary.
            reward: Base reward from environment.
            terminated: Whether episode has terminated.

        Returns:
            Shaped reward.
        """
        curr_features = extract_features_from_obs(obs, self.board_size)

        shaped_reward = compute_shaped_reward(
            base_reward=reward,
            prev_features=self._prev_features,
            curr_features=curr_features,
            config=self.config,
            terminated=terminated,
        )

        # Track rewards
        self._episode_rewards.append(reward)
        self._episode_shaped_rewards.append(shaped_reward)

        # Update state
        self._prev_features = curr_features

        return shaped_reward

    @property
    def episode_reward_sum(self) -> float:
        """Sum of original rewards in episode."""
        return sum(self._episode_rewards)

    @property
    def episode_shaped_reward_sum(self) -> float:
        """Sum of shaped rewards in episode."""
        return sum(self._episode_shaped_rewards)

    def get_episode_stats(self) -> dict[str, float]:
        """Get statistics for the episode.

        Returns:
            Dictionary with episode statistics.
        """
        if not self._episode_rewards:
            return {
                "original_reward_sum": 0.0,
                "shaped_reward_sum": 0.0,
                "shaping_contribution": 0.0,
                "num_steps": 0,
            }

        original_sum = sum(self._episode_rewards)
        shaped_sum = sum(self._episode_shaped_rewards)

        return {
            "original_reward_sum": original_sum,
            "shaped_reward_sum": shaped_sum,
            "shaping_contribution": shaped_sum - original_sum,
            "num_steps": len(self._episode_rewards),
        }


def create_reward_shaper(
    enabled: bool = True,
    board_size: int = 8,
    **kwargs,
) -> RewardShaper:
    """Factory function to create a reward shaper.

    Args:
        enabled: Whether reward shaping is enabled.
        board_size: Size of the game board.
        **kwargs: Additional arguments for RewardShapingConfig.

    Returns:
        Configured RewardShaper instance.
    """
    config = RewardShapingConfig(enabled=enabled, **kwargs)
    return RewardShaper(config=config, board_size=board_size)
