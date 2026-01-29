"""Enhanced reward signals for improved learning.

This module provides stronger intermediate reward signals to accelerate
learning while preserving optimal policy through potential-based shaping.

Features:
- Configurable intermediate rewards for various game events
- Territory control bonuses
- Unit preservation incentives
- Aggression bonuses for engaging combat
- Multiple win condition evaluation

Example usage:
    >>> from alpha_zero.utils.enhanced_rewards import (
    ...     EnhancedRewardsConfig,
    ...     EnhancedRewardShaper,
    ...     create_enhanced_reward_shaper,
    ... )
    >>> config = EnhancedRewardsConfig(scale_factor=2.0)
    >>> shaper = EnhancedRewardShaper(config, board_size=16)
    >>> shaped_reward = shaper.shape_reward(obs, base_reward, info)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from alpha_zero.utils.logging import get_logger

logger = get_logger(__name__)


class WinCondition(Enum):
    """Available win conditions."""

    DOMINATION = "domination"  # Control all cities
    ELIMINATION = "elimination"  # Destroy all enemy units
    SCORE = "score"  # Highest score at turn limit
    TERRITORY = "territory"  # Control most territory


@dataclass
class IntermediateRewardsConfig:
    """Configuration for intermediate rewards.

    Attributes:
        territory_control: Reward per territory tile controlled.
        unit_preservation: Reward for keeping units alive.
        city_proximity: Reward for units near enemy cities.
        exploration: Reward for exploring new tiles.
        production_progress: Reward for production advancement.
        combat_engagement: Reward for engaging in combat.
        city_founding: Reward for founding new cities.
        tech_progress: Reward for research advancement.
    """

    territory_control: float = 0.02
    unit_preservation: float = 0.01
    city_proximity: float = 0.005
    exploration: float = 0.001
    production_progress: float = 0.01  # Increased from 0.002
    combat_engagement: float = 0.01
    city_founding: float = 0.5  # Increased from 0.1
    tech_progress: float = 0.05
    step_penalty: float = -0.001  # Reduced from -0.005


@dataclass
class VictoryBonusConfig:
    """Configuration for victory bonuses.

    Attributes:
        domination: Bonus for domination victory.
        elimination: Bonus for elimination victory.
        score: Bonus for score victory.
        territory: Bonus for territory victory.
    """

    domination: float = 1.0
    elimination: float = 1.0
    score: float = 0.5
    territory: float = 0.3


@dataclass
class EnhancedRewardsConfig:
    """Complete enhanced rewards configuration.

    Attributes:
        enabled: Master switch for enhanced rewards.
        scale_factor: Global multiplier for all rewards.
        gamma: Discount factor for potential-based shaping.
        normalize: Whether to normalize rewards to [-1, 1] range.
        intermediate: Intermediate rewards configuration.
        victory_bonuses: Victory bonus configuration.
        active_win_conditions: Set of active win conditions.
        potential_weights: Weights for potential function components.
    """

    enabled: bool = True
    scale_factor: float = 1.0
    gamma: float = 0.99
    normalize: bool = True
    intermediate: IntermediateRewardsConfig = field(
        default_factory=IntermediateRewardsConfig
    )
    victory_bonuses: VictoryBonusConfig = field(
        default_factory=VictoryBonusConfig
    )
    active_win_conditions: set[WinCondition] = field(
        default_factory=lambda: {WinCondition.DOMINATION, WinCondition.ELIMINATION}
    )
    potential_weights: dict[str, float] = field(default_factory=lambda: {
        "unit_advantage": 0.1,
        "city_advantage": 0.3,  # Increased from 0.2
        "health_advantage": 0.05,
        "territory_advantage": 0.2, # Increased from 0.05
        "production_advantage": 0.03,
    })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnhancedRewardsConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            EnhancedRewardsConfig instance.
        """
        intermediate_data = data.get("intermediate", {})
        victory_data = data.get("victory_bonuses", {})

        win_conditions = set()
        for cond in data.get("active_win_conditions", ["domination", "elimination"]):
            try:
                win_conditions.add(WinCondition(cond))
            except ValueError:
                logger.warning(f"Unknown win condition: {cond}")

        return cls(
            enabled=data.get("enabled", True),
            scale_factor=data.get("scale_factor", 1.0),
            gamma=data.get("gamma", 0.99),
            normalize=data.get("normalize", True),
            intermediate=IntermediateRewardsConfig(
                territory_control=intermediate_data.get("territory_control", 0.02),
                unit_preservation=intermediate_data.get("unit_preservation", 0.01),
                city_proximity=intermediate_data.get("city_proximity", 0.005),
                exploration=intermediate_data.get("exploration", 0.001),
                production_progress=intermediate_data.get("production_progress", 0.002),
                combat_engagement=intermediate_data.get("combat_engagement", 0.01),
                city_founding=intermediate_data.get("city_founding", 0.1),
                tech_progress=intermediate_data.get("tech_progress", 0.05),
            ),
            victory_bonuses=VictoryBonusConfig(
                domination=victory_data.get("domination", 1.0),
                elimination=victory_data.get("elimination", 1.0),
                score=victory_data.get("score", 0.5),
                territory=victory_data.get("territory", 0.3),
            ),
            active_win_conditions=win_conditions,
            potential_weights=data.get("potential_weights", cls().potential_weights),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary.
        """
        return {
            "enabled": self.enabled,
            "scale_factor": self.scale_factor,
            "gamma": self.gamma,
            "normalize": self.normalize,
            "intermediate": {
                "territory_control": self.intermediate.territory_control,
                "unit_preservation": self.intermediate.unit_preservation,
                "city_proximity": self.intermediate.city_proximity,
                "exploration": self.intermediate.exploration,
                "production_progress": self.intermediate.production_progress,
                "combat_engagement": self.intermediate.combat_engagement,
                "city_founding": self.intermediate.city_founding,
                "tech_progress": self.intermediate.tech_progress,
            },
            "victory_bonuses": {
                "domination": self.victory_bonuses.domination,
                "elimination": self.victory_bonuses.elimination,
                "score": self.victory_bonuses.score,
                "territory": self.victory_bonuses.territory,
            },
            "active_win_conditions": [c.value for c in self.active_win_conditions],
            "potential_weights": self.potential_weights,
        }


@dataclass
class EnhancedStateFeatures:
    """Extended features for enhanced reward shaping.

    Attributes:
        own_units: Number of own units.
        enemy_units: Number of enemy units.
        own_cities: Number of own cities.
        enemy_cities: Number of enemy cities.
        own_avg_health: Average health of own units.
        enemy_avg_health: Average health of enemy units.
        own_territory: Tiles controlled by own units.
        enemy_territory: Tiles controlled by enemy units.
        own_production: Total production capacity.
        enemy_production: Enemy production capacity.
        units_near_enemy_cities: Own units adjacent to enemy cities.
        explored_tiles: Number of tiles explored.
        total_tiles: Total number of tiles.
        turn: Current turn number.
        max_turns: Maximum turns in game.
        combat_this_turn: Whether combat occurred this turn.
        city_founded_this_turn: Whether a city was founded.
    """

    own_units: int = 0
    enemy_units: int = 0
    own_cities: int = 0
    enemy_cities: int = 0
    own_avg_health: float = 100.0
    enemy_avg_health: float = 100.0
    own_territory: int = 0
    enemy_territory: int = 0
    own_production: int = 0
    enemy_production: int = 0
    units_near_enemy_cities: int = 0
    visited_tiles: int = 0  # Changed from explored_tiles
    total_tiles: int = 64
    turn: int = 0
    max_turns: int = 100
    combat_this_turn: bool = False
    city_founded_this_turn: bool = False


class WinConditionEvaluator:
    """Evaluates multiple win conditions and assigns appropriate rewards."""

    def __init__(self, config: EnhancedRewardsConfig):
        """Initialize win condition evaluator.

        Args:
            config: Enhanced rewards configuration.
        """
        self.config = config
        self._logger = get_logger(f"{__name__}.WinConditionEvaluator")

    def evaluate(
        self,
        features: EnhancedStateFeatures,
        game_ended: bool,
        winner: int | None,
        current_player: int,
    ) -> tuple[float, str]:
        """Evaluate win conditions and return bonus.

        Args:
            features: Current state features.
            game_ended: Whether the game has ended.
            winner: Winner player index (None for draw/ongoing).
            current_player: Current player index.

        Returns:
            Tuple of (victory_bonus, victory_type).
        """
        if not game_ended:
            return 0.0, ""

        if winner is None:
            return 0.0, "draw"

        is_winner = winner == current_player
        multiplier = 1.0 if is_winner else -1.0

        # Determine victory type
        victory_type = self._determine_victory_type(features, winner)

        # Get bonus based on victory type
        bonus = self._get_victory_bonus(victory_type)

        self._logger.debug(
            f"Victory evaluation: type={victory_type}, "
            f"bonus={bonus * multiplier:.3f}"
        )

        return bonus * multiplier, victory_type

    def _determine_victory_type(
        self,
        features: EnhancedStateFeatures,
        winner: int,
    ) -> str:
        """Determine the type of victory achieved.

        Args:
            features: State features.
            winner: Winner player index.

        Returns:
            Victory type string.
        """
        # Check domination (controls all cities)
        if winner == 0:
            if features.enemy_cities == 0 and features.own_cities > 0:
                if features.enemy_units == 0:
                    return "elimination"
                return "domination"
        else:
            if features.own_cities == 0 and features.enemy_cities > 0:
                if features.own_units == 0:
                    return "elimination"
                return "domination"

        # Default to score victory if at turn limit
        if features.turn >= features.max_turns:
            return "score"

        # Territory victory if enabled
        if WinCondition.TERRITORY in self.config.active_win_conditions:
            if features.own_territory > features.enemy_territory * 1.5:
                return "territory"

        return "default"

    def _get_victory_bonus(self, victory_type: str) -> float:
        """Get bonus for victory type.

        Args:
            victory_type: Type of victory.

        Returns:
            Victory bonus value.
        """
        bonuses = {
            "domination": self.config.victory_bonuses.domination,
            "elimination": self.config.victory_bonuses.elimination,
            "score": self.config.victory_bonuses.score,
            "territory": self.config.victory_bonuses.territory,
            "default": 1.0,
        }
        return bonuses.get(victory_type, 1.0)


class EnhancedRewardShaper:
    """Extended reward shaper with enhanced intermediate signals."""

    def __init__(
        self,
        config: EnhancedRewardsConfig | None = None,
        board_size: int = 16,
    ):
        """Initialize enhanced reward shaper.

        Args:
            config: Enhanced rewards configuration.
            board_size: Size of the game board.
        """
        self.config = config or EnhancedRewardsConfig()
        self.board_size = board_size

        self._win_evaluator = WinConditionEvaluator(self.config)
        self._prev_features: EnhancedStateFeatures | None = None
        self._episode_rewards: list[float] = []
        self._episode_base_rewards: list[float] = []
        self._logger = get_logger(f"{__name__}.EnhancedRewardShaper")

    def reset(self) -> None:
        """Reset shaper for a new episode."""
        self._prev_features = None
        self._episode_rewards = []
        self._episode_base_rewards = []

    def extract_features(
        self,
        obs: dict[str, np.ndarray],
        info: dict[str, Any] | None = None,
    ) -> EnhancedStateFeatures:
        """Extract enhanced features from observation.

        Args:
            obs: Observation dictionary.
            info: Additional info dictionary.

        Returns:
            EnhancedStateFeatures instance.
        """
        info = info or {}
        spatial = obs["spatial"]
        vector = obs["vector"]

        # Extract from vector encoding
        turn = int(vector[0] * 200) if len(vector) > 0 else 0
        own_units = int(vector[2] * 20) if len(vector) > 2 else 0
        enemy_units = int(vector[3] * 20) if len(vector) > 3 else 0
        own_cities = int(vector[4] * 5) if len(vector) > 4 else 0
        enemy_cities = int(vector[5] * 5) if len(vector) > 5 else 0
        own_avg_health = vector[6] * 100 if len(vector) > 6 else 100.0
        enemy_avg_health = vector[7] * 100 if len(vector) > 7 else 100.0

        # Extract territory from spatial
        own_unit_mask = np.sum(spatial[3:6], axis=0) > 0 if len(spatial) > 5 else np.zeros((self.board_size, self.board_size), dtype=bool)
        enemy_unit_mask = np.sum(spatial[6:9], axis=0) > 0 if len(spatial) > 8 else np.zeros((self.board_size, self.board_size), dtype=bool)

        own_territory = int(np.sum(own_unit_mask))
        enemy_territory = int(np.sum(enemy_unit_mask))

        # Extract from info
        num_units = info.get("num_units", [own_units, enemy_units])
        num_cities = info.get("num_cities", [own_cities, enemy_cities])

        return EnhancedStateFeatures(
            own_units=num_units[0] if isinstance(num_units, list) else own_units,
            enemy_units=num_units[1] if isinstance(num_units, list) else enemy_units,
            own_cities=num_cities[0] if isinstance(num_cities, list) else own_cities,
            enemy_cities=num_cities[1] if isinstance(num_cities, list) else enemy_cities,
            own_avg_health=own_avg_health,
            enemy_avg_health=enemy_avg_health,
            own_territory=own_territory,
            enemy_territory=enemy_territory,
            total_tiles=self.board_size * self.board_size,
            turn=info.get("turn", turn),
            max_turns=info.get("max_turns", 200),
            combat_this_turn=info.get("combat_this_turn", False),
            city_founded_this_turn=info.get("city_founded_this_turn", False),
            visited_tiles=info.get("visited_count", own_territory),  # Use visited count
        )

    def compute_potential(self, features: EnhancedStateFeatures) -> float:
        """Compute potential function value for state.

        Uses weighted combination of advantages across multiple dimensions.

        Args:
            features: State features.

        Returns:
            Potential value.
        """
        weights = self.config.potential_weights

        # Unit advantage
        max_units = 20
        unit_diff = (features.own_units - features.enemy_units) / max_units
        unit_potential = unit_diff * weights.get("unit_advantage", 0.1)

        # City advantage
        max_cities = 5
        city_diff = (features.own_cities - features.enemy_cities) / max_cities
        city_potential = city_diff * weights.get("city_advantage", 0.2)

        # Health advantage
        health_diff = (features.own_avg_health - features.enemy_avg_health) / 100.0
        health_potential = health_diff * weights.get("health_advantage", 0.05)

        # Territory advantage
        total = features.total_tiles
        if total > 0:
            territory_diff = (features.own_territory - features.enemy_territory) / total
        else:
            territory_diff = 0.0
        territory_potential = territory_diff * weights.get("territory_advantage", 0.05)

        return (
            unit_potential +
            city_potential +
            health_potential +
            territory_potential
        )

    def compute_intermediate_rewards(
        self,
        prev_features: EnhancedStateFeatures | None,
        curr_features: EnhancedStateFeatures,
    ) -> dict[str, float]:
        """Compute intermediate reward components.

        Args:
            prev_features: Previous state features.
            curr_features: Current state features.

        Returns:
            Dictionary of intermediate reward components.
        """
        rewards: dict[str, float] = {}
        cfg = self.config.intermediate

        if prev_features is None:
            return rewards

        # Territory control reward
        territory_gain = curr_features.own_territory - prev_features.own_territory
        if territory_gain > 0:
            rewards["territory_control"] = territory_gain * cfg.territory_control

        # Unit preservation reward (bonus for not losing units)
        if curr_features.own_units >= prev_features.own_units:
            rewards["unit_preservation"] = cfg.unit_preservation

        # Combat engagement reward
        if curr_features.combat_this_turn:
            rewards["combat_engagement"] = cfg.combat_engagement

        # City founding reward
        if curr_features.city_founded_this_turn:
            rewards["city_founding"] = cfg.city_founding

        # City proximity reward
        if curr_features.units_near_enemy_cities > prev_features.units_near_enemy_cities:
            rewards["city_proximity"] = cfg.city_proximity

        # Exploration reward (based on visited tiles)
        if curr_features.visited_tiles > (prev_features.visited_tiles if prev_features else 0):
            rewards["exploration"] = (
                curr_features.visited_tiles - (prev_features.visited_tiles if prev_features else 0)
            ) * cfg.exploration

        # Step penalty
        rewards["step_penalty"] = cfg.step_penalty

        return rewards

    def shape_reward(
        self,
        obs: dict[str, np.ndarray],
        base_reward: float,
        info: dict[str, Any] | None = None,
        terminated: bool = False,
    ) -> float:
        """Compute shaped reward.

        Uses potential-based reward shaping: F(s,a,s') = gamma * Phi(s') - Phi(s)
        Plus intermediate rewards for specific events.

        Args:
            obs: Current observation.
            base_reward: Base reward from environment.
            info: Additional info dictionary.
            terminated: Whether episode has terminated.

        Returns:
            Shaped reward.
        """
        if not self.config.enabled:
            return base_reward

        info = info or {}
        curr_features = self.extract_features(obs, info)

        # Base reward
        total_reward = base_reward

        # Potential-based shaping
        if self._prev_features is not None:
            prev_potential = self.compute_potential(self._prev_features)

            if terminated:
                # Terminal states have zero potential
                shaping_bonus = -prev_potential
            else:
                curr_potential = self.compute_potential(curr_features)
                shaping_bonus = self.config.gamma * curr_potential - prev_potential

            total_reward += shaping_bonus * self.config.scale_factor

        # Intermediate rewards
        intermediate = self.compute_intermediate_rewards(self._prev_features, curr_features)
        for component, value in intermediate.items():
            total_reward += value * self.config.scale_factor

        # Victory bonus
        if terminated:
            winner = info.get("winner")
            current_player = info.get("current_player", 0)

            victory_bonus, victory_type = self._win_evaluator.evaluate(
                curr_features,
                game_ended=True,
                winner=winner,
                current_player=current_player,
            )
            total_reward += victory_bonus * self.config.scale_factor

        # Normalize if configured
        if self.config.normalize:
            total_reward = np.clip(total_reward, -2.0, 2.0)

        # Track rewards
        self._episode_base_rewards.append(base_reward)
        self._episode_rewards.append(total_reward)

        # Update state
        self._prev_features = curr_features

        return float(total_reward)

    def get_episode_stats(self) -> dict[str, float]:
        """Get statistics for the current episode.

        Returns:
            Dictionary with episode statistics.
        """
        if not self._episode_rewards:
            return {
                "base_reward_sum": 0.0,
                "shaped_reward_sum": 0.0,
                "shaping_contribution": 0.0,
                "num_steps": 0,
                "reward_density": 0.0,
            }

        base_sum = sum(self._episode_base_rewards)
        shaped_sum = sum(self._episode_rewards)
        num_steps = len(self._episode_rewards)

        # Count non-zero rewards as density metric
        non_zero = sum(1 for r in self._episode_rewards if abs(r) > 1e-6)

        return {
            "base_reward_sum": base_sum,
            "shaped_reward_sum": shaped_sum,
            "shaping_contribution": shaped_sum - base_sum,
            "num_steps": num_steps,
            "reward_density": non_zero / num_steps if num_steps > 0 else 0.0,
        }


def create_enhanced_reward_shaper(
    scale_factor: float = 1.0,
    board_size: int = 16,
    **kwargs: Any,
) -> EnhancedRewardShaper:
    """Factory function to create enhanced reward shaper.

    Args:
        scale_factor: Global reward scale factor.
        board_size: Size of the game board.
        **kwargs: Additional configuration parameters.

    Returns:
        EnhancedRewardShaper instance.
    """
    config = EnhancedRewardsConfig(scale_factor=scale_factor, **kwargs)
    return EnhancedRewardShaper(config=config, board_size=board_size)


def get_aggressive_rewards_config() -> EnhancedRewardsConfig:
    """Get configuration that encourages aggressive play.

    Returns:
        EnhancedRewardsConfig optimized for aggressive strategies.
    """
    return EnhancedRewardsConfig(
        enabled=True,
        scale_factor=1.5,
        intermediate=IntermediateRewardsConfig(
            territory_control=0.03,
            unit_preservation=0.005,  # Lower preservation reward
            city_proximity=0.01,  # Higher proximity reward
            combat_engagement=0.02,  # Higher combat reward
            city_founding=0.05,  # Lower founding reward
        ),
        potential_weights={
            "unit_advantage": 0.15,
            "city_advantage": 0.25,
            "health_advantage": 0.03,
            "territory_advantage": 0.1,
        },
    )


def get_defensive_rewards_config() -> EnhancedRewardsConfig:
    """Get configuration that encourages defensive play.

    Returns:
        EnhancedRewardsConfig optimized for defensive strategies.
    """
    return EnhancedRewardsConfig(
        enabled=True,
        scale_factor=1.0,
        intermediate=IntermediateRewardsConfig(
            territory_control=0.01,
            unit_preservation=0.02,  # Higher preservation
            city_proximity=0.002,  # Lower proximity
            combat_engagement=0.005,  # Lower combat reward
            city_founding=0.6,  # Higher founding reward (default is 0.5)
        ),
        potential_weights={
            "unit_advantage": 0.05,
            "city_advantage": 0.3,
            "health_advantage": 0.1,
            "territory_advantage": 0.02,
        },
    )


def get_balanced_rewards_config() -> EnhancedRewardsConfig:
    """Get balanced configuration.

    Returns:
        EnhancedRewardsConfig with balanced settings.
    """
    return EnhancedRewardsConfig()
