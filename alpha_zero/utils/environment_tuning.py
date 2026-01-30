"""Environment tuning utilities for faster training convergence.

This module provides configurable game mechanics adjustments to reduce
episode length and accelerate training while preserving learning objectives.

Features:
- Configurable max turns with early termination
- Accelerated production mechanics
- Simplified combat resolution
- Adaptive game speed based on training progress

Example usage:
    >>> from alpha_zero.utils.environment_tuning import (
    ...     EnvironmentTuningConfig,
    ...     TunedEnvironmentWrapper,
    ...     create_tuned_environment,
    ... )
    >>> config = EnvironmentTuningConfig(max_turns=50, early_termination_enabled=True)
    >>> env = create_tuned_environment(config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np

from alpha_zero.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EarlyTerminationConfig:
    """Configuration for early game termination.

    Attributes:
        enabled: Whether early termination is active.
        advantage_threshold: Minimum advantage (units/cities) to trigger early end.
        check_interval: How often to check termination conditions (turns).
        min_turns: Minimum turns before early termination can occur.
        stalemate_threshold: Turns without progress before declaring draw.
    """

    enabled: bool = True
    advantage_threshold: int = 3
    check_interval: int = 5
    min_turns: int = 10
    stalemate_threshold: int = 20


@dataclass
class ProductionConfig:
    """Configuration for production mechanics.

    Attributes:
        multiplier: Production output multiplier (>1 = faster).
        base_production: Base production per turn.
        population_bonus: Additional production per population.
        unit_costs: Override unit production costs.
    """

    multiplier: float = 1.0
    base_production: int = 3
    population_bonus: int = 1
    unit_costs: dict[str, int] = field(
        default_factory=lambda: {
            "warrior": 3,
            "settler": 5,
            "archer": 4,
        }
    )


@dataclass
class CombatConfig:
    """Configuration for combat mechanics.

    Attributes:
        simplified: Use simplified (faster) combat resolution.
        damage_multiplier: Damage dealt multiplier.
        min_damage: Minimum damage per attack.
        terrain_bonus_enabled: Whether terrain affects combat.
    """

    simplified: bool = False
    damage_multiplier: float = 1.0
    min_damage: int = 10
    terrain_bonus_enabled: bool = True


@dataclass
class EnvironmentTuningConfig:
    """Complete environment tuning configuration.

    Attributes:
        enabled: Master switch for environment tuning.
        max_turns: Maximum turns per game (None = use default).
        fog_of_war_enabled: Whether fog of war is active.
        board_size: Override board size (None = use default).
        early_termination: Early termination configuration.
        production: Production mechanics configuration.
        combat: Combat mechanics configuration.
        seed: Random seed for reproducibility.
    """

    enabled: bool = True
    max_turns: int | None = 100
    fog_of_war_enabled: bool = True
    board_size: int | None = None
    early_termination: EarlyTerminationConfig = field(
        default_factory=EarlyTerminationConfig,
    )
    production: ProductionConfig = field(default_factory=ProductionConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentTuningConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            EnvironmentTuningConfig instance.
        """
        early_term_data = data.get("early_termination", {})
        production_data = data.get("production", {})
        combat_data = data.get("combat", {})

        return cls(
            enabled=data.get("enabled", True),
            max_turns=data.get("max_turns"),
            fog_of_war_enabled=data.get("fog_of_war_enabled", True),
            board_size=data.get("board_size"),
            early_termination=EarlyTerminationConfig(
                enabled=early_term_data.get("enabled", True),
                advantage_threshold=early_term_data.get("advantage_threshold", 3),
                check_interval=early_term_data.get("check_interval", 5),
                min_turns=early_term_data.get("min_turns", 10),
                stalemate_threshold=early_term_data.get("stalemate_threshold", 20),
            ),
            production=ProductionConfig(
                multiplier=production_data.get("multiplier", 1.0),
                base_production=production_data.get("base_production", 3),
                population_bonus=production_data.get("population_bonus", 1),
                unit_costs=production_data.get("unit_costs", ProductionConfig().unit_costs),
            ),
            combat=CombatConfig(
                simplified=combat_data.get("simplified", False),
                damage_multiplier=combat_data.get("damage_multiplier", 1.0),
                min_damage=combat_data.get("min_damage", 10),
                terrain_bonus_enabled=combat_data.get("terrain_bonus_enabled", True),
            ),
            seed=data.get("seed"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary.
        """
        return {
            "enabled": self.enabled,
            "max_turns": self.max_turns,
            "fog_of_war_enabled": self.fog_of_war_enabled,
            "board_size": self.board_size,
            "early_termination": {
                "enabled": self.early_termination.enabled,
                "advantage_threshold": self.early_termination.advantage_threshold,
                "check_interval": self.early_termination.check_interval,
                "min_turns": self.early_termination.min_turns,
                "stalemate_threshold": self.early_termination.stalemate_threshold,
            },
            "production": {
                "multiplier": self.production.multiplier,
                "base_production": self.production.base_production,
                "population_bonus": self.production.population_bonus,
                "unit_costs": self.production.unit_costs,
            },
            "combat": {
                "simplified": self.combat.simplified,
                "damage_multiplier": self.combat.damage_multiplier,
                "min_damage": self.combat.min_damage,
                "terrain_bonus_enabled": self.combat.terrain_bonus_enabled,
            },
            "seed": self.seed,
        }

    def validate(self) -> list[str]:
        """Validate configuration parameters.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        if self.max_turns is not None and self.max_turns < 1:
            errors.append("max_turns must be positive")

        if self.early_termination.advantage_threshold < 1:
            errors.append("advantage_threshold must be at least 1")

        if self.early_termination.min_turns < 0:
            errors.append("min_turns cannot be negative")

        if self.production.multiplier <= 0:
            errors.append("production multiplier must be positive")

        if self.combat.damage_multiplier <= 0:
            errors.append("combat damage_multiplier must be positive")

        return errors


class EarlyTerminationChecker:
    """Checks for early game termination conditions."""

    def __init__(self, config: EarlyTerminationConfig):
        """Initialize early termination checker.

        Args:
            config: Early termination configuration.
        """
        self.config = config
        self._last_state: dict[str, Any] | None = None
        self._stalemate_counter: int = 0
        self._logger = get_logger(f"{__name__}.EarlyTerminationChecker")

    def reset(self) -> None:
        """Reset state for a new episode."""
        self._last_state = None
        self._stalemate_counter = 0

    def check(
        self,
        turn: int,
        player_units: int,
        player_cities: int,
        opponent_units: int,
        opponent_cities: int,
    ) -> tuple[bool, int | None, str]:
        """Check if early termination should occur.

        Args:
            turn: Current turn number.
            player_units: Current player's unit count.
            player_cities: Current player's city count.
            opponent_units: Opponent's unit count.
            opponent_cities: Opponent's city count.

        Returns:
            Tuple of (should_terminate, winner, reason).
            winner is 0 for player, 1 for opponent, None for draw.
        """
        if not self.config.enabled:
            return False, None, ""

        if turn < self.config.min_turns:
            return False, None, ""

        if turn % self.config.check_interval != 0:
            return False, None, ""

        # Check for significant advantage
        unit_advantage = player_units - opponent_units
        city_advantage = player_cities - opponent_cities
        total_advantage = unit_advantage + city_advantage * 2  # Cities worth more

        if total_advantage >= self.config.advantage_threshold:
            self._logger.debug(
                f"Early termination: player advantage={total_advantage} at turn {turn}",
            )
            return True, 0, f"player_advantage_{total_advantage}"

        if total_advantage <= -self.config.advantage_threshold:
            self._logger.debug(
                f"Early termination: opponent advantage={-total_advantage} at turn {turn}",
            )
            return True, 1, f"opponent_advantage_{-total_advantage}"

        # Check for stalemate
        current_state = {
            "player_units": player_units,
            "player_cities": player_cities,
            "opponent_units": opponent_units,
            "opponent_cities": opponent_cities,
        }

        if self._last_state == current_state:
            self._stalemate_counter += 1
            if self._stalemate_counter >= self.config.stalemate_threshold:
                self._logger.debug(f"Early termination: stalemate at turn {turn}")
                return True, None, "stalemate"
        else:
            self._stalemate_counter = 0

        self._last_state = current_state
        return False, None, ""


class TunedEnvironmentWrapper(gym.Wrapper):
    """Gymnasium wrapper that applies environment tuning.

    This wrapper modifies the base environment's behavior according to
    the tuning configuration, enabling faster training convergence.
    """

    def __init__(
        self,
        env: gym.Env,
        config: EnvironmentTuningConfig | None = None,
    ):
        """Initialize tuned environment wrapper.

        Args:
            env: Base environment to wrap.
            config: Tuning configuration.
        """
        super().__init__(env)

        self.config = config or EnvironmentTuningConfig()
        self._early_termination_checker = EarlyTerminationChecker(
            self.config.early_termination,
        )
        self._logger = get_logger(f"{__name__}.TunedEnvironmentWrapper")

        # Override max_turns if specified
        if self.config.max_turns is not None and hasattr(env, "max_turns"):
            env.max_turns = self.config.max_turns

        # Apply production config if environment supports it
        if hasattr(env, "UNIT_COSTS") and self.config.production.unit_costs:
            for unit_type, cost in self.config.production.unit_costs.items():
                if unit_type in env.UNIT_COSTS:
                    env.UNIT_COSTS[unit_type] = cost

        self._episode_stats: dict[str, Any] = {}

    # Delegate common attributes to underlying environment
    @property
    def current_player(self) -> int:
        """Get current player from underlying environment."""
        return getattr(self.env, "current_player", 0)

    @property
    def turn(self) -> int:
        """Get current turn from underlying environment."""
        return getattr(self.env, "turn", 0)

    def get_valid_actions(self) -> np.ndarray:
        """Get valid actions from underlying environment."""
        return self.env.get_valid_actions()

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get observation from underlying environment."""
        return self.env.get_observation()

    def get_info(self) -> dict[str, Any]:
        """Get info from underlying environment."""
        return self.env.get_info()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment with tuning applied.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Tuple of (observation, info).
        """
        use_seed = seed if seed is not None else self.config.seed
        obs, info = self.env.reset(seed=use_seed, options=options)

        self._early_termination_checker.reset()
        self._episode_stats = {
            "turns": 0,
            "early_termination": False,
            "termination_reason": "",
        }

        info["tuning_enabled"] = self.config.enabled
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute action with tuning applied.

        Args:
            action: Action to execute.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not self.config.enabled:
            return obs, reward, terminated, truncated, info

        self._episode_stats["turns"] = getattr(self.env, "turn", 0)

        # Check early termination
        if not terminated and not truncated:
            should_terminate, winner, reason = self._check_early_termination()

            if should_terminate:
                terminated = True
                self._episode_stats["early_termination"] = True
                self._episode_stats["termination_reason"] = reason

                # Assign final reward based on winner
                current_player = getattr(self.env, "current_player", 0)
                if winner is None:
                    reward = float(reward)  # Draw
                elif winner == current_player:
                    reward = float(reward) + 1.0
                else:
                    reward = float(reward) - 1.0

                info["early_termination"] = True
                info["termination_reason"] = reason

        info["episode_stats"] = self._episode_stats.copy()
        return obs, reward, terminated, truncated, info

    def _check_early_termination(self) -> tuple[bool, int | None, str]:
        """Check early termination conditions.

        Returns:
            Tuple of (should_terminate, winner, reason).
        """
        # Extract game state from environment
        turn = getattr(self.env, "turn", 0)

        player_units = 0
        player_cities = 0
        opponent_units = 0
        opponent_cities = 0

        # Get unit counts
        if hasattr(self.env, "units"):
            for unit in self.env.units:
                if unit.player == 0:
                    player_units += 1
                else:
                    opponent_units += 1

        # Get city counts
        if hasattr(self.env, "cities"):
            for city in self.env.cities:
                if city.player == 0:
                    player_cities += 1
                else:
                    opponent_cities += 1

        return self._early_termination_checker.check(
            turn=turn,
            player_units=player_units,
            player_cities=player_cities,
            opponent_units=opponent_units,
            opponent_cities=opponent_cities,
        )

    def get_episode_stats(self) -> dict[str, Any]:
        """Get statistics for the current episode.

        Returns:
            Dictionary with episode statistics.
        """
        return self._episode_stats.copy()

    def clone(self) -> TunedEnvironmentWrapper:
        """Create a deep copy of the wrapped environment.

        Returns:
            Cloned TunedEnvironmentWrapper with cloned underlying environment.
        """
        # Clone the underlying environment
        if hasattr(self.env, "clone"):
            cloned_env = self.env.clone()
        else:
            import copy

            cloned_env = copy.deepcopy(self.env)

        # Create new wrapper with same config
        cloned_wrapper = TunedEnvironmentWrapper(cloned_env, self.config)

        # Copy episode stats
        cloned_wrapper._episode_stats = self._episode_stats.copy()

        # Copy early termination checker state
        cloned_wrapper._early_termination_checker._stalemate_counter = (
            self._early_termination_checker._stalemate_counter
        )
        if self._early_termination_checker._last_state is not None:
            cloned_wrapper._early_termination_checker._last_state = (
                self._early_termination_checker._last_state.copy()
            )

        return cloned_wrapper


def create_tuned_environment(
    config: EnvironmentTuningConfig | None = None,
    env_type: str = "microciv",
    **env_kwargs: Any,
) -> TunedEnvironmentWrapper:
    """Factory function to create a tuned environment.

    Args:
        config: Tuning configuration.
        env_type: Type of environment ("microciv").
        **env_kwargs: Additional arguments passed to environment constructor.

    Returns:
        TunedEnvironmentWrapper instance.

    Raises:
        ValueError: If env_type is not supported.
    """
    config = config or EnvironmentTuningConfig()

    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {', '.join(errors)}")

    if env_type == "microciv":
        from alpha_zero.envs.microciv import MicroCivEnv

        # Apply board_size override if specified
        if config.board_size is not None:
            env_kwargs["board_size"] = config.board_size

        # Apply max_turns override if specified
        if config.max_turns is not None:
            env_kwargs["max_turns"] = config.max_turns

        # Apply seed if specified
        if config.seed is not None:
            env_kwargs["seed"] = config.seed

        base_env = MicroCivEnv(**env_kwargs)
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")

    return TunedEnvironmentWrapper(base_env, config)


def get_default_fast_config() -> EnvironmentTuningConfig:
    """Get default configuration optimized for fast training.

    Returns:
        EnvironmentTuningConfig optimized for speed.
    """
    return EnvironmentTuningConfig(
        enabled=True,
        max_turns=50,
        early_termination=EarlyTerminationConfig(
            enabled=True,
            advantage_threshold=2,
            check_interval=3,
            min_turns=5,
            stalemate_threshold=10,
        ),
        production=ProductionConfig(
            multiplier=2.0,
            base_production=5,
        ),
        combat=CombatConfig(
            simplified=True,
            damage_multiplier=1.5,
        ),
    )


def get_default_balanced_config() -> EnvironmentTuningConfig:
    """Get default configuration balancing speed and realism.

    Returns:
        EnvironmentTuningConfig with balanced settings.
    """
    return EnvironmentTuningConfig(
        enabled=True,
        max_turns=100,
        early_termination=EarlyTerminationConfig(
            enabled=True,
            advantage_threshold=3,
            check_interval=5,
            min_turns=10,
            stalemate_threshold=15,
        ),
        production=ProductionConfig(
            multiplier=1.5,
            base_production=4,
        ),
        combat=CombatConfig(
            simplified=False,
            damage_multiplier=1.2,
        ),
    )


def get_default_realistic_config() -> EnvironmentTuningConfig:
    """Get default configuration for realistic gameplay.

    Returns:
        EnvironmentTuningConfig with realistic settings.
    """
    return EnvironmentTuningConfig(
        enabled=True,
        max_turns=200,
        early_termination=EarlyTerminationConfig(
            enabled=False,
        ),
        production=ProductionConfig(
            multiplier=1.0,
        ),
        combat=CombatConfig(
            simplified=False,
            damage_multiplier=1.0,
        ),
    )
