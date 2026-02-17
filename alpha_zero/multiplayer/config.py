"""Configuration for multiplayer components.

Provides dataclass-based configuration for all multiplayer subsystems.
All values are configurable via constructor, YAML, or environment variables.
No hardcoded magic numbers - defaults are documented and overridable.

Example:
    >>> config = MultiplayerConfig(num_players=4)
    >>> config.mcts.search_strategy
    'max_n'
    >>> config.scoring.method
    <ScoringMethod.LINEAR_RANKING: 'linear_ranking'>
"""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class TurnMode(enum.Enum):
    """Turn ordering modes for multiplayer games.

    Attributes:
        SEQUENTIAL: Players act one at a time in fixed order.
        SIMULTANEOUS: All players act at the same time.
        DYNAMIC: Simultaneous during peace, sequential during war.
    """

    SEQUENTIAL = "sequential"
    SIMULTANEOUS = "simultaneous"
    DYNAMIC = "dynamic"


class SearchStrategy(enum.Enum):
    """MCTS search strategy for multiplayer games.

    Attributes:
        MAX_N: Each player maximizes their own value vector component.
            Most principled but computationally expensive.
        PARANOID: Assumes all opponents collude against root player.
            Enables deeper search via alpha-beta-like pruning.
        BEST_REPLY: Only considers strongest opponent response.
            Middle ground between max_n and paranoid.
    """

    MAX_N = "max_n"
    PARANOID = "paranoid"
    BEST_REPLY = "best_reply"


@dataclass
class NPlayerMCTSConfig:
    """Configuration for N-player MCTS.

    Attributes:
        num_simulations: Total simulation budget per search.
        c_puct: Exploration constant for PUCT formula.
        dirichlet_alpha: Dirichlet noise parameter for root exploration.
        dirichlet_epsilon: Weight of Dirichlet noise vs prior.
        temperature: Sampling temperature for action selection.
        temperature_threshold: Turn after which temperature becomes greedy.
        search_strategy: Which multi-player search variant to use.
        use_opponent_modeling: Whether to incorporate opponent model predictions.
        value_normalization: Normalize values to [0, 1] range per player.
    """

    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30
    search_strategy: str = SearchStrategy.MAX_N.value
    use_opponent_modeling: bool = False
    value_normalization: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.num_simulations < 1:
            msg = f"num_simulations must be >= 1, got {self.num_simulations}"
            raise ValueError(msg)
        if self.c_puct <= 0:
            msg = f"c_puct must be > 0, got {self.c_puct}"
            raise ValueError(msg)
        if not 0 <= self.dirichlet_epsilon <= 1:
            msg = f"dirichlet_epsilon must be in [0, 1], got {self.dirichlet_epsilon}"
            raise ValueError(msg)
        # Validate search_strategy is a known value
        valid_strategies = {s.value for s in SearchStrategy}
        if self.search_strategy not in valid_strategies:
            msg = f"search_strategy must be one of {valid_strategies}, got {self.search_strategy}"
            raise ValueError(msg)

    @property
    def strategy(self) -> SearchStrategy:
        """Get search strategy as enum."""
        return SearchStrategy(self.search_strategy)


@dataclass
class ScoringConfig:
    """Configuration for game outcome scoring.

    Attributes:
        method: Scoring method name (linear_ranking, winner_take_all, proportional).
        draw_value: Value assigned to all players on draw.
        elimination_penalty: Extra penalty for eliminated players.
        score_weights: Optional per-metric weights for proportional scoring.
    """

    method: str = "linear_ranking"
    draw_value: float = 0.0
    elimination_penalty: float = 0.0
    score_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in ("linear_ranking", "winner_take_all", "proportional"):
            msg = f"Unknown scoring method: {self.method}"
            raise ValueError(msg)


@dataclass
class TurnManagerConfig:
    """Configuration for turn management.

    Attributes:
        mode: Turn ordering mode.
        turn_timeout_seconds: Maximum time per turn (0 = unlimited).
        war_detection_key: Info dict key indicating war state for dynamic mode.
        allow_concurrent_actions: Whether simultaneous actions can conflict.
    """

    mode: str = TurnMode.SEQUENTIAL.value
    turn_timeout_seconds: float = 0.0
    war_detection_key: str = "at_war"
    allow_concurrent_actions: bool = True

    def __post_init__(self) -> None:
        valid_modes = {m.value for m in TurnMode}
        if self.mode not in valid_modes:
            msg = f"mode must be one of {valid_modes}, got {self.mode}"
            raise ValueError(msg)

    @property
    def turn_mode(self) -> TurnMode:
        """Get turn mode as enum."""
        return TurnMode(self.mode)


@dataclass
class OpponentModelConfig:
    """Configuration for opponent modeling.

    Attributes:
        enabled: Whether opponent modeling is active.
        history_window: Number of past actions to consider per opponent.
        feature_dim: Dimension of opponent feature embedding.
        update_frequency: How often to update opponent model (in turns).
        decay_rate: Exponential decay for old observations.
    """

    enabled: bool = False
    history_window: int = 50
    feature_dim: int = 32
    update_frequency: int = 1
    decay_rate: float = 0.95

    def __post_init__(self) -> None:
        if self.history_window < 1:
            msg = f"history_window must be >= 1, got {self.history_window}"
            raise ValueError(msg)
        if not 0 < self.decay_rate <= 1:
            msg = f"decay_rate must be in (0, 1], got {self.decay_rate}"
            raise ValueError(msg)


@dataclass
class MultiplayerConfig:
    """Top-level configuration for multiplayer support.

    Combines all sub-configurations. Supports environment variable overrides
    with CIV6AZ_MP_ prefix.

    Attributes:
        num_players: Number of players in the game (>= 2).
        mcts: MCTS search configuration.
        scoring: Game outcome scoring configuration.
        turn_manager: Turn management configuration.
        opponent_model: Opponent modeling configuration.
        spatial_channels_per_player: Channels allocated per player in state encoding.
        city_channels_per_player: City channels per player in encoding.
        base_terrain_channels: Fixed terrain channels in encoding.
        fog_of_war: Whether fog of war is enabled.
    """

    num_players: int = 2
    mcts: NPlayerMCTSConfig = field(default_factory=NPlayerMCTSConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    turn_manager: TurnManagerConfig = field(default_factory=TurnManagerConfig)
    opponent_model: OpponentModelConfig = field(default_factory=OpponentModelConfig)
    spatial_channels_per_player: int = 3
    city_channels_per_player: int = 1
    base_terrain_channels: int = 3
    fog_of_war: bool = False

    def __post_init__(self) -> None:
        if self.num_players < 2:
            msg = f"num_players must be >= 2, got {self.num_players}"
            raise ValueError(msg)

    @property
    def total_spatial_channels(self) -> int:
        """Compute total spatial encoding channels for N players.

        Formula: base_terrain + num_players * (unit_channels + city_channels) + 2
        The +2 accounts for fog of war and constant bias channels.
        """
        player_channels = self.num_players * (
            self.spatial_channels_per_player + self.city_channels_per_player
        )
        extra_channels = 2  # fog of war + constant bias
        return self.base_terrain_channels + player_channels + extra_channels

    def apply_env_overrides(self) -> MultiplayerConfig:
        """Apply environment variable overrides with CIV6AZ_MP_ prefix.

        Returns:
            Self with environment overrides applied.
        """
        env_map: dict[str, tuple[str, type]] = {
            "CIV6AZ_MP_NUM_PLAYERS": ("num_players", int),
            "CIV6AZ_MP_SEARCH_STRATEGY": ("mcts.search_strategy", str),
            "CIV6AZ_MP_NUM_SIMULATIONS": ("mcts.num_simulations", int),
            "CIV6AZ_MP_C_PUCT": ("mcts.c_puct", float),
            "CIV6AZ_MP_TURN_MODE": ("turn_manager.mode", str),
            "CIV6AZ_MP_SCORING_METHOD": ("scoring.method", str),
            "CIV6AZ_MP_FOG_OF_WAR": ("fog_of_war", bool),
        }

        for env_var, (attr_path, attr_type) in env_map.items():
            value = os.getenv(env_var)
            if value is not None:
                parts = attr_path.split(".")
                target = self
                for part in parts[:-1]:
                    target = getattr(target, part)
                if attr_type is bool:
                    setattr(target, parts[-1], value.lower() in ("true", "1", "yes"))
                else:
                    setattr(target, parts[-1], attr_type(value))
                logger.info(
                    "Applied env override: %s=%s -> %s",
                    env_var,
                    value,
                    attr_path,
                )

        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiplayerConfig:
        """Create from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            MultiplayerConfig instance.
        """
        mcts_data = data.pop("mcts", {})
        scoring_data = data.pop("scoring", {})
        turn_data = data.pop("turn_manager", {})
        opp_data = data.pop("opponent_model", {})

        return cls(
            mcts=NPlayerMCTSConfig(**mcts_data),
            scoring=ScoringConfig(**scoring_data),
            turn_manager=TurnManagerConfig(**turn_data),
            opponent_model=OpponentModelConfig(**opp_data),
            **data,
        )
