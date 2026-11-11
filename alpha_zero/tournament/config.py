"""Configuration dataclasses for Tournament system.

This module defines configuration structures for the tournament system
including ELO settings, tier configuration, and player statistics.

All values are configurable via dataclasses, YAML files, or environment
variables with the prefix CIV6AZ_TOURNAMENT_.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.utils.compat import UTC


class MatchOutcome(str, Enum):
    """Possible outcomes of a match."""

    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


@dataclass
class ELOConfig:
    """Configuration for ELO rating system.

    Attributes:
        initial_rating: Starting rating for new players
        k_factor_base: Base K-factor for rating changes
        k_factor_min: Minimum K-factor after many games
        k_factor_provisional_games: Games before K-factor starts decreasing
        provisional_threshold: Games needed to leave provisional status
        rating_floor: Minimum possible rating
        rating_ceiling: Maximum possible rating
        draw_value: Score value for draws (0.5 standard)
    """

    initial_rating: float = 1500.0
    k_factor_base: float = 32.0
    k_factor_min: float = 10.0
    k_factor_provisional_games: int = 30
    provisional_threshold: int = 30
    rating_floor: float = 100.0
    rating_ceiling: float = 3000.0
    draw_value: float = 0.5

    def get_k_factor(self, games_played: int, is_provisional: bool = False) -> float:
        """Calculate K-factor based on games played.

        Args:
            games_played: Number of games the player has played
            is_provisional: Whether player is still provisional

        Returns:
            K-factor for rating calculation.
        """
        if is_provisional or games_played < self.provisional_threshold:
            return self.k_factor_base

        # Decay K-factor with more games
        decay = min(1.0, (games_played - self.provisional_threshold) / 100)
        return self.k_factor_base - decay * (self.k_factor_base - self.k_factor_min)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial_rating": self.initial_rating,
            "k_factor_base": self.k_factor_base,
            "k_factor_min": self.k_factor_min,
            "k_factor_provisional_games": self.k_factor_provisional_games,
            "provisional_threshold": self.provisional_threshold,
            "rating_floor": self.rating_floor,
            "rating_ceiling": self.rating_ceiling,
            "draw_value": self.draw_value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ELOConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class TierConfig:
    """Configuration for a single tier/league.

    Attributes:
        name: Display name of the tier
        rating_min: Minimum rating for this tier
        rating_max: Maximum rating for this tier
        promotion_threshold: Win rate needed for promotion consideration
        demotion_threshold: Win rate below which demotion is considered
        color: Display color for UI (hex string)
        icon: Optional icon identifier
    """

    name: str
    rating_min: float
    rating_max: float
    promotion_threshold: float = 0.55
    demotion_threshold: float = 0.45
    color: str = "#FFFFFF"
    icon: str | None = None

    def contains_rating(self, rating: float) -> bool:
        """Check if rating falls within this tier.

        Args:
            rating: Rating to check

        Returns:
            True if rating is within tier bounds.
        """
        return self.rating_min <= rating < self.rating_max

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "rating_min": self.rating_min,
            "rating_max": self.rating_max,
            "promotion_threshold": self.promotion_threshold,
            "demotion_threshold": self.demotion_threshold,
            "color": self.color,
            "icon": self.icon,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TierConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def default_tiers() -> list[TierConfig]:
    """Create default tier configuration."""
    return [
        TierConfig("Bronze", 0, 1200, color="#CD7F32"),
        TierConfig("Silver", 1200, 1500, color="#C0C0C0"),
        TierConfig("Gold", 1500, 1800, color="#FFD700"),
        TierConfig("Platinum", 1800, 2100, color="#E5E4E2"),
        TierConfig("Diamond", 2100, 2400, color="#B9F2FF"),
        TierConfig("Master", 2400, float("inf"), color="#9966CC"),
    ]


@dataclass
class MatchmakingConfig:
    """Configuration for matchmaking.

    Attributes:
        rating_range_initial: Initial rating range for matching
        rating_range_max: Maximum rating range after expansion
        rating_range_expand_rate: Points to expand range per second
        max_wait_time: Maximum time to wait before expanding to max
        avoid_recent_opponents: Number of recent opponents to avoid
        prefer_similar_games_played: Prefer opponents with similar experience
    """

    rating_range_initial: float = 200.0
    rating_range_max: float = 500.0
    rating_range_expand_rate: float = 10.0
    max_wait_time: float = 60.0
    avoid_recent_opponents: int = 3
    prefer_similar_games_played: bool = True

    def get_rating_range(self, wait_time: float) -> float:
        """Get current rating range based on wait time.

        Args:
            wait_time: Seconds spent waiting

        Returns:
            Current rating range.
        """
        expanded = self.rating_range_initial + wait_time * self.rating_range_expand_rate
        return min(expanded, self.rating_range_max)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rating_range_initial": self.rating_range_initial,
            "rating_range_max": self.rating_range_max,
            "rating_range_expand_rate": self.rating_range_expand_rate,
            "max_wait_time": self.max_wait_time,
            "avoid_recent_opponents": self.avoid_recent_opponents,
            "prefer_similar_games_played": self.prefer_similar_games_played,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchmakingConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TournamentConfig:
    """Complete tournament system configuration.

    Attributes:
        elo: ELO rating configuration
        matchmaking: Matchmaking configuration
        tiers: List of tier configurations
        games_per_match: Number of games per match (for best-of)
        time_control_seconds: Time control per player
        database_path: Path to SQLite database for persistence
        log_matches: Whether to log all matches
        log_dir: Directory for logs
        enable_seasons: Whether to enable seasonal resets
        season_duration_days: Days per season
    """

    elo: ELOConfig = field(default_factory=ELOConfig)
    matchmaking: MatchmakingConfig = field(default_factory=MatchmakingConfig)
    tiers: list[TierConfig] = field(default_factory=default_tiers)
    games_per_match: int = 2
    time_control_seconds: float = 300.0
    database_path: str = "data/tournament.db"
    log_matches: bool = True
    log_dir: str = "logs/tournament"
    enable_seasons: bool = False
    season_duration_days: int = 30

    ENV_PREFIX: str = field(default="CIV6AZ_TOURNAMENT_", repr=False)

    def get_tier_for_rating(self, rating: float) -> TierConfig | None:
        """Get the tier that contains the given rating.

        Args:
            rating: Player rating

        Returns:
            TierConfig or None if no tier matches.
        """
        for tier in self.tiers:
            if tier.contains_rating(rating):
                return tier
        return None

    def get_tier_name(self, rating: float) -> str:
        """Get tier name for a rating.

        Args:
            rating: Player rating

        Returns:
            Tier name or "Unknown".
        """
        tier = self.get_tier_for_rating(rating)
        return tier.name if tier else "Unknown"

    def get_tier_by_name(self, name: str) -> TierConfig | None:
        """Get tier by name.

        Args:
            name: Tier name

        Returns:
            TierConfig or None if not found.
        """
        for tier in self.tiers:
            if tier.name == name:
                return tier
        return None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TournamentConfig:
        """Load from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def from_env(cls, base_config: TournamentConfig | None = None) -> TournamentConfig:
        """Create with environment variable overrides."""
        config = base_config or cls()
        prefix = config.ENV_PREFIX

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            param_name = key[len(prefix) :].lower()

            # Handle nested configs
            if param_name.startswith("elo_"):
                nested_param = param_name[4:]
                if hasattr(config.elo, nested_param):
                    current = getattr(config.elo, nested_param)
                    setattr(config.elo, nested_param, cls._convert_value(value, current))
            elif param_name.startswith("matchmaking_"):
                nested_param = param_name[12:]
                if hasattr(config.matchmaking, nested_param):
                    current = getattr(config.matchmaking, nested_param)
                    setattr(config.matchmaking, nested_param, cls._convert_value(value, current))
            elif hasattr(config, param_name):
                current = getattr(config, param_name)
                setattr(config, param_name, cls._convert_value(value, current))

        return config

    @staticmethod
    def _convert_value(value: str, current: Any) -> Any:
        """Convert string value to appropriate type."""
        if isinstance(current, bool):
            return value.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            return int(value)
        elif isinstance(current, float):
            return float(value)
        return value

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> TournamentConfig:
        """Create from dictionary."""
        elo_data = data.pop("elo", {})
        matchmaking_data = data.pop("matchmaking", {})
        tiers_data = data.pop("tiers", None)

        config = cls(
            elo=ELOConfig.from_dict(elo_data) if elo_data else ELOConfig(),
            matchmaking=MatchmakingConfig.from_dict(matchmaking_data)
            if matchmaking_data
            else MatchmakingConfig(),
        )

        if tiers_data:
            config.tiers = [TierConfig.from_dict(t) for t in tiers_data]

        # Set remaining fields
        for key, value in data.items():
            if hasattr(config, key) and key != "ENV_PREFIX":
                setattr(config, key, value)

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "elo": self.elo.to_dict(),
            "matchmaking": self.matchmaking.to_dict(),
            "tiers": [t.to_dict() for t in self.tiers],
            "games_per_match": self.games_per_match,
            "time_control_seconds": self.time_control_seconds,
            "database_path": self.database_path,
            "log_matches": self.log_matches,
            "log_dir": self.log_dir,
            "enable_seasons": self.enable_seasons,
            "season_duration_days": self.season_duration_days,
        }

    def save(self, path: str | Path) -> None:
        """Save to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class PlayerStats:
    """Statistics for a player/agent.

    Attributes:
        player_id: Unique player identifier
        display_name: Display name
        rating: Current ELO rating
        rating_deviation: Rating uncertainty (for Glicko)
        games_played: Total games played
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws
        win_streak: Current win streak
        best_streak: Best win streak ever
        best_rating: Highest rating achieved
        tier: Current tier name
        is_provisional: Whether still in provisional period
        rating_history: History of (timestamp, rating) tuples
        created_at: Account creation timestamp
        last_game_at: Last game timestamp
    """

    player_id: str
    display_name: str = ""
    rating: float = 1500.0
    rating_deviation: float = 350.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    win_streak: int = 0
    best_streak: int = 0
    best_rating: float = 1500.0
    tier: str = "Bronze"
    is_provisional: bool = True
    rating_history: list[tuple[str, float]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_game_at: str | None = None

    def __post_init__(self) -> None:
        """Initialize display name if not set."""
        if not self.display_name:
            self.display_name = self.player_id

    @property
    def win_rate(self) -> float:
        """Legacy win rate ``wins / games_played`` (draws count as losses).

        Kept verbatim: ``games_played`` is an independent counter and the
        rating record is persisted with this key. Report :attr:`score_rate`
        beside it. Allow-listed in ``tests/regression/draw_aware_allowlist.toml``.
        """
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    @property
    def score_rate(self) -> float:
        """Draw-aware companion: ``(wins + 0.5 * draws) / (wins + draws + losses)``."""
        from alpha_zero.evaluation.scoring import score_rate

        return score_rate(self.wins, self.draws, self.losses)

    @property
    def loss_rate(self) -> float:
        """Calculate loss rate."""
        if self.games_played == 0:
            return 0.0
        return self.losses / self.games_played

    def record_game(
        self,
        outcome: MatchOutcome,
        rating_change: float,
        tier_config: TierConfig | None = None,
    ) -> None:
        """Record a game result.

        Args:
            outcome: Game outcome
            rating_change: Rating change from this game
            tier_config: Config for updating tier
        """
        self.games_played += 1
        self.rating += rating_change
        self.last_game_at = datetime.now(UTC).isoformat()

        if outcome == MatchOutcome.WIN:
            self.wins += 1
            self.win_streak += 1
            if self.win_streak > self.best_streak:
                self.best_streak = self.win_streak
        elif outcome == MatchOutcome.LOSS:
            self.losses += 1
            self.win_streak = 0
        else:
            self.draws += 1

        if self.rating > self.best_rating:
            self.best_rating = self.rating

        # Update provisional status
        if self.games_played >= 30:
            self.is_provisional = False

        # Update tier
        if tier_config and tier_config.contains_rating(self.rating):
            self.tier = tier_config.name

        # Record history
        self.rating_history.append((datetime.now(UTC).isoformat(), self.rating))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "display_name": self.display_name,
            "rating": self.rating,
            "rating_deviation": self.rating_deviation,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_streak": self.win_streak,
            "best_streak": self.best_streak,
            "best_rating": self.best_rating,
            "tier": self.tier,
            "is_provisional": self.is_provisional,
            "win_rate": self.win_rate,
            "rating_history": self.rating_history,
            "created_at": self.created_at,
            "last_game_at": self.last_game_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerStats:
        """Create from dictionary."""
        # Remove computed fields
        data = {k: v for k, v in data.items() if k != "win_rate"}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> PlayerStats:
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class MatchResult:
    """Result of a match between two players.

    Attributes:
        match_id: Unique match identifier
        player1_id: First player ID
        player2_id: Second player ID
        player1_score: Score/games won by player 1
        player2_score: Score/games won by player 2
        player1_rating_before: Player 1 rating before match
        player2_rating_before: Player 2 rating before match
        player1_rating_change: Player 1 rating change
        player2_rating_change: Player 2 rating change
        winner_id: ID of winner (None for draw)
        game_details: List of individual game results
        duration_seconds: Match duration
        timestamp: When match occurred
    """

    match_id: str
    player1_id: str
    player2_id: str
    player1_score: float = 0.0
    player2_score: float = 0.0
    player1_rating_before: float = 1500.0
    player2_rating_before: float = 1500.0
    player1_rating_change: float = 0.0
    player2_rating_change: float = 0.0
    winner_id: str | None = None
    game_details: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def is_draw(self) -> bool:
        """Check if match was a draw."""
        return self.winner_id is None and self.player1_score == self.player2_score

    def get_outcome_for_player(self, player_id: str) -> MatchOutcome:
        """Get outcome from a player's perspective.

        Args:
            player_id: Player to get outcome for

        Returns:
            MatchOutcome enum.
        """
        if self.is_draw:
            return MatchOutcome.DRAW
        if self.winner_id == player_id:
            return MatchOutcome.WIN
        return MatchOutcome.LOSS

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "match_id": self.match_id,
            "player1_id": self.player1_id,
            "player2_id": self.player2_id,
            "player1_score": self.player1_score,
            "player2_score": self.player2_score,
            "player1_rating_before": self.player1_rating_before,
            "player2_rating_before": self.player2_rating_before,
            "player1_rating_change": self.player1_rating_change,
            "player2_rating_change": self.player2_rating_change,
            "winner_id": self.winner_id,
            "game_details": self.game_details,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "is_draw": self.is_draw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchResult:
        """Create from dictionary."""
        # Remove computed fields
        data = {k: v for k, v in data.items() if k != "is_draw"}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
