"""Tournament system module for Civ6 AlphaZero.

This module provides an ELO-based tournament system for competitive
matchmaking and skill-based opponent selection.

Components:
- config: Configuration dataclasses for tournament system
- elo: ELO rating system implementation
- matchmaking: Opponent selection and queue management
- league: Tier/league system with promotion/demotion
- history: Match history persistence and statistics
"""

from alpha_zero.tournament.config import (
    ELOConfig,
    MatchmakingConfig,
    MatchOutcome,
    MatchResult,
    PlayerStats,
    TierConfig,
    TournamentConfig,
)
from alpha_zero.tournament.elo import (
    ELORatingSystem,
    RatingLeaderboard,
    RatingUpdate,
    calculate_expected_score,
    calculate_rating_change,
)
from alpha_zero.tournament.history import (
    MatchHistory,
    MatchRecord,
    PlayerStatistics,
)
from alpha_zero.tournament.league import (
    LeagueManager,
    SeasonInfo,
    TierProgress,
)
from alpha_zero.tournament.matchmaking import (
    MatchMaker,
    OpponentSelector,
    PendingMatch,
    QueueEntry,
)

__all__ = [
    # Config
    "ELOConfig",
    "MatchmakingConfig",
    "MatchOutcome",
    "MatchResult",
    "PlayerStats",
    "TierConfig",
    "TournamentConfig",
    # ELO
    "ELORatingSystem",
    "RatingLeaderboard",
    "RatingUpdate",
    "calculate_expected_score",
    "calculate_rating_change",
    # History
    "MatchHistory",
    "MatchRecord",
    "PlayerStatistics",
    # League
    "LeagueManager",
    "SeasonInfo",
    "TierProgress",
    # Matchmaking
    "MatchMaker",
    "OpponentSelector",
    "PendingMatch",
    "QueueEntry",
]
