"""Tournament system module for Civ6 AlphaZero.

This module provides an ELO-based tournament system for competitive
matchmaking and skill-based opponent selection.

Components:
- config: Configuration dataclasses for tournament system
- elo: ELO rating system implementation
- matchmaking: Opponent selection and queue management
- league: Tier/league system with promotion/demotion
- history: Match history persistence and statistics
- orchestrator: Tournament orchestration and bracket management
- wandb_dashboards: W&B visualization for tournament metrics
"""

from alpha_zero.tournament.analysis import (
    GenerationComparison,
    MetaAnalysis,
    PlayerTrend,
    TournamentAnalyzer,
)
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
from alpha_zero.tournament.orchestrator import (
    Bracket,
    Matchup,
    PlayerStanding,
    TournamentFormat,
    TournamentOrchestrator,
    TournamentResult,
)
from alpha_zero.tournament.wandb_dashboards import TournamentWandbLogger

__all__ = [
    # Supporting classes
    "Bracket",
    # Config
    "ELOConfig",
    # ELO
    "ELORatingSystem",
    "GenerationComparison",
    # League
    "LeagueManager",
    # History
    "MatchHistory",
    # Matchmaking
    "MatchMaker",
    "MatchOutcome",
    "MatchRecord",
    "MatchResult",
    "MatchmakingConfig",
    "Matchup",
    "MetaAnalysis",
    "OpponentSelector",
    "PendingMatch",
    "PlayerStanding",
    "PlayerStatistics",
    "PlayerStats",
    "PlayerTrend",
    "QueueEntry",
    "RatingLeaderboard",
    "RatingUpdate",
    "SeasonInfo",
    "TierConfig",
    "TierProgress",
    # Analysis
    "TournamentAnalyzer",
    "TournamentConfig",
    "TournamentFormat",
    # Orchestrator
    "TournamentOrchestrator",
    "TournamentResult",
    # W&B Dashboards
    "TournamentWandbLogger",
    # Utility functions
    "calculate_expected_score",
    "calculate_rating_change",
]
