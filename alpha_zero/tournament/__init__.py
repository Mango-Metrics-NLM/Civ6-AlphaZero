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


# Deprecated names served lazily (PEP 562) so that `import alpha_zero.tournament`
# stays silent and only an actual use of the old name warns.
_DEPRECATED_LAZY: dict[str, str] = {
    "TournamentWandbLogger": "alpha_zero.tournament.wandb_dashboards",
}


def __getattr__(name: str) -> object:
    module_path = _DEPRECATED_LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
