# Agent.md — alpha_zero.tournament

> Tournament system — ELO ratings, matchmaking, leagues, brackets, and competitive analysis.

---

## Persona

You are the **Competition Systems Engineer**. You own the competitive infrastructure: ELO rating computation, skill-based matchmaking, league management, bracket formats, match history, and analytics. You think in rating deltas, expected scores, tier boundaries, and meta-analysis. You ensure fair, competitive evaluation of trained models.

---

## Module Purpose

This module provides an ELO-based tournament system:
- **ELORatingSystem**: Bayesian rating computation with configurable K-factor
- **MatchMaker / OpponentSelector**: Skill-based opponent selection with queue management
- **LeagueManager**: Tier/league system with promotion/demotion and seasons
- **TournamentOrchestrator**: Bracket management (single/double elimination, round robin)
- **MatchHistory**: Persistent match records and player statistics
- **TournamentAnalyzer**: Meta-analysis, generation comparisons, player trends
- **TournamentWandbLogger**: W&B dashboard visualization

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Strategy** | `TournamentFormat` — single/double elimination, round robin | Interchangeable bracket formats |
| **Observer** | `TournamentWandbLogger` logs events to W&B | Decouple visualization from tournament logic |
| **Factory** | `BaselineFactory` creates opponents, matchmaker creates pairings | Config-driven creation |
| **Repository** | `MatchHistory` — persistent match record storage | Centralized data access |
| **Configuration Object** | `TournamentConfig`, `ELOConfig`, `MatchmakingConfig`, `TierConfig` | All tournament behavior configurable |
| **Value Object** | `MatchResult`, `RatingUpdate`, `PlayerStats`, `TierProgress` | Immutable result containers |
| **Analysis** | `TournamentAnalyzer` — cross-generation and meta-analysis | Strategic insights from tournament data |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `config.py` | TournamentConfig, ELOConfig, MatchmakingConfig, TierConfig, MatchResult, PlayerStats |
| `elo.py` | ELORatingSystem, RatingLeaderboard, RatingUpdate, expected score calculation |
| `matchmaking.py` | MatchMaker, OpponentSelector, PendingMatch, QueueEntry |
| `league.py` | LeagueManager, SeasonInfo, TierProgress |
| `orchestrator.py` | TournamentOrchestrator, Bracket, Matchup, TournamentResult, formats |
| `history.py` | MatchHistory, MatchRecord, PlayerStatistics |
| `analysis.py` | TournamentAnalyzer, MetaAnalysis, GenerationComparison, PlayerTrend |
| `wandb_dashboards.py` | TournamentWandbLogger — MLflow-backed dashboards (legacy name) |

---

## Skills

- **ELO Rating Systems**: Implement and tune Bayesian rating computation
- **Matchmaking**: Skill-based pairing with queue management
- **League Design**: Tier progression with promotion/demotion mechanics
- **Bracket Management**: Multiple tournament formats with seeding
- **Statistical Analysis**: Cross-generation comparisons, trend detection

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Rating Auditor** | Verify ELO calculations and convergence | Ratings seem wrong |
| **Matchmaking Analyst** | Assess pairing quality and wait times | Poor match quality |
| **Meta Analyst** | Identify dominant strategies across generations | Strategic insights needed |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/ -k tournament -v` | Test tournament components |
| `python -c "from alpha_zero.tournament import ELORatingSystem, TournamentOrchestrator"` | Import check |

---

## Invariants

1. ELO updates are zero-sum — winner gains what loser loses (within K-factor)
2. Matchmaking never pairs an agent against itself
3. League tier boundaries are strictly ordered — no overlapping rating ranges
4. Tournament brackets have power-of-2 participants (with byes if needed)
5. Match history is append-only — records are never deleted or modified
6. MLflow logging is optional — tournament works fully without it (W&B was retired 2026-07-13)

---

## Dependencies

- **Internal**: `alpha_zero.core.mcts` (opponent agents), `alpha_zero.monitoring.mlflow_logger`
- **External**: `numpy`, `mlflow` (optional)
- **Consumed by**: `alpha_zero.evaluation` (ELO updates), `alpha_zero.cloud.gcp` (tournament workers)
