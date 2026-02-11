"""Unified Training Orchestrator for Civ6 AlphaZero.

This module provides a single entry point for all training modes:
- Local self-play and training
- Distributed training with multiple actors
- Evaluation and benchmarking

Features:
- Dynamic hyperparameter scheduling
- Curriculum learning integration
- Enhanced reward shaping
- RAG-augmented strategy analysis
- W&B and TensorBoard logging
- Checkpoint management with resume support

Usage:
    from alpha_zero.orchestrator import TrainingOrchestrator, OrchestratorConfig

    config = OrchestratorConfig.from_yaml("configs/orchestrator.yaml")
    orchestrator = TrainingOrchestrator(config)
    orchestrator.run()

CLI:
    python -m alpha_zero.orchestrator --config configs/orchestrator.yaml
    python -m alpha_zero.orchestrator --preset fast --dry_run
"""

from __future__ import annotations

from alpha_zero.orchestrator.checkpointer import CheckpointInfo, CheckpointManager
from alpha_zero.orchestrator.config import (
    AutonomyLevel,
    CurriculumConfig,
    CurriculumStageConfig,
    EnvironmentConfig,
    LLMConfig,
    LoggingConfig,
    MCTSConfig,
    NetworkConfig,
    OrchestratorConfig,
    RAGConfig,
    ResourceBudget,
    RewardsConfig,
    ScheduleConfig,
    StrategyAdvisorConfig,
    TrainingConfig,
    WandbConfig,
)
from alpha_zero.orchestrator.orchestrator import (
    GameResult,
    TrainingOrchestrator,
    TrainingState,
)
from alpha_zero.orchestrator.rag_integration import (
    QueryResult,
    RAGIntegration,
    TrainingInsight,
)
from alpha_zero.orchestrator.scheduler import HyperparameterScheduler, SchedulerState
from alpha_zero.orchestrator.strategy_advisor import (
    AnalysisResult,
    GameReport,
    MoveExplanation,
    StrategyAdvisor,
    TrainingSummary,
)

__all__ = [
    "AnalysisResult",
    "AutonomyLevel",
    "CheckpointInfo",
    # Checkpointer
    "CheckpointManager",
    "CurriculumConfig",
    "CurriculumStageConfig",
    # Training component configs
    "EnvironmentConfig",
    "GameReport",
    "GameResult",
    # Scheduler
    "HyperparameterScheduler",
    "LLMConfig",
    "LoggingConfig",
    "MCTSConfig",
    "MoveExplanation",
    "NetworkConfig",
    # Configuration
    "OrchestratorConfig",
    "QueryResult",
    # RAG/LLM configs
    "RAGConfig",
    # RAG Integration
    "RAGIntegration",
    "ResourceBudget",
    "RewardsConfig",
    "ScheduleConfig",
    "SchedulerState",
    # Strategy Advisor
    "StrategyAdvisor",
    "StrategyAdvisorConfig",
    "TrainingConfig",
    "TrainingInsight",
    # Main orchestrator
    "TrainingOrchestrator",
    "TrainingState",
    "TrainingSummary",
    "WandbConfig",
]
