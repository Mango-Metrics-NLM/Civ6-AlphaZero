"""Model Evaluation Module for Civ6 AlphaZero.

This module provides comprehensive model evaluation capabilities including:
- Evaluation against configurable baselines (random, heuristic, neural network)
- ELO rating tracking using the existing tournament system
- Game diversity analysis (opening entropy, game length distribution)
- Report generation in multiple formats (JSON, Markdown)
- Sequential testing for early stopping when results are decisive

Quick Start:
    from alpha_zero.evaluation import ModelEvaluator, EvaluationConfig

    # Load configuration
    config = EvaluationConfig.from_yaml("configs/evaluation.yaml")

    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate_model("checkpoints/model_1000.pt")

    # Print results
    for baseline, result in results.items():
        print(f"{baseline}: {result.win_rate:.1%} win rate")

    # Generate report
    from alpha_zero.evaluation import ReportGenerator
    generator = ReportGenerator(config)
    report = generator.generate(results, evaluator.get_diversity_metrics())
    report.to_json("evaluation_report.json")

Available Classes:
    - EvaluationConfig: Master configuration for evaluation
    - BaselineConfig: Configuration for baseline opponents
    - DiversityConfig: Configuration for diversity tracking
    - ModelEvaluator: Main evaluation orchestrator
    - RandomPlayer: Random action selection baseline
    - HeuristicPlayer: Rule-based heuristic baseline
    - NeuralNetworkPlayer: Neural network baseline with optional MCTS
    - BaselineFactory: Factory for creating baselines from config
    - DiversityTracker: Tracks game diversity metrics
    - DiversityMetrics: Aggregated diversity metrics
    - EvaluationResult: Results from evaluating against one baseline
    - EvaluationReport: Complete evaluation report
    - ReportGenerator: Generates reports in various formats
    - SequentialTester: Sequential testing for early stopping in evaluation
"""

from alpha_zero.evaluation.baselines import (
    BaselineFactory,
    BaselinePlayer,
    HeuristicPlayer,
    NeuralNetworkPlayer,
    RandomPlayer,
)
from alpha_zero.evaluation.config import (
    BaselineConfig,
    BaselineType,
    DiversityConfig,
    EvaluationConfig,
    HeuristicWeights,
)
from alpha_zero.evaluation.diversity import DiversityMetrics, DiversityTracker
from alpha_zero.evaluation.evaluator import (
    EvaluationResult,
    GameResult,
    ModelEvaluator,
)
from alpha_zero.evaluation.reports import EvaluationReport, ReportGenerator
from alpha_zero.evaluation.sequential_testing import (
    EvaluationDecision,
    SequentialTestingConfig,
    SequentialTestResult,
    SequentialTester,
    get_sequential_tester,
)

__all__ = [
    # Configuration
    "EvaluationConfig",
    "BaselineConfig",
    "BaselineType",
    "DiversityConfig",
    "HeuristicWeights",
    # Baselines
    "BaselinePlayer",
    "RandomPlayer",
    "HeuristicPlayer",
    "NeuralNetworkPlayer",
    "BaselineFactory",
    # Evaluation
    "ModelEvaluator",
    "GameResult",
    "EvaluationResult",
    # Diversity
    "DiversityTracker",
    "DiversityMetrics",
    # Reports
    "EvaluationReport",
    "ReportGenerator",
    # Sequential Testing (Phase 6)
    "EvaluationDecision",
    "SequentialTestingConfig",
    "SequentialTestResult",
    "SequentialTester",
    "get_sequential_tester",
]
