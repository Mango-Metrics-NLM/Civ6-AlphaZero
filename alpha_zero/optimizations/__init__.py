"""Performance optimizations for AlphaZero training.

This module provides various performance optimizations based on 2025 research:

- **Playout Cap Randomization**: Variable MCTS simulations for up to 50x speedup
- **Policy Target Pruning**: Reduce action space by 50%+ while preserving quality
- **Cyclic Learning Rates**: 5 schedule types for improved training stability
- **Auxiliary Policy Targets**: Richer training signals (attention, opponent modeling)

Quick Start:
    from alpha_zero.optimizations import PerformanceOptimizationsConfig

    # Use a preset
    config = PerformanceOptimizationsConfig.balanced_preset()

    # Or customize
    config = PerformanceOptimizationsConfig(
        enable_playout_cap=True,
        enable_cyclic_lr=True,
    )

Individual Components:
    from alpha_zero.optimizations.playout_cap import PlayoutCapRandomizer
    from alpha_zero.optimizations.policy_pruning import PolicyTargetPruner
    from alpha_zero.optimizations.cyclic_lr import CyclicLRScheduler
    from alpha_zero.optimizations.auxiliary_targets import AuxiliaryTargetGenerator

Benchmarking:
    from alpha_zero.optimizations.benchmarks import OptimizationBenchmark

    benchmark = OptimizationBenchmark()
    results = benchmark.run_all_benchmarks()
    print(results)
"""

from alpha_zero.optimizations.auxiliary_targets import (
    AuxiliaryLossCalculator,
    AuxiliaryTargetGenerator,
    AuxiliaryTargets,
    AuxiliaryTargetsConfig,
    GamePhase,
    create_auxiliary_system,
)
from alpha_zero.optimizations.benchmarks import (
    BenchmarkResult,
    ComparisonResult,
    OptimizationBenchmark,
)
from alpha_zero.optimizations.config import PerformanceOptimizationsConfig
from alpha_zero.optimizations.cyclic_lr import (
    CycleInfo,
    CyclicLRConfig,
    CyclicLRScheduler,
    CyclicScheduleType,
)
from alpha_zero.optimizations.playout_cap import (
    PlayoutCapConfig,
    PlayoutCapRandomizer,
)
from alpha_zero.optimizations.policy_pruning import (
    PolicyPruningConfig,
    PolicyTargetPruner,
)

__all__ = [
    # Main config
    "PerformanceOptimizationsConfig",
    # Playout cap
    "PlayoutCapConfig",
    "PlayoutCapRandomizer",
    # Policy pruning
    "PolicyPruningConfig",
    "PolicyTargetPruner",
    # Cyclic LR
    "CyclicLRConfig",
    "CyclicLRScheduler",
    "CyclicScheduleType",
    "CycleInfo",
    # Auxiliary targets
    "AuxiliaryTargetsConfig",
    "AuxiliaryTargets",
    "AuxiliaryTargetGenerator",
    "AuxiliaryLossCalculator",
    "GamePhase",
    "create_auxiliary_system",
    # Benchmarks
    "OptimizationBenchmark",
    "BenchmarkResult",
    "ComparisonResult",
]
