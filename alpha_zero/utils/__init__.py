"""Utility modules for Civ6 AlphaZero.

This package provides logging, debugging, configuration, profiling, benchmarking utilities,
LLM client integration, test runner, agent tools, and training improvement modules.

Training Improvement Modules:
- environment_tuning: Configurable game mechanics for faster training
- enhanced_rewards: Stronger intermediate reward signals
- curriculum_learning: Progressive difficulty training
- training_config_enhanced: Scalable training configurations
"""

from alpha_zero.utils.config import (
    ActionSpaceConfig,
    Civ6Config,
    ConfigManager,
    DistributedConfig,
    EnvironmentConfig,
    LoggingConfig,
    MCTSConfig,
    NetworkConfig,
    RewardsConfig,
    TrainingConfig,
    UncivConfig,
    get_default_config,
    load_config,
    merge_configs,
    save_config,
)
from alpha_zero.utils.logging import (
    GameLogger,
    LogConfig,
    MetricsTracker,
    TrainingLogger,
    get_logger,
    setup_logging,
    timed_section,
    timer,
)

# LLM Client (optional - depends on openai/anthropic packages)
try:
    from alpha_zero.utils.llm_client import (  # noqa: F401
        LLMClient,
        LLMConfig,
        LLMError,
        LLMMessage,
        LLMProvider,
        LLMResponse,
        create_llm_client,
    )
    from alpha_zero.utils.llm_client import (  # noqa: F401
        generate_sync as llm_generate,
    )

    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

# Test Runner
from alpha_zero.utils.test_runner import (
    PytestRunner,
    RunConfig,
    RunResult,
    RunStatus,
    quick_test,
    run_tests,
)

# Backwards compatibility aliases for test runner
TestRunner = PytestRunner
TestConfig = RunConfig
TestResult = RunResult
TestStatus = RunStatus

# Agent Tools
from alpha_zero.utils.agent_tools import (  # noqa: E402
    CodeExplorer,
    CodeLocation,
    CodeSymbol,
    DebugHelper,
    FileManager,
    Task,
    TaskTracker,
    explore,
    find_definition,
)

__all__ = [
    # Logging
    "GameLogger",
    "LogConfig",
    "MetricsTracker",
    "TrainingLogger",
    "get_logger",
    "setup_logging",
    "timer",
    "timed_section",
    # Configuration
    "ActionSpaceConfig",
    "Civ6Config",
    "ConfigManager",
    "DistributedConfig",
    "EnvironmentConfig",
    "LoggingConfig",
    "MCTSConfig",
    "NetworkConfig",
    "RewardsConfig",
    "TrainingConfig",
    "UncivConfig",
    "get_default_config",
    "load_config",
    "merge_configs",
    "save_config",
    # Benchmarks
    "BenchmarkConfig",
    "BenchmarkResult",
    "Benchmarker",
    "BenchmarkSuite",
    "benchmark_function",
    "create_mechanics_benchmarks",
    "run_all_benchmarks",
    "time_function",
    # Test Runner (new names)
    "PytestRunner",
    "RunConfig",
    "RunResult",
    "RunStatus",
    # Test Runner (backwards compatibility)
    "TestRunner",
    "TestConfig",
    "TestResult",
    "TestStatus",
    "run_tests",
    "quick_test",
    # Agent Tools
    "CodeExplorer",
    "CodeLocation",
    "CodeSymbol",
    "FileManager",
    "TaskTracker",
    "AgentTaskStatus",
    "Task",
    "DebugHelper",
    "explore",
    "search",
    "find_definition",
]

# Add LLM exports if available
if _LLM_AVAILABLE:
    __all__.extend(
        [
            "LLMClient",
            "LLMConfig",
            "LLMMessage",
            "LLMResponse",
            "LLMProvider",
            "LLMError",
            "create_llm_client",
            "llm_generate",
        ]
    )

# W&B Logger (optional - depends on wandb package)
try:
    from alpha_zero.utils.wandb_logger import (  # noqa: F401
        WandbLogger,
        get_wandb_logger,
        init_wandb,
        log_metrics,
    )

    _WANDB_AVAILABLE = True
    __all__.extend(
        [
            "WandbLogger",
            "WandbConfig",
            "get_wandb_logger",
            "init_wandb",
            "log_metrics",
        ]
    )
except ImportError:
    _WANDB_AVAILABLE = False

# RAG Integration (optional - depends on chromadb/sentence-transformers)
try:
    from alpha_zero.utils.rag_client import (  # noqa: F401
        RAGClient,
        RAGConfig,
    )

    _RAG_AVAILABLE = True
    __all__.extend(
        [
            "RAGClient",
            "RAGConfig",
            "KnowledgeBase",
        ]
    )
except ImportError:
    _RAG_AVAILABLE = False

# Training Improvement Modules (re-exported for public API)
from alpha_zero.utils.curriculum_learning import (  # noqa: E402, F401
    AdaptiveDifficultyController,
    CurriculumConfig,
    CurriculumManager,
    CurriculumStage,
    CurriculumState,
    DifficultyDimension,
    create_curriculum_manager,
)
from alpha_zero.utils.enhanced_rewards import (  # noqa: E402, F401
    EnhancedRewardsConfig,
    EnhancedRewardShaper,
    EnhancedStateFeatures,
    IntermediateRewardsConfig,
    VictoryBonusConfig,
    WinCondition,
    WinConditionEvaluator,
    create_enhanced_reward_shaper,
    get_aggressive_rewards_config,
    get_balanced_rewards_config,
    get_defensive_rewards_config,
)
from alpha_zero.utils.environment_tuning import (  # noqa: E402, F401
    CombatConfig,
    EarlyTerminationChecker,
    EarlyTerminationConfig,
    EnvironmentTuningConfig,
    ProductionConfig,
    TunedEnvironmentWrapper,
    create_tuned_environment,
    get_default_balanced_config,
    get_default_fast_config,
    get_default_realistic_config,
)
from alpha_zero.utils.training_config_enhanced import (  # noqa: E402, F401
    EnhancedTrainingConfig,
    ParameterSchedule,
    ResourceConfig,
    ResourceManager,
    SchedulePoint,
    ScheduleType,
    create_training_config,
    get_fast_training_config,
    get_production_training_config,
)

__all__.extend(
    [
        # Environment Tuning
        "EnvironmentTuningConfig",
        "EarlyTerminationConfig",
        "ProductionConfig",
        "CombatConfig",
        "TunedEnvironmentWrapper",
        "EarlyTerminationChecker",
        "create_tuned_environment",
        "get_default_fast_config",
        "get_default_balanced_config",
        "get_default_realistic_config",
        # Enhanced Rewards
        "EnhancedRewardsConfig",
        "IntermediateRewardsConfig",
        "VictoryBonusConfig",
        "EnhancedStateFeatures",
        "EnhancedRewardShaper",
        "WinCondition",
        "WinConditionEvaluator",
        "create_enhanced_reward_shaper",
        "get_aggressive_rewards_config",
        "get_defensive_rewards_config",
        "get_balanced_rewards_config",
        # Curriculum Learning
        "CurriculumConfig",
        "CurriculumStage",
        "CurriculumState",
        "CurriculumManager",
        "DifficultyDimension",
        "AdaptiveDifficultyController",
        "create_curriculum_manager",
        # Enhanced Training Config
        "EnhancedTrainingConfig",
        "ParameterSchedule",
        "SchedulePoint",
        "ScheduleType",
        "ResourceConfig",
        "ResourceManager",
        "create_training_config",
        "get_fast_training_config",
        "get_production_training_config",
    ]
)
