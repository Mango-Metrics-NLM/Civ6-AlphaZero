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
    # Configuration
    "ActionSpaceConfig",
    "Civ6Config",
    # Agent Tools
    "CodeExplorer",
    "CodeLocation",
    "CodeSymbol",
    "ConfigManager",
    "DebugHelper",
    "DistributedConfig",
    "EnvironmentConfig",
    "FileManager",
    # Logging
    "GameLogger",
    "LogConfig",
    "LoggingConfig",
    "MCTSConfig",
    "MetricsTracker",
    "NetworkConfig",
    # Test Runner (new names)
    "PytestRunner",
    "RewardsConfig",
    "RunConfig",
    "RunResult",
    "RunStatus",
    "Task",
    "TaskTracker",
    # Test Runner (backwards compatibility)
    "TestConfig",
    "TestResult",
    "TestRunner",
    "TestStatus",
    "TrainingConfig",
    "TrainingLogger",
    "UncivConfig",
    "explore",
    "find_definition",
    "get_default_config",
    "get_logger",
    "load_config",
    "merge_configs",
    "quick_test",
    "run_tests",
    "save_config",
    "setup_logging",
    "timed_section",
    "timer",
]

# Add LLM exports if available
if _LLM_AVAILABLE:
    __all__.extend(
        [
            "LLMClient",
            "LLMConfig",
            "LLMError",
            "LLMMessage",
            "LLMProvider",
            "LLMResponse",
            "create_llm_client",
            "llm_generate",
        ]
    )

# Experiment tracker (MLflow-backed). log_metrics is the supported generic
# surface; the W&B-era names (WandbLogger, get_wandb_logger, init_wandb)
# remain importable as the LAST compat surface but are lazy and emit a
# DeprecationWarning — see __getattr__ below.
try:
    from alpha_zero.utils.experiment_tracker import (  # noqa: F401
        log_metrics,
    )

    _WANDB_AVAILABLE = True
    __all__.extend(
        [
            "WandbLogger",
            "get_wandb_logger",
            "init_wandb",
            "log_metrics",
        ]
    )
except ImportError:
    _WANDB_AVAILABLE = False

#: Legacy W&B-era names served lazily with a DeprecationWarning (PEP 562).
_LEGACY_TRACKER_NAMES = frozenset({"WandbLogger", "get_wandb_logger", "init_wandb"})


def __getattr__(name: str):  # noqa: ANN202
    """Lazy, deprecated aliases for the retired W&B-era tracker names."""
    if name in _LEGACY_TRACKER_NAMES:
        import warnings

        warnings.warn(
            f"alpha_zero.utils.{name} is deprecated; use "
            "alpha_zero.monitoring.mlflow_logger.WandbCompatAdapter / "
            "alpha_zero.utils.experiment_tracker instead (W&B is retired).",
            DeprecationWarning,
            stacklevel=2,
        )
        from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter
        from alpha_zero.utils.experiment_tracker import get_tracker, init_tracker

        return {
            "WandbLogger": WandbCompatAdapter,
            "get_wandb_logger": get_tracker,
            "init_wandb": init_tracker,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        "AdaptiveDifficultyController",
        "CombatConfig",
        # Curriculum Learning
        "CurriculumConfig",
        "CurriculumManager",
        "CurriculumStage",
        "CurriculumState",
        "DifficultyDimension",
        "EarlyTerminationChecker",
        "EarlyTerminationConfig",
        "EnhancedRewardShaper",
        # Enhanced Rewards
        "EnhancedRewardsConfig",
        "EnhancedStateFeatures",
        # Enhanced Training Config
        "EnhancedTrainingConfig",
        # Environment Tuning
        "EnvironmentTuningConfig",
        "IntermediateRewardsConfig",
        "ParameterSchedule",
        "ProductionConfig",
        "ResourceConfig",
        "ResourceManager",
        "SchedulePoint",
        "ScheduleType",
        "TunedEnvironmentWrapper",
        "VictoryBonusConfig",
        "WinCondition",
        "WinConditionEvaluator",
        "create_curriculum_manager",
        "create_enhanced_reward_shaper",
        "create_training_config",
        "create_tuned_environment",
        "get_aggressive_rewards_config",
        "get_balanced_rewards_config",
        "get_default_balanced_config",
        "get_default_fast_config",
        "get_default_realistic_config",
        "get_defensive_rewards_config",
        "get_fast_training_config",
        "get_production_training_config",
    ]
)
