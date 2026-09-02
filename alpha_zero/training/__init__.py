"""Training modules for Civ6 AlphaZero.

This package contains training infrastructure including:
- MuZero trainer with trajectory-based training
- Curriculum learning and transfer learning
- Replay buffers for experience storage
- Training profiling for diminishing returns detection
- Elo plateau detection for training stagnation
"""

from alpha_zero.training.muzero_trainer import (
    GameTrajectory,
    MuZeroTrainer,
    MuZeroTrainingConfig,
    TrajectoryBuffer,
)

# Phase 6: Elo Plateau Detector
from .elo_plateau_detector import (
    EloPlateauDetector,
    EloPlateauDetectorConfig,
    EloSnapshot,
    EloStatus,
    get_elo_detector,
)

# Phase 9: MuZero Integration
from .muzero_integration import (
    MuZeroIntegrationConfig,
    MuZeroTrainingBridge,
)

# Phase 6: Training Profiler
# Phase 8: Training Safeguards & Reports
from .training_report import (
    TrainingReport,
    TrainingReportConfig,
    TrainingReportData,
)
from .training_safeguards import (
    CheckpointValidation,
    GradientStatus,
    Intervention,
    InterventionAction,
    SafeguardReport,
    TrainingSafeguard,
    TrainingSafeguardConfig,
)

__all__ = [
    # Phase 8: Training Safeguards
    "CheckpointValidation",
    # Main pipeline
    # Configuration
    # Phase 6: Training Profiler
    "EloPlateauDetector",
    "EloPlateauDetectorConfig",
    "EloSnapshot",
    # Phase 6: Elo Plateau Detector
    "EloStatus",
    # Environment management
    # MuZero components
    "GameTrajectory",
    "GradientStatus",
    "Intervention",
    "InterventionAction",
    "MuZeroIntegrationConfig",
    "MuZeroTrainer",
    "MuZeroTrainingBridge",
    "MuZeroTrainingConfig",
    "SafeguardReport",
    "TrainingReport",
    "TrainingReportConfig",
    "TrainingReportData",
    "TrainingSafeguard",
    "TrainingSafeguardConfig",
    "TrajectoryBuffer",
    # Weight transfer
    # Factory functions
    "get_elo_detector",
]
