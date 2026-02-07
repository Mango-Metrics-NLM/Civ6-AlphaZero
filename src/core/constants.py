"""Shared numeric constants for the Strategic Planning SDK.

Centralises magic numbers that were previously scattered across modules.
Import from here rather than hard-coding values.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------
EPSILON: float = 1e-8
"""Small epsilon used to prevent division-by-zero or log-of-zero."""

EPSILON_SOFTMAX: float = 1e-10
"""Slightly smaller epsilon used in temperature-scaled softmax to avoid NaN."""

# ---------------------------------------------------------------------------
# Elo rating system
# ---------------------------------------------------------------------------
INITIAL_ELO_RATING: float = 1000.0
"""Starting Elo rating for new players/models."""

DEFAULT_ELO_K_FACTOR: float = 32.0
"""Default K-factor controlling Elo rating sensitivity."""

ELO_BASE: float = 10.0
"""Base of the exponent in the Elo expected-score formula."""

ELO_SCALE: float = 400.0
"""Denominator in the Elo exponent (rating difference for 10:1 expected score)."""

# ---------------------------------------------------------------------------
# Self-play defaults
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE_SCHEDULE: dict[int, float] = {0: 1.0, 30: 0.1}
"""Default temperature schedule mapping step-threshold -> temperature."""

DIRICHLET_NOISE_STEP_THRESHOLD: int = 30
"""Steps beyond which Dirichlet exploration noise is disabled."""

# ---------------------------------------------------------------------------
# Environment / adapter defaults
# ---------------------------------------------------------------------------
DEFAULT_BOARD_SIZE: int = 16
"""Default board dimension for grid-based environments."""

DEFAULT_MAX_STEPS: int = 200
"""Default maximum steps before a game is truncated."""

MICROCIV_SPATIAL_CHANNELS: int = 13
"""Number of spatial channels in MicroCiv state encoding."""

MICROCIV_VECTOR_DIM: int = 64
"""Vector feature dimension for MicroCiv."""

MICROCIV_ACTION_SPACE: int = 356
"""Fallback action-space size when MicroCiv env doesn't expose .action_space."""

STUB_SPATIAL_CHANNELS: int = 6
"""Number of spatial channels in stub mode."""

STUB_VECTOR_DIM: int = 4
"""Vector feature dimension in stub mode."""

RESOURCE_COLLECTION_REWARD: float = 0.1
"""Small intermediate reward for collecting a resource in MiniResource."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
MAX_LOGFILE_BYTES: int = 10 * 1024 * 1024
"""Maximum log file size before rotation (10 MiB)."""

LOGFILE_BACKUP_COUNT: int = 5
"""Number of rotated log files to keep."""
