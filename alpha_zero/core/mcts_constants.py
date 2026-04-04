"""Centralized constants for MCTS algorithms.

All values are configurable via environment variables with MCTS_ prefix.
This module provides sensible defaults that can be overridden at runtime.

Example:
    # Override rollout depth at runtime
    MCTS_ROLLOUT_DEPTH=30 python train.py
"""

from __future__ import annotations

import os

# =============================================================================
# Numerical Stability
# =============================================================================

EPSILON: float = float(os.getenv("MCTS_EPSILON", "1e-8"))
"""Small value for numerical stability in probability calculations."""

# =============================================================================
# Search Parameters
# =============================================================================

DEFAULT_ROLLOUT_DEPTH: int = int(os.getenv("MCTS_ROLLOUT_DEPTH", "20"))
"""Default depth for rollout simulations."""

DEFAULT_C_VISIT_DIVISOR: float = float(os.getenv("MCTS_C_VISIT_DIVISOR", "50.0"))
"""Default divisor for c_visit in node selection (controls exploration)."""

DEFAULT_C_VISIT: float = float(os.getenv("MCTS_C_VISIT", "50.0"))
"""Default c_visit value for MCTS configuration."""

GREEDY_THRESHOLD: float = float(os.getenv("MCTS_GREEDY_THRESHOLD", "0.99"))
"""Probability threshold above which action selection becomes greedy."""

# =============================================================================
# Logging Configuration
# =============================================================================

MCTS_LOG_LEVEL: str = os.getenv("MCTS_LOG_LEVEL", "INFO")
"""Log level for MCTS operations."""

LOG_SEARCH_STATS_INTERVAL: int = int(os.getenv("MCTS_LOG_INTERVAL", "100"))
"""Number of searches between statistics logging."""

LOG_NODE_EXPANSION: bool = os.getenv("MCTS_LOG_NODE_EXPANSION", "false").lower() == "true"
"""Whether to log detailed node expansion information."""

LOG_SEARCH_START: bool = os.getenv("MCTS_LOG_SEARCH_START", "true").lower() == "true"
"""Whether to log search start events."""
