"""Centralized test configuration constants.

All test files should import these values rather than hardcoding them.
This ensures consistency across tests and makes it easier to update
values when encoder or environment configurations change.

Usage:
    from tests.test_constants import TestDefaults

    def test_something():
        board_size = TestDefaults.BOARD_SIZE
        channels = TestDefaults.MICROCIV_CHANNELS
"""

from __future__ import annotations


class TestDefaults:
    """Default values for test configurations.

    These values are derived from the actual encoder and environment
    implementations to ensure test consistency.
    """

    # Board/Map sizes
    BOARD_SIZE: int = 16  # Default MicroCiv board size
    SMALL_BOARD_SIZE: int = 8  # For faster unit tests
    FULL_MAP_WIDTH: int = 84  # Full Civ6 map dimensions
    FULL_MAP_HEIGHT: int = 54

    # Encoder channel counts (from civ6_encoder.py)
    MICROCIV_CHANNELS: int = 13  # MicroCivEncoder.TOTAL_CHANNELS
    CIV6_CHANNELS: int = 84  # Civ6StateEncoder.TOTAL_CHANNELS
    CIVREALM_CHANNELS: int = 51  # CivRealmEncoder.TOTAL_CHANNELS

    # Training/batch sizes
    BATCH_SIZE: int = 4  # Small batch for fast tests
    LARGE_BATCH_SIZE: int = 32  # For integration tests

    # Game configuration
    MAX_TURNS: int = 50  # Short games for tests
    MAX_UNITS: int = 20  # Per player
    MAX_CITIES: int = 5  # Per player

    # MCTS configuration
    NUM_SIMULATIONS: int = 5  # Minimal for fast tests
    MCTS_C_PUCT: float = 1.5

    # Network configuration
    NUM_RES_BLOCKS: int = 2  # Minimal for fast tests
    NUM_FILTERS: int = 32  # Smaller for fast tests
    VECTOR_DIM: int = 64  # MicroCiv vector dimension

    # DDP/Distributed configuration
    WORLD_SIZE: int = 4
    MASTER_PORT: str = "29500"

    # Action space
    ACTION_SPACE_SIZE: int = 356  # MicroCiv default
    NUM_DIRECTIONS: int = 8
    NUM_UNIT_ACTIONS: int = 6


class EncoderChannels:
    """Encoder channel counts for different environments.

    Use these when creating test fixtures that need to match
    specific encoder configurations.
    """

    MICROCIV: int = 13
    CIV6_FULL: int = 84
    CIVREALM: int = 51


class TestTimeouts:
    """Timeout values for test configurations."""

    POLL_INTERVAL_MS: int = 10
    COMMAND_TIMEOUT_MS: int = 500
    HEARTBEAT_TIMEOUT_S: float = 1.0
    RETRY_DELAY_S: float = 0.1
    MAX_RETRIES: int = 3


class SpacetimeDBDefaults:
    """Default values for SpacetimeDB test configurations."""

    HOST: str = "http://localhost:3000"
    MODULE_ADDRESS: str = "test-module"
    RECONNECT_INTERVAL: float = 0.1  # Fast for tests
    MAX_RECONNECT_ATTEMPTS: int = 2
    CONNECTION_TIMEOUT: float = 1.0
    SUBSCRIPTION_TIMEOUT: float = 1.0
    MAX_WEIGHT_BLOB_MB: int = 10  # Small for tests
