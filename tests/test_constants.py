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

from src.core import constants as sdk_constants


class TestDefaults:
    """Default values for test configurations.

    Values that correspond to SDK constants reference them directly to
    prevent drift between tests and source.
    """

    # Board/Map sizes — derived from SDK constants
    BOARD_SIZE: int = sdk_constants.DEFAULT_BOARD_SIZE
    SMALL_BOARD_SIZE: int = 8  # For faster unit tests
    FULL_MAP_WIDTH: int = 84  # Full Civ6 map dimensions
    FULL_MAP_HEIGHT: int = 54

    # Encoder channel counts — derived from SDK constants
    MICROCIV_CHANNELS: int = sdk_constants.MICROCIV_SPATIAL_CHANNELS
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
    VECTOR_DIM: int = sdk_constants.MICROCIV_VECTOR_DIM

    # DDP/Distributed configuration
    WORLD_SIZE: int = 4
    MASTER_PORT: str = "29500"

    # Action space — derived from SDK constants
    ACTION_SPACE_SIZE: int = sdk_constants.MICROCIV_ACTION_SPACE
    NUM_DIRECTIONS: int = 8
    NUM_UNIT_ACTIONS: int = 6


class EncoderChannels:
    """Encoder channel counts for different environments.

    Use these when creating test fixtures that need to match
    specific encoder configurations.
    """

    MICROCIV: int = sdk_constants.MICROCIV_SPATIAL_CHANNELS
    CIV6_FULL: int = 84
    CIVREALM: int = 51


class TestTimeouts:
    """Timeout values for test configurations."""

    POLL_INTERVAL_MS: int = 10
    COMMAND_TIMEOUT_MS: int = 500
    HEARTBEAT_TIMEOUT_S: float = 1.0
    RETRY_DELAY_S: float = 0.1
    MAX_RETRIES: int = 3
