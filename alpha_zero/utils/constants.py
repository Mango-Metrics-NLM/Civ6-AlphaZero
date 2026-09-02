"""
Constants and Configuration Values for Civ6-AlphaZero

This module centralizes all constants and configurable values used throughout
the codebase. No magic numbers should be hardcoded elsewhere - all should
reference these constants or be loaded from configuration.

Design Principles:
- All numeric constants are named and documented
- Related constants are grouped in dataclasses
- All values can be overridden via configuration
- Clear separation between game rules and AI hyperparameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# =============================================================================
# ENCODER CONSTANTS
# =============================================================================


@dataclass(frozen=True)
class EncoderConstants:
    """Constants for state encoding.

    Controls normalization factors and channel counts for neural network input.
    """

    # Channel counts
    MICROCIV_CHANNELS: int = 13
    FULL_ENCODER_CHANNELS: int = 84

    # MicroCiv channel ranges
    TERRAIN_CHANNELS: tuple[int, int] = (0, 3)  # plains, hills, mountain
    OWN_UNIT_CHANNELS: tuple[int, int] = (3, 6)  # warrior, settler, archer
    ENEMY_UNIT_CHANNELS: tuple[int, int] = (6, 9)
    OWN_CITY_CHANNEL: int = 9
    ENEMY_CITY_CHANNEL: int = 10
    VISIBILITY_CHANNEL: int = 11
    BIAS_CHANNEL: int = 12

    # Full encoder channel ranges
    TERRAIN_TYPE_RANGE: tuple[int, int] = (0, 7)
    TERRAIN_FEATURE_RANGE: tuple[int, int] = (7, 15)
    RESOURCE_RANGE: tuple[int, int] = (15, 27)
    OWN_UNITS_RANGE: tuple[int, int] = (27, 39)
    ENEMY_UNITS_RANGE: tuple[int, int] = (39, 51)
    OWN_CITIES_RANGE: tuple[int, int] = (51, 55)
    ENEMY_CITIES_RANGE: tuple[int, int] = (55, 59)
    BORDER_RANGE: tuple[int, int] = (59, 63)
    FOG_RANGE: tuple[int, int] = (63, 66)
    YIELDS_RANGE: tuple[int, int] = (66, 71)
    IMPROVEMENTS_RANGE: tuple[int, int] = (71, 79)
    MOVEMENT_RANGE: tuple[int, int] = (79, 84)

    # Normalization factors
    YIELD_NORMALIZATION: float = 10.0
    # Score normalization (typical max score for progress calculation)
    MAX_TYPICAL_SCORE: float = 1000.0
    POPULATION_NORMALIZATION: float = 30.0
    HEALTH_NORMALIZATION: float = 100.0
    STRENGTH_NORMALIZATION: float = 100.0
    TURN_NORMALIZATION: float = 500.0
    PRODUCTION_NORMALIZATION: float = 500.0
    GOLD_NORMALIZATION: float = 1000.0
    SCIENCE_NORMALIZATION: float = 500.0
    CULTURE_NORMALIZATION: float = 500.0
    FAITH_NORMALIZATION: float = 500.0

    # MicroCiv vector normalization factors
    MICROCIV_TURN_NORMALIZATION: float = 200.0
    MICROCIV_GOLD_NORMALIZATION: float = 500.0
    MICROCIV_GOLD_PER_TURN_NORMALIZATION: float = 50.0

    # Full encoder vector normalization factors
    RESOURCE_NORMALIZATION: float = 20.0
    CITY_COUNT_NORMALIZATION: float = 20.0
    UNIT_COUNT_NORMALIZATION: float = 50.0
    MILITARY_STRENGTH_NORMALIZATION: float = 500.0
    PER_TURN_NORMALIZATION: float = 100.0
    CITY_PRODUCTION_NORMALIZATION: float = 100.0
    CITY_DEFENSE_NORMALIZATION: float = 100.0

    # Vector encoding dimensions
    MICROCIV_VECTOR_DIM: int = 64
    FULL_VECTOR_DIM: int = 512

    def get_channel_range(self, channel_type: str) -> tuple[int, int]:
        """Get channel range by type name.

        Args:
            channel_type: Name of the channel type.

        Returns:
            Tuple of (start, end) channel indices.
        """
        ranges = {
            "terrain_type": self.TERRAIN_TYPE_RANGE,
            "terrain_feature": self.TERRAIN_FEATURE_RANGE,
            "resource": self.RESOURCE_RANGE,
            "own_units": self.OWN_UNITS_RANGE,
            "enemy_units": self.ENEMY_UNITS_RANGE,
            "own_cities": self.OWN_CITIES_RANGE,
            "enemy_cities": self.ENEMY_CITIES_RANGE,
            "border": self.BORDER_RANGE,
            "fog": self.FOG_RANGE,
            "yields": self.YIELDS_RANGE,
            "improvements": self.IMPROVEMENTS_RANGE,
            "movement": self.MOVEMENT_RANGE,
        }
        return ranges.get(channel_type, (0, 0))


# =============================================================================
# NETWORK CONSTANTS
# =============================================================================


@dataclass(frozen=True)
class NetworkConstants:
    """Constants for neural network architecture."""

    # Default architecture parameters
    DEFAULT_NUM_FILTERS: int = 64
    DEFAULT_NUM_RES_BLOCKS: int = 6
    DEFAULT_FC_DIM: int = 256

    # Attention parameters
    DEFAULT_NUM_HEADS: int = 8
    DEFAULT_ATTENTION_DIM: int = 64
    DEFAULT_MLP_RATIO: float = 4.0
    DEFAULT_DROPOUT: float = 0.1

    # Action masking
    INVALID_ACTION_LOGIT: float = -1e9
    SOFTMAX_TEMPERATURE_MIN: float = 0.1
    SOFTMAX_TEMPERATURE_MAX: float = 2.0

    # Initialization
    WEIGHT_INIT_STD: float = 0.01
    BIAS_INIT_VALUE: float = 0.0

    # Gradient clipping
    MAX_GRADIENT_NORM: float = 1.0


# =============================================================================
# MCTS CONSTANTS
# =============================================================================


@dataclass(frozen=True)
class MCTSConstants:
    """Constants for Monte Carlo Tree Search.

    Per-variant simulation budgets are explicit so drift between Civ6MCTS,
    GumbelMCTS, MuZeroMCTS, and the orchestrator config is audited rather
    than silently accumulated across three files.
    """

    # PUCT formula parameters (shared across all variants)
    DEFAULT_C_PUCT: float = 1.5
    DEFAULT_C_PUCT_BASE: float = 19652.0
    DEFAULT_C_PUCT_INIT: float = 1.25

    # Exploration noise (shared across all variants)
    DEFAULT_DIRICHLET_ALPHA: float = 0.3
    DEFAULT_DIRICHLET_EPSILON: float = 0.25

    # Temperature annealing (shared across all variants)
    DEFAULT_TEMPERATURE: float = 1.0
    DEFAULT_TEMPERATURE_THRESHOLD: int = 30
    DEFAULT_TEMPERATURE_FINAL: float = 0.0

    # Virtual loss for parallel MCTS
    DEFAULT_VIRTUAL_LOSS: float = 3.0

    # Per-variant simulation budgets (previously duplicated and drifting)
    CLASSIC_NUM_SIMULATIONS: int = 200  # Civ6MCTS / SimpleMCTS
    GUMBEL_NUM_SIMULATIONS: int = 64  # GumbelMCTS — fewer needed via sequential halving
    MUZERO_NUM_SIMULATIONS: int = 50  # MuZeroMCTS — hidden-state based
    # Orchestrator training budget. Matches the orchestrator.MCTSConfig
    # dataclass default and configs/orchestrator.yaml (the default preset).
    # NOTE: configs/orchestrator_production.yaml overrides this to 200
    # and configs/orchestrator_fast.yaml to 10 — those are explicit
    # preset overrides, not the library default. Kept distinct from
    # CLASSIC_NUM_SIMULATIONS so the two can be tuned independently.
    ORCHESTRATOR_NUM_SIMULATIONS: int = 50

    # Backwards-compatible alias (older code referenced DEFAULT_NUM_SIMULATIONS).
    # Equals CLASSIC_NUM_SIMULATIONS — kept distinct so refactors that want the
    # classic value can be explicit.
    DEFAULT_NUM_SIMULATIONS: int = 200
    MIN_SIMULATIONS: int = 1
    MAX_SIMULATIONS: int = 10000

    # Gumbel-specific parameters
    GUMBEL_NUM_SAMPLED_ACTIONS: int = 16
    GUMBEL_MAX_CONSIDERED_ACTIONS: int = 32
    GUMBEL_C_SCALE: float = 1.0
    GUMBEL_MIN_ACTIONS_AFTER_HALVING: int = 1

    # MuZero-specific parameters
    MUZERO_DISCOUNT: float = 0.997
    MUZERO_REWARD_SUPPORT_SIZE: int = 601

    # Numerical stability
    PRIOR_EPSILON: float = 1e-8
    VALUE_EPSILON: float = 1e-6


# =============================================================================
# TRAINING CONSTANTS
# =============================================================================


@dataclass(frozen=True)
class TrainingConstants:
    """Constants for training pipeline."""

    # Learning rate
    DEFAULT_LEARNING_RATE: float = 0.001
    MIN_LEARNING_RATE: float = 1e-6
    LR_DECAY_FACTOR: float = 0.1

    # Batch parameters
    DEFAULT_BATCH_SIZE: int = 128
    MIN_BATCH_SIZE: int = 8
    MAX_BATCH_SIZE: int = 4096

    # Replay buffer
    DEFAULT_BUFFER_SIZE: int = 100000
    DEFAULT_MIN_REPLAY_SIZE: int = 1000
    SAMPLE_MULTIPLIER: int = 100

    # Optimization
    WEIGHT_DECAY: float = 1e-4
    MOMENTUM: float = 0.9
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.999

    # Loss computation
    VALUE_LOSS_WEIGHT: float = 1.0
    POLICY_LOSS_WEIGHT: float = 1.0
    L2_REG_WEIGHT: float = 1e-4

    # Checkpointing
    DEFAULT_SAVE_INTERVAL: int = 100
    DEFAULT_EVAL_INTERVAL: int = 50
    DEFAULT_EVAL_GAMES: int = 20

    # Warmup
    WARMUP_STEPS: int = 1000


# =============================================================================
# GAME CONSTANTS
# =============================================================================


@dataclass(frozen=True)
class GameConstants:
    """Constants for game mechanics."""

    # Board dimensions
    DEFAULT_BOARD_SIZE: int = 16
    MIN_BOARD_SIZE: int = 4
    MAX_BOARD_SIZE: int = 128

    # Turn limits
    DEFAULT_MAX_TURNS: int = 200
    MIN_TURNS: int = 10

    # Unit parameters
    MAX_UNITS_PER_PLAYER: int = 20
    MAX_CITIES_PER_PLAYER: int = 5
    MAX_PLAYERS: int = 8

    # Movement directions (8-directional)
    DIRECTIONS: tuple[tuple[int, int], ...] = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    # Terrain generation ratios
    PLAINS_RATIO: float = 0.7
    HILLS_RATIO: float = 0.2
    MOUNTAIN_RATIO: float = 0.1

    # Combat
    COMBAT_ADVANTAGE_MULTIPLIER: float = 1.5
    FORTIFICATION_BONUS: float = 0.25
    TERRAIN_DEFENSE_BONUS: float = 0.1

    # MicroCiv combat damage formula: damage = BASE * ratio + MIN
    COMBAT_BASE_DAMAGE: int = 30
    COMBAT_MIN_DAMAGE_ATTACKER: int = 5
    COMBAT_MIN_DAMAGE_DEFENDER: int = 10
    COMBAT_SMOOTHING_EPSILON: float = 0.1

    # MicroCiv city mechanics
    CITY_BASE_PRODUCTION: int = 2  # Added to population for production per turn
    PRODUCTION_COST_MULTIPLIER: int = 10  # Unit cost * this = production required
    MIN_CITY_DISTANCE: int = 3  # Minimum distance between cities

    # MicroCiv unit default health
    DEFAULT_UNIT_HEALTH: int = 100
    DEFAULT_CITY_DEFENSE: int = 20

    # MicroCiv settler auto-found probability
    SETTLER_AUTO_FOUND_PROBABILITY: float = 0.1

    # Shaped reward values
    PRODUCTION_START_REWARD: float = 0.01
    CITY_FOUND_REWARD: float = 1.0


@dataclass(frozen=True)
class UnitConstants:
    """Constants for unit types."""

    # Unit costs (production)
    WARRIOR_COST: int = 3
    SETTLER_COST: int = 5
    ARCHER_COST: int = 4
    HORSEMAN_COST: int = 6
    SWORDSMAN_COST: int = 5

    # Unit strength
    WARRIOR_STRENGTH: int = 20
    SETTLER_STRENGTH: int = 0
    ARCHER_STRENGTH: int = 15
    HORSEMAN_STRENGTH: int = 25
    SWORDSMAN_STRENGTH: int = 35

    # Unit movement
    WARRIOR_MOVES: int = 2
    SETTLER_MOVES: int = 2
    ARCHER_MOVES: int = 2
    HORSEMAN_MOVES: int = 4
    SWORDSMAN_MOVES: int = 2

    # Range
    ARCHER_RANGE: int = 2

    def get_unit_stats(self, unit_type: str) -> dict[str, int]:
        """Get all stats for a unit type.

        Args:
            unit_type: Name of the unit type.

        Returns:
            Dictionary with cost, strength, and moves.
        """
        unit_type = unit_type.upper()
        return {
            "cost": getattr(self, f"{unit_type}_COST", 0),
            "strength": getattr(self, f"{unit_type}_STRENGTH", 0),
            "moves": getattr(self, f"{unit_type}_MOVES", 2),
        }


# =============================================================================
# ACTION SPACE CONSTANTS
# =============================================================================


@dataclass(frozen=True)
class ActionSpaceConstants:
    """Constants for action space definition."""

    # MicroCiv action counts
    END_TURN_ACTIONS: int = 1
    DIRECTIONS_COUNT: int = 8

    # Hierarchical action space
    NUM_ACTION_CATEGORIES: int = 9
    MAX_UNIT_SELECTION: int = 256
    MAX_CITY_SELECTION: int = 64
    MAX_TECH_SELECTION: int = 128
    MAX_CIVIC_SELECTION: int = 128

    # Action type counts
    UNIT_ACTION_TYPES: int = 8
    CITY_ACTION_TYPES: int = 12
    DIPLOMACY_ACTION_TYPES: int = 6

    def compute_microciv_action_space_size(
        self,
        max_units: int = 20,
        max_cities: int = 5,
        unit_types: int = 3,
    ) -> int:
        """Compute total MicroCiv action space size.

        Args:
            max_units: Maximum units per player.
            max_cities: Maximum cities per player.
            unit_types: Number of producible unit types.

        Returns:
            Total action space size.
        """
        end_turn = self.END_TURN_ACTIONS
        move_actions = max_units * self.DIRECTIONS_COUNT
        attack_actions = max_units * self.DIRECTIONS_COUNT
        skip_actions = max_units
        # One found-city action per unit slot. Omitting this block returned 356
        # while MicroCivActionSpace.total_size is 376, and validate_constants
        # asserted the same wrong figure -- so the error validated itself green
        # and seeded the 356 literals scattered through the cloud workers.
        found_city_actions = max_units
        produce_actions = max_cities * unit_types

        return (
            end_turn
            + move_actions
            + attack_actions
            + skip_actions
            + found_city_actions
            + produce_actions
        )


# =============================================================================
# DISTRIBUTED TRAINING CONSTANTS
# =============================================================================


@dataclass(frozen=True)
class DistributedConstants:
    """Constants for distributed training."""

    # Actor configuration
    DEFAULT_NUM_ACTORS: int = 64
    MIN_ACTORS: int = 1
    MAX_ACTORS: int = 1024
    ACTOR_BATCH_SIZE: int = 8

    # Weight synchronization
    WEIGHT_SYNC_INTERVAL: int = 100
    WEIGHT_VERSION_STALENESS_THRESHOLD: int = 10

    # Queue management
    MAX_QUEUE_SIZE: int = 10000
    QUEUE_TIMEOUT_SECONDS: float = 1.0

    # Process management
    PROCESS_HEALTH_CHECK_INTERVAL: float = 30.0
    PROCESS_RESTART_DELAY: float = 5.0
    MAX_PROCESS_RESTARTS: int = 3

    # Compression
    COMPRESSION_ENABLED: bool = True
    COMPRESSION_LEVEL: int = 6  # LZMA compression level

    # Timeout values (seconds)
    TRAJECTORY_SUBMISSION_TIMEOUT: float = 5.0
    ACTOR_POOL_STOP_TIMEOUT: float = 10.0
    DEFAULT_STATE_TIMEOUT: float = 60.0
    DEFAULT_ACTION_TIMEOUT: float = 30.0


# =============================================================================
# REWARD CONSTANTS
# =============================================================================


@dataclass(frozen=True)
class RewardConstants:
    """Constants for reward shaping."""

    # Terminal rewards
    WIN_REWARD: float = 1.0
    LOSS_REWARD: float = -1.0
    DRAW_REWARD: float = 0.0

    # Intermediate rewards (optional shaping)
    CAPTURE_CITY_REWARD: float = 0.1
    DESTROY_UNIT_REWARD: float = 0.05
    BUILD_CITY_REWARD: float = 0.1
    TECH_DISCOVERED_REWARD: float = 0.05

    # Penalties
    UNIT_LOST_PENALTY: float = -0.05
    CITY_LOST_PENALTY: float = -0.1
    INVALID_ACTION_PENALTY: float = -0.01

    # MicroCiv-specific reward values
    MICROCIV_UNIT_KILL_REWARD: float = 0.1
    MICROCIV_UNIT_LOST_PENALTY: float = -0.1

    # Discount factor
    GAMMA: float = 0.99


# =============================================================================
# LOGGING CONSTANTS
# =============================================================================


@dataclass(frozen=True)
class LoggingConstants:
    """Constants for logging configuration."""

    # File sizes
    MAX_LOG_FILE_SIZE_MB: int = 10
    MAX_LOG_BACKUP_COUNT: int = 5

    # Circular buffer
    CIRCULAR_BUFFER_SIZE: int = 1000

    # Metrics
    METRICS_WINDOW_SIZE: int = 100

    # TensorBoard
    TENSORBOARD_FLUSH_SECS: int = 30


# =============================================================================
# PATH CONSTANTS
# =============================================================================


@dataclass(frozen=True)
class PathConstants:
    """Default filesystem paths used across configurable components.

    All paths are relative to the project root. Override via configuration
    (YAML / CIV6AZ_* env vars) when deploying. Centralising them here
    prevents the same literal appearing in both a class default and its
    from_dict fallback (a real source of silent drift in the codebase).
    """

    # RAG knowledge base
    RAG_PERSIST_DIR: str = "data/knowledge_db"
    RAG_KNOWLEDGE_DIR: str = "data/knowledge"
    RAG_COLLECTION_NAME: str = "civ6_knowledge"

    # LATS reflection memory
    REFLECTION_PERSIST_DIR: str = "data/reflections"
    REFLECTION_COLLECTION_NAME: str = "game_reflections"

    # Orchestrator artifacts
    ORCHESTRATOR_CHECKPOINT_DIR: str = "checkpoints/orchestrator"
    ORCHESTRATOR_LOG_DIR: str = "logs/orchestrator"

    # Storage backend root
    DEFAULT_LOCAL_STORAGE_ROOT: str = "./storage"


# =============================================================================
# GLOBAL INSTANCES (for convenience)
# =============================================================================

ENCODER = EncoderConstants()
NETWORK = NetworkConstants()
MCTS = MCTSConstants()
TRAINING = TrainingConstants()
GAME = GameConstants()
UNITS = UnitConstants()
ACTION_SPACE = ActionSpaceConstants()
DISTRIBUTED = DistributedConstants()
REWARDS = RewardConstants()
LOGGING = LoggingConstants()
PATHS = PathConstants()


def default_action_space_size() -> int:
    """Return the MicroCiv action-space size every live default must agree on.

    This is the single dynamic source for ``action_space_size`` defaults in
    ``NetworkConfig``, ``MuZeroConfig``, ``NPlayerNetworkConfig``,
    ``MicroCivNetwork``, the Civ6 bridge network builder and the DUCT factory.
    It is derived from the same per-slot arithmetic as
    ``alpha_zero.core.civ6_actions.MicroCivActionSpace.total_size`` (376 at the
    shipped ``max_units=20`` / ``max_cities=5`` / three unit types), so the two
    cannot drift independently. Before this existed, the live defaults carried a
    hand-copied ``381`` -- five wider than the environment -- which built policy
    heads whose last five logits could never be selected.

    Callers that pass ``action_space_size`` explicitly still win: a checkpoint
    saved with a 381-wide head loads unchanged when the caller says ``381``.
    """
    return ACTION_SPACE.compute_microciv_action_space_size()


#: Import-time snapshot of :func:`default_action_space_size` for use as a plain
#: default in signatures and dataclass fields. Use the function when the value
#: must be recomputed; use this when a constant expression is required.
DEFAULT_ACTION_SPACE_SIZE: int = default_action_space_size()


def get_all_constants() -> dict[str, Any]:
    """Get all constants as a dictionary.

    Returns:
        Dictionary mapping constant group names to their values.
    """
    return {
        "encoder": ENCODER,
        "network": NETWORK,
        "mcts": MCTS,
        "training": TRAINING,
        "game": GAME,
        "units": UNITS,
        "action_space": ACTION_SPACE,
        "distributed": DISTRIBUTED,
        "rewards": REWARDS,
        "logging": LOGGING,
        "paths": PATHS,
    }


def validate_constants() -> list[str]:
    """Validate all constants for consistency.

    Returns:
        List of validation errors (empty if all valid).
    """
    errors = []

    # Check game constants consistency
    terrain_sum = GAME.PLAINS_RATIO + GAME.HILLS_RATIO + GAME.MOUNTAIN_RATIO
    if abs(terrain_sum - 1.0) > 1e-9:
        errors.append("Terrain ratios must sum to 1.0")

    # Check action space computation
    expected = 376  # Must match MicroCivActionSpace.total_size (civ6_actions.py)
    computed = ACTION_SPACE.compute_microciv_action_space_size()
    if computed != expected:
        errors.append(f"Action space mismatch: expected {expected}, got {computed}")

    # Check encoder channel ranges don't overlap
    ranges = [
        ENCODER.TERRAIN_TYPE_RANGE,
        ENCODER.TERRAIN_FEATURE_RANGE,
        ENCODER.RESOURCE_RANGE,
        ENCODER.OWN_UNITS_RANGE,
        ENCODER.ENEMY_UNITS_RANGE,
    ]
    for i, r1 in enumerate(ranges):
        for r2 in ranges[i + 1 :]:
            if r1[1] > r2[0] and r2[1] > r1[0]:
                errors.append(f"Encoder channel ranges overlap: {r1} and {r2}")

    return errors


# Validate on import (can be disabled in production)
_validation_errors = validate_constants()
if _validation_errors:
    import warnings

    for error in _validation_errors:
        warnings.warn(f"Constant validation error: {error}", stacklevel=2)
