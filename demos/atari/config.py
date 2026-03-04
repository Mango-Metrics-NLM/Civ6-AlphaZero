"""Unified configuration for Atari AlphaZero demo.

All configuration is centralized here using dataclasses with sensible defaults.
No hardcoded values — everything is configurable via constructor, YAML, or
environment variables.

Example:
    >>> config = AtariConfig.from_yaml("demos/atari/configs/pong.yaml")
    >>> config = AtariConfig.for_game("Breakout")
    >>> config = AtariConfig()  # sensible defaults
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix for all Atari demo overrides
# ---------------------------------------------------------------------------
ENV_PREFIX: str = "ATARI_DEMO_"


def _env_override(key: str, default: Any, cast: type = str) -> Any:
    """Read an environment variable override with type casting.

    Args:
        key: Suffix after ENV_PREFIX (e.g. "GAME_NAME").
        default: Value to return when the variable is unset.
        cast: Type to cast the string value to.

    Returns:
        The overridden value or *default*.
    """
    raw = os.environ.get(f"{ENV_PREFIX}{key}")
    if raw is None:
        return default
    try:
        if cast is bool:
            return raw.lower() in ("1", "true", "yes")
        return cast(raw)
    except (ValueError, TypeError):
        logger.warning("Could not cast env %s%s=%r to %s", ENV_PREFIX, key, raw, cast)
        return default


# ============================================================================
# Sub-configs
# ============================================================================


@dataclass
class AtariEnvConfig:
    """Configuration for the Atari environment wrapper.

    Attributes:
        game_name: Gymnasium Atari game identifier (e.g. "Pong", "Breakout").
        frame_height: Height to resize frames to.
        frame_width: Width to resize frames to.
        frame_stack: Number of consecutive frames to stack.
        frame_skip: Number of frames to repeat each action.
        grayscale: Convert frames to grayscale.
        max_episode_steps: Hard limit on episode length (0 = use env default).
        noop_max: Maximum number of no-op actions at episode start.
        terminal_on_life_loss: Treat life loss as terminal (for training).
        clip_rewards: Clip rewards to {-1, 0, +1} (standard Atari practice).
        sticky_actions: Probability of repeating previous action (stochasticity).
        vector_dim: Dimension of the auxiliary vector observation.
    """

    game_name: str = "Pong"
    frame_height: int = 84
    frame_width: int = 84
    frame_stack: int = 4
    frame_skip: int = 4
    grayscale: bool = True
    grayscale_weights: tuple[float, float, float] = (0.2989, 0.5870, 0.1140)
    max_episode_steps: int = 0
    noop_max: int = 30
    terminal_on_life_loss: bool = True
    clip_rewards: bool = True
    sticky_actions: float = 0.0
    vector_dim: int = 32
    gymnasium_version: str = "v5"

    # Computed helpers -------------------------------------------------------

    @property
    def spatial_channels(self) -> int:
        """Number of spatial channels after preprocessing."""
        channels_per_frame = 1 if self.grayscale else 3
        return self.frame_stack * channels_per_frame

    @property
    def spatial_size(self) -> tuple[int, int]:
        """(height, width) of preprocessed spatial observation."""
        return (self.frame_height, self.frame_width)

    @property
    def gymnasium_id(self) -> str:
        """Full Gymnasium environment ID."""
        return f"ALE/{self.game_name}-{self.gymnasium_version}"

    def apply_env_overrides(self) -> AtariEnvConfig:
        """Apply ``ATARI_DEMO_*`` environment-variable overrides in-place."""
        self.game_name = _env_override("GAME_NAME", self.game_name)
        self.frame_height = _env_override("FRAME_HEIGHT", self.frame_height, int)
        self.frame_width = _env_override("FRAME_WIDTH", self.frame_width, int)
        self.frame_stack = _env_override("FRAME_STACK", self.frame_stack, int)
        self.frame_skip = _env_override("FRAME_SKIP", self.frame_skip, int)
        self.grayscale = _env_override("GRAYSCALE", self.grayscale, bool)
        self.clip_rewards = _env_override("CLIP_REWARDS", self.clip_rewards, bool)
        return self


@dataclass
class AtariNetworkConfig:
    """Configuration for the Atari neural network.

    Attributes:
        num_res_blocks: Number of residual blocks in the spatial encoder.
        num_filters: Convolutional filter count.
        fc_hidden_dim: Hidden dimension after flattening spatial features.
        value_head_hidden: Hidden dimension in the value head.
        dropout: Dropout probability (0 = disabled).
    """

    num_res_blocks: int = 6
    num_filters: int = 64
    fc_hidden_dim: int = 256
    value_head_hidden: int = 128
    dropout: float = 0.0

    def apply_env_overrides(self) -> AtariNetworkConfig:
        """Apply environment-variable overrides."""
        self.num_res_blocks = _env_override("NUM_RES_BLOCKS", self.num_res_blocks, int)
        self.num_filters = _env_override("NUM_FILTERS", self.num_filters, int)
        self.fc_hidden_dim = _env_override("FC_HIDDEN_DIM", self.fc_hidden_dim, int)
        return self


@dataclass
class AtariMCTSConfig:
    """Configuration for MCTS in the Atari demo.

    Attributes:
        num_simulations: Number of MCTS simulations per move.
        c_puct: Exploration constant for UCB selection.
        dirichlet_alpha: Alpha parameter for Dirichlet exploration noise.
        dirichlet_epsilon: Weight of Dirichlet noise at the root.
        temperature: Temperature for action selection.
        temperature_threshold: Step after which temperature drops to 0.
    """

    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.25
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 500

    def apply_env_overrides(self) -> AtariMCTSConfig:
        """Apply environment-variable overrides."""
        self.num_simulations = _env_override("NUM_SIMULATIONS", self.num_simulations, int)
        self.c_puct = _env_override("C_PUCT", self.c_puct, float)
        return self


@dataclass
class AtariTrainingConfig:
    """Configuration for the Atari training loop.

    Attributes:
        num_iterations: Number of training iterations.
        episodes_per_iteration: Self-play episodes per iteration.
        batch_size: Mini-batch size for gradient updates.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularisation coefficient.
        gradient_clip: Maximum gradient norm.
        num_epochs: Training epochs per iteration.
        replay_buffer_size: Maximum replay buffer capacity.
        min_replay_size: Minimum samples before training starts.
        discount_factor: Discount factor (gamma) for returns.
        checkpoint_dir: Directory for model checkpoints.
        log_dir: Directory for training logs.
        save_interval: Iterations between checkpoint saves.
        eval_episodes: Number of evaluation episodes.
        eval_interval: Iterations between evaluations.
        device: PyTorch device string.
    """

    num_iterations: int = 100
    episodes_per_iteration: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    num_epochs: int = 5
    replay_buffer_size: int = 100_000
    min_replay_size: int = 500
    discount_factor: float = 0.99
    checkpoint_dir: str = "checkpoints/atari"
    log_dir: str = "logs/atari"
    save_interval: int = 10
    eval_episodes: int = 5
    eval_interval: int = 10
    device: str = ""
    training_sample_multiplier: int = 50
    return_norm_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not self.device:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

    def apply_env_overrides(self) -> AtariTrainingConfig:
        """Apply environment-variable overrides."""
        self.num_iterations = _env_override("NUM_ITERATIONS", self.num_iterations, int)
        self.episodes_per_iteration = _env_override(
            "EPISODES_PER_ITER", self.episodes_per_iteration, int
        )
        self.batch_size = _env_override("BATCH_SIZE", self.batch_size, int)
        self.learning_rate = _env_override("LEARNING_RATE", self.learning_rate, float)
        self.discount_factor = _env_override("DISCOUNT_FACTOR", self.discount_factor, float)
        self.device = _env_override("DEVICE", self.device)
        return self


@dataclass
class AtariHierarchicalConfig:
    """Configuration for the optional hierarchical reasoning layer.

    When *enabled*, the Atari demo uses a 3-level goal hierarchy
    (strategic / tactical / operational) adapted from the core Civ6
    hierarchical reasoning system.

    Attributes:
        enabled: Whether hierarchical reasoning is active.
        strategic_update_freq: Steps between strategic-level re-planning.
        tactical_update_freq: Steps between tactical-level re-planning.
        strategic_horizon: Planning horizon (in steps) for strategic level.
        tactical_horizon: Planning horizon (in steps) for tactical level.
        operational_horizon: Planning horizon (in steps) for operational level.
        goal_reward_shaping_weight: How much goal progress shapes the reward.
        max_active_goals_per_level: Maximum concurrent goals per level.
    """

    enabled: bool = False
    strategic_update_freq: int = 500
    tactical_update_freq: int = 50
    strategic_horizon: int = 1000
    tactical_horizon: int = 200
    operational_horizon: int = 20
    goal_reward_shaping_weight: float = 0.1
    max_active_goals_per_level: int = 3
    tactical_progress_increment: float = 0.05
    strategic_progress_increment: float = 0.01

    def apply_env_overrides(self) -> AtariHierarchicalConfig:
        """Apply environment-variable overrides."""
        self.enabled = _env_override("HIERARCHICAL_ENABLED", self.enabled, bool)
        self.strategic_update_freq = _env_override(
            "STRATEGIC_UPDATE_FREQ", self.strategic_update_freq, int
        )
        self.tactical_update_freq = _env_override(
            "TACTICAL_UPDATE_FREQ", self.tactical_update_freq, int
        )
        return self


# ============================================================================
# Master config
# ============================================================================


@dataclass
class AtariConfig:
    """Master configuration aggregating all Atari demo sub-configs.

    Supports construction from:
        - Keyword arguments / dataclass defaults
        - YAML file (``AtariConfig.from_yaml(...)``)
        - Game-specific presets (``AtariConfig.for_game("Breakout")``)
        - Environment variable overrides (``config.apply_env_overrides()``)
    """

    environment: AtariEnvConfig = field(default_factory=AtariEnvConfig)
    network: AtariNetworkConfig = field(default_factory=AtariNetworkConfig)
    mcts: AtariMCTSConfig = field(default_factory=AtariMCTSConfig)
    training: AtariTrainingConfig = field(default_factory=AtariTrainingConfig)
    hierarchical: AtariHierarchicalConfig = field(default_factory=AtariHierarchicalConfig)
    novel_features: Any = field(default=None)  # Optional[NovelFeaturesConfig]

    # Pre-defined game presets ------------------------------------------------

    GAME_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {
        "Pong": {
            "environment": {"game_name": "Pong", "terminal_on_life_loss": True},
            "mcts": {"dirichlet_alpha": 0.25},
        },
        "Breakout": {
            "environment": {"game_name": "Breakout", "terminal_on_life_loss": True},
            "mcts": {"dirichlet_alpha": 0.25},
        },
        "SpaceInvaders": {
            "environment": {"game_name": "SpaceInvaders", "terminal_on_life_loss": True},
            "mcts": {"dirichlet_alpha": 0.25},
        },
        "MsPacman": {
            "environment": {"game_name": "MsPacman", "terminal_on_life_loss": True},
            "mcts": {"dirichlet_alpha": 0.25},
        },
    }

    # Factory methods ---------------------------------------------------------

    @classmethod
    def for_game(cls, game_name: str, **overrides: Any) -> AtariConfig:
        """Create configuration for a specific Atari game.

        Falls back to sensible defaults if *game_name* has no preset.

        Args:
            game_name: Atari game identifier.
            **overrides: Additional keyword overrides applied on top.

        Returns:
            AtariConfig instance.
        """
        preset = cls.GAME_PRESETS.get(game_name, {"environment": {"game_name": game_name}})
        return cls.from_dict({**preset, **overrides})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtariConfig:
        """Create configuration from a nested dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            AtariConfig instance.
        """
        env_data = data.get("environment", {})
        net_data = data.get("network", {})
        mcts_data = data.get("mcts", {})
        train_data = data.get("training", {})
        hier_data = data.get("hierarchical", {})
        novel_data = data.get("novel_features", {})

        novel_features = None
        if novel_data:
            from demos.atari.novel_features import NovelFeaturesConfig

            novel_features = NovelFeaturesConfig(**novel_data)

        return cls(
            environment=AtariEnvConfig(**env_data),
            network=AtariNetworkConfig(**net_data),
            mcts=AtariMCTSConfig(**mcts_data),
            training=AtariTrainingConfig(**train_data),
            hierarchical=AtariHierarchicalConfig(**hier_data),
            novel_features=novel_features,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> AtariConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            AtariConfig instance.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If *path* does not exist.
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML configuration loading")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}

        # Allow wrapping under an ``atari`` key
        if "atari" in data:
            data = data["atari"]

        return cls.from_dict(data)

    def apply_env_overrides(self) -> AtariConfig:
        """Apply all ``ATARI_DEMO_*`` environment-variable overrides."""
        self.environment.apply_env_overrides()
        self.network.apply_env_overrides()
        self.mcts.apply_env_overrides()
        self.training.apply_env_overrides()
        self.hierarchical.apply_env_overrides()

        # Novel features env overrides
        reflection_flag = _env_override("REFLECTION_ENABLED", None, bool)
        rag_flag = _env_override("RAG_ENABLED", None, bool)
        if reflection_flag is not None or rag_flag is not None:
            if self.novel_features is None:
                from demos.atari.novel_features import NovelFeaturesConfig

                self.novel_features = NovelFeaturesConfig()
            if reflection_flag is not None:
                self.novel_features.reflection_enabled = reflection_flag
            if rag_flag is not None:
                self.novel_features.rag_enabled = rag_flag

        return self
