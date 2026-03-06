"""
GCP Configuration for Civ6 AlphaZero Cloud Training.

Provides dataclass-based configuration with:
- Environment variable support (no hardcoded values)
- Preset configurations for budget/on-demand tiers
- Validation and serialization utilities
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Default GPU pricing (USD/hour) - can be overridden via CIV6AZ_GPU_COST_* env vars
DEFAULT_GPU_COSTS: dict[str, float] = {
    "NVIDIA_TESLA_T4": 0.35,
    "NVIDIA_TESLA_T4_SPOT": 0.11,
    "NVIDIA_L4": 0.70,
    "NVIDIA_L4_SPOT": 0.22,
    "NVIDIA_TESLA_V100": 2.48,
    "NVIDIA_TESLA_V100_SPOT": 0.74,
    "NVIDIA_TESLA_P100": 1.46,
    "NVIDIA_TESLA_P100_SPOT": 0.43,
    "NVIDIA_TESLA_A100": 3.67,
    "NVIDIA_TESLA_A100_SPOT": 1.10,
    "NVIDIA_H100_80GB": 11.60,
    "NVIDIA_H100_80GB_SPOT": 3.48,
}

# Canonical MCTS type choices for argparse across all cloud workers.
# Source of truth: aligned with alpha_zero.core.mcts_factory._MCTS_REGISTRY.
# Uses tuple (immutable) so callers can safely pass to argparse choices.
VALID_MCTS_TYPES: tuple[str, ...] = ("civ6", "gumbel", "batch_gumbel", "muzero")

# Default MCTS type when none is specified.
DEFAULT_MCTS_TYPE: str = "gumbel"

# Default batch self-play configuration (Phase 16 infrastructure, enabled in Sprint 6)
DEFAULT_NUM_PARALLEL_GAMES: int = 4  # Number of parallel games for batch MCTS
DEFAULT_MAX_BATCH_SIZE: int = 32  # Maximum batch size for network inference


def get_gpu_costs() -> dict[str, float]:
    """Get GPU costs, allowing env-var overrides.

    Each GPU type can be overridden via CIV6AZ_GPU_COST_{NAME} env var,
    where NAME is the GPU type with NVIDIA_ and TESLA_ prefixes stripped.
    Example: CIV6AZ_GPU_COST_T4=0.40 overrides NVIDIA_TESLA_T4 pricing.
    """
    costs = {}
    for gpu_type, default_cost in DEFAULT_GPU_COSTS.items():
        env_key = f"CIV6AZ_GPU_COST_{gpu_type.replace('NVIDIA_', '').replace('TESLA_', '')}"
        costs[gpu_type] = float(os.environ.get(env_key, str(default_cost)))
    return costs


class VertexTier(str, Enum):
    """Vertex AI pricing tier."""

    BUDGET = "budget"  # T4 for both self-play and training
    ON_DEMAND = "ondemand"  # A100 for both


@dataclass
class SelfPlayWorkerConfig:
    """Configuration for self-play worker jobs.

    Attributes:
        machine_type: GCE machine type (e.g., n1-standard-8).
        accelerator_type: GPU type (e.g., NVIDIA_TESLA_T4).
        accelerator_count: Number of GPUs per worker.
        replica_count: Number of parallel workers.
        spot: Use Spot VMs for cost savings.
        games_per_worker: Number of games each worker generates.
        max_retries: Maximum job retries on failure.
    """

    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    replica_count: int = 10
    spot: bool = True
    games_per_worker: int = 100
    max_retries: int = 3

    @classmethod
    def budget_preset(cls) -> SelfPlayWorkerConfig:
        """Create budget tier configuration (T4 Spot VMs)."""
        return cls(
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            replica_count=10,
            spot=True,
        )

    @classmethod
    def ondemand_preset(cls) -> SelfPlayWorkerConfig:
        """Create on-demand tier configuration (A100 GPUs)."""
        return cls(
            machine_type="a2-highgpu-1g",
            accelerator_type="NVIDIA_TESLA_A100",
            accelerator_count=1,
            replica_count=20,
            spot=False,
        )


@dataclass
class TrainingWorkerConfig:
    """Configuration for training jobs.

    Attributes:
        machine_type: GCE machine type.
        accelerator_type: GPU type.
        accelerator_count: Number of GPUs.
        spot: Use Spot VMs (not recommended for training).
        checkpoint_interval: Steps between checkpoints.
        epochs_per_iteration: Training epochs per iteration.
    """

    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    spot: bool = False
    checkpoint_interval: int = 1000
    epochs_per_iteration: int = 10
    replica_count: int = 1

    @classmethod
    def budget_preset(cls) -> TrainingWorkerConfig:
        """Create budget tier configuration (T4 GPU for better quota availability)."""
        return cls(
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            replica_count=1,
            spot=False,
        )

    @classmethod
    def ondemand_preset(cls) -> TrainingWorkerConfig:
        """Create on-demand tier configuration (8x A100)."""
        return cls(
            machine_type="a2-highgpu-8g",
            accelerator_type="NVIDIA_TESLA_A100",
            accelerator_count=8,
            replica_count=1,
            spot=False,
        )


@dataclass
class GCSPathConfig:
    """GCS path configuration.

    Defines the bucket structure for all artifacts.
    """

    models_prefix: str = "models/civ6az"
    trajectories_prefix: str = "trajectories/civ6az"
    checkpoints_prefix: str = "checkpoints/civ6az"
    logs_prefix: str = "logs/civ6az"
    agent_id: str | None = None

    def _resolve_prefix(self, base_prefix: str) -> str:
        """Return agent-scoped prefix when agent_id is set, otherwise base."""
        if self.agent_id:
            return f"{base_prefix}/{self.agent_id}"
        return base_prefix

    def model_path(self, version: int) -> str:
        """Get path for a model version."""
        return f"{self._resolve_prefix(self.models_prefix)}/v{version:03d}/model.pt"

    def metadata_path(self, version: int) -> str:
        """Get path for model metadata."""
        return f"{self._resolve_prefix(self.models_prefix)}/v{version:03d}/metadata.json"

    def trajectory_path(self, iteration: int, worker_id: int) -> str:
        """Get path for trajectory shard."""
        return f"{self._resolve_prefix(self.trajectories_prefix)}/iter_{iteration:03d}/shard_worker_{worker_id}.pt"

    def checkpoint_path(self, step: int) -> str:
        """Get path for training checkpoint."""
        return f"{self._resolve_prefix(self.checkpoints_prefix)}/training_step_{step}.pt"

    def latest_checkpoint_path(self) -> str:
        """Get path for latest checkpoint marker."""
        return f"{self._resolve_prefix(self.checkpoints_prefix)}/latest.pt"


# Architecture presets for different training scales.
# "small" matches current defaults (backwards compatible).
# "medium" matches v000's proven architecture (embed=256, layers=8).
# "large" is for future scaling experiments.
ARCHITECTURE_PRESETS: dict[str, dict[str, Any]] = {
    "small": {
        "embed_dim": 64,
        "num_layers": 6,
        "num_heads": 8,
        "policy_hidden_dim": 128,
        "value_head_dim": 64,
    },
    "medium": {
        "embed_dim": 256,
        "num_layers": 8,
        "num_heads": 8,
        "policy_hidden_dim": 256,
        "value_head_dim": 128,
    },
    "large": {
        "embed_dim": 512,
        "num_layers": 12,
        "num_heads": 16,
        "policy_hidden_dim": 512,
        "value_head_dim": 256,
    },
}


@dataclass
class NetworkArchitectureConfig:
    """Network architecture defaults for the Civ6 Transformer model.

    Provides a single source of truth for architecture parameters used by
    selfplay, training, and evaluation workers. Defaults match the current
    Civ6TransformerNetwork / MicroCivEncoder architecture ("small" preset).

    All values can be overridden via environment variables prefixed with CIV6AZ_.

    Attributes:
        board_size: Board size (spatial_size H=W).
        action_space_size: Total discrete actions (1 + 20*8 + 20*8 + 20 + 20 + 5*3 = 376).
        spatial_channels: Input spatial feature planes from MicroCivEncoder.
        vector_dim: Dimension of the vector (non-spatial) observation.
        num_layers: Number of transformer layers (maps to TransformerConfig.num_layers).
        embed_dim: Transformer embedding dimension (maps to TransformerConfig.embed_dim).
        num_heads: Number of attention heads (maps to TransformerConfig.num_heads).
        policy_hidden_dim: Hidden dimension for policy head.
        value_head_dim: Hidden dimension for value head.
        encoder_type: Spatial encoder type ("patch_embed" or "hex_crystalline").
    """

    board_size: int = 8
    action_space_size: int = 376
    spatial_channels: int = 28  # Extended from 13 to 28 for N-player support (Sprint 6)
    vector_dim: int = 64
    num_layers: int = 6
    embed_dim: int = 64
    num_heads: int = 8
    policy_hidden_dim: int = 128
    value_head_dim: int = 64
    encoder_type: str = "patch_embed"

    @classmethod
    def from_preset(cls, preset: str) -> NetworkArchitectureConfig:
        """Create from a named architecture preset.

        Available presets:
        - "small": embed=64, layers=6 (current default, backwards compatible)
        - "medium": embed=256, layers=8 (proven in v000 tournament results)
        - "large": embed=512, layers=12 (future scaling experiments)

        Args:
            preset: Preset name ('small', 'medium', 'large').

        Returns:
            NetworkArchitectureConfig instance.

        Raises:
            ConfigurationError: If preset name is not recognized.
        """
        if preset not in ARCHITECTURE_PRESETS:
            valid = ", ".join(sorted(ARCHITECTURE_PRESETS.keys()))
            raise ConfigurationError(
                f"Unknown architecture preset '{preset}'. Valid presets: {valid}"
            )
        return cls(**ARCHITECTURE_PRESETS[preset])

    @classmethod
    def from_env(cls) -> NetworkArchitectureConfig:
        """Create from environment variables.

        Environment Variables:
            CIV6AZ_BOARD_SIZE: Board size (default: 8)
            CIV6AZ_ACTION_SPACE_SIZE: Action space size (default: 376)
            CIV6AZ_SPATIAL_CHANNELS: Spatial channels (default: 13)
            CIV6AZ_VECTOR_DIM: Vector dimension (default: 64)
            CIV6AZ_NUM_LAYERS: Transformer layers (default: 6)
            CIV6AZ_EMBED_DIM: Embedding dimension (default: 64)
            CIV6AZ_NUM_HEADS: Attention heads (default: 8)
            CIV6AZ_POLICY_HIDDEN_DIM: Policy head hidden dim (default: 128)
            CIV6AZ_VALUE_HEAD_DIM: Value head hidden dim (default: 64)
            CIV6AZ_ENCODER_TYPE: Spatial encoder type (default: "patch_embed")

        Returns:
            NetworkArchitectureConfig instance.
        """
        return cls(
            board_size=int(os.environ.get("CIV6AZ_BOARD_SIZE", "8")),
            action_space_size=int(os.environ.get("CIV6AZ_ACTION_SPACE_SIZE", "376")),
            spatial_channels=int(os.environ.get("CIV6AZ_SPATIAL_CHANNELS", "28")),
            vector_dim=int(os.environ.get("CIV6AZ_VECTOR_DIM", "64")),
            num_layers=int(os.environ.get("CIV6AZ_NUM_LAYERS", "6")),
            embed_dim=int(os.environ.get("CIV6AZ_EMBED_DIM", "64")),
            num_heads=int(os.environ.get("CIV6AZ_NUM_HEADS", "8")),
            policy_hidden_dim=int(os.environ.get("CIV6AZ_POLICY_HIDDEN_DIM", "128")),
            value_head_dim=int(os.environ.get("CIV6AZ_VALUE_HEAD_DIM", "64")),
            encoder_type=os.environ.get("CIV6AZ_ENCODER_TYPE", "patch_embed"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "board_size": self.board_size,
            "action_space_size": self.action_space_size,
            "spatial_channels": self.spatial_channels,
            "vector_dim": self.vector_dim,
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "policy_hidden_dim": self.policy_hidden_dim,
            "value_head_dim": self.value_head_dim,
            "encoder_type": self.encoder_type,
        }


@dataclass
class GCPConfig:
    """Main GCP configuration for Vertex AI training.

    All values are loaded from environment variables by default,
    ensuring no hardcoded project IDs or bucket names.

    Environment Variables:
        CIV6AZ_VERTEX_PROJECT: GCP project ID
        CIV6AZ_VERTEX_REGION: GCP region (default: us-central1)
        CIV6AZ_GCS_BUCKET: GCS bucket name (without gs:// prefix)
        CIV6AZ_VERTEX_SERVICE_ACCOUNT: Service account email (optional)
        CIV6AZ_CONTAINER_IMAGE_URI: Container image URI (optional)
    """

    project_id: str = field(default_factory=lambda: os.environ.get("CIV6AZ_VERTEX_PROJECT", ""))
    region: str = field(
        default_factory=lambda: os.environ.get("CIV6AZ_VERTEX_REGION", "us-central1")
    )
    bucket: str = field(default_factory=lambda: os.environ.get("CIV6AZ_GCS_BUCKET", ""))
    service_account: str | None = field(
        default_factory=lambda: os.environ.get("CIV6AZ_VERTEX_SERVICE_ACCOUNT")
    )
    container_image_uri: str | None = field(
        default_factory=lambda: os.environ.get("CIV6AZ_CONTAINER_IMAGE_URI")
    )

    # Worker configurations
    self_play: SelfPlayWorkerConfig = field(default_factory=SelfPlayWorkerConfig)
    training: TrainingWorkerConfig = field(default_factory=TrainingWorkerConfig)

    # Path configuration
    paths: GCSPathConfig = field(default_factory=GCSPathConfig)

    # Network architecture
    network: NetworkArchitectureConfig = field(default_factory=NetworkArchitectureConfig)

    # Tier selection
    tier: VertexTier = VertexTier.BUDGET

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Normalize bucket name (remove gs:// prefix if present)
        if self.bucket.startswith("gs://"):
            self.bucket = self.bucket[5:]

    @classmethod
    def from_env(cls, validate: bool = True) -> GCPConfig:
        """Create configuration from environment variables.

        Args:
            validate: Whether to validate required fields.

        Returns:
            GCPConfig instance.

        Raises:
            ConfigurationError: If required variables are missing.
        """
        config = cls()
        if validate:
            config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> GCPConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            GCPConfig instance.

        Raises:
            ConfigurationError: If file is invalid or missing.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GCPConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            GCPConfig instance.
        """
        self_play_data = data.pop("self_play", {})
        training_data = data.pop("training", {})
        paths_data = data.pop("paths", {})
        network_data = data.pop("network", {})

        self_play = (
            SelfPlayWorkerConfig(**self_play_data) if self_play_data else SelfPlayWorkerConfig()
        )
        training = (
            TrainingWorkerConfig(**training_data) if training_data else TrainingWorkerConfig()
        )
        paths = GCSPathConfig(**paths_data) if paths_data else GCSPathConfig()
        network = (
            NetworkArchitectureConfig(**network_data)
            if network_data
            else NetworkArchitectureConfig()
        )

        tier_str = data.pop("tier", "budget")
        tier = VertexTier(tier_str) if isinstance(tier_str, str) else tier_str

        return cls(
            self_play=self_play,
            training=training,
            paths=paths,
            network=network,
            tier=tier,
            **data,
        )

    @classmethod
    def with_tier(cls, tier: VertexTier | str) -> GCPConfig:
        """Create configuration with specified tier presets.

        Args:
            tier: Pricing tier (budget or ondemand).

        Returns:
            GCPConfig with appropriate worker configs.
        """
        if isinstance(tier, str):
            tier = VertexTier(tier)

        if tier == VertexTier.BUDGET:
            return cls(
                tier=tier,
                self_play=SelfPlayWorkerConfig.budget_preset(),
                training=TrainingWorkerConfig.budget_preset(),
            )
        else:
            return cls(
                tier=tier,
                self_play=SelfPlayWorkerConfig.ondemand_preset(),
                training=TrainingWorkerConfig.ondemand_preset(),
            )

    def validate(self) -> None:
        """Validate that all required configuration is present.

        Raises:
            ConfigurationError: If required values are missing.
        """
        missing: list[str] = []

        if not self.project_id:
            missing.append("CIV6AZ_VERTEX_PROJECT")
        if not self.bucket:
            missing.append("CIV6AZ_GCS_BUCKET")

        if missing:
            raise ConfigurationError(
                "Missing required GCP configuration",
                missing_vars=missing,
            )

    @property
    def bucket_uri(self) -> str:
        """Get full GCS bucket URI."""
        return f"gs://{self.bucket}"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "project_id": self.project_id,
            "region": self.region,
            "bucket": self.bucket,
            "service_account": self.service_account,
            "container_image_uri": self.container_image_uri,
            "tier": self.tier.value,
            "self_play": {
                "machine_type": self.self_play.machine_type,
                "accelerator_type": self.self_play.accelerator_type,
                "accelerator_count": self.self_play.accelerator_count,
                "replica_count": self.self_play.replica_count,
                "spot": self.self_play.spot,
                "games_per_worker": self.self_play.games_per_worker,
                "max_retries": self.self_play.max_retries,
            },
            "training": {
                "machine_type": self.training.machine_type,
                "accelerator_type": self.training.accelerator_type,
                "accelerator_count": self.training.accelerator_count,
                "spot": self.training.spot,
                "checkpoint_interval": self.training.checkpoint_interval,
                "epochs_per_iteration": self.training.epochs_per_iteration,
            },
            "paths": {
                "models_prefix": self.paths.models_prefix,
                "trajectories_prefix": self.paths.trajectories_prefix,
                "checkpoints_prefix": self.paths.checkpoints_prefix,
                "logs_prefix": self.paths.logs_prefix,
            },
            "network": self.network.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"GCPConfig(project={self.project_id!r}, region={self.region!r}, "
            f"bucket={self.bucket!r}, tier={self.tier.value})"
        )
