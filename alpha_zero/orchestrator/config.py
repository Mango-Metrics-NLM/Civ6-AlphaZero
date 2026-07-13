"""Unified configuration system for the Training Orchestrator.

This module provides a hierarchical configuration system that unifies:
- Environment tuning
- Enhanced rewards
- Curriculum learning
- Training hyperparameters
- Distributed training
- RAG and LLM integration
- Logging and monitoring

All values can be:
- Loaded from YAML files
- Overridden via environment variables (CIV6AZ_ prefix)
- Set programmatically

Usage:
    from alpha_zero.orchestrator.config import OrchestratorConfig

    # Load from file
    config = OrchestratorConfig.from_yaml("configs/orchestrator.yaml")

    # Load from preset
    config = OrchestratorConfig.from_preset("fast")

    # Environment variable overrides
    # CIV6AZ_TRAINING_BATCH_SIZE=512 will set training.batch_size
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from alpha_zero.monitoring.config import MLflowConfig, parse_tag_string
from alpha_zero.utils.constants import ENCODER, MCTS, NETWORK, PATHS

if TYPE_CHECKING:
    from alpha_zero.spacetimedb.config import SpacetimeDBConfig


class ScheduleType(str, Enum):
    """Types of hyperparameter schedules."""

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    STEP = "step"
    WARMUP_COSINE = "warmup_cosine"
    EXPONENTIAL = "exponential"


@dataclass
class ScheduleConfig:
    """Configuration for hyperparameter schedules.

    Supports multiple schedule types for dynamic hyperparameter adjustment:
    - constant: Fixed value throughout training
    - linear: Linear interpolation from initial to final
    - cosine: Cosine annealing
    - step: Step-wise changes at specified iterations
    - warmup_cosine: Linear warmup followed by cosine decay
    - exponential: Exponential decay
    """

    schedule_type: str = "constant"
    initial_value: float = 0.001
    final_value: float | None = None
    decay_steps: int = 100000
    warmup_steps: int = 0
    step_points: list[dict[str, float]] = field(default_factory=list)

    def get_value(self, step: int) -> float:
        """Compute scheduled value at given step.

        Args:
            step: Current training step.

        Returns:
            The scheduled value at the given step.
        """
        if self.schedule_type == ScheduleType.CONSTANT.value:
            return self.initial_value

        final = self.final_value if self.final_value is not None else self.initial_value

        if self.schedule_type == ScheduleType.LINEAR.value:
            progress = min(step / max(self.decay_steps, 1), 1.0)
            return self.initial_value + progress * (final - self.initial_value)

        if self.schedule_type == ScheduleType.COSINE.value:
            progress = min(step / max(self.decay_steps, 1), 1.0)
            return final + 0.5 * (self.initial_value - final) * (1 + math.cos(math.pi * progress))

        if self.schedule_type == ScheduleType.STEP.value:
            current_value = self.initial_value
            for point in sorted(self.step_points, key=lambda x: x.get("iteration", 0)):
                if step >= point.get("iteration", 0):
                    current_value = point.get("value", current_value)
            return current_value

        if self.schedule_type == ScheduleType.WARMUP_COSINE.value:
            if step < self.warmup_steps:
                # Linear warmup
                return self.initial_value * step / max(self.warmup_steps, 1)
            # Cosine decay after warmup
            decay_length = max(self.decay_steps - self.warmup_steps, 1)
            progress = min((step - self.warmup_steps) / decay_length, 1.0)
            return final + 0.5 * (self.initial_value - final) * (1 + math.cos(math.pi * progress))

        if self.schedule_type == ScheduleType.EXPONENTIAL.value:
            if final <= 0 or self.initial_value <= 0:
                return self.initial_value
            decay_rate = (final / self.initial_value) ** (1 / max(self.decay_steps, 1))
            return float(self.initial_value * (decay_rate**step))

        return self.initial_value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduleConfig:
        """Create from dictionary."""
        return cls(
            schedule_type=data.get("schedule_type", "constant"),
            initial_value=float(data.get("initial_value", 0.001)),
            final_value=float(data["final_value"]) if "final_value" in data else None,
            decay_steps=int(data.get("decay_steps", 100000)),
            warmup_steps=int(data.get("warmup_steps", 0)),
            step_points=data.get("step_points", []),
        )


@dataclass
class ResourceBudget:
    """Resource budget constraints for autonomy control."""

    max_iterations_before_guidance: int = 5
    max_files_per_pass: int = 20
    max_lines_per_edit: int = 500
    exploration_timeout_minutes: float = 10.0
    test_timeout_seconds: int = 300
    max_memory_gb: float = 8.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceBudget:
        """Create from dictionary."""
        return cls(
            max_iterations_before_guidance=int(
                data.get("max_iterations_before_guidance", 5),
            ),
            max_files_per_pass=int(data.get("max_files_per_pass", 20)),
            max_lines_per_edit=int(data.get("max_lines_per_edit", 500)),
            exploration_timeout_minutes=float(
                data.get("exploration_timeout_minutes", 10.0),
            ),
            test_timeout_seconds=int(data.get("test_timeout_seconds", 300)),
            max_memory_gb=float(data.get("max_memory_gb", 8.0)),
        )


@dataclass
class AutonomyLevel:
    """Permission architecture for orchestrator operations."""

    # Autonomous actions (proceed without asking)
    auto_checkpoint: bool = True
    auto_log_metrics: bool = True
    auto_adjust_lr: bool = True
    auto_curriculum_transition: bool = True
    auto_restart_actors: bool = True

    # Confirm first actions
    confirm_architecture_changes: bool = True
    confirm_distributed_changes: bool = True
    confirm_large_deletions: bool = True

    # Prohibited actions
    prohibit_main_commits: bool = True
    prohibit_env_modification: bool = True
    prohibit_checkpoint_deletion: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutonomyLevel:
        """Create from dictionary."""
        return cls(
            auto_checkpoint=data.get("auto_checkpoint", True),
            auto_log_metrics=data.get("auto_log_metrics", True),
            auto_adjust_lr=data.get("auto_adjust_lr", True),
            auto_curriculum_transition=data.get("auto_curriculum_transition", True),
            auto_restart_actors=data.get("auto_restart_actors", True),
            confirm_architecture_changes=data.get("confirm_architecture_changes", True),
            confirm_distributed_changes=data.get("confirm_distributed_changes", True),
            confirm_large_deletions=data.get("confirm_large_deletions", True),
            prohibit_main_commits=data.get("prohibit_main_commits", True),
            prohibit_env_modification=data.get("prohibit_env_modification", True),
            prohibit_checkpoint_deletion=data.get("prohibit_checkpoint_deletion", True),
        )


@dataclass
class RAGConfig:
    """Configuration for RAG (Retrieval-Augmented Generation) system.

    Path defaults sourced from alpha_zero.utils.constants.PATHS.
    """

    enabled: bool = True
    persist_directory: str = PATHS.RAG_PERSIST_DIR
    collection_name: str = PATHS.RAG_COLLECTION_NAME
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    auto_ingest: bool = True
    knowledge_dir: str = PATHS.RAG_KNOWLEDGE_DIR
    store_insights: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGConfig:
        """Create from dictionary. Missing fields fall back to class defaults."""
        defaults = cls()
        return cls(
            enabled=data.get("enabled", defaults.enabled),
            persist_directory=data.get("persist_directory", defaults.persist_directory),
            collection_name=data.get("collection_name", defaults.collection_name),
            embedding_model=data.get("embedding_model", defaults.embedding_model),
            top_k=int(data.get("top_k", defaults.top_k)),
            auto_ingest=data.get("auto_ingest", defaults.auto_ingest),
            knowledge_dir=data.get("knowledge_dir", defaults.knowledge_dir),
            store_insights=data.get("store_insights", defaults.store_insights),
        )


@dataclass
class LLMConfig:
    """Configuration for LLM (Large Language Model) integration."""

    enabled: bool = True
    provider: str = "auto"  # auto | openai | anthropic
    openai_model: str = "gpt-4"
    anthropic_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7
    max_retries: int = 3
    timeout: float = 60.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMConfig:
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            provider=data.get("provider", "auto"),
            openai_model=data.get("openai_model", "gpt-4"),
            anthropic_model=data.get("anthropic_model", "claude-sonnet-4-20250514"),
            max_tokens=int(data.get("max_tokens", 4096)),
            temperature=float(data.get("temperature", 0.7)),
            max_retries=int(data.get("max_retries", 3)),
            timeout=float(data.get("timeout", 60.0)),
        )


@dataclass
class StrategyAdvisorConfig:
    """Configuration for LLM-based strategy advisor."""

    enabled: bool = True
    analysis_interval: int = 100  # Analyze every N games
    explain_moves: bool = False  # Generate move explanations (expensive)
    summarize_interval: int = 1000  # Summarize training every N iterations
    suggest_improvements: bool = True
    store_reports: bool = True
    max_context_length: int = 2000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyAdvisorConfig:
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            analysis_interval=int(data.get("analysis_interval", 100)),
            explain_moves=data.get("explain_moves", False),
            summarize_interval=int(data.get("summarize_interval", 1000)),
            suggest_improvements=data.get("suggest_improvements", True),
            store_reports=data.get("store_reports", True),
            max_context_length=int(data.get("max_context_length", 2000)),
        )


@dataclass
class ReflectionIntegrationConfig:
    """Configuration for LATS-style reflection memory integration.

    Path defaults sourced from alpha_zero.utils.constants.PATHS.
    """

    enabled: bool = False  # Disabled by default for backwards compatibility
    storage_backend: str = "memory"  # "memory", "chroma", "faiss"
    collection_name: str = PATHS.REFLECTION_COLLECTION_NAME
    embedding_dimension: int = 384
    similarity_threshold: float = 0.7
    max_retrievals: int = 5
    warning_penalty: float = 0.1  # Penalty applied to MCTS value
    max_entries: int = 10000
    persist_directory: str = PATHS.REFLECTION_PERSIST_DIR

    # LLM reflection generation settings
    generate_critiques: bool = True  # Use LLM to generate verbal critiques
    llm_temperature: float = 0.3
    llm_max_tokens: int = 500
    cache_critiques: bool = True
    use_fallback_on_failure: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReflectionIntegrationConfig:
        """Create from dictionary. Missing fields fall back to class defaults."""
        defaults = cls()
        return cls(
            enabled=data.get("enabled", defaults.enabled),
            storage_backend=data.get("storage_backend", defaults.storage_backend),
            collection_name=data.get("collection_name", defaults.collection_name),
            embedding_dimension=int(data.get("embedding_dimension", defaults.embedding_dimension)),
            similarity_threshold=float(
                data.get("similarity_threshold", defaults.similarity_threshold)
            ),
            max_retrievals=int(data.get("max_retrievals", defaults.max_retrievals)),
            warning_penalty=float(data.get("warning_penalty", defaults.warning_penalty)),
            max_entries=int(data.get("max_entries", defaults.max_entries)),
            persist_directory=data.get("persist_directory", defaults.persist_directory),
            generate_critiques=data.get("generate_critiques", defaults.generate_critiques),
            llm_temperature=float(data.get("llm_temperature", defaults.llm_temperature)),
            llm_max_tokens=int(data.get("llm_max_tokens", defaults.llm_max_tokens)),
            cache_critiques=data.get("cache_critiques", defaults.cache_critiques),
            use_fallback_on_failure=data.get(
                "use_fallback_on_failure", defaults.use_fallback_on_failure
            ),
        )

    @classmethod
    def from_env(cls) -> ReflectionIntegrationConfig:
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("CIV6AZ_REFLECTION_ENABLED", "false").lower() == "true",
            storage_backend=os.getenv("CIV6AZ_REFLECTION_BACKEND", "memory"),
            embedding_dimension=int(os.getenv("CIV6AZ_REFLECTION_EMBED_DIM", "384")),
            warning_penalty=float(os.getenv("CIV6AZ_REFLECTION_WARNING_PENALTY", "0.1")),
        )


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical planning integration."""

    enabled: bool = False  # Disabled by default for backwards compatibility

    # Strategic layer (long-term goals)
    strategic_interval: int = 50  # Re-evaluate strategy every N turns
    use_rag_for_strategy: bool = True

    # Tactical layer (medium-term objectives)
    tactical_interval: int = 10  # Re-evaluate tactics every N turns
    goal_types: list[str] = field(
        default_factory=lambda: ["expand_cities", "build_military", "research_tech"]
    )

    # Operational layer (per-turn actions)
    use_goal_guided_mcts: bool = True
    goal_weight: float = 0.3  # Weight of goal progress in value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HierarchicalConfig:
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            strategic_interval=int(data.get("strategic_interval", 50)),
            use_rag_for_strategy=data.get("use_rag_for_strategy", True),
            tactical_interval=int(data.get("tactical_interval", 10)),
            goal_types=data.get("goal_types", ["expand_cities", "build_military", "research_tech"]),
            use_goal_guided_mcts=data.get("use_goal_guided_mcts", True),
            goal_weight=float(data.get("goal_weight", 0.3)),
        )

    @classmethod
    def from_env(cls) -> HierarchicalConfig:
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("CIV6AZ_HIERARCHICAL_ENABLED", "false").lower() == "true",
            strategic_interval=int(os.getenv("CIV6AZ_STRATEGIC_INTERVAL", "50")),
            tactical_interval=int(os.getenv("CIV6AZ_TACTICAL_INTERVAL", "10")),
            goal_weight=float(os.getenv("CIV6AZ_GOAL_WEIGHT", "0.3")),
        )


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    type: str = "microciv"
    board_size: int = 16
    num_players: int = 2
    max_turns: int = 100
    fog_of_war_enabled: bool = True
    early_termination_enabled: bool = True
    early_termination_advantage_threshold: int = 3
    early_termination_min_turns: int = 10
    early_termination_stalemate_threshold: int = 20
    production_multiplier: float = 1.5
    base_production: int = 3
    #: When ``type == "chess"``, the gameplay preset routed through
    #: ``create_game_env``. Defaults to ``"default"`` to remain backwards
    #: compatible with MicroCiv configs.
    gameplay_preset: str = "default"
    #: Chess-only: number of past plies to encode (1..8). 1 → 21 channels,
    #: 8 → 119 channels per AlphaZero paper. Ignored for MicroCiv.
    chess_history_depth: int = 1
    #: Chess-only: max plies before truncation (default 512 = 256 full moves).
    chess_max_plies: int = 512
    #: Chess-only: optional FEN to start from (None → standard).
    chess_starting_fen: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentConfig:
        """Create from dictionary."""
        early_term = data.get("early_termination", {})
        chess_block = data.get("chess", {}) or {}
        return cls(
            type=data.get("type", "microciv"),
            board_size=int(data.get("board_size", 16)),
            num_players=int(data.get("num_players", 2)),
            max_turns=int(data.get("max_turns", 100)),
            fog_of_war_enabled=data.get("fog_of_war_enabled", True),
            early_termination_enabled=early_term.get("enabled", True),
            early_termination_advantage_threshold=int(
                early_term.get("advantage_threshold", 3),
            ),
            early_termination_min_turns=int(early_term.get("min_turns", 10)),
            early_termination_stalemate_threshold=int(
                early_term.get("stalemate_threshold", 20),
            ),
            production_multiplier=float(
                data.get("production", {}).get("multiplier", 1.5),
            ),
            base_production=int(data.get("production", {}).get("base_production", 3)),
            gameplay_preset=str(data.get("gameplay_preset", "default")),
            chess_history_depth=int(chess_block.get("history_depth", 1)),
            chess_max_plies=int(chess_block.get("max_plies", 512)),
            chess_starting_fen=chess_block.get("starting_fen"),
        )


@dataclass
class NetworkConfig:
    """Neural network configuration.

    Shape defaults sourced from alpha_zero.utils.constants ENCODER and
    NETWORK. num_heads and dropout intentionally diverge from
    NetworkConstants (which uses 8 and 0.1) because the orchestrator's
    default network is smaller for faster Phase 1 training.
    """

    spatial_channels: int = ENCODER.MICROCIV_CHANNELS
    vector_dim: int = ENCODER.MICROCIV_VECTOR_DIM
    num_res_blocks: int = NETWORK.DEFAULT_NUM_RES_BLOCKS
    num_filters: int = NETWORK.DEFAULT_NUM_FILTERS
    num_heads: int = 4  # orchestrator-specific, smaller than NETWORK.DEFAULT_NUM_HEADS
    value_head_hidden: int = NETWORK.DEFAULT_FC_DIM
    dropout: float = 0.0  # orchestrator-specific, no dropout for Phase 1 training
    framework: str = "pytorch"  # "pytorch" or "jax"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkConfig:
        """Create from dictionary. Missing fields fall back to class defaults."""
        defaults = cls()
        return cls(
            spatial_channels=int(data.get("spatial_channels", defaults.spatial_channels)),
            vector_dim=int(data.get("vector_dim", defaults.vector_dim)),
            num_res_blocks=int(data.get("num_res_blocks", defaults.num_res_blocks)),
            num_filters=int(data.get("num_filters", defaults.num_filters)),
            num_heads=int(data.get("num_heads", defaults.num_heads)),
            value_head_hidden=int(data.get("value_head_hidden", defaults.value_head_hidden)),
            dropout=float(data.get("dropout", defaults.dropout)),
            framework=str(data.get("framework", defaults.framework)),
        )


@dataclass
class MCTSConfig:
    """MCTS configuration for orchestrated training.

    Defaults sourced from alpha_zero.utils.constants.MCTS for auditable
    single source of truth.
    """

    num_simulations: int = MCTS.ORCHESTRATOR_NUM_SIMULATIONS
    c_puct: float = MCTS.DEFAULT_C_PUCT
    dirichlet_alpha: float = MCTS.DEFAULT_DIRICHLET_ALPHA
    dirichlet_epsilon: float = MCTS.DEFAULT_DIRICHLET_EPSILON
    temperature: float = MCTS.DEFAULT_TEMPERATURE
    temperature_threshold: int = MCTS.DEFAULT_TEMPERATURE_THRESHOLD

    def __post_init__(self) -> None:
        if not MCTS.MIN_SIMULATIONS <= self.num_simulations <= MCTS.MAX_SIMULATIONS:
            raise ValueError(
                f"num_simulations must be in [{MCTS.MIN_SIMULATIONS}, "
                f"{MCTS.MAX_SIMULATIONS}]; got {self.num_simulations}"
            )
        if self.c_puct <= 0:
            raise ValueError(f"c_puct must be > 0; got {self.c_puct}")
        if self.dirichlet_alpha <= 0:
            raise ValueError(f"dirichlet_alpha must be > 0; got {self.dirichlet_alpha}")
        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError(f"dirichlet_epsilon must be in [0, 1]; got {self.dirichlet_epsilon}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0; got {self.temperature}")
        if self.temperature_threshold < 0:
            raise ValueError(
                f"temperature_threshold must be >= 0; got {self.temperature_threshold}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCTSConfig:
        """Create from dictionary.

        Missing fields fall back to the class defaults via ``defaults = cls()``
        rather than duplicated literal fallbacks. This makes it structurally
        impossible for the class default and the from_dict fallback to drift
        apart over time (a recurring bug class on configs with many fields).
        """
        defaults = cls()
        return cls(
            num_simulations=int(data.get("num_simulations", defaults.num_simulations)),
            c_puct=float(data.get("c_puct", defaults.c_puct)),
            dirichlet_alpha=float(data.get("dirichlet_alpha", defaults.dirichlet_alpha)),
            dirichlet_epsilon=float(data.get("dirichlet_epsilon", defaults.dirichlet_epsilon)),
            temperature=float(data.get("temperature", defaults.temperature)),
            temperature_threshold=int(
                data.get("temperature_threshold", defaults.temperature_threshold)
            ),
        )


@dataclass
class TrainingConfig:
    """Training configuration."""

    num_iterations: int = 100000
    games_per_iteration: int = 200
    batch_size: int = 256
    num_epochs: int = 10
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    replay_buffer_size: int = 500000
    min_replay_size: int = 10000
    checkpoint_interval: int = 1000
    evaluation_interval: int = 500
    evaluation_games: int = 20
    # Iterations between rich EvaluationPipeline runs (vs random + heuristic).
    # The fast inline champion-vs-self eval still runs every iteration; this
    # gates the richer, more expensive opponent suite used to compute
    # value_correlation and update ELO.
    full_eval_interval: int = 5
    max_checkpoints: int = 10
    resume: bool = True  # Auto-resume from latest checkpoint if available

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Create from dictionary.

        Missing fields fall back to the class defaults via ``defaults = cls()``
        rather than duplicated literal fallbacks, so a future change to a class
        default automatically propagates to the from_dict path.
        """
        defaults = cls()
        return cls(
            num_iterations=int(data.get("num_iterations", defaults.num_iterations)),
            games_per_iteration=int(data.get("games_per_iteration", defaults.games_per_iteration)),
            batch_size=int(data.get("batch_size", defaults.batch_size)),
            num_epochs=int(data.get("num_epochs", defaults.num_epochs)),
            weight_decay=float(data.get("weight_decay", defaults.weight_decay)),
            gradient_clip=float(data.get("gradient_clip", defaults.gradient_clip)),
            replay_buffer_size=int(data.get("replay_buffer_size", defaults.replay_buffer_size)),
            min_replay_size=int(data.get("min_replay_size", defaults.min_replay_size)),
            checkpoint_interval=int(data.get("checkpoint_interval", defaults.checkpoint_interval)),
            evaluation_interval=int(data.get("evaluation_interval", defaults.evaluation_interval)),
            evaluation_games=int(data.get("evaluation_games", defaults.evaluation_games)),
            full_eval_interval=int(data.get("full_eval_interval", defaults.full_eval_interval)),
            max_checkpoints=int(data.get("max_checkpoints", defaults.max_checkpoints)),
            resume=data.get("resume", defaults.resume),
        )


@dataclass
class RewardsConfig:
    """Enhanced rewards configuration."""

    enabled: bool = True
    scale_factor: float = 1.0
    gamma: float = 0.99
    normalize: bool = True
    territory_control: float = 0.02
    unit_preservation: float = 0.01
    city_proximity: float = 0.005
    exploration: float = 0.001
    combat_engagement: float = 0.01
    city_founding: float = 0.1
    unit_advantage_weight: float = 0.1
    city_advantage_weight: float = 0.2
    health_advantage_weight: float = 0.05
    territory_advantage_weight: float = 0.05

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardsConfig:
        """Create from dictionary."""
        intermediate = data.get("intermediate", {})
        potential = data.get("potential_weights", {})
        return cls(
            enabled=data.get("enabled", True),
            scale_factor=float(data.get("scale_factor", 1.0)),
            gamma=float(data.get("gamma", 0.99)),
            normalize=data.get("normalize", True),
            territory_control=float(intermediate.get("territory_control", 0.02)),
            unit_preservation=float(intermediate.get("unit_preservation", 0.01)),
            city_proximity=float(intermediate.get("city_proximity", 0.005)),
            exploration=float(intermediate.get("exploration", 0.001)),
            combat_engagement=float(intermediate.get("combat_engagement", 0.01)),
            city_founding=float(intermediate.get("city_founding", 0.1)),
            unit_advantage_weight=float(potential.get("unit_advantage", 0.1)),
            city_advantage_weight=float(potential.get("city_advantage", 0.2)),
            health_advantage_weight=float(potential.get("health_advantage", 0.05)),
            territory_advantage_weight=float(potential.get("territory_advantage", 0.05)),
        )


@dataclass
class CurriculumStageConfig:
    """Configuration for a single curriculum stage."""

    name: str
    level: int
    board_size: int
    max_turns: int
    promotion_threshold: float
    min_games: int = 50

    @classmethod
    def from_dict(cls, data: dict[str, Any], level: int = 0) -> CurriculumStageConfig:
        """Create from dictionary."""
        return cls(
            name=data.get("name", f"stage_{level}"),
            level=data.get("level", level),
            board_size=int(data.get("board_size", 8 + level * 4)),
            max_turns=int(data.get("max_turns", 30 + level * 30)),
            promotion_threshold=float(data.get("promotion_threshold", 0.7 - level * 0.05)),
            min_games=int(data.get("min_games", 50)),
        )


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""

    enabled: bool = True
    allow_demotion: bool = True
    metrics_window: int = 100
    stages: list[CurriculumStageConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize default stages if none provided."""
        if not self.stages:
            self.stages = [
                CurriculumStageConfig(
                    name="beginner",
                    level=0,
                    board_size=8,
                    max_turns=30,
                    promotion_threshold=0.7,
                    min_games=50,
                ),
                CurriculumStageConfig(
                    name="intermediate",
                    level=1,
                    board_size=12,
                    max_turns=60,
                    promotion_threshold=0.6,
                    min_games=75,
                ),
                CurriculumStageConfig(
                    name="advanced",
                    level=2,
                    board_size=16,
                    max_turns=100,
                    promotion_threshold=0.55,
                    min_games=100,
                ),
            ]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumConfig:
        """Create from dictionary."""
        stages_data = data.get("stages", [])
        stages = [CurriculumStageConfig.from_dict(s, i) for i, s in enumerate(stages_data)]
        return cls(
            enabled=data.get("enabled", True),
            allow_demotion=data.get("allow_demotion", True),
            metrics_window=int(data.get("metrics_window", 100)),
            stages=stages if stages else [],
        )


@dataclass
class WandbConfig:
    """Legacy Weights & Biases configuration (deprecated bridge).

    W&B is retired; MLflow is the experiment-tracking system of record
    (PROJECT_CHARTER.md NG-5). This dataclass survives so existing YAML
    ``wandb:`` sections, presets, ``--no-wandb``, and ``CIV6AZ_WANDB_*``
    overrides keep parsing. At runtime the orchestrator converts it via
    :meth:`to_mlflow_config` and drives the MLflow backend.
    """

    enabled: bool = False
    project: str = "civ6-alphazero"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)
    log_model: bool = True
    log_code: bool = True
    mode: str = "online"  # online | offline | disabled

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WandbConfig:
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            project=data.get("project", "civ6-alphazero"),
            entity=data.get("entity"),
            tags=data.get("tags", []),
            log_model=data.get("log_model", True),
            log_code=data.get("log_code", True),
            mode=data.get("mode", "online"),
        )

    def to_mlflow_config(self) -> Any:
        """Map this legacy W&B config onto the canonical ``MLflowConfig``.

        Mapping: ``project`` -> ``experiment_name``; list ``tags`` -> the
        MLflow dict form ``{tag: "true"}``; ``entity`` is preserved as the
        ``wandb.entity`` tag rather than dropped; ``mode == "disabled"``
        forces ``enabled=False``; ``log_model`` -> ``log_models``.

        Returns:
            alpha_zero.monitoring.config.MLflowConfig
        """
        from alpha_zero.monitoring.config import MLflowConfig

        tags: dict[str, str] = dict.fromkeys(self.tags or [], "true")
        if self.entity:
            tags["wandb.entity"] = self.entity
        return MLflowConfig(
            enabled=self.enabled and self.mode != "disabled",
            experiment_name=self.project,
            tags=tags,
            log_models=self.log_model,
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    json_format: bool = False
    tensorboard: bool = True
    tensorboard_dir: str = "logs/tensorboard"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoggingConfig:
        """Create from dictionary."""
        return cls(
            level=data.get("level", "INFO"),
            console_output=data.get("console_output", True),
            file_output=data.get("file_output", True),
            json_format=data.get("json_format", False),
            tensorboard=data.get("tensorboard", True),
            tensorboard_dir=data.get("tensorboard_dir", "logs/tensorboard"),
        )


@dataclass
class StagnationConfig:
    """Training-stagnation detection settings (opt-in).

    Wraps the standalone ``EloPlateauDetector`` (and optionally
    ``ConvergenceMonitor``). Detector thresholds themselves come from their own
    ``CIV6AZ_ELO_*`` / ``CIV6AZ_CONVERGENCE_*`` env vars; this config only governs
    whether stagnation detection runs and how the orchestrator reacts.

    Attributes:
        enabled: Master switch (default off — zero behaviour change).
        use_elo_plateau: Use the ELO/win-rate plateau detector (recommended).
        use_convergence: Use the loss/win-rate convergence monitor. Off by
            default: a loss signal can fall while the champion never improves
            (the v1.2.0 "0 promotions" trap), so it is not a safe default.
        auto_stop: Request a graceful shutdown on plateau (else alert only).
        alert_to_mlflow: Emit ``stagnation/*`` metrics + a warning tag to MLflow.
    """

    enabled: bool = False
    use_elo_plateau: bool = True
    use_convergence: bool = False
    auto_stop: bool = False
    alert_to_mlflow: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StagnationConfig:
        """Create from dictionary."""
        defaults = cls()
        return cls(
            enabled=bool(data.get("enabled", defaults.enabled)),
            use_elo_plateau=bool(data.get("use_elo_plateau", defaults.use_elo_plateau)),
            use_convergence=bool(data.get("use_convergence", defaults.use_convergence)),
            auto_stop=bool(data.get("auto_stop", defaults.auto_stop)),
            alert_to_mlflow=bool(data.get("alert_to_mlflow", defaults.alert_to_mlflow)),
        )


@dataclass
class OrchestratorConfig:
    """Master configuration for the Training Orchestrator.

    Unifies all component configurations with:
    - YAML file loading
    - Environment variable overrides (prefix: CIV6AZ_)
    - Validation and defaults
    - Schedule support for all hyperparameters
    """

    # Mode selection
    mode: str = "local"  # local | distributed | evaluation

    # Experiment metadata
    experiment_name: str = "civ6_alphazero_v1"
    experiment_tags: list[str] = field(default_factory=list)
    random_seed: int | None = None

    # Sub-configurations
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    strategy_advisor: StrategyAdvisorConfig = field(
        default_factory=StrategyAdvisorConfig,
    )

    # Reflection memory (LATS-style learning from failures)
    reflection: ReflectionIntegrationConfig = field(
        default_factory=ReflectionIntegrationConfig,
    )

    # Hierarchical planning (strategic/tactical/operational layers)
    hierarchical: HierarchicalConfig = field(
        default_factory=HierarchicalConfig,
    )

    # Training-stagnation detection (opt-in)
    stagnation: StagnationConfig = field(default_factory=StagnationConfig)

    # Resource management
    resources: ResourceBudget = field(default_factory=ResourceBudget)
    autonomy: AutonomyLevel = field(default_factory=AutonomyLevel)

    # Schedules for dynamic hyperparameters
    lr_schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    batch_size_schedule: ScheduleConfig = field(
        default_factory=lambda: ScheduleConfig(initial_value=64),
    )
    temperature_schedule: ScheduleConfig = field(
        default_factory=lambda: ScheduleConfig(initial_value=1.0),
    )
    exploration_schedule: ScheduleConfig = field(
        default_factory=lambda: ScheduleConfig(initial_value=0.25),
    )

    # SpacetimeDB (optional real-time distributed coordination)
    spacetimedb: SpacetimeDBConfig | None = field(default=None)

    # Paths (defaults sourced from alpha_zero.utils.constants.PATHS)
    checkpoint_dir: str = PATHS.ORCHESTRATOR_CHECKPOINT_DIR
    log_dir: str = PATHS.ORCHESTRATOR_LOG_DIR
    config_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> OrchestratorConfig:
        """Load configuration from YAML file with env overrides.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Loaded and validated configuration.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = cls._from_dict(data)
        config.config_path = str(path)
        config.apply_env_overrides()
        return config

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> OrchestratorConfig:
        """Create from dictionary."""
        return cls(
            mode=data.get("mode", "local"),
            experiment_name=data.get("experiment_name", "civ6_alphazero_v1"),
            experiment_tags=data.get("experiment_tags", []),
            random_seed=data.get("random_seed"),
            environment=EnvironmentConfig.from_dict(data.get("environment", {})),
            network=NetworkConfig.from_dict(data.get("network", {})),
            mcts=MCTSConfig.from_dict(data.get("mcts", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
            rewards=RewardsConfig.from_dict(data.get("rewards", {})),
            curriculum=CurriculumConfig.from_dict(data.get("curriculum", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
            wandb=WandbConfig.from_dict(data.get("wandb", {})),
            mlflow=MLflowConfig.from_dict(data.get("mlflow", {})),
            rag=RAGConfig.from_dict(data.get("rag", {})),
            llm=LLMConfig.from_dict(data.get("llm", {})),
            strategy_advisor=StrategyAdvisorConfig.from_dict(
                data.get("strategy_advisor", {}),
            ),
            reflection=ReflectionIntegrationConfig.from_dict(
                data.get("reflection", {}),
            ),
            hierarchical=HierarchicalConfig.from_dict(
                data.get("hierarchical", {}),
            ),
            stagnation=StagnationConfig.from_dict(data.get("stagnation", {})),
            resources=ResourceBudget.from_dict(data.get("resources", {})),
            autonomy=AutonomyLevel.from_dict(data.get("autonomy", {})),
            lr_schedule=ScheduleConfig.from_dict(data.get("lr_schedule", {})),
            batch_size_schedule=ScheduleConfig.from_dict(
                data.get("batch_size_schedule", {"initial_value": 64}),
            ),
            temperature_schedule=ScheduleConfig.from_dict(
                data.get("temperature_schedule", {"initial_value": 1.0}),
            ),
            exploration_schedule=ScheduleConfig.from_dict(
                data.get("exploration_schedule", {"initial_value": 0.25}),
            ),
            spacetimedb=cls._parse_spacetimedb_config(data.get("spacetimedb")),
            checkpoint_dir=data.get("checkpoint_dir", PATHS.ORCHESTRATOR_CHECKPOINT_DIR),
            log_dir=data.get("log_dir", PATHS.ORCHESTRATOR_LOG_DIR),
        )

    @staticmethod
    def _parse_spacetimedb_config(data: dict[str, Any] | None) -> SpacetimeDBConfig | None:
        """Parse SpacetimeDB config from dict if present.

        Args:
            data: SpacetimeDB configuration dictionary.

        Returns:
            SpacetimeDBConfig or None.
        """
        if data is None:
            return None
        try:
            from alpha_zero.spacetimedb.config import SpacetimeDBConfig

            return SpacetimeDBConfig.from_dict(data)
        except ImportError:
            return None

    @classmethod
    def from_preset(cls, preset: str) -> OrchestratorConfig:
        """Load from predefined preset.

        Args:
            preset: One of "fast", "balanced", "production".

        Returns:
            Configuration for the specified preset.
        """
        presets = {
            "fast": cls._fast_preset(),
            "balanced": cls._balanced_preset(),
            "production": cls._production_preset(),
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")
        config = presets[preset]
        config.apply_env_overrides()
        return config

    @classmethod
    def _fast_preset(cls) -> OrchestratorConfig:
        """Fast preset for quick testing."""
        config = cls()
        config.experiment_name = "fast_test"
        # Environment settings for fast testing
        config.environment.board_size = 6
        config.environment.max_turns = 20
        # Network settings (smaller for speed)
        config.network.num_res_blocks = 4
        config.network.num_filters = 32
        # Training settings
        config.training.num_iterations = 10
        config.training.games_per_iteration = 5
        config.training.batch_size = 32
        config.training.num_epochs = 2
        config.training.min_replay_size = 100
        config.mcts.num_simulations = 10
        config.curriculum.stages = [
            CurriculumStageConfig(
                name="fast_test",
                level=0,
                board_size=6,
                max_turns=20,
                promotion_threshold=0.5,
                min_games=5,
            ),
        ]
        config.rag.enabled = False
        config.llm.enabled = False
        config.strategy_advisor.enabled = False
        config.wandb.enabled = False
        config.logging.tensorboard = False
        return config

    @classmethod
    def _balanced_preset(cls) -> OrchestratorConfig:
        """Balanced preset for initial experiments."""
        config = cls()
        config.experiment_name = "balanced_training"
        config.training.num_iterations = 500
        config.training.games_per_iteration = 50
        config.training.batch_size = 128
        config.training.num_epochs = 5
        config.mcts.num_simulations = 50
        config.strategy_advisor.analysis_interval = 50
        return config

    @classmethod
    def _production_preset(cls) -> OrchestratorConfig:
        """Production preset for full training."""
        config = cls()
        config.experiment_name = "production_training"
        config.training.num_iterations = 500000
        config.training.games_per_iteration = 200
        config.training.batch_size = 512
        config.training.num_epochs = 10
        config.mcts.num_simulations = 200
        config.network.num_res_blocks = 10
        config.network.num_filters = 128
        config.lr_schedule = ScheduleConfig(
            schedule_type="warmup_cosine",
            initial_value=0.001,
            final_value=0.00001,
            decay_steps=500000,
            warmup_steps=10000,
        )
        config.wandb.enabled = True
        return config

    def apply_env_overrides(self) -> OrchestratorConfig:
        """Apply environment variable overrides with CIV6AZ_ prefix.

        Environment variables follow the pattern:
        CIV6AZ_<SECTION>_<KEY>=value

        Examples:
        - CIV6AZ_TRAINING_BATCH_SIZE=512
        - CIV6AZ_MCTS_NUM_SIMULATIONS=200
        - CIV6AZ_WANDB_ENABLED=true  (deprecated; prefer CIV6AZ_MLFLOW_ENABLED)

        Returns:
            Self for method chaining.
        """
        prefix = "CIV6AZ_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            parts = key[len(prefix) :].lower().split("_", 1)
            if len(parts) != 2:
                continue

            section, attr = parts
            self._apply_override(section, attr, value)

        return self

    def _apply_override(self, section: str, attr: str, value: str) -> None:
        """Apply a single environment override."""
        section_map = {
            "training": self.training,
            "mcts": self.mcts,
            "network": self.network,
            "environment": self.environment,
            "rewards": self.rewards,
            "logging": self.logging,
            "wandb": self.wandb,
            "mlflow": self.mlflow,
            "rag": self.rag,
            "llm": self.llm,
            "resources": self.resources,
            "reflection": self.reflection,
            "hierarchical": self.hierarchical,
            "stagnation": self.stagnation,
        }

        if section not in section_map:
            return

        target = section_map[section]
        if not hasattr(target, attr):
            return

        current_value = getattr(target, attr)
        try:
            if isinstance(current_value, bool):
                setattr(target, attr, value.lower() in ("true", "1", "yes"))
            elif isinstance(current_value, int):
                setattr(target, attr, int(value))
            elif isinstance(current_value, float):
                setattr(target, attr, float(value))
            elif isinstance(current_value, dict):
                # Parse "k1=v1,k2=v2" form used by env-var overrides such as
                # CIV6AZ_MLFLOW_TAGS. Without this, a raw string would clobber
                # the dict field.
                setattr(target, attr, parse_tag_string(value))
            else:
                setattr(target, attr, value)
        except (ValueError, TypeError):
            pass  # Skip invalid overrides

    def validate(self) -> list[str]:
        """Validate configuration, return list of warnings.

        Returns:
            List of warning messages for potentially problematic settings.
        """
        warnings = []

        # Check replay buffer size
        if self.training.replay_buffer_size < self.training.batch_size * 10:
            warnings.append(
                f"Replay buffer ({self.training.replay_buffer_size}) should be "
                f"at least 10x batch size ({self.training.batch_size})",
            )

        # Check curriculum stages
        if self.curriculum.enabled and not self.curriculum.stages:
            warnings.append("Curriculum enabled but no stages defined")

        # Check MCTS simulations
        if self.mcts.num_simulations < 10:
            warnings.append(
                f"MCTS simulations ({self.mcts.num_simulations}) is very low",
            )

        # Check RAG dependencies
        if self.rag.enabled and not self.strategy_advisor.enabled:
            warnings.append("RAG enabled but strategy_advisor disabled")

        # Check distributed mode
        if self.mode == "distributed":
            if not self.autonomy.auto_restart_actors:
                warnings.append(
                    "Distributed mode without auto_restart_actors may cause issues",
                )

        return warnings

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "mode": self.mode,
            "experiment_name": self.experiment_name,
            "experiment_tags": self.experiment_tags,
            "random_seed": self.random_seed,
            "environment": {
                "type": self.environment.type,
                "board_size": self.environment.board_size,
                "num_players": self.environment.num_players,
                "max_turns": self.environment.max_turns,
                "fog_of_war_enabled": self.environment.fog_of_war_enabled,
                "early_termination": {
                    "enabled": self.environment.early_termination_enabled,
                    "advantage_threshold": self.environment.early_termination_advantage_threshold,
                    "min_turns": self.environment.early_termination_min_turns,
                },
            },
            "network": {
                "spatial_channels": self.network.spatial_channels,
                "num_res_blocks": self.network.num_res_blocks,
                "num_filters": self.network.num_filters,
                "num_heads": self.network.num_heads,
            },
            "mcts": {
                "num_simulations": self.mcts.num_simulations,
                "c_puct": self.mcts.c_puct,
                "dirichlet_alpha": self.mcts.dirichlet_alpha,
                "dirichlet_epsilon": self.mcts.dirichlet_epsilon,
            },
            "training": {
                "num_iterations": self.training.num_iterations,
                "games_per_iteration": self.training.games_per_iteration,
                "batch_size": self.training.batch_size,
                "num_epochs": self.training.num_epochs,
                "replay_buffer_size": self.training.replay_buffer_size,
            },
            "rewards": {
                "enabled": self.rewards.enabled,
                "scale_factor": self.rewards.scale_factor,
            },
            "curriculum": {
                "enabled": self.curriculum.enabled,
                "stages": [
                    {
                        "name": s.name,
                        "board_size": s.board_size,
                        "max_turns": s.max_turns,
                        "promotion_threshold": s.promotion_threshold,
                    }
                    for s in self.curriculum.stages
                ],
            },
            "wandb": {
                "enabled": self.wandb.enabled,
                "project": self.wandb.project,
            },
            "rag": {
                "enabled": self.rag.enabled,
                "persist_directory": self.rag.persist_directory,
            },
            "llm": {
                "enabled": self.llm.enabled,
                "provider": self.llm.provider,
            },
            "strategy_advisor": {
                "enabled": self.strategy_advisor.enabled,
                "analysis_interval": self.strategy_advisor.analysis_interval,
            },
            "reflection": {
                "enabled": self.reflection.enabled,
                "storage_backend": self.reflection.storage_backend,
                "warning_penalty": self.reflection.warning_penalty,
                "generate_critiques": self.reflection.generate_critiques,
            },
            "hierarchical": {
                "enabled": self.hierarchical.enabled,
                "strategic_interval": self.hierarchical.strategic_interval,
                "tactical_interval": self.hierarchical.tactical_interval,
                "goal_weight": self.hierarchical.goal_weight,
            },
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save the YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
