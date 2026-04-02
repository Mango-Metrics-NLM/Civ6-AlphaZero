"""
Goal Encoders for Hierarchical Reasoning.

This module provides encoders that transform goals into tensor representations
suitable for conditioning neural network decisions at different hierarchy levels.

Example:
    >>> from alpha_zero.hierarchical.encoders import GoalEncoder, HierarchicalGoalEncoder
    >>> encoder = GoalEncoder(embed_dim=64)
    >>> goal_tensor = encoder.encode(goal, game_state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from alpha_zero.hierarchical.goals.goal_types import (
    Goal,
    GoalStatus,
    GoalTarget,
    HierarchyLevel,
    OperationalGoalType,
    StrategicGoalType,
    TacticalGoalType,
)

logger = logging.getLogger(__name__)


@dataclass
class GoalEmbedding:
    """Container for goal embedding with metadata.

    Attributes:
        embedding: The goal embedding vector.
        goal_id: ID of the encoded goal.
        level: Hierarchy level of the goal.
        priority: Goal priority at encoding time.
    """

    embedding: np.ndarray
    goal_id: str
    level: HierarchyLevel
    priority: float = 1.0


@dataclass
class GoalEncoderConfig:
    """Configuration for goal encoders.

    Attributes:
        embed_dim: Output embedding dimension.
        include_context: Whether to include goal context features.
        include_progress: Whether to include progress information.
        max_goals: Maximum number of goals to encode simultaneously.
    """

    embed_dim: int = 64
    include_context: bool = True
    include_progress: bool = True
    max_goals: int = 10

    @classmethod
    def for_level(cls, level: HierarchyLevel) -> GoalEncoderConfig:
        """Create level-specific configuration.

        Args:
            level: Hierarchy level.

        Returns:
            GoalEncoderConfig with appropriate settings.
        """
        configs = {
            HierarchyLevel.STRATEGIC: cls(embed_dim=128, max_goals=5),
            HierarchyLevel.TACTICAL: cls(embed_dim=64, max_goals=10),
            HierarchyLevel.OPERATIONAL: cls(embed_dim=32, max_goals=15),
        }
        return configs.get(level, cls())


class GoalEncoder:
    """Encoder that transforms goals into vector representations.

    This encoder creates fixed-size embeddings from goals that can be used
    to condition neural network decisions. It handles:
    - Goal type encoding (one-hot or learned)
    - Target location/entity encoding
    - Progress and status encoding
    - Context from parent goals
    """

    # Goal type vocabulary sizes
    NUM_STRATEGIC_TYPES = len(StrategicGoalType)
    NUM_TACTICAL_TYPES = len(TacticalGoalType)
    NUM_OPERATIONAL_TYPES = len(OperationalGoalType)
    NUM_STATUSES = len(GoalStatus)

    def __init__(
        self,
        config: GoalEncoderConfig | None = None,
        board_size: tuple[int, int] = (16, 16),
    ) -> None:
        """Initialize the goal encoder.

        Args:
            config: Encoder configuration.
            board_size: Game board size for location normalization.
        """
        self.config = config or GoalEncoderConfig()
        self.board_size = board_size

        # Calculate feature dimensions
        self._type_dim = max(
            self.NUM_STRATEGIC_TYPES,
            self.NUM_TACTICAL_TYPES,
            self.NUM_OPERATIONAL_TYPES,
        )
        self._status_dim = self.NUM_STATUSES
        self._level_dim = len(HierarchyLevel)
        self._target_dim = 8  # Location (2) + entity features (6)
        self._progress_dim = 4  # Progress, deadline, creation, last_update
        self._context_dim = 16  # Parent goal context

        self._raw_features = (
            self._type_dim
            + self._status_dim
            + self._level_dim
            + self._target_dim
            + self._progress_dim
            + (self._context_dim if self.config.include_context else 0)
        )

        logger.debug(
            "GoalEncoder initialized: raw_features=%d, embed_dim=%d",
            self._raw_features,
            self.config.embed_dim,
        )

    @property
    def output_dim(self) -> int:
        """Get output embedding dimension."""
        return self.config.embed_dim

    def encode(
        self,
        goal: Goal,
        game_state: dict[str, Any] | None = None,
        parent_goal: Goal | None = None,
    ) -> GoalEmbedding:
        """Encode a single goal into a vector representation.

        Args:
            goal: Goal to encode.
            game_state: Optional game state for context.
            parent_goal: Optional parent goal for context.

        Returns:
            GoalEmbedding with the encoded goal.
        """
        features = []

        # Goal type (one-hot)
        type_vec = self._encode_goal_type(goal)
        features.append(type_vec)

        # Goal status (one-hot)
        status_vec = np.zeros(self._status_dim, dtype=np.float32)
        status_vec[goal.status.value - 1] = 1.0  # Status enum is 1-indexed
        features.append(status_vec)

        # Hierarchy level (one-hot)
        level_vec = np.zeros(self._level_dim, dtype=np.float32)
        level_idx = list(HierarchyLevel).index(goal.level)
        level_vec[level_idx] = 1.0
        features.append(level_vec)

        # Target encoding
        target_vec = self._encode_target(goal.target, game_state)
        features.append(target_vec)

        # Progress information
        if self.config.include_progress:
            progress_vec = self._encode_progress(goal, game_state)
            features.append(progress_vec)

        # Parent goal context
        if self.config.include_context and parent_goal:
            context_vec = self._encode_context(parent_goal)
            features.append(context_vec)
        elif self.config.include_context:
            features.append(np.zeros(self._context_dim, dtype=np.float32))

        # Concatenate all features
        combined = np.concatenate(features)

        # Project or pad to embed_dim
        embedding = self._project_to_embedding(combined)

        return GoalEmbedding(
            embedding=embedding,
            goal_id=goal.id,
            level=goal.level,
            priority=goal.priority,
        )

    def encode_batch(
        self,
        goals: list[Goal],
        game_state: dict[str, Any] | None = None,
        parent_goals: dict[str, Goal] | None = None,
    ) -> np.ndarray:
        """Encode multiple goals into a batch tensor.

        Args:
            goals: List of goals to encode.
            game_state: Optional game state for context.
            parent_goals: Optional mapping of goal_id -> parent goal.

        Returns:
            Array of shape (max_goals, embed_dim) with encoded goals.
        """
        parent_goals = parent_goals or {}
        batch = np.zeros(
            (self.config.max_goals, self.config.embed_dim),
            dtype=np.float32,
        )

        for i, goal in enumerate(goals[: self.config.max_goals]):
            parent = parent_goals.get(goal.parent_goal_id) if goal.parent_goal_id else None
            embedding = self.encode(goal, game_state, parent)
            batch[i] = embedding.embedding

        return batch

    def _encode_goal_type(self, goal: Goal) -> np.ndarray:
        """Encode goal type as one-hot vector."""
        type_vec = np.zeros(self._type_dim, dtype=np.float32)

        if goal.level == HierarchyLevel.STRATEGIC:
            if isinstance(goal.goal_type, StrategicGoalType):
                idx = list(StrategicGoalType).index(goal.goal_type)
                type_vec[idx] = 1.0
        elif goal.level == HierarchyLevel.TACTICAL:
            if isinstance(goal.goal_type, TacticalGoalType):
                idx = list(TacticalGoalType).index(goal.goal_type)
                type_vec[idx] = 1.0
        elif goal.level == HierarchyLevel.OPERATIONAL:
            if isinstance(goal.goal_type, OperationalGoalType):
                idx = list(OperationalGoalType).index(goal.goal_type)
                type_vec[idx] = 1.0

        return type_vec

    def _encode_target(
        self,
        target: GoalTarget,
        game_state: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Encode goal target information."""
        target_vec = np.zeros(self._target_dim, dtype=np.float32)

        # Location (normalized)
        if target.position:
            target_vec[0] = target.position[0] / self.board_size[0]
            target_vec[1] = target.position[1] / self.board_size[1]
        else:
            target_vec[0] = 0.5  # Default to center
            target_vec[1] = 0.5

        # Quantity (normalized)
        target_vec[2] = min(target.quantity / 10.0, 1.0) if target.quantity else 0.0

        # Entity ID hash (for consistent encoding)
        if target.entity_id:
            # Use hash for consistent encoding across different entity IDs
            hash_val = hash(target.entity_id) % 1000 / 1000.0
            target_vec[3] = hash_val

        # Target type encoding (simple hash)
        if target.target_type:
            type_hash = hash(target.target_type) % 1000 / 1000.0
            target_vec[4] = type_hash

        # Distance to target (if location known and game state available)
        if target.position and game_state:
            own_cities = [
                c
                for c in game_state.get("cities", [])
                if c.get("player") == game_state.get("current_player", 0)
            ]
            if own_cities:
                # Distance from nearest own city
                min_dist = min(
                    abs(c.get("x", 0) - target.position[0])
                    + abs(c.get("y", 0) - target.position[1])
                    for c in own_cities
                )
                target_vec[5] = min(min_dist / 20.0, 1.0)

        # Visibility (if target location is visible)
        if target.position and game_state:
            visibility = game_state.get("visibility", {})
            x, y = target.position
            target_vec[6] = 1.0 if visibility.get(f"{x},{y}", False) else 0.0

        # Threat level at target
        if target.position and game_state:
            enemies = [
                u
                for u in game_state.get("units", [])
                if u.get("player") != game_state.get("current_player", 0)
            ]
            nearby_enemies = sum(
                1
                for e in enemies
                if abs(e.get("x", 0) - target.position[0]) <= 3
                and abs(e.get("y", 0) - target.position[1]) <= 3
            )
            target_vec[7] = min(nearby_enemies / 5.0, 1.0)

        return target_vec

    def _encode_progress(
        self,
        goal: Goal,
        game_state: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Encode goal progress information."""
        current_turn = game_state.get("turn", 0) if game_state else 0

        progress_vec = np.array(
            [
                goal.progress,  # Current progress (0-1)
                # Turns since creation (normalized)
                min((current_turn - goal.created_turn) / 100.0, 1.0),
                # Turns until deadline (normalized, 1.0 if no deadline)
                (
                    min((goal.deadline_turn - current_turn) / 50.0, 1.0)
                    if goal.deadline_turn
                    else 1.0
                ),
                # Turns since last progress (normalized)
                min((current_turn - goal.last_progress_turn) / 20.0, 1.0),
            ],
            dtype=np.float32,
        )

        return progress_vec

    def _encode_context(self, parent_goal: Goal) -> np.ndarray:
        """Encode parent goal context."""
        context_vec = np.zeros(self._context_dim, dtype=np.float32)

        # Parent type encoding
        type_vec = self._encode_goal_type(parent_goal)
        context_vec[: min(len(type_vec), 8)] = type_vec[:8]

        # Parent progress
        context_vec[8] = parent_goal.progress

        # Parent priority
        context_vec[9] = min(parent_goal.priority, 1.0)

        # Parent status
        context_vec[10] = (parent_goal.status.value - 1) / (len(GoalStatus) - 1)

        # Number of siblings (children of parent)
        context_vec[11] = min(len(parent_goal.child_goal_ids) / 10.0, 1.0)

        # Parent level
        level_idx = list(HierarchyLevel).index(parent_goal.level)
        context_vec[12] = level_idx / (len(HierarchyLevel) - 1)

        # Parent urgency (based on deadline)
        if parent_goal.deadline_turn and parent_goal.created_turn:
            total_time = parent_goal.deadline_turn - parent_goal.created_turn
            if total_time > 0:
                context_vec[13] = 1.0 - parent_goal.progress  # Remaining work
                context_vec[14] = min(total_time / 100.0, 1.0)

        # Parent has reward modifier
        context_vec[15] = parent_goal.reward_modifier

        return context_vec

    def _project_to_embedding(self, features: np.ndarray) -> np.ndarray:
        """Project raw features to embedding dimension."""
        if len(features) >= self.config.embed_dim:
            # Truncate if too large
            return features[: self.config.embed_dim].astype(np.float32)
        else:
            # Pad if too small
            embedding = np.zeros(self.config.embed_dim, dtype=np.float32)
            embedding[: len(features)] = features
            return embedding


class HierarchicalGoalEncoder:
    """Encoder that manages goal encoding across all hierarchy levels.

    This class provides a unified interface for encoding goals at different
    levels with appropriate level-specific configurations.
    """

    def __init__(
        self,
        board_size: tuple[int, int] = (16, 16),
        strategic_config: GoalEncoderConfig | None = None,
        tactical_config: GoalEncoderConfig | None = None,
        operational_config: GoalEncoderConfig | None = None,
    ) -> None:
        """Initialize hierarchical goal encoder.

        Args:
            board_size: Game board size.
            strategic_config: Config for strategic level encoding.
            tactical_config: Config for tactical level encoding.
            operational_config: Config for operational level encoding.
        """
        self.board_size = board_size

        self.strategic_encoder = GoalEncoder(
            strategic_config or GoalEncoderConfig.for_level(HierarchyLevel.STRATEGIC),
            board_size,
        )
        self.tactical_encoder = GoalEncoder(
            tactical_config or GoalEncoderConfig.for_level(HierarchyLevel.TACTICAL),
            board_size,
        )
        self.operational_encoder = GoalEncoder(
            operational_config or GoalEncoderConfig.for_level(HierarchyLevel.OPERATIONAL),
            board_size,
        )

        self._encoders = {
            HierarchyLevel.STRATEGIC: self.strategic_encoder,
            HierarchyLevel.TACTICAL: self.tactical_encoder,
            HierarchyLevel.OPERATIONAL: self.operational_encoder,
        }

        logger.info(
            "HierarchicalGoalEncoder initialized with dims: "
            "strategic=%d, tactical=%d, operational=%d",
            self.strategic_encoder.output_dim,
            self.tactical_encoder.output_dim,
            self.operational_encoder.output_dim,
        )

    def encode(
        self,
        goal: Goal,
        game_state: dict[str, Any] | None = None,
        parent_goal: Goal | None = None,
    ) -> GoalEmbedding:
        """Encode a goal using the appropriate level encoder.

        Args:
            goal: Goal to encode.
            game_state: Optional game state for context.
            parent_goal: Optional parent goal.

        Returns:
            GoalEmbedding for the goal.
        """
        encoder = self._encoders.get(goal.level, self.operational_encoder)
        return encoder.encode(goal, game_state, parent_goal)

    def encode_goals_for_level(
        self,
        goals: list[Goal],
        level: HierarchyLevel,
        game_state: dict[str, Any] | None = None,
        goal_lookup: dict[str, Goal] | None = None,
    ) -> np.ndarray:
        """Encode goals for a specific hierarchy level.

        Args:
            goals: List of goals to encode.
            level: Target hierarchy level.
            game_state: Optional game state.
            goal_lookup: Optional mapping of goal_id -> Goal for parent lookup.

        Returns:
            Batch array of encoded goals.
        """
        # Filter goals for the specified level
        level_goals = [g for g in goals if g.level == level]

        # Build parent goal mapping
        parent_goals = {}
        if goal_lookup:
            for goal in level_goals:
                if goal.parent_goal_id and goal.parent_goal_id in goal_lookup:
                    parent_goals[goal.parent_goal_id] = goal_lookup[goal.parent_goal_id]

        encoder = self._encoders.get(level, self.operational_encoder)
        return encoder.encode_batch(level_goals, game_state, parent_goals)

    def encode_active_goals(
        self,
        goals: list[Goal],
        game_state: dict[str, Any] | None = None,
        goal_lookup: dict[str, Goal] | None = None,
    ) -> dict[HierarchyLevel, np.ndarray]:
        """Encode active goals across all hierarchy levels.

        Args:
            goals: List of all active goals.
            game_state: Optional game state.
            goal_lookup: Optional mapping of goal_id -> Goal.

        Returns:
            Dictionary mapping levels to encoded goal arrays.
        """
        return {
            level: self.encode_goals_for_level(goals, level, game_state, goal_lookup)
            for level in HierarchyLevel
        }

    def get_output_dims(self) -> dict[HierarchyLevel, int]:
        """Get output dimensions for all level encoders.

        Returns:
            Dictionary mapping levels to output dimensions.
        """
        return {level: encoder.output_dim for level, encoder in self._encoders.items()}

    def get_encoder(self, level: HierarchyLevel) -> GoalEncoder:
        """Get encoder for a specific level.

        Args:
            level: Hierarchy level.

        Returns:
            Level-specific goal encoder.
        """
        return self._encoders[level]


if TORCH_AVAILABLE:

    class LearnedGoalEmbedding(nn.Module):
        """Learned goal embedding using neural networks.

        This module provides trainable embeddings for goals that can be
        fine-tuned during training for better goal conditioning.
        """

        def __init__(
            self,
            num_goal_types: int = 64,  # Max across all levels
            embed_dim: int = 64,
            include_location: bool = True,
            include_progress: bool = True,
            board_size: tuple[int, int] = (16, 16),
        ) -> None:
            """Initialize learned goal embedding.

            Args:
                num_goal_types: Maximum number of goal types.
                embed_dim: Embedding dimension.
                include_location: Whether to include location encoding.
                include_progress: Whether to include progress features.
                board_size: Game board size.
            """
            super().__init__()
            self.embed_dim = embed_dim
            self.include_location = include_location
            self.include_progress = include_progress
            self.board_size = board_size

            # Goal type embedding
            self.type_embedding = nn.Embedding(num_goal_types, embed_dim // 2)

            # Level embedding
            self.level_embedding = nn.Embedding(len(HierarchyLevel), embed_dim // 4)

            # Status embedding
            self.status_embedding = nn.Embedding(len(GoalStatus), embed_dim // 4)

            # Location encoder (if included)
            if include_location:
                self.location_encoder = nn.Sequential(
                    nn.Linear(2, embed_dim // 4),
                    nn.ReLU(),
                    nn.Linear(embed_dim // 4, embed_dim // 4),
                )

            # Progress encoder (if included)
            if include_progress:
                self.progress_encoder = nn.Sequential(
                    nn.Linear(4, embed_dim // 4),
                    nn.ReLU(),
                    nn.Linear(embed_dim // 4, embed_dim // 4),
                )

            # Final projection
            input_dim = (
                embed_dim // 2  # Type
                + embed_dim // 4  # Level
                + embed_dim // 4  # Status
                + (embed_dim // 4 if include_location else 0)
                + (embed_dim // 4 if include_progress else 0)
            )
            self.projection = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        def forward(
            self,
            goal_types: torch.Tensor,  # (batch, max_goals)
            goal_levels: torch.Tensor,  # (batch, max_goals)
            goal_statuses: torch.Tensor,  # (batch, max_goals)
            goal_locations: torch.Tensor | None = None,  # (batch, max_goals, 2)
            goal_progress: torch.Tensor | None = None,  # (batch, max_goals, 4)
        ) -> torch.Tensor:
            """Forward pass for goal embedding.

            Args:
                goal_types: Goal type indices.
                goal_levels: Hierarchy level indices.
                goal_statuses: Goal status indices.
                goal_locations: Optional normalized locations.
                goal_progress: Optional progress features.

            Returns:
                Goal embeddings of shape (batch, max_goals, embed_dim).
            """
            # Get base embeddings
            type_emb = self.type_embedding(goal_types)  # (B, G, E/2)
            level_emb = self.level_embedding(goal_levels)  # (B, G, E/4)
            status_emb = self.status_embedding(goal_statuses)  # (B, G, E/4)

            components = [type_emb, level_emb, status_emb]

            # Add location encoding if available
            if self.include_location and goal_locations is not None:
                loc_emb = self.location_encoder(goal_locations)
                components.append(loc_emb)

            # Add progress encoding if available
            if self.include_progress and goal_progress is not None:
                prog_emb = self.progress_encoder(goal_progress)
                components.append(prog_emb)

            # Concatenate and project
            combined = torch.cat(components, dim=-1)
            return self.projection(combined)
