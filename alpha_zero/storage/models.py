"""Data models for storage backends.

Provides dataclasses for:
- Model metadata (version, metrics, timestamps)
- Trajectory metadata (game info, worker info)
- Checkpoint metadata (iteration, state)
- Evaluation results (win rates, match history)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from alpha_zero.utils.compat import UTC


@dataclass
class ModelMetadata:
    """Metadata for a stored model checkpoint.

    Attributes:
        iteration: Training iteration when model was saved.
        version: Model version number (monotonically increasing).
        elo_rating: Optional Elo rating from evaluation.
        win_rate: Win rate against previous champion.
        timestamp: ISO timestamp when saved.
        config_hash: Hash of training config for reproducibility.
        training_step: Global training step count.
        metrics: Additional training metrics.
    """

    iteration: int
    version: int
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    elo_rating: float | None = None
    win_rate: float | None = None
    config_hash: str | None = None
    training_step: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create from dictionary."""
        return cls(
            iteration=data.get("iteration", 0),
            version=data.get("version", 0),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            elo_rating=data.get("elo_rating"),
            win_rate=data.get("win_rate"),
            config_hash=data.get("config_hash"),
            training_step=data.get("training_step", 0),
            metrics=data.get("metrics", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ModelMetadata:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TrajectoryMetadata:
    """Metadata for a stored game trajectory.

    Attributes:
        game_id: Unique game identifier.
        worker_id: Worker that generated this trajectory.
        iteration: Training iteration when game was played.
        game_length: Number of turns in the game.
        winner: Winning player (0, 1) or None for draw.
        model_version: Version of model used for self-play.
        timestamp: ISO timestamp when game completed.
        total_reward: Cumulative reward for the trajectory.
        num_examples: Number of training examples in trajectory.
    """

    game_id: str
    worker_id: int
    iteration: int
    game_length: int = 0
    winner: int | None = None
    model_version: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    total_reward: float = 0.0
    num_examples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryMetadata:
        """Create from dictionary."""
        return cls(
            game_id=data.get("game_id", ""),
            worker_id=data.get("worker_id", 0),
            iteration=data.get("iteration", 0),
            game_length=data.get("game_length", 0),
            winner=data.get("winner"),
            model_version=data.get("model_version", 0),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            total_reward=data.get("total_reward", 0.0),
            num_examples=data.get("num_examples", 0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> TrajectoryMetadata:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class CheckpointMetadata:
    """Metadata for a full training checkpoint.

    Attributes:
        iteration: Training iteration.
        training_step: Global step count.
        model_version: Model version number.
        timestamp: ISO timestamp.
        games_played: Total games played so far.
        total_examples: Total training examples generated.
        curriculum_stage: Current curriculum stage.
        best_win_rate: Best win rate achieved.
        optimizer_included: Whether optimizer state is included.
        scheduler_included: Whether scheduler state is included.
    """

    iteration: int
    training_step: int = 0
    model_version: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    games_played: int = 0
    total_examples: int = 0
    curriculum_stage: int = 0
    best_win_rate: float = 0.0
    optimizer_included: bool = True
    scheduler_included: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary."""
        return cls(
            iteration=data.get("iteration", 0),
            training_step=data.get("training_step", 0),
            model_version=data.get("model_version", 0),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            games_played=data.get("games_played", 0),
            total_examples=data.get("total_examples", 0),
            curriculum_stage=data.get("curriculum_stage", 0),
            best_win_rate=data.get("best_win_rate", 0.0),
            optimizer_included=data.get("optimizer_included", True),
            scheduler_included=data.get("scheduler_included", True),
        )

    @classmethod
    def from_json(cls, json_str: str) -> CheckpointMetadata:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class EvaluationResult:
    """Result of model evaluation (candidate vs champion).

    Attributes:
        iteration: Training iteration of candidate model.
        candidate_version: Version of candidate model.
        champion_version: Version of champion model.
        num_games: Total games played in evaluation.
        candidate_wins: Games won by candidate.
        champion_wins: Games won by champion.
        draws: Games that ended in draw.
        win_rate: Candidate win rate (wins / total).
        promoted: Whether candidate was promoted to champion.
        timestamp: ISO timestamp of evaluation completion.
        game_results: Optional detailed results per game.
    """

    iteration: int
    candidate_version: int
    champion_version: int
    num_games: int
    candidate_wins: int
    champion_wins: int
    draws: int
    win_rate: float
    promoted: bool
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    game_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        """Create from dictionary."""
        return cls(
            iteration=data.get("iteration", 0),
            candidate_version=data.get("candidate_version", 0),
            champion_version=data.get("champion_version", 0),
            num_games=data.get("num_games", 0),
            candidate_wins=data.get("candidate_wins", 0),
            champion_wins=data.get("champion_wins", 0),
            draws=data.get("draws", 0),
            win_rate=data.get("win_rate", 0.0),
            promoted=data.get("promoted", False),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            game_results=data.get("game_results", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> EvaluationResult:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


__all__ = [
    "CheckpointMetadata",
    "EvaluationResult",
    "ModelMetadata",
    "TrajectoryMetadata",
]
