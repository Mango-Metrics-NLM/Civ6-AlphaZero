"""Debug self-play script for flushing out bugs in Civ6 AlphaZero.

This script runs a SHORT self-play session with comprehensive validation
and debugging output to identify issues in the training pipeline.

Usage:
    python run_debug_selfplay.py --num-games 3 --max-turns 15
    python run_debug_selfplay.py --output-json debug_report.json
    python run_debug_selfplay.py --use-network-mcts --num-simulations 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig, SimpleMCTS
from alpha_zero.core.civ6_network import MicroCivNetwork, compute_loss
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.utils.constants import ENCODER
from alpha_zero.utils.constants import MCTS as MCTS_CONSTANTS
from alpha_zero.utils.logging import (
    GameLogger,
    MetricsTracker,
    PerformanceProfiler,
    get_logger,
    setup_logging,
)

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class DebugSelfPlayConfig:
    """Configuration for debugging self-play runs.

    All parameters are configurable - no hardcoded values.
    """

    # Game parameters
    num_games: int = 3
    max_turns_per_game: int = 15
    board_size: int = 8

    # Network parameters (dynamically determined based on board_size)
    num_res_blocks: int = 2
    num_filters: int = 16

    # MCTS parameters
    num_simulations: int = 10
    c_puct: float = MCTS_CONSTANTS.DEFAULT_C_PUCT
    use_simple_mcts: bool = True  # Start with SimpleMCTS for debugging

    # Logging
    log_level: str = "DEBUG"
    log_dir: str = "logs/debug_selfplay"
    save_game_logs: bool = True

    # Validation toggles
    validate_observations: bool = True
    validate_actions: bool = True
    validate_network_outputs: bool = True
    validate_mcts_search: bool = True
    validate_replay_buffer: bool = True

    # Training step test
    run_training_step: bool = True
    training_batch_size: int = 8

    # Error handling
    fail_fast: bool = False  # Stop on first error vs collect all errors

    # Device
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> DebugSelfPlayConfig:
        """Load config from environment variables with CIV6AZ_DEBUG_ prefix."""
        config = cls()

        env_mappings = {
            "CIV6AZ_DEBUG_NUM_GAMES": ("num_games", int),
            "CIV6AZ_DEBUG_MAX_TURNS": ("max_turns_per_game", int),
            "CIV6AZ_DEBUG_BOARD_SIZE": ("board_size", int),
            "CIV6AZ_DEBUG_NUM_SIMULATIONS": ("num_simulations", int),
            "CIV6AZ_DEBUG_LOG_LEVEL": ("log_level", str),
            "CIV6AZ_DEBUG_FAIL_FAST": ("fail_fast", lambda x: x.lower() == "true"),
        }

        for env_key, (attr, converter) in env_mappings.items():
            value = os.environ.get(env_key)
            if value is not None:
                setattr(config, attr, converter(value))

        return config


# =============================================================================
# VALIDATORS
# =============================================================================


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ObservationValidator:
    """Validates environment observations for correctness."""

    def __init__(
        self,
        board_size: int,
        spatial_channels: int,
        vector_dim: int,
        action_space_size: int,
    ):
        self.board_size = board_size
        self.spatial_channels = spatial_channels
        self.vector_dim = vector_dim
        self.action_space_size = action_space_size
        self.logger = get_logger("validator.observation")

    def validate(self, obs: dict[str, np.ndarray], step: int) -> ValidationResult:
        """Validate an observation dictionary."""
        errors = []
        warnings = []

        # Check required keys
        required_keys = {"spatial", "vector", "action_mask"}
        missing_keys = required_keys - set(obs.keys())
        if missing_keys:
            errors.append(f"Step {step}: Missing keys: {missing_keys}")

        # Check spatial shape
        if "spatial" in obs:
            expected_shape = (self.spatial_channels, self.board_size, self.board_size)
            if obs["spatial"].shape != expected_shape:
                errors.append(
                    f"Step {step}: Spatial shape {obs['spatial'].shape} != expected {expected_shape}"
                )

            # Check for NaN/Inf
            if np.any(np.isnan(obs["spatial"])):
                errors.append(f"Step {step}: Spatial contains NaN values")
            if np.any(np.isinf(obs["spatial"])):
                errors.append(f"Step {step}: Spatial contains Inf values")

            # Check value range
            if np.any(obs["spatial"] < -10) or np.any(obs["spatial"] > 10):
                warnings.append(
                    f"Step {step}: Spatial values outside [-10, 10] range"
                )

        # Check vector shape
        if "vector" in obs:
            expected_shape = (self.vector_dim,)
            if obs["vector"].shape != expected_shape:
                errors.append(
                    f"Step {step}: Vector shape {obs['vector'].shape} != expected {expected_shape}"
                )

            # Check for NaN/Inf
            if np.any(np.isnan(obs["vector"])):
                errors.append(f"Step {step}: Vector contains NaN values")
            if np.any(np.isinf(obs["vector"])):
                errors.append(f"Step {step}: Vector contains Inf values")

        # Check action mask
        if "action_mask" in obs:
            expected_shape = (self.action_space_size,)
            if obs["action_mask"].shape != expected_shape:
                errors.append(
                    f"Step {step}: Action mask shape {obs['action_mask'].shape} != expected {expected_shape}"
                )

            # Check mask is binary
            unique_values = np.unique(obs["action_mask"])
            if not np.all(np.isin(unique_values, [0, 1])):
                errors.append(
                    f"Step {step}: Action mask contains non-binary values: {unique_values}"
                )

            # Check at least one valid action
            if np.sum(obs["action_mask"]) == 0:
                errors.append(f"Step {step}: No valid actions in mask")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


class NetworkValidator:
    """Validates network outputs."""

    def __init__(self, action_space_size: int):
        self.action_space_size = action_space_size
        self.logger = get_logger("validator.network")

    def validate_forward(
        self,
        policy_logits: torch.Tensor,
        value: torch.Tensor,
        action_mask: torch.Tensor,
        step: int,
    ) -> ValidationResult:
        """Validate network forward pass outputs."""
        errors = []
        warnings = []

        # Check policy logits shape
        if policy_logits.dim() == 2:
            if policy_logits.shape[1] != self.action_space_size:
                errors.append(
                    f"Step {step}: Policy logits shape {policy_logits.shape} != expected (batch, {self.action_space_size})"
                )
        elif policy_logits.dim() == 1:
            if policy_logits.shape[0] != self.action_space_size:
                errors.append(
                    f"Step {step}: Policy logits shape {policy_logits.shape} != expected ({self.action_space_size},)"
                )

        # Check for NaN/Inf in policy logits
        if torch.any(torch.isnan(policy_logits)):
            errors.append(f"Step {step}: Policy logits contain NaN values")
        if torch.any(torch.isinf(policy_logits)):
            # Check if it's the expected masking value
            inf_count = torch.sum(torch.isinf(policy_logits)).item()
            masked_count = torch.sum(action_mask == 0).item()
            if inf_count != masked_count:
                warnings.append(
                    f"Step {step}: Policy logits Inf count ({inf_count}) != masked actions ({masked_count})"
                )

        # Check value bounds
        if value.dim() == 0:
            value_scalar = value.item()
        else:
            value_scalar = value.squeeze().item()

        if value_scalar < -1.0 or value_scalar > 1.0:
            warnings.append(
                f"Step {step}: Value {value_scalar:.4f} outside [-1, 1] range"
            )

        # Check for NaN in value
        if torch.isnan(value).any():
            errors.append(f"Step {step}: Value contains NaN")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_predict(
        self, policy_probs: np.ndarray, value: float, step: int
    ) -> ValidationResult:
        """Validate predict() outputs."""
        errors = []
        warnings = []

        # Check probabilities sum to ~1
        prob_sum = np.sum(policy_probs)
        if abs(prob_sum - 1.0) > 1e-4:
            errors.append(
                f"Step {step}: Policy probs sum to {prob_sum:.6f}, expected ~1.0"
            )

        # Check all probs are non-negative
        if np.any(policy_probs < 0):
            errors.append(f"Step {step}: Negative policy probabilities found")

        # Check for NaN
        if np.any(np.isnan(policy_probs)):
            errors.append(f"Step {step}: Policy probs contain NaN")

        # Check value bounds
        if value < -1.0 or value > 1.0:
            warnings.append(f"Step {step}: Value {value:.4f} outside [-1, 1] range")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


class MCTSValidator:
    """Validates MCTS search results."""

    def __init__(self, action_space_size: int):
        self.action_space_size = action_space_size
        self.logger = get_logger("validator.mcts")

    def validate_search(
        self,
        action_probs: np.ndarray,
        root_value: float,
        action_mask: np.ndarray,
        step: int,
    ) -> ValidationResult:
        """Validate MCTS search output."""
        errors = []
        warnings = []

        # Check action_probs shape
        if action_probs.shape[0] != self.action_space_size:
            errors.append(
                f"Step {step}: Action probs shape {action_probs.shape} != expected ({self.action_space_size},)"
            )

        # Check probabilities sum to ~1
        prob_sum = np.sum(action_probs)
        if abs(prob_sum - 1.0) > 1e-4:
            errors.append(
                f"Step {step}: Action probs sum to {prob_sum:.6f}, expected ~1.0"
            )

        # Check only valid actions have non-zero probability
        invalid_probs = action_probs * (1 - action_mask)
        if np.sum(invalid_probs) > 1e-6:
            errors.append(
                f"Step {step}: Invalid actions have non-zero probability: {np.sum(invalid_probs):.6f}"
            )

        # Check root_value bounds
        if root_value < -1.0 or root_value > 1.0:
            warnings.append(
                f"Step {step}: Root value {root_value:.4f} outside [-1, 1] range"
            )

        # Check for NaN
        if np.any(np.isnan(action_probs)):
            errors.append(f"Step {step}: Action probs contain NaN")
        if np.isnan(root_value):
            errors.append(f"Step {step}: Root value is NaN")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


class ReplayBufferValidator:
    """Validates replay buffer operations."""

    def __init__(
        self,
        spatial_channels: int,
        board_size: int,
        vector_dim: int,
        action_space_size: int,
    ):
        self.spatial_channels = spatial_channels
        self.board_size = board_size
        self.vector_dim = vector_dim
        self.action_space_size = action_space_size
        self.logger = get_logger("validator.buffer")

    def validate_example(
        self, example: dict[str, Any], index: int
    ) -> ValidationResult:
        """Validate a single training example."""
        errors = []
        warnings = []

        # Check spatial
        expected_spatial = (self.spatial_channels, self.board_size, self.board_size)
        if example["spatial"].shape != expected_spatial:
            errors.append(
                f"Example {index}: Spatial shape {example['spatial'].shape} != {expected_spatial}"
            )

        # Check vector
        if example["vector"].shape != (self.vector_dim,):
            errors.append(
                f"Example {index}: Vector shape {example['vector'].shape} != ({self.vector_dim},)"
            )

        # Check action_mask
        if example["action_mask"].shape != (self.action_space_size,):
            errors.append(
                f"Example {index}: Action mask shape {example['action_mask'].shape} != ({self.action_space_size},)"
            )

        # Check policy
        if example["policy"].shape != (self.action_space_size,):
            errors.append(
                f"Example {index}: Policy shape {example['policy'].shape} != ({self.action_space_size},)"
            )

        # Check policy sums to ~1
        policy_sum = np.sum(example["policy"])
        if abs(policy_sum - 1.0) > 1e-4:
            errors.append(
                f"Example {index}: Policy sums to {policy_sum:.6f}, expected ~1.0"
            )

        # Check value in valid range
        if example["value"] < -1.0 or example["value"] > 1.0:
            errors.append(
                f"Example {index}: Value {example['value']:.4f} outside [-1, 1] range"
            )

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# =============================================================================
# DIAGNOSTICS
# =============================================================================


@dataclass
class DebugDiagnostics:
    """Collected diagnostics from the debug run."""

    # Error tracking by category
    observation_errors: list[tuple[int, int, str]] = field(default_factory=list)
    network_errors: list[tuple[int, int, str]] = field(default_factory=list)
    mcts_errors: list[tuple[int, int, str]] = field(default_factory=list)
    action_errors: list[tuple[int, int, str]] = field(default_factory=list)
    buffer_errors: list[str] = field(default_factory=list)
    training_errors: list[str] = field(default_factory=list)

    # Warnings
    warnings: list[str] = field(default_factory=list)

    # Statistics
    games_completed: int = 0
    total_steps: int = 0
    total_examples: int = 0

    # Timing
    total_time: float = 0.0
    game_times: list[float] = field(default_factory=list)

    # Game results
    game_results: list[dict[str, Any]] = field(default_factory=list)

    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return (
            len(self.observation_errors) > 0
            or len(self.network_errors) > 0
            or len(self.mcts_errors) > 0
            or len(self.action_errors) > 0
            or len(self.buffer_errors) > 0
            or len(self.training_errors) > 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "observation_errors": self.observation_errors,
            "network_errors": self.network_errors,
            "mcts_errors": self.mcts_errors,
            "action_errors": self.action_errors,
            "buffer_errors": self.buffer_errors,
            "training_errors": self.training_errors,
            "warnings": self.warnings,
            "games_completed": self.games_completed,
            "total_steps": self.total_steps,
            "total_examples": self.total_examples,
            "total_time": self.total_time,
            "game_times": self.game_times,
            "game_results": self.game_results,
        }


# =============================================================================
# TERMINATION AND ACTION DIAGNOSTICS
# =============================================================================


@dataclass
class ActionRecord:
    """Record of a single action taken during a game."""

    game_id: int
    step: int
    turn: int
    player: int
    action_index: int
    action_type: str
    was_valid: bool
    action_mask_sum: int
    execution_result: str
    mcts_root_value: float = 0.0


class ActionTracer:
    """Traces actions for debugging invalid action selection."""

    def __init__(self, action_space_handler: Any | None = None):
        self.action_space = action_space_handler
        self.logger = get_logger("tracer.action")

        self.records: list[ActionRecord] = []
        self.invalid_action_count = 0
        self.action_type_counts: dict[str, int] = {}

    def trace_action(
        self,
        game_id: int,
        step: int,
        turn: int,
        player: int,
        action: int,
        action_mask: np.ndarray,
        action_dict: dict[str, Any],
        mcts_value: float = 0.0,
    ) -> ActionRecord:
        """Record an action selection."""
        action_type = action_dict.get("type", "unknown")
        was_valid = action_mask[action] > 0

        if not was_valid:
            self.invalid_action_count += 1
            self.logger.error(
                f"Game {game_id} Step {step}: INVALID ACTION {action} "
                f"({action_type}) selected!"
            )

        self.action_type_counts[action_type] = (
            self.action_type_counts.get(action_type, 0) + 1
        )

        record = ActionRecord(
            game_id=game_id,
            step=step,
            turn=turn,
            player=player,
            action_index=action,
            action_type=action_type,
            was_valid=was_valid,
            action_mask_sum=int(action_mask.sum()),
            execution_result="pending",
            mcts_root_value=mcts_value,
        )

        self.records.append(record)
        return record

    def update_execution_result(self, success: bool, error: str = "") -> None:
        """Update the last record with execution result."""
        if self.records:
            result = "success" if success else f"failed: {error}"
            # Use object.__setattr__ since dataclass might be frozen
            self.records[-1] = ActionRecord(
                game_id=self.records[-1].game_id,
                step=self.records[-1].step,
                turn=self.records[-1].turn,
                player=self.records[-1].player,
                action_index=self.records[-1].action_index,
                action_type=self.records[-1].action_type,
                was_valid=self.records[-1].was_valid,
                action_mask_sum=self.records[-1].action_mask_sum,
                execution_result=result,
                mcts_root_value=self.records[-1].mcts_root_value,
            )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of traced actions."""
        if not self.records:
            return {"total_actions": 0}

        total_actions = len(self.records)
        invalid_rate = self.invalid_action_count / max(total_actions, 1)
        end_turn_count = self.action_type_counts.get("end_turn", 0)

        return {
            "total_actions": total_actions,
            "invalid_action_count": self.invalid_action_count,
            "invalid_action_rate": invalid_rate,
            "action_type_distribution": dict(self.action_type_counts),
            "end_turn_rate": end_turn_count / max(total_actions, 1),
            "avg_valid_actions_available": np.mean(
                [r.action_mask_sum for r in self.records]
            ),
        }


class GameTerminationValidator:
    """Validates game termination conditions and tracks early endings."""

    def __init__(self, min_expected_turns: int = 2):
        self.min_expected_turns = min_expected_turns
        self.logger = get_logger("validator.termination")
        self.termination_records: list[dict[str, Any]] = []

    def validate_termination(
        self,
        game_id: int,
        turn: int,
        winner: int | None,
        player_units: tuple[int, int],
        player_cities: tuple[int, int],
        terminated: bool,
        truncated: bool,
    ) -> ValidationResult:
        """Validate game termination event."""
        errors = []
        warnings = []

        # Determine termination reason
        if truncated:
            reason = "timeout"
        elif winner is not None:
            loser = 1 - winner
            if player_units[loser] == 0 and player_cities[loser] == 0:
                reason = f"player_{loser}_eliminated"
            else:
                reason = "unknown_victory"
        else:
            reason = "draw"

        # Record termination
        self.termination_records.append({
            "game_id": game_id,
            "turn": turn,
            "winner": winner,
            "reason": reason,
            "player_units": player_units,
            "player_cities": player_cities,
        })

        # Check for premature termination
        if turn < self.min_expected_turns and terminated and not truncated:
            errors.append(
                f"Game {game_id}: CRITICAL - Ended turn {turn}, "
                f"winner={winner}, units={player_units}, cities={player_cities}"
            )

        # Check for turn 0 termination (should never happen)
        if turn == 0 and terminated:
            errors.append(
                f"Game {game_id}: Game terminated at turn 0 - this is a bug"
            )

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all termination events."""
        if not self.termination_records:
            return {"total_games": 0}

        turns = [r["turn"] for r in self.termination_records]
        reasons = [r["reason"] for r in self.termination_records]

        reason_counts: dict[str, int] = {}
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Group by turn buckets
        turn_buckets = {
            "turn_0": sum(1 for t in turns if t == 0),
            "turn_1": sum(1 for t in turns if t == 1),
            "turns_2-5": sum(1 for t in turns if 2 <= t <= 5),
            "turns_6-10": sum(1 for t in turns if 6 <= t <= 10),
            "turns_10+": sum(1 for t in turns if t > 10),
        }

        return {
            "total_games": len(self.termination_records),
            "avg_turns": sum(turns) / len(turns) if turns else 0,
            "min_turns": min(turns) if turns else 0,
            "max_turns": max(turns) if turns else 0,
            "early_terminations": sum(1 for t in turns if t < self.min_expected_turns),
            "reason_distribution": reason_counts,
            "games_by_turn": turn_buckets,
        }


@dataclass
class StateSnapshot:
    """Snapshot of environment state at a point in time."""

    game_id: int
    step: int
    turn: int
    current_player: int
    player_units: tuple[int, int]
    player_cities: tuple[int, int]
    winner: int | None
    done: bool
    action_mask_sum: int
    last_action: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "game_id": self.game_id,
            "step": self.step,
            "turn": self.turn,
            "current_player": self.current_player,
            "player_units": self.player_units,
            "player_cities": self.player_cities,
            "winner": self.winner,
            "done": self.done,
            "action_mask_sum": self.action_mask_sum,
            "last_action": self.last_action,
        }


class StateSnapshotManager:
    """Manages state snapshots for post-mortem analysis."""

    def __init__(self, snapshot_dir: str = "logs/snapshots"):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("snapshot.manager")

        self.termination_snapshots: list[StateSnapshot] = []

    def capture_termination(
        self,
        env: Any,
        game_id: int,
        step: int,
        last_action: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """Capture snapshot at game termination."""
        # Count units/cities per player
        p0_units = len([u for u in env.units if u.player == 0])
        p0_cities = len([c for c in env.cities if c.player == 0])
        p1_units = len([u for u in env.units if u.player == 1])
        p1_cities = len([c for c in env.cities if c.player == 1])

        action_mask = env.get_valid_actions()

        snapshot = StateSnapshot(
            game_id=game_id,
            step=step,
            turn=env.turn,
            current_player=env.current_player,
            player_units=(p0_units, p1_units),
            player_cities=(p0_cities, p1_cities),
            winner=env.winner,
            done=env.done,
            action_mask_sum=int(action_mask.sum()),
            last_action=last_action,
        )

        self.termination_snapshots.append(snapshot)

        # Save early terminations
        if env.turn < 2:
            self.logger.warning(
                f"Early termination: game={game_id}, turn={env.turn}, winner={env.winner}"
            )
            self.save_snapshot(snapshot, f"early_term_game{game_id}_turn{env.turn}.json")

        return snapshot

    def save_snapshot(self, snapshot: StateSnapshot, filename: str) -> Path:
        """Save snapshot to file."""
        filepath = self.snapshot_dir / filename
        with open(filepath, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        return filepath

    def get_early_terminations(self, max_turns: int = 2) -> list[StateSnapshot]:
        """Get snapshots of games that ended early."""
        return [s for s in self.termination_snapshots if s.turn < max_turns]


@dataclass
class DebugReport:
    """Final report from debug run."""

    config: dict[str, Any]
    diagnostics: DebugDiagnostics
    profiler_summary: dict[str, Any]
    component_status: dict[str, str]  # "OK", "WARNING", "ERROR"
    recommendations: list[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("DEBUG SELF-PLAY REPORT")
        print("=" * 70)

        print("\nConfiguration:")
        for key, value in self.config.items():
            print(f"  - {key}: {value}")

        print("\nComponent Status:")
        for component, status in self.component_status.items():
            status_symbol = {"OK": "[OK]", "WARNING": "[!]", "ERROR": "[X]"}.get(
                status, "[?]"
            )
            print(f"  {status_symbol} {component}: {status}")

        print("\nValidation Results:")
        d = self.diagnostics
        print(f"  - Observation errors: {len(d.observation_errors)}")
        print(f"  - Network errors: {len(d.network_errors)}")
        print(f"  - MCTS errors: {len(d.mcts_errors)}")
        print(f"  - Action errors: {len(d.action_errors)}")
        print(f"  - Buffer errors: {len(d.buffer_errors)}")
        print(f"  - Training errors: {len(d.training_errors)}")
        print(f"  - Warnings: {len(d.warnings)}")

        print("\nStatistics:")
        print(f"  - Games completed: {d.games_completed}")
        print(f"  - Total steps: {d.total_steps}")
        print(f"  - Total examples: {d.total_examples}")
        print(f"  - Total time: {d.total_time:.2f}s")
        if d.game_times:
            print(f"  - Avg game time: {sum(d.game_times) / len(d.game_times):.2f}s")

        if self.profiler_summary:
            print("\nPerformance Summary:")
            for section, stats in self.profiler_summary.items():
                if isinstance(stats, dict) and "mean" in stats:
                    print(
                        f"  - {section}: {stats['mean']:.4f}s avg ({stats['count']} calls)"
                    )

        if self.recommendations:
            print("\nRecommendations:")
            for rec in self.recommendations:
                print(f"  - {rec}")

        result = "PASS" if not d.has_errors() else "FAIL"
        print(f"\nRESULT: {result}")
        if d.has_errors():
            print("  Critical errors detected - review error logs for details")
        print("=" * 70 + "\n")

    def save_json(self, path: str) -> None:
        """Save full report as JSON."""
        report_dict = {
            "config": self.config,
            "diagnostics": self.diagnostics.to_dict(),
            "profiler_summary": self.profiler_summary,
            "component_status": self.component_status,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }
        with open(path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)


# =============================================================================
# DEBUG SELF-PLAY RUNNER
# =============================================================================


class DebugSelfPlayRunner:
    """Runs debugging self-play with comprehensive validation."""

    def __init__(self, config: DebugSelfPlayConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # Create log directory
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.game_logger = GameLogger(log_dir=config.log_dir)
        self.profiler = PerformanceProfiler()
        self.metrics = MetricsTracker()
        self.diagnostics = DebugDiagnostics()

        # New diagnostic components for termination analysis
        self.action_tracer = ActionTracer()
        self.termination_validator = GameTerminationValidator(min_expected_turns=2)
        self.snapshot_manager = StateSnapshotManager(
            snapshot_dir=f"{config.log_dir}/snapshots"
        )

        # Components (initialized in _setup_components)
        self.env: MicroCivEnv | None = None
        self.network: MicroCivNetwork | None = None
        self.mcts: Civ6MCTS | SimpleMCTS | None = None

        # Validators (initialized in _setup_components)
        self.obs_validator: ObservationValidator | None = None
        self.network_validator: NetworkValidator | None = None
        self.mcts_validator: MCTSValidator | None = None
        self.buffer_validator: ReplayBufferValidator | None = None

        # Training examples storage
        self.training_examples: list[dict[str, Any]] = []

    def run(self) -> DebugReport:
        """Run the full debugging session."""
        start_time = time.time()

        self.logger.info("=" * 60)
        self.logger.info("DEBUG SELF-PLAY SESSION")
        self.logger.info("=" * 60)
        self.logger.info(f"Games: {self.config.num_games}")
        self.logger.info(f"Max turns: {self.config.max_turns_per_game}")
        self.logger.info(f"Board size: {self.config.board_size}")
        self.logger.info(f"MCTS simulations: {self.config.num_simulations}")
        self.logger.info(f"Using SimpleMCTS: {self.config.use_simple_mcts}")

        # Setup components
        self._setup_components()

        # Run self-play games
        self._run_self_play_games()

        # Run training step test
        if self.config.run_training_step and len(self.training_examples) > 0:
            self._run_training_step_test()

        # Calculate total time
        self.diagnostics.total_time = time.time() - start_time

        # Generate report
        return self._generate_report()

    def _setup_components(self) -> None:
        """Initialize all components with validation."""
        self.logger.info("\nSetting up components...")

        # Create environment
        with self.profiler.section("env_setup"):
            self.env = MicroCivEnv(
                board_size=self.config.board_size,
                max_turns=self.config.max_turns_per_game,
            )
            action_space_size = self.env.action_space_handler.total_size
            self.logger.info(f"Environment created: action_space={action_space_size}")

        # Create network
        with self.profiler.section("network_setup"):
            self.network = MicroCivNetwork(
                board_size=self.config.board_size,
                action_space_size=action_space_size,
                num_res_blocks=self.config.num_res_blocks,
                num_filters=self.config.num_filters,
            ).to(self.config.device)
            self.network.eval()

            param_count = sum(p.numel() for p in self.network.parameters())
            self.logger.info(f"Network created: {param_count:,} parameters")

        # Create MCTS
        with self.profiler.section("mcts_setup"):
            if self.config.use_simple_mcts:
                self.mcts = SimpleMCTS(
                    num_simulations=self.config.num_simulations,
                    c_puct=self.config.c_puct,
                )
                self.logger.info("SimpleMCTS created (random rollout)")
            else:
                mcts_config = MCTSConfig(
                    num_simulations=self.config.num_simulations,
                    c_puct=self.config.c_puct,
                    dirichlet_alpha=MCTS_CONSTANTS.DEFAULT_DIRICHLET_ALPHA,
                    dirichlet_epsilon=MCTS_CONSTANTS.DEFAULT_DIRICHLET_EPSILON,
                )
                self.mcts = Civ6MCTS(
                    self.network, mcts_config, device=self.config.device
                )
                self.logger.info("Civ6MCTS created (network-guided)")

        # Create validators
        spatial_channels = ENCODER.MICROCIV_CHANNELS
        vector_dim = ENCODER.MICROCIV_VECTOR_DIM

        self.obs_validator = ObservationValidator(
            board_size=self.config.board_size,
            spatial_channels=spatial_channels,
            vector_dim=vector_dim,
            action_space_size=action_space_size,
        )
        self.network_validator = NetworkValidator(action_space_size=action_space_size)
        self.mcts_validator = MCTSValidator(action_space_size=action_space_size)
        self.buffer_validator = ReplayBufferValidator(
            spatial_channels=spatial_channels,
            board_size=self.config.board_size,
            vector_dim=vector_dim,
            action_space_size=action_space_size,
        )

        self.logger.info("Validators created")

    def _run_self_play_games(self) -> None:
        """Run all self-play games."""
        self.logger.info(f"\nRunning {self.config.num_games} self-play games...")

        for game_id in range(self.config.num_games):
            game_start = time.time()
            self.logger.info(f"\n--- Game {game_id + 1}/{self.config.num_games} ---")

            try:
                result = self._run_single_game(game_id)
                self.diagnostics.game_results.append(result)
                self.diagnostics.games_completed += 1

            except Exception as e:
                self.logger.error(f"Game {game_id} failed: {e}", exc_info=True)
                self.diagnostics.game_results.append(
                    {"game_id": game_id, "error": str(e)}
                )
                if self.config.fail_fast:
                    raise

            game_time = time.time() - game_start
            self.diagnostics.game_times.append(game_time)
            self.logger.info(f"Game {game_id + 1} completed in {game_time:.2f}s")

    def _run_single_game(self, game_id: int) -> dict[str, Any]:
        """Run a single game with step-by-step validation."""
        assert self.env is not None
        assert self.mcts is not None

        self.game_logger.start_game(game_id)

        obs, info = self.env.reset()
        game_history: list[dict[str, Any]] = []
        step = 0

        # Safety limit
        max_steps = self.config.max_turns_per_game * 10

        while step < max_steps:
            # Validate observation
            if self.config.validate_observations:
                result = self.obs_validator.validate(obs, step)
                if not result.valid:
                    for error in result.errors:
                        self.diagnostics.observation_errors.append(
                            (game_id, step, error)
                        )
                        self.logger.error(f"Observation error: {error}")
                self.diagnostics.warnings.extend(result.warnings)

            # Network inference (if using network MCTS)
            if (
                self.config.validate_network_outputs
                and not self.config.use_simple_mcts
            ):
                with self.profiler.section("network_inference"):
                    with torch.no_grad():
                        spatial_t = (
                            torch.from_numpy(obs["spatial"])
                            .unsqueeze(0)
                            .float()
                            .to(self.config.device)
                        )
                        vector_t = (
                            torch.from_numpy(obs["vector"])
                            .unsqueeze(0)
                            .float()
                            .to(self.config.device)
                        )
                        mask_t = (
                            torch.from_numpy(obs["action_mask"])
                            .unsqueeze(0)
                            .float()
                            .to(self.config.device)
                        )

                        policy_logits, value = self.network(spatial_t, vector_t, mask_t)

                        result = self.network_validator.validate_forward(
                            policy_logits, value, mask_t, step
                        )
                        if not result.valid:
                            for error in result.errors:
                                self.diagnostics.network_errors.append(
                                    (game_id, step, error)
                                )
                                self.logger.error(f"Network error: {error}")
                        self.diagnostics.warnings.extend(result.warnings)

            # MCTS search
            with self.profiler.section("mcts_search"):
                action_probs, root_value = self.mcts.search(self.env)

            # Validate MCTS results
            if self.config.validate_mcts_search:
                result = self.mcts_validator.validate_search(
                    action_probs, root_value, obs["action_mask"], step
                )
                if not result.valid:
                    for error in result.errors:
                        self.diagnostics.mcts_errors.append((game_id, step, error))
                        self.logger.error(f"MCTS error: {error}")
                self.diagnostics.warnings.extend(result.warnings)

            # Store example
            game_history.append(
                {
                    "spatial": obs["spatial"].copy(),
                    "vector": obs["vector"].copy(),
                    "action_mask": obs["action_mask"].copy(),
                    "policy": action_probs.copy(),
                    "player": self.env.current_player,
                }
            )

            # Select action with action mask enforcement
            temp = 1.0 if step < 10 else 0.0
            action = self.mcts.select_action(
                action_probs, temp, action_mask=obs["action_mask"]
            )

            # Decode action for tracing
            action_dict = self.env.action_space_handler.decode_action(action)

            # Trace action for debugging
            self.action_tracer.trace_action(
                game_id=game_id,
                step=step,
                turn=self.env.turn,
                player=self.env.current_player,
                action=action,
                action_mask=obs["action_mask"],
                action_dict=action_dict,
                mcts_value=root_value,
            )

            # Validate action
            if self.config.validate_actions:
                if obs["action_mask"][action] == 0:
                    error = f"Step {step}: Selected invalid action {action}"
                    self.diagnostics.action_errors.append((game_id, step, error))
                    self.logger.error(f"Action error: {error}")

            # Log to game logger
            self.game_logger.log_state(
                turn=self.env.turn,
                player=self.env.current_player,
                action=action,
                action_probs=action_probs,
                value_estimate=root_value,
                valid_actions=obs["action_mask"],
            )

            # Take step
            obs, reward, terminated, truncated, info = self.env.step(action)
            step += 1
            self.diagnostics.total_steps += 1

            # Update action tracer with execution result
            action_success = info.get("action_success", True)
            action_error = info.get("action_error", "")
            self.action_tracer.update_execution_result(action_success, action_error)

            if terminated or truncated:
                # Capture termination snapshot
                self.snapshot_manager.capture_termination(
                    env=self.env,
                    game_id=game_id,
                    step=step,
                    last_action=action_dict,
                )

                # Validate termination
                p0_units = len([u for u in self.env.units if u.player == 0])
                p1_units = len([u for u in self.env.units if u.player == 1])
                p0_cities = len([c for c in self.env.cities if c.player == 0])
                p1_cities = len([c for c in self.env.cities if c.player == 1])

                term_result = self.termination_validator.validate_termination(
                    game_id=game_id,
                    turn=self.env.turn,
                    winner=info.get("winner"),
                    player_units=(p0_units, p1_units),
                    player_cities=(p0_cities, p1_cities),
                    terminated=terminated,
                    truncated=truncated,
                )

                if not term_result.valid:
                    for error in term_result.errors:
                        self.diagnostics.action_errors.append((game_id, step, error))
                        self.logger.error(f"Termination error: {error}")

                break

        # Assign values based on outcome
        winner = info.get("winner")

        for hist in game_history:
            if winner is None:
                value = 0.0
            elif winner == hist["player"]:
                value = 1.0
            else:
                value = -1.0

            example = {
                "spatial": hist["spatial"],
                "vector": hist["vector"],
                "action_mask": hist["action_mask"],
                "policy": hist["policy"],
                "value": value,
            }

            # Validate example
            if self.config.validate_replay_buffer:
                result = self.buffer_validator.validate_example(
                    example, len(self.training_examples)
                )
                if not result.valid:
                    for error in result.errors:
                        self.diagnostics.buffer_errors.append(error)
                        self.logger.error(f"Buffer error: {error}")

            self.training_examples.append(example)
            self.diagnostics.total_examples += 1

        # End game logging
        game_data = self.game_logger.end_game(
            winner=winner, total_turns=self.env.turn, reason="normal"
        )

        if self.config.save_game_logs:
            self.game_logger.save_game(game_data)

        return {
            "game_id": game_id,
            "steps": step,
            "winner": winner,
            "turn": self.env.turn,
            "examples": len(game_history),
        }

    def _run_training_step_test(self) -> None:
        """Test a training step with collected examples."""
        self.logger.info("\nRunning training step test...")

        if len(self.training_examples) < self.config.training_batch_size:
            self.logger.warning(
                f"Not enough examples ({len(self.training_examples)}) "
                f"for batch size ({self.config.training_batch_size})"
            )
            return

        try:
            with self.profiler.section("training_step"):
                assert self.network is not None

                # Sample batch
                indices = np.random.choice(
                    len(self.training_examples),
                    size=self.config.training_batch_size,
                    replace=False,
                )
                batch = [self.training_examples[i] for i in indices]

                # Convert to tensors
                spatial = torch.stack(
                    [torch.from_numpy(ex["spatial"]).float() for ex in batch]
                ).to(self.config.device)
                vector = torch.stack(
                    [torch.from_numpy(ex["vector"]).float() for ex in batch]
                ).to(self.config.device)
                mask = torch.stack(
                    [torch.from_numpy(ex["action_mask"]).float() for ex in batch]
                ).to(self.config.device)
                target_policy = torch.stack(
                    [torch.from_numpy(ex["policy"]).float() for ex in batch]
                ).to(self.config.device)
                target_value = torch.tensor([ex["value"] for ex in batch]).float().to(
                    self.config.device
                )

                # Forward pass
                self.network.train()
                policy_logits, value = self.network(spatial, vector, mask)

                # Compute loss
                loss, policy_loss, value_loss = compute_loss(
                    policy_logits, value, target_policy, target_value, mask
                )

                # Check for NaN loss
                if torch.isnan(loss):
                    raise ValueError("Training loss is NaN")

                # Backward pass (without optimizer step)
                loss.backward()

                # Check gradients
                grad_norm = 0.0
                for p in self.network.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm**0.5

                self.logger.info(f"Training step: loss={loss.item():.4f}")
                self.logger.info(f"  Policy loss: {policy_loss.item():.4f}")
                self.logger.info(f"  Value loss: {value_loss.item():.4f}")
                self.logger.info(f"  Gradient norm: {grad_norm:.4f}")

                # Zero gradients (cleanup)
                self.network.zero_grad()
                self.network.eval()

        except Exception as e:
            self.diagnostics.training_errors.append(str(e))
            self.logger.error(f"Training step failed: {e}", exc_info=True)

    def _generate_report(self) -> DebugReport:
        """Generate the final debug report."""
        d = self.diagnostics

        # Determine component status
        component_status = {
            "Environment": "OK",
            "Network": "OK" if len(d.network_errors) == 0 else "ERROR",
            "MCTS": "OK" if len(d.mcts_errors) == 0 else "ERROR",
            "Observations": "OK" if len(d.observation_errors) == 0 else "ERROR",
            "Actions": "OK" if len(d.action_errors) == 0 else "ERROR",
            "Replay Buffer": "OK" if len(d.buffer_errors) == 0 else "ERROR",
            "Training": "OK" if len(d.training_errors) == 0 else "ERROR",
        }

        # Add warnings
        if d.warnings:
            for component in component_status:
                if component_status[component] == "OK":
                    # Check if any warnings relate to this component
                    component_status[component] = "OK"  # Keep OK, warnings are minor

        # Generate recommendations
        recommendations = []

        if d.observation_errors:
            recommendations.append(
                "Check environment observation generation for shape/value issues"
            )

        if d.network_errors:
            recommendations.append(
                "Check network forward pass for NaN/Inf values in policy/value outputs"
            )

        if d.mcts_errors:
            recommendations.append(
                "Check MCTS search for policy probability normalization issues"
            )

        if d.action_errors:
            recommendations.append(
                "Check action selection and masking logic in MCTS"
            )

        if d.buffer_errors:
            recommendations.append(
                "Check training example generation and value assignment"
            )

        if d.training_errors:
            recommendations.append(
                "Check loss computation and gradient flow for numerical stability"
            )

        # Add termination analysis recommendations
        term_summary = self.termination_validator.get_summary()
        action_summary = self.action_tracer.get_summary()

        if term_summary.get("early_terminations", 0) > 0:
            recommendations.append(
                f"Early terminations detected: {term_summary['early_terminations']} games ended before turn 2. "
                "Check victory conditions and action execution."
            )

        if action_summary.get("invalid_action_count", 0) > 0:
            recommendations.append(
                f"Invalid actions detected: {action_summary['invalid_action_count']}. "
                "Check action masking in MCTS selection."
            )

        end_turn_rate = action_summary.get("end_turn_rate", 0)
        if end_turn_rate > 0.5:
            recommendations.append(
                f"High END_TURN rate: {end_turn_rate:.1%}. "
                "MCTS may be selecting end_turn too often."
            )

        # Log termination summary
        self.logger.info(f"\nTermination Summary: {term_summary}")
        self.logger.info(f"Action Summary: {action_summary}")

        if not recommendations:
            recommendations.append("No critical issues detected - system looks healthy")

        # Create config dict
        config_dict = {
            "num_games": self.config.num_games,
            "max_turns_per_game": self.config.max_turns_per_game,
            "board_size": self.config.board_size,
            "num_simulations": self.config.num_simulations,
            "use_simple_mcts": self.config.use_simple_mcts,
            "run_training_step": self.config.run_training_step,
        }

        return DebugReport(
            config=config_dict,
            diagnostics=d,
            profiler_summary=self.profiler.get_summary(),
            component_status=component_status,
            recommendations=recommendations,
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main() -> int:
    """Main entry point for debug self-play."""
    parser = argparse.ArgumentParser(description="Debug Self-Play for Civ6 AlphaZero")
    parser.add_argument("--num-games", type=int, default=3, help="Number of games")
    parser.add_argument("--max-turns", type=int, default=15, help="Max turns per game")
    parser.add_argument("--board-size", type=int, default=8, help="Board size")
    parser.add_argument("--num-simulations", type=int, default=10, help="MCTS simulations")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="DEBUG",
        help="Log level",
    )
    parser.add_argument("--log-dir", type=str, default="logs/debug_selfplay")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first error")
    parser.add_argument(
        "--no-training-step", action="store_true", help="Skip training step test"
    )
    parser.add_argument(
        "--use-network-mcts", action="store_true", help="Use network-guided MCTS"
    )
    parser.add_argument(
        "--output-json", type=str, default=None, help="Output JSON report path"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_dir=args.log_dir)

    # Build config from args (no hardcoded values)
    config = DebugSelfPlayConfig(
        num_games=args.num_games,
        max_turns_per_game=args.max_turns,
        board_size=args.board_size,
        num_simulations=args.num_simulations,
        log_level=args.log_level,
        log_dir=args.log_dir,
        fail_fast=args.fail_fast,
        run_training_step=not args.no_training_step,
        use_simple_mcts=not args.use_network_mcts,
        device=args.device,
    )

    # Run debug session
    runner = DebugSelfPlayRunner(config)
    report = runner.run()

    # Output results
    report.print_summary()

    if args.output_json:
        report.save_json(args.output_json)
        print(f"Report saved to: {args.output_json}")

    # Exit code based on errors
    return 1 if report.diagnostics.has_errors() else 0


if __name__ == "__main__":
    sys.exit(main())
