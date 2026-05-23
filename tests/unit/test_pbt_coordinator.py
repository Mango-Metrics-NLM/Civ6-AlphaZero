"""Tests for PBT Coordinator.

Tests cover:
- PBTProgress dataclass and to_dict
- PBTCoordinator initialization and configuration
- Training callbacks setup
- Initialize with and without checkpoint
- Training loop (run) with max_steps and max_generations
- Training step execution with callbacks
- Evaluation cycle
- Exploit/explore cycle
- Checkpoint save/load
- Progress tracking
- Shutdown and stop
- State machine transitions
- Factory function create_pbt_coordinator
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alpha_zero.pbt.config import PBTConfig
from alpha_zero.pbt.coordinator import (
    PBTCoordinator,
    PBTProgress,
    create_pbt_coordinator,
)
from alpha_zero.pbt.scheduler import EvaluationResult

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Create seeded RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def pbt_config(tmp_path: Path) -> PBTConfig:
    """Create a minimal PBT configuration for testing."""
    return PBTConfig(
        population_size=4,
        min_population_size=4,
        evaluation_interval=5,
        evaluation_games=2,
        selection_method="truncation",
        truncation_fraction=0.25,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_interval=10,
        convergence_window=3,
        convergence_threshold=0.001,
        wandb_enabled=False,
    )


@pytest.fixture
def coordinator(pbt_config: PBTConfig, rng: np.random.Generator) -> PBTCoordinator:
    """Create an initialized PBT coordinator."""
    coord = PBTCoordinator(pbt_config, rng=rng)
    coord.initialize()
    return coord


def make_train_step_fn() -> MagicMock:
    """Create a mock train step function."""
    return MagicMock(return_value={"loss": 0.5, "policy_loss": 0.3, "value_loss": 0.2})


def make_evaluate_fn(fitness: float = 0.6) -> MagicMock:
    """Create a mock evaluate function."""

    def _evaluate(agent_id: str) -> EvaluationResult:
        return EvaluationResult(
            agent_id=agent_id,
            fitness=fitness,
            games_played=10,
            wins=int(fitness * 10),
            losses=10 - int(fitness * 10),
        )

    return MagicMock(side_effect=_evaluate)


# =============================================================================
# PBTProgress Tests
# =============================================================================


class TestPBTProgress:
    """Tests for PBTProgress dataclass."""

    def test_default_values(self) -> None:
        """Test default progress values."""
        progress = PBTProgress()
        assert progress.current_step == 0
        assert progress.generation == 0
        assert progress.best_fitness == 0.0
        assert progress.mean_fitness == 0.0
        assert progress.population_size == 0
        assert progress.evaluations_completed == 0
        assert progress.exploit_explore_cycles == 0
        assert progress.elapsed_time == 0.0
        assert progress.timestamp is not None

    def test_custom_values(self) -> None:
        """Test custom progress values."""
        progress = PBTProgress(
            current_step=100,
            generation=5,
            best_fitness=0.85,
            mean_fitness=0.7,
            population_size=16,
            evaluations_completed=10,
            exploit_explore_cycles=3,
            elapsed_time=120.5,
        )
        assert progress.current_step == 100
        assert progress.generation == 5
        assert progress.best_fitness == 0.85
        assert progress.mean_fitness == 0.7

    def test_to_dict(self) -> None:
        """Test converting progress to dictionary."""
        progress = PBTProgress(
            current_step=50,
            generation=3,
            best_fitness=0.8,
            mean_fitness=0.65,
            population_size=8,
            evaluations_completed=5,
            exploit_explore_cycles=2,
            elapsed_time=60.0,
        )
        data = progress.to_dict()

        assert data["current_step"] == 50
        assert data["generation"] == 3
        assert data["best_fitness"] == 0.8
        assert data["mean_fitness"] == 0.65
        assert data["population_size"] == 8
        assert data["evaluations_completed"] == 5
        assert data["exploit_explore_cycles"] == 2
        assert data["elapsed_time"] == 60.0
        assert "timestamp" in data

    def test_to_dict_contains_all_fields(self) -> None:
        """Test that to_dict returns all expected keys."""
        progress = PBTProgress()
        data = progress.to_dict()

        expected_keys = {
            "current_step",
            "generation",
            "best_fitness",
            "mean_fitness",
            "population_size",
            "evaluations_completed",
            "exploit_explore_cycles",
            "elapsed_time",
            "timestamp",
        }
        assert set(data.keys()) == expected_keys


# =============================================================================
# PBTCoordinator Initialization Tests
# =============================================================================


class TestPBTCoordinatorInit:
    """Tests for PBTCoordinator initialization."""

    def test_basic_initialization(self, pbt_config: PBTConfig, rng: np.random.Generator) -> None:
        """Test basic initialization without calling initialize()."""
        coord = PBTCoordinator(pbt_config, rng=rng)

        assert coord.config is pbt_config
        assert coord.rng is rng
        assert coord.wandb_logger is None
        assert coord._current_step == 0
        assert not coord._running
        assert not coord._shutdown_requested
        assert coord.population is not None
        assert coord.scheduler is not None
        assert coord.mutation_operator is not None

    def test_initialization_with_wandb_logger(self, pbt_config: PBTConfig) -> None:
        """Test initialization with W&B logger."""
        mock_logger = MagicMock()
        coord = PBTCoordinator(pbt_config, wandb_logger=mock_logger)

        assert coord.wandb_logger is mock_logger

    def test_initialize_fresh_population(self, pbt_config: PBTConfig, rng: np.random.Generator) -> None:
        """Test initializing with a fresh population."""
        coord = PBTCoordinator(pbt_config, rng=rng)
        coord.initialize()

        assert coord.population.population_size == pbt_config.population_size
        assert coord._start_time is not None

    def test_initialize_from_checkpoint(
        self, pbt_config: PBTConfig, rng: np.random.Generator, tmp_path: Path
    ) -> None:
        """Test initializing from a checkpoint."""
        # First, create a checkpoint
        coord1 = PBTCoordinator(pbt_config, rng=rng)
        coord1.initialize()

        # Update some state
        agent_ids = coord1.population.get_agent_ids()
        for i, aid in enumerate(agent_ids):
            coord1.population.update_fitness(aid, 0.1 * (i + 1))
        coord1._current_step = 50

        checkpoint_path = tmp_path / "test_checkpoint"
        coord1._save_checkpoint()

        # Now load from checkpoint
        coord2 = PBTCoordinator(pbt_config, rng=np.random.default_rng(99))
        actual_checkpoint = tmp_path / "checkpoints" / "checkpoint_50"
        coord2.initialize(checkpoint_path=actual_checkpoint)

        assert coord2.population.population_size == pbt_config.population_size
        assert coord2._current_step == 50

    def test_initialize_nonexistent_checkpoint_creates_fresh(
        self, pbt_config: PBTConfig, rng: np.random.Generator
    ) -> None:
        """Test that nonexistent checkpoint path falls back to fresh init."""
        coord = PBTCoordinator(pbt_config, rng=rng)
        coord.initialize(checkpoint_path="/nonexistent/path/checkpoint")

        # Should have created a fresh population
        assert coord.population.population_size == pbt_config.population_size


# =============================================================================
# Training Callbacks Tests
# =============================================================================


class TestTrainingCallbacks:
    """Tests for training callback setup."""

    def test_set_training_callbacks(self, coordinator: PBTCoordinator) -> None:
        """Test setting all training callbacks."""
        train_fn = MagicMock()
        eval_fn = MagicMock()
        save_fn = MagicMock()
        load_fn = MagicMock()
        copy_fn = MagicMock()

        coordinator.set_training_callbacks(
            train_step_fn=train_fn,
            evaluate_fn=eval_fn,
            save_weights_fn=save_fn,
            load_weights_fn=load_fn,
            copy_weights_fn=copy_fn,
        )

        assert coordinator._train_step_fn is train_fn
        assert coordinator._evaluate_fn is eval_fn
        assert coordinator._save_weights_fn is save_fn
        assert coordinator._load_weights_fn is load_fn
        assert coordinator._copy_weights_fn is copy_fn

    def test_set_partial_callbacks(self, coordinator: PBTCoordinator) -> None:
        """Test setting only some callbacks."""
        train_fn = MagicMock()

        coordinator.set_training_callbacks(train_step_fn=train_fn)

        assert coordinator._train_step_fn is train_fn
        assert coordinator._evaluate_fn is None

    def test_callbacks_propagated_to_population(self, coordinator: PBTCoordinator) -> None:
        """Test that weight callbacks are propagated to population manager."""
        save_fn = MagicMock()
        load_fn = MagicMock()
        copy_fn = MagicMock()

        coordinator.set_training_callbacks(
            save_weights_fn=save_fn,
            load_weights_fn=load_fn,
            copy_weights_fn=copy_fn,
        )

        assert coordinator.population._save_weights_fn is save_fn
        assert coordinator.population._load_weights_fn is load_fn
        assert coordinator.population._copy_weights_fn is copy_fn


# =============================================================================
# Training Step Tests
# =============================================================================


class TestTrainingStep:
    """Tests for training step execution."""

    def test_run_training_step_with_callback(self, coordinator: PBTCoordinator) -> None:
        """Test that training step calls train_step_fn for each agent."""
        train_fn = make_train_step_fn()
        coordinator.set_training_callbacks(train_step_fn=train_fn)

        coordinator._run_training_step()

        # Should be called once for each agent
        assert train_fn.call_count == coordinator.population.population_size

    def test_run_training_step_no_callback(self, coordinator: PBTCoordinator) -> None:
        """Test training step does nothing without callback."""
        # No callbacks set
        coordinator._run_training_step()
        # Should not raise

    def test_run_training_step_increments_agent_steps(self, coordinator: PBTCoordinator) -> None:
        """Test that training step increments agent training_steps."""
        train_fn = make_train_step_fn()
        coordinator.set_training_callbacks(train_step_fn=train_fn)

        coordinator._run_training_step()

        for agent in coordinator.population:
            assert agent.training_steps == 1

    def test_run_training_step_handles_errors(self, coordinator: PBTCoordinator) -> None:
        """Test that training step handles exceptions gracefully."""
        error_fn = MagicMock(side_effect=RuntimeError("Training crashed"))
        coordinator.set_training_callbacks(train_step_fn=error_fn)

        # Should not raise
        coordinator._run_training_step()

    def test_run_training_step_partial_errors(self, coordinator: PBTCoordinator) -> None:
        """Test training step where some agents fail."""
        call_count = 0

        def partial_fail(agent_id: str, hyperparams: dict) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Agent 2 crashed")
            return {"loss": 0.5}

        coordinator.set_training_callbacks(train_step_fn=partial_fail)
        coordinator._run_training_step()

        # Should have attempted all agents
        assert call_count == coordinator.population.population_size


# =============================================================================
# Evaluation Tests
# =============================================================================


class TestEvaluation:
    """Tests for evaluation cycle."""

    def test_run_evaluation_with_callback(self, coordinator: PBTCoordinator) -> None:
        """Test evaluation updates fitness for all agents."""
        eval_fn = make_evaluate_fn(fitness=0.7)
        coordinator.set_training_callbacks(evaluate_fn=eval_fn)

        coordinator._run_evaluation()

        # All agents should have been evaluated
        assert eval_fn.call_count == coordinator.population.population_size

        # All agents should have updated fitness
        for agent in coordinator.population:
            assert len(agent.fitness_history) > 0

    def test_run_evaluation_records_to_scheduler(self, coordinator: PBTCoordinator) -> None:
        """Test evaluation records results to scheduler."""
        eval_fn = make_evaluate_fn(fitness=0.6)
        coordinator.set_training_callbacks(evaluate_fn=eval_fn)

        coordinator._run_evaluation()

        state = coordinator.scheduler.get_state()
        assert state.evaluation_count == 1

    def test_run_evaluation_no_callback_uses_scheduler(self, coordinator: PBTCoordinator) -> None:
        """Test evaluation uses scheduler evaluator when no callback set."""
        # No evaluate_fn set, so it should use scheduler's evaluator
        coordinator._run_evaluation()

        state = coordinator.scheduler.get_state()
        assert state.evaluation_count == 1

    def test_run_evaluation_handles_errors(self, coordinator: PBTCoordinator) -> None:
        """Test evaluation handles exceptions gracefully."""
        error_fn = MagicMock(side_effect=RuntimeError("Evaluation crashed"))
        coordinator.set_training_callbacks(evaluate_fn=error_fn)

        # Should not raise
        coordinator._run_evaluation()

    def test_run_evaluation_updates_stats_history(self, coordinator: PBTCoordinator) -> None:
        """Test evaluation updates stats history."""
        eval_fn = make_evaluate_fn(fitness=0.5)
        coordinator.set_training_callbacks(evaluate_fn=eval_fn)

        coordinator._run_evaluation()

        assert len(coordinator._stats_history) == 1


# =============================================================================
# Exploit/Explore Tests
# =============================================================================


class TestExploitExplore:
    """Tests for exploit/explore cycle."""

    def test_run_exploit_explore(self, coordinator: PBTCoordinator) -> None:
        """Test exploit/explore replaces weak agents."""
        # First set up fitness values
        agent_ids = coordinator.population.get_agent_ids()
        for i, aid in enumerate(agent_ids):
            coordinator.population.update_fitness(aid, 0.1 * (i + 1))

        initial_gen = coordinator.population.generation

        coordinator._run_exploit_explore()

        assert coordinator.population.generation == initial_gen + 1

    def test_run_exploit_explore_records_to_scheduler(self, coordinator: PBTCoordinator) -> None:
        """Test exploit/explore records to scheduler."""
        agent_ids = coordinator.population.get_agent_ids()
        for i, aid in enumerate(agent_ids):
            coordinator.population.update_fitness(aid, 0.1 * (i + 1))

        coordinator._run_exploit_explore()

        state = coordinator.scheduler.get_state()
        assert state.exploit_explore_count == 1

    def test_run_exploit_explore_increments_mutation_generation(
        self, coordinator: PBTCoordinator
    ) -> None:
        """Test exploit/explore increments mutation operator generation."""
        agent_ids = coordinator.population.get_agent_ids()
        for i, aid in enumerate(agent_ids):
            coordinator.population.update_fitness(aid, 0.1 * (i + 1))

        assert coordinator.mutation_operator._generation == 0
        coordinator._run_exploit_explore()
        assert coordinator.mutation_operator._generation == 1


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestCheckpoints:
    """Tests for checkpoint save/load."""

    def test_save_checkpoint(self, coordinator: PBTCoordinator, tmp_path: Path) -> None:
        """Test saving checkpoint creates expected files."""
        coordinator.config.checkpoint_dir = str(tmp_path / "ckpt")
        coordinator._current_step = 42

        coordinator._save_checkpoint()

        checkpoint_path = tmp_path / "ckpt" / "checkpoint_42"
        assert checkpoint_path.exists()
        assert (checkpoint_path / "coordinator_state.json").exists()
        assert (checkpoint_path / "metadata.json").exists()

    def test_save_checkpoint_state_contents(
        self, coordinator: PBTCoordinator, tmp_path: Path
    ) -> None:
        """Test saved checkpoint state contains correct data."""
        coordinator.config.checkpoint_dir = str(tmp_path / "ckpt")
        coordinator._current_step = 100
        coordinator._start_time = 1000.0

        coordinator._save_checkpoint()

        checkpoint_path = tmp_path / "ckpt" / "checkpoint_100"
        with open(checkpoint_path / "coordinator_state.json") as f:
            state = json.load(f)

        assert state["current_step"] == 100
        assert state["start_time"] == 1000.0
        assert "config" in state

    def test_save_checkpoint_updates_last_checkpoint_step(
        self, coordinator: PBTCoordinator, tmp_path: Path
    ) -> None:
        """Test that saving updates last_checkpoint_step."""
        coordinator.config.checkpoint_dir = str(tmp_path / "ckpt")
        coordinator._current_step = 75

        coordinator._save_checkpoint()

        assert coordinator._last_checkpoint_step == 75

    def test_load_checkpoint(self, coordinator: PBTCoordinator, tmp_path: Path) -> None:
        """Test loading checkpoint restores state."""
        coordinator.config.checkpoint_dir = str(tmp_path / "ckpt")
        coordinator._current_step = 200
        coordinator._last_checkpoint_step = 100
        coordinator._start_time = 500.0

        coordinator._save_checkpoint()

        # Create a new coordinator and load
        new_coord = PBTCoordinator(coordinator.config, rng=np.random.default_rng(99))
        checkpoint_path = tmp_path / "ckpt" / "checkpoint_200"
        new_coord.load_checkpoint(checkpoint_path)

        assert new_coord._current_step == 200
        assert new_coord._last_checkpoint_step == 100


# =============================================================================
# Training Loop Tests
# =============================================================================


class TestTrainingLoop:
    """Tests for the main training loop (run method)."""

    def test_run_with_max_steps(self, coordinator: PBTCoordinator, tmp_path: Path) -> None:
        """Test run stops at max_steps."""
        coordinator.config.checkpoint_dir = str(tmp_path / "ckpt")
        train_fn = make_train_step_fn()
        coordinator.set_training_callbacks(train_step_fn=train_fn)

        results = coordinator.run(max_steps=3)

        assert results["total_steps"] == 3
        assert "best_fitness" in results
        assert "elapsed_time" in results

    def test_run_with_max_generations(self, coordinator: PBTCoordinator, tmp_path: Path) -> None:
        """Test run stops at max_generations."""
        coordinator.config.checkpoint_dir = str(tmp_path / "ckpt")
        coordinator.config.evaluation_interval = 1  # Evaluate every step

        # Set up fitness so exploit/explore can work
        for i, aid in enumerate(coordinator.population.get_agent_ids()):
            coordinator.population.update_fitness(aid, 0.1 * (i + 1))

        train_fn = make_train_step_fn()
        eval_fn = make_evaluate_fn(fitness=0.5)
        coordinator.set_training_callbacks(train_step_fn=train_fn, evaluate_fn=eval_fn)

        results = coordinator.run(max_steps=1000, max_generations=2)

        # Should have stopped when reaching max_generations
        assert results["total_generations"] <= 2

    def test_run_auto_initializes(self, pbt_config: PBTConfig, rng: np.random.Generator, tmp_path: Path) -> None:
        """Test that run initializes population if not already done."""
        pbt_config.checkpoint_dir = str(tmp_path / "ckpt")
        coord = PBTCoordinator(pbt_config, rng=rng)
        # Don't call initialize()

        results = coord.run(max_steps=1)

        assert coord.population.population_size == pbt_config.population_size
        assert results["total_steps"] == 1

    def test_run_returns_best_agent_info(self, coordinator: PBTCoordinator, tmp_path: Path) -> None:
        """Test run returns information about best agent."""
        coordinator.config.checkpoint_dir = str(tmp_path / "ckpt")

        # Set up some fitness
        for i, aid in enumerate(coordinator.population.get_agent_ids()):
            coordinator.population.update_fitness(aid, 0.1 * (i + 1))

        results = coordinator.run(max_steps=1)

        assert "best_agent_id" in results
        assert "best_hyperparameters" in results
        assert results["best_agent_id"] is not None
        assert isinstance(results["best_hyperparameters"], dict)

    def test_run_stop_request(self, coordinator: PBTCoordinator, tmp_path: Path) -> None:
        """Test that stop() causes run to exit."""
        coordinator.config.checkpoint_dir = str(tmp_path / "ckpt")

        # Set up a train function that requests stop after a few iterations
        call_count = 0

        def train_and_stop(agent_id: str, hyperparams: dict) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count > 5:
                coordinator.stop()
            return {"loss": 0.5}

        coordinator.set_training_callbacks(train_step_fn=train_and_stop)

        results = coordinator.run(max_steps=1000)

        # Should have stopped early
        assert results["total_steps"] < 1000

    def test_run_convergence_detection(self, coordinator: PBTCoordinator, tmp_path: Path) -> None:
        """Test that run stops on convergence."""
        coordinator.config.checkpoint_dir = str(tmp_path / "ckpt")
        coordinator.config.convergence_window = 3
        coordinator.config.convergence_threshold = 1.0  # Very easy to converge

        # Pre-fill stats history with converged data
        from alpha_zero.pbt.population import PopulationStats

        for _ in range(5):
            stats = PopulationStats(std_fitness=0.0)  # Zero variance = converged
            coordinator.population.stats_history.append(stats)

        results = coordinator.run(max_steps=1000)

        # Should have detected convergence and stopped
        assert results["total_steps"] < 1000


# =============================================================================
# Progress and Properties Tests
# =============================================================================


class TestProgressAndProperties:
    """Tests for progress tracking and properties."""

    def test_get_progress_initial(self, coordinator: PBTCoordinator) -> None:
        """Test getting initial progress."""
        progress = coordinator.get_progress()

        assert isinstance(progress, PBTProgress)
        assert progress.current_step == 0
        assert progress.generation == 0
        assert progress.population_size == coordinator.config.population_size

    def test_get_progress_after_steps(self, coordinator: PBTCoordinator, tmp_path: Path) -> None:
        """Test progress after some training steps."""
        coordinator.config.checkpoint_dir = str(tmp_path / "ckpt")
        train_fn = make_train_step_fn()
        coordinator.set_training_callbacks(train_step_fn=train_fn)

        coordinator.run(max_steps=3)

        progress = coordinator.get_progress()
        assert progress.current_step == 3

    def test_get_progress_empty_population(self, pbt_config: PBTConfig) -> None:
        """Test progress with uninitialized population."""
        coord = PBTCoordinator(pbt_config)
        progress = coord.get_progress()

        assert progress.population_size == 0
        assert progress.best_fitness == 0.0
        assert progress.mean_fitness == 0.0

    def test_is_running_property(self, coordinator: PBTCoordinator) -> None:
        """Test is_running property."""
        assert not coordinator.is_running

    def test_current_step_property(self, coordinator: PBTCoordinator) -> None:
        """Test current_step property."""
        assert coordinator.current_step == 0

        coordinator._current_step = 42
        assert coordinator.current_step == 42

    def test_stop_method(self, coordinator: PBTCoordinator) -> None:
        """Test stop method sets shutdown flag."""
        coordinator.stop()
        assert coordinator._shutdown_requested


# =============================================================================
# Final Results Tests
# =============================================================================


class TestFinalResults:
    """Tests for computing final results."""

    def test_compute_final_results(self, coordinator: PBTCoordinator) -> None:
        """Test computing final results."""
        coordinator._start_time = 1000.0

        # Set up fitness
        for i, aid in enumerate(coordinator.population.get_agent_ids()):
            coordinator.population.update_fitness(aid, 0.1 * (i + 1))

        results = coordinator._compute_final_results()

        assert "total_steps" in results
        assert "total_generations" in results
        assert "elapsed_time" in results
        assert "best_agent_id" in results
        assert "best_fitness" in results
        assert "mean_fitness" in results
        assert "best_hyperparameters" in results
        assert "final_population_size" in results

    def test_compute_final_results_empty_population(self, pbt_config: PBTConfig) -> None:
        """Test final results with empty population."""
        coord = PBTCoordinator(pbt_config)
        coord._start_time = 1000.0

        results = coord._compute_final_results()

        assert results["best_agent_id"] is None
        assert results["best_hyperparameters"] == {}


# =============================================================================
# Logging Tests
# =============================================================================


class TestLogging:
    """Tests for logging methods (no-op without wandb)."""

    def test_log_training_metrics_no_wandb(self, coordinator: PBTCoordinator) -> None:
        """Test logging training metrics without wandb logger."""
        # wandb_logger is None, should be no-op
        coordinator._log_training_metrics("agent_0", {"loss": 0.5})

    def test_log_evaluation_results_no_wandb(self, coordinator: PBTCoordinator) -> None:
        """Test logging evaluation results without wandb logger."""
        from alpha_zero.pbt.population import PopulationStats

        results = [EvaluationResult(agent_id="a", fitness=0.5)]
        stats = PopulationStats(mean_fitness=0.5, max_fitness=0.5)

        coordinator._log_evaluation_results(results, stats)

    def test_log_exploit_explore_no_wandb(self, coordinator: PBTCoordinator) -> None:
        """Test logging exploit/explore without wandb logger."""
        changes = [("weak_0", "strong_0", {"lr": 0.001})]
        coordinator._log_exploit_explore(changes)

    def test_log_training_metrics_with_wandb(self, pbt_config: PBTConfig) -> None:
        """Test logging metrics with wandb mock."""
        mock_wandb = MagicMock()
        coord = PBTCoordinator(pbt_config, wandb_logger=mock_wandb)
        coord.initialize()

        with patch.dict("sys.modules", {"wandb": MagicMock()}):
            coord._log_training_metrics("agent_0", {"loss": 0.5})

    def test_log_evaluation_results_with_wandb(self, pbt_config: PBTConfig) -> None:
        """Test logging evaluation results with wandb mock."""
        from alpha_zero.pbt.population import PopulationStats

        mock_wandb = MagicMock()
        coord = PBTCoordinator(pbt_config, wandb_logger=mock_wandb)

        results = [EvaluationResult(agent_id="a", fitness=0.5)]
        stats = PopulationStats(
            mean_fitness=0.5,
            max_fitness=0.5,
            min_fitness=0.5,
            std_fitness=0.0,
            hyperparameter_stats={"lr": {"mean": 0.001, "std": 0.0001, "min": 0.0005, "max": 0.002}},
        )

        with patch.dict("sys.modules", {"wandb": MagicMock()}):
            coord._log_evaluation_results(results, stats)

    def test_log_exploit_explore_with_wandb(self, pbt_config: PBTConfig) -> None:
        """Test logging exploit/explore with wandb mock."""
        mock_wandb = MagicMock()
        coord = PBTCoordinator(pbt_config, wandb_logger=mock_wandb)

        changes = [("weak_0", "strong_0", {"lr": 0.001})]

        with patch.dict("sys.modules", {"wandb": MagicMock()}):
            coord._log_exploit_explore(changes)


# =============================================================================
# Signal Handling Tests
# =============================================================================


class TestSignalHandling:
    """Tests for signal handling."""

    def test_setup_signal_handlers(self, coordinator: PBTCoordinator) -> None:
        """Test signal handler setup doesn't raise."""
        coordinator._setup_signal_handlers()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreatePBTCoordinator:
    """Tests for create_pbt_coordinator factory function."""

    def test_default_creation(self) -> None:
        """Test creating coordinator with defaults."""
        coord = create_pbt_coordinator()

        assert coord.config.population_size == 16
        assert coord.config.evaluation_interval == 1000
        assert coord.config.evaluation_games == 20
        assert coord.config.mutation_strength == 0.2

    def test_custom_creation(self) -> None:
        """Test creating coordinator with custom values."""
        coord = create_pbt_coordinator(
            population_size=8,
            evaluation_interval=500,
            evaluation_games=10,
            mutation_strength=0.3,
        )

        assert coord.config.population_size == 8
        assert coord.config.evaluation_interval == 500
        assert coord.config.evaluation_games == 10
        assert coord.config.mutation_strength == 0.3

    def test_factory_with_extra_kwargs(self) -> None:
        """Test factory passes extra kwargs to PBTConfig."""
        coord = create_pbt_coordinator(
            population_size=8,
            truncation_fraction=0.3,
            selection_method="tournament",
        )

        assert coord.config.truncation_fraction == 0.3
        assert coord.config.selection_method == "tournament"


# =============================================================================
# Integration Tests
# =============================================================================


class TestCoordinatorIntegration:
    """Integration tests for coordinator workflow."""

    def test_full_workflow(self, pbt_config: PBTConfig, rng: np.random.Generator, tmp_path: Path) -> None:
        """Test a complete PBT workflow: init -> train -> evaluate -> exploit/explore."""
        pbt_config.checkpoint_dir = str(tmp_path / "ckpt")
        pbt_config.evaluation_interval = 2
        pbt_config.checkpoint_interval = 100  # Don't checkpoint during test
        pbt_config.convergence_threshold = 0.0001  # Avoid false convergence
        pbt_config.convergence_window = 100  # Large window to avoid convergence

        coord = PBTCoordinator(pbt_config, rng=rng)
        coord.initialize()

        # Set up callbacks with varying fitness to avoid convergence
        call_counter = 0

        def varying_eval(agent_id: str) -> EvaluationResult:
            nonlocal call_counter
            call_counter += 1
            fitness = 0.5 + 0.01 * call_counter
            return EvaluationResult(
                agent_id=agent_id,
                fitness=fitness,
                games_played=10,
                wins=int(fitness * 10),
                losses=10 - int(fitness * 10),
            )

        train_fn = make_train_step_fn()

        coord.set_training_callbacks(
            train_step_fn=train_fn,
            evaluate_fn=varying_eval,
        )

        # Run a few steps
        results = coord.run(max_steps=8)

        # Verify results
        assert results["total_steps"] == 8
        assert results["final_population_size"] == pbt_config.population_size
        assert results["elapsed_time"] > 0

    def test_train_evaluate_exploit_sequence(self, coordinator: PBTCoordinator) -> None:
        """Test manual sequence of train -> evaluate -> exploit/explore."""
        # Set fitness
        for i, aid in enumerate(coordinator.population.get_agent_ids()):
            coordinator.population.update_fitness(aid, 0.1 * (i + 1))

        # Manual training step
        train_fn = make_train_step_fn()
        coordinator.set_training_callbacks(train_step_fn=train_fn)
        coordinator._run_training_step()

        # Manual evaluation
        eval_fn = make_evaluate_fn(fitness=0.7)
        coordinator.set_training_callbacks(
            train_step_fn=train_fn, evaluate_fn=eval_fn
        )
        coordinator._run_evaluation()

        # Manual exploit/explore
        coordinator._run_exploit_explore()

        assert coordinator.population.generation == 1
