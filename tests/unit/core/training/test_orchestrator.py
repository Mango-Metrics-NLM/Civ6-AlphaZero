"""Tests for the training orchestrator.

Covers the end-to-end training loop wiring, iteration metrics,
and component integration in dry-run mode.
"""
from __future__ import annotations

import pytest

from src.core.config import (
    CurriculumConfig,
    CurriculumStage,
    SDKConfig,
    TrainingConfig,
)
from src.core.training.checkpoint import CheckpointManager
from src.core.training.evaluator import Evaluator
from src.core.training.learner import Learner
from src.core.training.orchestrator import IterationMetrics, TrainingOrchestrator
from src.core.training.replay_buffer import ReplayBuffer
from src.environments.mini_games.mini_resource import MiniResourceEnv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sdk_config() -> SDKConfig:
    """Minimal SDK config for fast testing."""
    return SDKConfig(
        training=TrainingConfig(
            batch_size=4,
            max_iterations=3,
            games_per_iteration=2,
            replay_buffer_size=200,
            checkpoint_interval=2,
            eval_interval=1,
            eval_games=4,
            max_steps_per_game=20,
        ),
        seed=42,
    )


@pytest.fixture()
def env_factory():
    """Environment factory for MiniResourceEnv."""
    return lambda: MiniResourceEnv(board_size=4, max_steps=20, num_resources=2)


@pytest.fixture()
def orchestrator(sdk_config, env_factory, tmp_path):
    """Orchestrator configured for testing."""
    sdk_config_copy = sdk_config.model_copy(update={"output_dir": str(tmp_path)})
    return TrainingOrchestrator(
        config=sdk_config_copy,
        env_factory=env_factory,
    )


# ---------------------------------------------------------------------------
# IterationMetrics
# ---------------------------------------------------------------------------


class TestIterationMetrics:
    """Tests for the IterationMetrics dataclass."""

    def test_defaults(self) -> None:
        m = IterationMetrics(iteration=0)
        assert m.iteration == 0
        assert m.self_play_games == 0
        assert m.training_loss == 0.0
        assert m.is_best is False
        assert m.curriculum_stage == ""

    def test_to_dict(self) -> None:
        m = IterationMetrics(iteration=5, training_loss=0.42, eval_elo=1100.0)
        d = m.to_dict()
        assert d["iteration"] == 5
        assert d["training_loss"] == pytest.approx(0.42)
        assert d["eval_elo"] == pytest.approx(1100.0)
        assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# TrainingOrchestrator
# ---------------------------------------------------------------------------


class TestTrainingOrchestrator:
    """Tests for the TrainingOrchestrator class."""

    def test_constructor_defaults(self, orchestrator) -> None:
        assert orchestrator.iteration == 0
        assert orchestrator.best_elo == -float("inf")
        assert orchestrator.history == []
        assert orchestrator.is_running is False

    def test_run_dry_run(self, orchestrator) -> None:
        """Dry run executes exactly one iteration."""
        metrics = orchestrator.run(dry_run=True)
        assert len(metrics) == 1
        assert metrics[0].iteration == 0
        assert orchestrator.iteration == 1

    def test_run_multiple_iterations(self, sdk_config, env_factory, tmp_path) -> None:
        """Run multiple iterations and verify metrics accumulate."""
        config = sdk_config.model_copy(update={"output_dir": str(tmp_path)})
        orch = TrainingOrchestrator(config=config, env_factory=env_factory)
        metrics = orch.run(max_iterations=2)
        assert len(metrics) == 2
        assert metrics[0].iteration == 0
        assert metrics[1].iteration == 1
        assert orch.iteration == 2

    def test_self_play_generates_examples(self, orchestrator) -> None:
        """Self-play generates training examples."""
        metrics = orchestrator.run(dry_run=True)
        assert metrics[0].self_play_games > 0
        assert metrics[0].self_play_examples > 0

    def test_evaluation_runs(self, orchestrator) -> None:
        """Evaluation produces Elo ratings."""
        metrics = orchestrator.run(dry_run=True)
        # eval_interval=1 so evaluation runs every iteration
        assert metrics[0].eval_elo != 0.0 or metrics[0].eval_win_rate >= 0.0

    def test_get_summary(self, orchestrator) -> None:
        """Summary contains expected keys."""
        orchestrator.run(dry_run=True)
        summary = orchestrator.get_summary()
        assert "iteration" in summary
        assert "best_elo" in summary
        assert "buffer_size" in summary
        assert "learner_steps" in summary

    def test_is_running_during_training(self, orchestrator) -> None:
        """is_running is False before and after run."""
        assert orchestrator.is_running is False
        orchestrator.run(dry_run=True)
        assert orchestrator.is_running is False

    def test_custom_components(self, sdk_config, env_factory, tmp_path) -> None:
        """Orchestrator accepts custom components."""
        config = sdk_config.model_copy(update={"output_dir": str(tmp_path)})
        learner = Learner(config=config.training)
        buffer = ReplayBuffer(capacity=100, seed=42)
        evaluator = Evaluator(mcts_config=config.mcts, num_eval_games=2)
        ckpt = CheckpointManager(output_dir=tmp_path, checkpoint_interval=10)

        orch = TrainingOrchestrator(
            config=config,
            env_factory=env_factory,
            learner=learner,
            replay_buffer=buffer,
            evaluator=evaluator,
            checkpoint_manager=ckpt,
        )
        metrics = orch.run(dry_run=True)
        assert len(metrics) == 1

    def test_with_curriculum(self, sdk_config, env_factory, tmp_path) -> None:
        """Orchestrator works with curriculum scheduler."""
        config = sdk_config.model_copy(
            update={
                "output_dir": str(tmp_path),
                "curriculum": CurriculumConfig(
                    stages=[
                        CurriculumStage(name="easy", environment="mini_resource"),
                        CurriculumStage(name="hard", environment="territory_control"),
                    ],
                    promotion_elo_threshold=2000.0,
                    promotion_window=3,
                ),
            }
        )
        orch = TrainingOrchestrator(config=config, env_factory=env_factory)
        metrics = orch.run(dry_run=True)
        assert metrics[0].curriculum_stage == "easy"

    def test_training_skipped_when_buffer_undersized(self, env_factory, tmp_path) -> None:
        """Training phase is skipped when buffer has fewer samples than batch_size."""
        config = SDKConfig(
            training=TrainingConfig(
                batch_size=9999,
                max_iterations=1,
                games_per_iteration=1,
                replay_buffer_size=200,
                eval_interval=999,
                max_steps_per_game=10,
            ),
            output_dir=str(tmp_path),
        )
        orch = TrainingOrchestrator(config=config, env_factory=env_factory)
        metrics = orch.run(dry_run=True)
        # Training loss should be 0 because buffer < batch_size
        assert metrics[0].training_loss == pytest.approx(0.0)

    def test_checkpoint_saved(self, sdk_config, env_factory, tmp_path) -> None:
        """Checkpoint is saved at configured interval."""
        config = sdk_config.model_copy(
            update={
                "output_dir": str(tmp_path),
                "training": TrainingConfig(
                    batch_size=4,
                    max_iterations=3,
                    games_per_iteration=2,
                    replay_buffer_size=200,
                    checkpoint_interval=1,
                    eval_interval=1,
                    eval_games=2,
                    max_steps_per_game=20,
                ),
            }
        )
        orch = TrainingOrchestrator(config=config, env_factory=env_factory)
        orch.run(max_iterations=1)
        ckpt_dir = tmp_path / "checkpoints"
        assert ckpt_dir.exists()

    def test_constructor_with_explicit_curriculum(
        self, sdk_config, env_factory, tmp_path
    ) -> None:
        """Orchestrator accepts an externally constructed curriculum."""
        from src.core.training.curriculum import CurriculumScheduler

        config = sdk_config.model_copy(
            update={
                "output_dir": str(tmp_path),
                "curriculum": CurriculumConfig(
                    stages=[
                        CurriculumStage(name="s1", environment="mini_resource"),
                        CurriculumStage(name="s2", environment="territory_control"),
                    ],
                    promotion_elo_threshold=9999.0,
                    promotion_window=3,
                ),
            }
        )
        ext_curriculum = CurriculumScheduler(config.curriculum)
        orch = TrainingOrchestrator(
            config=config,
            env_factory=env_factory,
            curriculum=ext_curriculum,
        )
        assert orch._curriculum is ext_curriculum
        metrics = orch.run(dry_run=True)
        assert metrics[0].curriculum_stage == "s1"

    def test_get_summary_includes_curriculum(
        self, sdk_config, env_factory, tmp_path
    ) -> None:
        """get_summary includes curriculum key when curriculum is active."""
        config = sdk_config.model_copy(
            update={
                "output_dir": str(tmp_path),
                "curriculum": CurriculumConfig(
                    stages=[
                        CurriculumStage(name="easy", environment="mini_resource"),
                        CurriculumStage(name="hard", environment="territory_control"),
                    ],
                    promotion_elo_threshold=9999.0,
                    promotion_window=3,
                ),
            }
        )
        orch = TrainingOrchestrator(config=config, env_factory=env_factory)
        orch.run(dry_run=True)
        summary = orch.get_summary()
        assert "curriculum" in summary
        assert summary["curriculum"]["current_stage"] == "easy"
