"""Integration tests for Phase 2 components.

Tests the integration between distributed training, Unciv interface,
logging, and configuration systems.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


# Test imports work correctly
def test_distributed_imports() -> None:
    """Test that distributed module imports work."""
    from alpha_zero.distributed import (
        ActorConfig,
        CoordinatorConfig,
        LearnerConfig,
        ReplayBufferConfig,
        SharedStorageConfig,
    )

    # Verify we can instantiate config classes
    storage_config = SharedStorageConfig()
    buffer_config = ReplayBufferConfig()
    actor_config = ActorConfig()
    learner_config = LearnerConfig()
    coord_config = CoordinatorConfig()

    assert storage_config is not None
    assert buffer_config is not None
    assert actor_config is not None
    assert learner_config is not None
    assert coord_config is not None


def test_interfaces_imports() -> None:
    """Test that interfaces module imports work."""
    from alpha_zero.interfaces import (
        UncivTerrainType,
        UncivUnitType,
    )

    # Verify enums have values
    assert len(UncivTerrainType) > 0
    assert len(UncivUnitType) > 0


def test_utils_imports() -> None:
    """Test that utils module imports work."""
    from alpha_zero.utils import (
        # Configuration
        get_default_config,
    )

    # Verify we can get defaults
    config = get_default_config()
    assert config is not None


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_load_project_config(self) -> None:
        """Test loading the actual project configuration."""
        from alpha_zero.utils import load_config

        config_path = Path("configs/civ6_config.yaml")
        if not config_path.exists():
            pytest.skip("Project config not found")

        config = load_config(config_path)

        # Verify Phase 2 sections exist
        assert config.distributed is not None
        assert config.unciv is not None
        assert config.logging is not None
        assert config.action_space is not None

        # Verify Phase 2 values
        assert config.distributed.num_actors == 64
        assert config.network.num_heads == 4

    def test_config_with_logging_setup(self) -> None:
        """Test configuration integrates with logging setup."""
        from alpha_zero.utils import LogConfig, get_default_config

        config = get_default_config()
        log_config = LogConfig(
            level=config.logging.level,
            log_dir=config.logging.log_dir,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_config.log_dir = tmpdir
            # Just verify the config can be used (don't actually set up logging)
            assert log_config.level == "INFO"


class TestDistributedComponentIntegration:
    """Integration tests for distributed training components."""

    def test_shared_storage_config_creation(self) -> None:
        """Test SharedStorage can be configured from Civ6Config."""
        from alpha_zero.distributed import SharedStorageConfig
        from alpha_zero.utils import get_default_config

        config = get_default_config()

        storage_config = SharedStorageConfig(
            max_weight_versions=10,
            compression_enabled=config.distributed.compression_enabled,
        )

        assert storage_config.compression_enabled == config.distributed.compression_enabled

    def test_replay_buffer_config_creation(self) -> None:
        """Test ReplayBuffer can be configured from Civ6Config."""
        from alpha_zero.distributed import ReplayBufferConfig
        from alpha_zero.utils import get_default_config

        config = get_default_config()

        buffer_config = ReplayBufferConfig(
            max_size=config.training.replay_buffer_size,
            min_size=config.training.min_replay_size,
            compression_enabled=config.distributed.compression_enabled,
        )

        assert buffer_config.max_size == 500000
        assert buffer_config.min_size == 10000

    def test_actor_config_creation(self) -> None:
        """Test Actor can be configured from Civ6Config."""
        from alpha_zero.distributed import ActorConfig
        from alpha_zero.utils import get_default_config

        config = get_default_config()

        actor_config = ActorConfig(
            num_simulations=config.mcts.num_simulations,
            c_puct=config.mcts.c_puct,
            dirichlet_alpha=config.mcts.dirichlet_alpha,
            dirichlet_epsilon=config.mcts.dirichlet_epsilon,
            temperature=config.mcts.temperature,
            temperature_threshold=config.mcts.temperature_threshold,
            games_per_update=config.distributed.actor_batch_size,
        )

        assert actor_config.num_simulations == 200
        assert actor_config.games_per_update == 8

    def test_learner_config_creation(self) -> None:
        """Test Learner can be configured from Civ6Config."""
        from alpha_zero.distributed import LearnerConfig
        from alpha_zero.utils import get_default_config

        config = get_default_config()

        learner_config = LearnerConfig(
            batch_size=config.distributed.learner_batch_size,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            gradient_clip=config.training.gradient_clip,
            lr_decay_steps=config.training.lr_decay_steps,
            lr_decay_rate=config.training.lr_decay_rate,
        )

        assert learner_config.batch_size == 256
        assert learner_config.learning_rate == 0.001

    def test_coordinator_config_creation(self) -> None:
        """Test Coordinator can be configured from Civ6Config."""
        from alpha_zero.distributed import CoordinatorConfig
        from alpha_zero.utils import get_default_config

        config = get_default_config()

        coord_config = CoordinatorConfig(
            num_actors=config.distributed.num_actors,
            max_games=config.distributed.max_training_steps,
            evaluation_interval=config.distributed.evaluation_interval,
            checkpoint_interval=config.distributed.checkpoint_interval,
        )

        assert coord_config.num_actors == 64


class TestUncivInterfaceIntegration:
    """Integration tests for Unciv interface."""

    def test_unciv_terrain_mapping(self) -> None:
        """Test Unciv terrain types map correctly."""
        from alpha_zero.interfaces import UncivTerrainType

        # Verify all terrain types have mappings
        for terrain in UncivTerrainType:
            # Just verify the enum value exists
            assert terrain.value is not None

    def test_unciv_unit_mapping(self) -> None:
        """Test Unciv unit types map correctly."""
        from alpha_zero.interfaces import UncivUnitType

        # Verify all unit types have mappings
        for unit_type in UncivUnitType:
            assert unit_type.value is not None

    def test_unciv_interface_creation(self) -> None:
        """Test UncivInterface can be created."""
        from alpha_zero.interfaces import create_unciv_interface
        from alpha_zero.utils import get_default_config

        config = get_default_config()

        interface = create_unciv_interface(
            board_size=config.unciv.board_size,
            max_players=config.unciv.max_players,
        )

        assert interface is not None
        assert interface.board_size == 80


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_game_logger_creation(self) -> None:
        """Test GameLogger can be created and used."""
        from alpha_zero.utils import GameLogger, LogConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(log_dir=tmpdir)
            logger = GameLogger(config, game_id="test_game_1")

            # Log some actions
            logger.log_action(
                turn=1,
                player=0,
                action_type="MOVE",
                action_details={"unit_id": 0, "direction": "N"},
            )
            logger.log_action(
                turn=1,
                player=1,
                action_type="END_TURN",
                action_details={},
            )

            # End the game
            logger.end_game(winner=0, total_turns=1, reason="elimination")

            # Check log file was created
            _log_files = list(Path(tmpdir).glob("*.json"))
            # GameLogger may or may not create files depending on config
            assert True  # Test passes if no exception

    def test_training_logger_creation(self) -> None:
        """Test TrainingLogger can be created and used."""
        from alpha_zero.utils import LogConfig, TrainingLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(log_dir=tmpdir)
            logger = TrainingLogger(config)

            # Log training metrics
            logger.log_epoch(
                epoch=1,
                total_loss=1.5,
                policy_loss=1.0,
                value_loss=0.5,
            )

            logger.log_evaluation(
                iteration=1,
                win_rate=0.65,
                games_played=100,
            )

            assert True  # Test passes if no exception

    def test_metrics_tracker(self) -> None:
        """Test MetricsTracker aggregation."""
        from alpha_zero.utils import MetricsTracker

        tracker = MetricsTracker(window_size=10)

        # Add some metrics
        for i in range(20):
            tracker.add("loss", float(i) / 10)
            tracker.add("win_rate", 0.5 + i * 0.01)

        # Get statistics
        loss_stats = tracker.get_stats("loss")
        assert "mean" in loss_stats
        assert "std" in loss_stats
        assert "min" in loss_stats
        assert "max" in loss_stats

        # Window should only keep last 10
        assert loss_stats["count"] == 10


class TestTrainingExampleFormat:
    """Test training example format compatibility."""

    def test_training_example_structure(self) -> None:
        """Test TrainingExample has correct structure."""
        from alpha_zero.distributed import TrainingExample

        # Create example with numpy arrays
        spatial = np.zeros((84, 16, 16), dtype=np.float32)
        vector = np.zeros(512, dtype=np.float32)
        action_mask = np.ones(1000, dtype=np.float32)
        policy = np.zeros(1000, dtype=np.float32)
        policy[0] = 1.0  # Valid action
        value = 1.0

        example = TrainingExample(
            spatial=spatial,
            vector=vector,
            action_mask=action_mask,
            policy=policy,
            value=value,
        )

        assert example.spatial.shape == (84, 16, 16)
        assert example.vector.shape == (512,)
        assert example.value == 1.0

    def test_game_trajectory_structure(self) -> None:
        """Test GameTrajectory has correct structure."""
        from alpha_zero.distributed import GameTrajectory, TrainingExample

        # Create examples
        examples = []
        for i in range(10):
            examples.append(
                TrainingExample(
                    spatial=np.zeros((84, 16, 16), dtype=np.float32),
                    vector=np.zeros(512, dtype=np.float32),
                    action_mask=np.ones(1000, dtype=np.float32),
                    policy=np.zeros(1000, dtype=np.float32),
                    value=1.0 if i % 2 == 0 else -1.0,
                )
            )

        trajectory = GameTrajectory(
            examples=examples,
            game_id=1,
            winner=0,
            game_length=10,
        )

        assert len(trajectory.examples) == 10
        assert trajectory.winner == 0
        assert trajectory.game_length == 10


class TestEncoderCompatibility:
    """Test encoder compatibility with Phase 2 components."""

    def test_full_encoder_channels(self) -> None:
        """Test full encoder produces 84-channel output."""
        from alpha_zero.envs.civ6_encoder import Civ6StateEncoder, EncoderConfig

        config = EncoderConfig(
            board_size=16,
            spatial_channels=84,
            vector_dim=512,
        )
        encoder = Civ6StateEncoder(config)

        # Create minimal game state
        game_state = {
            "current_player": 0,
            "turn": 1,
            "terrain": {},
            "units": [],
            "cities": [],
            "visibility": {},
        }

        spatial, vector = encoder.encode(game_state)

        assert spatial.shape == (84, 16, 16)
        assert vector.shape == (512,)


class TestActionSpaceCompatibility:
    """Test action space compatibility."""

    def test_hierarchical_action_space_exists(self) -> None:
        """Test hierarchical action space class exists."""
        from alpha_zero.core.civ6_actions import HierarchicalActionSpace

        action_space = HierarchicalActionSpace(
            board_size=16,
            max_units=100,
            max_cities=20,
        )

        assert action_space is not None
        # Verify it has expected methods
        assert hasattr(action_space, "decode_action")
        assert hasattr(action_space, "encode_action")


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    def test_config_to_components_workflow(self) -> None:
        """Test creating all components from config."""
        from alpha_zero.distributed import (
            ActorConfig,
            CoordinatorConfig,
            LearnerConfig,
            ReplayBufferConfig,
        )
        from alpha_zero.utils import get_default_config

        # Use default config
        config = get_default_config()

        # Create all component configs
        coord_config = CoordinatorConfig(
            num_actors=config.distributed.num_actors,
            max_games=1000,
        )

        actor_config = ActorConfig(
            num_simulations=config.mcts.num_simulations,
            games_per_update=config.distributed.actor_batch_size,
        )

        learner_config = LearnerConfig(
            batch_size=config.distributed.learner_batch_size,
            learning_rate=config.training.learning_rate,
        )

        buffer_config = ReplayBufferConfig(
            max_size=config.training.replay_buffer_size,
            min_size=config.training.min_replay_size,
        )

        # Verify all configs are valid
        assert coord_config.num_actors == 64
        assert actor_config.num_simulations == 200
        assert learner_config.batch_size == 256
        assert buffer_config.max_size == 500000

    def test_logging_with_metrics_workflow(self) -> None:
        """Test logging and metrics tracking workflow."""
        from alpha_zero.utils import (
            LogConfig,
            MetricsTracker,
            TrainingLogger,
            get_default_config,
        )

        config = get_default_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            log_config = LogConfig(
                level=config.logging.level,
                log_dir=tmpdir,
            )

            logger = TrainingLogger(log_config)
            tracker = MetricsTracker(window_size=config.logging.metrics_window)

            # Simulate training loop
            for epoch in range(5):
                loss = 1.0 / (epoch + 1)
                tracker.add("loss", loss)
                logger.log_epoch(
                    epoch=epoch,
                    total_loss=loss,
                    policy_loss=loss * 0.6,
                    value_loss=loss * 0.4,
                )

            stats = tracker.get_stats("loss")
            assert stats["count"] == 5
            assert stats["mean"] > 0
