"""Tests for distributed actor components.

Tests cover:
- ActorConfig defaults and validation
- GameHistoryItem dataclass
- SelfPlayActor initialization and basic operations
- ActorPool management (without spawning actual processes)
"""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alpha_zero.distributed.actor import (
    ActorConfig,
    ActorPool,
    GameHistoryItem,
    SelfPlayActor,
)
from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    ReplayBufferConfig,
    SharedStorage,
    SharedStorageConfig,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def actor_config() -> ActorConfig:
    """Create a minimal actor configuration for testing."""
    return ActorConfig(
        actor_id=0,
        board_size=8,
        max_turns=50,
        num_res_blocks=2,
        num_filters=16,
        num_simulations=10,
        max_games=5,
        log_interval=1,
    )


@pytest.fixture
def shared_storage() -> SharedStorage:
    """Create shared storage for testing."""
    config = SharedStorageConfig(compress_weights=False)
    storage = SharedStorage(config=config)
    yield storage
    storage.shutdown()


@pytest.fixture
def replay_buffer() -> DistributedReplayBuffer:
    """Create replay buffer for testing."""
    config = ReplayBufferConfig(max_size=1000, min_size=10)
    buffer = DistributedReplayBuffer(config=config)
    yield buffer
    buffer.shutdown()


@pytest.fixture
def sample_game_history_item() -> GameHistoryItem:
    """Create a sample game history item."""
    return GameHistoryItem(
        spatial=np.random.rand(13, 8, 8).astype(np.float32),
        vector=np.random.rand(64).astype(np.float32),
        action_mask=np.ones(381, dtype=np.float32),
        policy=np.random.rand(381).astype(np.float32),
        player=0,
    )


# =============================================================================
# ActorConfig Tests
# =============================================================================


class TestActorConfig:
    """Tests for ActorConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ActorConfig()
        assert config.actor_id == 0
        assert config.board_size == 16
        assert config.max_turns == 200
        assert config.num_res_blocks == 6
        assert config.num_filters == 64
        assert config.num_simulations == 50
        assert config.c_puct == 1.5
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25
        assert config.temperature == 1.0
        assert config.temperature_threshold == 30
        assert config.games_per_update == 1
        assert config.max_games == 0
        assert config.device == "cpu"
        assert config.weight_update_interval == 1.0
        assert config.trajectory_submission_timeout == 5.0
        assert config.log_interval == 10

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ActorConfig(
            actor_id=5,
            board_size=8,
            max_turns=100,
            num_simulations=20,
            max_games=10,
            device="cuda:0",
        )
        assert config.actor_id == 5
        assert config.board_size == 8
        assert config.max_turns == 100
        assert config.num_simulations == 20
        assert config.max_games == 10
        assert config.device == "cuda:0"

    def test_config_to_dict(self) -> None:
        """Test config can be converted to dict."""
        config = ActorConfig(actor_id=1)
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert config_dict["actor_id"] == 1
        assert "board_size" in config_dict
        assert "num_simulations" in config_dict


# =============================================================================
# GameHistoryItem Tests
# =============================================================================


class TestGameHistoryItem:
    """Tests for GameHistoryItem dataclass."""

    def test_creation(self, sample_game_history_item: GameHistoryItem) -> None:
        """Test GameHistoryItem can be created."""
        item = sample_game_history_item
        assert item.spatial.shape == (13, 8, 8)
        assert item.vector.shape == (64,)
        assert item.action_mask.shape == (381,)
        assert item.policy.shape == (381,)
        assert item.player == 0

    def test_creation_with_different_player(self) -> None:
        """Test GameHistoryItem with different player."""
        item = GameHistoryItem(
            spatial=np.zeros((13, 8, 8), dtype=np.float32),
            vector=np.zeros(64, dtype=np.float32),
            action_mask=np.ones(381, dtype=np.float32),
            policy=np.ones(381, dtype=np.float32) / 381,
            player=1,
        )
        assert item.player == 1


# =============================================================================
# SelfPlayActor Tests
# =============================================================================


class TestSelfPlayActor:
    """Tests for SelfPlayActor class.

    These tests use mocking to avoid actually playing games or
    loading real networks.
    """

    def test_initialization(
        self,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test SelfPlayActor initializes correctly."""
        actor = SelfPlayActor(actor_config, shared_storage, replay_buffer)

        assert actor.config == actor_config
        assert actor.storage == shared_storage
        assert actor.replay_buffer == replay_buffer
        assert actor.games_played == 0
        assert actor.total_examples == 0
        assert actor.is_running() is False
        assert actor.network is None  # Not initialized until run()
        assert actor.mcts is None

    def test_get_statistics_before_running(
        self,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test get_statistics returns correct values before running."""
        actor = SelfPlayActor(actor_config, shared_storage, replay_buffer)
        stats = actor.get_statistics()

        assert stats["actor_id"] == actor_config.actor_id
        assert stats["games_played"] == 0
        assert stats["total_examples"] == 0
        assert stats["model_version"] == 0
        assert stats["is_running"] is False

    def test_stop_method(
        self,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test stop method sets running flag to False."""
        actor = SelfPlayActor(actor_config, shared_storage, replay_buffer)

        # Simulate running state
        actor._running = True
        assert actor.is_running() is True

        actor.stop()
        assert actor.is_running() is False

    @patch("alpha_zero.distributed.actor.MicroCivNetwork")
    @patch("alpha_zero.distributed.actor.SimpleMCTS")
    def test_init_network_without_weights(
        self,
        mock_simple_mcts_class: MagicMock,
        mock_network_class: MagicMock,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test _init_network uses SimpleMCTS when no weights available."""
        # Setup mock network
        mock_network = MagicMock()
        mock_network.to.return_value = mock_network
        mock_network_class.return_value = mock_network

        # Setup mock SimpleMCTS
        mock_mcts = MagicMock()
        mock_simple_mcts_class.return_value = mock_mcts

        actor = SelfPlayActor(actor_config, shared_storage, replay_buffer)
        actor._init_network()

        # Should have created network
        mock_network_class.assert_called_once()
        mock_network.eval.assert_called_once()

        # Should use SimpleMCTS since no weights
        mock_simple_mcts_class.assert_called_once_with(
            num_simulations=actor_config.num_simulations,
            c_puct=actor_config.c_puct,
        )
        assert actor.mcts is mock_mcts

    @patch("alpha_zero.distributed.actor.MicroCivNetwork")
    def test_update_weights_false_initially(
        self,
        mock_network_class: MagicMock,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test _update_weights returns False when not initialized."""
        actor = SelfPlayActor(actor_config, shared_storage, replay_buffer)

        # Should return False since network/sync not initialized
        result = actor._update_weights()
        assert result is False


# =============================================================================
# ActorPool Tests
# =============================================================================


class TestActorPool:
    """Tests for ActorPool class.

    These tests avoid actually spawning processes by mocking
    multiprocessing components.
    """

    def test_initialization(
        self,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test ActorPool initializes correctly."""
        pool = ActorPool(
            num_actors=4,
            actor_config=actor_config,
            storage=shared_storage,
            replay_buffer=replay_buffer,
        )

        assert pool.num_actors == 4
        assert pool.base_config == actor_config
        assert pool.storage == shared_storage
        assert pool.replay_buffer == replay_buffer
        assert len(pool.processes) == 0
        assert pool._started is False

    def test_is_running_before_start(
        self,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test is_running returns False before start."""
        pool = ActorPool(
            num_actors=2,
            actor_config=actor_config,
            storage=shared_storage,
            replay_buffer=replay_buffer,
        )
        assert pool.is_running() is False

    def test_get_running_count_before_start(
        self,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test get_running_count returns 0 before start."""
        pool = ActorPool(
            num_actors=2,
            actor_config=actor_config,
            storage=shared_storage,
            replay_buffer=replay_buffer,
        )
        assert pool.get_running_count() == 0

    @patch("alpha_zero.distributed.actor.mp.Process")
    def test_start_creates_processes(
        self,
        mock_process_class: MagicMock,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test start creates the correct number of processes."""
        # Setup mock process
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process_class.return_value = mock_process

        pool = ActorPool(
            num_actors=3,
            actor_config=actor_config,
            storage=shared_storage,
            replay_buffer=replay_buffer,
        )

        pool.start()

        # Should have created 3 processes
        assert mock_process_class.call_count == 3
        assert mock_process.start.call_count == 3
        assert len(pool.processes) == 3
        assert pool._started is True

    @patch("alpha_zero.distributed.actor.mp.Process")
    def test_start_twice_warns(
        self,
        mock_process_class: MagicMock,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test starting twice doesn't create more processes."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process_class.return_value = mock_process

        pool = ActorPool(
            num_actors=2,
            actor_config=actor_config,
            storage=shared_storage,
            replay_buffer=replay_buffer,
        )

        pool.start()
        initial_count = mock_process_class.call_count

        pool.start()  # Second start should be no-op

        assert mock_process_class.call_count == initial_count

    def test_stop_before_start(
        self,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test stop before start is safe."""
        pool = ActorPool(
            num_actors=2,
            actor_config=actor_config,
            storage=shared_storage,
            replay_buffer=replay_buffer,
        )

        # Should not raise
        pool.stop()
        assert pool._started is False

    @patch("alpha_zero.distributed.actor.mp.Process")
    def test_stop_terminates_processes(
        self,
        mock_process_class: MagicMock,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test stop terminates running processes."""
        # Setup mock process that appears to stop gracefully
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.is_alive.return_value = False  # Process stopped
        mock_process_class.return_value = mock_process

        pool = ActorPool(
            num_actors=2,
            actor_config=actor_config,
            storage=shared_storage,
            replay_buffer=replay_buffer,
        )

        pool.start()
        pool.stop(timeout=1.0)

        # Should have called join on processes
        assert mock_process.join.called
        assert pool._started is False
        assert len(pool.processes) == 0

    @patch("alpha_zero.distributed.actor.mp.Process")
    def test_is_running_with_live_processes(
        self,
        mock_process_class: MagicMock,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test is_running returns True when processes are alive."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.is_alive.return_value = True
        mock_process_class.return_value = mock_process

        pool = ActorPool(
            num_actors=2,
            actor_config=actor_config,
            storage=shared_storage,
            replay_buffer=replay_buffer,
        )

        pool.start()

        assert pool.is_running() is True
        assert pool.get_running_count() == 2

        # Cleanup
        mock_process.is_alive.return_value = False
        pool.stop()

    @patch("alpha_zero.distributed.actor.mp.Process")
    def test_restart_dead_actors(
        self,
        mock_process_class: MagicMock,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test restart_dead_actors restarts crashed processes."""
        # Create initial processes (for start())
        initial_processes = []
        for i in range(3):
            mock_process = MagicMock()
            mock_process.pid = 12345 + i
            mock_process.is_alive.return_value = i != 1  # Process 1 is dead
            initial_processes.append(mock_process)

        # Create restart process
        restart_process = MagicMock()
        restart_process.pid = 99999
        restart_process.is_alive.return_value = True

        # Use side_effect to return different mocks in sequence
        mock_process_class.side_effect = [*initial_processes, restart_process]

        pool = ActorPool(
            num_actors=3,
            actor_config=actor_config,
            storage=shared_storage,
            replay_buffer=replay_buffer,
        )

        pool.start()

        restarted = pool.restart_dead_actors()

        assert restarted == 1
        # Verify a new process was started
        assert restart_process.start.called

        # Cleanup
        for p in pool.processes:
            p.is_alive.return_value = False
        pool.stop()

    def test_restart_dead_actors_not_started(
        self,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test restart_dead_actors returns 0 when not started."""
        pool = ActorPool(
            num_actors=2,
            actor_config=actor_config,
            storage=shared_storage,
            replay_buffer=replay_buffer,
        )

        restarted = pool.restart_dead_actors()
        assert restarted == 0

    def test_restart_dead_actors_after_stop_requested(
        self,
        actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Test restart_dead_actors returns 0 when stop requested."""
        pool = ActorPool(
            num_actors=2,
            actor_config=actor_config,
            storage=shared_storage,
            replay_buffer=replay_buffer,
        )

        # Request stop
        shared_storage.request_stop()

        restarted = pool.restart_dead_actors()
        assert restarted == 0
