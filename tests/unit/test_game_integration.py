"""Unit tests for game integration layer."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def basic_config(temp_checkpoint_dir):
    """Create basic connection configuration."""
    from alpha_zero.integration.game_connector import ConnectionConfig, RecoveryConfig

    return ConnectionConfig(
        recovery=RecoveryConfig(
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_enabled=True,
        ),
    )


@pytest.fixture
def mock_backend():
    """Create mock game backend."""
    backend = MagicMock()
    backend.connect.return_value = True
    backend.is_connected.return_value = True
    backend.get_state.return_value = {"turn": 1, "player": 0}
    backend.apply_action.return_value = {"success": True}
    backend.reset.return_value = {"observation": {}, "info": {}}
    backend.get_valid_actions.return_value = [0, 1, 2, 3]
    return backend


# =============================================================================
# RetryConfig Tests
# =============================================================================


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from alpha_zero.integration.game_connector import RetryConfig

        config = RetryConfig()
        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0

    def test_get_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        from alpha_zero.integration.game_connector import RetryConfig

        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=0.0,  # Disable jitter for predictable testing
        )

        # Delay should increase exponentially
        delay0 = config.get_delay(0)
        delay1 = config.get_delay(1)
        delay2 = config.get_delay(2)

        assert delay0 == 1.0  # 1 * 2^0
        assert delay1 == 2.0  # 1 * 2^1
        assert delay2 == 4.0  # 1 * 2^2

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        from alpha_zero.integration.game_connector import RetryConfig

        config = RetryConfig(
            base_delay=1.0,
            max_delay=10.0,
            jitter=0.0,
        )

        # High attempt should be capped
        delay = config.get_delay(100)
        assert delay == 10.0


# =============================================================================
# TimeoutConfig Tests
# =============================================================================


class TestTimeoutConfig:
    """Tests for TimeoutConfig dataclass."""

    def test_default_values(self):
        """Test default timeout values."""
        from alpha_zero.integration.game_connector import TimeoutConfig

        config = TimeoutConfig()
        assert config.connect_timeout == 30.0
        assert config.action_timeout == 10.0
        assert config.state_timeout == 5.0


# =============================================================================
# ConnectionConfig Tests
# =============================================================================


class TestConnectionConfig:
    """Tests for ConnectionConfig dataclass."""

    def test_default_preset(self):
        """Test default configuration preset."""
        from alpha_zero.integration.game_connector import ConnectionConfig

        config = ConnectionConfig.default()
        assert config.retry.max_retries == 5

    def test_aggressive_preset(self):
        """Test aggressive retry preset."""
        from alpha_zero.integration.game_connector import ConnectionConfig

        config = ConnectionConfig.aggressive_retry()
        assert config.retry.max_retries == 10
        assert config.retry.base_delay == 0.5

    def test_low_latency_preset(self):
        """Test low latency preset."""
        from alpha_zero.integration.game_connector import ConnectionConfig

        config = ConnectionConfig.low_latency()
        assert config.timeout.action_timeout == 5.0
        assert config.latency.latency_threshold_ms == 50.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from alpha_zero.integration.game_connector import ConnectionConfig

        config = ConnectionConfig.default()
        d = config.to_dict()

        assert "retry" in d
        assert "timeout" in d
        assert "latency" in d
        assert "recovery" in d


# =============================================================================
# LatencyTracker Tests
# =============================================================================


class TestLatencyTracker:
    """Tests for LatencyTracker class."""

    def test_record_sample(self):
        """Test recording latency samples."""
        from alpha_zero.integration.game_connector import LatencyConfig, LatencyTracker

        config = LatencyConfig(measurement_window=10)
        tracker = LatencyTracker(config)

        tracker.record_sample(50.0)
        tracker.record_sample(100.0)

        assert tracker.average_latency == 75.0

    def test_p95_latency(self):
        """Test 95th percentile latency calculation."""
        from alpha_zero.integration.game_connector import LatencyConfig, LatencyTracker

        config = LatencyConfig(measurement_window=100)
        tracker = LatencyTracker(config)

        # Add samples
        for i in range(100):
            tracker.record_sample(float(i))

        p95 = tracker.p95_latency
        assert p95 >= 90  # Should be around 95

    def test_action_latency_tracking(self):
        """Test separate action latency tracking."""
        from alpha_zero.integration.game_connector import LatencyConfig, LatencyTracker

        config = LatencyConfig()
        tracker = LatencyTracker(config)

        # Record regular sample
        tracker.record_sample(50.0, is_action=False)
        # Record action sample
        tracker.record_sample(100.0, is_action=True)

        assert tracker.action_latency == 100.0
        assert tracker.average_latency == 75.0

    def test_compensation_delay(self):
        """Test latency compensation calculation."""
        from alpha_zero.integration.game_connector import LatencyConfig, LatencyTracker

        config = LatencyConfig(
            compensation_enabled=True,
            target_latency_ms=50.0,
        )
        tracker = LatencyTracker(config)

        # High latency
        tracker.record_sample(150.0)

        # Should compensate for 100ms difference
        delay = tracker.compensation_delay
        assert delay > 0

    def test_reset(self):
        """Test tracker reset."""
        from alpha_zero.integration.game_connector import LatencyConfig, LatencyTracker

        tracker = LatencyTracker(LatencyConfig())
        tracker.record_sample(100.0)
        tracker.reset()

        assert tracker.average_latency == 0.0


# =============================================================================
# ConnectionMetrics Tests
# =============================================================================


class TestConnectionMetrics:
    """Tests for ConnectionMetrics dataclass."""

    def test_initial_values(self):
        """Test initial metric values."""
        from alpha_zero.integration.game_connector import ConnectionMetrics

        metrics = ConnectionMetrics()
        assert metrics.total_actions == 0
        assert metrics.success_rate == 1.0
        assert metrics.retry_rate == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        from alpha_zero.integration.game_connector import ConnectionMetrics

        metrics = ConnectionMetrics(
            total_actions=100,
            successful_actions=80,
            failed_actions=20,
        )

        assert metrics.success_rate == 0.8

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from alpha_zero.integration.game_connector import ConnectionMetrics

        metrics = ConnectionMetrics(
            total_actions=50,
            successful_actions=45,
        )

        d = metrics.to_dict()
        assert d["total_actions"] == 50
        assert d["success_rate"] == 0.9


# =============================================================================
# CheckpointManager Tests
# =============================================================================


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_save_checkpoint(self, temp_checkpoint_dir):
        """Test saving checkpoints."""
        from alpha_zero.integration.game_connector import CheckpointManager, RecoveryConfig

        config = RecoveryConfig(
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_enabled=True,
        )
        manager = CheckpointManager(config)

        state = {"turn": 10, "player": 0}
        path = manager.save_checkpoint(state, "game_001", 10)

        assert path is not None
        assert path.exists()

    def test_load_latest_checkpoint(self, temp_checkpoint_dir):
        """Test loading latest checkpoint."""
        from alpha_zero.integration.game_connector import CheckpointManager, RecoveryConfig

        config = RecoveryConfig(
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_enabled=True,
        )
        manager = CheckpointManager(config)

        # Save multiple checkpoints
        manager.save_checkpoint({"turn": 1}, "game_001", 1)
        manager.save_checkpoint({"turn": 2}, "game_001", 2)
        manager.save_checkpoint({"turn": 3}, "game_001", 3)

        # Load latest
        checkpoint = manager.load_latest_checkpoint()

        assert checkpoint is not None
        assert checkpoint["state"]["turn"] == 3

    def test_load_by_game_id(self, temp_checkpoint_dir):
        """Test loading checkpoint by game ID."""
        from alpha_zero.integration.game_connector import CheckpointManager, RecoveryConfig

        config = RecoveryConfig(
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_enabled=True,
        )
        manager = CheckpointManager(config)

        # Save for different games
        manager.save_checkpoint({"turn": 1}, "game_001", 1)
        manager.save_checkpoint({"turn": 2}, "game_002", 2)

        # Load specific game
        checkpoint = manager.load_latest_checkpoint(game_id="game_001")

        assert checkpoint is not None
        assert checkpoint["game_id"] == "game_001"

    def test_max_checkpoints_cleanup(self, temp_checkpoint_dir):
        """Test old checkpoints are cleaned up."""
        from alpha_zero.integration.game_connector import CheckpointManager, RecoveryConfig

        config = RecoveryConfig(
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_enabled=True,
            max_checkpoints=3,
        )
        manager = CheckpointManager(config)

        # Save more than max
        for i in range(5):
            manager.save_checkpoint({"turn": i}, "game_001", i)

        # Should only have max_checkpoints
        assert len(manager._checkpoints) == 3

    def test_clear_checkpoints(self, temp_checkpoint_dir):
        """Test clearing checkpoints."""
        from alpha_zero.integration.game_connector import CheckpointManager, RecoveryConfig

        config = RecoveryConfig(
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_enabled=True,
        )
        manager = CheckpointManager(config)

        manager.save_checkpoint({"turn": 1}, "game_001", 1)
        manager.save_checkpoint({"turn": 2}, "game_001", 2)

        removed = manager.clear_checkpoints()

        assert removed == 2
        assert len(manager._checkpoints) == 0


# =============================================================================
# GameConnector Tests
# =============================================================================


class TestGameConnector:
    """Tests for GameConnector class."""

    def test_connect_success(self, basic_config, mock_backend):
        """Test successful connection."""
        from alpha_zero.integration.game_connector import ConnectionStatus, GameConnector, GameType

        connector = GameConnector(
            basic_config,
            game_type=GameType.MICROCIV,
            backend=mock_backend,
        )

        result = connector.connect()

        assert result is True
        assert connector.is_connected is True
        assert connector.status == ConnectionStatus.CONNECTED

    def test_connect_callback(self, basic_config, mock_backend):
        """Test connection callback is triggered."""
        from alpha_zero.integration.game_connector import GameConnector, GameType

        connector = GameConnector(
            basic_config,
            game_type=GameType.MICROCIV,
            backend=mock_backend,
        )

        callback_data = {"called": False}

        def on_connect():
            callback_data["called"] = True

        connector.on_connect(on_connect)
        connector.connect()

        assert callback_data["called"] is True

    def test_disconnect(self, basic_config, mock_backend):
        """Test disconnection."""
        from alpha_zero.integration.game_connector import ConnectionStatus, GameConnector, GameType

        connector = GameConnector(
            basic_config,
            game_type=GameType.MICROCIV,
            backend=mock_backend,
        )

        connector.connect()
        connector.disconnect()

        assert connector.status == ConnectionStatus.DISCONNECTED

    def test_get_state(self, basic_config, mock_backend):
        """Test state retrieval."""
        from alpha_zero.integration.game_connector import GameConnector, GameType

        connector = GameConnector(
            basic_config,
            game_type=GameType.MICROCIV,
            backend=mock_backend,
        )
        connector.connect()

        state = connector.get_state()

        assert state is not None
        assert "turn" in state

    def test_apply_action(self, basic_config, mock_backend):
        """Test action application."""
        from alpha_zero.integration.game_connector import GameConnector, GameType

        connector = GameConnector(
            basic_config,
            game_type=GameType.MICROCIV,
            backend=mock_backend,
        )
        connector.connect()

        result = connector.apply_action(0)

        assert result is not None
        mock_backend.apply_action.assert_called_once_with(0)

    def test_action_metrics(self, basic_config, mock_backend):
        """Test action metrics tracking."""
        from alpha_zero.integration.game_connector import GameConnector, GameType

        connector = GameConnector(
            basic_config,
            game_type=GameType.MICROCIV,
            backend=mock_backend,
        )
        connector.connect()

        # Apply some actions
        connector.apply_action(0)
        connector.apply_action(1)
        connector.apply_action(2)

        metrics = connector.get_metrics()

        assert metrics["total_actions"] == 3
        assert metrics["successful_actions"] == 3

    def test_reset(self, basic_config, mock_backend):
        """Test game reset."""
        from alpha_zero.integration.game_connector import GameConnector, GameType

        connector = GameConnector(
            basic_config,
            game_type=GameType.MICROCIV,
            backend=mock_backend,
        )
        connector.connect()

        result = connector.reset()

        assert result is not None
        mock_backend.reset.assert_called()

    def test_get_valid_actions(self, basic_config, mock_backend):
        """Test getting valid actions."""
        from alpha_zero.integration.game_connector import GameConnector, GameType

        connector = GameConnector(
            basic_config,
            game_type=GameType.MICROCIV,
            backend=mock_backend,
        )
        connector.connect()

        actions = connector.get_valid_actions()

        assert actions == [0, 1, 2, 3]

    def test_reconnect(self, basic_config, mock_backend):
        """Test reconnection."""
        from alpha_zero.integration.game_connector import GameConnector, GameType

        connector = GameConnector(
            basic_config,
            game_type=GameType.MICROCIV,
            backend=mock_backend,
        )
        connector.connect()

        result = connector.reconnect()

        assert result is True
        assert connector._metrics.reconnections == 1

    def test_error_callback(self, basic_config):
        """Test error callback is triggered on failure."""
        from alpha_zero.integration.game_connector import (
            ConnectionConfig,
            GameConnector,
            GameType,
            RetryConfig,
        )

        # Create backend that fails
        failing_backend = MagicMock()
        failing_backend.connect.return_value = True
        failing_backend.is_connected.return_value = True
        failing_backend.apply_action.side_effect = Exception("Test error")

        # Use minimal retries for faster test
        config = ConnectionConfig(
            retry=RetryConfig(max_retries=1, base_delay=0.01),
        )

        connector = GameConnector(
            config,
            game_type=GameType.MICROCIV,
            backend=failing_backend,
        )

        callback_data = {"error": None}

        def on_error(e):
            callback_data["error"] = e

        connector.on_error(on_error)
        connector.connect()

        with pytest.raises(Exception):
            connector.apply_action(0)

        assert callback_data["error"] is not None

    def test_latency_stats(self, basic_config, mock_backend):
        """Test latency statistics retrieval."""
        from alpha_zero.integration.game_connector import GameConnector, GameType

        connector = GameConnector(
            basic_config,
            game_type=GameType.MICROCIV,
            backend=mock_backend,
        )
        connector.connect()

        # Perform some operations
        connector.get_state()
        connector.apply_action(0)

        stats = connector.get_latency_stats()

        assert "average_ms" in stats
        assert "samples" in stats


# =============================================================================
# GameType Tests
# =============================================================================


class TestGameType:
    """Tests for GameType enum."""

    def test_game_types(self):
        """Test game type enumeration."""
        from alpha_zero.integration.game_connector import GameType

        assert GameType.MICROCIV.name == "MICROCIV"
        assert GameType.UNCIV.name == "UNCIV"
        assert GameType.CIV6.name == "CIV6"


# =============================================================================
# ConnectionStatus Tests
# =============================================================================


class TestConnectionStatus:
    """Tests for ConnectionStatus enum."""

    def test_connection_statuses(self):
        """Test connection status enumeration."""
        from alpha_zero.integration.game_connector import ConnectionStatus

        assert ConnectionStatus.DISCONNECTED.name == "DISCONNECTED"
        assert ConnectionStatus.CONNECTED.name == "CONNECTED"
        assert ConnectionStatus.ERROR.name == "ERROR"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_game_connector(self, temp_checkpoint_dir):
        """Test creating game connector with factory."""
        from alpha_zero.integration.game_connector import GameType, create_game_connector

        connector = create_game_connector(
            game_type=GameType.MICROCIV,
            preset="default",
        )

        assert connector is not None
        assert connector.game_type == GameType.MICROCIV

    def test_create_microciv_connector(self):
        """Test creating MicroCiv connector with factory."""
        from alpha_zero.integration.game_connector import create_microciv_connector

        connector = create_microciv_connector(
            board_size=8,
            max_turns=100,
        )

        assert connector is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_connection_timeout_error(self):
        """Test connection timeout error."""
        from alpha_zero.integration.game_connector import ConnectionTimeoutError, ErrorSeverity

        error = ConnectionTimeoutError()
        assert error.severity == ErrorSeverity.RECOVERABLE

    def test_game_connection_error(self):
        """Test game connection error."""
        from alpha_zero.integration.game_connector import ErrorSeverity, GameConnectionError

        error = GameConnectionError(
            "Test error",
            severity=ErrorSeverity.CRITICAL,
        )

        assert error.severity == ErrorSeverity.CRITICAL
        assert str(error) == "Test error"
