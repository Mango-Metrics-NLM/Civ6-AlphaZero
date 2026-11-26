"""Unit tests for bridge health checker.

Tests for BridgeHealthConfig, GameBridgeHealthChecker, circuit breaker,
exponential backoff retry, and health monitoring.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.bridge.health_checker import (
    BridgeHealth,
    BridgeHealthConfig,
    CircuitBreakerOpenError,
    CircuitState,
    GameBridgeHealthChecker,
    HealthStatus,
    OperationTimeoutError,
)

# =============================================================================
# BridgeHealthConfig Tests
# =============================================================================


class TestBridgeHealthConfig:
    """Tests for BridgeHealthConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BridgeHealthConfig()
        assert config.check_interval_seconds == 5.0
        assert config.timeout_seconds == 10.0
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 1.0
        assert config.backoff_multiplier == 2.0
        assert config.max_backoff_seconds == 60.0
        assert config.circuit_breaker_threshold == 3
        assert config.circuit_breaker_timeout_seconds == 30.0
        assert config.success_threshold == 2
        assert config.health_window_size == 10
        assert config.degraded_threshold == 0.3

    def test_config_from_env(self) -> None:
        """Test configuration from environment variables."""
        env_vars = {
            "CIV6AZ_BRIDGE_CHECK_INTERVAL": "10.0",
            "CIV6AZ_BRIDGE_TIMEOUT": "20.0",
            "CIV6AZ_BRIDGE_MAX_RETRIES": "3",
            "CIV6AZ_BRIDGE_RETRY_DELAY": "2.0",
            "CIV6AZ_BRIDGE_BACKOFF_MULTIPLIER": "3.0",
            "CIV6AZ_BRIDGE_MAX_BACKOFF": "120.0",
            "CIV6AZ_BRIDGE_CIRCUIT_THRESHOLD": "5",
            "CIV6AZ_BRIDGE_CIRCUIT_TIMEOUT": "60.0",
            "CIV6AZ_BRIDGE_SUCCESS_THRESHOLD": "3",
            "CIV6AZ_BRIDGE_HEALTH_WINDOW": "20",
            "CIV6AZ_BRIDGE_DEGRADED_THRESHOLD": "0.5",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = BridgeHealthConfig.from_env()

        assert config.check_interval_seconds == 10.0
        assert config.timeout_seconds == 20.0
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 2.0
        assert config.backoff_multiplier == 3.0
        assert config.max_backoff_seconds == 120.0
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_timeout_seconds == 60.0
        assert config.success_threshold == 3
        assert config.health_window_size == 20
        assert config.degraded_threshold == 0.5

    def test_config_from_yaml(self) -> None:
        """Test configuration from YAML file."""
        yaml_content = """
bridge_health:
  check_interval_seconds: 15.0
  timeout_seconds: 30.0
  max_retries: 10
  retry_delay_seconds: 0.5
  backoff_multiplier: 1.5
  max_backoff_seconds: 30.0
  circuit_breaker_threshold: 4
  circuit_breaker_timeout_seconds: 45.0
  success_threshold: 4
  health_window_size: 15
  degraded_threshold: 0.4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = BridgeHealthConfig.from_yaml(Path(temp_path))
        finally:
            os.unlink(temp_path)

        assert config.check_interval_seconds == 15.0
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 10
        assert config.retry_delay_seconds == 0.5
        assert config.backoff_multiplier == 1.5
        assert config.max_backoff_seconds == 30.0
        assert config.circuit_breaker_threshold == 4
        assert config.circuit_breaker_timeout_seconds == 45.0
        assert config.success_threshold == 4
        assert config.health_window_size == 15
        assert config.degraded_threshold == 0.4

    def test_config_from_yaml_missing_file(self) -> None:
        """Test from_yaml raises for missing file."""
        with pytest.raises(FileNotFoundError):
            BridgeHealthConfig.from_yaml(Path("/nonexistent/config.yaml"))

    def test_config_from_dict(self) -> None:
        """Test configuration from dictionary."""
        data = {
            "check_interval_seconds": 7.5,
            "timeout_seconds": 15.0,
            "max_retries": 4,
            "circuit_breaker_threshold": 2,
        }
        config = BridgeHealthConfig.from_dict(data)
        assert config.check_interval_seconds == 7.5
        assert config.timeout_seconds == 15.0
        assert config.max_retries == 4
        assert config.circuit_breaker_threshold == 2
        # Defaults for unspecified
        assert config.retry_delay_seconds == 1.0

    def test_config_to_dict(self) -> None:
        """Test configuration to dictionary."""
        config = BridgeHealthConfig(
            check_interval_seconds=8.0,
            timeout_seconds=12.0,
            max_retries=6,
        )
        data = config.to_dict()
        assert data["check_interval_seconds"] == 8.0
        assert data["timeout_seconds"] == 12.0
        assert data["max_retries"] == 6
        assert "circuit_breaker_threshold" in data

    def test_check_interval_validation(self) -> None:
        """Test check_interval_seconds validation."""
        with pytest.raises(ValueError, match="check_interval_seconds must be positive"):
            BridgeHealthConfig(check_interval_seconds=0)

    def test_timeout_validation(self) -> None:
        """Test timeout_seconds validation."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            BridgeHealthConfig(timeout_seconds=-1)

    def test_max_retries_validation(self) -> None:
        """Test max_retries validation."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            BridgeHealthConfig(max_retries=-1)

    def test_backoff_multiplier_validation(self) -> None:
        """Test backoff_multiplier validation."""
        with pytest.raises(ValueError, match=r"backoff_multiplier must be >= 1\.0"):
            BridgeHealthConfig(backoff_multiplier=0.5)

    def test_circuit_threshold_validation(self) -> None:
        """Test circuit_breaker_threshold validation."""
        with pytest.raises(ValueError, match="circuit_breaker_threshold must be >= 1"):
            BridgeHealthConfig(circuit_breaker_threshold=0)

    def test_degraded_threshold_validation(self) -> None:
        """Test degraded_threshold validation."""
        with pytest.raises(ValueError, match="degraded_threshold must be between 0 and 1"):
            BridgeHealthConfig(degraded_threshold=1.5)


# =============================================================================
# BridgeHealth Tests
# =============================================================================


class TestBridgeHealth:
    """Tests for BridgeHealth dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        health = BridgeHealth(
            status=HealthStatus.HEALTHY,
            circuit_state=CircuitState.CLOSED,
            is_connected=True,
            last_check_time=1000.0,
            consecutive_failures=0,
            consecutive_successes=5,
            failure_rate=0.1,
            latency_ms=10.5,
            error_message=None,
            details={"key": "value"},
        )
        data = health.to_dict()
        assert data["status"] == "healthy"
        assert data["circuit_state"] == "closed"
        assert data["is_connected"] is True
        assert data["failure_rate"] == 0.1
        assert data["latency_ms"] == 10.5
        assert data["details"] == {"key": "value"}


# =============================================================================
# Mock Bridge for Testing
# =============================================================================


class MockBridge:
    """Mock bridge for testing health checker."""

    def __init__(self, heartbeat_result: bool = True):
        self._heartbeat_result = heartbeat_result
        self._heartbeat_call_count = 0

    def check_heartbeat(self) -> bool:
        self._heartbeat_call_count += 1
        return self._heartbeat_result

    def set_heartbeat_result(self, result: bool) -> None:
        self._heartbeat_result = result


# =============================================================================
# GameBridgeHealthChecker Tests
# =============================================================================


class TestGameBridgeHealthChecker:
    """Tests for GameBridgeHealthChecker initialization."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)
        assert checker.config is not None
        assert checker.bridge is bridge
        assert checker.circuit_state == CircuitState.CLOSED

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        bridge = MockBridge()
        config = BridgeHealthConfig(
            circuit_breaker_threshold=5,
            timeout_seconds=15.0,
        )
        checker = GameBridgeHealthChecker(bridge, config)
        assert checker.config.circuit_breaker_threshold == 5
        assert checker.config.timeout_seconds == 15.0


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_initially_closed(self) -> None:
        """Test circuit starts closed."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)
        assert checker.circuit_state == CircuitState.CLOSED
        assert not checker.is_circuit_open()

    def test_circuit_opens_after_threshold_failures(self) -> None:
        """Test circuit opens after threshold failures."""
        bridge = MockBridge()
        config = BridgeHealthConfig(circuit_breaker_threshold=3)
        checker = GameBridgeHealthChecker(bridge, config)

        # Record failures up to threshold
        for _ in range(3):
            checker.record_failure("Test error")

        assert checker.circuit_state == CircuitState.OPEN
        assert checker.is_circuit_open()

    def test_circuit_stays_closed_below_threshold(self) -> None:
        """Test circuit stays closed below threshold."""
        bridge = MockBridge()
        config = BridgeHealthConfig(circuit_breaker_threshold=3)
        checker = GameBridgeHealthChecker(bridge, config)

        # Record failures below threshold
        for _ in range(2):
            checker.record_failure("Test error")

        assert checker.circuit_state == CircuitState.CLOSED
        assert not checker.is_circuit_open()

    def test_success_resets_failure_count(self) -> None:
        """Test success resets consecutive failure count."""
        bridge = MockBridge()
        config = BridgeHealthConfig(circuit_breaker_threshold=3)
        checker = GameBridgeHealthChecker(bridge, config)

        # Record some failures
        checker.record_failure("Error 1")
        checker.record_failure("Error 2")

        # Success resets count
        checker.record_success()
        assert checker._consecutive_failures == 0

        # Two more failures shouldn't open circuit
        checker.record_failure("Error 3")
        checker.record_failure("Error 4")
        assert checker.circuit_state == CircuitState.CLOSED

    def test_circuit_transitions_to_half_open(self) -> None:
        """Test circuit transitions to half-open after timeout."""
        bridge = MockBridge()
        config = BridgeHealthConfig(
            circuit_breaker_threshold=2,
            circuit_breaker_timeout_seconds=0.1,  # Short timeout for test
        )
        checker = GameBridgeHealthChecker(bridge, config)

        # Open circuit
        checker.record_failure("Error 1")
        checker.record_failure("Error 2")
        assert checker.circuit_state == CircuitState.OPEN

        # Wait for timeout
        import time

        time.sleep(0.15)

        # Check should transition to half-open
        assert not checker.is_circuit_open()
        assert checker.circuit_state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success_threshold(self) -> None:
        """Test half-open circuit closes after success threshold."""
        bridge = MockBridge()
        config = BridgeHealthConfig(
            circuit_breaker_threshold=2,
            circuit_breaker_timeout_seconds=0.01,
            success_threshold=2,
        )
        checker = GameBridgeHealthChecker(bridge, config)

        # Open and transition to half-open
        checker.record_failure("Error 1")
        checker.record_failure("Error 2")

        import time

        time.sleep(0.02)
        checker.is_circuit_open()  # Triggers transition

        # Record successes
        checker.record_success()
        assert checker.circuit_state == CircuitState.HALF_OPEN

        checker.record_success()
        assert checker.circuit_state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self) -> None:
        """Test half-open circuit reopens on single failure."""
        bridge = MockBridge()
        config = BridgeHealthConfig(
            circuit_breaker_threshold=2,
            circuit_breaker_timeout_seconds=0.01,
        )
        checker = GameBridgeHealthChecker(bridge, config)

        # Open and transition to half-open
        checker.record_failure("Error 1")
        checker.record_failure("Error 2")

        import time

        time.sleep(0.02)
        checker.is_circuit_open()  # Triggers transition
        assert checker.circuit_state == CircuitState.HALF_OPEN

        # Single failure in half-open reopens
        checker.record_failure("Error 3")
        assert checker.circuit_state == CircuitState.OPEN


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_check_health_connected(self) -> None:
        """Test health check when connected."""
        bridge = MockBridge(heartbeat_result=True)
        checker = GameBridgeHealthChecker(bridge)

        health = await checker.check_health()

        assert health.status == HealthStatus.HEALTHY
        assert health.is_connected is True
        assert health.circuit_state == CircuitState.CLOSED
        assert bridge._heartbeat_call_count == 1

    @pytest.mark.asyncio
    async def test_check_health_disconnected(self) -> None:
        """Test health check when disconnected."""
        bridge = MockBridge(heartbeat_result=False)
        checker = GameBridgeHealthChecker(bridge)

        health = await checker.check_health()

        assert health.status == HealthStatus.UNHEALTHY
        assert health.is_connected is False
        assert health.error_message == "Heartbeat check failed"

    @pytest.mark.asyncio
    async def test_check_health_with_exception(self) -> None:
        """Test health check handles exception."""
        bridge = MagicMock()
        bridge.check_heartbeat.side_effect = Exception("Connection error")
        checker = GameBridgeHealthChecker(bridge)

        health = await checker.check_health()

        assert health.status == HealthStatus.UNHEALTHY
        assert health.is_connected is False
        assert "Connection error" in health.error_message

    @pytest.mark.asyncio
    async def test_check_health_degraded_state(self) -> None:
        """Test health check reports degraded state."""
        bridge = MockBridge(heartbeat_result=True)
        config = BridgeHealthConfig(
            health_window_size=10,
            degraded_threshold=0.3,  # 30% failures = degraded
            circuit_breaker_threshold=100,  # High to prevent circuit opening
        )
        checker = GameBridgeHealthChecker(bridge, config)

        # Record some failures (below circuit threshold)
        for _ in range(3):
            checker.record_failure("Error")

        # Now connected but with failure history
        health = await checker.check_health()

        # 3 failures + 1 success = 4 results, 75% failure rate > 30%
        assert health.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_health_records_latency(self) -> None:
        """Test health check records latency."""
        bridge = MockBridge(heartbeat_result=True)
        checker = GameBridgeHealthChecker(bridge)

        health = await checker.check_health()

        assert health.latency_ms >= 0
        assert len(checker._latencies) == 1


class TestExecuteWithRetry:
    """Tests for execute_with_retry functionality."""

    @pytest.mark.asyncio
    async def test_execute_success_first_try(self) -> None:
        """Test successful execution on first try."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)

        async def successful_operation() -> str:
            return "success"

        result = await checker.execute_with_retry(successful_operation)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_success_after_retries(self) -> None:
        """Test successful execution after retries."""
        bridge = MockBridge()
        config = BridgeHealthConfig(
            max_retries=3,
            retry_delay_seconds=0.01,
        )
        checker = GameBridgeHealthChecker(bridge, config)

        call_count = 0

        async def flaky_operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary error")
            return "success"

        result = await checker.execute_with_retry(flaky_operation)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_failure_after_retries(self) -> None:
        """Test failure after exhausting retries."""
        bridge = MockBridge()
        config = BridgeHealthConfig(
            max_retries=2,
            retry_delay_seconds=0.01,
        )
        checker = GameBridgeHealthChecker(bridge, config)

        async def failing_operation() -> str:
            raise RuntimeError("Persistent error")

        with pytest.raises(RuntimeError, match="Persistent error"):
            await checker.execute_with_retry(failing_operation)

    @pytest.mark.asyncio
    async def test_execute_timeout(self) -> None:
        """Test operation timeout."""
        bridge = MockBridge()
        config = BridgeHealthConfig(
            timeout_seconds=0.05,
            max_retries=0,
        )
        checker = GameBridgeHealthChecker(bridge, config)

        async def slow_operation() -> str:
            await asyncio.sleep(0.2)
            return "never reached"

        with pytest.raises(OperationTimeoutError):
            await checker.execute_with_retry(slow_operation)

    @pytest.mark.asyncio
    async def test_execute_circuit_open_rejects(self) -> None:
        """Test circuit open rejects execution."""
        bridge = MockBridge()
        config = BridgeHealthConfig(circuit_breaker_threshold=1)
        checker = GameBridgeHealthChecker(bridge, config)

        # Open circuit
        checker.record_failure("Error")

        async def operation() -> str:
            return "should not run"

        with pytest.raises(CircuitBreakerOpenError):
            await checker.execute_with_retry(operation)

    @pytest.mark.asyncio
    async def test_execute_with_exponential_backoff(self) -> None:
        """Test exponential backoff between retries."""
        bridge = MockBridge()
        config = BridgeHealthConfig(
            max_retries=3,
            retry_delay_seconds=0.01,
            backoff_multiplier=2.0,
            max_backoff_seconds=1.0,
            circuit_breaker_threshold=100,  # High to prevent circuit opening
        )
        checker = GameBridgeHealthChecker(bridge, config)

        attempts = 0
        delays: list[float] = []

        async def failing_operation() -> str:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("Error")

        async def record_sleep(seconds: float) -> None:
            delays.append(seconds)

        # Assert the delays the checker REQUESTS, not wall-clock gaps: a measured
        # gap absorbs scheduler stalls on a loaded CI runner, which made this test
        # fail with delays[2] (0.04 s) < delays[1] (0.27 s).
        with (
            patch("alpha_zero.bridge.health_checker.asyncio.sleep", new=record_sleep),
            pytest.raises(RuntimeError),
        ):
            await checker.execute_with_retry(failing_operation)

        assert attempts == 4  # Initial + 3 retries
        assert delays == pytest.approx([0.01, 0.02, 0.04])

    @pytest.mark.asyncio
    async def test_execute_records_success(self) -> None:
        """Test successful execution records success."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)

        async def operation() -> str:
            return "success"

        await checker.execute_with_retry(operation)

        assert checker._consecutive_successes >= 1
        assert checker._consecutive_failures == 0


class TestCallbacks:
    """Tests for callback functionality."""

    def test_circuit_open_callback(self) -> None:
        """Test circuit open callback is called."""
        bridge = MockBridge()
        config = BridgeHealthConfig(circuit_breaker_threshold=2)
        checker = GameBridgeHealthChecker(bridge, config)

        callback_called = []

        def on_circuit_open() -> None:
            callback_called.append(True)

        checker.register_circuit_open_callback(on_circuit_open)

        # Open circuit
        checker.record_failure("Error 1")
        checker.record_failure("Error 2")

        assert len(callback_called) == 1

    def test_circuit_close_callback(self) -> None:
        """Test circuit close callback is called."""
        bridge = MockBridge()
        config = BridgeHealthConfig(
            circuit_breaker_threshold=1,
            circuit_breaker_timeout_seconds=0.01,
            success_threshold=1,
        )
        checker = GameBridgeHealthChecker(bridge, config)

        callback_called = []

        def on_circuit_close() -> None:
            callback_called.append(True)

        checker.register_circuit_close_callback(on_circuit_close)

        # Open circuit
        checker.record_failure("Error")

        import time

        time.sleep(0.02)
        checker.is_circuit_open()  # Transition to half-open

        # Close with success
        checker.record_success()

        assert len(callback_called) == 1

    @pytest.mark.asyncio
    async def test_health_change_callback(self) -> None:
        """Test health change callback is called."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)

        health_reports: list[BridgeHealth] = []

        def on_health_change(health: BridgeHealth) -> None:
            health_reports.append(health)

        checker.register_health_change_callback(on_health_change)

        await checker.check_health()

        assert len(health_reports) == 1
        assert health_reports[0].status == HealthStatus.HEALTHY


class TestMonitoringLifecycle:
    """Tests for monitoring lifecycle."""

    @pytest.mark.asyncio
    async def test_start_monitoring(self) -> None:
        """Test starting monitoring."""
        bridge = MockBridge()
        config = BridgeHealthConfig(check_interval_seconds=0.05)
        checker = GameBridgeHealthChecker(bridge, config)

        await checker.start_monitoring()
        assert checker._monitoring is True
        assert checker._monitor_task is not None

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Should have checked health
        assert bridge._heartbeat_call_count >= 1

        await checker.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stop_monitoring(self) -> None:
        """Test stopping monitoring."""
        bridge = MockBridge()
        config = BridgeHealthConfig(check_interval_seconds=0.05)
        checker = GameBridgeHealthChecker(bridge, config)

        await checker.start_monitoring()
        await checker.stop_monitoring()

        assert checker._monitoring is False
        assert checker._monitor_task is None

    @pytest.mark.asyncio
    async def test_double_start_warning(self) -> None:
        """Test double start logs warning."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)

        await checker.start_monitoring()
        await checker.start_monitoring()  # Should warn

        await checker.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stop_without_start(self) -> None:
        """Test stop without start is safe."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)

        await checker.stop_monitoring()  # Should not error
        assert checker._monitoring is False


class TestStatisticsAndReset:
    """Tests for statistics and reset."""

    def test_get_statistics(self) -> None:
        """Test get_statistics returns correct data."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)

        checker.record_success()
        checker.record_failure("Error")

        stats = checker.get_statistics()

        assert stats["circuit_state"] == "closed"
        assert stats["consecutive_failures"] == 1
        assert stats["consecutive_successes"] == 0
        assert "failure_rate" in stats
        assert "avg_latency_ms" in stats

    def test_reset(self) -> None:
        """Test reset clears all state."""
        bridge = MockBridge()
        config = BridgeHealthConfig(circuit_breaker_threshold=1)
        checker = GameBridgeHealthChecker(bridge, config)

        # Build up some state
        checker.record_failure("Error")
        assert checker.circuit_state == CircuitState.OPEN

        # Reset
        checker.reset()

        assert checker.circuit_state == CircuitState.CLOSED
        assert checker._consecutive_failures == 0
        assert checker._consecutive_successes == 0
        assert len(checker._health_results) == 0

    def test_repr(self) -> None:
        """Test string representation."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)

        repr_str = repr(checker)
        assert "GameBridgeHealthChecker" in repr_str
        assert "circuit=closed" in repr_str


class TestFailureRateCalculation:
    """Tests for failure rate calculation."""

    def test_empty_window(self) -> None:
        """Test failure rate with empty window."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)

        assert checker._calculate_failure_rate() == 0.0

    def test_all_successes(self) -> None:
        """Test failure rate with all successes."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)

        for _ in range(5):
            checker.record_success()

        assert checker._calculate_failure_rate() == 0.0

    def test_all_failures(self) -> None:
        """Test failure rate with all failures."""
        bridge = MockBridge()
        config = BridgeHealthConfig(circuit_breaker_threshold=100)  # Don't open
        checker = GameBridgeHealthChecker(bridge, config)

        for _ in range(5):
            checker.record_failure("Error")

        assert checker._calculate_failure_rate() == 1.0

    def test_mixed_results(self) -> None:
        """Test failure rate with mixed results."""
        bridge = MockBridge()
        config = BridgeHealthConfig(circuit_breaker_threshold=100)  # Don't open
        checker = GameBridgeHealthChecker(bridge, config)

        for _ in range(3):
            checker.record_success()
        for _ in range(2):
            checker.record_failure("Error")

        assert checker._calculate_failure_rate() == 0.4  # 2/5

    def test_window_size_limit(self) -> None:
        """Test failure rate respects window size."""
        bridge = MockBridge()
        config = BridgeHealthConfig(
            health_window_size=5,
            circuit_breaker_threshold=100,
        )
        checker = GameBridgeHealthChecker(bridge, config)

        # Fill window with failures
        for _ in range(5):
            checker.record_failure("Error")

        # Now add successes - old failures drop out
        for _ in range(5):
            checker.record_success()

        assert checker._calculate_failure_rate() == 0.0


class TestLatencyCalculation:
    """Tests for latency calculation."""

    @pytest.mark.asyncio
    async def test_avg_latency_empty(self) -> None:
        """Test average latency with no data."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)

        assert checker._calculate_avg_latency() == 0.0

    @pytest.mark.asyncio
    async def test_avg_latency_calculation(self) -> None:
        """Test average latency calculation."""
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)

        # Simulate some latencies
        checker._latencies.append(10.0)
        checker._latencies.append(20.0)
        checker._latencies.append(30.0)

        assert checker._calculate_avg_latency() == 20.0


class TestExceptionTypes:
    """Tests for custom exception types."""

    def test_circuit_breaker_open_error(self) -> None:
        """Test CircuitBreakerOpenError."""
        error = CircuitBreakerOpenError("Test message", retry_after=30.0)
        assert str(error) == "Test message"
        assert error.retry_after == 30.0

    def test_operation_timeout_error(self) -> None:
        """Test OperationTimeoutError."""
        error = OperationTimeoutError("Timeout message", timeout=10.0)
        assert str(error) == "Timeout message"
        assert error.timeout == 10.0
