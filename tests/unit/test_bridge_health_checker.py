"""Tests for bridge health checker with circuit breaker pattern."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from alpha_zero.bridge.health_checker import (
    BridgeHealth,
    BridgeHealthConfig,
    CircuitBreakerOpenError,
    CircuitState,
    GameBridgeHealthChecker,
    HealthStatus,
    OperationTimeoutError,
)


# ── Config Tests ──────────────────────────────────────────────────────────


class TestBridgeHealthConfig:
    """Tests for BridgeHealthConfig."""

    def test_default_values(self):
        config = BridgeHealthConfig()
        assert config.check_interval_seconds == 5.0
        assert config.timeout_seconds == 10.0
        assert config.max_retries == 5
        assert config.circuit_breaker_threshold == 3
        assert config.success_threshold == 2
        assert config.health_window_size == 10
        assert 0 <= config.degraded_threshold <= 1

    def test_custom_values(self):
        config = BridgeHealthConfig(
            check_interval_seconds=2.0,
            timeout_seconds=5.0,
            max_retries=3,
            circuit_breaker_threshold=5,
        )
        assert config.check_interval_seconds == 2.0
        assert config.timeout_seconds == 5.0
        assert config.max_retries == 3
        assert config.circuit_breaker_threshold == 5

    @pytest.mark.parametrize(
        "field,value",
        [
            ("check_interval_seconds", 0),
            ("check_interval_seconds", -1),
            ("timeout_seconds", 0),
            ("timeout_seconds", -1),
            ("max_retries", -1),
            ("retry_delay_seconds", 0),
            ("backoff_multiplier", 0.5),
            ("max_backoff_seconds", 0),
            ("circuit_breaker_threshold", 0),
            ("circuit_breaker_timeout_seconds", 0),
            ("success_threshold", 0),
            ("health_window_size", 0),
            ("degraded_threshold", -0.1),
            ("degraded_threshold", 1.1),
        ],
    )
    def test_validation_rejects_invalid(self, field: str, value: Any):
        with pytest.raises(ValueError):
            BridgeHealthConfig(**{field: value})

    def test_from_dict(self):
        data = {
            "check_interval_seconds": 3.0,
            "timeout_seconds": 7.0,
            "max_retries": 2,
        }
        config = BridgeHealthConfig.from_dict(data)
        assert config.check_interval_seconds == 3.0
        assert config.timeout_seconds == 7.0
        assert config.max_retries == 2

    def test_to_dict_roundtrip(self):
        original = BridgeHealthConfig(check_interval_seconds=2.0, max_retries=3)
        d = original.to_dict()
        restored = BridgeHealthConfig.from_dict(d)
        assert restored.check_interval_seconds == original.check_interval_seconds
        assert restored.max_retries == original.max_retries

    def test_from_yaml(self, tmp_path: Path):
        yaml_path = tmp_path / "config.yaml"
        data = {
            "bridge_health": {
                "check_interval_seconds": 2.0,
                "timeout_seconds": 5.0,
            }
        }
        yaml_path.write_text(yaml.dump(data))
        config = BridgeHealthConfig.from_yaml(yaml_path)
        assert config.check_interval_seconds == 2.0
        assert config.timeout_seconds == 5.0

    def test_from_yaml_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            BridgeHealthConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_from_yaml_invalid_content(self, tmp_path: Path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("just a string")
        with pytest.raises(ValueError, match="mapping"):
            BridgeHealthConfig.from_yaml(yaml_path)

    def test_from_env(self):
        config = BridgeHealthConfig.from_env()
        assert isinstance(config, BridgeHealthConfig)


# ── BridgeHealth Tests ────────────────────────────────────────────────────


class TestBridgeHealth:
    """Tests for BridgeHealth dataclass."""

    def test_to_dict(self):
        health = BridgeHealth(
            status=HealthStatus.HEALTHY,
            circuit_state=CircuitState.CLOSED,
            is_connected=True,
            last_check_time=1234.5,
            consecutive_failures=0,
        )
        d = health.to_dict()
        assert d["status"] == "healthy"
        assert d["circuit_state"] == "closed"
        assert d["is_connected"] is True

    def test_unhealthy_state(self):
        health = BridgeHealth(
            status=HealthStatus.UNHEALTHY,
            circuit_state=CircuitState.OPEN,
            is_connected=False,
            last_check_time=0.0,
            error_message="Connection lost",
        )
        assert health.error_message == "Connection lost"
        assert health.status == HealthStatus.UNHEALTHY


# ── Exception Tests ───────────────────────────────────────────────────────


class TestExceptions:
    """Tests for custom exceptions."""

    def test_circuit_breaker_open_error(self):
        err = CircuitBreakerOpenError("test", retry_after=5.0)
        assert err.retry_after == 5.0
        assert "test" in str(err)

    def test_operation_timeout_error(self):
        err = OperationTimeoutError("timed out", timeout=10.0)
        assert err.timeout == 10.0


# ── GameBridgeHealthChecker Tests ─────────────────────────────────────────


class MockBridge:
    """Mock bridge for testing."""

    def __init__(self, healthy: bool = True):
        self._healthy = healthy
        self.check_count = 0

    def check_heartbeat(self) -> bool:
        self.check_count += 1
        return self._healthy

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy


class TestGameBridgeHealthChecker:
    """Tests for GameBridgeHealthChecker."""

    @pytest.fixture
    def bridge(self):
        return MockBridge(healthy=True)

    @pytest.fixture
    def config(self):
        return BridgeHealthConfig(
            check_interval_seconds=0.1,
            timeout_seconds=1.0,
            max_retries=2,
            retry_delay_seconds=0.01,
            backoff_multiplier=2.0,
            max_backoff_seconds=1.0,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout_seconds=0.1,
            success_threshold=2,
            health_window_size=10,
            degraded_threshold=0.3,
        )

    @pytest.fixture
    def checker(self, bridge, config):
        return GameBridgeHealthChecker(bridge, config)

    def test_initial_state(self, checker):
        assert checker.circuit_state == CircuitState.CLOSED
        assert not checker.is_circuit_open()

    @pytest.mark.asyncio
    async def test_check_health_healthy(self, checker):
        health = await checker.check_health()
        assert health.status == HealthStatus.HEALTHY
        assert health.is_connected is True
        assert health.circuit_state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_check_health_unhealthy(self, bridge, config):
        bridge.set_healthy(False)
        checker = GameBridgeHealthChecker(bridge, config)
        health = await checker.check_health()
        assert health.is_connected is False

    def test_record_success(self, checker):
        checker.record_success()
        assert checker._consecutive_failures == 0
        assert checker._consecutive_successes == 1

    def test_record_failure(self, checker):
        checker.record_failure("test error")
        assert checker._consecutive_failures == 1
        assert checker._consecutive_successes == 0
        assert checker._last_error == "test error"

    def test_circuit_opens_after_threshold(self, checker, config):
        """Circuit opens after enough consecutive failures."""
        for i in range(config.circuit_breaker_threshold):
            checker.record_failure(f"error {i}")
        assert checker.circuit_state == CircuitState.OPEN

    def test_circuit_half_open_after_timeout(self, checker, config):
        """Circuit transitions to half-open after timeout."""
        for i in range(config.circuit_breaker_threshold):
            checker.record_failure(f"error {i}")
        assert checker.circuit_state == CircuitState.OPEN

        # Simulate timeout passing
        checker._circuit_opened_at = time.time() - config.circuit_breaker_timeout_seconds - 1
        assert not checker.is_circuit_open()
        assert checker.circuit_state == CircuitState.HALF_OPEN

    def test_circuit_closes_after_successes(self, checker, config):
        """Circuit closes after enough successes in half-open."""
        # Open the circuit
        for i in range(config.circuit_breaker_threshold):
            checker.record_failure(f"error {i}")

        # Transition to half-open
        checker._transition_to_half_open()
        assert checker.circuit_state == CircuitState.HALF_OPEN

        # Record enough successes
        for _ in range(config.success_threshold):
            checker.record_success()
        assert checker.circuit_state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self, checker, config):
        """Single failure in half-open reopens the circuit."""
        for i in range(config.circuit_breaker_threshold):
            checker.record_failure(f"error {i}")
        checker._transition_to_half_open()
        checker.record_failure("half-open failure")
        assert checker.circuit_state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, checker):
        """Successful operation returns result."""
        async def success_op():
            return 42

        result = await checker.execute_with_retry(success_op)
        assert result == 42

    @pytest.mark.asyncio
    async def test_execute_with_retry_eventual_success(self, checker):
        """Operation succeeds after retries."""
        call_count = 0

        async def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient failure")
            return "ok"

        result = await checker.execute_with_retry(flaky_op)
        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_retry_all_fail(self, checker, config):
        """All retries exhausted raises the last error."""
        async def failing_op():
            raise RuntimeError("persistent failure")

        with pytest.raises(RuntimeError, match="persistent failure"):
            await checker.execute_with_retry(failing_op)

    @pytest.mark.asyncio
    async def test_execute_with_retry_circuit_open(self, checker, config):
        """Operation rejected when circuit is open."""
        for i in range(config.circuit_breaker_threshold):
            checker.record_failure(f"error {i}")

        async def some_op():
            return "should not reach"

        with pytest.raises(CircuitBreakerOpenError):
            await checker.execute_with_retry(some_op)

    def test_failure_rate_calculation(self, checker):
        checker.record_success()
        checker.record_success()
        checker.record_failure("x")
        rate = checker._calculate_failure_rate()
        assert rate == pytest.approx(1 / 3)

    def test_avg_latency_empty(self, checker):
        assert checker._calculate_avg_latency() == 0.0

    def test_health_status_determination(self, checker):
        assert checker._determine_health_status(True) == HealthStatus.HEALTHY
        assert checker._determine_health_status(False) == HealthStatus.UNHEALTHY

    def test_health_status_degraded(self, checker, config):
        """High failure rate marks as degraded."""
        # Interleave failures and successes to exceed degraded threshold
        # without tripping the circuit breaker (threshold=3 consecutive).
        # degraded_threshold=0.3, so 4/10 = 0.4 > 0.3 should trigger degraded.
        for _ in range(4):
            checker.record_failure("x")
            checker.record_success()  # Reset consecutive failures
        for _ in range(2):
            checker.record_success()
        status = checker._determine_health_status(True)
        assert status == HealthStatus.DEGRADED

    def test_callbacks_on_circuit_open(self, checker, config):
        """Callbacks fire when circuit opens."""
        events = []
        checker.register_circuit_open_callback(lambda: events.append("open"))

        for i in range(config.circuit_breaker_threshold):
            checker.record_failure(f"error {i}")

        assert "open" in events

    def test_callbacks_on_circuit_close(self, checker, config):
        """Callbacks fire when circuit closes."""
        events = []
        checker.register_circuit_close_callback(lambda: events.append("close"))

        for i in range(config.circuit_breaker_threshold):
            checker.record_failure(f"error {i}")
        checker._transition_to_half_open()
        for _ in range(config.success_threshold):
            checker.record_success()

        assert "close" in events

    @pytest.mark.asyncio
    async def test_health_change_callback(self, checker):
        """Health change callback fires on check."""
        events = []
        checker.register_health_change_callback(lambda h: events.append(h))
        await checker.check_health()
        assert len(events) == 1
        assert isinstance(events[0], BridgeHealth)

    def test_reset(self, checker, config):
        """Reset clears all state."""
        for i in range(config.circuit_breaker_threshold):
            checker.record_failure(f"error {i}")
        checker.reset()
        assert checker.circuit_state == CircuitState.CLOSED
        assert checker._consecutive_failures == 0

    def test_get_statistics(self, checker):
        stats = checker.get_statistics()
        assert "circuit_state" in stats
        assert "failure_rate" in stats
        assert "monitoring_active" in stats

    def test_repr(self, checker):
        r = repr(checker)
        assert "GameBridgeHealthChecker" in r
        assert "circuit=" in r

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, checker):
        """Start and stop monitoring without errors."""
        await checker.start_monitoring()
        assert checker._monitoring is True
        await asyncio.sleep(0.05)
        await checker.stop_monitoring()
        assert checker._monitoring is False

    @pytest.mark.asyncio
    async def test_double_start_monitoring(self, checker):
        """Starting monitoring twice doesn't create duplicate tasks."""
        await checker.start_monitoring()
        await checker.start_monitoring()  # Should log warning, not crash
        await checker.stop_monitoring()
