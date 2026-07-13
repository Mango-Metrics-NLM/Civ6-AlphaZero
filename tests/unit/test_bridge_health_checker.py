"""Comprehensive pytest tests for alpha_zero.bridge.health_checker.

Coverage targets (≥75%):
- BridgeHealthConfig: defaults, validate(), from_env(), from_dict(), to_dict(), from_yaml()
- CircuitBreakerOpenError / OperationTimeoutError: construction, attributes
- BridgeHealth: construction, to_dict()
- GameBridgeHealthChecker: instantiation, check_health() healthy/unhealthy,
  circuit breaker state transitions (CLOSED->OPEN->HALF_OPEN->CLOSED),
  execute_with_retry(), callbacks, monitoring start/stop, reset(),
  get_statistics(), __repr__, _determine_health_status().
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _fast_config(**overrides: Any) -> BridgeHealthConfig:
    """Return a BridgeHealthConfig with fast defaults for unit tests."""
    defaults: dict[str, Any] = {
        "check_interval_seconds": 0.1,
        "timeout_seconds": 1.0,
        "max_retries": 2,
        "retry_delay_seconds": 0.01,
        "backoff_multiplier": 2.0,
        "max_backoff_seconds": 1.0,
        "circuit_breaker_threshold": 3,
        "circuit_breaker_timeout_seconds": 0.1,
        "success_threshold": 2,
        "health_window_size": 10,
        "degraded_threshold": 0.3,
    }
    defaults.update(overrides)
    return BridgeHealthConfig(**defaults)


class MockBridge:
    """Mock bridge for testing."""

    def __init__(self, healthy: bool = True) -> None:
        self._healthy = healthy
        self.check_count = 0

    def check_heartbeat(self) -> bool:
        self.check_count += 1
        return self._healthy

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy


def _make_checker(healthy: bool = True, **config_overrides: Any) -> GameBridgeHealthChecker:
    return GameBridgeHealthChecker(MockBridge(healthy=healthy), _fast_config(**config_overrides))


# ---------------------------------------------------------------------------
# BridgeHealthConfig – defaults
# ---------------------------------------------------------------------------


class TestBridgeHealthConfigDefaults:
    """Verify every default value listed in the docstring."""

    def test_default_check_interval(self) -> None:
        assert BridgeHealthConfig().check_interval_seconds == 5.0

    def test_default_timeout(self) -> None:
        assert BridgeHealthConfig().timeout_seconds == 10.0

    def test_default_max_retries(self) -> None:
        assert BridgeHealthConfig().max_retries == 5

    def test_default_retry_delay(self) -> None:
        assert BridgeHealthConfig().retry_delay_seconds == 1.0

    def test_default_backoff_multiplier(self) -> None:
        assert BridgeHealthConfig().backoff_multiplier == 2.0

    def test_default_max_backoff(self) -> None:
        assert BridgeHealthConfig().max_backoff_seconds == 60.0

    def test_default_circuit_threshold(self) -> None:
        assert BridgeHealthConfig().circuit_breaker_threshold == 3

    def test_default_circuit_timeout(self) -> None:
        assert BridgeHealthConfig().circuit_breaker_timeout_seconds == 30.0

    def test_default_success_threshold(self) -> None:
        assert BridgeHealthConfig().success_threshold == 2

    def test_default_health_window_size(self) -> None:
        assert BridgeHealthConfig().health_window_size == 10

    def test_default_degraded_threshold(self) -> None:
        assert BridgeHealthConfig().degraded_threshold == 0.3


# ---------------------------------------------------------------------------
# BridgeHealthConfig – validate() error cases
# ---------------------------------------------------------------------------


class TestBridgeHealthConfigValidate:
    """Each violated constraint must raise ValueError."""

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
    def test_validation_rejects_invalid(self, field: str, value: Any) -> None:
        with pytest.raises(ValueError):
            BridgeHealthConfig(**{field: value})

    def test_max_retries_zero_is_valid(self) -> None:
        cfg = BridgeHealthConfig(max_retries=0)
        assert cfg.max_retries == 0

    def test_degraded_threshold_boundary_zero_valid(self) -> None:
        cfg = BridgeHealthConfig(degraded_threshold=0.0)
        assert cfg.degraded_threshold == 0.0

    def test_degraded_threshold_boundary_one_valid(self) -> None:
        cfg = BridgeHealthConfig(degraded_threshold=1.0)
        assert cfg.degraded_threshold == 1.0

    def test_backoff_multiplier_exactly_one_valid(self) -> None:
        cfg = BridgeHealthConfig(backoff_multiplier=1.0)
        assert cfg.backoff_multiplier == 1.0


# ---------------------------------------------------------------------------
# BridgeHealthConfig – factories and round-trip
# ---------------------------------------------------------------------------


class TestBridgeHealthConfigFactories:
    def test_from_env_returns_instance(self) -> None:
        assert isinstance(BridgeHealthConfig.from_env(), BridgeHealthConfig)

    def test_from_env_reads_env_variable(self) -> None:
        with patch.dict(os.environ, {"CIV6AZ_BRIDGE_MAX_RETRIES": "7"}):
            cfg = BridgeHealthConfig.from_env()
        assert cfg.max_retries == 7

    def test_from_dict_partial_override(self) -> None:
        data = {"max_retries": 3, "timeout_seconds": 20.0}
        cfg = BridgeHealthConfig.from_dict(data)
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 20.0
        assert cfg.check_interval_seconds == 5.0  # default untouched

    def test_from_dict_empty_uses_defaults(self) -> None:
        cfg = BridgeHealthConfig.from_dict({})
        assert cfg.max_retries == 5

    def test_from_dict_coerces_string_to_int(self) -> None:
        cfg = BridgeHealthConfig.from_dict({"max_retries": "4"})
        assert cfg.max_retries == 4

    def test_from_dict_coerces_string_to_float(self) -> None:
        cfg = BridgeHealthConfig.from_dict({"check_interval_seconds": "3"})
        assert cfg.check_interval_seconds == 3.0

    def test_to_dict_round_trip(self) -> None:
        original = _fast_config()
        restored = BridgeHealthConfig.from_dict(original.to_dict())
        assert original.to_dict() == restored.to_dict()

    def test_to_dict_contains_all_expected_keys(self) -> None:
        expected = {
            "check_interval_seconds",
            "timeout_seconds",
            "max_retries",
            "retry_delay_seconds",
            "backoff_multiplier",
            "max_backoff_seconds",
            "circuit_breaker_threshold",
            "circuit_breaker_timeout_seconds",
            "success_threshold",
            "health_window_size",
            "degraded_threshold",
        }
        assert set(_fast_config().to_dict().keys()) == expected

    def test_custom_values_stored_correctly(self) -> None:
        cfg = BridgeHealthConfig(
            check_interval_seconds=2.0,
            timeout_seconds=5.0,
            max_retries=3,
            circuit_breaker_threshold=5,
        )
        assert cfg.check_interval_seconds == 2.0
        assert cfg.timeout_seconds == 5.0
        assert cfg.max_retries == 3
        assert cfg.circuit_breaker_threshold == 5

    def test_from_yaml_with_bridge_health_section(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "cfg.yaml"
        data = {"bridge_health": {"check_interval_seconds": 2.0, "timeout_seconds": 5.0}}
        yaml_path.write_text(yaml.dump(data))
        cfg = BridgeHealthConfig.from_yaml(yaml_path)
        assert cfg.check_interval_seconds == 2.0
        assert cfg.timeout_seconds == 5.0

    def test_from_yaml_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            BridgeHealthConfig.from_yaml(tmp_path / "nope.yaml")

    def test_from_yaml_invalid_content_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("just a string")
        with pytest.raises(ValueError, match="mapping"):
            BridgeHealthConfig.from_yaml(yaml_path)


# ---------------------------------------------------------------------------
# CircuitBreakerOpenError & OperationTimeoutError
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpenError:
    def test_default_message_contains_open(self) -> None:
        err = CircuitBreakerOpenError()
        assert "open" in str(err).lower()

    def test_custom_message(self) -> None:
        err = CircuitBreakerOpenError("custom msg")
        assert "custom msg" in str(err)

    def test_retry_after_default_zero(self) -> None:
        assert CircuitBreakerOpenError().retry_after == 0

    def test_retry_after_custom_value(self) -> None:
        err = CircuitBreakerOpenError(retry_after=15.5)
        assert err.retry_after == 15.5

    def test_is_subclass_of_exception(self) -> None:
        assert issubclass(CircuitBreakerOpenError, Exception)


class TestOperationTimeoutError:
    def test_default_message_contains_timeout(self) -> None:
        err_msg = str(OperationTimeoutError()).lower()
        assert "timed out" in err_msg or "timeout" in err_msg

    def test_timeout_attribute(self) -> None:
        err = OperationTimeoutError(timeout=7.5)
        assert err.timeout == 7.5

    def test_is_subclass_of_exception(self) -> None:
        assert issubclass(OperationTimeoutError, Exception)

    def test_custom_message(self) -> None:
        err = OperationTimeoutError("took too long")
        assert "took too long" in str(err)


# ---------------------------------------------------------------------------
# BridgeHealth dataclass
# ---------------------------------------------------------------------------


class TestBridgeHealth:
    def _make(self, **kwargs: Any) -> BridgeHealth:
        defaults: dict[str, Any] = {
            "status": HealthStatus.HEALTHY,
            "circuit_state": CircuitState.CLOSED,
            "is_connected": True,
            "last_check_time": 1234.5,
        }
        defaults.update(kwargs)
        return BridgeHealth(**defaults)

    def test_to_dict_status_is_string_value(self) -> None:
        assert self._make().to_dict()["status"] == "healthy"

    def test_to_dict_circuit_state_is_string_value(self) -> None:
        assert self._make().to_dict()["circuit_state"] == "closed"

    def test_to_dict_is_connected_false(self) -> None:
        assert self._make(is_connected=False).to_dict()["is_connected"] is False

    def test_to_dict_includes_error_message(self) -> None:
        d = self._make(error_message="oops").to_dict()
        assert d["error_message"] == "oops"

    def test_default_failure_rate_zero(self) -> None:
        assert self._make().failure_rate == 0.0

    def test_default_details_empty_dict(self) -> None:
        assert self._make().details == {}

    def test_unhealthy_state_preserved(self) -> None:
        h = self._make(
            status=HealthStatus.UNHEALTHY,
            circuit_state=CircuitState.OPEN,
            is_connected=False,
            error_message="Connection lost",
        )
        assert h.status == HealthStatus.UNHEALTHY
        assert h.error_message == "Connection lost"


# ---------------------------------------------------------------------------
# GameBridgeHealthChecker – instantiation
# ---------------------------------------------------------------------------


class TestGameBridgeHealthCheckerInstantiation:
    def test_initial_circuit_state_closed(self) -> None:
        assert _make_checker().circuit_state == CircuitState.CLOSED

    def test_is_circuit_open_initially_false(self) -> None:
        assert _make_checker().is_circuit_open() is False

    def test_default_config_created_when_none(self) -> None:
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge)
        assert isinstance(checker.config, BridgeHealthConfig)

    def test_bridge_stored(self) -> None:
        bridge = MockBridge()
        checker = GameBridgeHealthChecker(bridge, _fast_config())
        assert checker.bridge is bridge

    def test_repr_contains_class_name(self) -> None:
        assert "GameBridgeHealthChecker" in repr(_make_checker())

    def test_repr_contains_circuit_state(self) -> None:
        assert "circuit=" in repr(_make_checker())

    def test_repr_contains_failure_rate(self) -> None:
        assert "failure_rate=" in repr(_make_checker())


# ---------------------------------------------------------------------------
# GameBridgeHealthChecker – check_health()
# ---------------------------------------------------------------------------


class TestCheckHealthHealthy:
    @pytest.mark.asyncio
    async def test_returns_healthy_status(self) -> None:
        checker = _make_checker(healthy=True)
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            health = await checker.check_health()
        assert health.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_is_connected_true(self) -> None:
        checker = _make_checker(healthy=True)
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            health = await checker.check_health()
        assert health.is_connected is True

    @pytest.mark.asyncio
    async def test_circuit_state_closed(self) -> None:
        checker = _make_checker(healthy=True)
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            health = await checker.check_health()
        assert health.circuit_state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_no_error_message(self) -> None:
        checker = _make_checker(healthy=True)
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            health = await checker.check_health()
        assert health.error_message is None

    @pytest.mark.asyncio
    async def test_consecutive_successes_incremented(self) -> None:
        checker = _make_checker(healthy=True)
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            await checker.check_health()
        assert checker._consecutive_successes == 1

    @pytest.mark.asyncio
    async def test_to_dict_has_expected_keys(self) -> None:
        checker = _make_checker(healthy=True)
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            health = await checker.check_health()
        d = health.to_dict()
        for key in ("status", "circuit_state", "is_connected", "failure_rate", "latency_ms"):
            assert key in d


class TestCheckHealthUnhealthy:
    @pytest.mark.asyncio
    async def test_heartbeat_false_gives_unhealthy(self) -> None:
        checker = _make_checker(healthy=False)
        with patch("asyncio.to_thread", new=AsyncMock(return_value=False)):
            health = await checker.check_health()
        assert health.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_heartbeat_false_is_connected_false(self) -> None:
        checker = _make_checker(healthy=False)
        with patch("asyncio.to_thread", new=AsyncMock(return_value=False)):
            health = await checker.check_health()
        assert health.is_connected is False

    @pytest.mark.asyncio
    async def test_heartbeat_false_sets_error_message(self) -> None:
        checker = _make_checker(healthy=False)
        with patch("asyncio.to_thread", new=AsyncMock(return_value=False)):
            health = await checker.check_health()
        assert health.error_message is not None

    @pytest.mark.asyncio
    async def test_exception_in_heartbeat_is_connected_false(self) -> None:
        checker = _make_checker()
        with patch("asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("boom"))):
            health = await checker.check_health()
        assert health.is_connected is False

    @pytest.mark.asyncio
    async def test_exception_stores_error_message(self) -> None:
        checker = _make_checker()
        with patch("asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("boom"))):
            health = await checker.check_health()
        assert "boom" in (health.error_message or "")

    @pytest.mark.asyncio
    async def test_health_callback_invoked_once(self) -> None:
        checker = _make_checker(healthy=True)
        cb = MagicMock()
        checker.register_health_change_callback(cb)
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            await checker.check_health()
        cb.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_callback_receives_bridge_health(self) -> None:
        checker = _make_checker(healthy=True)
        received: list[BridgeHealth] = []
        checker.register_health_change_callback(received.append)
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            await checker.check_health()
        assert isinstance(received[0], BridgeHealth)

    @pytest.mark.asyncio
    async def test_health_callback_exception_does_not_propagate(self) -> None:
        checker = _make_checker(healthy=True)
        checker.register_health_change_callback(MagicMock(side_effect=RuntimeError("cb error")))
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            health = await checker.check_health()
        assert health is not None


# ---------------------------------------------------------------------------
# Circuit breaker state transitions
# ---------------------------------------------------------------------------


class TestCircuitBreakerTransitions:
    def test_closed_to_open_after_n_failures(self) -> None:
        checker = _make_checker(circuit_breaker_threshold=3)
        checker.record_failure("e1")
        checker.record_failure("e2")
        assert checker.circuit_state == CircuitState.CLOSED
        checker.record_failure("e3")
        assert checker.circuit_state == CircuitState.OPEN

    def test_open_circuit_is_open_returns_true(self) -> None:
        checker = _make_checker(circuit_breaker_threshold=1, circuit_breaker_timeout_seconds=30.0)
        checker.record_failure("fail")
        assert checker.is_circuit_open() is True

    def test_open_transitions_to_half_open_after_timeout(self) -> None:
        checker = _make_checker(circuit_breaker_threshold=1, circuit_breaker_timeout_seconds=0.01)
        checker.record_failure("fail")
        time.sleep(0.05)
        assert checker.is_circuit_open() is False
        assert checker.circuit_state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_after_enough_successes(self) -> None:
        checker = _make_checker(
            circuit_breaker_threshold=1,
            circuit_breaker_timeout_seconds=0.01,
            success_threshold=2,
        )
        checker.record_failure("fail")
        time.sleep(0.05)
        checker.is_circuit_open()  # triggers HALF_OPEN
        checker.record_success()
        assert checker.circuit_state == CircuitState.HALF_OPEN
        checker.record_success()
        assert checker.circuit_state == CircuitState.CLOSED

    def test_half_open_to_open_on_single_failure(self) -> None:
        checker = _make_checker(circuit_breaker_threshold=1, circuit_breaker_timeout_seconds=0.01)
        checker.record_failure("fail")
        time.sleep(0.05)
        checker.is_circuit_open()
        checker.record_failure("another")
        assert checker.circuit_state == CircuitState.OPEN

    def test_circuit_open_callback_invoked_once(self) -> None:
        checker = _make_checker(circuit_breaker_threshold=2, circuit_breaker_timeout_seconds=30.0)
        cb = MagicMock()
        checker.register_circuit_open_callback(cb)
        checker.record_failure("e1")
        checker.record_failure("e2")
        cb.assert_called_once()

    def test_circuit_close_callback_invoked_on_recovery(self) -> None:
        checker = _make_checker(
            circuit_breaker_threshold=1,
            circuit_breaker_timeout_seconds=0.01,
            success_threshold=1,
        )
        close_cb = MagicMock()
        checker.register_circuit_close_callback(close_cb)
        checker.record_failure("fail")
        time.sleep(0.05)
        checker.is_circuit_open()
        checker.record_success()
        close_cb.assert_called_once()

    def test_callback_exception_does_not_propagate_on_open(self) -> None:
        checker = _make_checker(circuit_breaker_threshold=1)
        checker.register_circuit_open_callback(MagicMock(side_effect=RuntimeError("oops")))
        checker.record_failure("fail")  # should NOT raise
        assert checker.circuit_state == CircuitState.OPEN

    def test_record_success_resets_consecutive_failures(self) -> None:
        checker = _make_checker()
        checker.record_failure("e")
        checker.record_failure("e")
        checker.record_success()
        assert checker._consecutive_failures == 0

    def test_record_success_increments_consecutive_successes(self) -> None:
        checker = _make_checker()
        checker.record_success()
        checker.record_success()
        assert checker._consecutive_successes == 2

    def test_record_failure_clears_consecutive_successes(self) -> None:
        checker = _make_checker()
        checker.record_success()
        checker.record_failure("e")
        assert checker._consecutive_successes == 0

    def test_failure_rate_calculation(self) -> None:
        checker = _make_checker(health_window_size=4)
        checker.record_success()
        checker.record_failure("f")
        checker.record_success()
        checker.record_failure("f")
        assert checker._calculate_failure_rate() == pytest.approx(0.5)

    def test_failure_rate_empty_window_is_zero(self) -> None:
        assert _make_checker()._calculate_failure_rate() == 0.0

    def test_avg_latency_empty_is_zero(self) -> None:
        assert _make_checker()._calculate_avg_latency() == 0.0

    def test_avg_latency_calculated_correctly(self) -> None:
        checker = _make_checker()
        checker._latencies.append(100.0)
        checker._latencies.append(200.0)
        assert checker._calculate_avg_latency() == pytest.approx(150.0)

    def test_is_circuit_open_half_open_returns_false(self) -> None:
        checker = _make_checker(circuit_breaker_threshold=1, circuit_breaker_timeout_seconds=0.01)
        checker.record_failure("fail")
        time.sleep(0.05)
        checker._transition_to_half_open()
        assert checker.is_circuit_open() is False


# ---------------------------------------------------------------------------
# _determine_health_status()
# ---------------------------------------------------------------------------


class TestDetermineHealthStatus:
    def test_open_circuit_always_unhealthy(self) -> None:
        checker = _make_checker(circuit_breaker_threshold=1)
        checker.record_failure("fail")
        assert checker._determine_health_status(is_connected=True) == HealthStatus.UNHEALTHY

    def test_not_connected_always_unhealthy(self) -> None:
        checker = _make_checker()
        assert checker._determine_health_status(is_connected=False) == HealthStatus.UNHEALTHY

    def test_high_failure_rate_gives_degraded(self) -> None:
        # Use alternating failures/successes to avoid triggering circuit breaker
        checker = _make_checker(degraded_threshold=0.3, health_window_size=10)
        for _ in range(6):
            checker.record_success()
        for _ in range(4):
            checker._health_results.append(False)  # add failures without opening circuit
        checker._circuit_state = CircuitState.CLOSED
        assert checker._determine_health_status(is_connected=True) == HealthStatus.DEGRADED

    def test_low_failure_rate_gives_healthy(self) -> None:
        checker = _make_checker(degraded_threshold=0.5, health_window_size=10)
        for _ in range(9):
            checker.record_success()
        checker._health_results.append(False)
        checker._circuit_state = CircuitState.CLOSED
        assert checker._determine_health_status(is_connected=True) == HealthStatus.HEALTHY

    def test_no_results_connected_gives_healthy(self) -> None:
        checker = _make_checker()
        assert checker._determine_health_status(is_connected=True) == HealthStatus.HEALTHY


# ---------------------------------------------------------------------------
# execute_with_retry()
# ---------------------------------------------------------------------------


class TestExecuteWithRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        checker = _make_checker()
        op = AsyncMock(return_value=42)
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await checker.execute_with_retry(op)
        assert result == 42

    @pytest.mark.asyncio
    async def test_raises_circuit_breaker_error_when_open(self) -> None:
        checker = _make_checker(
            circuit_breaker_threshold=1, circuit_breaker_timeout_seconds=30.0
        )
        checker.record_failure("force open")
        with pytest.raises(CircuitBreakerOpenError):
            await checker.execute_with_retry(AsyncMock())

    @pytest.mark.asyncio
    async def test_retry_after_positive_when_circuit_open(self) -> None:
        checker = _make_checker(
            circuit_breaker_threshold=1, circuit_breaker_timeout_seconds=30.0
        )
        checker.record_failure("force open")
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await checker.execute_with_retry(AsyncMock())
        assert exc_info.value.retry_after > 0

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure_then_succeeds(self) -> None:
        checker = _make_checker(max_retries=2, retry_delay_seconds=0.01)
        call_count = 0

        async def flaky_op() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "ok"

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await checker.execute_with_retry(flaky_op)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_last_exception_after_all_retries_exhausted(self) -> None:
        checker = _make_checker(max_retries=2, retry_delay_seconds=0.01)
        op = AsyncMock(side_effect=ValueError("permanent"))
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ValueError, match="permanent"):
                await checker.execute_with_retry(op)

    @pytest.mark.asyncio
    async def test_operation_called_with_args_and_kwargs(self) -> None:
        checker = _make_checker(max_retries=0)
        op = AsyncMock(return_value="done")
        with patch("asyncio.sleep", new=AsyncMock()):
            await checker.execute_with_retry(op, "a", "b", key="val")
        op.assert_awaited_once_with("a", "b", key="val")

    @pytest.mark.asyncio
    async def test_records_latency_after_success(self) -> None:
        checker = _make_checker(max_retries=0)
        op = AsyncMock(return_value=None)
        with patch("asyncio.sleep", new=AsyncMock()):
            await checker.execute_with_retry(op)
        assert len(checker._latencies) >= 1

    @pytest.mark.asyncio
    async def test_backoff_capped_at_max_backoff(self) -> None:
        checker = _make_checker(
            max_retries=3,
            retry_delay_seconds=10.0,
            backoff_multiplier=10.0,
            max_backoff_seconds=15.0,
            circuit_breaker_threshold=100,
        )
        call_count = 0

        async def always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"fail {call_count}")

        sleep_calls: list[float] = []

        async def capture_sleep(t: float) -> None:
            sleep_calls.append(t)

        with patch("asyncio.sleep", side_effect=capture_sleep):
            with pytest.raises(RuntimeError):
                await checker.execute_with_retry(always_fail)

        for delay in sleep_calls[1:]:
            assert delay <= 15.0

    @pytest.mark.asyncio
    async def test_backoff_grows_exponentially(self) -> None:
        checker = _make_checker(
            max_retries=3,
            retry_delay_seconds=0.1,
            backoff_multiplier=3.0,
            max_backoff_seconds=100.0,
            circuit_breaker_threshold=100,
        )
        call_count = 0

        async def always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"fail {call_count}")

        sleep_calls: list[float] = []

        async def capture_sleep(t: float) -> None:
            sleep_calls.append(t)

        with patch("asyncio.sleep", side_effect=capture_sleep):
            with pytest.raises(RuntimeError):
                await checker.execute_with_retry(always_fail)

        assert len(sleep_calls) == 3
        assert sleep_calls[0] == pytest.approx(0.1)
        assert sleep_calls[1] == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_circuit_opens_mid_retry_raises_circuit_breaker_error(self) -> None:
        checker = _make_checker(
            max_retries=5,
            retry_delay_seconds=0.01,
            circuit_breaker_threshold=1,
        )
        op = AsyncMock(side_effect=RuntimeError("fail"))
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises((CircuitBreakerOpenError, RuntimeError)):
                await checker.execute_with_retry(op)

    @pytest.mark.asyncio
    async def test_zero_retries_fails_on_first_attempt(self) -> None:
        checker = _make_checker(max_retries=0)
        op = AsyncMock(side_effect=ValueError("single shot"))
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ValueError, match="single shot"):
                await checker.execute_with_retry(op)


# ---------------------------------------------------------------------------
# reset() and get_statistics()
# ---------------------------------------------------------------------------


class TestResetAndStatistics:
    def test_reset_clears_failures(self) -> None:
        checker = _make_checker(circuit_breaker_threshold=1)
        checker.record_failure("e")
        checker.reset()
        assert checker._consecutive_failures == 0
        assert checker.circuit_state == CircuitState.CLOSED

    def test_reset_clears_health_results(self) -> None:
        checker = _make_checker()
        checker.record_success()
        checker.reset()
        assert len(checker._health_results) == 0

    def test_reset_clears_latencies(self) -> None:
        checker = _make_checker()
        checker._latencies.append(123.0)
        checker.reset()
        assert len(checker._latencies) == 0

    def test_reset_clears_last_error(self) -> None:
        checker = _make_checker()
        checker.record_failure("err")
        checker.reset()
        assert checker._last_error is None

    def test_get_statistics_has_expected_keys(self) -> None:
        expected_keys = {
            "circuit_state",
            "consecutive_failures",
            "consecutive_successes",
            "failure_rate",
            "avg_latency_ms",
            "health_window_size",
            "last_check_time",
            "last_error",
            "monitoring_active",
        }
        assert expected_keys.issubset(_make_checker().get_statistics().keys())

    def test_get_statistics_circuit_state_is_string_value(self) -> None:
        stats = _make_checker().get_statistics()
        assert stats["circuit_state"] == CircuitState.CLOSED.value

    def test_get_statistics_monitoring_false_initially(self) -> None:
        assert _make_checker().get_statistics()["monitoring_active"] is False


# ---------------------------------------------------------------------------
# Monitoring start / stop
# ---------------------------------------------------------------------------


class TestMonitoring:
    @pytest.mark.asyncio
    async def test_start_monitoring_sets_flag(self) -> None:
        checker = _make_checker()
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            await checker.start_monitoring()
            assert checker._monitoring is True
            await checker.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stop_monitoring_clears_flag(self) -> None:
        checker = _make_checker()
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            await checker.start_monitoring()
            await checker.stop_monitoring()
        assert checker._monitoring is False

    @pytest.mark.asyncio
    async def test_start_monitoring_idempotent(self) -> None:
        checker = _make_checker()
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            await checker.start_monitoring()
            task_first = checker._monitor_task
            await checker.start_monitoring()  # second call must be a no-op
            assert checker._monitor_task is task_first
            await checker.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stop_monitoring_when_not_started_no_error(self) -> None:
        checker = _make_checker()
        await checker.stop_monitoring()  # must not raise

    @pytest.mark.asyncio
    async def test_monitoring_updates_statistics(self) -> None:
        checker = _make_checker()
        with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
            await checker.start_monitoring()
            await asyncio.sleep(0.05)
            await checker.stop_monitoring()
        assert checker.get_statistics()["monitoring_active"] is False
