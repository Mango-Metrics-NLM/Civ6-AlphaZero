"""Integration tests for CivRealm Docker connection.

Tests the connection to CivRealm Docker container, including
connection establishment, recovery, and timeout handling.

These tests require a running CivRealm Docker container:
    docker run -d -p 6300:6300 civrealm/civrealm:latest

Use pytest marker to skip when Docker is unavailable:
    pytest tests/integration/test_civrealm_docker.py -v -m requires_env
"""

from __future__ import annotations

import os
import socket
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from alpha_zero.envs.docker_config import (
    ConnectionState,
    ConnectionStatus,
    DockerConnectionConfig,
)

if TYPE_CHECKING:
    pass


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def docker_config() -> DockerConnectionConfig:
    """Create Docker connection config from environment or defaults."""
    return DockerConnectionConfig()


@pytest.fixture
def connection_state() -> ConnectionState:
    """Create fresh connection state."""
    return ConnectionState()


def _is_docker_available() -> bool:
    """Check if Docker is accessible by trying to connect to CivRealm port."""
    config = DockerConnectionConfig()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        result = sock.connect_ex((config.host, config.port))
        sock.close()
        return result == 0
    except OSError:
        return False


# Skip marker for tests requiring Docker
requires_docker = pytest.mark.skipif(
    not _is_docker_available(),
    reason="CivRealm Docker container not available",
)


# ============================================================================
# Configuration Tests
# ============================================================================


class TestDockerConnectionConfig:
    """Tests for DockerConnectionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DockerConnectionConfig()
        assert config.host == "localhost"
        assert config.port == 6300
        assert config.client_port == 6001
        assert config.connection_timeout_s == 30.0
        assert config.retry_count == 3
        assert config.retry_delay_s == 2.0

    def test_environment_override(self) -> None:
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "CIVREALM_HOST": "192.168.1.100",
                "CIVREALM_PORT": "6400",
            },
        ):
            config = DockerConnectionConfig()
            assert config.host == "192.168.1.100"
            assert config.port == 6400

    def test_explicit_values_override_environment(self) -> None:
        """Test explicit values override environment."""
        with patch.dict(os.environ, {"CIVREALM_HOST": "env-host"}):
            config = DockerConnectionConfig(host="explicit-host")
            assert config.host == "explicit-host"

    def test_server_address_property(self, docker_config: DockerConnectionConfig) -> None:
        """Test server_address property formatting."""
        assert docker_config.server_address == f"{docker_config.host}:{docker_config.port}"

    def test_retry_delay_exponential_backoff(self, docker_config: DockerConnectionConfig) -> None:
        """Test exponential backoff calculation."""
        base_delay = docker_config.retry_delay_s

        # Attempt 0: base delay
        assert docker_config.get_retry_delay(0) == base_delay

        # Attempt 1: 2x base delay
        assert docker_config.get_retry_delay(1) == base_delay * 2

        # Attempt 2: 4x base delay
        assert docker_config.get_retry_delay(2) == base_delay * 4

    def test_retry_delay_capped_at_max(self, docker_config: DockerConnectionConfig) -> None:
        """Test retry delay is capped at max_backoff_s."""
        # Large attempt number should be capped
        delay = docker_config.get_retry_delay(100)
        assert delay <= docker_config.max_backoff_s

    def test_validation_rejects_invalid_port(self) -> None:
        """Test validation rejects invalid port numbers."""
        with pytest.raises(ValueError, match="Invalid port"):
            DockerConnectionConfig(port=0)

        with pytest.raises(ValueError, match="Invalid port"):
            DockerConnectionConfig(port=70000)

    def test_validation_rejects_negative_timeout(self) -> None:
        """Test validation rejects negative timeout."""
        with pytest.raises(ValueError, match="Invalid connection_timeout_s"):
            DockerConnectionConfig(connection_timeout_s=-1.0)

    def test_to_dict_serialization(self, docker_config: DockerConnectionConfig) -> None:
        """Test configuration serializes to dict."""
        data = docker_config.to_dict()
        assert isinstance(data, dict)
        assert data["host"] == docker_config.host
        assert data["port"] == docker_config.port
        assert data["retry_count"] == docker_config.retry_count

    def test_from_dict_deserialization(self) -> None:
        """Test configuration deserializes from dict."""
        data = {
            "host": "test-host",
            "port": 7000,
            "retry_count": 5,
        }
        config = DockerConnectionConfig.from_dict(data)
        assert config.host == "test-host"
        assert config.port == 7000
        assert config.retry_count == 5


# ============================================================================
# Connection State Tests
# ============================================================================


class TestConnectionState:
    """Tests for ConnectionState tracking."""

    def test_initial_state(self, connection_state: ConnectionState) -> None:
        """Test initial connection state."""
        assert connection_state.status == ConnectionStatus.DISCONNECTED
        assert connection_state.last_connected_at is None
        assert connection_state.last_error is None
        assert connection_state.retry_attempts == 0
        assert connection_state.consecutive_failures == 0

    def test_on_connected_updates_state(self, connection_state: ConnectionState) -> None:
        """Test state update on successful connection."""
        timestamp = time.time()
        connection_state.on_connected(timestamp)

        assert connection_state.status == ConnectionStatus.CONNECTED
        assert connection_state.last_connected_at == timestamp
        assert connection_state.last_error is None
        assert connection_state.retry_attempts == 0
        assert connection_state.consecutive_failures == 0

    def test_on_disconnected_updates_state(self, connection_state: ConnectionState) -> None:
        """Test state update on disconnection."""
        connection_state.on_disconnected("Connection lost")

        assert connection_state.status == ConnectionStatus.DISCONNECTED
        assert connection_state.last_error == "Connection lost"
        assert connection_state.consecutive_failures == 1

    def test_on_retry_updates_state(self, connection_state: ConnectionState) -> None:
        """Test state update during retry."""
        connection_state.on_retry()

        assert connection_state.status == ConnectionStatus.RECONNECTING
        assert connection_state.retry_attempts == 1

    def test_on_failure_updates_state(self, connection_state: ConnectionState) -> None:
        """Test state update on connection failure."""
        connection_state.on_failure("Connection refused")

        assert connection_state.status == ConnectionStatus.FAILED
        assert connection_state.last_error == "Connection refused"
        assert connection_state.consecutive_failures == 1

    def test_reset_clears_state(self, connection_state: ConnectionState) -> None:
        """Test reset clears all state."""
        # Set some state
        connection_state.on_failure("Some error")
        connection_state.retry_attempts = 5

        # Reset
        connection_state.reset()

        assert connection_state.status == ConnectionStatus.DISCONNECTED
        assert connection_state.last_error is None
        assert connection_state.retry_attempts == 0
        assert connection_state.consecutive_failures == 0

    def test_consecutive_failures_accumulate(self, connection_state: ConnectionState) -> None:
        """Test consecutive failures accumulate."""
        connection_state.on_disconnected("Error 1")
        connection_state.on_disconnected("Error 2")
        connection_state.on_failure("Error 3")

        assert connection_state.consecutive_failures == 3

    def test_connected_resets_failures(self, connection_state: ConnectionState) -> None:
        """Test successful connection resets failure count."""
        connection_state.on_failure("Error")
        connection_state.on_failure("Error")
        assert connection_state.consecutive_failures == 2

        connection_state.on_connected(time.time())
        assert connection_state.consecutive_failures == 0


# ============================================================================
# Docker Availability Tests (requires Docker)
# ============================================================================


@requires_docker
class TestDockerContainerAvailable:
    """Tests that require running CivRealm Docker container."""

    def test_can_connect_to_docker_port(self, docker_config: DockerConnectionConfig) -> None:
        """Test basic TCP connection to Docker port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(docker_config.connection_timeout_s)

        try:
            result = sock.connect_ex((docker_config.host, docker_config.port))
            assert result == 0, f"Failed to connect: error code {result}"
        finally:
            sock.close()

    @pytest.mark.requires_env
    def test_civrealm_environment_reset(self, docker_config: DockerConnectionConfig) -> None:
        """Test CivRealm environment can be reset."""
        # Import here to allow test collection without CivRealm
        try:
            from alpha_zero.envs.civrealm_wrapper import (
                CIVREALM_AVAILABLE,
                CivRealmConfig,
                CivRealmWrapper,
            )
        except ImportError:
            pytest.skip("CivRealm wrapper not available")

        if not CIVREALM_AVAILABLE:
            pytest.skip("CivRealm not installed")

        config = CivRealmConfig(
            host=docker_config.host,
            port=docker_config.port,
            task="development_build_city",
        )
        env = CivRealmWrapper(config)

        try:
            obs, info = env.reset()
            assert obs is not None
            assert isinstance(info, dict)
        finally:
            env.close()


# ============================================================================
# Connection Recovery Tests (mocked)
# ============================================================================


class TestConnectionRecovery:
    """Tests for connection recovery logic (mocked)."""

    def test_connection_retry_on_failure(
        self, docker_config: DockerConnectionConfig, connection_state: ConnectionState
    ) -> None:
        """Test connection retry logic on failure."""
        max_retries = docker_config.retry_count

        # Simulate failed connection attempts
        for attempt in range(max_retries):
            connection_state.on_retry()
            connection_state.on_failure(f"Attempt {attempt} failed")

        assert connection_state.retry_attempts == max_retries
        assert connection_state.consecutive_failures == max_retries
        assert connection_state.status == ConnectionStatus.FAILED

    def test_connection_succeeds_after_retries(
        self, docker_config: DockerConnectionConfig, connection_state: ConnectionState
    ) -> None:
        """Test connection succeeds after some retries."""
        # Simulate two failed attempts then success
        connection_state.on_retry()
        connection_state.on_failure("Attempt 0 failed")

        connection_state.on_retry()
        connection_state.on_failure("Attempt 1 failed")

        # Third attempt succeeds
        connection_state.on_retry()
        connection_state.on_connected(time.time())

        assert connection_state.status == ConnectionStatus.CONNECTED
        assert connection_state.consecutive_failures == 0


# ============================================================================
# Timeout Handling Tests
# ============================================================================


class TestTimeoutHandling:
    """Tests for connection timeout handling."""

    def test_socket_timeout_configuration(self, docker_config: DockerConnectionConfig) -> None:
        """Test socket timeout is properly configured."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(docker_config.connection_timeout_s)

        # Verify timeout was set
        assert sock.gettimeout() == docker_config.connection_timeout_s
        sock.close()

    def test_connection_times_out_to_unreachable_host(
        self, docker_config: DockerConnectionConfig, connection_state: ConnectionState
    ) -> None:
        """Test connection times out for unreachable host."""
        # Use a non-routable IP for timeout test
        config = DockerConnectionConfig(
            host="10.255.255.1",  # Non-routable IP
            port=6300,
            connection_timeout_s=0.5,  # Short timeout for test
        )

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(config.connection_timeout_s)

        start_time = time.time()
        try:
            sock.connect((config.host, config.port))
            pytest.fail("Connection should have timed out")
        except (OSError, TimeoutError):
            elapsed = time.time() - start_time
            # Should timeout around the configured timeout
            assert elapsed < config.connection_timeout_s + 1.0
            connection_state.on_failure("Connection timed out")
        finally:
            sock.close()

        assert connection_state.status == ConnectionStatus.FAILED
