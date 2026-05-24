"""Unit tests for SpacetimeDB configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from alpha_zero.spacetimedb.config import SpacetimeDBConfig


class TestSpacetimeDBConfig:
    """Tests for SpacetimeDBConfig dataclass."""

    def test_default_values(self):
        """Config has sensible defaults when created with no env vars."""
        with patch.dict(os.environ, {}, clear=True):
            config = SpacetimeDBConfig()

        assert config.enabled is False
        assert config.host == "http://localhost:3000"
        assert config.module_address == ""
        assert config.auth_token is None
        assert config.ssl_enabled is False
        assert config.reconnect_interval == 5.0
        assert config.max_reconnect_attempts == 10
        assert config.connection_timeout == 30.0
        assert config.compress_weights is True
        assert config.max_weight_blob_mb == 500
        assert config.subscribe_model_versions is True
        assert config.subscribe_training_state is True
        assert config.subscribe_metrics is False

    def test_env_override_enabled(self):
        """CIV6_SPACETIMEDB_ENABLED overrides enabled flag."""
        with patch.dict(os.environ, {"CIV6_SPACETIMEDB_ENABLED": "true"}, clear=True):
            config = SpacetimeDBConfig()
        assert config.enabled is True

    def test_env_override_enabled_false(self):
        """CIV6_SPACETIMEDB_ENABLED=false keeps disabled."""
        with patch.dict(os.environ, {"CIV6_SPACETIMEDB_ENABLED": "false"}, clear=True):
            config = SpacetimeDBConfig()
        assert config.enabled is False

    def test_env_override_host(self):
        """CIV6_SPACETIMEDB_HOST overrides host."""
        with patch.dict(
            os.environ,
            {"CIV6_SPACETIMEDB_HOST": "https://prod.spacetimedb.com:443"},
            clear=True,
        ):
            config = SpacetimeDBConfig()
        assert config.host == "https://prod.spacetimedb.com:443"

    def test_env_override_module(self):
        """CIV6_SPACETIMEDB_MODULE overrides module_address."""
        with patch.dict(
            os.environ,
            {"CIV6_SPACETIMEDB_MODULE": "civ6-alphazero-prod"},
            clear=True,
        ):
            config = SpacetimeDBConfig()
        assert config.module_address == "civ6-alphazero-prod"

    def test_env_override_auth_token(self):
        """CIV6_SPACETIMEDB_AUTH_TOKEN overrides auth_token."""
        with patch.dict(
            os.environ,
            {"CIV6_SPACETIMEDB_AUTH_TOKEN": "secret-token-123"},
            clear=True,
        ):
            config = SpacetimeDBConfig()
        assert config.auth_token == "secret-token-123"  # noqa: S105

    def test_env_override_ssl(self):
        """CIV6_SPACETIMEDB_SSL overrides ssl_enabled."""
        with patch.dict(os.environ, {"CIV6_SPACETIMEDB_SSL": "1"}, clear=True):
            config = SpacetimeDBConfig()
        assert config.ssl_enabled is True

    def test_from_env(self):
        """from_env creates config from environment."""
        with patch.dict(
            os.environ,
            {
                "CIV6_SPACETIMEDB_ENABLED": "true",
                "CIV6_SPACETIMEDB_HOST": "http://test:3000",
                "CIV6_SPACETIMEDB_MODULE": "test-module",
            },
            clear=True,
        ):
            config = SpacetimeDBConfig.from_env()
        assert config.enabled is True
        assert config.host == "http://test:3000"
        assert config.module_address == "test-module"

    def test_from_dict(self):
        """from_dict creates config from dictionary."""
        data = {
            "enabled": True,
            "host": "http://custom:4000",
            "module_address": "my-module",
            "ssl_enabled": True,
            "reconnect_interval": 10.0,
            "compress_weights": False,
            "max_weight_blob_mb": 200,
            "subscribe_metrics": True,
        }
        config = SpacetimeDBConfig.from_dict(data)
        assert config.enabled is True
        assert config.host == "http://custom:4000"
        assert config.module_address == "my-module"
        assert config.ssl_enabled is True
        assert config.reconnect_interval == 10.0
        assert config.compress_weights is False
        assert config.max_weight_blob_mb == 200
        assert config.subscribe_metrics is True

    def test_from_dict_defaults(self):
        """from_dict uses defaults for missing keys."""
        config = SpacetimeDBConfig.from_dict({})
        assert config.enabled is False
        assert config.host == "http://localhost:3000"
        assert config.module_address == ""
        assert config.compress_weights is True

    def test_to_dict(self):
        """to_dict serializes config without auth_token."""
        with patch.dict(os.environ, {}, clear=True):
            config = SpacetimeDBConfig(
                enabled=True,
                host="http://test:3000",
                module_address="test",
                auth_token="secret",  # noqa: S106
            )
        result = config.to_dict()
        assert result["enabled"] is True
        assert result["host"] == "http://test:3000"
        assert result["module_address"] == "test"
        # auth_token should NOT be in serialized output
        assert "auth_token" not in result

    def test_to_dict_roundtrip(self):
        """from_dict(to_dict()) produces equivalent config."""
        with patch.dict(os.environ, {}, clear=True):
            original = SpacetimeDBConfig(
                enabled=True,
                host="http://test:3000",
                module_address="test-module",
                reconnect_interval=3.0,
                max_weight_blob_mb=100,
            )
        data = original.to_dict()
        restored = SpacetimeDBConfig.from_dict(data)
        assert restored.enabled == original.enabled
        assert restored.host == original.host
        assert restored.module_address == original.module_address
        assert restored.reconnect_interval == original.reconnect_interval
        assert restored.max_weight_blob_mb == original.max_weight_blob_mb

    def test_for_local(self):
        """for_local creates config for local development."""
        config = SpacetimeDBConfig.for_local("my-local-module")
        assert config.enabled is True
        assert config.host == "http://localhost:3000"
        assert config.module_address == "my-local-module"

    def test_for_local_custom_host(self):
        """for_local accepts custom host."""
        config = SpacetimeDBConfig.for_local(
            "my-module",
            host="http://localhost:5000",
        )
        assert config.host == "http://localhost:5000"
        assert config.module_address == "my-module"

    @pytest.mark.parametrize(
        "env_val,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("", False),
        ],
    )
    def test_boolean_env_parsing(self, env_val, expected):
        """Boolean env vars parsed correctly."""
        with patch.dict(
            os.environ,
            {"CIV6_SPACETIMEDB_ENABLED": env_val},
            clear=True,
        ):
            config = SpacetimeDBConfig()
        assert config.enabled is expected

    def test_programmatic_override_survives_env(self):
        """Explicitly set values are overridden by env vars."""
        with patch.dict(
            os.environ,
            {"CIV6_SPACETIMEDB_HOST": "http://env-host:3000"},
            clear=True,
        ):
            config = SpacetimeDBConfig(host="http://programmatic:3000")
        # env should override
        assert config.host == "http://env-host:3000"
