"""Unit tests for JAX debug utilities."""

import logging

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from alpha_zero.core.jax_debug import (
    JaxDebugLogger,
    jax_debug_mode,
    validate_jax_outputs,
)

pytestmark = pytest.mark.requires_jax


class TestJaxDebugLogger:
    """Tests for JaxDebugLogger."""

    def test_from_env_default_disabled(self, monkeypatch):
        monkeypatch.delenv("CIV6AZ_JAX_DEBUG", raising=False)
        logger = JaxDebugLogger.from_env()
        assert logger.enabled is False

    def test_from_env_enabled(self, monkeypatch):
        monkeypatch.setenv("CIV6AZ_JAX_DEBUG", "true")
        logger = JaxDebugLogger.from_env()
        assert logger.enabled is True

    def test_log_array_when_disabled(self):
        logger = JaxDebugLogger(enabled=False)
        # Should not raise
        logger.log_array("test", jnp.ones((3, 3)))

    def test_log_array_when_enabled(self, caplog):
        logger = JaxDebugLogger(enabled=True, log_level="DEBUG")
        with caplog.at_level(logging.DEBUG):
            logger.log_array("test_arr", jnp.array([1.0, 2.0, 3.0]))
        assert "test_arr" in caplog.text

    def test_log_params(self, caplog):
        from alpha_zero.core.jax_network import JaxMicroCivNetwork

        logger = JaxDebugLogger(enabled=True, log_level="DEBUG")
        model = JaxMicroCivNetwork(
            num_res_blocks=1, num_filters=16, num_groups=4, key=jax.random.PRNGKey(0)
        )
        with caplog.at_level(logging.DEBUG):
            logger.log_params(model, label="test_model")
        assert "test_model" in caplog.text

    def test_log_compilation(self, caplog):
        logger = JaxDebugLogger(enabled=True, log_level="DEBUG")
        with caplog.at_level(logging.DEBUG):
            logger.log_compilation("forward", {"spatial": (13, 16, 16)})
        assert "forward" in caplog.text


class TestJaxDebugMode:
    """Tests for jax_debug_mode context manager."""

    def test_disables_jit(self):
        assert jax.config.jax_disable_jit is False
        with jax_debug_mode():
            assert jax.config.jax_disable_jit is True
        assert jax.config.jax_disable_jit is False

    def test_enables_nan_check(self):
        original = jax.config.jax_debug_nans
        with jax_debug_mode():
            assert jax.config.jax_debug_nans is True
        assert jax.config.jax_debug_nans == original

    def test_restores_on_exception(self):
        original_jit = jax.config.jax_disable_jit
        try:
            with jax_debug_mode():
                raise ValueError("test error")
        except ValueError:
            pass
        assert jax.config.jax_disable_jit == original_jit


class TestValidateJaxOutputs:
    """Tests for validate_jax_outputs."""

    def test_valid_outputs(self):
        policy = jnp.ones(381) / 381
        value = jnp.array(0.5)
        # Should not raise
        validate_jax_outputs(policy, value)

    def test_valid_batched_outputs(self):
        policy = jnp.ones((4, 381)) / 381
        value = jnp.array([0.5, -0.3, 0.1, 0.0])
        validate_jax_outputs(policy, value)

    def test_rejects_3d_policy(self):
        policy = jnp.ones((2, 3, 381))
        value = jnp.array(0.5)
        with pytest.raises(ValueError, match="1D or 2D"):
            validate_jax_outputs(policy, value)

    def test_rejects_nan_policy(self):
        policy = jnp.array([1.0, float("nan"), 0.5])
        value = jnp.array(0.5)
        with pytest.raises(ValueError, match="NaN"):
            validate_jax_outputs(policy, value)

    def test_rejects_nan_value(self):
        policy = jnp.ones(10)
        value = jnp.array(float("nan"))
        with pytest.raises(ValueError, match="NaN"):
            validate_jax_outputs(policy, value)

    def test_checks_action_size(self):
        policy = jnp.ones(381)
        value = jnp.array(0.5)
        validate_jax_outputs(policy, value, expected_action_size=381)

    def test_rejects_wrong_action_size(self):
        policy = jnp.ones(100)
        value = jnp.array(0.5)
        with pytest.raises(ValueError, match="action dim"):
            validate_jax_outputs(policy, value, expected_action_size=381)
