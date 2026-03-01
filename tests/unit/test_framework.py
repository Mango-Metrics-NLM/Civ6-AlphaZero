"""Unit tests for framework abstraction layer."""

import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from alpha_zero.core.framework import (
    MASKED_LOGIT_VALUE,
    MICROCIV_MAX_BOARD_SIZE,
    MICROCIV_MAX_CHANNELS,
    RESERVED_CHECKPOINT_KEYS,
    FrameworkType,
    framework_available,
    from_numpy,
    get_framework,
    is_microciv_config,
    to_numpy,
)


class TestConstants:
    """Tests for shared constants — prevent accidental changes."""

    def test_masked_logit_value(self):
        assert MASKED_LOGIT_VALUE == -1e9

    def test_microciv_max_board_size(self):
        assert MICROCIV_MAX_BOARD_SIZE == 16

    def test_microciv_max_channels(self):
        assert MICROCIV_MAX_CHANNELS == 37

    def test_reserved_checkpoint_keys(self):
        assert RESERVED_CHECKPOINT_KEYS == frozenset({"framework", "config", "param_count"})
        assert isinstance(RESERVED_CHECKPOINT_KEYS, frozenset)


class TestIsMicrocivConfig:
    """Tests for is_microciv_config() boundary logic."""

    def test_small_board_small_channels(self):
        assert is_microciv_config(8, 5) is True

    def test_exact_boundary(self):
        assert is_microciv_config(16, 37) is True

    def test_board_too_large(self):
        assert is_microciv_config(17, 13) is False

    def test_channels_too_many(self):
        assert is_microciv_config(16, 38) is False

    def test_full_civ6_config(self):
        assert is_microciv_config(44, 84) is False


class TestFrameworkType:
    """Tests for FrameworkType enum."""

    def test_pytorch_value(self):
        assert FrameworkType.PYTORCH.value == "pytorch"

    def test_jax_value(self):
        assert FrameworkType.JAX.value == "jax"


class TestGetFramework:
    """Tests for get_framework()."""

    def test_default_is_pytorch(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CIV6AZ_FRAMEWORK", None)
            result = get_framework()
            assert result == FrameworkType.PYTORCH

    def test_override_parameter(self):
        result = get_framework(override="jax")
        assert result == FrameworkType.JAX

    def test_override_case_insensitive(self):
        result = get_framework(override="JAX")
        assert result == FrameworkType.JAX

    def test_override_with_whitespace(self):
        result = get_framework(override="  pytorch  ")
        assert result == FrameworkType.PYTORCH

    def test_env_var(self):
        with patch.dict(os.environ, {"CIV6AZ_FRAMEWORK": "jax"}):
            result = get_framework()
            assert result == FrameworkType.JAX

    def test_override_takes_precedence_over_env(self):
        with patch.dict(os.environ, {"CIV6AZ_FRAMEWORK": "jax"}):
            result = get_framework(override="pytorch")
            assert result == FrameworkType.PYTORCH

    def test_unknown_framework_raises(self):
        with pytest.raises(ValueError, match="Unknown framework"):
            get_framework(override="tensorflow")


class TestFrameworkAvailable:
    """Tests for framework_available()."""

    def test_pytorch_available(self):
        assert framework_available(FrameworkType.PYTORCH) is True

    def test_pytorch_available_by_string(self):
        assert framework_available("pytorch") is True

    @pytest.mark.requires_jax
    def test_jax_available_when_installed(self):
        assert framework_available(FrameworkType.JAX) is True

    def test_unknown_framework_returns_false(self):
        assert framework_available("tensorflow") is False


class TestToNumpy:
    """Tests for to_numpy()."""

    def test_numpy_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert result is arr

    def test_torch_tensor(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_torch_tensor_with_grad(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)

    @pytest.mark.requires_jax
    def test_jax_array(self):
        import jax.numpy as jnp

        arr = jnp.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Cannot convert"):
            to_numpy("not a tensor")


class TestFromNumpy:
    """Tests for from_numpy()."""

    def test_to_pytorch(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = from_numpy(arr, FrameworkType.PYTORCH)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        np.testing.assert_array_equal(result.numpy(), arr)

    def test_to_pytorch_by_string(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = from_numpy(arr, "pytorch")
        assert isinstance(result, torch.Tensor)

    @pytest.mark.requires_jax
    def test_to_jax(self):
        import jax.numpy as jnp

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = from_numpy(arr, FrameworkType.JAX)
        assert isinstance(result, jnp.ndarray)
        np.testing.assert_array_equal(np.asarray(result), arr)

    @pytest.mark.requires_jax
    def test_to_jax_by_string(self):
        import jax.numpy as jnp

        arr = np.array([1.0], dtype=np.float32)
        result = from_numpy(arr, "jax")
        assert isinstance(result, jnp.ndarray)

    def test_unknown_framework_string_raises(self):
        arr = np.array([1.0])
        with pytest.raises(ValueError):
            from_numpy(arr, "tensorflow")
