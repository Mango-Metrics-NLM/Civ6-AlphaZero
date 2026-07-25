"""Unit tests for JAX/Equinox neural networks."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

from alpha_zero.core.jax_network import (
    JaxCiv6Network,
    JaxGroupNorm,
    JaxMicroCivNetwork,
    JaxNetworkConfig,
    JaxNetworkWrapper,
    JaxResidualBlock,
    JaxSpatialEncoder,
    JaxVectorEncoder,
    create_jax_network,
)

pytestmark = pytest.mark.requires_jax


class TestJaxGroupNorm:
    """Tests for GroupNorm implementation."""

    def test_output_shape(self):
        gn = JaxGroupNorm(num_channels=64, num_groups=8)
        x = jnp.ones((64, 16, 16))
        y = gn(x)
        assert y.shape == (64, 16, 16)

    def test_normalizes_channels(self):
        gn = JaxGroupNorm(num_channels=8, num_groups=2)
        x = jnp.ones((8, 4, 4)) * 5.0
        y = gn(x)
        # After normalization of constant input, output should be near zero
        # (before affine: exactly zero; after affine with weight=1, bias=0: still zero)
        assert jnp.allclose(y, jnp.zeros_like(y), atol=1e-5)


class TestJaxResidualBlock:
    """Tests for residual block."""

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    def test_output_shape(self, key):
        block = JaxResidualBlock(num_filters=64, num_groups=8, key=key)
        x = jax.random.normal(key, (64, 16, 16))
        y = block(x)
        assert y.shape == x.shape

    def test_residual_connection(self, key):
        """Output includes input (residual path)."""
        block = JaxResidualBlock(num_filters=64, num_groups=8, key=key)
        x = jax.random.normal(key, (64, 16, 16))
        y = block(x)
        # Output should not be zero (has residual)
        assert jnp.abs(y).sum() > 0


class TestJaxSpatialEncoder:
    """Tests for spatial encoder."""

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    def test_output_shape(self, key):
        encoder = JaxSpatialEncoder(
            in_channels=13, num_filters=64, num_res_blocks=2, num_groups=8, key=key
        )
        x = jax.random.normal(key, (13, 16, 16))
        y = encoder(x)
        assert y.shape == (64, 16, 16)


class TestJaxVectorEncoder:
    """Tests for vector encoder."""

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    def test_output_shape(self, key):
        encoder = JaxVectorEncoder(input_dim=64, hidden_dim=128, output_dim=128, key=key)
        x = jax.random.normal(key, (64,))
        y = encoder(x, inference=True)
        assert y.shape == (128,)


class TestJaxMicroCivNetwork:
    """Tests for MicroCiv JAX network."""

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def network(self, key):
        return JaxMicroCivNetwork(
            board_size=16,
            action_space_size=381,
            num_res_blocks=2,
            num_filters=32,
            num_groups=8,
            key=key,
        )

    def test_forward_shapes(self, network, key):
        spatial = jax.random.normal(key, (13, 16, 16))
        vector = jax.random.normal(key, (64,))
        policy_logits, value = network(spatial, vector)

        assert policy_logits.shape == (381,)
        assert value.shape == ()

    def test_value_range(self, network, key):
        spatial = jax.random.normal(key, (13, 16, 16))
        vector = jax.random.normal(key, (64,))
        _, value = network(spatial, vector)

        assert float(value) >= -1.0
        assert float(value) <= 1.0

    def test_action_masking(self, network, key):
        spatial = jax.random.normal(key, (13, 16, 16))
        vector = jax.random.normal(key, (64,))
        mask = jnp.zeros(381)
        mask = mask.at[:10].set(1.0)

        policy_logits, _ = network(spatial, vector, mask)

        # Invalid actions should have very negative logits
        assert jnp.all(policy_logits[10:] < -1e8)

    def test_no_mask_all_actions_valid(self, network, key):
        spatial = jax.random.normal(key, (13, 16, 16))
        vector = jax.random.normal(key, (64,))
        policy_logits, _ = network(spatial, vector, action_mask=None)

        # Should produce finite logits for all actions
        assert jnp.all(jnp.isfinite(policy_logits))

    def test_config_stored(self, network):
        assert network.config.spatial_channels == 13
        assert network.config.spatial_size == 16
        assert network.config.action_space_size == 381


class TestJaxCiv6Network:
    """Tests for full Civ6 JAX network."""

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def network(self, key):
        config = JaxNetworkConfig(
            spatial_channels=84,
            spatial_size=44,
            vector_dim=512,
            num_res_blocks=2,
            num_filters=64,
            action_space_size=10000,
            num_groups=8,
        )
        return JaxCiv6Network(config=config, key=key)

    def test_forward_shapes(self, network, key):
        spatial = jax.random.normal(key, (84, 44, 44))
        vector = jax.random.normal(key, (512,))
        policy_logits, value = network(spatial, vector)

        assert policy_logits.shape == (10000,)
        assert value.shape == ()


class TestJaxNetworkWrapper:
    """Tests for the stateful wrapper."""

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def wrapper(self, key):
        model = JaxMicroCivNetwork(
            board_size=16,
            action_space_size=381,
            num_res_blocks=2,
            num_filters=32,
            num_groups=8,
            key=key,
        )
        return JaxNetworkWrapper(model, seed=42, jit_enabled=False)

    def test_predict_single_sample(self, wrapper):
        spatial = np.random.randn(13, 16, 16).astype(np.float32)
        vector = np.random.randn(64).astype(np.float32)
        mask = np.ones(381, dtype=np.float32)

        policy, value = wrapper.predict_single(spatial, vector, mask)

        assert policy.shape == (381,)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_predict_batch(self, wrapper):
        spatial = np.random.randn(4, 13, 16, 16).astype(np.float32)
        vector = np.random.randn(4, 64).astype(np.float32)
        mask = np.ones((4, 381), dtype=np.float32)

        policy, value = wrapper.predict(spatial, vector, mask)

        assert policy.shape == (4, 381)
        assert value.shape == (4,)

    def test_predict_returns_probabilities(self, wrapper):
        spatial = np.random.randn(13, 16, 16).astype(np.float32)
        vector = np.random.randn(64).astype(np.float32)
        mask = np.ones(381, dtype=np.float32)

        policy, _ = wrapper.predict_single(spatial, vector, mask)

        assert np.all(policy >= 0)
        assert abs(policy.sum() - 1.0) < 1e-5

    def test_param_count(self, wrapper):
        assert wrapper.param_count > 0

    def test_get_set_params(self, wrapper):
        params = wrapper.get_params()
        assert params is not None

        # Setting same params should not change outputs
        spatial = np.random.randn(13, 16, 16).astype(np.float32)
        vector = np.random.randn(64).astype(np.float32)
        mask = np.ones(381, dtype=np.float32)

        policy1, value1 = wrapper.predict_single(spatial, vector, mask)
        wrapper.set_params(params)
        policy2, value2 = wrapper.predict_single(spatial, vector, mask)

        np.testing.assert_allclose(policy1, policy2, atol=1e-6)
        assert abs(value1 - value2) < 1e-6

    def test_save_load_roundtrip(self, wrapper, tmp_path):
        spatial = np.random.randn(13, 16, 16).astype(np.float32)
        vector = np.random.randn(64).astype(np.float32)
        mask = np.ones(381, dtype=np.float32)

        policy_before, value_before = wrapper.predict_single(spatial, vector, mask)

        path = str(tmp_path / "test_model")
        wrapper.save(path, metadata={"test": True})

        loaded = JaxNetworkWrapper.load(path, seed=42, jit_enabled=False)
        policy_after, value_after = loaded.predict_single(spatial, vector, mask)

        np.testing.assert_allclose(policy_before, policy_after, atol=1e-5)
        assert abs(value_before - value_after) < 1e-5


class TestJaxNetworkConfig:
    """Tests for configuration."""

    def test_default_values(self):
        config = JaxNetworkConfig()
        assert config.spatial_channels == 13
        assert config.num_groups == 8

    def test_to_dict_roundtrip(self):
        config = JaxNetworkConfig(num_filters=128, num_groups=16)
        d = config.to_dict()
        restored = JaxNetworkConfig.from_dict(d)
        assert restored.num_filters == 128
        assert restored.num_groups == 16

    def test_from_network_config(self):
        """Test conversion from PyTorch NetworkConfig."""
        from alpha_zero.core.civ6_network import NetworkConfig

        torch_config = NetworkConfig(num_filters=128, num_res_blocks=10)
        jax_config = JaxNetworkConfig.from_network_config(torch_config)
        assert jax_config.num_filters == 128
        assert jax_config.num_res_blocks == 10


class TestJaxNetworkWrapperValidation:
    """Tests for input validation error paths."""

    @pytest.fixture
    def wrapper(self):
        model = JaxMicroCivNetwork(
            board_size=16,
            action_space_size=381,
            num_res_blocks=2,
            num_filters=32,
            num_groups=8,
            key=jax.random.PRNGKey(42),
        )
        return JaxNetworkWrapper(model, seed=42, jit_enabled=False)

    def test_invalid_spatial_ndim_raises(self, wrapper):
        """2D spatial input should be rejected."""
        spatial = np.zeros((13, 16), dtype=np.float32)
        vector = np.zeros(64, dtype=np.float32)
        mask = np.ones(381, dtype=np.float32)

        with pytest.raises(ValueError, match="spatial must be 3D"):
            wrapper.predict(spatial, vector, mask)

    def test_invalid_vector_ndim_raises(self, wrapper):
        """3D vector input should be rejected."""
        spatial = np.zeros((13, 16, 16), dtype=np.float32)
        vector = np.zeros((1, 1, 64), dtype=np.float32)
        mask = np.ones(381, dtype=np.float32)

        with pytest.raises(ValueError, match="vector must be 1D"):
            wrapper.predict(spatial, vector, mask)

    def test_groupnorm_indivisible_raises(self):
        """num_channels not divisible by num_groups should raise."""
        with pytest.raises(ValueError, match="divisible"):
            JaxGroupNorm(num_channels=7, num_groups=8)

    def test_residual_block_indivisible_raises(self):
        """num_filters not divisible by num_groups should raise."""
        with pytest.raises(ValueError, match="divisible"):
            JaxResidualBlock(num_filters=7, num_groups=8, key=jax.random.PRNGKey(0))


class TestJaxNetworkWrapperJIT:
    """Tests for JIT-compiled wrapper path (production default)."""

    def test_jit_single_matches_nojit(self):
        """JIT-compiled single predict should match non-JIT."""
        model = JaxMicroCivNetwork(
            board_size=16,
            action_space_size=381,
            num_res_blocks=2,
            num_filters=32,
            num_groups=8,
            key=jax.random.PRNGKey(42),
        )
        wrapper_nojit = JaxNetworkWrapper(model, seed=42, jit_enabled=False)
        wrapper_jit = JaxNetworkWrapper(model, seed=42, jit_enabled=True)

        spatial = np.zeros((13, 16, 16), dtype=np.float32)
        vector = np.zeros(64, dtype=np.float32)
        mask = np.ones(381, dtype=np.float32)

        p1, v1 = wrapper_nojit.predict_single(spatial, vector, mask)
        p2, v2 = wrapper_jit.predict_single(spatial, vector, mask)

        np.testing.assert_allclose(p1, p2, atol=1e-6)
        assert abs(v1 - v2) < 1e-6

    def test_jit_batched_matches_nojit(self):
        """JIT-compiled batched predict should match non-JIT."""
        model = JaxMicroCivNetwork(
            board_size=16,
            action_space_size=381,
            num_res_blocks=2,
            num_filters=32,
            num_groups=8,
            key=jax.random.PRNGKey(42),
        )
        wrapper_nojit = JaxNetworkWrapper(model, seed=42, jit_enabled=False)
        wrapper_jit = JaxNetworkWrapper(model, seed=42, jit_enabled=True)

        spatial = np.zeros((4, 13, 16, 16), dtype=np.float32)
        vector = np.zeros((4, 64), dtype=np.float32)
        mask = np.ones((4, 381), dtype=np.float32)

        p1, v1 = wrapper_nojit.predict(spatial, vector, mask)
        p2, v2 = wrapper_jit.predict(spatial, vector, mask)

        np.testing.assert_allclose(p1, p2, atol=1e-5)
        np.testing.assert_allclose(v1, v2, atol=1e-5)


class TestCreateJaxNetwork:
    """Tests for factory function."""

    def test_creates_micro_network(self):
        from alpha_zero.core.civ6_network import NetworkConfig

        config = NetworkConfig()
        wrapper = create_jax_network(config, board_size=16, action_space_size=381)
        assert isinstance(wrapper, JaxNetworkWrapper)
        assert wrapper.param_count > 0

    def test_creates_large_network(self):
        from alpha_zero.core.civ6_network import NetworkConfig

        config = NetworkConfig(spatial_channels=84)
        wrapper = create_jax_network(config, board_size=44, action_space_size=10000, seed=123)
        assert isinstance(wrapper, JaxNetworkWrapper)
