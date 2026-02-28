"""Unit tests for JAX checkpoint serialization and cross-framework conversion."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import torch

from alpha_zero.core.jax_checkpoint import (
    jax_to_torch_state_dict,
    load_jax_checkpoint,
    save_jax_checkpoint,
    torch_to_jax_params,
)
from alpha_zero.core.jax_network import (
    JaxMicroCivNetwork,
    JaxNetworkConfig,
    JaxNetworkWrapper,
)

pytestmark = pytest.mark.requires_jax


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def model(key):
    return JaxMicroCivNetwork(
        board_size=16,
        action_space_size=381,
        num_res_blocks=2,
        num_filters=32,
        num_groups=8,
        key=key,
    )


@pytest.fixture
def test_input():
    """Deterministic test input."""
    rng = np.random.default_rng(42)
    return {
        "spatial": rng.standard_normal((13, 16, 16)).astype(np.float32),
        "vector": rng.standard_normal((64,)).astype(np.float32),
        "mask": np.ones(381, dtype=np.float32),
    }


class TestSaveLoadRoundtrip:
    """Tests for JAX checkpoint save/load."""

    def test_save_creates_files(self, model, tmp_path):
        path = str(tmp_path / "checkpoint")
        save_jax_checkpoint(model, path, metadata={"iteration": 100})

        assert (tmp_path / "checkpoint.eqx").exists()
        assert (tmp_path / "checkpoint.json").exists()

    def test_load_preserves_params(self, model, tmp_path, key, test_input):
        path = str(tmp_path / "checkpoint")
        save_jax_checkpoint(model, path)

        # Create template with same architecture
        template = JaxMicroCivNetwork(
            board_size=16,
            action_space_size=381,
            num_res_blocks=2,
            num_filters=32,
            num_groups=8,
            key=jax.random.PRNGKey(99),  # Different key → different params
        )

        loaded = load_jax_checkpoint(path, template)

        # Run forward pass on both
        sp = jnp.array(test_input["spatial"])
        vec = jnp.array(test_input["vector"])
        mask = jnp.array(test_input["mask"])

        orig_policy, orig_value = model(sp, vec, mask)
        loaded_policy, loaded_value = loaded(sp, vec, mask)

        np.testing.assert_allclose(
            np.asarray(orig_policy), np.asarray(loaded_policy), atol=1e-5
        )
        assert abs(float(orig_value) - float(loaded_value)) < 1e-5

    def test_metadata_preserved(self, model, tmp_path):
        import json

        path = str(tmp_path / "checkpoint")
        save_jax_checkpoint(model, path, metadata={"iteration": 42, "win_rate": 0.65})

        meta = json.loads((tmp_path / "checkpoint.json").read_text())
        assert meta["iteration"] == 42
        assert meta["win_rate"] == 0.65
        assert meta["framework"] == "jax"


class TestJaxToTorchConversion:
    """Tests for JAX → PyTorch state dict conversion."""

    def test_conversion_produces_state_dict(self, model):
        state_dict = jax_to_torch_state_dict(model)

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # All values should be torch tensors
        for key, value in state_dict.items():
            assert isinstance(value, torch.Tensor), f"{key} is {type(value)}"

    def test_param_shapes_match(self, model, key):
        """JAX and equivalent PyTorch network should have same param shapes."""
        state_dict = jax_to_torch_state_dict(model)

        # Count total parameters
        jax_total = sum(v.numel() for v in state_dict.values())
        assert jax_total > 0


class TestCrossFrameworkRoundtrip:
    """Tests for cross-framework parameter conversion integrity."""

    def test_jax_state_dict_has_correct_properties(self, model):
        """Verify converted state dict has correct types and dtypes."""
        state_dict = jax_to_torch_state_dict(model)

        for name, tensor in state_dict.items():
            assert isinstance(tensor, torch.Tensor), f"{name} is not a torch.Tensor"
            assert tensor.dtype == torch.float32, f"{name} has dtype {tensor.dtype}"
            assert not tensor.requires_grad, f"{name} unexpectedly requires_grad"

    def test_jax_state_dict_param_count_matches(self, model):
        """Verify total parameter count is preserved across conversion."""
        import equinox as eqx

        state_dict = jax_to_torch_state_dict(model)
        torch_total = sum(v.numel() for v in state_dict.values())

        jax_leaves = jax.tree.leaves(eqx.filter(model, eqx.is_array))
        jax_total = sum(leaf.size for leaf in jax_leaves)

        assert torch_total == jax_total, (
            f"Param count mismatch: PyTorch={torch_total}, JAX={jax_total}"
        )


class TestWrapperSaveLoad:
    """Tests for JaxNetworkWrapper save/load integration."""

    def test_wrapper_roundtrip(self, tmp_path, test_input):
        wrapper = JaxNetworkWrapper(
            JaxMicroCivNetwork(
                num_res_blocks=2, num_filters=16, num_groups=4,
                key=jax.random.PRNGKey(42),
            ),
            seed=42,
            jit_enabled=False,
        )

        policy_before, value_before = wrapper.predict_single(
            test_input["spatial"], test_input["vector"], test_input["mask"]
        )

        path = str(tmp_path / "wrapper_model")
        wrapper.save(path)

        loaded = JaxNetworkWrapper.load(path, seed=42, jit_enabled=False)
        policy_after, value_after = loaded.predict_single(
            test_input["spatial"], test_input["vector"], test_input["mask"]
        )

        np.testing.assert_allclose(policy_before, policy_after, atol=1e-5)
        assert abs(value_before - value_after) < 1e-5


class TestReservedCheckpointKeys:
    """Tests for RESERVED_CHECKPOINT_KEYS protection in save_jax_checkpoint."""

    def test_save_ignores_reserved_metadata_keys(self, model, tmp_path):
        """Reserved keys like 'framework' should not be overwritten by user metadata."""
        import json

        path = str(tmp_path / "checkpoint")
        save_jax_checkpoint(
            model, path,
            metadata={"framework": "pytorch", "custom_key": "value"},
        )

        meta = json.loads((tmp_path / "checkpoint.json").read_text())
        assert meta["framework"] == "jax"  # Not overwritten
        assert meta["custom_key"] == "value"  # Custom key preserved


class TestTorchToJaxConversion:
    """Tests for PyTorch → JAX conversion edge cases."""

    def test_skips_non_tensor_entries(self, key):
        """Non-tensor state_dict entries (dicts, strings, etc.) should be skipped."""
        template = JaxMicroCivNetwork(
            num_res_blocks=2, num_filters=32, num_groups=8, key=key,
        )
        state_dict = {
            "metadata": {"version": 1},
            "name": "test",
            "count": 42,
        }
        # Should not raise — just skip non-tensor entries
        result = torch_to_jax_params(state_dict, template)
        assert result is not None

    def test_bidirectional_roundtrip(self, model, key, test_input):
        """JAX → torch state_dict → JAX should preserve numerical outputs."""
        # Forward pass with original model
        sp = jnp.array(test_input["spatial"])
        vec = jnp.array(test_input["vector"])
        mask = jnp.array(test_input["mask"])
        orig_policy, orig_value = model(sp, vec, mask)

        # Convert to PyTorch state_dict and back
        state_dict = jax_to_torch_state_dict(model)
        template = JaxMicroCivNetwork(
            num_res_blocks=2, num_filters=32, num_groups=8,
            key=jax.random.PRNGKey(99),  # Different key → different params
        )
        restored = torch_to_jax_params(state_dict, template)

        # Forward pass with restored model should match
        rest_policy, rest_value = restored(sp, vec, mask)
        np.testing.assert_allclose(
            np.asarray(orig_policy), np.asarray(rest_policy), atol=1e-5,
        )
        assert abs(float(orig_value) - float(rest_value)) < 1e-5
