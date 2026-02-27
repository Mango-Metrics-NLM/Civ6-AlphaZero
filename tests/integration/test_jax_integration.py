"""Integration tests for JAX backend.

Tests end-to-end workflows: network creation via factory, MCTS integration,
training loop, and cross-framework checkpoint conversion.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from alpha_zero.core.jax_mcts_adapter import JaxMCTSAdapter
from alpha_zero.core.jax_network import (
    JaxMicroCivNetwork,
    JaxNetworkWrapper,
    create_jax_network,
)
from alpha_zero.core.jax_trainer import (
    JaxDataLoader,
    JaxTrainConfig,
    create_train_state,
    jax_train_step,
)

pytestmark = [pytest.mark.requires_jax, pytest.mark.integration]


@dataclass
class MockTrainingExample:
    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    policy: np.ndarray
    value: float


def make_example(board_size: int = 16, action_size: int = 381):
    policy = np.random.dirichlet(np.ones(action_size)).astype(np.float32)
    return MockTrainingExample(
        spatial=np.random.randn(13, board_size, board_size).astype(np.float32),
        vector=np.random.randn(64).astype(np.float32),
        action_mask=np.ones(action_size, dtype=np.float32),
        policy=policy,
        value=np.random.uniform(-1, 1),
    )


class TestFactoryJaxCreation:
    """Test that the factory creates JAX networks correctly."""

    def test_create_network_with_framework_param(self):
        from alpha_zero.core.civ6_network import NetworkConfig
        from alpha_zero.utils.factories import create_network

        config = NetworkConfig()
        net = create_network(
            config, board_size=16, action_space_size=381, framework="jax"
        )
        assert isinstance(net, JaxNetworkWrapper)

    def test_create_network_with_env_var(self, monkeypatch):
        from alpha_zero.core.civ6_network import NetworkConfig
        from alpha_zero.utils.factories import create_network

        monkeypatch.setenv("CIV6AZ_FRAMEWORK", "jax")
        config = NetworkConfig()
        net = create_network(config, board_size=16, action_space_size=381)
        assert isinstance(net, JaxNetworkWrapper)

    def test_create_network_default_is_pytorch(self, monkeypatch):
        from alpha_zero.core.civ6_network import NetworkConfig
        from alpha_zero.utils.factories import create_network

        monkeypatch.delenv("CIV6AZ_FRAMEWORK", raising=False)
        config = NetworkConfig()
        net = create_network(config, board_size=16, action_space_size=381)
        assert isinstance(net, torch.nn.Module)

    def test_create_jax_network_directly(self):
        from alpha_zero.core.civ6_network import NetworkConfig

        config = NetworkConfig()
        wrapper = create_jax_network(config, board_size=16, action_space_size=381)

        spatial = np.random.randn(1, 13, 16, 16).astype(np.float32)
        vector = np.random.randn(1, 64).astype(np.float32)
        mask = np.ones((1, 381), dtype=np.float32)

        policy, value = wrapper.predict(spatial, vector, mask)
        assert policy.shape == (1, 381)
        assert value.shape == (1,)


class TestJaxMCTSIntegration:
    """Test JAX network with MCTS search."""

    @pytest.fixture
    def adapter(self):
        from alpha_zero.envs.microciv import MicroCivEnv

        # Use actual environment action space size for correct shape matching
        env = MicroCivEnv(board_size=16, max_turns=50, seed=0)
        env.reset()
        action_space_size = env.action_space.n

        model = JaxMicroCivNetwork(
            board_size=16,
            action_space_size=action_space_size,
            num_res_blocks=2,
            num_filters=16,
            num_groups=4,
            key=jax.random.PRNGKey(42),
        )
        wrapper = JaxNetworkWrapper(model, seed=42, jit_enabled=False)
        return JaxMCTSAdapter(wrapper, device="cpu")

    def test_mcts_with_jax_adapter(self, adapter):
        """Run Civ6MCTS search using JAX network via adapter."""
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=16, max_turns=50, seed=42)
        env.reset()

        mcts_config = MCTSConfig(
            num_simulations=10,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            temperature=1.0,
        )
        mcts = Civ6MCTS(adapter, mcts_config, device="cpu")

        action_probs, root_value = mcts.search(env)
        assert action_probs.shape[0] > 0
        assert isinstance(root_value, float)

    def test_play_full_game(self, adapter):
        """Play a complete game using JAX network + SimpleMCTS."""
        from alpha_zero.core.civ6_mcts import SimpleMCTS
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=8, max_turns=20, seed=42)
        env.reset()

        mcts = SimpleMCTS(num_simulations=5, c_puct=1.5)

        steps = 0
        done = False
        while not done and steps < 50:
            action_probs, _ = mcts.search(env)
            valid_actions = np.where(action_probs > 0)[0]
            if len(valid_actions) == 0:
                break
            action = np.random.choice(valid_actions, p=action_probs[valid_actions] / action_probs[valid_actions].sum())
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps > 0


class TestJaxTrainingLoop:
    """Test end-to-end JAX training."""

    def test_training_reduces_loss(self):
        """Train for a few steps and verify loss decreases."""
        model = JaxMicroCivNetwork(
            board_size=16,
            action_space_size=381,
            num_res_blocks=2,
            num_filters=16,
            num_groups=4,
            key=jax.random.PRNGKey(42),
        )

        config = JaxTrainConfig(
            learning_rate=0.01,
            weight_decay=0.0,
            gradient_clip=1.0,
            jit_enabled=False,
            seed=42,
        )

        state = create_train_state(model, config)
        examples = [make_example() for _ in range(32)]
        loader = JaxDataLoader(examples, batch_size=16, shuffle=False, seed=0)
        batch = next(iter(loader))

        losses = []
        for _ in range(10):
            state, metrics = jax_train_step(state, batch)
            losses.append(metrics["total_loss"])

        # Loss should decrease
        assert losses[-1] < losses[0]
        assert state.step == 10


class TestCrossFrameworkCheckpoint:
    """Test checkpoint conversion between PyTorch and JAX."""

    def test_jax_save_load_produces_same_output(self, tmp_path):
        """Save JAX model, load it, verify identical outputs."""
        wrapper = JaxNetworkWrapper(
            JaxMicroCivNetwork(
                num_res_blocks=2, num_filters=16, num_groups=4,
                key=jax.random.PRNGKey(42),
            ),
            seed=42,
            jit_enabled=False,
        )

        spatial = np.random.randn(13, 16, 16).astype(np.float32)
        vector = np.random.randn(64).astype(np.float32)
        mask = np.ones(381, dtype=np.float32)

        policy_before, value_before = wrapper.predict_single(spatial, vector, mask)

        path = str(tmp_path / "model")
        wrapper.save(path)
        loaded = JaxNetworkWrapper.load(path, seed=42, jit_enabled=False)

        policy_after, value_after = loaded.predict_single(spatial, vector, mask)

        np.testing.assert_allclose(policy_before, policy_after, atol=1e-5)
        assert abs(value_before - value_after) < 1e-5

    def test_jax_to_torch_state_dict(self):
        """Convert JAX model to PyTorch state dict format."""
        from alpha_zero.core.jax_checkpoint import jax_to_torch_state_dict

        model = JaxMicroCivNetwork(
            num_res_blocks=2, num_filters=16, num_groups=4,
            key=jax.random.PRNGKey(42),
        )

        state_dict = jax_to_torch_state_dict(model)

        assert len(state_dict) > 0
        for name, tensor in state_dict.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32


class TestFrameworkAbstraction:
    """Test the framework abstraction layer."""

    def test_to_numpy_jax(self):
        from alpha_zero.core.framework import to_numpy

        arr = jnp.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)

    def test_from_numpy_jax(self):
        from alpha_zero.core.framework import from_numpy

        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = from_numpy(arr, "jax")
        assert isinstance(result, jnp.ndarray)

    def test_jax_available(self):
        from alpha_zero.core.framework import framework_available

        assert framework_available("jax") is True
        assert framework_available("pytorch") is True
