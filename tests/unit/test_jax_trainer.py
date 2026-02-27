"""Unit tests for JAX training utilities."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

from alpha_zero.core.jax_network import JaxMicroCivNetwork, JaxNetworkWrapper
from alpha_zero.core.jax_trainer import (
    JaxDataLoader,
    JaxTrainConfig,
    JaxTrainState,
    create_train_state,
    jax_compute_loss,
    jax_train_step,
)
from tests.conftest import MockTrainingExample, make_training_example

pytestmark = pytest.mark.requires_jax


def make_example(board_size: int = 16, action_size: int = 381) -> MockTrainingExample:
    """Create a random training example (delegates to shared fixture)."""
    return make_training_example(board_size=board_size, action_size=action_size)


@pytest.fixture
def model():
    return JaxMicroCivNetwork(
        board_size=16,
        action_space_size=381,
        num_res_blocks=2,
        num_filters=16,
        num_groups=4,
        key=jax.random.PRNGKey(42),
    )


@pytest.fixture
def config():
    return JaxTrainConfig(
        learning_rate=0.001,
        weight_decay=1e-4,
        gradient_clip=1.0,
        jit_enabled=False,
        seed=42,
    )


@pytest.fixture
def train_state(model, config):
    return create_train_state(model, config)


@pytest.fixture
def examples():
    return [make_example() for _ in range(32)]


class TestJaxTrainConfig:
    """Tests for training configuration."""

    def test_default_values(self):
        config = JaxTrainConfig()
        assert config.learning_rate == 0.001
        assert config.gradient_clip == 1.0
        assert config.jit_enabled is True

    def test_from_training_config(self):
        class MockConfig:
            learning_rate = 0.01
            weight_decay = 0.001
            gradient_clip = 0.5

        config = JaxTrainConfig.from_training_config(MockConfig())
        assert config.learning_rate == 0.01
        assert config.weight_decay == 0.001
        assert config.gradient_clip == 0.5


class TestJaxTrainState:
    """Tests for training state."""

    def test_create_train_state(self, train_state):
        assert train_state.step == 0
        assert train_state.model is not None
        assert train_state.opt_state is not None
        assert train_state.rng_key is not None

    def test_increment_step(self, train_state):
        new_state = train_state.increment_step()
        assert new_state.step == 1
        assert train_state.step == 0  # Original unchanged


class TestJaxComputeLoss:
    """Tests for loss computation."""

    def test_loss_is_finite(self, model):
        key = jax.random.PRNGKey(0)
        spatial = jax.random.normal(key, (13, 16, 16))
        vector = jax.random.normal(key, (64,))
        mask = jnp.ones(381)
        target_policy = jnp.ones(381) / 381
        target_value = jnp.array(0.5)

        loss, metrics = jax_compute_loss(
            model, spatial, vector, mask, target_policy, target_value
        )

        assert jnp.isfinite(loss)
        assert "total_loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics

    def test_loss_nonnegative(self, model):
        key = jax.random.PRNGKey(0)
        spatial = jax.random.normal(key, (13, 16, 16))
        vector = jax.random.normal(key, (64,))
        mask = jnp.ones(381)
        target_policy = jnp.ones(381) / 381
        target_value = jnp.array(0.0)

        loss, metrics = jax_compute_loss(
            model, spatial, vector, mask, target_policy, target_value
        )

        assert float(loss) >= 0

    def test_action_mask_applied(self, model):
        key = jax.random.PRNGKey(0)
        spatial = jax.random.normal(key, (13, 16, 16))
        vector = jax.random.normal(key, (64,))

        # Only first 10 actions valid
        mask = jnp.zeros(381).at[:10].set(1.0)
        target_policy = jnp.zeros(381).at[:10].set(0.1)
        target_value = jnp.array(0.0)

        loss, _ = jax_compute_loss(
            model, spatial, vector, mask, target_policy, target_value
        )

        assert jnp.isfinite(loss)


class TestJaxTrainStep:
    """Tests for the training step."""

    def test_single_step(self, train_state, examples):
        loader = JaxDataLoader(examples, batch_size=8, shuffle=False, seed=0)
        batch = next(iter(loader))

        new_state, metrics = jax_train_step(train_state, batch)

        assert new_state.step == 1
        assert "total_loss" in metrics
        assert metrics["total_loss"] > 0

    def test_loss_decreases_over_steps(self, train_state, examples):
        """After multiple steps, loss should generally decrease."""
        loader = JaxDataLoader(examples, batch_size=16, shuffle=False, seed=0)
        batch = next(iter(loader))

        state = train_state
        losses = []
        for _ in range(5):
            state, metrics = jax_train_step(state, batch)
            losses.append(metrics["total_loss"])

        # Loss should decrease (at least the last value should be less than the first)
        assert losses[-1] < losses[0]

    def test_step_counter_increments(self, train_state, examples):
        loader = JaxDataLoader(examples, batch_size=8, shuffle=False, seed=0)
        batch = next(iter(loader))

        state = train_state
        for i in range(3):
            state, _ = jax_train_step(state, batch)
            assert state.step == i + 1


class TestJaxDataLoader:
    """Tests for data loading."""

    def test_iteration(self, examples):
        loader = JaxDataLoader(examples, batch_size=8, shuffle=False, seed=0)

        batches = list(loader)
        assert len(batches) == len(examples) // 8

    def test_batch_shapes(self, examples):
        loader = JaxDataLoader(examples, batch_size=8, shuffle=False, seed=0)
        batch = next(iter(loader))

        assert batch["spatial"].shape == (8, 13, 16, 16)
        assert batch["vector"].shape == (8, 64)
        assert batch["action_mask"].shape == (8, 381)
        assert batch["policy"].shape == (8, 381)
        assert batch["value"].shape == (8,)

    def test_shuffle(self, examples):
        loader1 = JaxDataLoader(examples, batch_size=8, shuffle=True, seed=0)
        loader2 = JaxDataLoader(examples, batch_size=8, shuffle=True, seed=1)

        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # Different seeds should produce different orderings
        # (not guaranteed but extremely likely with 32 examples)
        # Check values are not identical
        assert not np.array_equal(np.asarray(batch1["value"]), np.asarray(batch2["value"]))

    def test_len(self, examples):
        loader = JaxDataLoader(examples, batch_size=8)
        assert len(loader) == 4  # 32 // 8
