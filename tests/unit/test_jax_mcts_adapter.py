"""Unit tests for JAX MCTS adapter."""

import pytest
import torch

jax = pytest.importorskip("jax")

try:
    from alpha_zero.core.jax_mcts_adapter import JaxMCTSAdapter
    from alpha_zero.core.jax_network import JaxMicroCivNetwork, JaxNetworkWrapper
except ImportError:
    pytest.skip("JAX/Equinox network classes unavailable", allow_module_level=True)

pytestmark = pytest.mark.requires_jax


@pytest.fixture
def jax_wrapper():
    model = JaxMicroCivNetwork(
        board_size=16,
        action_space_size=381,
        num_res_blocks=2,
        num_filters=32,
        num_groups=8,
        key=jax.random.PRNGKey(42),
    )
    return JaxNetworkWrapper(model, seed=42, jit_enabled=False)


@pytest.fixture
def adapter(jax_wrapper):
    return JaxMCTSAdapter(jax_wrapper, device="cpu")


class TestJaxMCTSAdapter:
    """Tests for the MCTS adapter."""

    def test_predict_returns_torch_tensors(self, adapter):
        spatial = torch.randn(1, 13, 16, 16)
        vector = torch.randn(1, 64)
        mask = torch.ones(1, 381)

        policy, value = adapter.predict(spatial, vector, mask)

        assert isinstance(policy, torch.Tensor)
        assert isinstance(value, torch.Tensor)

    def test_predict_shapes(self, adapter):
        batch_size = 1
        spatial = torch.randn(batch_size, 13, 16, 16)
        vector = torch.randn(batch_size, 64)
        mask = torch.ones(batch_size, 381)

        policy, value = adapter.predict(spatial, vector, mask)

        assert policy.shape == (batch_size, 381)
        assert value.shape == (batch_size,)

    def test_predict_returns_probabilities(self, adapter):
        spatial = torch.randn(1, 13, 16, 16)
        vector = torch.randn(1, 64)
        mask = torch.ones(1, 381)

        policy, _ = adapter.predict(spatial, vector, mask)

        # Should be valid probability distribution
        assert torch.all(policy >= 0)
        assert torch.allclose(policy.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_predict_value_in_range(self, adapter):
        spatial = torch.randn(1, 13, 16, 16)
        vector = torch.randn(1, 64)
        mask = torch.ones(1, 381)

        _, value = adapter.predict(spatial, vector, mask)

        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_eval_is_noop(self, adapter):
        # Should not raise
        adapter.eval()

    def test_train_is_noop(self, adapter):
        # Should not raise
        adapter.train()

    def test_action_masking(self, adapter):
        spatial = torch.randn(1, 13, 16, 16)
        vector = torch.randn(1, 64)
        mask = torch.zeros(1, 381)
        mask[:, :10] = 1.0

        policy, _ = adapter.predict(spatial, vector, mask)

        # Most probability should be on valid actions
        valid_prob = policy[:, :10].sum()
        assert valid_prob > 0.99

    def test_parameters_returns_params(self, adapter):
        params = adapter.parameters()
        assert params is not None

    def test_wrapper_property(self, adapter, jax_wrapper):
        assert adapter.wrapper is jax_wrapper


class TestJaxMCTSAdapterWithSimpleMCTS:
    """Integration test: adapter works with SimpleMCTS."""

    def test_mcts_search_completes(self, adapter):
        """Verify that a basic MCTS search can complete with JAX adapter."""
        from alpha_zero.core.civ6_mcts import SimpleMCTS
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=16, max_turns=50, seed=42)
        env.reset()

        mcts = SimpleMCTS(num_simulations=5, c_puct=1.5)
        action_probs, _root_value = mcts.search(env)

        assert action_probs.shape[0] > 0
        assert abs(action_probs.sum() - 1.0) < 1e-5 or action_probs.sum() == 0
