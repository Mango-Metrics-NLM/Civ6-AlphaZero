"""Unit tests for the Atari neural network."""

from __future__ import annotations

import pytest
import torch

from demos.atari.atari_network import AtariNetwork
from demos.atari.config import AtariNetworkConfig as NetConfig


@pytest.fixture
def small_network():
    """Create a small network for fast testing."""
    config = NetConfig(num_res_blocks=2, num_filters=16, fc_hidden_dim=32, value_head_hidden=16)
    return AtariNetwork(
        config=config,
        num_actions=6,
        spatial_size=(84, 84),
        spatial_channels=4,
        vector_dim=32,
    )


class TestAtariNetwork:
    def test_forward_shape(self, small_network):
        batch = 2
        spatial = torch.randn(batch, 4, 84, 84)
        vector = torch.randn(batch, 32)
        logits, value = small_network(spatial, vector)
        assert logits.shape == (batch, 6)
        assert value.shape == (batch,)

    def test_value_range(self, small_network):
        spatial = torch.randn(1, 4, 84, 84)
        vector = torch.randn(1, 32)
        _, value = small_network(spatial, vector)
        # Value head uses tanh → should be in [-1, 1]
        assert -1.0 <= value.item() <= 1.0

    def test_action_mask(self, small_network):
        spatial = torch.randn(1, 4, 84, 84)
        vector = torch.randn(1, 32)
        mask = torch.tensor([[1, 0, 1, 0, 1, 0]], dtype=torch.float32)
        logits, _ = small_network(spatial, vector, action_mask=mask)
        # Masked actions should have very low logits
        assert logits[0, 1].item() < -1e6
        assert logits[0, 3].item() < -1e6

    def test_predict_returns_probs(self, small_network):
        spatial = torch.randn(1, 4, 84, 84)
        vector = torch.randn(1, 32)
        probs, _value = small_network.predict(spatial, vector)
        assert probs.shape == (1, 6)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert (probs >= 0).all()

    def test_predict_no_mask(self, small_network):
        spatial = torch.randn(2, 4, 84, 84)
        vector = torch.randn(2, 32)
        probs, _value = small_network.predict(spatial, vector, action_mask=None)
        assert probs.shape == (2, 6)

    def test_different_spatial_sizes(self):
        config = NetConfig(num_res_blocks=1, num_filters=8, fc_hidden_dim=16, value_head_hidden=8)
        for size in [(42, 42), (84, 84), (96, 96)]:
            net = AtariNetwork(
                config=config,
                num_actions=4,
                spatial_size=size,
                spatial_channels=4,
                vector_dim=16,
            )
            spatial = torch.randn(1, 4, size[0], size[1])
            vector = torch.randn(1, 16)
            logits, _value = net(spatial, vector)
            assert logits.shape == (1, 4)

    def test_different_action_spaces(self):
        config = NetConfig(num_res_blocks=1, num_filters=8, fc_hidden_dim=16, value_head_hidden=8)
        for n_actions in [4, 6, 9, 18]:
            net = AtariNetwork(
                config=config,
                num_actions=n_actions,
                spatial_size=(84, 84),
                spatial_channels=4,
                vector_dim=16,
            )
            spatial = torch.randn(1, 4, 84, 84)
            vector = torch.randn(1, 16)
            logits, _ = net(spatial, vector)
            assert logits.shape == (1, n_actions)

    def test_gradient_flow(self, small_network):
        spatial = torch.randn(2, 4, 84, 84)
        vector = torch.randn(2, 32)
        logits, value = small_network(spatial, vector)
        loss = logits.mean() + value.mean()
        loss.backward()
        # Check at least one param has grad
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in small_network.parameters())
        assert has_grad

    def test_save_and_load(self, small_network, tmp_path):
        path = str(tmp_path / "test_model.pt")
        small_network.save(path)

        loaded = AtariNetwork.load(path, spatial_channels=4, vector_dim=32)
        assert loaded.num_actions == small_network.num_actions

        # Verify same outputs
        spatial = torch.randn(1, 4, 84, 84)
        vector = torch.randn(1, 32)
        small_network.eval()
        loaded.eval()
        with torch.no_grad():
            orig_logits, orig_val = small_network(spatial, vector)
            load_logits, load_val = loaded(spatial, vector)
        torch.testing.assert_close(orig_logits, load_logits)
        torch.testing.assert_close(orig_val, load_val)

    def test_eval_mode(self, small_network):
        small_network.eval()
        spatial = torch.randn(1, 4, 84, 84)
        vector = torch.randn(1, 32)
        with torch.no_grad():
            logits, _value = small_network(spatial, vector)
        assert logits.shape == (1, 6)

    def test_parameter_count(self, small_network):
        count = sum(p.numel() for p in small_network.parameters())
        assert count > 0

    def test_uses_core_residual_block(self):
        """Verify we're reusing the core project's ResidualBlock."""
        from alpha_zero.core.civ6_network import ResidualBlock

        config = NetConfig(num_res_blocks=2, num_filters=8, fc_hidden_dim=16, value_head_hidden=8)
        net = AtariNetwork(config=config, num_actions=4, spatial_size=(84, 84), spatial_channels=4, vector_dim=16)
        # The res_tower should contain ResidualBlock instances
        for module in net.res_tower:
            assert isinstance(module, ResidualBlock)
