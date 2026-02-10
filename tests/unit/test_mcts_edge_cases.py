"""Edge case tests for MCTS variants."""
import numpy as np
import pytest
import torch

from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.core.civ6_mcts import GumbelMCTS, GumbelMCTSConfig
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.utils.constants import ENCODER


class TestMCTSEdgeCases:
    """Edge cases for MCTS search."""

    def test_single_valid_action_deterministic(self):
        """When only one action is valid, network output should be deterministic after masking."""
        action_space_size = 376  # MicroCiv default
        network = MicroCivNetwork(board_size=8, num_res_blocks=1, num_filters=8, action_space_size=action_space_size)
        network.eval()

        # Create input
        spatial = torch.randn(1, ENCODER.MICROCIV_CHANNELS, 8, 8)
        vector = torch.randn(1, ENCODER.MICROCIV_VECTOR_DIM)

        # Create mask with only one valid action
        action_mask = torch.zeros(1, action_space_size)
        action_mask[0, 0] = 1.0  # Only action 0 is valid

        with torch.no_grad():
            policy_logits, value = network(spatial, vector, action_mask)
            # Apply softmax to get probabilities
            policy_probs = torch.softmax(policy_logits, dim=1)

        # The single valid action should have probability ~1.0
        assert policy_probs[0, 0].item() > 0.99, f"Single valid action should dominate: {policy_probs[0, 0].item()}"
        # All other actions should have near-zero probability
        assert policy_probs[0, 1:].sum().item() < 0.01

    def test_extreme_value_network_outputs(self):
        """Network should handle normally even with extreme inputs."""
        action_space_size = 376  # MicroCiv default
        network = MicroCivNetwork(board_size=8, num_res_blocks=1, num_filters=8, action_space_size=action_space_size)
        network.eval()

        # Large magnitude inputs
        spatial = torch.randn(1, ENCODER.MICROCIV_CHANNELS, 8, 8) * 100.0
        vector = torch.randn(1, ENCODER.MICROCIV_VECTOR_DIM) * 100.0
        action_mask = torch.ones(1, action_space_size)

        with torch.no_grad():
            policy_logits, value = network(spatial, vector, action_mask)

        assert not torch.isnan(policy_logits).any(), "Policy contains NaN"
        assert not torch.isinf(policy_logits).any(), "Policy contains Inf"
        assert not torch.isnan(value).any(), "Value contains NaN"

    def test_zero_input_does_not_crash(self):
        """Zero spatial/vector input should not crash the network."""
        action_space_size = 376  # MicroCiv default
        network = MicroCivNetwork(board_size=8, num_res_blocks=1, num_filters=8, action_space_size=action_space_size)
        network.eval()

        spatial = torch.zeros(1, ENCODER.MICROCIV_CHANNELS, 8, 8)
        vector = torch.zeros(1, ENCODER.MICROCIV_VECTOR_DIM)
        action_mask = torch.ones(1, action_space_size)

        with torch.no_grad():
            policy_logits, value = network(spatial, vector, action_mask)

        assert policy_logits.shape == (1, action_space_size)
        assert value.shape == (1, 1) or value.shape == (1,)

    def test_batch_size_one_forward(self):
        """Single-sample batch should work correctly."""
        action_space_size = 376  # MicroCiv default
        network = MicroCivNetwork(board_size=8, num_res_blocks=1, num_filters=8, action_space_size=action_space_size)
        network.eval()

        spatial = torch.randn(1, ENCODER.MICROCIV_CHANNELS, 8, 8)
        vector = torch.randn(1, ENCODER.MICROCIV_VECTOR_DIM)
        action_mask = torch.ones(1, action_space_size)

        with torch.no_grad():
            policy_logits, value = network(spatial, vector, action_mask)

        assert policy_logits.shape[0] == 1
        assert value.shape[0] == 1
