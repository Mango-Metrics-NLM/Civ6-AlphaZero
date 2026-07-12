"""Regression test for the bridge encoder/network channel mismatch.

Root cause (see conversation history / commit message for full diagnosis):

1. ``BridgeAgentLoop._build_observation`` used to slice ``spatial_t[0]`` /
   ``vector_t[0]`` believing ``StateConverter.convert_and_encode`` returns a
   batched tensor. It does not — ``Civ6StateEncoder.encode`` returns
   *unbatched* ``(84, 16, 16)`` spatial and ``(512,)`` vector tensors, so the
   ``[0]`` indexing silently corrupted the observation (collapsed the
   spatial tensor to a bare ``(16, 16)`` 2-D array by slicing off channel 0
   instead of stripping a batch dim).
2. ``_build_network_for_bridge`` built ``MicroCivNetwork`` without passing
   ``spatial_channels``/``vector_dim``, so it silently defaulted to the
   13-channel/64-dim MicroCiv shape while ``StateConverter`` encodes with
   the full 84-channel/512-dim ``Civ6StateEncoder``.

This test builds the network exactly as the (fixed) bridge network builder
does, encodes a realistic state via ``StateConverter`` with the bridge's
demo ``ActionSpaceConfig``, and runs ``network.forward`` end to end —
reproducing the exact failure path described above.
"""

from __future__ import annotations

import torch

from alpha_zero.bridge.action_converter import ActionConverter
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.civ6_local import Civ6LocalConfig, _build_network_for_bridge
from alpha_zero.core.civ6_actions import ActionSpaceConfig
from tests.fixtures.bridge_snapshots import minimal_state


def _build_bridge_pieces(board_size: int = 16):
    """Mirror the composition in ``civ6_local.run_civ6_bridge``."""
    config = Civ6LocalConfig(board_size=board_size, num_players=2, device="cpu")

    state_converter = StateConverter(board_size=board_size)

    demo_action_space = ActionSpaceConfig(
        board_size=board_size,
        max_units=20,
        max_cities=5,
        num_production_items=20,
    )
    action_converter = ActionConverter(
        board_size=board_size,
        action_space_config=demo_action_space,
    )

    network = _build_network_for_bridge(
        config, action_space_size=action_converter.action_space_size
    )
    return state_converter, action_converter, network


class TestBridgeEncoderNetworkChannelMatch:
    """StateConverter.convert_and_encode -> network.forward must not raise."""

    def test_network_forward_accepts_state_converter_encoding(self) -> None:
        """Encoding a realistic state and feeding it to the bridge network
        must succeed and produce correctly-shaped policy/value tensors.

        Before the fix this raised:
            RuntimeError: Given groups=1, weight of size [64, 13, 3, 3],
            expected input[1, 1, 16, 16] to have 13 channels, but got 1
            channels instead
        """
        state_converter, action_converter, network = _build_bridge_pieces()

        state = minimal_state(board_size=16)
        spatial_t, vector_t = state_converter.convert_and_encode(state)

        # Unbatched tensors from the encoder - the network builder/caller is
        # responsible for adding the batch dim (mirrors GreedyNetworkPolicy).
        spatial_batched = spatial_t.unsqueeze(0)
        vector_batched = vector_t.unsqueeze(0)
        mask = torch.ones(1, action_converter.action_space_size)

        network.eval()
        with torch.no_grad():
            policy_logits, value = network(spatial_batched, vector_batched, mask)

        assert policy_logits.shape == (1, action_converter.action_space_size)
        assert value.numel() == 1

    def test_build_network_for_bridge_matches_encoder_channels(self) -> None:
        """The constructed network's input dims must equal the encoder's
        actual output dims (84 spatial channels, 512-dim vector) rather
        than silently defaulting to the 13/64 MicroCiv shape."""
        state_converter, _action_converter, network = _build_bridge_pieces()

        assert state_converter.encoder is not None
        assert network.config.spatial_channels == state_converter.encoder.config.spatial_channels
        assert network.config.vector_dim == state_converter.encoder.config.vector_dim
