"""Unit tests for alpha_zero.core.hex_conv module.

Tests hexagonal topology-aware convolution for Civ6 AlphaZero, including:
- HexAdjacency: pre-computed hex neighbor indices for even-q offset coordinates
- HexConvolution: 7-tap kernel (center + 6 neighbors) with proper hex topology
- HexConvBlock: convolution block with batch norm, activation, and residual
"""

import pytest
import torch

from alpha_zero.core.hex_conv import (
    HexAdjacency,
    HexConvBlock,
    HexConvConfig,
    HexConvolution,
)


class TestHexAdjacency:
    """Tests for HexAdjacency pre-computation of hex neighbor indices."""

    def test_neighbor_count_shape(self):
        """Test that neighbor indices have correct shape (board_size*board_size, 6)."""
        board_size = 8
        hex_adj = HexAdjacency(board_size=board_size)

        neighbors = hex_adj.get_neighbor_indices()

        assert neighbors.shape == (board_size * board_size, 6)
        assert neighbors.dtype == torch.long

    def test_center_cell_has_six_valid_neighbors(self):
        """Test that a center cell (even col) has 6 valid neighbors (no -1)."""
        board_size = 8
        hex_adj = HexAdjacency(board_size=board_size)
        neighbors = hex_adj.get_neighbor_indices()

        # Center cell at (col=2, row=3) - even column, well inside board
        col, row = 2, 3
        flat_idx = row * board_size + col

        cell_neighbors = neighbors[flat_idx]

        # All 6 neighbors should be valid (>= 0)
        assert (cell_neighbors >= 0).all()
        assert cell_neighbors.numel() == 6

    def test_corner_cell_has_fewer_valid_neighbors(self):
        """Test that corner cell (0,0) has fewer than 6 valid neighbors (some -1)."""
        board_size = 8
        hex_adj = HexAdjacency(board_size=board_size)
        neighbors = hex_adj.get_neighbor_indices()

        # Top-left corner at (col=0, row=0)
        col, row = 0, 0
        flat_idx = row * board_size + col

        cell_neighbors = neighbors[flat_idx]

        # Should have some invalid neighbors (-1)
        num_valid = (cell_neighbors >= 0).sum().item()
        assert num_valid < 6
        assert num_valid >= 2  # Corner should have at least 2 neighbors

    def test_edge_cell_has_valid_neighbor_count(self):
        """Test that edge cell has between 3-5 valid neighbors."""
        board_size = 8
        hex_adj = HexAdjacency(board_size=board_size)
        neighbors = hex_adj.get_neighbor_indices()

        # Edge cell at (col=0, row=4) - left edge, middle row
        col, row = 0, 4
        flat_idx = row * board_size + col

        cell_neighbors = neighbors[flat_idx]

        num_valid = (cell_neighbors >= 0).sum().item()
        assert 3 <= num_valid <= 5  # Edge cells have 3-5 neighbors

    def test_even_vs_odd_column_neighbor_rules(self):
        """Test that even and odd columns have different neighbor patterns."""
        board_size = 8
        hex_adj = HexAdjacency(board_size=board_size)
        neighbors = hex_adj.get_neighbor_indices()

        # Even column cell at (col=2, row=3)
        # Expected neighbors: (c-1,r-1), (c-1,r), (c,r-1), (c,r+1), (c+1,r-1), (c+1,r)
        # = (1,2), (1,3), (2,2), (2,4), (3,2), (3,3)
        even_col, even_row = 2, 3
        even_flat_idx = even_row * board_size + even_col
        even_neighbors = neighbors[even_flat_idx]

        # Convert expected neighbors to flat indices
        expected_even = [
            2 * board_size + 1,  # (1,2)
            3 * board_size + 1,  # (1,3)
            2 * board_size + 2,  # (2,2)
            4 * board_size + 2,  # (2,4)
            2 * board_size + 3,  # (3,2)
            3 * board_size + 3,  # (3,3)
        ]

        assert even_neighbors.tolist() == expected_even

        # Odd column cell at (col=3, row=3)
        # Expected neighbors: (c-1,r), (c-1,r+1), (c,r-1), (c,r+1), (c+1,r), (c+1,r+1)
        # = (2,3), (2,4), (3,2), (3,4), (4,3), (4,4)
        odd_col, odd_row = 3, 3
        odd_flat_idx = odd_row * board_size + odd_col
        odd_neighbors = neighbors[odd_flat_idx]

        expected_odd = [
            3 * board_size + 2,  # (2,3)
            4 * board_size + 2,  # (2,4)
            2 * board_size + 3,  # (3,2)
            4 * board_size + 3,  # (3,4)
            3 * board_size + 4,  # (4,3)
            4 * board_size + 4,  # (4,4)
        ]

        assert odd_neighbors.tolist() == expected_odd

    def test_invalid_offset_type_raises_error(self):
        """Test that invalid offset_type raises ValueError."""
        with pytest.raises(ValueError, match="Only 'even-q' offset type is currently supported"):
            HexAdjacency(board_size=8, offset_type="odd-q")


class TestHexConvolution:
    """Tests for HexConvolution layer with 7-tap kernel."""

    def test_output_shape(self):
        """Test that output has correct shape (B, C_out, H, W)."""
        torch.manual_seed(42)

        batch_size = 2
        in_channels = 13
        out_channels = 64
        board_size = 8

        layer = HexConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            board_size=board_size,
        )

        x = torch.randn(batch_size, in_channels, board_size, board_size)
        output = layer(x)

        assert output.shape == (batch_size, out_channels, board_size, board_size)

    def test_various_channel_and_board_sizes(self):
        """Test various input/output channel combinations and board sizes."""
        torch.manual_seed(42)

        test_cases = [
            (1, 3, 16, 4),  # (batch, in_ch, out_ch, board_size)
            (2, 13, 64, 8),
            (4, 64, 128, 16),
            (1, 32, 32, 12),
        ]

        for batch_size, in_ch, out_ch, board_size in test_cases:
            layer = HexConvolution(
                in_channels=in_ch,
                out_channels=out_ch,
                board_size=board_size,
            )

            x = torch.randn(batch_size, in_ch, board_size, board_size)
            output = layer(x)

            assert output.shape == (batch_size, out_ch, board_size, board_size)

    def test_with_bias_and_without_bias(self):
        """Test convolution with and without bias."""
        torch.manual_seed(42)

        in_channels = 13
        out_channels = 32
        board_size = 8

        # With bias
        layer_with_bias = HexConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            board_size=board_size,
            bias=True,
        )
        assert layer_with_bias.bias is not None
        assert layer_with_bias.bias.shape == (out_channels,)

        # Without bias
        layer_no_bias = HexConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            board_size=board_size,
            bias=False,
        )
        assert layer_no_bias.bias is None

        # Both should produce valid output
        x = torch.randn(2, in_channels, board_size, board_size)

        out_with_bias = layer_with_bias(x)
        out_no_bias = layer_no_bias(x)

        assert out_with_bias.shape == (2, out_channels, board_size, board_size)
        assert out_no_bias.shape == (2, out_channels, board_size, board_size)

    def test_zero_input_with_no_bias(self):
        """Test that zero input produces small output when no bias is used."""
        torch.manual_seed(42)

        in_channels = 13
        out_channels = 32
        board_size = 8

        layer = HexConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            board_size=board_size,
            bias=False,
        )

        # Zero input
        x = torch.zeros(1, in_channels, board_size, board_size)
        output = layer(x)

        # Output should be exactly zero (no bias, zero input)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow through input and weight."""
        torch.manual_seed(42)

        in_channels = 8
        out_channels = 16
        board_size = 8

        layer = HexConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            board_size=board_size,
        )

        x = torch.randn(2, in_channels, board_size, board_size, requires_grad=True)
        output = layer(x)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert layer.weight.grad is not None

        # Check gradients are non-zero
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        assert not torch.allclose(layer.weight.grad, torch.zeros_like(layer.weight.grad))


class TestHexConvBlock:
    """Tests for HexConvBlock with batch norm, activation, and residual."""

    def test_output_shape_with_residual_same_channels(self):
        """Test output shape when residual is enabled and channels match."""
        torch.manual_seed(42)

        batch_size = 2
        channels = 64
        board_size = 8

        block = HexConvBlock(
            in_channels=channels,
            out_channels=channels,
            board_size=board_size,
            residual=True,
        )

        x = torch.randn(batch_size, channels, board_size, board_size)
        output = block(x)

        assert output.shape == (batch_size, channels, board_size, board_size)

    def test_output_shape_without_residual(self):
        """Test output shape when residual is disabled."""
        torch.manual_seed(42)

        batch_size = 2
        in_channels = 32
        out_channels = 64
        board_size = 8

        block = HexConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            board_size=board_size,
            residual=False,
        )

        x = torch.randn(batch_size, in_channels, board_size, board_size)
        output = block(x)

        assert output.shape == (batch_size, out_channels, board_size, board_size)

    def test_different_in_out_channels_with_residual(self):
        """Test that skip projection is used when in_channels != out_channels."""
        torch.manual_seed(42)

        batch_size = 2
        in_channels = 32
        out_channels = 64
        board_size = 8

        block = HexConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            board_size=board_size,
            residual=True,
        )

        # Should have skip_conv for channel projection
        assert block.skip_conv is not None

        x = torch.randn(batch_size, in_channels, board_size, board_size)
        output = block(x)

        assert output.shape == (batch_size, out_channels, board_size, board_size)

    def test_activation_gelu(self):
        """Test block with GELU activation."""
        torch.manual_seed(42)

        block = HexConvBlock(
            in_channels=32,
            out_channels=64,
            board_size=8,
            activation="gelu",
        )

        assert isinstance(block.act, torch.nn.GELU)

        x = torch.randn(1, 32, 8, 8)
        output = block(x)

        assert output.shape == (1, 64, 8, 8)

    def test_activation_relu(self):
        """Test block with ReLU activation."""
        torch.manual_seed(42)

        block = HexConvBlock(
            in_channels=32,
            out_channels=64,
            board_size=8,
            activation="relu",
        )

        assert isinstance(block.act, torch.nn.ReLU)

        x = torch.randn(1, 32, 8, 8)
        output = block(x)

        assert output.shape == (1, 64, 8, 8)

    def test_activation_silu(self):
        """Test block with SiLU activation."""
        torch.manual_seed(42)

        block = HexConvBlock(
            in_channels=32,
            out_channels=64,
            board_size=8,
            activation="silu",
        )

        assert isinstance(block.act, torch.nn.SiLU)

        x = torch.randn(1, 32, 8, 8)
        output = block(x)

        assert output.shape == (1, 64, 8, 8)

    def test_with_batch_norm(self):
        """Test block with batch normalization enabled."""
        torch.manual_seed(42)

        block = HexConvBlock(
            in_channels=32,
            out_channels=64,
            board_size=8,
            use_batch_norm=True,
        )

        assert block.batch_norm is not None
        assert isinstance(block.batch_norm, torch.nn.BatchNorm2d)

        # Convolution should not have bias when batch norm is used
        assert block.hex_conv.bias is None

        x = torch.randn(2, 32, 8, 8)
        output = block(x)

        assert output.shape == (2, 64, 8, 8)

    def test_without_batch_norm(self):
        """Test block without batch normalization."""
        torch.manual_seed(42)

        block = HexConvBlock(
            in_channels=32,
            out_channels=64,
            board_size=8,
            use_batch_norm=False,
        )

        assert block.batch_norm is None

        # Convolution should have bias when batch norm is not used
        assert block.hex_conv.bias is not None

        x = torch.randn(2, 32, 8, 8)
        output = block(x)

        assert output.shape == (2, 64, 8, 8)


class TestHexConvConfig:
    """Tests for HexConvConfig dataclass."""

    def test_default_config_values(self):
        """Test that default config values match expected values."""
        config = HexConvConfig()

        assert config.in_channels == 13
        assert config.out_channels == 64
        assert config.board_size == 8
        assert config.offset_type == "even-q"
        assert config.num_layers == 2
        assert config.activation == "gelu"
        assert config.use_bias is True
        assert config.use_batch_norm is True

    def test_custom_config_values(self):
        """Test creating config with custom values."""
        config = HexConvConfig(
            in_channels=32,
            out_channels=128,
            board_size=16,
            num_layers=4,
            activation="relu",
            use_bias=False,
            use_batch_norm=False,
        )

        assert config.in_channels == 32
        assert config.out_channels == 128
        assert config.board_size == 16
        assert config.num_layers == 4
        assert config.activation == "relu"
        assert config.use_bias is False
        assert config.use_batch_norm is False
