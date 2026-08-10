"""Direct tests for the encoder's less-travelled branches.

The existing suites reach these layers through the full encoder, which leaves
the alternative symmetry groups, the attention combination mode, the error
paths, and ``PatchEmbed.forward`` covered only incidentally or not at all. Since
the subpackage is published for reuse, a caller may pick any of these
combinations on day one — so each is exercised directly here.
"""

from __future__ import annotations

import pytest
import torch

from alpha_zero.hexcrystal import (
    CrystallineLatticeLayer,
    HexConvBlock,
    HexConvolution,
    PatchEmbed,
    SymmetryType,
)

#: Transform counts implied by each symmetry group.
_EXPECTED_TRANSFORMS = {
    SymmetryType.HEXAGONAL: 6,
    SymmetryType.CUBIC: 8,
    SymmetryType.TETRAHEDRAL: 12,
}


class TestPatchEmbed:
    """The default encoder, tested on its own rather than through the network."""

    def test_forward_produces_one_token_per_patch(self) -> None:
        embed = PatchEmbed(in_channels=13, embed_dim=64, patch_size=4, spatial_size=8)

        out = embed(torch.randn(2, 13, 8, 8))

        assert out.shape == (2, embed.num_patches, 64)
        assert embed.num_patches == (8 // 4) ** 2

    def test_patch_count_scales_with_board_and_patch_size(self) -> None:
        assert PatchEmbed(3, 16, patch_size=2, spatial_size=8).num_patches == 16
        assert PatchEmbed(3, 16, patch_size=8, spatial_size=8).num_patches == 1

    def test_gradients_flow(self) -> None:
        embed = PatchEmbed(in_channels=4, embed_dim=8, patch_size=2, spatial_size=4)

        embed(torch.randn(1, 4, 4, 4)).sum().backward()

        assert embed.proj.weight.grad is not None


class TestSymmetryGroups:
    """Every advertised lattice type must actually build and run."""

    @pytest.mark.parametrize("symmetry", list(SymmetryType))
    def test_each_group_builds_its_transform_set(self, symmetry: SymmetryType) -> None:
        layer = CrystallineLatticeLayer(dim=16, lattice_type=symmetry)

        assert layer.lattice_transform.shape[0] == _EXPECTED_TRANSFORMS[symmetry]

    @pytest.mark.parametrize("symmetry", list(SymmetryType))
    def test_each_group_runs_a_forward_pass(self, symmetry: SymmetryType) -> None:
        layer = CrystallineLatticeLayer(dim=16, lattice_type=symmetry).eval()

        out = layer(torch.randn(2, 4, 16))

        assert out.shape == (2, 4, 16)
        assert torch.isfinite(out).all()

    def test_string_and_enum_select_the_same_group(self) -> None:
        """Config round-trips carry the string form, so both must agree."""
        from_enum = CrystallineLatticeLayer(dim=8, lattice_type=SymmetryType.CUBIC)
        from_string = CrystallineLatticeLayer(dim=8, lattice_type="cubic")

        assert from_enum.lattice_type == from_string.lattice_type
        assert from_enum.lattice_transform.shape == from_string.lattice_transform.shape

    def test_unknown_lattice_type_names_the_valid_options(self) -> None:
        with pytest.raises(ValueError, match="hexagonal"):
            CrystallineLatticeLayer(dim=8, lattice_type="dodecahedral")


class TestCombinationModes:
    """Both ways of combining the transforms are reachable and differ."""

    @pytest.mark.parametrize("mode", ["weighted_sum", "attention"])
    def test_mode_runs(self, mode: str) -> None:
        layer = CrystallineLatticeLayer(dim=16, combination_mode=mode).eval()  # type: ignore[arg-type]

        out = layer(torch.randn(2, 3, 16))

        assert out.shape == (2, 3, 16)
        assert torch.isfinite(out).all()

    def test_weighted_sum_uses_a_learned_parameter(self) -> None:
        """The combination weights are learned, not re-randomised per pass."""
        layer = CrystallineLatticeLayer(dim=8, combination_mode="weighted_sum")

        assert layer.combination_weights is not None
        assert layer.combination_weights.requires_grad
        assert layer.attention_proj is None

    def test_attention_mode_uses_a_projection(self) -> None:
        layer = CrystallineLatticeLayer(dim=8, combination_mode="attention")

        assert layer.attention_proj is not None
        assert layer.combination_weights is None

    def test_unknown_combination_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="combination_mode"):
            CrystallineLatticeLayer(dim=8, combination_mode="telepathy")  # type: ignore[arg-type]

    @pytest.mark.parametrize("mode", ["weighted_sum", "attention"])
    def test_gradients_reach_the_transforms(self, mode: str) -> None:
        layer = CrystallineLatticeLayer(dim=8, combination_mode=mode)  # type: ignore[arg-type]

        layer(torch.randn(2, 3, 8)).sum().backward()

        assert layer.lattice_transform.grad is not None

    def test_eval_mode_is_deterministic(self) -> None:
        """A learned combination must not vary between identical passes."""
        layer = CrystallineLatticeLayer(dim=8).eval()
        sample = torch.randn(1, 2, 8)

        with torch.no_grad():
            assert torch.allclose(layer(sample), layer(sample), atol=1e-7)


class TestHexConvolution:
    """Convolution shape and activation handling."""

    @pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
    def test_activations_are_supported(self, activation: str) -> None:
        block = HexConvBlock(
            in_channels=4, out_channels=8, board_size=8, activation=activation
        ).eval()

        out = block(torch.randn(2, 4, 8, 8))

        assert out.shape == (2, 8, 8, 8)
        assert torch.isfinite(out).all()

    def test_batch_norm_can_be_disabled(self) -> None:
        block = HexConvBlock(
            in_channels=4, out_channels=8, board_size=8, use_batch_norm=False
        ).eval()

        assert block(torch.randn(1, 4, 8, 8)).shape == (1, 8, 8, 8)

    def test_convolution_preserves_spatial_dimensions(self) -> None:
        """The 7-tap gather is same-resolution; nothing downsamples."""
        conv = HexConvolution(in_channels=3, out_channels=5, board_size=6)

        assert conv(torch.randn(2, 3, 6, 6)).shape == (2, 5, 6, 6)

    @pytest.mark.parametrize("board_size", [4, 8, 16])
    def test_various_board_sizes(self, board_size: int) -> None:
        conv = HexConvolution(in_channels=2, out_channels=2, board_size=board_size)

        out = conv(torch.randn(1, 2, board_size, board_size))

        assert out.shape == (1, 2, board_size, board_size)
