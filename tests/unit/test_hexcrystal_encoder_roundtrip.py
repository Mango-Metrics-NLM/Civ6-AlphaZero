"""Standalone weight round-trip and device behaviour for the hex encoder.

Existing suites exercise the encoder only through ``Civ6TransformerNetwork``.
For the encoder to be publishable on its own, it has to be saveable and
loadable on its own too ΓÇö which means a strict ``load_state_dict`` between two
independently-constructed instances must reproduce identical outputs.

The adjacency tests cover a subtler property: ``HexAdjacency`` indices are
registered buffers holding ``-1`` sentinels for off-board neighbours. Buffers
travel with ``.to(device)`` and appear in the ``state_dict``, and the gather
path must keep tolerating the sentinel after a device move.
"""

from __future__ import annotations

import pytest
import torch

from alpha_zero.hexcrystal import (
    HexAdjacency,
    HexConvolution,
    HexEncoderConfig,
    HexSpatialEncoder,
)

_CONFIG = HexEncoderConfig(in_channels=13, spatial_size=8, embed_dim=64, num_output_tokens=4)


def _sample(batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    return torch.randn(
        batch, _CONFIG.in_channels, _CONFIG.spatial_size, _CONFIG.spatial_size, generator=generator
    )


@pytest.mark.parametrize("use_crystalline", [True, False])
def test_state_dict_round_trip_reproduces_outputs(use_crystalline: bool) -> None:
    """A strict state_dict transfer makes two encoders numerically identical."""
    config = HexEncoderConfig(**{**_CONFIG.to_dict(), "use_crystalline": use_crystalline})

    source = HexSpatialEncoder(config).eval()
    target = HexSpatialEncoder(config).eval()

    sample = _sample()
    with torch.no_grad():
        before = target(sample)

    target.load_state_dict(source.state_dict(), strict=True)

    with torch.no_grad():
        expected = source(sample)
        after = target(sample)

    assert torch.allclose(after, expected, atol=1e-6), "weights did not transfer"
    assert not torch.allclose(before, after, atol=1e-6), (
        "the two encoders were already identical; the test proves nothing"
    )


def test_round_trip_through_a_saved_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The encoder survives a real save/load cycle on disk."""
    source = HexSpatialEncoder(_CONFIG).eval()
    path = tmp_path / "encoder.pt"
    torch.save({"config": _CONFIG.to_dict(), "state_dict": source.state_dict()}, path)

    payload = torch.load(path, weights_only=True)
    restored = HexSpatialEncoder(HexEncoderConfig.from_dict(payload["config"])).eval()
    restored.load_state_dict(payload["state_dict"], strict=True)

    sample = _sample()
    with torch.no_grad():
        assert torch.allclose(source(sample), restored(sample), atol=1e-6)


def test_output_shape_matches_configured_token_count() -> None:
    """The encoder emits exactly the configured number of tokens."""
    encoder = HexSpatialEncoder(_CONFIG)

    out = encoder(_sample(batch=3))

    assert out.shape == (3, _CONFIG.num_output_tokens, _CONFIG.embed_dim)


def test_eval_mode_is_deterministic() -> None:
    """Two forward passes in eval mode agree (dropout is disabled)."""
    encoder = HexSpatialEncoder(_CONFIG).eval()
    sample = _sample()

    with torch.no_grad():
        assert torch.allclose(encoder(sample), encoder(sample), atol=1e-7)


def test_adjacency_indices_are_registered_buffers() -> None:
    """Neighbour indices ship in the state_dict so they move with the module."""
    conv = HexConvolution(in_channels=4, out_channels=8, board_size=8)

    buffer_names = {name for name, _ in conv.named_buffers()}

    assert "neighbor_indices" in buffer_names
    assert "neighbor_indices" in conv.state_dict()


def test_adjacency_marks_off_board_neighbours() -> None:
    """Corner tiles have fewer than six neighbours, flagged with -1."""
    adjacency = HexAdjacency(board_size=8)
    indices = adjacency.get_neighbor_indices()

    assert indices.shape == (64, 6)
    assert (indices == -1).any(), "expected off-board sentinels on a bounded board"
    assert indices.max().item() < 64


def test_adjacency_rejects_unsupported_offset_system() -> None:
    """Only even-q is implemented; anything else fails loudly at construction."""
    with pytest.raises(ValueError, match="even-q"):
        HexAdjacency(board_size=8, offset_type="odd-q")


def test_module_moves_to_cpu_without_disturbing_buffers() -> None:
    """An explicit ``.to('cpu')`` is a no-op that preserves the gather path."""
    encoder = HexSpatialEncoder(_CONFIG).eval()
    sample = _sample()

    with torch.no_grad():
        before = encoder(sample)
    encoder.to(torch.device("cpu"))
    with torch.no_grad():
        after = encoder(sample)

    assert torch.allclose(before, after, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_module_moves_to_cuda() -> None:
    """Buffers follow the module to GPU and the gather still works."""
    encoder = HexSpatialEncoder(_CONFIG).eval().to("cuda")

    with torch.no_grad():
        out = encoder(_sample().to("cuda"))

    assert out.is_cuda
    assert torch.isfinite(out).all()
