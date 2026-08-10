"""Checkpoint key-namespace regression guard for the hexcrystal extraction.

Every checkpoint ever written by this repository stores encoder weights under
``encoder.patch_embed.*`` ΓÇö the attribute name is deliberately preserved for both
encoder types (see the comment in ``TransformerEncoder.__init__``). Moving the
encoder modules must not perturb that namespace, or every existing checkpoint
becomes unloadable.

The expected key sets below were captured by running the *pre-move* code and are
committed as literal data. They are intentionally not regenerated from the
current implementation: a snapshot written after a change cannot detect that
change.
"""

from __future__ import annotations

import pytest

from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

# Captured from the pre-move tree on 2026-08-09 with the config below.
_EXPECTED: dict[str, dict[str, int]] = {
    "patch_embed": {"n_keys": 41, "n_params": 231_225},
    "hex_crystalline": {"n_keys": 65, "n_params": 290_751},
}

# Encoder-owned keys, pre-move. Both encoders live under the same prefix.
_EXPECTED_ENCODER_KEYS: dict[str, set[str]] = {
    "patch_embed": {
        "encoder.patch_embed.proj.bias",
        "encoder.patch_embed.proj.weight",
    },
    "hex_crystalline": {
        "encoder.patch_embed.crystalline.combination_weights",
        "encoder.patch_embed.crystalline.lattice_basis",
        "encoder.patch_embed.crystalline.lattice_norm.bias",
        "encoder.patch_embed.crystalline.lattice_norm.weight",
        "encoder.patch_embed.crystalline.lattice_transform",
    },
}

_ENCODER_PREFIX = "encoder.patch_embed."


def _build(encoder_type: str) -> Civ6TransformerNetwork:
    """Build a network with the exact config used to capture the snapshot."""
    config = TransformerConfig(
        encoder_type=encoder_type,
        spatial_size=8,
        patch_size=4,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
    )
    return Civ6TransformerNetwork(config)


@pytest.mark.parametrize("encoder_type", ["patch_embed", "hex_crystalline"])
def test_state_dict_key_count_unchanged(encoder_type: str) -> None:
    """The number of state_dict entries matches the pre-move capture."""
    net = _build(encoder_type)

    assert len(net.state_dict()) == _EXPECTED[encoder_type]["n_keys"]


@pytest.mark.parametrize("encoder_type", ["patch_embed", "hex_crystalline"])
def test_parameter_count_unchanged(encoder_type: str) -> None:
    """Total parameter count matches the pre-move capture."""
    net = _build(encoder_type)
    total = sum(p.numel() for p in net.parameters())

    assert total == _EXPECTED[encoder_type]["n_params"]


@pytest.mark.parametrize("encoder_type", ["patch_embed", "hex_crystalline"])
def test_encoder_weights_keep_patch_embed_prefix(encoder_type: str) -> None:
    """Encoder weights stay under ``encoder.patch_embed.`` for both encoders.

    This is the load-bearing compatibility property: the attribute name, not the
    module path, determines checkpoint keys.
    """
    net = _build(encoder_type)
    encoder_keys = {k for k in net.state_dict() if k.startswith(_ENCODER_PREFIX)}

    assert encoder_keys, f"no keys under {_ENCODER_PREFIX!r} for {encoder_type}"
    missing = _EXPECTED_ENCODER_KEYS[encoder_type] - encoder_keys
    assert not missing, f"expected encoder keys vanished after the move: {sorted(missing)}"


@pytest.mark.parametrize("encoder_type", ["patch_embed", "hex_crystalline"])
def test_state_dict_round_trips_strictly(encoder_type: str) -> None:
    """A freshly built network loads its own state_dict with ``strict=True``."""
    source = _build(encoder_type)
    target = _build(encoder_type)

    target.load_state_dict(source.state_dict(), strict=True)


def test_encoders_have_distinct_key_sets() -> None:
    """The two encoders remain distinguishable by their key sets.

    Guards the assumption behind ``validate_encoder_state``: a hex checkpoint
    must not silently load into a patch_embed skeleton.
    """
    patch_keys = set(_build("patch_embed").state_dict())
    hex_keys = set(_build("hex_crystalline").state_dict())

    assert patch_keys != hex_keys
    assert any("crystalline" in k or "hex_conv" in k for k in hex_keys - patch_keys)
