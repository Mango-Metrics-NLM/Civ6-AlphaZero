"""Backwards-compatibility guarantees for the hexcrystal extraction.

The encoder implementation moved from ``alpha_zero.core.*`` to
``alpha_zero.hexcrystal.*``. Shim modules remain at the historical paths. These
tests pin the guarantee that the move is invisible to existing callers: the old
paths must resolve to the *same objects*, not merely to equivalent ones, and the
encoder registry must remain a single shared registry regardless of which import
path registered an entry.

Companion suites:
  - ``test_hexcrystal_state_dict_keys.py`` pins the checkpoint key namespace.
  - ``test_hexcrystal_isolation.py`` pins the torch-only dependency contract.
"""

from __future__ import annotations

import importlib

import pytest
import torch.nn as nn

# (old module path, new module path, exported names that must be identical)
_SHIM_MAP: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "alpha_zero.core.hex_conv",
        "alpha_zero.hexcrystal.hex_conv",
        ("HexAdjacency", "HexConvBlock", "HexConvConfig", "HexConvolution"),
    ),
    (
        "alpha_zero.core.crystalline_layers",
        "alpha_zero.hexcrystal.crystalline_layers",
        ("CrystallineLatticeConfig", "CrystallineLatticeLayer", "SymmetryType"),
    ),
    (
        "alpha_zero.core.hex_encoder",
        "alpha_zero.hexcrystal.hex_encoder",
        ("HexEncoderConfig", "HexSpatialEncoder", "SpatialToTokens", "create_hex_encoder"),
    ),
    (
        "alpha_zero.core.encoder_registry",
        "alpha_zero.hexcrystal.encoder_registry",
        (
            "EncoderFactory",
            "EncoderType",
            "create_spatial_encoder",
            "get_registered_encoders",
            "register_encoder",
        ),
    ),
]


@pytest.mark.parametrize(
    ("old_path", "new_path", "names"),
    _SHIM_MAP,
    ids=[old.rsplit(".", 1)[-1] for old, _, _ in _SHIM_MAP],
)
def test_shim_exports_are_identical_objects(
    old_path: str, new_path: str, names: tuple[str, ...]
) -> None:
    """Every shimmed name resolves to the very same object as the new path."""
    old_mod = importlib.import_module(old_path)
    new_mod = importlib.import_module(new_path)

    for name in names:
        old_obj = getattr(old_mod, name)
        new_obj = getattr(new_mod, name)
        assert old_obj is new_obj, (
            f"{old_path}.{name} is not the same object as {new_path}.{name}; "
            "the shim is re-implementing rather than re-exporting"
        )


@pytest.mark.parametrize(
    ("old_path", "names"),
    [(old, names) for old, _, names in _SHIM_MAP],
    ids=[old.rsplit(".", 1)[-1] for old, _, _ in _SHIM_MAP],
)
def test_shim_declares_all(old_path: str, names: tuple[str, ...]) -> None:
    """Shims declare ``__all__``.

    Beyond documenting the surface, ``__all__`` marks the re-exports as used so
    the repository's lint autofix cannot strip them as unused imports.
    """
    old_mod = importlib.import_module(old_path)

    assert hasattr(old_mod, "__all__"), f"{old_path} must declare __all__"
    assert set(old_mod.__all__) == set(names), (
        f"{old_path}.__all__ drifted from its re-exported names"
    )


def test_patch_embed_alias_is_identical() -> None:
    """``transformer_network.PatchEmbed`` still resolves after the move."""
    from alpha_zero.core.transformer_network import PatchEmbed as TransformerPatchEmbed
    from alpha_zero.hexcrystal.patch_embed import PatchEmbed as HexcrystalPatchEmbed

    assert TransformerPatchEmbed is HexcrystalPatchEmbed


def test_registry_is_shared_across_import_paths() -> None:
    """An encoder registered via the old path is visible through the new one."""
    from alpha_zero.core.encoder_registry import register_encoder as register_old
    from alpha_zero.hexcrystal.encoder_registry import (
        get_registered_encoders as get_new,
    )

    sentinel = "_backcompat_probe_encoder"

    @register_old(sentinel)
    def _factory(
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        spatial_size: int,
        **_kwargs: object,
    ) -> nn.Module:
        return nn.Identity()

    try:
        assert sentinel in get_new(), (
            "registry is not shared between the shim and the implementation module"
        )
    finally:
        # Keep global state clean for other tests.
        from alpha_zero.hexcrystal import encoder_registry as impl

        impl._ENCODER_REGISTRY.pop(sentinel, None)


def test_both_builtin_encoders_registered() -> None:
    """The move must not drop either built-in encoder registration."""
    from alpha_zero.hexcrystal.encoder_registry import get_registered_encoders

    registered = get_registered_encoders()
    assert "patch_embed" in registered
    assert "hex_crystalline" in registered


def test_hexcrystal_package_reexports_public_api() -> None:
    """The subpackage ``__init__`` exposes the full public surface."""
    import alpha_zero.hexcrystal as pkg

    for name in pkg.__all__:
        assert getattr(pkg, name, None) is not None, f"hexcrystal.{name} is missing"
