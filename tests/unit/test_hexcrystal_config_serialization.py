"""Serialization contract for the hexcrystal configuration objects.

``HexEncoderConfig`` is the schema behind the published ``config.json``, so its
round-trip behaviour is a compatibility surface, not an implementation detail.
These tests pin three properties: every field survives a dict round-trip, the
JSON form contains only primitives, and a config written by a future version
(carrying fields this version does not know) still loads.

``SymmetryType`` is covered here too because it is the one enum in the stack, and
the crystalline layer accepts either the enum or its string value ΓÇö a JSON round
trip must not change which layer you get.
"""

from __future__ import annotations

import json
from dataclasses import fields

import pytest
import torch

from alpha_zero.hexcrystal import (
    CrystallineLatticeLayer,
    HexEncoderConfig,
    HexSpatialEncoder,
    SymmetryType,
)

_PRIMITIVES = (str, int, float, bool, type(None))


def test_to_dict_covers_every_field() -> None:
    """No field is silently dropped from the serialized form."""
    config = HexEncoderConfig()
    data = config.to_dict()

    assert set(data) == {f.name for f in fields(HexEncoderConfig)}


def test_to_dict_values_are_json_primitives() -> None:
    """Every value is a JSON primitive, so no custom encoder is required."""
    data = HexEncoderConfig().to_dict()

    for key, value in data.items():
        assert isinstance(value, _PRIMITIVES), f"{key} is {type(value).__name__}, not a primitive"


def test_dict_round_trip_is_lossless() -> None:
    """dict -> config -> dict reproduces the original exactly."""
    original = HexEncoderConfig(
        in_channels=7,
        embed_dim=32,
        spatial_size=16,
        hex_hidden_dim=48,
        hex_conv_layers=3,
        lattice_type="cubic",
        num_output_tokens=16,
        dropout=0.25,
        activation="relu",
        use_batch_norm=False,
        use_crystalline=False,
    )

    restored = HexEncoderConfig.from_dict(original.to_dict())

    assert restored == original


def test_json_round_trip_is_lossless() -> None:
    """config -> JSON text -> config reproduces the original exactly."""
    original = HexEncoderConfig(embed_dim=128, lattice_type="tetrahedral", dropout=0.0)

    restored = HexEncoderConfig.from_json(original.to_json())

    assert restored == original


def test_to_json_emits_a_flat_object() -> None:
    """The JSON form is a flat object ΓÇö required for a hub ``config.json``."""
    payload = json.loads(HexEncoderConfig().to_json())

    assert isinstance(payload, dict)
    assert all(not isinstance(v, (dict, list)) for v in payload.values())


def test_from_dict_ignores_unknown_keys() -> None:
    """Unknown fields are dropped, not fatal.

    A config written by a newer version of the package must still load, so
    forward compatibility is a tolerated-unknowns policy rather than an error.
    """
    data = HexEncoderConfig().to_dict()
    data["a_field_from_the_future"] = 42

    restored = HexEncoderConfig.from_dict(data)

    assert restored == HexEncoderConfig()


def test_from_dict_warns_about_unknown_keys(caplog: pytest.LogCaptureFixture) -> None:
    """Dropped fields are logged, so the loss is visible rather than silent."""
    data = HexEncoderConfig().to_dict()
    data["mystery_field"] = "x"

    with caplog.at_level("WARNING"):
        HexEncoderConfig.from_dict(data)

    assert "mystery_field" in caplog.text


def test_from_dict_accepts_a_partial_mapping() -> None:
    """Omitted fields fall back to their defaults."""
    restored = HexEncoderConfig.from_dict({"embed_dim": 256})

    assert restored.embed_dim == 256
    assert restored.in_channels == HexEncoderConfig().in_channels


@pytest.mark.parametrize("symmetry", list(SymmetryType))
def test_symmetry_type_survives_json_round_trip(symmetry: SymmetryType) -> None:
    """A symmetry value serializes to a string and rebuilds the same layer.

    ``SymmetryType`` is a ``(str, Enum)``, so ``json.dumps`` emits its value
    directly and the layer constructor accepts the string back.
    """
    encoded = json.dumps({"lattice_type": symmetry})
    decoded = json.loads(encoded)["lattice_type"]

    assert decoded == symmetry.value

    from_enum = CrystallineLatticeLayer(dim=16, lattice_type=symmetry)
    from_string = CrystallineLatticeLayer(dim=16, lattice_type=decoded)

    assert from_enum.lattice_type == from_string.lattice_type
    assert from_enum.lattice_transform.shape == from_string.lattice_transform.shape


def test_config_drives_an_equivalent_encoder_after_round_trip() -> None:
    """A round-tripped config builds a structurally identical encoder.

    Equality of the dataclass is necessary but not sufficient ΓÇö this asserts the
    config is actually *sufficient* to rebuild the module.
    """
    original = HexEncoderConfig(in_channels=13, spatial_size=8, embed_dim=64, num_output_tokens=4)
    restored = HexEncoderConfig.from_json(original.to_json())

    torch.manual_seed(0)
    first = HexSpatialEncoder(original)
    torch.manual_seed(0)
    second = HexSpatialEncoder(restored)

    assert first.state_dict().keys() == second.state_dict().keys()

    sample = torch.randn(2, 13, 8, 8)
    first.eval()
    second.eval()
    with torch.no_grad():
        assert torch.allclose(first(sample), second(sample), atol=1e-6)
