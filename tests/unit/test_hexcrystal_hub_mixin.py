"""Hugging Face Hub round-trip for the hex-crystalline encoder.

Everything here runs offline against a temporary directory. A published model is
only as good as its reload path, so these tests assert the three properties a
consumer actually depends on: the artifact set is exactly what the Hub expects,
the configuration survives as readable JSON, and the reloaded weights reproduce
the original outputs numerically.

The final test covers the negative path — importing the module without the extra
installed must fail with a message naming the extra, not with an obscure
``NameError`` further down.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub", reason="requires the 'hub' extra")
pytest.importorskip("safetensors", reason="requires the 'hub' extra")

import torch

from alpha_zero.hexcrystal import HexEncoderConfig, HexSpatialEncoder
from alpha_zero.hexcrystal.hub_mixin import HexSpatialEncoderForHub

_CONFIG = HexEncoderConfig(in_channels=13, spatial_size=8, embed_dim=64, num_output_tokens=4)


@pytest.fixture
def sample() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(2, 13, 8, 8, generator=generator)


@pytest.fixture
def saved_encoder(tmp_path: Path) -> tuple[HexSpatialEncoderForHub, Path]:
    """A hub encoder saved to a temporary directory."""
    encoder = HexSpatialEncoderForHub.from_encoder_config(_CONFIG).eval()
    destination = tmp_path / "encoder"
    encoder.save_pretrained(destination)
    return encoder, destination


def test_save_pretrained_writes_expected_artifacts(
    saved_encoder: tuple[HexSpatialEncoderForHub, Path],
) -> None:
    """Saving produces safetensors weights and a JSON config."""
    _, destination = saved_encoder
    written = {p.name for p in destination.iterdir()}

    assert "model.safetensors" in written, "weights must be safetensors, never a pickle"
    assert "config.json" in written
    assert not any(name.endswith((".pt", ".pth", ".bin")) for name in written), (
        f"unexpected pickled artifact among {sorted(written)}"
    )


def test_saved_config_round_trips_into_the_dataclass(
    saved_encoder: tuple[HexSpatialEncoderForHub, Path],
) -> None:
    """``config.json`` is a flat document that rebuilds the config exactly."""
    _, destination = saved_encoder
    payload = json.loads((destination / "config.json").read_text())

    assert all(not isinstance(v, (dict, list)) for v in payload.values()), "config must be flat"
    assert HexEncoderConfig.from_dict(payload) == _CONFIG


def test_from_pretrained_reproduces_weights(
    saved_encoder: tuple[HexSpatialEncoderForHub, Path], sample: torch.Tensor
) -> None:
    """A reloaded encoder matches the original parameter for parameter."""
    original, destination = saved_encoder

    reloaded = HexSpatialEncoderForHub.from_pretrained(destination).eval()

    original_state = original.state_dict()
    reloaded_state = reloaded.state_dict()
    assert original_state.keys() == reloaded_state.keys()
    for key, value in original_state.items():
        assert torch.allclose(value, reloaded_state[key], atol=1e-6), f"weight drift in {key}"


def test_from_pretrained_reproduces_outputs(
    saved_encoder: tuple[HexSpatialEncoderForHub, Path], sample: torch.Tensor
) -> None:
    """The reloaded encoder is numerically equivalent on a real forward pass."""
    original, destination = saved_encoder
    reloaded = HexSpatialEncoderForHub.from_pretrained(destination).eval()

    with torch.no_grad():
        assert torch.allclose(original(sample), reloaded(sample), atol=1e-6)


def test_round_trip_needs_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving and loading a local directory performs no HTTP calls.

    Guards against a future change that resolves a local path through the Hub
    API, which would make offline use impossible.
    """
    import huggingface_hub

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unexpected network access during a local round trip")

    for attribute in ("snapshot_download", "hf_hub_download"):
        if hasattr(huggingface_hub, attribute):
            monkeypatch.setattr(huggingface_hub, attribute, _explode)

    destination = tmp_path / "offline"
    HexSpatialEncoderForHub.from_encoder_config(_CONFIG).save_pretrained(destination)

    assert HexSpatialEncoderForHub.from_pretrained(destination) is not None


def test_hub_subclass_is_weight_compatible_with_plain_encoder(sample: torch.Tensor) -> None:
    """Weights move between the plain and hub classes without translation.

    This is what makes publication possible: training produces a plain encoder,
    and only the published artifact needs the hub subclass.
    """
    plain = HexSpatialEncoder(_CONFIG).eval()

    hub = HexSpatialEncoderForHub.from_encoder(plain).eval()

    with torch.no_grad():
        assert torch.allclose(plain(sample), hub(sample), atol=1e-6)


def test_from_encoder_rejects_mismatched_architecture() -> None:
    """A strict transfer refuses weights from a differently-shaped encoder."""
    source = HexSpatialEncoder(_CONFIG)
    mismatched = HexSpatialEncoderForHub.from_encoder_config(
        HexEncoderConfig(**{**_CONFIG.to_dict(), "embed_dim": 128})
    )

    with pytest.raises(RuntimeError):
        mismatched.load_state_dict(source.state_dict(), strict=True)


def test_import_without_extra_names_the_extra() -> None:
    """Without the hub dependency the module raises a directive ImportError.

    Run in a subprocess with ``huggingface_hub`` blocked, since the parent
    process has it installed.
    """
    script = textwrap.dedent(
        """
        import sys, importlib.abc


        class _Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] == "huggingface_hub":
                    raise ModuleNotFoundError("blocked by test")
                return None


        sys.meta_path.insert(0, _Blocker())

        try:
            import alpha_zero.hexcrystal.hub_mixin  # noqa: F401
        except ImportError as exc:
            print("MESSAGE:" + str(exc))
        else:
            print("NO_ERROR")

        # The base subpackage must remain usable without the extra.
        import alpha_zero.hexcrystal  # noqa: F401
        print("BASE_OK")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )

    assert result.returncode == 0, result.stderr
    assert "NO_ERROR" not in result.stdout, "import unexpectedly succeeded without the extra"
    assert "hub" in result.stdout, f"error message should name the extra: {result.stdout}"
    assert "BASE_OK" in result.stdout, "base subpackage must survive a missing optional extra"
