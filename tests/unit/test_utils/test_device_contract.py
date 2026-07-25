"""Repository-wide device-contract guards.

Two regressions are cheap to reintroduce and expensive to notice:

1. A checkpoint written on a GPU box becoming unloadable on a CPU-only box,
   which breaks the charter's NG-4 ("no workflow requires a GPU").
2. A bare ``.numpy()`` on a tensor that might live on CUDA, which raises
   ``TypeError`` only once someone actually has a GPU.

Neither needs a GPU present to be caught, which is the point.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import torch

from alpha_zero.utils.checkpoint import (
    compute_weight_checksum,
    load_model_state,
    safe_load_checkpoint,
    save_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "alpha_zero"

# ``.numpy()`` is only safe once the tensor is on the host, and ``.cpu()`` is
# the only call that guarantees that — ``.detach()`` drops the grad history but
# leaves a CUDA tensor on the GPU, so ``.detach().numpy()`` is flagged too.
_BARE_NUMPY = re.compile(r"(?<!\.cpu\(\))\.numpy\(\)")


def _iter_package_sources() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


class TestNoBareNumpyConversions:
    """Every ``.numpy()`` in the package must be preceded by ``.cpu()``."""

    def test_no_bare_numpy_calls(self) -> None:
        offenders: list[str] = []
        for source in _iter_package_sources():
            for lineno, line in enumerate(source.read_text().splitlines(), start=1):
                if _BARE_NUMPY.search(line):
                    rel = source.relative_to(REPO_ROOT)
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")

        assert not offenders, (
            "Found .numpy() calls not preceded by .cpu(). These raise TypeError "
            "on CUDA tensors:\n" + "\n".join(offenders)
        )


class TestCheckpointMapLocationDefault:
    """The loaders must default to host memory, not the save-time device."""

    @pytest.mark.parametrize("func", [safe_load_checkpoint, load_model_state])
    def test_map_location_defaults_to_cpu(self, func: object) -> None:
        default = inspect.signature(func).parameters["map_location"].default
        assert default == "cpu", (
            f"{func.__name__} defaults map_location to {default!r}. "
            "torch.load then restores tensors to the device recorded at save "
            "time, so a GPU-trained checkpoint raises on a CPU-only host."
        )


class TestCrossDeviceCheckpointRoundTrip:
    """A checkpoint saved on one device must load on the other."""

    @staticmethod
    def _tiny_state_dict(device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "layer.weight": torch.randn(4, 4, device=device),
            "layer.bias": torch.zeros(4, device=device),
        }

    def test_round_trip_on_available_device(self, tmp_path: Path) -> None:
        """Save on whatever device exists, reload mapped to CPU."""
        from alpha_zero.utils.device import resolve_device

        device = resolve_device("auto")
        state = self._tiny_state_dict(device)
        path = tmp_path / "model.pt"
        save_checkpoint(path, {k: v.cpu() for k, v in state.items()})

        reloaded = load_model_state(path)

        assert all(t.device.type == "cpu" for t in reloaded.values())
        for key, tensor in state.items():
            assert torch.allclose(reloaded[key], tensor.cpu())

    def test_default_load_needs_no_explicit_map_location(self, tmp_path: Path) -> None:
        """The NG-4 path: a bare load must not depend on the save-time device."""
        path = tmp_path / "model.pt"
        save_checkpoint(path, self._tiny_state_dict(torch.device("cpu")))

        result = safe_load_checkpoint(path)

        assert result.data["model_state_dict"]["layer.weight"].device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_gpu_saved_checkpoint_loads_on_cpu(self, tmp_path: Path) -> None:
        """The direction that actually broke: CUDA-tagged tensors, CPU host."""
        path = tmp_path / "gpu_model.pt"
        torch.save(
            {"model_state_dict": self._tiny_state_dict(torch.device("cuda"))},
            path,
        )

        reloaded = load_model_state(path)

        assert all(t.device.type == "cpu" for t in reloaded.values())


class TestChecksumIsDeviceAgnostic:
    """``compute_weight_checksum`` is used as the cross-device equality proof."""

    def test_checksum_survives_a_round_trip(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(8, 4)
        before = compute_weight_checksum(model)

        path = tmp_path / "linear.pt"
        save_checkpoint(path, model.state_dict())
        restored = torch.nn.Linear(8, 4)
        restored.load_state_dict(load_model_state(path))

        # approx, not ==: the checksum normalizes neither dtype nor device.
        assert compute_weight_checksum(restored) == pytest.approx(before, rel=1e-6)
