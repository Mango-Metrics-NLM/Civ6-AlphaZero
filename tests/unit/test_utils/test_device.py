"""Tests for :mod:`alpha_zero.utils.device`.

These cover the fallback behaviour that keeps NG-4 ("no workflow requires a
GPU") true: a CUDA request on a CPU-only host must resolve to cpu and say so,
rather than crashing inside ``.to()`` further down the call stack.
"""

from __future__ import annotations

import logging

import pytest
import torch

from alpha_zero.utils.device import resolve_device, resolve_device_str


class TestResolveDeviceStr:
    """String resolution, the form used by config dataclasses."""

    def test_cpu_is_always_honoured(self) -> None:
        assert resolve_device_str("cpu") == "cpu"

    def test_none_and_auto_agree(self) -> None:
        assert resolve_device_str(None) == resolve_device_str("auto")

    def test_auto_matches_availability(self) -> None:
        resolved = resolve_device_str("auto")
        if torch.cuda.is_available():
            assert resolved == "cuda"
        else:
            assert resolved in {"cpu", "mps"}

    def test_case_and_whitespace_are_normalised(self) -> None:
        assert resolve_device_str("  CPU  ") == "cpu"

    def test_empty_string_is_treated_as_auto(self) -> None:
        assert resolve_device_str("") == resolve_device_str("auto")

    def test_unknown_device_raises_with_grammar(self) -> None:
        with pytest.raises(ValueError, match="Invalid device"):
            resolve_device_str("tpu")

    def test_malformed_cuda_index_raises(self) -> None:
        # Only reachable when CUDA is present; otherwise the unavailable
        # branch short-circuits to cpu first, which is also acceptable.
        if not torch.cuda.is_available():
            assert resolve_device_str("cuda:x") == "cpu"
        else:
            with pytest.raises(ValueError, match="integer N"):
                resolve_device_str("cuda:x")


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CPU-only host")
class TestCpuOnlyHostFallback:
    """The NG-4 guarantee: asking for CUDA on a CPU box must not explode."""

    def test_cuda_falls_back_to_cpu(self) -> None:
        assert resolve_device_str("cuda") == "cpu"

    def test_indexed_cuda_falls_back_to_cpu(self) -> None:
        assert resolve_device_str("cuda:2") == "cpu"

    def test_fallback_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="alpha_zero.utils.device"):
            resolve_device_str("cuda")
        assert any("CUDA is unavailable" in record.getMessage() for record in caplog.records)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestCudaHost:
    """Behaviour on a GPU box, so the second direction is not untested."""

    def test_cuda_is_honoured(self) -> None:
        assert resolve_device_str("cuda").startswith("cuda")

    def test_out_of_range_index_falls_back(self) -> None:
        too_high = torch.cuda.device_count() + 5
        assert resolve_device_str(f"cuda:{too_high}") == "cpu"

    def test_cpu_still_honoured_on_gpu_host(self) -> None:
        assert resolve_device_str("cpu") == "cpu"


class TestResolveDevice:
    """The ``torch.device`` form used at ``.to()`` call sites."""

    def test_returns_torch_device(self) -> None:
        assert isinstance(resolve_device("cpu"), torch.device)

    def test_agrees_with_string_form(self) -> None:
        assert resolve_device("auto").type == resolve_device_str("auto").split(":")[0]

    def test_usable_for_tensor_placement(self) -> None:
        device = resolve_device("auto")
        tensor = torch.zeros(2, 2).to(device)
        assert tensor.device.type == device.type
