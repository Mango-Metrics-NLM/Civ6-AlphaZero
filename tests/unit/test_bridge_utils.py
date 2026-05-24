"""Tests for ``alpha_zero.bridge.utils``.

Covers the shape-normalising ``extract_scalar_value`` helper that is shared
between :mod:`alpha_zero.bridge.agent_loop` and
:mod:`alpha_zero.inference.inference_engine` (PR #82 invariant).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from alpha_zero.bridge.utils import extract_scalar_value


class TestExtractScalarValue:
    """``extract_scalar_value`` must accept every shape the project's value
    heads emit and reduce to a single Python ``float``."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (torch.tensor(0.42), 0.42),  # ()
            (torch.tensor([0.42]), 0.42),  # (1,)
            (torch.tensor([[0.42]]), 0.42),  # (1, 1)
            (torch.tensor([[0.7, -0.4, 0.1]]), 0.7),  # (1, N)
            (torch.tensor([0.42, -0.4]), 0.42),  # (N,) flat
        ],
    )
    def test_torch_shapes_round_trip(self, value: torch.Tensor, expected: float) -> None:
        assert extract_scalar_value(value) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (np.array(0.42, dtype=np.float32), 0.42),
            (np.array([0.42], dtype=np.float32), 0.42),
            (np.array([[0.7, -0.4]], dtype=np.float32), 0.7),
        ],
    )
    def test_numpy_shapes_round_trip(self, value: np.ndarray, expected: float) -> None:
        assert extract_scalar_value(value) == pytest.approx(expected)

    def test_python_scalar(self) -> None:
        assert extract_scalar_value(0.3) == pytest.approx(0.3)
        assert extract_scalar_value(7) == pytest.approx(7.0)

    def test_none_returns_default(self) -> None:
        assert extract_scalar_value(None) == 0.0
        assert extract_scalar_value(None, default=-1.5) == -1.5

    def test_empty_tensor_returns_default(self) -> None:
        assert extract_scalar_value(torch.empty(0)) == 0.0
        assert extract_scalar_value(np.empty(0)) == 0.0

    def test_unparseable_value_returns_default(self) -> None:
        """Strings / non-numeric objects must not raise; return the default."""

        class _NoReshape:
            pass

        assert extract_scalar_value(_NoReshape()) == 0.0
        assert extract_scalar_value("not a number") == 0.0

    def test_per_player_debug_log_off_by_default(self, caplog) -> None:
        """``log_per_player=False`` suppresses the per-player DEBUG message."""
        value = torch.tensor([[0.7, -0.4, 0.1]])
        with caplog.at_level(logging.DEBUG, logger="alpha_zero.bridge.utils"):
            extract_scalar_value(value)
        assert not any("per-player value tensor" in r.message for r in caplog.records)

    def test_per_player_debug_log_emitted_when_requested(self, caplog) -> None:
        """``log_per_player=True`` emits the operator-facing audit log."""
        value = torch.tensor([[0.7, -0.4, 0.1]])
        with caplog.at_level(logging.DEBUG, logger="alpha_zero.bridge.utils"):
            extract_scalar_value(value, log_per_player=True)
        matching = [r for r in caplog.records if "per-player value tensor" in r.message]
        assert len(matching) == 1
        # The shape tuple is in the log so operators can audit which network
        # produced the value — pin that contract.
        assert "1, 3" in matching[0].message

    def test_single_element_tensor_does_not_log(self, caplog) -> None:
        """``log_per_player=True`` is a no-op when the tensor has 1 element."""
        value = torch.tensor([[0.42]])
        with caplog.at_level(logging.DEBUG, logger="alpha_zero.bridge.utils"):
            extract_scalar_value(value, log_per_player=True)
        assert not any("per-player value tensor" in r.message for r in caplog.records)
