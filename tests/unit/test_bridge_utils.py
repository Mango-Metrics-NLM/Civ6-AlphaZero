"""Tests for ``alpha_zero.bridge.utils``.

Covers the shape-normalising ``extract_scalar_value`` helper that is shared
between :mod:`alpha_zero.bridge.agent_loop` and
:mod:`alpha_zero.inference.inference_engine` (PR #82 invariant), plus the
monotonic message-ID and boolean-env-var helpers shared across
:mod:`alpha_zero.bridge.file_bridge`, :mod:`alpha_zero.bridge.tuner_bridge`,
and :mod:`alpha_zero.bridge.config`.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest
import torch

from alpha_zero.bridge.utils import (
    MAX_MSG_ID,
    extract_scalar_value,
    next_monotonic_msg_id,
    parse_bool_env,
    seed_msg_counter,
)


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


class TestSeedMsgCounter:
    """``seed_msg_counter`` backs the ``FileBridge``/``TunerBridge`` seed."""

    def test_seed_is_wall_clock_milliseconds_mod_int32(self) -> None:
        # The seed is the millisecond timestamp reduced modulo MAX_MSG_ID so
        # it fits in a signed int32 (the game truncates the ack msg_id to
        # int32). It equals int(time.time()*1000) % MAX_MSG_ID.
        before = int(time.time() * 1000) % MAX_MSG_ID
        seeded = seed_msg_counter()
        after = int(time.time() * 1000) % MAX_MSG_ID
        # No modulo wrap occurred within this fast window unless before>after.
        if before <= after:
            assert before <= seeded <= after

    def test_seed_is_int32_safe(self) -> None:
        for _ in range(50):
            assert 0 <= seed_msg_counter() < MAX_MSG_ID

    def test_seed_returns_int(self) -> None:
        assert isinstance(seed_msg_counter(), int)


class TestNextMonotonicMsgId:
    """``next_monotonic_msg_id`` backs ``next_msg_id``/``_next_msg_id``."""

    def test_increments_by_one(self) -> None:
        assert next_monotonic_msg_id(100) == 101

    def test_repeated_calls_are_monotonic(self) -> None:
        counter = 0
        ids = []
        for _ in range(5):
            counter = next_monotonic_msg_id(counter)
            ids.append(counter)
        assert ids == [1, 2, 3, 4, 5]

    def test_wraps_to_one_before_int32_overflow(self) -> None:
        # At the ceiling the counter must wrap to 1, never emit MAX_MSG_ID or
        # above (which would truncate to a negative int32 in the game's ack).
        assert next_monotonic_msg_id(MAX_MSG_ID - 2) == MAX_MSG_ID - 1
        assert next_monotonic_msg_id(MAX_MSG_ID - 1) == 1


class TestParseBoolEnv:
    """``parse_bool_env`` backs ``BridgeConfig``/``AgentLoopConfig`` bool parsing."""

    @pytest.mark.parametrize("raw", ["1", "true", "True", "TRUE", "yes", "Yes", "on", "On"])
    def test_truthy_values(self, raw: str) -> None:
        assert parse_bool_env(raw, default=False) is True

    @pytest.mark.parametrize("raw", ["0", "false", "False", "FALSE", "no", "No", "off", "Off"])
    def test_falsy_values(self, raw: str) -> None:
        assert parse_bool_env(raw, default=True) is False

    def test_none_returns_default(self) -> None:
        assert parse_bool_env(None, default=True) is True
        assert parse_bool_env(None, default=False) is False

    def test_unrecognised_value_returns_default(self) -> None:
        assert parse_bool_env("garbage", default=True) is True
        assert parse_bool_env("garbage", default=False) is False

    def test_empty_string_returns_default(self) -> None:
        assert parse_bool_env("", default=True) is True
