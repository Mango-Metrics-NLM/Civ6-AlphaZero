"""Tests for the :class:`Opponent` Protocol and minimal compliance checks."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from alpha_zero.opponents import (
    NetworkOpponent,
    Opponent,
    StockfishOpponent,
    StockfishOpponentConfig,
)


class _DummyOpponent:
    """Minimal Opponent implementation used to exercise the Protocol check."""

    name = "dummy"

    def get_action(self, env: Any, action_mask: np.ndarray) -> int:
        return int(np.flatnonzero(action_mask > 0)[0])

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None


class TestProtocolCompliance:
    def test_dummy_satisfies_protocol(self) -> None:
        assert isinstance(_DummyOpponent(), Opponent)

    def test_stockfish_opponent_satisfies_protocol(self) -> None:
        engine = MagicMock()
        sf = StockfishOpponent(
            StockfishOpponentConfig(path="/nonexistent/bin/stockfish"),
            engine=engine,
        )
        try:
            assert isinstance(sf, Opponent)
        finally:
            sf.close()

    def test_network_opponent_satisfies_protocol(self) -> None:
        net = MagicMock()
        op = NetworkOpponent(network=net, name="frozen")
        assert isinstance(op, Opponent)


class TestProtocolName:
    def test_name_is_required_attribute(self) -> None:
        engine = MagicMock()
        sf = StockfishOpponent(
            StockfishOpponentConfig(path="/nonexistent", name="sf-test"),
            engine=engine,
        )
        try:
            assert sf.name == "sf-test"
        finally:
            sf.close()
