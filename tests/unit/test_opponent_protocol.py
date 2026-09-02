"""Minimal compliance checks for the opponent adapters.

The ``Opponent`` Protocol and ``NetworkOpponent`` adapter were removed on
2026-09-02 (no consumer); the protocol-conformance tests went with them.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from alpha_zero.opponents import StockfishOpponent, StockfishOpponentConfig


class TestOpponentName:
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
