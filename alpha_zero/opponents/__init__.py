"""Opponent adapters for AlphaZero training and evaluation.

An opponent is the "thing that picks an action given a state" — distinct
from ``multiplayer/opponent_model.OpponentModel`` (a behavioural *predictor*,
not a *player*).

Adapters available:

- :class:`StockfishOpponent` — wraps a Stockfish UCI engine via
  ``python-chess.engine``.

The ``Opponent`` Protocol and ``NetworkOpponent`` adapter that once lived here
had no consumer and were removed on 2026-09-02; the arena path uses
``alpha_zero.evaluation.baselines.NeuralNetworkPlayer``.
"""

from __future__ import annotations

from alpha_zero.opponents.stockfish_opponent import (
    StockfishOpponent,
    StockfishOpponentConfig,
)

__all__ = [
    "StockfishOpponent",
    "StockfishOpponentConfig",
]
