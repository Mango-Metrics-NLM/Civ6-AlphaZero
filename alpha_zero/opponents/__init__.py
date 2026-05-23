"""Opponent adapters for AlphaZero training and evaluation.

This module introduces a thin ``Opponent`` Protocol that abstracts the
"thing that picks an action given a state" — distinct from the existing
``multiplayer/opponent_model.OpponentModel`` (which is a behavioural
*predictor*, not a *player*).

Adapters available:

- :class:`StockfishOpponent` — wraps a Stockfish UCI engine via
  ``python-chess.engine``.
- :class:`NetworkOpponent` — wraps a trained ``MicroCivNetwork`` for arena
  and evaluation matchups.

Both implement the runtime-checkable :class:`Opponent` Protocol, so
downstream code (training loop, tournament harness) can treat them
identically.
"""

from __future__ import annotations

from alpha_zero.opponents.base import Opponent
from alpha_zero.opponents.network_opponent import NetworkOpponent
from alpha_zero.opponents.stockfish_opponent import (
    StockfishOpponent,
    StockfishOpponentConfig,
)

__all__ = [
    "NetworkOpponent",
    "Opponent",
    "StockfishOpponent",
    "StockfishOpponentConfig",
]
