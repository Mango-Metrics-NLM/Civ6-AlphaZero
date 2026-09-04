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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from alpha_zero.opponents.stockfish_opponent import (
        StockfishOpponent,
        StockfishOpponentConfig,
    )

__all__ = [
    "StockfishOpponent",
    "StockfishOpponentConfig",
]

# `stockfish_opponent` imports `chess` at module scope, and `chess` is an
# OPTIONAL extra. Importing it eagerly here made `import alpha_zero.opponents`
# -- and therefore any test that touches this package -- fail outright without
# that extra, which is how two unit modules came to abort collection for a
# dependency they never use. Resolved lazily (PEP 562), the same pattern the
# root package uses: the public names still resolve on attribute access, so this
# is behaviour-preserving for every caller that has `chess` installed, and a
# clear ImportError naming the extra for one that does not.
_LAZY_EXPORTS = {
    "StockfishOpponent": "alpha_zero.opponents.stockfish_opponent",
    "StockfishOpponentConfig": "alpha_zero.opponents.stockfish_opponent",
}


def __getattr__(name: str) -> Any:
    """Resolve a public attribute on first access (PEP 562).

    Args:
        name: Attribute looked up on ``alpha_zero.opponents``.

    Returns:
        The requested object, imported from its defining module.

    Raises:
        AttributeError: The name is not part of this package's public surface.
        ImportError: The name requires the optional ``chess`` extra.
    """
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import importlib

        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"{name} requires the optional 'chess' extra: pip install -e '.[chess]'"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value  # cache, so this runs once per process
    return value


def __dir__() -> list[str]:
    """Keep tab-completion and ``dir()`` reporting the public surface."""
    return sorted(__all__)
