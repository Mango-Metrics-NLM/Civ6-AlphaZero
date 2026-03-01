"""Atari AlphaZero web-based UI.

Provides a FastAPI-based web interface for:
- Human-vs-AI Atari play with real-time frame streaming
- Live training dashboards with Chart.js curves
- MCTS tree visualization with D3.js
- Episode replay with timeline scrubbing

Quick Start:
    python -m demos.atari.ui --game Pong --port 8080

Usage:
    from demos.atari.ui import create_app, AtariUIConfig

    config = AtariUIConfig(game="Pong")
    app = create_app(config)
"""

from __future__ import annotations

__version__ = "0.1.0"

try:
    from fastapi import FastAPI  # noqa: F401

    _UI_AVAILABLE = True
except ImportError:
    _UI_AVAILABLE = False


def __getattr__(name: str) -> object:
    """Lazy imports to avoid loading FastAPI when not needed."""
    if name == "create_app":
        from demos.atari.ui.app import create_app

        return create_app
    if name == "AtariUIConfig":
        from demos.atari.ui.config import AtariUIConfig

        return AtariUIConfig
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["_UI_AVAILABLE", "AtariUIConfig", "create_app"]
