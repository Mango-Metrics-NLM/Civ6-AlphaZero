"""Web-based Human vs AI interface for Civ6 AlphaZero.

This module provides a FastAPI-based web interface for playing MicroCiv
against trained AI agents in the browser.

Components:
- config: Web server and game configuration
- app: FastAPI application factory
- adapter: Bridges between MicroCivEnv/agents and GameController
- session: Concurrent game session management
- routes: REST, WebSocket, and HTMX endpoints
- templates: Jinja2 HTML templates with HTMX

Quick Start:
    python -m alpha_zero.web --preset dev

Usage:
    from alpha_zero.web import create_app, WebConfig

    config = WebConfig.from_preset("dev")
    app = create_app(config)
"""

from __future__ import annotations

# Module version (single source of truth)
__version__ = "0.1.0"

# Session ID generation length (hex characters from uuid4)
SESSION_ID_LENGTH = 12

# Checkpoint dict key for model state
CHECKPOINT_MODEL_KEY = "model_state_dict"

# Default number of top moves to show in UI
DEFAULT_TOP_MOVES = 5

# Session cleanup interval in seconds
DEFAULT_CLEANUP_INTERVAL = 60.0


def __getattr__(name: str) -> object:
    """Lazy imports to avoid loading FastAPI when not needed."""
    if name == "create_app":
        from alpha_zero.web.app import create_app

        return create_app
    if name == "WebConfig":
        from alpha_zero.web.config import WebConfig

        return WebConfig
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "WebConfig",
    "create_app",
]
