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
