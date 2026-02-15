"""FastAPI application factory for the web interface.

Creates and configures the FastAPI app with all routes,
middleware, and background tasks.

Usage:
    from alpha_zero.web.app import create_app
    from alpha_zero.web.config import WebConfig

    config = WebConfig.from_preset("dev")
    app = create_app(config)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from alpha_zero.web import DEFAULT_CLEANUP_INTERVAL, __version__
from alpha_zero.web.config import WebConfig
from alpha_zero.web.session import GameSessionManager

logger = logging.getLogger(__name__)

# Static and template directories
_MODULE_DIR = Path(__file__).parent
STATIC_DIR = _MODULE_DIR / "static"
TEMPLATE_DIR = _MODULE_DIR / "templates"


def create_app(config: WebConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Web configuration. If None, loads from environment.

    Returns:
        Configured FastAPI application instance.
    """

    config = config or WebConfig.from_env()

    # Session manager
    session_manager = GameSessionManager(config)

    # Replay manager (optional)
    replay_manager = _create_replay_manager(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Application lifespan: startup and shutdown."""
        logger.info(
            "Starting Civ6 AlphaZero Web UI on %s:%d (debug=%s)",
            config.host,
            config.port,
            config.debug,
        )

        # Start background cleanup task
        cleanup_task = asyncio.create_task(_session_cleanup_loop(session_manager))

        yield

        # Shutdown
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Web UI shutdown complete")

    app = FastAPI(
        title="Civ6 AlphaZero - Human vs AI",
        description="Web interface for playing MicroCiv against trained AI agents",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Templates
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    # Import and mount routes
    from alpha_zero.web.routes.game import create_game_router
    from alpha_zero.web.routes.pages import create_pages_router
    from alpha_zero.web.routes.replay import create_replay_router
    from alpha_zero.web.routes.websocket import create_websocket_router

    app.include_router(create_game_router(session_manager), prefix="/api/game")
    app.include_router(create_replay_router(replay_manager), prefix="/api/replay")
    app.include_router(create_websocket_router(session_manager), prefix="/ws")
    app.include_router(create_pages_router(session_manager, templates))

    # Static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Health check
    @app.get("/api/health")
    async def health_check() -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "ok",
            "active_sessions": session_manager.active_count,
            "version": __version__,
        }

    # Store config and manager on app for test access
    app.state.config = config
    app.state.session_manager = session_manager

    logger.info("FastAPI app created with %d route(s)", len(app.routes))
    return app


def _create_replay_manager(config: WebConfig) -> Any | None:
    """Create replay manager if replays are enabled.

    Args:
        config: Web configuration.

    Returns:
        ReplayManager instance or None if disabled.
    """
    if not config.enable_replay:
        return None

    try:
        from alpha_zero.human_interface.replay import ReplayManager

        hi_config = config.to_human_interface_config()
        replay_mgr = ReplayManager(hi_config)
        logger.debug("ReplayManager initialized (dir=%s)", config.replay_dir)
        return replay_mgr
    except ImportError:
        logger.warning("Could not import ReplayManager, replays disabled")
        return None
    except Exception:
        logger.exception("Failed to create ReplayManager")
        return None


async def _session_cleanup_loop(
    session_manager: GameSessionManager,
    interval_seconds: float = DEFAULT_CLEANUP_INTERVAL,
) -> None:
    """Background task that periodically cleans up expired sessions.

    Args:
        session_manager: Session manager to clean up.
        interval_seconds: Seconds between cleanup runs.
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            removed = session_manager.cleanup_expired()
            if removed > 0:
                logger.info("Cleaned up %d expired session(s)", removed)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Error in session cleanup loop")
