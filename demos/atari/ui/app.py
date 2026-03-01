"""FastAPI application factory for the Atari UI."""

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

from demos.atari.ui import __version__
from demos.atari.ui.config import AtariUIConfig
from demos.atari.ui.session import SessionManager

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).parent
STATIC_DIR = _MODULE_DIR / "static"
TEMPLATE_DIR = _MODULE_DIR / "templates"


def create_app(config: AtariUIConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: UI configuration. Defaults are used when None.

    Returns:
        Configured FastAPI application instance.
    """
    config = config or AtariUIConfig()
    session_manager = SessionManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Starting Atari UI on %s:%d (game=%s, device=%s)",
            config.host,
            config.port,
            config.game,
            config.device,
        )
        cleanup_task = asyncio.create_task(_cleanup_loop(session_manager))
        yield
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Atari UI shutdown complete")

    app = FastAPI(
        title="Atari AlphaZero UI",
        description="Web interface for Atari AlphaZero demo",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    # Import and mount routes
    from demos.atari.ui.routes.game import create_game_router
    from demos.atari.ui.routes.mcts import create_mcts_router
    from demos.atari.ui.routes.pages import create_pages_router
    from demos.atari.ui.routes.training import create_training_router
    from demos.atari.ui.routes.websocket import create_websocket_router

    app.include_router(create_game_router(session_manager, config), prefix="/api/game")
    app.include_router(create_training_router(config), prefix="/api/training")
    app.include_router(create_mcts_router(session_manager), prefix="/api/mcts")
    app.include_router(create_websocket_router(session_manager), prefix="/ws")
    app.include_router(create_pages_router(session_manager, templates, config))

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/api/health")
    async def health_check() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "game": config.game,
            "active_sessions": session_manager.active_count,
        }

    app.state.config = config
    app.state.session_manager = session_manager

    logger.info("Atari UI app created with %d route(s)", len(app.routes))
    return app


async def _cleanup_loop(
    session_manager: SessionManager,
    interval: float = 60.0,
) -> None:
    """Background task to periodically clean up expired sessions."""
    while True:
        try:
            await asyncio.sleep(interval)
            removed = session_manager.cleanup_expired()
            if removed > 0:
                logger.info("Cleaned up %d expired session(s)", removed)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Error in session cleanup loop")
