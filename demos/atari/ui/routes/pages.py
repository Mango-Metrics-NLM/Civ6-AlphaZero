"""HTML page routes serving Jinja2 templates."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from demos.atari.ui.config import AtariUIConfig
from demos.atari.ui.session import SessionManager


def create_pages_router(
    session_manager: SessionManager,
    templates: Jinja2Templates,
    config: AtariUIConfig,
) -> APIRouter:
    """Create the HTML page router."""
    router = APIRouter(tags=["pages"])

    @router.get("/")
    async def index(request: Request) -> Any:
        """Home page — redirects to play."""
        return templates.TemplateResponse(
            "play.html",
            {
                "request": request,
                "game": config.game,
                "config": config,
            },
        )

    @router.get("/play")
    async def play_page(request: Request) -> Any:
        """Human-vs-AI play mode."""
        return templates.TemplateResponse(
            "play.html",
            {
                "request": request,
                "game": config.game,
                "config": config,
            },
        )

    @router.get("/train")
    async def train_page(request: Request) -> Any:
        """Training dashboard."""
        return templates.TemplateResponse(
            "train.html",
            {
                "request": request,
                "game": config.game,
                "config": config,
            },
        )

    @router.get("/evaluate")
    async def evaluate_page(request: Request) -> Any:
        """Model evaluation page."""
        return templates.TemplateResponse(
            "evaluate.html",
            {
                "request": request,
                "game": config.game,
                "config": config,
            },
        )

    @router.get("/replay")
    async def replay_page(request: Request) -> Any:
        """Episode replay page."""
        return templates.TemplateResponse(
            "replay.html",
            {
                "request": request,
                "game": config.game,
                "config": config,
            },
        )

    return router
