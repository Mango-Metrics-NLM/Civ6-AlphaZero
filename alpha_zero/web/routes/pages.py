"""HTML page routes for the HTMX frontend.

Serves full HTML pages and HTMX partial responses for
in-browser game interaction.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from alpha_zero.web.serializers import (
    decode_action_readable,
    serialize_board_for_template,
    serialize_confidence,
    serialize_move_history,
)

logger = logging.getLogger(__name__)


def create_pages_router(session_manager: Any, templates: Any) -> APIRouter:
    """Create the HTML pages router.

    Args:
        session_manager: GameSessionManager instance.
        templates: Jinja2Templates instance.

    Returns:
        FastAPI APIRouter with page routes.
    """
    router = APIRouter(tags=["pages"])

    def _require_session(session_id: str) -> Any:
        """Look up session or raise 404."""
        try:
            return session_manager.get_session_or_error(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Render the lobby / new game page."""
        sessions = session_manager.list_sessions()
        return cast(
            HTMLResponse,
            templates.TemplateResponse(
                request,
                "index.html",
                {
                    "sessions": sessions,
                    "active_count": session_manager.active_count,
                },
            ),
        )

    @router.post("/game/new")
    async def create_game_form(
        request: Request,
        human_player: int = Form(default=0),
        opponent_type: str = Form(default="random"),
        board_size: int = Form(default=16),
    ) -> Response:
        """Handle new game form submission."""
        try:
            session = session_manager.create_session(
                human_player=human_player,
                opponent_type=opponent_type,
                board_size=board_size,
            )
        except (RuntimeError, ValueError) as e:
            logger.error("Failed to create game: %s", e)
            return cast(
                Response,
                templates.TemplateResponse(
                    request,
                    "index.html",
                    {
                        "error": str(e),
                        "sessions": session_manager.list_sessions(),
                        "active_count": session_manager.active_count,
                    },
                    status_code=400,
                ),
            )

        logger.info("Game created via form: %s", session.session_id)
        return RedirectResponse(
            url=f"/game/{session.session_id}",
            status_code=303,
        )

    @router.get("/game/{session_id}", response_class=HTMLResponse)
    async def game_page(request: Request, session_id: str) -> HTMLResponse:
        """Render the main game page."""
        session = _require_session(session_id)

        state = session.controller.current_state
        board = serialize_board_for_template(
            session.adapter,
            state.valid_actions,
            session.action_space,
            human_player=session.human_player,
        )
        history = serialize_move_history(state.history, session.action_space)

        # Get last AI confidence if available
        last_confidence = None
        for turn in reversed(state.history):
            if not turn.is_human and turn.ai_evaluation is not None:
                last_confidence = serialize_confidence(turn, session.action_space)
                break

        valid_actions = [
            {"index": idx, "label": decode_action_readable(idx, session.action_space)}
            for idx in state.valid_actions
        ]

        return cast(
            HTMLResponse,
            templates.TemplateResponse(
                request,
                "game.html",
                {
                    "session_id": session_id,
                    "phase": state.phase.value,
                    "turn_number": state.turn_number,
                    "current_player": state.current_player,
                    "human_player": session.human_player,
                    "is_human_turn": session.controller.is_human_turn,
                    "is_game_over": session.controller.is_game_over,
                    "board": board,
                    "history": history,
                    "valid_actions": valid_actions,
                    "confidence": last_confidence,
                    "opponent_type": session.opponent_type,
                    "config": session.config.to_dict(),
                },
            ),
        )

    @router.post("/htmx/game/{session_id}/move", response_class=HTMLResponse)
    async def htmx_move(
        request: Request,
        session_id: str,
        action: int = Form(...),
    ) -> HTMLResponse:
        """Handle move via HTMX form submission.

        Returns updated board + info partials for HTMX swap.
        """
        session = _require_session(session_id)
        controller = session.controller

        if not controller.is_human_turn:
            raise HTTPException(status_code=409, detail="Not human's turn")

        if not controller.is_action_valid(action):
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")

        # Execute human move
        controller.play_human_move(action)

        # Auto-play AI if needed
        if not controller.is_game_over and not controller.is_human_turn:
            if hasattr(session.ai_agent, "set_env"):
                session.ai_agent.set_env(session.adapter.env)
            await asyncio.to_thread(controller.play_ai_move)

        # Re-render full game page (HTMX will swap the target)
        return await game_page(request, session_id)

    @router.post("/htmx/game/{session_id}/undo", response_class=HTMLResponse)
    async def htmx_undo(request: Request, session_id: str) -> HTMLResponse:
        """Handle undo via HTMX."""
        session = _require_session(session_id)

        controller = session.controller
        controller.undo_move()
        if not controller.is_human_turn:
            controller.undo_move()

        return await game_page(request, session_id)

    @router.post("/htmx/game/{session_id}/resign", response_class=HTMLResponse)
    async def htmx_resign(request: Request, session_id: str) -> HTMLResponse:
        """Handle resign via HTMX."""
        session = _require_session(session_id)

        try:
            session.controller.resign()
        except RuntimeError:
            pass

        return await game_page(request, session_id)

    return router
