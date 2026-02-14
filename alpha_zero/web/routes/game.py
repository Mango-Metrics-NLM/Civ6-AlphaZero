"""Game REST API routes.

Provides endpoints for game CRUD operations: creating games,
submitting moves, undo, resign, and querying game state.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from alpha_zero.web.serializers import (
    serialize_confidence,
    serialize_game_state,
    serialize_turn_result,
)

logger = logging.getLogger(__name__)


class NewGameRequest(BaseModel):
    """Request body for creating a new game."""

    human_player: int = Field(default=0, ge=0, le=1)
    seed: int | None = None
    opponent_type: str | None = None
    board_size: int | None = Field(default=None, ge=4, le=32)


class MoveRequest(BaseModel):
    """Request body for submitting a move."""

    action: int = Field(ge=0)


def create_game_router(session_manager: Any) -> APIRouter:
    """Create the game API router.

    Args:
        session_manager: GameSessionManager instance.

    Returns:
        FastAPI APIRouter with game endpoints.
    """
    router = APIRouter(tags=["game"])

    @router.post("/new")
    async def create_game(request: NewGameRequest) -> dict[str, Any]:
        """Create a new game session.

        Returns session_id and initial game state.
        """
        try:
            session = session_manager.create_session(
                human_player=request.human_player,
                seed=request.seed,
                opponent_type=request.opponent_type,
                board_size=request.board_size,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        state = serialize_game_state(
            session.session_id,
            session.controller.current_state,
            session.adapter,
            session.action_space,
        )

        logger.info("Game created: %s", session.session_id)
        return {"session_id": session.session_id, "state": state}

    @router.get("/{session_id}")
    async def get_game_state(session_id: str) -> dict[str, Any]:
        """Get current game state for a session."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        state = serialize_game_state(
            session.session_id,
            session.controller.current_state,
            session.adapter,
            session.action_space,
        )
        return {"state": state}

    @router.post("/{session_id}/move")
    async def submit_move(session_id: str, request: MoveRequest) -> dict[str, Any]:
        """Submit a human move and get AI response.

        If it becomes the AI's turn after the human move, the AI move
        is computed and included in the response.
        """
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        controller = session.controller

        if not controller.is_human_turn:
            raise HTTPException(status_code=409, detail="Not human's turn")

        if not controller.is_action_valid(request.action):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {request.action}",
            )

        # Execute human move
        try:
            human_result = controller.play_human_move(request.action)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        response: dict[str, Any] = {
            "human_move": serialize_turn_result(human_result, session.action_space),
        }

        # Auto-play AI move if it's AI's turn and game not over
        if not controller.is_game_over and not controller.is_human_turn:
            # Update env reference for MCTS
            if hasattr(session.ai_agent, "set_env"):
                session.ai_agent.set_env(session.adapter.env)

            # Run AI move in thread pool to avoid blocking
            ai_result = await asyncio.to_thread(controller.play_ai_move)
            response["ai_move"] = serialize_turn_result(ai_result, session.action_space)
            response["confidence"] = serialize_confidence(ai_result, session.action_space)

        # Include updated state
        response["state"] = serialize_game_state(
            session.session_id,
            controller.current_state,
            session.adapter,
            session.action_space,
        )

        return response

    @router.post("/{session_id}/undo")
    async def undo_move(session_id: str) -> dict[str, Any]:
        """Undo the last move (or last two if AI moved after human)."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        controller = session.controller
        success = controller.undo_move()

        # If we undid an AI move, also undo the preceding human move
        if success and not controller.is_human_turn:
            controller.undo_move()

        state = serialize_game_state(
            session.session_id,
            controller.current_state,
            session.adapter,
            session.action_space,
        )

        return {"success": success, "state": state}

    @router.post("/{session_id}/resign")
    async def resign_game(session_id: str) -> dict[str, Any]:
        """Resign the current game."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            result = session.controller.resign()
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        state = serialize_game_state(
            session.session_id,
            session.controller.current_state,
            session.adapter,
            session.action_space,
        )

        return {
            "result": serialize_turn_result(result, session.action_space),
            "state": state,
        }

    @router.post("/{session_id}/pause")
    async def pause_game(session_id: str) -> dict[str, Any]:
        """Pause the current game."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        session.controller.pause_game()

        state = serialize_game_state(
            session.session_id,
            session.controller.current_state,
            session.adapter,
            session.action_space,
        )
        return {"state": state}

    @router.post("/{session_id}/resume")
    async def resume_game(session_id: str) -> dict[str, Any]:
        """Resume a paused game."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        session.controller.resume_game()

        state = serialize_game_state(
            session.session_id,
            session.controller.current_state,
            session.adapter,
            session.action_space,
        )
        return {"state": state}

    @router.get("/{session_id}/valid-actions")
    async def get_valid_actions(session_id: str) -> dict[str, Any]:
        """Get decoded list of valid actions."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        from alpha_zero.web.serializers import decode_action_readable

        actions = [
            {
                "index": idx,
                "label": decode_action_readable(idx, session.action_space),
            }
            for idx in session.controller.valid_actions
        ]

        return {"actions": actions}

    @router.get("/{session_id}/summary")
    async def get_game_summary(session_id: str) -> dict[str, Any]:
        """Get game summary statistics."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        controller = session.controller
        state = controller.current_state

        total_moves = len(state.history)
        human_moves = sum(1 for t in state.history if t.is_human)
        ai_moves = total_moves - human_moves
        ai_thinking_total = sum(t.thinking_time for t in state.history if not t.is_human)

        return {
            "session_id": session.session_id,
            "opponent_type": session.opponent_type,
            "human_player": session.human_player,
            "total_moves": total_moves,
            "human_moves": human_moves,
            "ai_moves": ai_moves,
            "ai_thinking_time_total": round(ai_thinking_total, 3),
            "game_over": controller.is_game_over,
            "phase": state.phase.value,
            "turn_number": state.turn_number,
            "winner": state.history[-1].winner if state.history and state.history[-1].game_over else None,
            "age_seconds": round(session.age_seconds, 1),
        }

    return router
