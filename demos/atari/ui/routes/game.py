"""REST endpoints for game control."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from demos.atari.ui.config import AtariUIConfig
from demos.atari.ui.session import SessionManager

logger = logging.getLogger(__name__)


def create_game_router(session_manager: SessionManager, config: AtariUIConfig) -> APIRouter:
    """Create the game REST API router."""
    router = APIRouter(tags=["game"])

    @router.post("/start")
    async def start_game() -> dict[str, Any]:
        """Start a new game session."""
        session = session_manager.create_session(config)
        return session.get_state()

    @router.post("/step")
    async def step(session_id: str, action: int) -> dict[str, Any]:
        """Execute a human action."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        result = await asyncio.to_thread(session.human_step, action)
        return result

    @router.post("/ai_step")
    async def ai_step(session_id: str) -> dict[str, Any]:
        """Execute an AI step (MCTS search + action)."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        result = await asyncio.to_thread(session.ai_step)
        return result

    @router.get("/state")
    async def get_state(session_id: str) -> dict[str, Any]:
        """Get current game state."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.get_state()

    @router.post("/reset")
    async def reset_game(session_id: str) -> dict[str, Any]:
        """Reset the current game."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.reset()

    @router.post("/save_episode")
    async def save_episode(session_id: str) -> dict[str, str]:
        """Save episode recording."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        path = session.save_episode()
        return {"path": path}

    @router.post("/load_model")
    async def load_model(session_id: str, path: str) -> dict[str, str]:
        """Load a model checkpoint into the session."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        session.load_model(path)
        return {"status": "ok", "path": path}

    return router
