"""Replay REST API routes.

Provides endpoints for listing, loading, and managing game replays.
Delegates to the existing ReplayManager from human_interface.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)


def create_replay_router(replay_manager: Any | None) -> APIRouter:
    """Create the replay API router.

    Args:
        replay_manager: ReplayManager instance (or None if replays disabled).

    Returns:
        FastAPI APIRouter with replay endpoints.
    """
    router = APIRouter(tags=["replay"])

    @router.get("/list")
    async def list_replays() -> dict[str, Any]:
        """List all saved game replays."""
        if replay_manager is None:
            return {"replays": [], "enabled": False}

        try:
            replays = replay_manager.list_replays()
        except Exception:
            logger.exception("Error listing replays")
            replays = []

        return {"replays": replays, "enabled": True}

    @router.get("/{game_id}")
    async def get_replay(game_id: str) -> dict[str, Any]:
        """Get full replay data for a game."""
        if replay_manager is None:
            raise HTTPException(status_code=404, detail="Replays not enabled")

        try:
            replay = replay_manager.load_replay(game_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"Replay not found: {game_id}") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        summary = replay.get_summary()
        return {"game_id": game_id, "summary": summary}

    @router.get("/{game_id}/turn/{turn_number}")
    async def get_replay_turn(game_id: str, turn_number: int) -> dict[str, Any]:
        """Get data for a specific turn in a replay."""
        if replay_manager is None:
            raise HTTPException(status_code=404, detail="Replays not enabled")

        try:
            replay = replay_manager.load_replay(game_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"Replay not found: {game_id}") from e

        turn_data = replay.get_turn(turn_number)
        if turn_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Turn {turn_number} not found in replay {game_id}",
            )

        return {"game_id": game_id, "turn_number": turn_number, "data": turn_data}

    @router.delete("/{game_id}")
    async def delete_replay(game_id: str) -> dict[str, Any]:
        """Delete a saved replay."""
        if replay_manager is None:
            raise HTTPException(status_code=404, detail="Replays not enabled")

        try:
            deleted = replay_manager.delete_replay(game_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Replay not found: {game_id}")

        logger.info("Replay deleted: %s", game_id)
        return {"deleted": True, "game_id": game_id}

    return router
