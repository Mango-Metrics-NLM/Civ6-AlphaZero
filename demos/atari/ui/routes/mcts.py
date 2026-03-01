"""REST endpoint for MCTS tree inspection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from demos.atari.ui.serializers import mcts_tree_to_json
from demos.atari.ui.session import SessionManager

# Atari action names for common games
ATARI_ACTIONS: dict[str, list[str]] = {
    "Pong": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
    "Breakout": ["NOOP", "FIRE", "RIGHT", "LEFT"],
    "SpaceInvaders": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
}


def create_mcts_router(session_manager: SessionManager) -> APIRouter:
    """Create the MCTS inspection router."""
    router = APIRouter(tags=["mcts"])

    @router.get("/tree")
    async def get_mcts_tree(
        session_id: str,
        max_depth: int = 4,
        min_visits: int = 1,
    ) -> dict[str, Any]:
        """Get the last MCTS search tree as JSON for visualization."""
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        root = session.get_mcts_root()
        if root is None:
            return {"nodes": [], "edges": [], "message": "No MCTS search performed yet"}

        action_names = ATARI_ACTIONS.get(session.ui_config.game)
        tree = mcts_tree_to_json(
            root, max_depth=max_depth, min_visits=min_visits, action_names=action_names
        )
        return tree

    return router
