"""WebSocket endpoint for real-time game updates.

Provides bidirectional communication for:
- Human moves submitted via WebSocket
- AI thinking notifications
- AI move results pushed to client
- Game state change events
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from alpha_zero.web.serializers import (
    serialize_confidence,
    serialize_game_state,
    serialize_turn_result,
)

logger = logging.getLogger(__name__)


def create_websocket_router(session_manager: Any) -> APIRouter:
    """Create the WebSocket router.

    Args:
        session_manager: GameSessionManager instance.

    Returns:
        FastAPI APIRouter with WebSocket endpoint.
    """
    router = APIRouter(tags=["websocket"])

    @router.websocket("/game/{session_id}")
    async def game_websocket(websocket: WebSocket, session_id: str) -> None:
        """WebSocket endpoint for real-time game interaction.

        Protocol:
        Client → Server:
            {"type": "move", "action": int}
            {"type": "undo"}
            {"type": "resign"}
            {"type": "ping"}

        Server → Client:
            {"type": "state", "data": {...}}
            {"type": "move_result", "data": {...}}
            {"type": "ai_thinking"}
            {"type": "ai_move", "data": {...}}
            {"type": "game_over", "data": {...}}
            {"type": "error", "message": str}
            {"type": "pong"}
        """
        session = session_manager.get_session(session_id)
        if session is None:
            await websocket.close(code=4004, reason="Session not found")
            return

        await websocket.accept()
        logger.info("WebSocket connected: session=%s", session_id)

        try:
            # Send initial state
            state = serialize_game_state(
                session.session_id,
                session.controller.current_state,
                session.adapter,
                session.action_space,
            )
            await _send_json(websocket, {"type": "state", "data": state})

            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await _send_json(
                        websocket,
                        {"type": "error", "message": "Invalid JSON"},
                    )
                    continue

                session.touch()
                msg_type = message.get("type", "")

                if msg_type == "ping":
                    await _send_json(websocket, {"type": "pong"})

                elif msg_type == "move":
                    await _handle_move(websocket, session, message)

                elif msg_type == "undo":
                    await _handle_undo(websocket, session)

                elif msg_type == "resign":
                    await _handle_resign(websocket, session)

                else:
                    await _send_json(
                        websocket,
                        {"type": "error", "message": f"Unknown message type: {msg_type}"},
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: session=%s", session_id)
        except Exception:
            logger.exception("WebSocket error: session=%s", session_id)
            try:
                await _send_json(
                    websocket,
                    {"type": "error", "message": "Internal server error"},
                )
            except Exception:
                logger.debug("Failed to send error message on broken WebSocket")

    async def _handle_move(
        websocket: WebSocket,
        session: Any,
        message: dict[str, Any],
    ) -> None:
        """Handle a move message from the client."""
        action = message.get("action")
        if action is None:
            await _send_json(
                websocket,
                {"type": "error", "message": "Missing 'action' field"},
            )
            return

        controller = session.controller

        if not controller.is_human_turn:
            await _send_json(
                websocket,
                {"type": "error", "message": "Not human's turn"},
            )
            return

        if not controller.is_action_valid(action):
            await _send_json(
                websocket,
                {"type": "error", "message": f"Invalid action: {action}"},
            )
            return

        # Execute human move
        try:
            human_result = controller.play_human_move(action)
        except (ValueError, RuntimeError) as e:
            await _send_json(
                websocket,
                {"type": "error", "message": str(e)},
            )
            return

        await _send_json(
            websocket,
            {
                "type": "move_result",
                "data": serialize_turn_result(human_result, session.action_space),
            },
        )

        # Check game over
        if controller.is_game_over:
            await _send_game_over(websocket, session)
            return

        # Auto-play AI move
        if not controller.is_human_turn:
            await _send_json(websocket, {"type": "ai_thinking"})

            # Update env reference
            if hasattr(session.ai_agent, "set_env"):
                session.ai_agent.set_env(session.adapter.env)

            # Run AI in thread pool
            ai_result = await asyncio.to_thread(controller.play_ai_move)

            ai_data = serialize_turn_result(ai_result, session.action_space)
            confidence = serialize_confidence(ai_result, session.action_space)
            ai_data["confidence"] = confidence

            await _send_json(websocket, {"type": "ai_move", "data": ai_data})

            if controller.is_game_over:
                await _send_game_over(websocket, session)
                return

        # Send updated state
        state = serialize_game_state(
            session.session_id,
            controller.current_state,
            session.adapter,
            session.action_space,
        )
        await _send_json(websocket, {"type": "state", "data": state})

    async def _handle_undo(websocket: WebSocket, session: Any) -> None:
        """Handle an undo request."""
        controller = session.controller
        success = controller.undo_move()

        if success and not controller.is_human_turn:
            controller.undo_move()

        state = serialize_game_state(
            session.session_id,
            controller.current_state,
            session.adapter,
            session.action_space,
        )
        await _send_json(
            websocket,
            {"type": "state", "data": state},
        )

    async def _handle_resign(websocket: WebSocket, session: Any) -> None:
        """Handle a resign request."""
        try:
            result = session.controller.resign()
            await _send_json(
                websocket,
                {
                    "type": "move_result",
                    "data": serialize_turn_result(result, session.action_space),
                },
            )
            await _send_game_over(websocket, session)
        except RuntimeError as e:
            await _send_json(
                websocket,
                {"type": "error", "message": str(e)},
            )

    async def _send_game_over(websocket: WebSocket, session: Any) -> None:
        """Send game over notification."""
        state = session.controller.current_state
        winner = None
        if state.history and state.history[-1].game_over:
            winner = state.history[-1].winner

        await _send_json(
            websocket,
            {
                "type": "game_over",
                "data": {
                    "winner": winner,
                    "human_player": session.human_player,
                    "human_won": winner == session.human_player,
                    "turn_number": state.turn_number,
                },
            },
        )

    return router


async def _send_json(websocket: Any, data: dict[str, Any]) -> None:
    """Send JSON message over WebSocket.

    Args:
        websocket: WebSocket connection.
        data: Data to serialize and send.
    """
    try:
        await websocket.send_text(json.dumps(data, default=str))
    except Exception:
        logger.debug("Failed to send WebSocket message", exc_info=True)
