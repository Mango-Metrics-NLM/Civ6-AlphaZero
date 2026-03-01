"""WebSocket endpoint for real-time game frame streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from demos.atari.ui.session import SessionManager

logger = logging.getLogger(__name__)


def create_websocket_router(session_manager: SessionManager) -> APIRouter:
    """Create the WebSocket router for real-time game interaction."""
    router = APIRouter(tags=["websocket"])

    @router.websocket("/game/{session_id}")
    async def game_websocket(websocket: WebSocket, session_id: str) -> None:
        """WebSocket for real-time Atari game interaction.

        Protocol:
        Client -> Server:
            {"type": "action", "action": 2}       # Human action
            {"type": "ai_move"}                    # Request AI step
            {"type": "reset"}                      # Reset episode
            {"type": "load_model", "path": "..."}  # Switch checkpoint
            {"type": "ping"}

        Server -> Client:
            {"type": "frame", "data": "<base64>", "reward": 0.0, "done": false, "step": 42}
            {"type": "ai_move", "action": 3, "probs": [...], "value": 0.73}
            {"type": "episode_end", "total_reward": 21.0, "steps": 3200}
            {"type": "error", "message": "..."}
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
            state = session.get_state()
            await _send(websocket, {"type": "state", "data": state})

            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await _send(websocket, {"type": "error", "message": "Invalid JSON"})
                    continue

                session.touch()
                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    await _send(websocket, {"type": "pong"})

                elif msg_type == "action":
                    action = msg.get("action")
                    if action is None:
                        await _send(websocket, {"type": "error", "message": "Missing action"})
                        continue
                    result = await asyncio.to_thread(session.human_step, int(action))
                    await _send(websocket, {"type": "frame", "data": result})
                    if result.get("done"):
                        await _send(
                            websocket,
                            {
                                "type": "episode_end",
                                "total_reward": result["total_reward"],
                                "steps": result["step"],
                            },
                        )

                elif msg_type == "ai_move":
                    await _send(websocket, {"type": "ai_thinking"})
                    result = await asyncio.to_thread(session.ai_step)
                    await _send(
                        websocket,
                        {
                            "type": "ai_move",
                            "action": result["action"],
                            "probs": result["probs"],
                            "value": result["value"],
                            "frame": result["frame"],
                            "reward": result["reward"],
                            "done": result["done"],
                            "step": result["step"],
                        },
                    )
                    if result.get("done"):
                        await _send(
                            websocket,
                            {
                                "type": "episode_end",
                                "total_reward": result["total_reward"],
                                "steps": result["step"],
                            },
                        )

                elif msg_type == "reset":
                    state = session.reset()
                    await _send(websocket, {"type": "state", "data": state})

                elif msg_type == "load_model":
                    path = msg.get("path", "")
                    session.load_model(path)
                    await _send(websocket, {"type": "model_loaded", "path": path})

                else:
                    await _send(
                        websocket,
                        {
                            "type": "error",
                            "message": f"Unknown type: {msg_type}",
                        },
                    )

        except (WebSocketDisconnect, RuntimeError):
            logger.info("WebSocket disconnected: session=%s", session_id)
        except Exception:
            logger.exception("WebSocket error: session=%s", session_id)
            try:
                await _send(websocket, {"type": "error", "message": "Internal error"})
            except Exception:
                logger.debug("Failed to send error on broken WebSocket")

    return router


async def _send(websocket: WebSocket, data: dict[str, Any]) -> None:
    """Send JSON message over WebSocket."""
    try:
        await websocket.send_text(json.dumps(data, default=str))
    except Exception:
        logger.debug("Failed to send WebSocket message", exc_info=True)
