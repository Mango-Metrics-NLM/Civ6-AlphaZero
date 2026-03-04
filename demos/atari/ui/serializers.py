"""Serializers for converting game state to JSON-friendly formats."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert an RGB numpy frame to a base64-encoded PNG string.

    Args:
        frame: RGB image array (H, W, 3) uint8 or float32.

    Returns:
        Base64-encoded PNG string.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed — returning empty frame")
        return ""

    if frame.dtype == np.float32 or frame.dtype == np.float64:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

    if frame.ndim == 2:
        img = Image.fromarray(frame, mode="L")
    else:
        img = Image.fromarray(frame, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def mcts_tree_to_json(
    root: Any,
    max_depth: int = 4,
    min_visits: int = 1,
    action_names: list[str] | None = None,
) -> dict[str, Any]:
    """Serialize an MCTS search tree to JSON for D3.js visualization.

    Args:
        root: MCTS root node (must have children, visit_count, value attrs).
        max_depth: Maximum tree depth to serialize.
        min_visits: Minimum visit count to include a node.
        action_names: Optional action index -> name mapping.

    Returns:
        Dict with "nodes" and "edges" lists.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_id = 0

    def _traverse(node: Any, depth: int, parent_id: int | None) -> None:
        nonlocal node_id
        current_id = node_id
        node_id += 1

        visit_count = getattr(node, "visit_count", 0)
        if visit_count < min_visits and depth > 0:
            return

        value = 0.0
        if visit_count > 0:
            value = getattr(node, "value_sum", 0.0) / visit_count

        prior = getattr(node, "prior", 0.0)

        nodes.append(
            {
                "id": current_id,
                "visits": visit_count,
                "value": round(float(value), 4),
                "prior": round(float(prior), 4),
                "depth": depth,
            }
        )

        if parent_id is not None:
            action = getattr(node, "action", None)
            action_label = str(action)
            if action_names and action is not None and 0 <= action < len(action_names):
                action_label = action_names[action]
            edges.append(
                {
                    "source": parent_id,
                    "target": current_id,
                    "action": action_label,
                }
            )

        if depth < max_depth:
            children = getattr(node, "children", {})
            if isinstance(children, dict):
                for child in children.values():
                    _traverse(child, depth + 1, current_id)
            elif isinstance(children, list):
                for child in children:
                    _traverse(child, depth + 1, current_id)

    _traverse(root, 0, None)
    return {"nodes": nodes, "edges": edges}


def metrics_to_json(metrics: dict[str, Any]) -> dict[str, Any]:
    """Sanitize training metrics for JSON serialization.

    Args:
        metrics: Raw metrics dict (may contain numpy types).

    Returns:
        JSON-safe dict.
    """
    result = {}
    for key, val in metrics.items():
        if isinstance(val, (np.floating, np.integer)):
            result[key] = float(val)
        elif isinstance(val, np.ndarray):
            result[key] = val.tolist()
        else:
            result[key] = val
    return result
