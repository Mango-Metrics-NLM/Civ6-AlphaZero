"""Shared utilities for the Civ6 bridge layer.

Houses tiny, dependency-light helpers that would otherwise duplicate
across :mod:`alpha_zero.bridge.agent_loop`, :mod:`alpha_zero.inference.
inference_engine`, and the future :mod:`alpha_zero.bridge.mcts_policy`.

The bias is hard: nothing goes here unless at least two existing call
sites would benefit. This file is **not** a junk drawer.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def extract_scalar_value(
    value: Any | None,
    *,
    default: float = 0.0,
    log_per_player: bool = False,
) -> float:
    """Reduce a value-head tensor of any supported shape to a Python ``float``.

    Networks across the project return ``value`` with one of:
        - ``()``     legacy scalar
        - ``(1,)``   2-player network (current convention)
        - ``(1, 1)`` squeeze-omitted 2-player output
        - ``(1, N)`` N-player value vector (multiplayer networks)
    PR #82 fixed the inference engine to handle all four; this helper
    centralises the same logic so :mod:`alpha_zero.bridge.agent_loop`
    and any future caller stay in sync.

    Args:
        value: Torch tensor, numpy array, scalar, or ``None``.
        default: Returned when ``value`` is ``None`` or has zero elements.
        log_per_player: When ``True`` and the input has more than one
            element, emit a DEBUG log noting which player's value was
            returned. Off by default since the agent loop reports this
            via its own diagnostics path.

    Returns:
        The first element interpreted as a Python ``float``. Multi-player
        tensors return player 0's value (consistent with PR #82).
    """
    if value is None:
        return default

    try:
        flat = value.reshape(-1)
    except AttributeError:
        # Plain Python scalar or other duck-typed numeric.
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    numel = getattr(flat, "numel", None)
    if callable(numel):
        size = int(numel())
    else:
        # numpy / array-like
        size = int(getattr(flat, "size", 0))
    if size == 0:
        return default

    first = flat[0]
    try:
        scalar = float(first.item())
    except AttributeError:
        scalar = float(first)

    if size > 1 and log_per_player:
        shape = getattr(value, "shape", None)
        # Message intentionally contains the phrase "per-player value tensor"
        # — consumers (and tests) grep for this substring as the operator-
        # facing signal that a multi-player network output was flattened.
        logger.debug(
            "Reduced per-player value tensor (shape=%s) to player-0 scalar (%.4f)",
            tuple(shape) if shape is not None else "<unknown>",
            scalar,
        )

    return scalar


def seed_msg_counter() -> int:
    """Seed a monotonic message-ID counter from the current wall-clock time.

    Shared by :class:`alpha_zero.bridge.file_bridge.FileBridge` and
    :class:`alpha_zero.bridge.tuner_bridge.TunerBridge` so their message
    IDs are seeded identically (milliseconds since the epoch), keeping IDs
    unlikely to collide across process restarts.

    Returns:
        The current time in integer milliseconds.
    """
    return int(time.time() * 1000)


def next_monotonic_msg_id(current: int) -> int:
    """Advance a monotonic message-ID counter by one.

    Callers own the counter's storage (e.g. ``self._msg_counter``) and pass
    its current value in; this keeps the counter a plain, directly
    inspectable/settable attribute on the owning bridge (several tests do
    exactly that) while sharing the increment algorithm itself.

    Args:
        current: The counter's current value.

    Returns:
        ``current + 1`` - the new counter value and next message ID.
    """
    return current + 1


def parse_bool_env(value: str | None, default: bool) -> bool:
    """Parse a boolean-flavoured environment variable value.

    Recognises the vocabulary shared across the repo's config parsers:
    ``"1"``/``"true"``/``"yes"``/``"on"`` (case-insensitive) are truthy,
    ``"0"``/``"false"``/``"no"``/``"off"`` are falsy, and anything else -
    including ``None`` or an unrecognised string - returns ``default``.

    Args:
        value: Raw environment variable value, or ``None`` if unset.
        default: Value to return when ``value`` is unset or unrecognised.

    Returns:
        The parsed boolean.
    """
    if value is None:
        return default
    normalized = value.lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return default
