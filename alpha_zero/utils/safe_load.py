"""The repository's single gated ``torch.load`` with an unsafe path.

This is a **leaf** module: it imports only the standard library and torch, so
both ``alpha_zero.core.checkpoint_utils`` and ``alpha_zero.utils.checkpoint``
can depend on it without forming a cycle (``core/__init__`` eagerly imports
``civ6_network``, which imports ``utils.checkpoint`` at module scope, so
``utils.checkpoint`` can never import from ``core``).

Policy (tech-debt hardening spec WS4.1):

* ``torch.load(..., weights_only=True)`` is always attempted first;
* a rejection **never** downgrades automatically -- it raises
  :class:`UnsafeCheckpointError` naming the source and the two opt-ins;
* the opt-ins are the per-call ``allow_unsafe=True`` keyword and the
  process-wide environment variable :data:`UNSAFE_CHECKPOINT_ENV_VAR`
  (``CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1``), read from ``os.environ`` at call
  time; either logs a WARNING and loads with ``weights_only=False``, which can
  execute arbitrary code from the file.

``tests/security/test_no_unsafe_torch_load.py`` pins that :func:`safe_torch_load`
is the only ``weights_only=False`` call in ``alpha_zero/``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import IO, Any

import torch

logger = logging.getLogger(__name__)

#: Environment variable that authorises ``weights_only=False`` loading for the
#: whole process. Read from ``os.environ`` at call time, so it can be set after
#: import (e.g. by a test fixture or a job launcher).
UNSAFE_CHECKPOINT_ENV_VAR = "CIV6AZ_ALLOW_UNSAFE_CHECKPOINT"

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

__all__ = [
    "UNSAFE_CHECKPOINT_ENV_VAR",
    "UnsafeCheckpointError",
    "safe_torch_load",
    "unsafe_checkpoint_loading_allowed",
]


class UnsafeCheckpointError(RuntimeError):
    """A checkpoint was rejected by ``weights_only=True`` and no opt-in was given.

    Raised instead of silently retrying with ``weights_only=False``. The message
    names the checkpoint and both opt-in mechanisms.
    """


def unsafe_checkpoint_loading_allowed(allow_unsafe: bool = False) -> bool:
    """Return True when a ``weights_only=False`` load has been authorised.

    Args:
        allow_unsafe: Explicit per-call opt-in. Wins when True.

    Returns:
        ``allow_unsafe``, or whether :data:`UNSAFE_CHECKPOINT_ENV_VAR` holds a
        truthy value (``1``/``true``/``yes``/``on``, case-insensitive) in the
        environment *right now*.
    """
    if allow_unsafe:
        return True
    return os.environ.get(UNSAFE_CHECKPOINT_ENV_VAR, "").strip().lower() in _TRUTHY_VALUES


def safe_torch_load(
    source: str | Path | IO[bytes],
    *,
    map_location: str | torch.device | None = "cpu",
    allow_unsafe: bool = False,
) -> Any:
    """The repository's single ``torch.load`` site with an unsafe path.

    Tries ``weights_only=True`` first. If torch's tensor-only unpickler rejects
    the payload, the outcome depends on authorisation:

    * not authorised (default): raise :class:`UnsafeCheckpointError` chained to
      torch's error, naming the source, the ``allow_unsafe`` keyword and
      :data:`UNSAFE_CHECKPOINT_ENV_VAR`;
    * authorised (``allow_unsafe=True`` or the env var): log a WARNING and load
      again with ``weights_only=False``.

    Args:
        source: Path, or a seekable binary file object (its position is reset
            before the second attempt).
        map_location: Passed to ``torch.load``.
        allow_unsafe: Explicit opt-in; see
            :func:`unsafe_checkpoint_loading_allowed`.

    Returns:
        Whatever ``torch.load`` returned.

    Raises:
        UnsafeCheckpointError: Safe load rejected and no opt-in was given.
    """
    label = str(source) if isinstance(source, (str, Path)) else f"<{type(source).__name__}>"
    try:
        return torch.load(source, map_location=map_location, weights_only=True)
    except Exception as exc:
        if not unsafe_checkpoint_loading_allowed(allow_unsafe):
            raise UnsafeCheckpointError(
                f"Checkpoint {label} could not be loaded with weights_only=True "
                f"({type(exc).__name__}: {exc}). Automatic fallback to "
                "weights_only=False has been removed because it executes arbitrary "
                "code from the file. If this checkpoint comes from a trusted source, "
                f"pass allow_unsafe=True or set {UNSAFE_CHECKPOINT_ENV_VAR}=1."
            ) from exc
        authority = "allow_unsafe=True" if allow_unsafe else f"{UNSAFE_CHECKPOINT_ENV_VAR}=1"
        logger.warning(
            "Loading checkpoint %s with weights_only=False (authorised by %s); this can "
            "execute arbitrary code from the file. Safe load was rejected with: %s",
            label,
            authority,
            exc,
        )
        if hasattr(source, "seek"):
            source.seek(0)
        return torch.load(source, map_location=map_location, weights_only=False)
