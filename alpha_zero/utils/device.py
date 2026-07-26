"""Single source of truth for resolving a requested torch device.

Before this module the repository resolved devices in ~30 places using three
mutually incompatible idioms: unconditional ``"cuda" if torch.cuda.is_available()
else "cpu"`` (which ignores what the user asked for), ``config.device if
torch.cuda.is_available() else "cpu"`` (which discards a valid ``"cuda:1"``), and
bare ``device: str = "cpu"`` defaults that passed an unvalidated user string
straight to ``.to()``. The grammars disagreed too — the bridge rejected
``"cuda:0"`` and ``"auto"`` that the training entrypoint accepted.

The rules here are deliberately boring:

* ``None`` / ``"auto"`` — use CUDA when available, else MPS when available, else CPU.
* ``"cpu"`` — always honoured.
* ``"cuda"`` / ``"cuda:N"`` — honoured when available; otherwise a warning is
  logged and CPU is returned. An out-of-range ``N`` also falls back rather than
  raising deep inside ``.to()``.
* ``"mps"`` — honoured when available, else warn and fall back to CPU.
* Anything else raises ``ValueError`` naming the accepted grammar.

Falling back is always *reported*. A caller that warns and then keeps using the
unresolved string is the bug this module exists to prevent, so callers should
store what these functions return, never what the user typed.

Torch is imported lazily so that ``--help`` on a CLI does not pay the import
cost, and so a broken install degrades to a clear message instead of an
``ImportError`` traceback at module import.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

__all__ = ["place_network", "resolve_device", "resolve_device_str"]

_VALID_PREFIXES = ("cpu", "cuda", "mps")


def _normalise(requested: str | None) -> str:
    """Lower-case and default an incoming device request."""
    if requested is None:
        return "auto"
    text = str(requested).strip().lower()
    return text or "auto"


def _resolve_name(requested: str | None) -> str:
    """Resolve ``requested`` to a concrete device string.

    Returns ``"cpu"`` when torch is unavailable so that ``--help`` paths and
    import-time defaults keep working on a broken install.
    """
    text = _normalise(requested)

    try:
        import torch
    except Exception:  # pragma: no cover - exercised only on a broken install
        logger.warning("torch is unavailable; resolving device to cpu")
        return "cpu"

    cuda_ok = torch.cuda.is_available()

    if text == "auto":
        if cuda_ok:
            return "cuda"
        if _mps_available(torch):
            return "mps"
        return "cpu"

    if text == "cpu":
        return "cpu"

    if text.startswith("cuda"):
        if not cuda_ok:
            logger.warning("device %r requested but CUDA is unavailable; using cpu", text)
            return "cpu"
        if ":" in text:
            index = text.split(":", 1)[1]
            if not index.isdigit():
                raise ValueError(
                    f"Invalid device {requested!r}: expected 'cuda:N' with an integer N"
                )
            if int(index) >= torch.cuda.device_count():
                logger.warning(
                    "device %r requested but only %d CUDA device(s) present; using cpu",
                    text,
                    torch.cuda.device_count(),
                )
                return "cpu"
        return text

    if text == "mps":
        if not _mps_available(torch):
            logger.warning("device 'mps' requested but unavailable; using cpu")
            return "cpu"
        return "mps"

    raise ValueError(
        f"Invalid device {requested!r}: expected one of "
        f"{_VALID_PREFIXES} (optionally 'cuda:N'), or 'auto'"
    )


def _mps_available(torch_module: object) -> bool:
    """Return True when Apple MPS is usable, tolerating older torch builds."""
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    if is_available is None:
        return False
    try:
        return bool(is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def resolve_device_str(requested: str | None = None) -> str:
    """Resolve ``requested`` to a device string such as ``"cpu"`` or ``"cuda:1"``.

    Use this for config dataclass fields and anything serialized, where a
    ``torch.device`` would not round-trip through YAML or JSON.
    """
    return _resolve_name(requested)


def place_network(network: object, requested: str | None = None) -> str:
    """Resolve ``requested`` and move ``network`` onto it, returning the name.

    Search classes historically stored a device string and assumed *someone
    else* had already moved the network there. That held only by luck — the web
    session happened to call ``.to()`` afterwards — so constructing a search
    object directly with ``device="cuda"`` produced a CPU network fed CUDA
    inputs. Doing both here makes the two agree by construction.

    Test doubles that stand in for a network need not implement ``.to``; those
    are left untouched rather than being forced to grow one. A network that
    *does* have ``.to`` but fails to move — CUDA OOM, an invalid device — raises
    here rather than being swallowed: the alternative is a mixed-device
    RuntimeError surfacing much further from the cause.
    """
    name = _resolve_name(requested)
    mover = getattr(network, "to", None)
    if callable(mover):
        mover(name)
    return name


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve ``requested`` to a concrete :class:`torch.device`.

    Raises:
        RuntimeError: If torch is not importable, since no ``torch.device`` can
            be constructed without it. Use :func:`resolve_device_str` on paths
            that must survive a missing torch.
        ValueError: If ``requested`` is not a recognised device string.
    """
    name = _resolve_name(requested)
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only on a broken install
        raise RuntimeError(
            "torch is required to construct a torch.device; "
            "use resolve_device_str() if a string is sufficient"
        ) from exc
    return torch.device(name)
