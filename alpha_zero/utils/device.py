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

import contextlib
import logging
import os
from collections.abc import Iterator
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_INFERENCE_THREADS",
    "INFERENCE_THREADS_ENV_VAR",
    "cuda_ordinal",
    "inference_thread_limit",
    "place_network",
    "resolve_device",
    "resolve_device_str",
    "resolve_inference_threads",
]

_VALID_PREFIXES = ("cpu", "cuda", "mps")


def _normalise(requested: str | None) -> str:
    """Lower-case and default an incoming device request."""
    if requested is None:
        return "auto"
    text = str(requested).strip().lower()
    return text or "auto"


def cuda_ordinal(device: object) -> int | None:
    """The CUDA ordinal a device string names, or ``None`` if it names no CUDA device.

    Bare ``cuda`` is ordinal 0; ``cuda:N`` is ``N``; anything else -- ``cpu``,
    ``auto``, ``mps``, a malformed ``cuda:x``, ``None`` -- is ``None``. Needs no
    torch, so torch-free modules (the run-card driver, evidence records) can ask
    which GPU a device string means without importing it.
    """
    name = str(device or "").strip().lower()
    if name == "cuda":
        return 0
    if name.startswith("cuda:") and name[5:].isdigit():
        return int(name[5:])
    return None


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
            if (cuda_ordinal(text) or 0) >= torch.cuda.device_count():
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


def _import_torch() -> ModuleType | None:
    """Import torch lazily, returning None on a broken or absent install.

    Mirrors the inline try/import the resolver functions above use; the thread
    helpers need the module object itself rather than just a fallback value.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - exercised only on a broken install
        logger.warning("torch is unavailable; leaving thread configuration untouched")
        return None
    return torch


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


# --------------------------------------------------------------------------
# CPU thread count for single-state inference
# --------------------------------------------------------------------------
#
# MCTS evaluates one position at a time, so every self-play forward pass is a
# batch of exactly 1. Torch defaults its intra-op pool to the core count, and
# splitting a tiny tensor across every core costs far more than it saves: the
# OpenMP barrier is entered per operator, and its default *active* wait policy
# spins. On an oversubscribed VM the spinning threads and the worker threads
# fight for the same cores.
#
# Measured on this project's 4-vCPU CI-class box (MicroCivNetwork, 561k params,
# 6x6 board, torch 2.13, batch of 1):
#
#     threads=1   1.72 ms      threads=3    20.72 ms
#     threads=2   1.54 ms      threads=4   347.12 ms
#
# 225x, from thread count alone -- which is why network-guided self-play looked
# unusably slow while the random-rollout baseline did not: SimpleMCTS never
# calls the network.
#
# The cap is applied only around inference. Batched training genuinely does
# scale with cores on a real machine, so it keeps torch's own default; capping
# globally would trade this bug for a slower trainer.

#: Threads to use for single-state inference when nothing else is configured.
#: Above 2 the intra-op split costs more than it saves for a batch of 1.
DEFAULT_MAX_INFERENCE_THREADS = 2

#: Set to a positive integer to pin the inference thread count, or to ``0`` /
#: ``"auto"`` to leave torch's default untouched (opting out entirely).
INFERENCE_THREADS_ENV_VAR = "CIV6AZ_INFERENCE_THREADS"


def resolve_inference_threads(requested: int | None = None) -> int | None:
    """Resolve the intra-op thread count to use for single-state inference.

    Precedence: explicit *requested* argument, then ``CIV6AZ_INFERENCE_THREADS``,
    then :data:`DEFAULT_MAX_INFERENCE_THREADS` capped by torch's current setting
    (so a single-core box is never asked for two threads).

    Returns ``None`` to mean "change nothing", which is what ``0``, ``"auto"``
    and an unparseable env value all resolve to. Returning ``None`` rather than
    raising keeps a typo in an env var from taking down a training run; it is
    logged.
    """
    if requested is not None:
        return requested if requested > 0 else None

    raw = os.environ.get(INFERENCE_THREADS_ENV_VAR)
    if raw is not None:
        value = raw.strip().lower()
        if value in {"auto", ""}:
            return None
        try:
            parsed = int(value)
        except ValueError:
            logger.warning(
                "%s=%r is not an integer or 'auto'; leaving torch threads unchanged.",
                INFERENCE_THREADS_ENV_VAR,
                raw,
            )
            return None
        return parsed if parsed > 0 else None

    torch_module = _import_torch()
    if torch_module is None:
        return None
    try:
        current = int(torch_module.get_num_threads())
    except Exception:  # pragma: no cover - defensive
        return None
    return min(current, DEFAULT_MAX_INFERENCE_THREADS)


@contextlib.contextmanager
def inference_thread_limit(requested: int | None = None) -> Iterator[int | None]:
    """Temporarily cap torch's intra-op threads for a single-state inference phase.

    Yields the thread count actually applied, or ``None`` when the setting was
    left alone. The previous value is always restored, including on exception,
    so a training phase that follows self-play keeps torch's default.

    A no-op when torch is missing or refuses the call, so callers can wrap a
    phase unconditionally without guarding for either case.
    """
    threads = resolve_inference_threads(requested)
    torch_module = _import_torch()
    if threads is None or torch_module is None:
        yield None
        return

    try:
        previous = int(torch_module.get_num_threads())
    except Exception:  # pragma: no cover - defensive
        yield None
        return

    if previous == threads:
        yield threads
        return

    try:
        torch_module.set_num_threads(threads)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not set torch threads to %s: %s", threads, exc)
        yield None
        return

    logger.debug("Inference thread cap: %s -> %s", previous, threads)
    try:
        yield threads
    finally:
        try:
            torch_module.set_num_threads(previous)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Failed to restore torch thread count to %s", previous)
