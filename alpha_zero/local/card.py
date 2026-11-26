"""A run card: the YAML a registered run is described by, parsed and validated.

A card (``configs/run_cards/*.yaml``) names a commit, a game budget, the torch
build and GPUs it expects, and a list of invocations. This module is the model
and the loader only: it reads a file and either returns a :class:`RunCard` or
raises :class:`RunCardError`. Pricing, pinning, the device check and execution
live in :mod:`alpha_zero.local.run_card`, which re-exports these names so
existing imports keep working.

Card keys
---------
``card_id`` (a run id), ``sha`` (7-40 hex digits), ``max_games``, ``invocations``
(``name`` / ``runner`` / ``argv`` each), and optionally ``approval_ref``,
``cuda_visible_devices``, ``torch_spec`` + ``torch_index_url`` (both or neither),
``gpus`` (``cuda:N`` -> the GPU name expected there) and ``description``. An
unknown key is refused: a typo must not silently drop a constraint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.local.pinned_env import PinnedEnvError, TorchPin
from alpha_zero.utils.data_root import validate_run_id
from alpha_zero.utils.device import cuda_ordinal

__all__ = ["Invocation", "RunCard", "RunCardError", "load_card"]

_CARD_KEYS = frozenset(
    {
        "card_id",
        "sha",
        "max_games",
        "approval_ref",
        "invocations",
        "cuda_visible_devices",
        "torch_spec",
        "torch_index_url",
        "gpus",
        "description",
    }
)
_INVOCATION_KEYS = frozenset({"name", "runner", "argv"})
_SHA_CHARS = frozenset("0123456789abcdefABCDEF")


class RunCardError(RuntimeError):
    """The card is malformed, unpriceable or cannot be run."""


@dataclass(frozen=True)
class Invocation:
    name: str
    runner: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class RunCard:
    card_id: str
    sha: str
    max_games: int
    invocations: tuple[Invocation, ...]
    approval_ref: str | None = None
    cuda_visible_devices: str | None = None
    path: Path | None = None
    torch_pin: TorchPin | None = None
    #: CUDA ordinal -> the GPU name the card expects there (``gpus:`` in the YAML).
    gpus: Mapping[int, str] = field(default_factory=dict)


def _scalar_arg(value: Any, where: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise RunCardError(f"{where}: argv entries must be strings or numbers, got {value!r}")
    return str(value)


def load_card(path: str | Path) -> RunCard:
    """Parse and validate a run card. Raises :class:`RunCardError` (or ``DataRootError``)."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RunCardError(f"{path}: a run card is a YAML mapping")
    unknown = set(raw) - _CARD_KEYS
    if unknown:
        raise RunCardError(f"{path}: unknown card keys {sorted(unknown)}")
    card_id = validate_run_id(str(raw.get("card_id", "")))
    sha = str(raw.get("sha", ""))
    if not (7 <= len(sha) <= 40 and set(sha) <= _SHA_CHARS):
        raise RunCardError(f"{path}: sha must be a 7-40 hex-digit commit, got {sha!r}")
    max_games = raw.get("max_games")
    if isinstance(max_games, bool) or not isinstance(max_games, int) or max_games < 0:
        raise RunCardError(f"{path}: max_games must be a non-negative integer")
    items = raw.get("invocations")
    if not isinstance(items, list) or not items:
        raise RunCardError(f"{path}: invocations must be a non-empty list")
    invocations: list[Invocation] = []
    for i, item in enumerate(items):
        where = f"{path}: invocations[{i}]"
        if not isinstance(item, dict) or set(item) - _INVOCATION_KEYS:
            raise RunCardError(f"{where} must be a mapping of {sorted(_INVOCATION_KEYS)}")
        name = validate_run_id(str(item.get("name", "")))
        argv = item.get("argv") or []
        if not isinstance(argv, list):
            raise RunCardError(f"{where}: argv must be a list")
        invocations.append(
            Invocation(
                name=name,
                runner=str(item.get("runner", "")),
                argv=tuple(_scalar_arg(a, where) for a in argv),
            )
        )
    names = [inv.name for inv in invocations]
    if len(set(names)) != len(names):
        raise RunCardError(f"{path}: duplicate invocation names in {names}")
    approval = raw.get("approval_ref")
    cuda = raw.get("cuda_visible_devices")
    torch_spec, torch_index = raw.get("torch_spec"), raw.get("torch_index_url")
    torch_pin: TorchPin | None = None
    if (torch_spec is None) != (torch_index is None):
        raise RunCardError(f"{path}: torch_spec and torch_index_url go together (both or neither)")
    if torch_spec is not None:
        try:
            torch_pin = TorchPin(str(torch_spec), str(torch_index))
        except PinnedEnvError as exc:
            raise RunCardError(f"{path}: {exc}") from exc
    return RunCard(
        card_id=card_id,
        sha=sha.lower(),
        max_games=max_games,
        invocations=tuple(invocations),
        approval_ref=(str(approval).strip() or None) if approval is not None else None,
        cuda_visible_devices=None if cuda is None else str(cuda),
        path=path,
        torch_pin=torch_pin,
        gpus=_load_gpus(raw.get("gpus"), path),
    )


def _load_gpus(raw: Any, path: Path) -> dict[int, str]:
    """``gpus:`` -- ``{"cuda:0": "<GPU name>", ...}`` -- as ``{ordinal: name}``."""
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not raw:
        raise RunCardError(f"{path}: gpus must be a non-empty mapping of cuda:N to a GPU name")
    gpus: dict[int, str] = {}
    for key, name in raw.items():
        ordinal = cuda_ordinal(key)
        if ordinal is None or str(key).strip().lower() == "cuda" or not str(name).strip():
            raise RunCardError(f"{path}: gpus entry {key!r}: {name!r} is not 'cuda:N: <GPU name>'")
        gpus[ordinal] = str(name).strip()
    return gpus
