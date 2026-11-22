"""Configuration and search-mode guard for :mod:`alpha_zero.civ6_local`.

Extracted from ``alpha_zero/civ6_local.py`` to keep configuration parsing,
environment/YAML/preset loading, and validation cleanly separated from CLI
runner dispatch while avoiding heavyweight ``torch``/``numpy`` imports at
config-load time.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

VALID_MODES: tuple[str, ...] = ("microciv", "civ6", "web")
VALID_LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")
VALID_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")  # "cuda:N" also accepted

ENV_PREFIX = "CIV6AZ_LOCAL_"

#: Bound demo action space sizes used when running the live Civ6 bridge without
#: a full-scale checkpoint.
DEFAULT_DEMO_MAX_UNITS: int = 20
DEFAULT_DEMO_MAX_CITIES: int = 5
DEFAULT_DEMO_NUM_PRODUCTION_ITEMS: int = 20

#: The gap this refusal answers: the live search mode cannot work as wired.
SEARCH_MODE_GAP_ID = "GM-B5"
#: The gap that has to close first: the bridge action translator.
SEARCH_TRANSLATOR_GAP_ID = "GM-B6"

__all__ = [
    "DEFAULT_DEMO_MAX_CITIES",
    "DEFAULT_DEMO_MAX_UNITS",
    "DEFAULT_DEMO_NUM_PRODUCTION_ITEMS",
    "ENV_PREFIX",
    "SEARCH_MODE_GAP_ID",
    "SEARCH_TRANSLATOR_GAP_ID",
    "VALID_DEVICES",
    "VALID_LOG_LEVELS",
    "VALID_MODES",
    "Civ6LocalConfig",
    "_get_default_bridge_ipc_dir",
    "_get_default_civ6_log_path",
    "search_mode_refusal",
]


def _get_default_bridge_ipc_dir() -> Path:
    """Return the default IPC directory for the Civ6 bridge.

    Delegates to ``alpha_zero.bridge.config.get_default_comm_dir`` which
    already handles OneDrive-redirected *Documents* folders on Windows.
    Falls back to ``%TEMP%/civ6az_bridge`` when the bridge package is not
    importable (e.g., during unit tests that stub out the bridge layer).
    """
    try:
        from alpha_zero.bridge.config import get_default_comm_dir

        resolved = get_default_comm_dir()
        logger.debug("Resolved bridge IPC directory: %s", resolved)
        return resolved
    except Exception:
        fallback = Path(tempfile.gettempdir()) / "civ6az_bridge"
        logger.debug("Using fallback bridge IPC directory: %s", fallback)
        return fallback


def _get_default_civ6_log_path() -> Path:
    """Return the default path to Civilization VI's ``Lua.log`` file.

    Delegates to :func:`alpha_zero.bridge.config.get_default_civ6_log_path`
    so CLI callers do not duplicate the default layout or import torch.
    """
    try:
        from alpha_zero.bridge.config import get_default_civ6_log_path

        resolved = get_default_civ6_log_path()
        logger.debug("Resolved Civ6 log path: %s", resolved)
        return resolved
    except Exception:
        fallback = (
            Path(os.getenv("LOCALAPPDATA", ""))
            / "Firaxis Games"
            / "Sid Meier's Civilization VI"
            / "Logs"
            / "Lua.log"
        )
        logger.debug("Using fallback Civ6 log path: %s", fallback)
        return fallback


@dataclass
class Civ6LocalConfig:
    """Configuration for local Civ6 gameplay.

    Attributes:
        mode: Play mode — microciv | civ6 | web
        checkpoint_path: Path to a trained model checkpoint (None = use random/heuristic AI)
        board_size: Board size for MicroCiv environment
        num_players: Number of players (2-8, MicroCiv only)
        num_simulations: MCTS simulations per move
        device: PyTorch inference device (cpu | cuda)
        use_mcts: Use MCTS policy instead of greedy argmax
        use_eval_harness: Enable MLflow-based BridgeEvalHarness for local civ6 play
        web_host: Bind address for web mode
        web_port: Port for web mode
        bridge_ipc_dir: IPC directory for civ6 bridge mode
        civ6_log_path: Path to Civilization VI's Lua.log, watched by the
            bridge in civ6 mode
        log_level: Logging verbosity (DEBUG | INFO | WARNING | ERROR)
        dry_run: Validate config and exit without starting gameplay
    """

    mode: str = "microciv"
    checkpoint_path: str | None = None
    board_size: int = 16
    num_players: int = 2
    num_simulations: int = 50
    device: str = "cpu"
    use_mcts: bool = False
    use_eval_harness: bool = False
    web_host: str = "127.0.0.1"
    web_port: int = 8080
    bridge_ipc_dir: str = field(default_factory=lambda: str(_get_default_bridge_ipc_dir()))
    civ6_log_path: str = field(default_factory=lambda: str(_get_default_civ6_log_path()))
    log_level: str = "INFO"
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.board_size < 4 or self.board_size > 64:
            raise ValueError(f"board_size must be 4-64, got {self.board_size}")
        if self.num_players < 2 or self.num_players > 8:
            raise ValueError(f"num_players must be 2-8, got {self.num_players}")
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {self.num_simulations}")
        # Resolve to a concrete device here so every downstream consumer sees
        # the same value. Previously validate() warned about falling back to
        # cpu without mutating self.device, so run_civ6_bridge kept handing the
        # unresolved "cuda" to the policy and WebConfig while the network had
        # already been placed on cpu. Import lazily so importing this module
        # alone does not trigger alpha_zero.utils package initialization.
        from alpha_zero.utils.device import resolve_device_str

        self.device = resolve_device_str(self.device)
        if self.web_port < 1 or self.web_port > 65535:
            raise ValueError(f"web_port must be 1-65535, got {self.web_port}")
        log_level_upper = self.log_level.upper()
        if log_level_upper not in VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of {VALID_LOG_LEVELS}, got {self.log_level!r}")
        self.log_level = log_level_upper

    @classmethod
    def from_yaml(cls, path: str) -> Civ6LocalConfig:
        """Load configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open() as f:
            raw = yaml.safe_load(f)

        if raw is None:
            return cls()

        # Support both top-level keys and a nested "civ6_local" section
        data: dict[str, Any] = raw.get("civ6_local", raw)

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}

        unknown = set(data.keys()) - known_fields
        if unknown:
            logger.warning("Ignoring unknown config keys: %s", sorted(unknown))

        return cls(**filtered)

    @classmethod
    def from_env(cls) -> Civ6LocalConfig:
        """Create configuration from ``CIV6AZ_LOCAL_`` environment variables."""
        base = cls()
        kwargs: dict[str, Any] = {}

        def _get(suffix: str) -> str | None:
            return os.environ.get(f"{ENV_PREFIX}{suffix}")

        def _get_bool(suffix: str, default: bool) -> bool:
            val = (_get(suffix) or "").lower()
            if val in ("1", "true", "yes"):
                return True
            if val in ("0", "false", "no"):
                return False
            return default

        if (v := _get("MODE")) is not None:
            kwargs["mode"] = v
        if (v := _get("CHECKPOINT_PATH")) is not None:
            kwargs["checkpoint_path"] = v if v else None
        if (v := _get("BOARD_SIZE")) is not None:
            kwargs["board_size"] = int(v)
        if (v := _get("NUM_PLAYERS")) is not None:
            kwargs["num_players"] = int(v)
        if (v := _get("NUM_SIMULATIONS")) is not None:
            kwargs["num_simulations"] = int(v)
        if (v := _get("DEVICE")) is not None:
            kwargs["device"] = v
        if (v := _get("USE_MCTS")) is not None:
            kwargs["use_mcts"] = _get_bool("USE_MCTS", False)
        if (v := _get("USE_EVAL_HARNESS")) is not None:
            kwargs["use_eval_harness"] = _get_bool("USE_EVAL_HARNESS", False)
        if (v := _get("WEB_HOST")) is not None:
            kwargs["web_host"] = v
        if (v := _get("WEB_PORT")) is not None:
            kwargs["web_port"] = int(v)
        if (v := _get("BRIDGE_IPC_DIR")) is not None:
            kwargs["bridge_ipc_dir"] = v
        if (v := _get("LOG_PATH")) is not None:
            kwargs["civ6_log_path"] = v
        if (v := _get("LOG_LEVEL")) is not None:
            kwargs["log_level"] = v

        # dry_run is additive — env can enable it but not disable an already-set flag
        if _get_bool("DRY_RUN", False):
            kwargs["dry_run"] = True

        if kwargs:
            return replace(base, **kwargs)
        return base

    @classmethod
    def from_preset(cls, preset: str) -> Civ6LocalConfig:
        """Create configuration from a named preset (``microciv_fast``, ``microciv_full``, ``civ6``)."""
        presets: dict[str, dict[str, Any]] = {
            "microciv_fast": {
                "mode": "microciv",
                "board_size": 8,
                "num_simulations": 10,
                "log_level": "WARNING",
            },
            "microciv_full": {
                "mode": "microciv",
                "board_size": 16,
                "num_simulations": 50,
                "log_level": "INFO",
            },
            "civ6": {
                "mode": "civ6",
                "num_simulations": 200,
                "log_level": "INFO",
            },
        }

        if preset not in presets:
            valid = sorted(presets.keys())
            raise ValueError(f"Unknown preset {preset!r}. Valid presets: {valid}")

        return cls(**presets[preset])

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a plain dictionary."""
        return {
            "mode": self.mode,
            "checkpoint_path": self.checkpoint_path,
            "board_size": self.board_size,
            "num_players": self.num_players,
            "num_simulations": self.num_simulations,
            "device": self.device,
            "use_mcts": self.use_mcts,
            "use_eval_harness": self.use_eval_harness,
            "web_host": self.web_host,
            "web_port": self.web_port,
            "bridge_ipc_dir": self.bridge_ipc_dir,
            "civ6_log_path": self.civ6_log_path,
            "log_level": self.log_level,
            "dry_run": self.dry_run,
        }

    def validate(self) -> list[str]:
        """Return a list of non-fatal warning strings."""
        warnings: list[str] = []

        if self.checkpoint_path is not None and not Path(self.checkpoint_path).exists():
            warnings.append(
                f"checkpoint_path does not exist: {self.checkpoint_path!r} — AI will use random play"
            )

        if self.mode == "civ6":
            bridge_dir = Path(self.bridge_ipc_dir)
            if not bridge_dir.exists():
                warnings.append(
                    f"bridge_ipc_dir does not exist: {bridge_dir} — "
                    "Civ6 must create it before the bridge starts"
                )

        return warnings


def search_mode_refusal(config: Civ6LocalConfig) -> str | None:
    """Why this config cannot start, when it asks for the search on a live game."""
    if config.mode != "civ6" or not config.use_mcts:
        return None
    return (
        f"--mcts is refused in civ6 mode ({SEARCH_MODE_GAP_ID}): the search plays in the "
        "MicroCiv simulator and nothing translates its actions into live commands yet "
        f"({SEARCH_TRANSLATOR_GAP_ID}). Run without --mcts (and without "
        f"{ENV_PREFIX}USE_MCTS) to use the greedy network policy. See docs/LIVE_CIV6.md."
    )
