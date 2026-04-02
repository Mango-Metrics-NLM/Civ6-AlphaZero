"""Unified local entry point for full Civ6 gameplay.

Supports three modes of play:
- microciv: Play vs AI in the MicroCiv training environment
- civ6:     Play real Civilization 6 via the file-based Lua mod bridge
- web:      Launch the FastAPI web UI for browser-based play

Usage:
    python -m alpha_zero.civ6_local --help
    python -m alpha_zero.civ6_local --dry-run
    python -m alpha_zero.civ6_local --mode microciv
    python -m alpha_zero.civ6_local --mode civ6
    python -m alpha_zero.civ6_local --mode web
    python -m alpha_zero.civ6_local --config configs/civ6_local_gameplay.yaml
    python -m alpha_zero.civ6_local --checkpoint checkpoints/best.pt
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODES = ("microciv", "civ6", "web")
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
VALID_DEVICES = ("cpu", "cuda")

ENV_PREFIX = "CIV6AZ_LOCAL_"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


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
        web_host: Bind address for web mode
        web_port: Port for web mode
        bridge_ipc_dir: IPC directory for civ6 bridge mode
        log_level: Logging verbosity (DEBUG | INFO | WARNING | ERROR)
        dry_run: Validate config and exit without starting gameplay
    """

    mode: str = "microciv"
    checkpoint_path: str | None = None
    board_size: int = 16
    num_players: int = 2
    num_simulations: int = 50
    device: str = "cpu"
    web_host: str = "127.0.0.1"
    web_port: int = 8080
    bridge_ipc_dir: str = "/tmp/civ6az_bridge"  # noqa: S108
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
        if self.device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {VALID_DEVICES}, got {self.device!r}")
        if self.web_port < 1 or self.web_port > 65535:
            raise ValueError(f"web_port must be 1-65535, got {self.web_port}")
        log_level_upper = self.log_level.upper()
        if log_level_upper not in VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of {VALID_LOG_LEVELS}, got {self.log_level!r}")
        self.log_level = log_level_upper

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Civ6LocalConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Civ6LocalConfig with values from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML structure is invalid.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open() as f:
            raw = yaml.safe_load(f)

        if raw is None:
            return cls()

        # Support both top-level keys and a nested "civ6_local" section
        data: dict[str, Any] = raw.get("civ6_local", raw)

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known_fields}

        unknown = set(data.keys()) - known_fields
        if unknown:
            logger.warning("Ignoring unknown config keys: %s", sorted(unknown))

        return cls(**filtered)

    @classmethod
    def from_env(cls) -> "Civ6LocalConfig":
        """Create configuration from CIV6AZ_LOCAL_ environment variables.

        Supported variables (all optional):
            CIV6AZ_LOCAL_MODE              — play mode
            CIV6AZ_LOCAL_CHECKPOINT_PATH   — checkpoint file path
            CIV6AZ_LOCAL_BOARD_SIZE        — board size (int)
            CIV6AZ_LOCAL_NUM_PLAYERS       — number of players (int)
            CIV6AZ_LOCAL_NUM_SIMULATIONS   — MCTS simulations (int)
            CIV6AZ_LOCAL_DEVICE            — inference device
            CIV6AZ_LOCAL_WEB_HOST          — web server bind address
            CIV6AZ_LOCAL_WEB_PORT          — web server port (int)
            CIV6AZ_LOCAL_BRIDGE_IPC_DIR    — bridge IPC directory
            CIV6AZ_LOCAL_LOG_LEVEL         — logging level
            CIV6AZ_LOCAL_DRY_RUN           — dry run flag (1/true/yes)

        Returns:
            Civ6LocalConfig with environment overrides applied.
        """
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
        if (v := _get("WEB_HOST")) is not None:
            kwargs["web_host"] = v
        if (v := _get("WEB_PORT")) is not None:
            kwargs["web_port"] = int(v)
        if (v := _get("BRIDGE_IPC_DIR")) is not None:
            kwargs["bridge_ipc_dir"] = v
        if (v := _get("LOG_LEVEL")) is not None:
            kwargs["log_level"] = v

        # dry_run is additive — env can enable it but not disable an already-set flag
        if _get_bool("DRY_RUN", False):
            kwargs["dry_run"] = True

        if kwargs:
            from dataclasses import replace

            return replace(base, **kwargs)
        return base

    @classmethod
    def from_preset(cls, preset: str) -> "Civ6LocalConfig":
        """Create configuration from a named preset.

        Presets:
            microciv_fast  — MicroCiv with 8x8 board, 10 simulations (quick games)
            microciv_full  — MicroCiv with 16x16 board, 50 simulations (full quality)
            civ6           — Real Civ6 via bridge, 200 simulations

        Args:
            preset: Preset name.

        Returns:
            Civ6LocalConfig for the named preset.

        Raises:
            ValueError: If the preset is unknown.
        """
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
        """Serialise config to a plain dictionary.

        Returns:
            Dictionary representation of all config fields.
        """
        return {
            "mode": self.mode,
            "checkpoint_path": self.checkpoint_path,
            "board_size": self.board_size,
            "num_players": self.num_players,
            "num_simulations": self.num_simulations,
            "device": self.device,
            "web_host": self.web_host,
            "web_port": self.web_port,
            "bridge_ipc_dir": self.bridge_ipc_dir,
            "log_level": self.log_level,
            "dry_run": self.dry_run,
        }

    def validate(self) -> list[str]:
        """Return a list of non-fatal warning strings.

        Returns:
            List of warning messages (empty if everything looks fine).
        """
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

        if self.mode == "web" and self.web_host == "0.0.0.0":  # noqa: S104
            warnings.append(
                "web_host is 0.0.0.0 — server will accept connections from any interface"
            )

        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            warnings.append("device=cuda requested but CUDA is not available — falling back to cpu")

        return warnings


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


def run_microciv(config: Civ6LocalConfig) -> int:
    """Run a local MicroCiv game against the AI.

    Args:
        config: Validated local gameplay config.

    Returns:
        Exit code (0 = success).
    """
    import numpy as np
    import torch

    from alpha_zero.core.civ6_mcts import GumbelMCTS, GumbelMCTSConfig
    from alpha_zero.core.civ6_network import MicroCivNetwork
    from alpha_zero.envs.microciv import MicroCivEnv

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    env = MicroCivEnv(board_size=config.board_size)
    action_size = env.action_space.n
    obs_sample, _ = env.reset()
    spatial_channels = obs_sample["spatial"].shape[0]

    net = MicroCivNetwork(
        board_size=config.board_size,
        action_space_size=action_size,
        in_channels=spatial_channels,
    ).to(device)

    if config.checkpoint_path:
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        net.load_state_dict(state_dict)
        logger.info("Loaded checkpoint: %s", config.checkpoint_path)
    else:
        logger.info("No checkpoint — AI uses untrained (random) network")

    net.eval()

    gumbel_cfg = GumbelMCTSConfig(
        num_simulations=config.num_simulations,
        num_sampled_actions=min(config.num_simulations, 16),
    )
    _mcts = GumbelMCTS(net, gumbel_cfg, device=str(device))  # reserved for future use

    obs, _ = env.reset()
    done = False
    turn = 0

    logger.info(
        "Starting MicroCiv game: board=%dx%d, sims=%d",
        config.board_size,
        config.board_size,
        config.num_simulations,
    )

    while not done:
        turn += 1
        spatial = torch.from_numpy(obs["spatial"]).float().unsqueeze(0).to(device)
        vector = torch.from_numpy(obs["vector"]).float().unsqueeze(0).to(device)
        _action_mask = torch.from_numpy(obs["action_mask"]).float().unsqueeze(0).to(device)

        with torch.no_grad():
            policy_logits, _ = net(spatial, vector)

        # Use masked greedy selection (MCTS can be plugged in later)
        mask = obs["action_mask"].astype(bool)
        if mask.any():
            logits_np = policy_logits.cpu().numpy().flatten()
            logits_np[~mask] = -1e9
            action = int(np.argmax(logits_np))
        else:
            action = 0

        obs, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

        if turn % 10 == 0:
            logger.info("Turn %d | reward=%.2f", turn, float(reward))

    logger.info("Game over after %d turns. Final reward: %.2f", turn, float(reward))
    return 0


def run_civ6_bridge(config: Civ6LocalConfig) -> int:
    """Connect to real Civilization 6 via the file-based IPC bridge.

    Args:
        config: Validated local gameplay config.

    Returns:
        Exit code (0 = success).
    """
    try:
        from alpha_zero.bridge.config import BridgeConfig
        from alpha_zero.bridge.file_bridge import FileBridge
    except ImportError as exc:
        logger.error("Bridge module not available: %s", exc)
        return 1

    bridge_cfg = BridgeConfig(
        comm_dir=Path(config.bridge_ipc_dir),
        log_level=config.log_level,
    )
    bridge_cfg.ensure_comm_dir()

    logger.info("Connecting to Civ6 bridge at: %s", bridge_cfg.comm_dir)
    logger.info("Start Civilization 6 with the AlphaZeroAI mod to begin.")

    bridge = FileBridge(bridge_cfg)
    try:
        bridge.run()
    except KeyboardInterrupt:
        logger.info("Bridge stopped by user.")
    finally:
        bridge.close()

    return 0


def run_web(config: Civ6LocalConfig) -> int:
    """Launch the FastAPI web interface.

    Args:
        config: Validated local gameplay config.

    Returns:
        Exit code (0 = success).
    """
    try:
        import uvicorn

        from alpha_zero.web.app import create_app
        from alpha_zero.web.config import WebConfig
    except ImportError as exc:
        logger.error("Web dependencies not installed: %s", exc)
        logger.error("Install with: pip install -e '.[web]'")
        return 1

    web_cfg = WebConfig(
        host=config.web_host,
        port=config.web_port,
        board_size=config.board_size,
        mcts_simulations=config.num_simulations,
        device=config.device,
        checkpoint_path=config.checkpoint_path,
        log_level=config.log_level.lower(),
    )

    app = create_app(web_cfg)

    logger.info("Starting web interface at http://%s:%d", config.web_host, config.web_port)

    try:
        uvicorn.run(
            app,
            host=config.web_host,
            port=config.web_port,
            log_level=config.log_level.lower(),
        )
    except KeyboardInterrupt:
        logger.info("Web server stopped by user.")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for civ6_local.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="alpha_zero.civ6_local",
        description="Civ6 AlphaZero — Local Gameplay Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  microciv  Play the simplified MicroCiv training environment vs the AI.
  civ6      Connect to real Civilization 6 via the file-based Lua mod bridge.
  web       Launch the FastAPI web UI (open browser to play).

Environment variables (CIV6AZ_LOCAL_ prefix):
  CIV6AZ_LOCAL_MODE, CIV6AZ_LOCAL_BOARD_SIZE, CIV6AZ_LOCAL_NUM_SIMULATIONS,
  CIV6AZ_LOCAL_DEVICE, CIV6AZ_LOCAL_CHECKPOINT_PATH, CIV6AZ_LOCAL_WEB_HOST,
  CIV6AZ_LOCAL_WEB_PORT, CIV6AZ_LOCAL_BRIDGE_IPC_DIR, CIV6AZ_LOCAL_LOG_LEVEL

Examples:
  python -m alpha_zero.civ6_local --dry-run
  python -m alpha_zero.civ6_local --mode microciv --board-size 8
  python -m alpha_zero.civ6_local --mode web --web-port 3000
  python -m alpha_zero.civ6_local --preset microciv_fast --dry-run
""",
    )

    # Config source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--config",
        type=str,
        metavar="FILE",
        help="Path to YAML configuration file",
    )
    source_group.add_argument(
        "--preset",
        type=str,
        choices=["microciv_fast", "microciv_full", "civ6"],
        help="Use a named configuration preset",
    )

    # Core options
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(VALID_MODES),
        help="Play mode: microciv | civ6 | web",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        dest="checkpoint_path",
        metavar="PATH",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        dest="board_size",
        metavar="N",
        help="Board size for MicroCiv (default: 16)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        dest="num_players",
        metavar="N",
        help="Number of players for MicroCiv (2-8)",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        dest="num_simulations",
        metavar="N",
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=list(VALID_DEVICES),
        help="PyTorch inference device (default: cpu)",
    )

    # Web options
    parser.add_argument(
        "--web-host",
        type=str,
        dest="web_host",
        metavar="HOST",
        help="Web server bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        dest="web_port",
        metavar="PORT",
        help="Web server port (default: 8080)",
    )

    # Bridge options
    parser.add_argument(
        "--bridge-ipc-dir",
        type=str,
        dest="bridge_ipc_dir",
        metavar="DIR",
        help="IPC directory for Civ6 bridge (default: /tmp/civ6az_bridge)",
    )

    # General
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without starting gameplay",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        dest="log_level",
        choices=[lv.lower() for lv in VALID_LOG_LEVELS],
        help="Logging level (default: info)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging (shorthand for --log-level debug)",
    )

    return parser


def build_config(args: argparse.Namespace) -> Civ6LocalConfig:
    """Construct Civ6LocalConfig from parsed CLI arguments.

    Args:
        args: Namespace from ArgumentParser.parse_args().

    Returns:
        Configured Civ6LocalConfig.
    """
    if args.preset:
        config = Civ6LocalConfig.from_preset(args.preset)
    elif args.config:
        config = Civ6LocalConfig.from_yaml(args.config)
    else:
        config = Civ6LocalConfig.from_env()

    # Apply CLI overrides
    from dataclasses import replace

    overrides: dict[str, Any] = {}

    if args.mode is not None:
        overrides["mode"] = args.mode
    if args.checkpoint_path is not None:
        overrides["checkpoint_path"] = args.checkpoint_path
    if args.board_size is not None:
        overrides["board_size"] = args.board_size
    if args.num_players is not None:
        overrides["num_players"] = args.num_players
    if args.num_simulations is not None:
        overrides["num_simulations"] = args.num_simulations
    if args.device is not None:
        overrides["device"] = args.device
    if args.web_host is not None:
        overrides["web_host"] = args.web_host
    if args.web_port is not None:
        overrides["web_port"] = args.web_port
    if args.bridge_ipc_dir is not None:
        overrides["bridge_ipc_dir"] = args.bridge_ipc_dir
    if args.dry_run:
        overrides["dry_run"] = True

    log_level = args.log_level
    if args.verbose:
        log_level = "debug"
    if log_level is not None:
        overrides["log_level"] = log_level.upper()

    if overrides:
        config = replace(config, **overrides)

    return config


def setup_logging(log_level: str) -> None:
    """Configure root logger.

    Args:
        log_level: One of DEBUG / INFO / WARNING / ERROR.
    """
    numeric = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Determine log level before building config so early errors are visible
    early_log = "DEBUG" if args.verbose else (args.log_level or "INFO").upper()
    setup_logging(early_log)

    try:
        config = build_config(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    # Update logging to final level
    setup_logging(config.log_level)

    # Print non-fatal warnings
    for warning in config.validate():
        logger.warning("Config: %s", warning)

    if config.dry_run:
        print("Configuration valid:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")
        print("\nDry run complete — no game started.")
        return 0

    logger.info("Starting civ6_local — mode=%s", config.mode)

    try:
        if config.mode == "microciv":
            return run_microciv(config)
        elif config.mode == "civ6":
            return run_civ6_bridge(config)
        elif config.mode == "web":
            return run_web(config)
        else:
            logger.error("Unknown mode: %s", config.mode)
            return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
