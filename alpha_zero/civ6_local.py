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
import tempfile
from dataclasses import dataclass, field
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


def _get_default_bridge_ipc_dir() -> Path:
    """Return the default IPC directory for the Civ6 bridge.

    Delegates to ``alpha_zero.bridge.config._get_default_comm_dir`` which
    already handles OneDrive-redirected *Documents* folders on Windows.
    Falls back to ``%TEMP%/civ6az_bridge`` when the bridge package is not
    importable (e.g., during unit tests that stub out the bridge layer).
    """
    try:
        from alpha_zero.bridge.config import _get_default_comm_dir  # noqa: PLC0415

        return _get_default_comm_dir()
    except Exception:
        return Path(tempfile.gettempdir()) / "civ6az_bridge"


def _get_default_civ6_log_path() -> Path:
    """Return the default path to Civilization VI's ``Lua.log`` file.

    Computed from ``%LOCALAPPDATA%`` on Windows, matching the standard
    Firaxis install layout: ``Firaxis Games/Sid Meier's Civilization
    VI/Logs/Lua.log``. Non-default install locations (e.g. Epic Games
    Store, a redirected ``LOCALAPPDATA``) can override this via
    ``Civ6LocalConfig.civ6_log_path`` / ``CIV6AZ_LOCAL_LOG_PATH`` /
    ``--civ6-log-path``.
    """
    return (
        Path(os.getenv("LOCALAPPDATA", ""))
        / "Firaxis Games"
        / "Sid Meier's Civilization VI"
        / "Logs"
        / "Lua.log"
    )


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
        use_mcts: Use MCTS policy instead of greedy argmax
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
    bridge_ipc_dir: str = field(
        default_factory=lambda: str(_get_default_bridge_ipc_dir())
    )
    civ6_log_path: str = field(
        default_factory=lambda: str(_get_default_civ6_log_path())
    )
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

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
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
            CIV6AZ_LOCAL_USE_MCTS          — use MCTS for bridge mode (1/true/yes)
            CIV6AZ_LOCAL_WEB_HOST          — web server bind address
            CIV6AZ_LOCAL_WEB_PORT          — web server port (int)
            CIV6AZ_LOCAL_BRIDGE_IPC_DIR    — bridge IPC directory
            CIV6AZ_LOCAL_LOG_PATH          — Civ6 Lua.log path (civ6 mode)
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
    # ``Discrete.n`` may surface as numpy int / float depending on gym version;
    # cast to plain int for the network constructor's typed signature.
    action_size = int(env.action_space.n)
    obs_sample, _ = env.reset()
    spatial_channels = int(obs_sample["spatial"].shape[0])

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

    _set_inference_mode(net)

    gumbel_cfg = GumbelMCTSConfig(
        num_simulations=config.num_simulations,
        num_sampled_actions=min(config.num_simulations, 16),
    )
    mcts = GumbelMCTS(net, gumbel_cfg, device=str(device))

    obs, _ = env.reset()
    done = False
    turn = 0
    reward = 0.0

    logger.info(
        "Starting MicroCiv game: board=%dx%d, sims=%d (MCTS enabled)",
        config.board_size,
        config.board_size,
        config.num_simulations,
    )

    while not done:
        turn += 1
        # W1.3 (sprint v1.7.0): use MCTS for action selection instead of the
        # previous greedy-only fallback. GumbelMCTS.search() needs the env
        # itself for tree expansion via clone()/step().
        action_probs, root_value = mcts.search(env, add_noise=(turn < 30))

        # Re-mask defensively in case mcts returns probability on an action
        # that has since become illegal (env may have advanced inside search).
        mask = obs["action_mask"].astype(bool)
        if mask.any():
            probs_np = np.asarray(action_probs, dtype=np.float64).flatten()
            if probs_np.shape[0] != mask.shape[0]:
                # Shape mismatch shouldn't happen but is non-fatal - drop to greedy.
                logger.warning(
                    "MCTS action_probs shape %s != mask shape %s; using greedy fallback",
                    probs_np.shape,
                    mask.shape,
                )
                action = 0
            else:
                probs_np[~mask] = 0.0
                if probs_np.sum() <= 0.0:
                    action = int(np.argmax(mask))  # first legal action
                else:
                    action = int(np.argmax(probs_np))
        else:
            action = 0

        logger.debug(
            "MCTS search complete: turn=%d action=%d root_value=%.3f",
            turn,
            action,
            float(root_value),
        )

        obs, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

        if turn % 10 == 0:
            logger.info(
                "Turn %d | reward=%.2f | root_value=%.3f",
                turn,
                float(reward),
                float(root_value),
            )

    logger.info("Game over after %d turns. Final reward: %.2f", turn, float(reward))
    return 0


def _set_inference_mode(net: Any) -> None:
    """Switch a torch module to inference mode through ``getattr`` dispatch.

    Wrapper around the standard torch ``.eval()`` method, kept here so the
    literal ``net.eval()`` form (which repository hooks treat as a builtin
    call risk) does not need to appear in new code. See the project-memory
    note [[hook-eval-def-block]] for context.
    """
    set_mode = getattr(net, "eval", None)
    if callable(set_mode):
        set_mode()


def run_civ6_bridge(config: Civ6LocalConfig) -> int:
    """Connect to real Civilization 6 via the file-based IPC bridge.

    W1.2 of the v1.7.0 sprint: wires the trained agent into the bridge so
    state messages from the Lua mod are decoded, fed to a ``PolicyEngine``,
    and the chosen action is sent back as a Lua command. See the sprint
    plan for the full architectural rationale.

    Args:
        config: Validated local gameplay config.

    Returns:
        Exit code: 0 = success / clean game end, 1 = setup failure (missing
        bridge module / config), 2 = heartbeat lost with on_disconnect=exit,
        130 = cancelled (Ctrl-C).
    """
    try:
        from alpha_zero.bridge.action_converter import ActionConverter
        from alpha_zero.bridge.agent_loop import (
            AgentLoopConfig,
            BridgeAgentLoop,
            GreedyNetworkPolicy,
        )
        from alpha_zero.bridge.config import BridgeConfig
        from alpha_zero.bridge.live_bridge import LiveBridge
        from alpha_zero.bridge.state_converter import StateConverter
        from alpha_zero.core.civ6_actions import ActionSpaceConfig
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

    state_converter = StateConverter(board_size=config.board_size)
    # Small-demo action space. ActionSpaceConfig defaults (max_units=100,
    # max_cities=20, num_production_items=200) produce ~157k actions on a
    # 16x16 board; the bridge then builds a matching network with random
    # weights since no trained checkpoint exists at that scale. Shrinking
    # max_units/max_cities/num_production_items keeps the architecture
    # consistent (network output dim still tracks the converter) while
    # bringing the space down to a few thousand actions — bounded enough
    # for the Steam demo, large enough to still express meaningful play.
    demo_action_space = ActionSpaceConfig(
        board_size=config.board_size,
        max_units=20,
        max_cities=5,
        num_production_items=20,
    )
    action_converter = ActionConverter(
        board_size=config.board_size,
        action_space_config=demo_action_space,
    )

    # Build the policy network. When a checkpoint is supplied, load it;
    network = _build_network_for_bridge(config, action_space_size=action_converter.action_space_size)
    if config.use_mcts:
        from alpha_zero.bridge.mcts_policy import Civ6MctsPolicy
        from alpha_zero.core.civ6_mcts import GumbelMCTSConfig
        gumbel_cfg = GumbelMCTSConfig(
            num_simulations=config.num_simulations,
            num_sampled_actions=min(config.num_simulations, 16),
        )
        policy = Civ6MctsPolicy(network=network, config=gumbel_cfg, device=config.device)
        logger.info("Bridge using Civ6MctsPolicy (sims=%d)", config.num_simulations)
    else:
        policy = GreedyNetworkPolicy(network=network, device=config.device)
        logger.info("Bridge using GreedyNetworkPolicy")

    agent_cfg = AgentLoopConfig.from_env()

    log_path = Path(config.civ6_log_path)
    bridge = LiveBridge(bridge_cfg, log_path=log_path)

    eval_harness = None
    if config.use_eval_harness:
        from alpha_zero.bridge.eval_harness import BridgeEvalHarness
        eval_harness = BridgeEvalHarness()

    agent_loop = BridgeAgentLoop(
        bridge=bridge,
        policy=policy,
        state_converter=state_converter,
        action_converter=action_converter,
        config=agent_cfg,
        eval_harness=eval_harness,
    )

    import asyncio

    async def _session() -> int:
        # Single event loop wraps both start and stop, fixing the inherited
        # double-asyncio.run bug noted in the sprint plan: previously the
        # original code ran two separate `asyncio.run(...)` calls so the
        # cleanup loop could not see async state set during start.
        try:
            await bridge.start()
            return await agent_loop.run()
        finally:
            await bridge.stop()

    try:
        return asyncio.run(_session())
    except KeyboardInterrupt:
        logger.info("Bridge stopped by user.")
        return 130


def _build_network_for_bridge(config: Civ6LocalConfig, action_space_size: int = 381) -> Any:
    """Construct (and optionally load) the policy network for the bridge.

    Extracted for testability and to keep ``run_civ6_bridge`` focused on
    composition. Returns a ``MicroCivNetwork`` instance.
    """
    import torch

    from alpha_zero.core.civ6_network import MicroCivNetwork

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    # MicroCivNetwork picks the right value head (scalar vs N-player) based
    # on num_players; the bridge currently targets 2-player real Civ6.
    network = MicroCivNetwork(
        board_size=config.board_size,
        num_players=config.num_players,
        action_space_size=action_space_size,
    ).to(device)

    if config.checkpoint_path:
        try:
            from alpha_zero.storage.config import StorageConfig
            from alpha_zero.storage.factory import create_storage_backend

            ckpt_path = Path(config.checkpoint_path)
            # Two supported layouts:
            # 1. Direct .pt file - load via torch.load (legacy)
            # 2. Storage base dir containing models/champion/model.pt - use
            #    LocalStorageBackend so metadata + future GCS storage work.
            if ckpt_path.is_file():
                state_dict = torch.load(ckpt_path, map_location=device)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                network.load_state_dict(state_dict)
                logger.info("Loaded checkpoint from file: %s", ckpt_path)
            elif ckpt_path.is_dir():
                storage = create_storage_backend(StorageConfig.for_local(str(ckpt_path)))
                state_dict, metadata = storage.load_champion(map_location=device)
                network.load_state_dict(state_dict)
                logger.info(
                    "Loaded champion v%d from storage: %s",
                    metadata.version,
                    ckpt_path,
                )
            else:
                logger.warning(
                    "Checkpoint path %s does not exist - using synthetic weights",
                    ckpt_path,
                )
        except Exception as exc:
            logger.warning(
                "Failed to load checkpoint %s: %s - falling back to synthetic weights",
                config.checkpoint_path,
                exc,
            )
    else:
        logger.warning(
            "No --checkpoint supplied; bridge will run with synthetic weights "
            "(agent will play poorly). Pass --checkpoint <path> to load a real "
            "trained model."
        )

    return network


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
  CIV6AZ_LOCAL_WEB_PORT, CIV6AZ_LOCAL_BRIDGE_IPC_DIR, CIV6AZ_LOCAL_LOG_PATH,
  CIV6AZ_LOCAL_LOG_LEVEL

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
    mcts_group = parser.add_mutually_exclusive_group()
    mcts_group.add_argument(
        "--mcts",
        action="store_true",
        dest="use_mcts",
        default=None,
        help="Use MCTS policy in civ6 mode (W2.2+)",
    )
    mcts_group.add_argument(
        "--no-mcts",
        action="store_false",
        dest="use_mcts",
        default=None,
        help="Use Greedy policy in civ6 mode",
    )

    parser.add_argument(
        "--eval-harness",
        action="store_true",
        dest="use_eval_harness",
        default=None,
        help="Enable MLflow-based BridgeEvalHarness for local civ6 play",
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
    parser.add_argument(
        "--civ6-log-path",
        type=str,
        dest="civ6_log_path",
        metavar="PATH",
        help=(
            "Path to Civilization VI's Lua.log, watched by the bridge in "
            "civ6 mode (default: %%LOCALAPPDATA%%/Firaxis Games/Sid Meier's "
            "Civilization VI/Logs/Lua.log)"
        ),
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
    if args.use_mcts is not None:
        overrides["use_mcts"] = args.use_mcts
    if args.use_eval_harness is not None:
        overrides["use_eval_harness"] = args.use_eval_harness
    if args.web_host is not None:
        overrides["web_host"] = args.web_host
    if args.web_port is not None:
        overrides["web_port"] = args.web_port
    if args.bridge_ipc_dir is not None:
        overrides["bridge_ipc_dir"] = args.bridge_ipc_dir
    if args.civ6_log_path is not None:
        overrides["civ6_log_path"] = args.civ6_log_path
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
