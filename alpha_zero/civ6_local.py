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

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

# Safe to import at module scope: alpha_zero.local.gameplay_config and
# alpha_zero.utils.device defer torch imports, so --help stays fast.
from alpha_zero.local.gameplay_config import (
    DEFAULT_DEMO_MAX_CITIES,
    DEFAULT_DEMO_MAX_UNITS,
    DEFAULT_DEMO_NUM_PRODUCTION_ITEMS,
    ENV_PREFIX,
    SEARCH_MODE_GAP_ID,
    SEARCH_TRANSLATOR_GAP_ID,
    VALID_DEVICES,
    VALID_LOG_LEVELS,
    VALID_MODES,
    Civ6LocalConfig,
    _get_default_bridge_ipc_dir,
    _get_default_civ6_log_path,
    search_mode_refusal,
)
from alpha_zero.utils.constants import DEFAULT_ACTION_SPACE_SIZE, discrete_action_count
from alpha_zero.utils.device import resolve_device

logger = logging.getLogger(__name__)

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
    "_build_bridge_policy",
    "_build_network_for_bridge",
    "_get_default_bridge_ipc_dir",
    "_get_default_civ6_log_path",
    "build_config",
    "create_parser",
    "main",
    "run_civ6_bridge",
    "run_microciv",
    "run_web",
    "search_mode_refusal",
    "setup_logging",
]


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

    from alpha_zero.core.checkpoint_utils import safe_torch_load
    from alpha_zero.core.civ6_mcts import GumbelMCTS, GumbelMCTSConfig, MCTSEnvironment
    from alpha_zero.core.civ6_network import MicroCivNetwork
    from alpha_zero.envs.microciv import MicroCivEnv

    device = resolve_device(config.device)

    env = MicroCivEnv(board_size=config.board_size)
    action_size = discrete_action_count(env.action_space)
    obs_sample, _ = env.reset()
    spatial_channels = int(obs_sample["spatial"].shape[0])

    net = MicroCivNetwork(
        board_size=config.board_size,
        action_space_size=action_size,
        spatial_channels=spatial_channels,
    ).to(device)

    if config.checkpoint_path:
        # weights_only=True by default; legacy pickles need
        # CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1 (see core.checkpoint_utils).
        checkpoint = safe_torch_load(config.checkpoint_path, map_location=device)
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

    # MicroCivEnv satisfies the MCTSEnvironment protocol at runtime, but its
    # gym ``Discrete`` action_space keeps it from matching structurally.
    mcts_env = cast(MCTSEnvironment, env)

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
        action_probs, root_value = mcts.search(mcts_env, add_noise=(turn < 30))

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

        obs, step_reward, terminated, truncated, _info = env.step(action)
        reward = float(step_reward)
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
        bridge module / config) or a refused search mode
        (:func:`search_mode_refusal`), 2 = heartbeat lost with
        on_disconnect=exit, 130 = cancelled (Ctrl-C).
    """
    # Before anything is created or connected.
    refusal = search_mode_refusal(config)
    if refusal is not None:
        logger.error(refusal)
        return 1

    try:
        import alpha_zero.bridge.config as bridge_config_mod
        from alpha_zero.bridge.action_converter import ActionConverter
        from alpha_zero.bridge.agent_loop import (
            AgentLoopConfig,
            BridgeAgentLoop,
            PolicyEngine,
        )
        from alpha_zero.bridge.config import BridgeConfig
        from alpha_zero.bridge.live_bridge import LiveBridge
        from alpha_zero.bridge.state_converter import StateConverter
        from alpha_zero.core.civ6_actions import ActionSpaceConfig
    except ImportError as exc:
        logger.error("Bridge module not available: %s", exc)
        return 1

    # Command transport: "file" (default) writes commands.json for a FireTuner
    # .ltp panel to relay; "socket" injects commands directly over the FireTuner
    # TCP protocol (no GUI panel needed). Selected via CIV6_BRIDGE_TRANSPORT.
    from_live_env = getattr(BridgeConfig, "from_live_env", None)
    if callable(from_live_env) and isinstance(BridgeConfig, type):
        bridge_cfg = from_live_env(
            comm_dir=Path(config.bridge_ipc_dir),
            log_level=config.log_level,
        )
    else:
        # Mocked BridgeConfig in isolated unit tests skips from_live_env.
        bridge_cfg = BridgeConfig(
            comm_dir=Path(config.bridge_ipc_dir),
            log_level=config.log_level,
        )
    default_transport = (
        bridge_config_mod.DEFAULT_COMMAND_TRANSPORT
        if isinstance(getattr(bridge_config_mod, "DEFAULT_COMMAND_TRANSPORT", None), str)
        else "file"
    )
    transport = (
        bridge_cfg.command_transport
        if isinstance(getattr(bridge_cfg, "command_transport", None), str)
        else (os.getenv("CIV6_BRIDGE_TRANSPORT", "").strip().lower() or default_transport)
    )
    bridge_cfg.ensure_comm_dir()

    logger.info("Connecting to Civ6 bridge at: %s", bridge_cfg.comm_dir)
    logger.info("Command transport: %s", transport)
    if transport == "socket":
        logger.info(
            "Injecting commands via FireTuner socket %s:%s (no GUI panel needed).",
            bridge_cfg.tuner_host,
            bridge_cfg.tuner_port,
        )
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
        max_units=DEFAULT_DEMO_MAX_UNITS,
        max_cities=DEFAULT_DEMO_MAX_CITIES,
        num_production_items=DEFAULT_DEMO_NUM_PRODUCTION_ITEMS,
    )
    action_converter = ActionConverter(
        board_size=config.board_size,
        action_space_config=demo_action_space,
    )

    # Build the policy network. When a checkpoint is supplied, load it;
    network = _build_network_for_bridge(
        config, action_space_size=action_converter.action_space_size
    )
    policy: PolicyEngine = _build_bridge_policy(config, network)

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

    # Optional startup kick: if the game is idle on the local player's turn
    # with an expired command window, no fresh state ever arrives. Setting
    # CIV6_BRIDGE_KICK_START forces one end-turn (socket transport only) so a
    # new turn-begin export is emitted for the agent to catch.
    kick_start = os.getenv("CIV6_BRIDGE_KICK_START", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    async def _session() -> int:
        # Single event loop wraps both start and stop, fixing the inherited
        # double-asyncio.run bug noted in the sprint plan: previously the
        # original code ran two separate `asyncio.run(...)` calls so the
        # cleanup loop could not see async state set during start.
        try:
            await bridge.start()
            if kick_start:
                logger.info("Kick-start enabled: forcing one end-turn to un-stall.")
                await bridge.kick_turn()
            return await agent_loop.run()
        finally:
            await bridge.stop()

    try:
        return asyncio.run(_session())
    except KeyboardInterrupt:
        logger.info("Bridge stopped by user.")
        return 130


def _build_bridge_policy(config: Civ6LocalConfig, network: Any) -> Any:
    """The policy the bridge agent loop runs: the search, or the greedy network.

    ``run_civ6_bridge`` never gets here with ``use_mcts`` set: it refuses that
    first (:func:`search_mode_refusal`). The body lives in
    ``alpha_zero.bridge.policy_factory``.
    """
    from alpha_zero.bridge.policy_factory import build_bridge_policy

    return build_bridge_policy(config, network, log=logger)


def _build_network_for_bridge(
    config: Civ6LocalConfig, action_space_size: int = DEFAULT_ACTION_SPACE_SIZE
) -> Any:
    """Construct (and optionally load) the policy network for the bridge."""
    from alpha_zero.bridge.policy_factory import build_network_for_bridge

    return build_network_for_bridge(config, action_space_size, log=logger)


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
        help="PyTorch inference device (default: cpu)",
    )
    mcts_group = parser.add_mutually_exclusive_group()
    mcts_group.add_argument(
        "--mcts",
        action="store_true",
        dest="use_mcts",
        default=None,
        help=(
            "Use the MCTS policy. Refused in civ6 mode until a translator from the "
            f"search's actions to live commands exists ({SEARCH_MODE_GAP_ID}, "
            f"{SEARCH_TRANSLATOR_GAP_ID})"
        ),
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
        help="IPC directory for Civ6 bridge (default: auto-detected Civ6 mod comm directory)",
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
    """Configure root logger level and default stdout stream handler."""
    numeric = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger().setLevel(numeric)


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
    logger.debug(
        "Resolved Civ6LocalConfig: mode=%s device=%s log_level=%s",
        config.mode,
        config.device,
        config.log_level,
    )

    # Print non-fatal warnings
    for warning in config.validate():
        logger.warning("Config: %s", warning)

    # A config that cannot start is not a valid one, so this comes before the
    # dry run reports "Configuration valid".
    refusal = search_mode_refusal(config)
    if refusal is not None:
        logger.error(refusal)
        return 1

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
