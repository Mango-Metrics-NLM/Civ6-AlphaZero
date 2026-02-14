"""CLI entry point for the web interface.

Usage:
    python -m alpha_zero.web                          # Start with defaults
    python -m alpha_zero.web --preset dev             # Dev preset
    python -m alpha_zero.web --config configs/web.yaml
    python -m alpha_zero.web --host 127.0.0.1 --port 3000
    python -m alpha_zero.web --dry-run                # Validate config only
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpha_zero.web.config import WebConfig

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python -m alpha_zero.web",
        description="Civ6 AlphaZero - Web-based Human vs AI Interface",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["dev", "production"],
        default=None,
        help="Use a named configuration preset",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server bind address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "heuristic", "neural"],
        default=None,
        help="Default opponent type",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=None,
        help="Default board size",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Enable debug mode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit",
    )

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> WebConfig:
    """Build WebConfig from CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        WebConfig instance.
    """
    from alpha_zero.web.config import WebConfig

    # Start from preset, config file, or defaults
    if args.preset:
        config = WebConfig.from_preset(args.preset)
    elif args.config:
        config = WebConfig.from_yaml(args.config)
    else:
        config = WebConfig.from_env()

    # Apply CLI overrides
    overrides: dict[str, object] = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.checkpoint is not None:
        overrides["checkpoint_path"] = args.checkpoint
    if args.opponent is not None:
        overrides["default_opponent"] = args.opponent
    if args.board_size is not None:
        overrides["board_size"] = args.board_size
    if args.debug is True:
        overrides["debug"] = True
        overrides["log_level"] = "debug"

    if overrides:
        config_dict = config.to_dict()
        config_dict.update(overrides)
        config = WebConfig(**config_dict)

    return config


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the web interface CLI.

    Args:
        argv: Command line arguments.

    Returns:
        Exit code.
    """
    args = parse_args(argv)

    try:
        config = build_config(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("Configuration valid:")
        from alpha_zero.web.config import WebConfig

        assert isinstance(config, WebConfig)
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")
        print("\nDry run complete. No server started.")
        return 0

    # Configure logging
    log_level = getattr(config, "log_level", "info").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        import uvicorn

        from alpha_zero.web.app import create_app

        app = create_app(config)

        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level=config.log_level.lower(),
            reload=config.debug,
        )
    except ImportError as e:
        print(
            f"Missing dependency: {e}\n"
            "Install web dependencies: pip install -e '.[web]'",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
