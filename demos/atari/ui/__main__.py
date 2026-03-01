"""CLI entry point for the Atari AlphaZero UI.

Usage:
    python -m demos.atari.ui [--port 8080] [--game Pong] [--model path/to/checkpoint.pt]
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    """Parse arguments and launch the UI server."""
    parser = argparse.ArgumentParser(
        description="Atari AlphaZero Web UI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--game", default="Pong", help="Atari game name")
    parser.add_argument("--model", default=None, help="Path to model checkpoint")
    parser.add_argument("--device", default="", help="PyTorch device (auto-detect if empty)")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations per AI move")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        from demos.atari.ui import _UI_AVAILABLE

        if not _UI_AVAILABLE:
            print(
                "ERROR: UI dependencies not installed.\n"
                "Install with: pip install -e '.[atari_ui]'",
                file=sys.stderr,
            )
            sys.exit(1)
    except ImportError:
        print(
            "ERROR: UI module not available.\n" "Install with: pip install -e '.[atari_ui]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from demos.atari.ui.app import create_app
    from demos.atari.ui.config import AtariUIConfig

    config = AtariUIConfig(
        host=args.host,
        port=args.port,
        game=args.game,
        model_path=args.model,
        device=args.device,
        mcts_simulations=args.mcts_sims,
        debug=args.debug,
    )

    app = create_app(config)

    import uvicorn

    print("\n  Atari AlphaZero UI")
    print(f"  Game:   {config.game}")
    print(f"  Device: {config.device}")
    print(f"  URL:    http://{config.host}:{config.port}\n")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()
