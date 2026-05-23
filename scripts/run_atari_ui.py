"""Launcher script for Atari UI that sets up sys.path correctly."""

import os
import sys


def _setup_and_run() -> None:
    """Set up sys.path and launch the Atari UI server."""
    # Ensure project root is on sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    os.chdir(project_root)

    from demos.atari.ui.__main__ import main

    main()


if __name__ == "__main__":
    _setup_and_run()
