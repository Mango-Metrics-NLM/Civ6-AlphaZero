#!/usr/bin/env python3
"""Civ6 AlphaZero Lua Mod Installer.

W2.3b of the v1.7.0 local-gameplay sprint. This script automates installing
the AlphaZeroAI Lua mod into the local Civilization 6 Mods directory on Windows.

Usage:
    python scripts/install_civ6_mod.py
    python scripts/install_civ6_mod.py --mod-dir "C:\\Custom\\Path\\To\\Mods"
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — no hardcoded strings scattered through the code body
# ---------------------------------------------------------------------------

#: Folder under Documents that Steam uses for save games on Windows.
_WIN_GAMES_FOLDER: str = "My Games"

#: Civ6 game subdirectory inside the My Games folder.
_CIV6_FOLDER: str = "Sid Meier's Civilization VI"

#: The mod subfolder inside the Civ6 user-data directory.
_MODS_SUBFOLDER: str = "Mods"

#: Name of the AlphaZeroAI mod directory (must match the .modinfo filename).
MOD_NAME: str = "AlphaZeroAI"

#: Default relative path to the mod source tree inside this repository.
_DEFAULT_SOURCE_DIR: str = "mod/AlphaZeroAI"


def get_default_mod_dir() -> Path:
    """Resolve the default Civ6 Mods directory on Windows.

    Prefers OneDrive-redirected Documents when the Civ6 folder exists there
    (common on modern Windows where Documents is backed by OneDrive).  Falls
    back to the bare ``%USERPROFILE%\\Documents`` path otherwise.

    On non-Windows systems, a warning is logged and the same path layout is
    constructed using ``~`` so the caller can still override it with
    ``--mod-dir``.
    """
    if sys.platform != "win32":
        logger.warning(
            "This script is designed for Windows (Steam). "
            "You may need to provide --mod-dir manually."
        )

    user_profile = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    base = Path(user_profile)

    civ6_rel = Path(_WIN_GAMES_FOLDER) / _CIV6_FOLDER

    # Candidate paths in preference order: OneDrive, then bare Documents
    candidates = [
        base / "OneDrive" / "Documents" / civ6_rel,
        base / "Documents" / civ6_rel,
    ]

    for candidate in candidates:
        if candidate.exists():
            chosen = candidate / _MODS_SUBFOLDER
            logger.info("Auto-detected Civ6 user-data at: %s", candidate)
            return chosen

    # Neither exists yet (Civ6 not installed / first run) — use OneDrive as default
    logger.warning(
        "Civ6 user-data directory not found. Defaulting to OneDrive path. "
        "Pass --mod-dir to override."
    )
    return base / "OneDrive" / "Documents" / civ6_rel / _MODS_SUBFOLDER



def main() -> int:
    parser = argparse.ArgumentParser(description="Install the Civ6 AlphaZero Lua Mod.")
    parser.add_argument(
        "--mod-dir",
        type=str,
        help="Custom path to the Civ6 Mods directory. Defaults to the standard My Games path.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=_DEFAULT_SOURCE_DIR,
        help=f"Path to the mod source directory (default: {_DEFAULT_SOURCE_DIR}).",
    )
    args = parser.parse_args()

    source_path = Path(args.source_dir).resolve()
    if not source_path.exists() or not source_path.is_dir():
        logger.error(f"Mod source directory not found: {source_path}")
        logger.error("Please run this script from the project root.")
        return 1

    if args.mod_dir:
        target_mods_dir = Path(args.mod_dir).resolve()
    else:
        target_mods_dir = get_default_mod_dir()

    if not target_mods_dir.exists():
        logger.warning(f"Mods directory does not exist: {target_mods_dir}")
        try:
            logger.info("Attempting to create Mods directory...")
            target_mods_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create Mods directory: {e}")
            return 1

    target_path = target_mods_dir / MOD_NAME

    logger.info(f"Source: {source_path}")
    logger.info(f"Target: {target_path}")

    if target_path.exists():
        logger.info("Existing mod installation found. Removing...")
        try:
            shutil.rmtree(target_path)
        except OSError as e:
            logger.error(f"Failed to remove existing mod: {e}")
            return 1

    try:
        logger.info("Copying mod files...")
        shutil.copytree(source_path, target_path)
        logger.info("Mod installed successfully.")
        logger.info("Please enable 'AlphaZeroAI' in the Civilization VI Additional Content menu.")
    except OSError as e:
        logger.error(f"Failed to copy mod files: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
