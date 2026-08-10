#!/usr/bin/env python3
"""Civ6 AlphaZero Lua Mod Installer.

W2.3b of the v1.7.0 local-gameplay sprint. This script automates installing
the AlphaZeroAI Lua mod into the local Civilization 6 Mods directory on Windows.

When the Mods directory is OneDrive-synced, OneDrive can hold locks that make
remove-then-copy fail with ``WinError 5``; in that case the installer falls
back to an SHA256-verified in-place overwrite.

Usage:
    python scripts/install_civ6_mod.py
    python scripts/install_civ6_mod.py --mod-dir "C:\\Custom\\Path\\To\\Mods"
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

# Ensure project root is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpha_zero.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

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

#: Chunk size for streaming SHA256 computation.
_HASH_CHUNK_BYTES: int = 65536


class InstallVerificationError(RuntimeError):
    """Raised when an installed file's SHA256 does not match its source."""


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


def _sha256(path: Path) -> str:
    """Stream a file's SHA256 hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_files(root: Path) -> set[Path]:
    """All files under ``root`` as paths relative to it."""
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def overwrite_in_place(source: Path, target: Path) -> None:
    """Overwrite ``target`` with ``source`` without deleting the tree first.

    OneDrive keeps handles on synced directories, so removing the whole
    install can fail with ``WinError 5`` while every file inside remains
    replaceable. Copies each source file over its destination, removes files
    that exist only in the destination, then verifies every copy against the
    source by SHA256.

    Raises:
        OSError: If a file cannot be replaced or a stale file removed.
        InstallVerificationError: If a copied file fails SHA256 verification.
    """
    source_files = _relative_files(source)

    for relative in sorted(source_files):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    logger.info("Replaced %d file(s) in place.", len(source_files))

    stale_files = _relative_files(target) - source_files
    for relative in sorted(stale_files):
        logger.info("Removing stale file: %s", relative)
        (target / relative).unlink()

    # Stale directories are pruned best-effort: an empty leftover folder is
    # harmless, and OneDrive may still hold a lock on the directory itself.
    for directory in sorted((path for path in target.rglob("*") if path.is_dir()), reverse=True):
        if not (source / directory.relative_to(target)).is_dir():
            try:
                directory.rmdir()
            except OSError:
                logger.warning("Could not remove stale directory: %s", directory)

    mismatched = [
        relative
        for relative in sorted(source_files)
        if _sha256(source / relative) != _sha256(target / relative)
    ]
    if mismatched:
        raise InstallVerificationError(
            "SHA256 mismatch after in-place overwrite: "
            + ", ".join(str(path) for path in mismatched)
        )
    logger.info("Verified %d file(s) by SHA256.", len(source_files))


def install_mod(source_path: Path, target_path: Path) -> int:
    """Install the mod tree at ``target_path``, returning a process exit code.

    Prefers the original remove-then-copy strategy. When the existing
    install cannot be removed (OneDrive-synced Mods directories hold locks
    that make ``shutil.rmtree`` fail even though the files are writable),
    falls back to an SHA256-verified in-place overwrite.
    """
    logger.info("Source: %s", source_path)
    logger.info("Target: %s", target_path)

    if target_path.exists():
        logger.info("Existing mod installation found. Removing...")
        try:
            shutil.rmtree(target_path)
        except OSError as err:
            logger.warning(
                "Could not remove existing mod (%s); falling back to in-place overwrite.",
                err,
            )
            try:
                overwrite_in_place(source_path, target_path)
            except (OSError, InstallVerificationError) as fallback_err:
                logger.error("In-place overwrite failed: %s", fallback_err)
                return 1
            _log_success()
            return 0

    try:
        logger.info("Copying mod files...")
        shutil.copytree(source_path, target_path)
    except OSError as err:
        logger.error("Failed to copy mod files: %s", err)
        return 1

    _log_success()
    return 0


def _log_success() -> None:
    logger.info("Mod installed successfully.")
    logger.info("Please enable 'AlphaZeroAI' in the Civilization VI Additional Content menu.")


def main() -> int:
    setup_logging()
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

    return install_mod(source_path, target_path)


if __name__ == "__main__":
    sys.exit(main())
