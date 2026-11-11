"""
Shared utility for loading Civ6-AlphaZero environment variables.

Loads CIV6AZ_* variables from a .env file (if present) or system environment,
with clear warnings when required variables are missing.

Usage:
    from alpha_zero.utils.env_loader import load_civ6az_env
    load_civ6az_env()  # call early in any script
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Required GCP variables and their descriptions
_REQUIRED_VARS: dict[str, str] = {
    "CIV6AZ_VERTEX_PROJECT": "GCP project ID (e.g. my-gcp-project)",
    "CIV6AZ_GCS_BUCKET": "GCS bucket name (e.g. my-bucket)",
}

# Optional variables with their defaults
_OPTIONAL_VARS: dict[str, str] = {
    "CIV6AZ_VERTEX_REGION": "us-central1",
    "CIV6AZ_BOARD_SIZE": "8",
    "CIV6AZ_ACTION_SPACE_SIZE": "376",
    "CIV6AZ_STORAGE_BACKEND": "gcs",
}


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict of key=value pairs.

    Ignores blank lines and lines starting with ``#``.

    Args:
        path: Absolute path to the .env file.

    Returns:
        Dict of variable name → value (values stripped of quotes).
    """
    result: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                logger.debug(".env line %d skipped (no '='): %s", line_no, line)
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                result[key] = value
    return result


def load_civ6az_env(
    dotenv_path: str | Path | None = None,
    *,
    override: bool = False,
    warn_missing: bool = True,
) -> dict[str, str]:
    """Load CIV6AZ_* environment variables from a .env file and/or os.environ.

    Search order for the .env file (first found wins):
    1. ``dotenv_path`` argument (if given)
    2. ``.env`` in the current working directory
    3. ``.env`` in the project root (two levels up from this file)

    Args:
        dotenv_path: Explicit path to a .env file.  ``None`` = auto-detect.
        override: If ``True``, values from the .env file overwrite existing
            environment variables.  Default is ``False`` (existing env wins).
        warn_missing: If ``True``, log a ``WARNING`` for each required variable
            that is still unset after loading.

    Returns:
        Dict of all environment variable values that were loaded or already set
        (keys are variable names; values are their resolved strings).
    """
    # --- Locate .env file ---
    candidate_paths: list[Path] = []
    skip_dotenv = os.environ.get("CIV6AZ_SKIP_DOTENV") == "1"
    if dotenv_path is not None:
        candidate_paths.append(Path(dotenv_path).resolve())

    cwd_env = Path.cwd() / ".env"
    if cwd_env not in candidate_paths:
        candidate_paths.append(cwd_env)

    root_env = Path(__file__).resolve().parent.parent.parent / ".env"
    if root_env not in candidate_paths:
        candidate_paths.append(root_env)

    loaded_from: Path | None = None
    for candidate in [] if skip_dotenv else candidate_paths:
        if candidate.is_file():
            loaded_from = candidate
            break

    if loaded_from is not None:
        try:
            file_vars = _parse_dotenv(loaded_from)
            loaded_count = 0
            for key, value in file_vars.items():
                if override or key not in os.environ:
                    os.environ[key] = value
                    loaded_count += 1
            logger.debug(
                "Loaded %d variable(s) from %s (override=%s)",
                loaded_count,
                loaded_from,
                override,
            )
        except OSError as exc:
            logger.warning("Failed to read .env file at %s: %s", loaded_from, exc)
    else:
        logger.debug("No .env file found (searched: %s)", [str(p) for p in candidate_paths])

    # --- Warn about missing required variables ---
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for var, description in _REQUIRED_VARS.items():
        value = os.environ.get(var, "")
        resolved[var] = value
        if not value and warn_missing:
            missing.append(f"  {var}: {description}")

    if missing and warn_missing:
        logger.warning(
            "Missing required CIV6AZ environment variables:\n%s\n"
            "Set them in your shell or create a .env file at the project root.",
            "\n".join(missing),
        )

    # Collect optional vars too
    for var, default in _OPTIONAL_VARS.items():
        resolved[var] = os.environ.get(var, default)

    return resolved


def require_civ6az_env(dotenv_path: str | Path | None = None) -> dict[str, str]:
    """Like :func:`load_civ6az_env` but raises ``RuntimeError`` if any required var is missing.

    Args:
        dotenv_path: Explicit path to a .env file.

    Returns:
        Dict of resolved environment variable values.

    Raises:
        RuntimeError: If one or more required variables are unset after loading.
    """
    resolved = load_civ6az_env(dotenv_path=dotenv_path, warn_missing=False)
    missing = [var for var in _REQUIRED_VARS if not resolved.get(var)]
    if missing:
        descriptions = "\n".join(f"  {var}: {_REQUIRED_VARS[var]}" for var in missing)
        raise RuntimeError(
            f"Missing required CIV6AZ environment variables:\n{descriptions}\n"
            "Set them in your shell or create a .env file at the project root."
        )
    return resolved
