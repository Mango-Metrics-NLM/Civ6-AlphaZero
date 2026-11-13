"""The local data root: where big artifacts live (LF-E4).

Policy
------
Checkpoints, replay data, venvs, logs and temp files must not live in the
OneDrive-synced repository. ``CIV6AZ_DATA_ROOT`` names a directory outside it.
Unset, every caller keeps its historical relative-path behaviour, so nothing
changes for a run that never opts in.

Design notes
------------
* :func:`resolve_data_root` only *reads* the environment; it never creates or
  validates anything, so it is safe to call from import-light code.
* :func:`assert_safe_data_root` is the validation (pure checks plus read-only
  Windows volume probes); :func:`harden_data_root` is the one function that
  mutates ACLs. They are separate so tests (and dry runs) can validate a path
  without touching its permissions.
* The Windows probes (:func:`_volume_filesystem`, :func:`_is_removable_drive`)
  are module-level so tests replace them; off Windows they are skipped.

Public surface: :func:`resolve_data_root`, :func:`storage_root`,
:func:`assert_safe_data_root`, :func:`harden_data_root`, :func:`validate_run_id`,
:func:`default_run_id`, :func:`default_checkpoint_dir`, :func:`scoped_temp_env`.
"""

from __future__ import annotations

import getpass
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_VAR = "CIV6AZ_DATA_ROOT"
MAX_ROOT_CHARS = 120
MIN_FREE_GB = 50
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
_RUN_ID_RE = re.compile(RUN_ID_PATTERN)
ALLOWED_FILESYSTEMS = frozenset({"NTFS", "REFS"})
_ONEDRIVE_ENV_VARS = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
SYSTEM_SID = "*S-1-5-18"
ADMINISTRATORS_SID = "*S-1-5-32-544"
DRIVE_REMOVABLE = 2
SUBDIRS = ("envs", "runs", "storage", "logs", "tmp")

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class DataRootError(ValueError):
    """The data root (or a run id) is unset where required, or unsafe."""


def _is_windows() -> bool:
    return sys.platform == "win32"


def resolve_data_root(*, required: bool = False) -> Path | None:
    """Return ``CIV6AZ_DATA_ROOT`` as an absolute path, or ``None`` when unset."""
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        if required:
            raise DataRootError(f"{ENV_VAR} is not set; point it at a directory outside the repo")
        return None
    return Path(raw).expanduser().resolve()


def storage_root() -> Path:
    """Root of run-scoped storage: ``<data root>/storage``. Requires the data root."""
    root = resolve_data_root(required=True)
    if root is None:  # unreachable: required=True raises
        raise DataRootError(f"{ENV_VAR} is not set")
    return root / "storage"


def validate_run_id(run_id: str) -> str:
    """Return ``run_id`` if it is a safe path component, else raise."""
    if not _RUN_ID_RE.match(run_id):
        raise DataRootError(f"invalid run id {run_id!r}: must match {RUN_ID_PATTERN}")
    return run_id


def default_run_id(seed: int | None, *, now: datetime | None = None) -> str:
    """``train-<seed or "unseeded">-<UTC %Y%m%dT%H%M%SZ>``."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return validate_run_id(f"train-{'unseeded' if seed is None else seed}-{stamp}")


def default_checkpoint_dir(root: Path, run_id: str) -> Path:
    """``<root>/runs/<run_id>/checkpoints``."""
    return root / "runs" / validate_run_id(run_id) / "checkpoints"


def _is_inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _onedrive_roots() -> list[Path]:
    roots = []
    for name in _ONEDRIVE_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            roots.append(Path(value).resolve())
    return roots


def _kernel32() -> Any:
    import ctypes

    if sys.platform == "win32":
        return ctypes.windll.kernel32
    raise DataRootError("Windows volume probes are unavailable off Windows")


def _volume_filesystem(path: Path) -> str:
    """Filesystem name of the volume holding ``path`` (Windows only)."""
    import ctypes

    anchor = Path(path.anchor or path.resolve().anchor)
    name = ctypes.create_unicode_buffer(261)
    ok = _kernel32().GetVolumeInformationW(str(anchor), None, 0, None, None, None, name, 261)
    if not ok:
        raise DataRootError(f"cannot read volume information for {anchor}")
    return str(name.value)


def _is_removable_drive(path: Path) -> bool:
    """True for a removable drive such as a USB stick (Windows only)."""
    anchor = Path(path.anchor or path.resolve().anchor)
    kind = _kernel32().GetDriveTypeW(str(anchor))
    return int(kind) == DRIVE_REMOVABLE


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def assert_safe_data_root(root: Path, repo_root: Path) -> None:
    """Refuse a data root that would sync, overflow, or sit on a bad volume.

    Raises :class:`DataRootError` when ``root`` is inside the repository, under
    OneDrive (by environment variable or by any component named ``OneDrive*``),
    longer than 120 characters, or (Windows) not NTFS/ReFS or on a removable
    drive. Logs a warning below 50 GB free. Creates nothing.
    """
    resolved = Path(root).resolve()
    repo = Path(repo_root).resolve()
    if _is_inside(resolved, repo):
        raise DataRootError(f"data root {resolved} is inside the repo {repo}")
    for od in _onedrive_roots():
        if _is_inside(resolved, od):
            raise DataRootError(f"data root {resolved} is under OneDrive ({od})")
    if any(part.lower().startswith("onedrive") for part in resolved.parts):
        raise DataRootError(f"data root {resolved} has a OneDrive-named path component")
    if len(str(resolved)) > MAX_ROOT_CHARS:
        raise DataRootError(
            f"data root path is {len(str(resolved))} chars; the limit is {MAX_ROOT_CHARS}"
        )
    if _is_windows():
        fs = _volume_filesystem(resolved)
        if fs.upper() not in ALLOWED_FILESYSTEMS:
            raise DataRootError(f"data root volume is {fs}; NTFS or ReFS is required")
        if _is_removable_drive(resolved):
            raise DataRootError("data root is on a removable (USB) drive")
    else:
        logger.debug("skipping NTFS/removable-drive checks off Windows")
    free_gb = _free_bytes(resolved) / 1024**3
    if free_gb < MIN_FREE_GB:
        logger.warning("data root has only %.1f GB free (< %d GB)", free_gb, MIN_FREE_GB)


def harden_data_root(root: Path, *, runner: Runner = subprocess.run) -> None:
    """Create the data root layout and restrict its ACL to the user, SYSTEM and Administrators.

    Windows: ``icacls <root> /inheritance:r`` then explicit grants by SID (so a
    localised group name cannot break it). Elsewhere: mode ``0o700``. This is the
    only function here that changes permissions; callers decide when to run it.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (root / sub).mkdir(exist_ok=True)
    if not _is_windows():
        root.chmod(0o700)
        return
    domain = os.environ.get("USERDOMAIN", "")
    user = os.environ.get("USERNAME") or getpass.getuser()
    account = f"{domain}\\{user}" if domain else user
    grants = [f"{account}:(OI)(CI)F", f"{SYSTEM_SID}:(OI)(CI)F", f"{ADMINISTRATORS_SID}:(OI)(CI)F"]
    cmd = ["icacls", str(root), "/inheritance:r", "/grant:r", *grants]
    result: Any = runner(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise DataRootError(f"icacls failed ({result.returncode}): {result.stderr.strip()}")
    logger.info("hardened data root %s", root)


def scoped_temp_env(root: Path) -> dict[str, str]:
    """``TMP``/``TEMP`` overrides that keep child-process temp files under the root."""
    tmp = str(Path(root) / "tmp")
    return {"TMP": tmp, "TEMP": tmp}
