"""Write a file so that a killed process never leaves half of it behind.

A run that is interrupted mid-write must not leave a torn checkpoint or a torn
``result.json``: the re-run rule (``docs/EVALUATION.md`` section 7.1, "After a
VOID") compares what an interrupted attempt left with what its re-run produced,
and a file that cannot be read cannot be shown to reproduce.

Both helpers write to a sibling temporary file in the same directory (the same
filesystem) and then give it the target's name in one step:

* :func:`write_atomically` writes to ``<name>.tmp`` and replaces the target
  (``os.replace``), assuming one writer per target (e.g. a claimed run directory);
* :func:`create_atomically` writes to a unique sibling temporary file
  (``tempfile.mkstemp``) and refuses an existing target (never calling
  ``os.replace``), for files that must never be overwritten even under concurrent
  writers.

The bytes written are exactly what the caller's writer produces; nothing here
changes a file's content. Needs only the standard library.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

#: Appended to the target's name for the file being written.
TEMPORARY_SUFFIX = ".tmp"
DEFAULT_ENCODING = "utf-8"

Writer = Callable[[Path], None]


def temporary_path(path: str | os.PathLike[str]) -> Path:
    """Where the content of ``path`` is written before it is published."""
    target = Path(path)
    return target.with_name(target.name + TEMPORARY_SUFFIX)


def _discard(tmp: Path) -> None:
    try:
        tmp.unlink(missing_ok=True)
    except OSError as exc:  # a leftover .tmp is harmless; never mask the real error
        logger.warning("could not remove %s: %s", tmp, exc)


def _is_windows() -> bool:
    return os.name == "nt"


def _create_unique_temporary_file(target: Path) -> Path:
    fd, raw_path = tempfile.mkstemp(
        prefix=f"{target.name}.",
        suffix=TEMPORARY_SUFFIX,
        dir=target.parent,
    )
    tmp = Path(raw_path)
    try:
        os.close(fd)
    except BaseException:
        _discard(tmp)
        raise
    return tmp


def _publish_exclusive(tmp: Path, target: Path) -> None:
    try:
        os.link(tmp, target)  # atomic, and fails if the target appeared meanwhile
    except FileExistsError:
        raise
    except OSError as exc:  # a filesystem without hard links (exFAT, some shares)
        if _is_windows():
            logger.debug("no hard link for %s (%s); renaming instead", target, exc)
            os.rename(tmp, target)
            return
        logger.warning("atomic exclusive creation requires hard links on %s (%s)", target, exc)
        raise


def write_atomically(path: str | os.PathLike[str], write: Writer) -> None:
    """Call ``write(tmp)`` and then rename ``tmp`` onto ``path``, replacing it.

    If ``write`` raises, or the process dies inside it, ``path`` is left exactly
    as it was (absent, or its previous content). A leftover ``.tmp`` from an
    earlier kill is overwritten.
    """
    target = Path(path)
    tmp = temporary_path(target)
    try:
        write(tmp)
        os.replace(tmp, target)
    except BaseException:
        _discard(tmp)
        raise


def create_atomically(path: str | os.PathLike[str], write: Writer) -> None:
    """Like :func:`write_atomically`, but refuse to replace an existing ``path``.

    Raises:
        FileExistsError: ``path`` already exists (also when another process
            created it while ``write`` ran). It is left untouched.
    """
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists; it is never overwritten")
    tmp = _create_unique_temporary_file(target)
    try:
        write(tmp)
        _publish_exclusive(tmp, target)
    finally:
        _discard(tmp)


def write_text_atomically(
    path: str | os.PathLike[str], text: str, *, encoding: str = DEFAULT_ENCODING
) -> None:
    """:func:`write_atomically` for text."""
    write_atomically(path, lambda tmp: _write_text(tmp, text, encoding))


def create_text_atomically(
    path: str | os.PathLike[str], text: str, *, encoding: str = DEFAULT_ENCODING
) -> None:
    """:func:`create_atomically` for text."""
    create_atomically(path, lambda tmp: _write_text(tmp, text, encoding))


def _write_text(tmp: Path, text: str, encoding: str) -> None:
    tmp.write_text(text, encoding=encoding)
