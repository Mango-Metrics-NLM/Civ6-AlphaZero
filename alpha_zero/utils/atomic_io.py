"""Write a file so that a killed process never leaves half of it behind.

A run that is interrupted mid-write must not leave a torn checkpoint or a torn
``result.json``: the re-run rule (``docs/EVALUATION.md`` section 7.1, "After a
VOID") compares what an interrupted attempt left with what its re-run produced,
and a file that cannot be read cannot be shown to reproduce.

Both helpers write to ``<name>.tmp`` beside the target (the same directory, so
the same filesystem) and then give it the target's name in one step:

* :func:`write_atomically` replaces the target (``os.replace``);
* :func:`create_atomically` refuses an existing target, for files that must never
  be overwritten.

The bytes written are exactly what the caller's writer produces; nothing here
changes a file's content. Needs only the standard library.

One writer per target: the temporary name is fixed, so two processes writing the
same target at once would share it. Callers own that (a run directory is claimed
by one run; a benchmark ``--out`` belongs to one arm).
"""

from __future__ import annotations

import logging
import os
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
    if target.exists():
        raise FileExistsError(f"{target} already exists; it is never overwritten")
    tmp = temporary_path(target)
    try:
        write(tmp)
        try:
            os.link(tmp, target)  # atomic, and fails if the target appeared meanwhile
        except FileExistsError:
            raise
        except OSError as exc:  # a filesystem without hard links (exFAT, some shares)
            logger.debug("no hard link for %s (%s); renaming instead", target, exc)
            if target.exists():
                raise FileExistsError(f"{target} already exists; it is never overwritten") from exc
            os.replace(tmp, target)
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
