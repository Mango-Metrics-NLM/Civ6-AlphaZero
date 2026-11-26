"""The per-iteration run record, ``<checkpoint_dir>/iterations.jsonl`` (LF-A13, T10 review I2).

One JSON object per line, one line per completed training iteration. Three rules
keep it readable when a directory is reused:

* **Identity.** Every line carries the run's identity (``run_id``, ``seed``,
  ``git_sha``, ``started_at``, ``resumed_from``, ``start_iteration``), so lines
  from different processes can always be told apart.
* **Fresh runs rotate, never delete.** A fresh (non-resumed) run moves an existing
  ``iterations.jsonl`` and every ``eval_*.json`` into ``previous_<UTC stamp>/``
  before it starts, so a shorter re-run cannot leave an earlier run's
  higher-numbered evaluations beside its own.
* **Resumes mark, never hide.** A resumed run appends to the existing file; a line
  for an iteration the file already holds gets ``"supersedes_earlier_record": true``.
  Readers wanting one line per iteration take the last one.

Crash safety: each line is one ``write`` followed by ``flush`` and ``os.fsync``.
A crash mid-write can still leave a torn line, so :func:`read_iteration_records`
skips any line that does not parse (with a warning), and an appender terminates
a torn tail before writing so a new line is never glued onto it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: File name of the per-iteration record inside the checkpoint directory.
ITERATIONS_FILE = "iterations.jsonl"

#: Files a fresh run moves aside: the record and every in-loop evaluation.
_ROTATED_GLOBS = (ITERATIONS_FILE, "eval_*.json")


def read_iteration_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Every parseable line of an ``iterations.jsonl``; unparseable (torn) lines are skipped."""
    target = Path(path)
    if not target.is_file():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("%s:%d is not valid JSON (a torn write?); skipped", target, number)
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def rotate_previous_run(
    checkpoint_dir: str | os.PathLike[str], *, stamp: str | None = None
) -> Path | None:
    """Move the record and ``eval_*.json`` into a new ``previous_<stamp>/``; never delete.

    Returns the directory created, or ``None`` when there was nothing to move. An
    existing ``previous_<stamp>`` is never reused, so rotations cannot overwrite
    one another.
    """
    root = Path(checkpoint_dir)
    stale = sorted({p for pattern in _ROTATED_GLOBS for p in root.glob(pattern) if p.is_file()})
    if not stale:
        return None
    base = f"previous_{stamp or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    destination = root / base
    suffix = 1
    while destination.exists():
        destination = root / f"{base}_{suffix}"
        suffix += 1
    destination.mkdir()
    for path in stale:
        os.replace(path, destination / path.name)
    logger.warning(
        "Fresh run into %s: moved %d file(s) from an earlier run to %s (nothing deleted).",
        root,
        len(stale),
        destination,
    )
    return destination


class IterationLog:
    """Appends identity-stamped, crash-safe lines to ``iterations.jsonl``.

    Args:
        checkpoint_dir: Directory holding the record.
        identity: Fields stamped onto every line (the run's identity).
        fresh: ``True`` for a run starting from the top: earlier files are rotated
            aside first. ``False`` for a resume: the file is appended to, and lines
            for iterations it already holds are marked as superseding them.
    """

    def __init__(
        self, checkpoint_dir: str | os.PathLike[str], *, identity: dict[str, Any], fresh: bool
    ) -> None:
        self.path = Path(checkpoint_dir) / ITERATIONS_FILE
        self.identity = dict(identity)
        self.rotated_to = rotate_previous_run(checkpoint_dir) if fresh else None
        self._seen = {r.get("iteration") for r in read_iteration_records(self.path)}
        self.records_written = 0
        self.prequential_failures = 0
        self._terminate_torn_tail()

    def _terminate_torn_tail(self) -> None:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return
        with open(self.path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) == b"\n":
                return
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write("\n")

    def append(self, record: dict[str, Any]) -> None:
        """Stamp ``record`` with the identity and append it as one fsync'd line."""
        line: dict[str, Any] = {**self.identity, **record}
        iteration = record.get("iteration")
        if iteration in self._seen:
            line["supersedes_earlier_record"] = True
        self._seen.add(iteration)
        prequential = record.get("prequential")
        if isinstance(prequential, dict) and "error" in prequential:
            self.prequential_failures += 1
        text = json.dumps(line, sort_keys=True) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        self.records_written += 1
