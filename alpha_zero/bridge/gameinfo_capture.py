"""Capture a snapshot of the game's own rule tables from a running game (GM-B4).

The mod writes the snapshot once per game load (``GameInfoExporter.lua``): a few
GameInfo tables (technologies, civics, buildings, units and their
prerequisites), identifiers and whole numbers only. With file access it writes
``gameinfo.json`` into its communication directory; without (the usual case)
it prints it to ``Lua.log`` as ``AZ_GAMEINFO:`` lines, chunked like states.

This module reads either, checks that the rules importer accepts it, and files
it under the data root (``CIV6AZ_DATA_ROOT``), never inside the repository: the
tables are the user's own game data and are not committed.

    python -m alpha_zero.bridge.gameinfo_capture --from-log PATH/Lua.log
    python -m alpha_zero.rules import-gameinfo <snapshot>     # the rule set
    CIV6AZ_BRIDGE_GAMEINFO=<snapshot>                         # the civic order
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alpha_zero.rules import gameinfo as rules_gameinfo

#: The data root's variable. Read here, not through ``alpha_zero.utils.data_root``:
#: importing ``alpha_zero.utils`` loads numpy, and this tool needs neither.
DATA_ROOT_ENV = rules_gameinfo.DATA_ROOT_ENV

LINE_PREFIX = "AZ_GAMEINFO:"
CHUNK_PREFIX = "AZ_GAMEINFO_CHUNK:"
END_PREFIX = "AZ_GAMEINFO_END:"
SUBDIR = "gameinfo"

_CHUNK = re.compile(r"(\d+)/(\d+):(.*)", re.DOTALL)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class GameInfoCaptureError(ValueError):
    """No usable snapshot was found, or it may not be written where asked."""


def _after(line: str, prefix: str) -> str | None:
    """What follows ``prefix`` in ``line`` (the game puts a context name before it), or None."""
    at = line.find(prefix)
    return None if at < 0 else line[at + len(prefix) :]


def _parse_document(text: str, where: str) -> dict[str, Any]:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GameInfoCaptureError(f"{where}: not valid JSON ({exc})") from exc
    if not isinstance(document, dict):
        raise GameInfoCaptureError(f"{where}: expected a JSON object of tables")
    return document


def snapshot_from_log(text: str) -> dict[str, Any]:
    """The last complete GameInfo message in the text of a ``Lua.log``.

    A message is one ``AZ_GAMEINFO:`` line, or ``AZ_GAMEINFO_CHUNK:i/n:`` lines
    closed by ``AZ_GAMEINFO_END:n`` with every piece present. An unfinished
    chunk set is passed over for an earlier complete message.

    Raises:
        GameInfoCaptureError: If the log holds no complete message, or the
            last complete one is not a JSON object.
    """
    last: str | None = None
    pieces: dict[int, str] = {}
    for line in text.splitlines():
        # The chunk and end markers start with the single-line marker's
        # letters, so they are tried first.
        chunk = _after(line, CHUNK_PREFIX)
        if chunk is not None:
            match = _CHUNK.fullmatch(chunk)
            if match:
                if int(match.group(1)) == 1:
                    pieces = {}
                pieces[int(match.group(1))] = match.group(3)
            continue
        end = _after(line, END_PREFIX)
        if end is not None:
            total = int(end.strip()) if end.strip().isdigit() else -1
            if total > 0 and sorted(pieces) == list(range(1, total + 1)):
                last = "".join(pieces[i] for i in range(1, total + 1))
            pieces = {}
            continue
        single = _after(line, LINE_PREFIX)
        if single is not None:
            last = single.strip()
            pieces = {}
    if last is None:
        raise GameInfoCaptureError(
            "no complete GameInfo message in the log: the mod writes one when a game "
            "loads (ENABLE_GAMEINFO_EXPORT), so load a game with the mod enabled first"
        )
    return _parse_document(last, "the GameInfo message in the log")


def snapshot_from_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The document in a ``gameinfo.json`` the mod wrote.

    Raises:
        GameInfoCaptureError: If the file cannot be read or is not a JSON object.
    """
    source = Path(path).expanduser()
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise GameInfoCaptureError(f"cannot read {source}: {exc}") from exc
    return _parse_document(text, str(source))


def check_snapshot(document: Any) -> rules_gameinfo.ImportSummary:
    """Prove the rules importer accepts ``document``; return what it would import.

    Raises:
        GameInfoCaptureError: With the importer's own message, if it does not.
    """
    try:
        return rules_gameinfo.convert(document)
    except ValueError as exc:
        raise GameInfoCaptureError(f"the snapshot is not importable: {exc}") from exc


def write_snapshot(document: Any, *, name: str | None = None) -> Path:
    """Check the snapshot and file it as ``<CIV6AZ_DATA_ROOT>/gameinfo/<name>.json``.

    Raises:
        GameInfoCaptureError: If the snapshot is not importable, the data root
            is unset or inside the repository, the name is not a plain file
            stem, or the file already exists. Nothing is written then.
    """
    check_snapshot(document)

    stem = name if name is not None else datetime.now(UTC).strftime("gameinfo-%Y%m%dT%H%M%SZ")
    if not _NAME.fullmatch(stem):
        raise GameInfoCaptureError(
            f"{stem!r} is not a plain file name (letters, digits, '.', '-', '_')"
        )

    raw_root = os.environ.get(DATA_ROOT_ENV, "").strip()
    if not raw_root:
        raise GameInfoCaptureError(
            f"{DATA_ROOT_ENV} is not set: a snapshot is the user's own game data and is "
            "written under the data root, outside the repository"
        )
    destination = Path(raw_root).expanduser().resolve() / SUBDIR / f"{stem}.json"
    checkout = rules_gameinfo.enclosing_checkout(destination)
    if checkout is not None:
        raise GameInfoCaptureError(
            f"{destination} is inside the repository checkout {checkout}: "
            f"point {DATA_ROOT_ENV} at a directory outside it"
        )
    if destination.exists():
        raise GameInfoCaptureError(f"{destination} already exists; snapshots are never overwritten")

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=destination.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            json.dump(document, out, indent=1, sort_keys=True)
            out.write("\n")
        _publish(Path(temporary), destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def _publish(temporary: Path, destination: Path) -> None:
    """Give the finished file its name, without ever replacing an existing one.

    A hard link fails if the name is taken, which closes the gap between the
    check above and the write. Where the file system has no hard links, fall
    back to a rename after checking again.
    """
    taken = GameInfoCaptureError(f"{destination} already exists; snapshots are never overwritten")
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise taken from None
    except OSError:
        if destination.exists():
            raise taken from None
        os.replace(temporary, destination)


def main(argv: Sequence[str] | None = None) -> int:
    """Capture a snapshot. Exit 0 written, 1 refused or unusable, 2 usage error."""
    parser = argparse.ArgumentParser(
        prog="python -m alpha_zero.bridge.gameinfo_capture",
        description="File the game's rule tables, as the mod exported them, under the data root.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-log", metavar="PATH", help="the game's Lua.log")
    source.add_argument("--from-file", metavar="PATH", help="the mod's gameinfo.json")
    parser.add_argument("--name", help="file stem of the snapshot (default: gameinfo-<UTC time>)")
    args = parser.parse_args(argv)

    try:
        if args.from_log is not None:
            log = Path(args.from_log).expanduser()
            try:
                text = log.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise GameInfoCaptureError(f"cannot read {log}: {exc}") from exc
            document = snapshot_from_log(text)
        else:
            document = snapshot_from_file(args.from_file)
        summary = check_snapshot(document)
        written = write_snapshot(document, name=args.name)
    except GameInfoCaptureError as exc:
        print(f"gameinfo_capture: {exc}", file=sys.stderr)
        return 1

    print(f"snapshot: {written}")
    for table, rows in sorted(document.items()):
        print(f"  {table}: {len(rows)} rows")
    print(
        f"  importable: {summary.techs} technologies, {summary.buildings} buildings, "
        f"{summary.units} units; {summary.civics_not_imported} civics counted"
    )
    print("next:")
    print(f"  python -m alpha_zero.rules import-gameinfo {written}")
    print(f"  CIV6AZ_BRIDGE_GAMEINFO={written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
