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
import shlex
import sys
import tempfile
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

from alpha_zero.bridge import tech_registry
from alpha_zero.rules import gameinfo as rules_gameinfo

#: The data root's variable. Read here, not through ``alpha_zero.utils.data_root``:
#: importing ``alpha_zero.utils`` loads numpy, and this tool needs neither.
DATA_ROOT_ENV = rules_gameinfo.DATA_ROOT_ENV

LINE_PREFIX = "AZ_GAMEINFO:"
CHUNK_PREFIX = "AZ_GAMEINFO_CHUNK:"
END_PREFIX = "AZ_GAMEINFO_END:"
#: Printed at the start of every export attempt, before any table is read.
BEGIN_MARKER = "AZ_GAMEINFO_BEGIN"
SUBDIR = "gameinfo"

#: The tables a capture must hold. The mod leaves one out when it could not
#: read it, and the importer would take a missing ``Units`` for "no units".
REQUIRED_TABLES: tuple[str, ...] = (
    "Technologies",
    "TechnologyPrereqs",
    "Civics",
    "CivicPrereqs",
    "Buildings",
    "Units",
)

#: Piece numbers and counts have at most nine digits: a longer run of digits is
#: not a count, and Python refuses to convert a very long one at all.
_CHUNK = re.compile(r"([0-9]{1,9})/([0-9]{1,9}):(.*)", re.DOTALL)
#: ``n:checksum``, ASCII digits only; blanks around the checksum are noise.
_END = re.compile(r"([0-9]{1,9}):\s*([0-9]{1,10})")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MAX_NAME_LENGTH = 100
#: Names Windows gives to devices, whatever follows a dot.
_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{n}" for n in range(1, 10)}
    | {f"lpt{n}" for n in range(1, 10)}
)


class GameInfoCaptureError(ValueError):
    """No usable snapshot was found, or it may not be written where asked."""


def _after(line: str, prefix: str) -> str | None:
    """What follows ``prefix`` in ``line`` (the game puts a context name before it), or None."""
    at = line.find(prefix)
    return None if at < 0 else line[at + len(prefix) :]


def _parse_document(text: str, where: str) -> dict[str, Any]:
    """The snapshot in ``text``, read as strictly as the rules importer reads it.

    A repeated key or a number too long for Python to convert is refused here,
    in one line: read loosely, the first would be filed with one of its values
    silently dropped and the second would end the tool with a traceback.
    """
    try:
        document = rules_gameinfo.loads(text)
    except rules_gameinfo.GameInfoError as exc:
        raise GameInfoCaptureError(f"{where}: not valid JSON ({exc})") from exc
    if not isinstance(document, dict):
        raise GameInfoCaptureError(f"{where}: expected a JSON object of tables")
    return document


@dataclass(frozen=True)
class LogScan:
    """What a ``Lua.log`` holds.

    Attributes:
        document: The last complete GameInfo message, or None when there is none.
        complete: How many complete messages the log holds (one per game load).
        later_incomplete: True when an unfinished message follows the last
            complete one, so ``document`` is not the newest load's.
    """

    document: dict[str, Any] | None
    complete: int
    later_incomplete: bool


def scan_log(text: str) -> LogScan:
    """Find the GameInfo messages in the text of a ``Lua.log``.

    A message is one ``AZ_GAMEINFO:`` line, or ``AZ_GAMEINFO_CHUNK:i/n:`` lines
    closed by ``AZ_GAMEINFO_END:n:checksum`` with every piece present, every
    line agreeing on ``n``, and the Adler-32 of the joined text equal to the
    checksum. The checksum is what keeps pieces of two exports from being
    joined into one snapshot when lines are lost between them. A log covers a
    whole launch, so it can hold one message per game loaded, and their tables
    can differ (other mods, other rules).

    The mod prints ``AZ_GAMEINFO_BEGIN`` when an export starts, before it reads
    a table. An attempt that then exports nothing (no table could be read)
    leaves only that line, and it is what says the last complete message is
    not the newest load's.

    Raises:
        GameInfoCaptureError: If the last complete message is not a JSON object.
    """
    last: str | None = None
    complete = 0
    later_incomplete = False
    pieces: dict[int, str] = {}
    totals: set[int] = set()

    lines = text.splitlines()
    # A log read while the game is writing can end in half a line.
    tail_unterminated = bool(text) and not text.endswith(("\n", "\r"))
    for number, line in enumerate(lines):
        # The chunk and end markers start with the single-line marker's
        # letters, so they are tried first.
        chunk = _after(line, CHUNK_PREFIX)
        if chunk is not None:
            match = _CHUNK.fullmatch(chunk)
            if match:
                if int(match.group(1)) == 1:
                    pieces, totals = {}, set()
                pieces[int(match.group(1))] = match.group(3)
                totals.add(int(match.group(2)))
            # A chunk set is open: until it is closed, the newest message is unfinished.
            later_incomplete = last is not None
            continue
        end = _after(line, END_PREFIX)
        if end is not None:
            joined = _checked_join(end, pieces, totals)
            if joined is not None:
                last = joined
                complete += 1
                later_incomplete = False
            else:
                later_incomplete = last is not None
            pieces, totals = {}, set()
            continue
        single = _after(line, LINE_PREFIX)
        if single is not None:
            if tail_unterminated and number == len(lines) - 1 and not _is_document(single):
                # Half a line: the newest message is unfinished, like an open
                # chunk set. It must not hide an earlier complete one.
                later_incomplete = last is not None
                pieces, totals = {}, set()
                continue
            last = single.strip()
            complete += 1
            later_incomplete = False
            pieces, totals = {}, set()
            continue
        # Tried last and only at the end of a line, so that snapshot data that
        # happened to contain the marker's letters is never read as one.
        if line.rstrip().endswith(BEGIN_MARKER):
            # A newer export started: whatever was being collected is not its.
            pieces, totals = {}, set()
            later_incomplete = last is not None

    document = None if last is None else _parse_document(last, "the GameInfo message in the log")
    return LogScan(document=document, complete=complete, later_incomplete=later_incomplete)


def _is_document(text: str) -> bool:
    """Whether ``text`` is a whole JSON object, whatever rules it breaks.

    Only "was the line cut short" is asked here. A whole object with a key written
    twice or an oversized number is complete, and ``_parse_document`` refuses it; calling
    it unfinished would let ``--allow-earlier`` file an older snapshot in its place.
    Digits are not converted, so a number of any length does not fail the reading.
    """
    try:
        return isinstance(json.loads(text, parse_int=lambda _digits: 0), dict)
    except (ValueError, RecursionError):
        return False


def _checked_join(end: str, pieces: dict[int, str], totals: set[int]) -> str | None:
    """The chunk set's text when the END line ``n:checksum`` closes it, else None.

    Complete means: every piece 1..n is there, every chunk line named the same
    n, and the joined text has the checksum the mod computed over the whole
    text before cutting it. An END line without a checksum closes nothing.
    """
    match = _END.fullmatch(end.strip())
    if match is None:
        return None
    total, checksum = int(match.group(1)), int(match.group(2))
    # The count comes from the log. Compare it with how many pieces arrived
    # before walking 1..n, so a line claiming a hundred million pieces costs
    # nothing.
    if total <= 0 or totals != {total} or len(pieces) != total:
        return None
    if any(index not in pieces for index in range(1, total + 1)):
        return None
    joined = "".join(pieces[i] for i in range(1, total + 1))
    if zlib.adler32(joined.encode("utf-8")) != checksum:
        return None
    return joined


def snapshot_from_log(text: str) -> dict[str, Any]:
    """The last complete GameInfo message in the text of a ``Lua.log``.

    An unfinished chunk set is passed over for an earlier complete message;
    :func:`scan_log` says when that happened.

    Raises:
        GameInfoCaptureError: If the log holds no complete message, or the
            last complete one is not a JSON object.
    """
    document = scan_log(text).document
    if document is None:
        raise GameInfoCaptureError(
            "no complete GameInfo message in the log: the mod writes one when a game "
            "loads (ENABLE_GAMEINFO_EXPORT), so load a game with the mod enabled first. "
            "A message without a checksum on its END line is from an older copy of the "
            "mod: reinstall it (python scripts/install_civ6_mod.py)"
        )
    return document


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
    if text.strip() == BEGIN_MARKER:
        raise GameInfoCaptureError(
            f"{source}: the mod began a snapshot and did not finish it (it could read no "
            "table, or was still writing); load a game with the mod enabled again"
        )
    return _parse_document(text, str(source))


def check_snapshot(document: Any) -> rules_gameinfo.ImportSummary:
    """Prove the rules importer accepts ``document``; return what it would import.

    Raises:
        GameInfoCaptureError: With the importer's own message, if it does not.
    """
    if isinstance(document, dict):
        missing = [table for table in REQUIRED_TABLES if table not in document]
        if missing:
            raise GameInfoCaptureError(
                "the snapshot is incomplete: the mod could not read "
                f"{', '.join(missing)} (its log says why); a rule set built from it "
                "would silently lack them"
            )
    try:
        summary = rules_gameinfo.convert(document)
    except ValueError as exc:
        raise GameInfoCaptureError(f"the snapshot is not importable: {exc}") from exc
    # The importer only counts civics; the research registry numbers its civic
    # slots from them (CIV6AZ_BRIDGE_GAMEINFO). Apply the registry's own rule
    # now, so a snapshot that is filed is one the registry accepts.
    try:
        tech_registry.civic_ids(tech_registry.civic_types_of(document, "the snapshot"))
    except tech_registry.ResearchRegistryError as exc:
        raise GameInfoCaptureError(
            f"the snapshot's civics cannot number research slots: {exc}"
        ) from exc
    return summary


def write_snapshot(document: Any, *, name: str | None = None) -> Path:
    """Check the snapshot and file it as ``<CIV6AZ_DATA_ROOT>/gameinfo/<name>.json``.

    Raises:
        GameInfoCaptureError: If the snapshot is not importable, the data root
            is unset or inside the repository, the name is not a plain file
            stem, or the file already exists. Nothing is written then.
    """
    check_snapshot(document)

    stem = name if name is not None else datetime.now(UTC).strftime("gameinfo-%Y%m%dT%H%M%SZ")
    if (
        not _NAME.fullmatch(stem)
        or len(stem) > _MAX_NAME_LENGTH
        or stem.split(".", 1)[0].lower() in _DEVICE_NAMES
    ):
        raise GameInfoCaptureError(
            f"{stem!r} is not a usable file name (letters, digits, '.', '-', '_'; "
            f"at most {_MAX_NAME_LENGTH} characters; not a device name)"
        )

    raw_root = os.environ.get(DATA_ROOT_ENV, "").strip()
    if not raw_root:
        raise GameInfoCaptureError(
            f"{DATA_ROOT_ENV} is not set: a snapshot is the user's own game data and is "
            "written under the data root, outside the repository"
        )

    # Where the file would really land: symbolic links are followed first, so a
    # data root or a gameinfo/ directory that points into a repository is seen
    # for what it is. Nothing is created before this check.
    try:
        directory = (Path(raw_root).expanduser() / SUBDIR).resolve()
    except (OSError, RuntimeError) as exc:
        raise GameInfoCaptureError(f"cannot resolve {raw_root!r}: {exc}") from exc
    destination = directory / f"{stem}.json"
    checkout = rules_gameinfo.enclosing_checkout(destination)
    source_tree = rules_gameinfo.checkout_root()
    if checkout is None and source_tree is not None and source_tree in destination.parents:
        checkout = source_tree
    if checkout is not None:
        raise GameInfoCaptureError(
            f"{destination} is inside the repository {checkout}: "
            f"point {DATA_ROOT_ENV} at a directory outside it"
        )
    if destination.exists():
        raise GameInfoCaptureError(f"{destination} already exists; snapshots are never overwritten")

    temporary: str | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=directory, prefix=destination.name + ".", suffix=".tmp"
        )
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            json.dump(document, out, indent=1, sort_keys=True)
            out.write("\n")
        _publish(Path(temporary), destination)
    except OSError as exc:
        raise GameInfoCaptureError(f"cannot write {destination}: {exc}") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return destination


def _publish(temporary: Path, destination: Path) -> None:
    """Give the finished file its name, without ever replacing an existing one.

    A hard link fails if the name is taken, which closes the gap between the
    check above and the write. Where the file system has no hard links, the
    name is claimed with an exclusive create, which fails the same way, and
    the finished file is then renamed over that empty claim. A check followed
    by a rename would replace a file that appeared in between.
    """
    taken = GameInfoCaptureError(f"{destination} already exists; snapshots are never overwritten")
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise taken from None
    except OSError:
        try:
            claim = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise taken from None
        os.close(claim)
        try:
            os.replace(temporary, destination)
        except OSError:
            # The claim is ours and empty: do not leave it looking like a snapshot.
            destination.unlink(missing_ok=True)
            raise


#: Characters no shell treats specially (``%`` excluded so Windows cmd ``%VAR%`` never passes unquoted).
_SHELL_PLAIN = re.compile(r"[A-Za-z0-9@+=:,./\\_-]*")
_WINDOWS_EXPAND_CHARS = frozenset("$`%")


def _for_a_shell(path: PurePath, *, windows: bool | None = None) -> str:
    """``path`` as it can be pasted into a command line on this platform.

    Unchanged when it holds nothing a shell treats specially; in double quotes
    when only whitespace (or Windows non-expanding meta characters like ``&``)
    appears; single-quoted for POSIX shells via ``shlex.quote`` when other
    characters appear; and single-quoted for PowerShell when ``$``, backtick, or
    ``%`` appears on Windows.
    """
    text = str(path)
    if _SHELL_PLAIN.fullmatch(text):
        return text
    if windows is None:
        windows = os.name == "nt"
    only_whitespace = all(c.isspace() or _SHELL_PLAIN.fullmatch(c) for c in text)
    if windows:
        if any(c in _WINDOWS_EXPAND_CHARS for c in text):
            escaped = text.replace("'", "''")
            return f"'{escaped}'"
        return f'"{text}"'
    if only_whitespace:
        return f'"{text}"'
    return shlex.quote(text)


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
    parser.add_argument(
        "--allow-earlier",
        action="store_true",
        help="when the newest GameInfo message in the log is unfinished, file the last "
        "complete one instead of refusing (it may be another game load's tables)",
    )
    args = parser.parse_args(argv)

    try:
        if args.from_log is not None:
            log = Path(args.from_log).expanduser()
            try:
                text = log.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise GameInfoCaptureError(f"cannot read {log}: {exc}") from exc
            scan = scan_log(text)
            if scan.document is None:
                document = snapshot_from_log(text)  # raises, with the explanation
            else:
                document = scan.document
            if scan.later_incomplete:
                if not args.allow_earlier:
                    raise GameInfoCaptureError(
                        "the newest GameInfo export in the log is incomplete (the game was "
                        "still writing it, it stopped, it could read no table, or lines of it "
                        "are missing); an earlier complete one exists, but it may be another "
                        "game load's tables. Load the game again, or pass --allow-earlier to "
                        "file the earlier one"
                    )
                print(
                    "gameinfo_capture: the newest GameInfo message is incomplete; "
                    "filing the last complete one (--allow-earlier)",
                    file=sys.stderr,
                )
            elif scan.complete > 1:
                print(
                    f"gameinfo_capture: the log holds {scan.complete} GameInfo messages "
                    "(one per game load); filing the last"
                )
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
    shown = _for_a_shell(written)
    print(f"  python -m alpha_zero.rules import-gameinfo {shown}")
    print(f"  CIV6AZ_BRIDGE_GAMEINFO={shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
