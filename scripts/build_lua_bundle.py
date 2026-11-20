#!/usr/bin/env python3
"""Build the Lua bundle the Civ6 mod loads from its module files.

Civ6 loads one file, ``mod/AlphaZeroAI/Scripts/AlphaZeroMain.lua``. It is the
module files of the same directory, concatenated in the order the ``.modinfo``
lists them, each between a ``BEGIN`` and an ``END`` marker. Until this script the
bundle was maintained by hand and had drifted from the module files, so an edit
to a module changed nothing in the game.

The module files are the source; the bundle is generated and committed.

Usage::

    python scripts/build_lua_bundle.py            # rewrite the bundle
    python scripts/build_lua_bundle.py --check    # exit 1 when it is out of date

Exit codes: 0 up to date (or written), 1 out of date under ``--check``,
2 usage or layout error (a module the modinfo lists is missing, say).

Line endings: everything is read with CRLF normalised to LF and the bundle is
written with LF, so a Windows checkout with ``core.autocrlf`` agrees with a
Linux one.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOD_DIR = REPO_ROOT / "mod" / "AlphaZeroAI"
SCRIPTS_SUBDIR = "Scripts"
BUNDLE_NAME = "AlphaZeroMain.lua"

EXIT_OK = 0
EXIT_OUT_OF_DATE = 1
EXIT_USAGE = 2

_RULE = "-- " + "=" * 76
HEADER = (
    f"{_RULE}\n"
    f"-- {BUNDLE_NAME} - Bundled entry point for AlphaZero AI Bridge\n"
    f"{_RULE}\n"
    "-- All modules concatenated into a single file for Civ6 AddUserInterfaces.\n"
    f"{_RULE}\n"
    "\n"
    "\n"
)
_LUA_SUFFIX = ".lua"
# The modinfo is read with three small patterns, not an XML parser: the standard
# library's parsers are flagged as unsafe for untrusted input, and this needs
# only comments removed, the <Files> block found and its <File> entries listed.
_XML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_FILES_BLOCK = re.compile(r"<Files\s*>(.*?)</Files\s*>", re.DOTALL)
_FILE_ENTRY = re.compile(r"<File\s*>(.*?)</File\s*>", re.DOTALL)
_FILE_OPENING = re.compile(r"<File\b")
_PLAIN_PATH = re.compile(r"[A-Za-z0-9_.\\/ -]+")
_BOM = "\ufeff"


class BundleError(Exception):
    """The mod directory is not laid out as the bundle needs."""


def begin_marker(name: str) -> str:
    """The line that opens module ``name`` in the bundle."""
    return f"-- ========== BEGIN {name} ==========\n"


def end_marker(name: str) -> str:
    """The line that closes module ``name`` in the bundle."""
    return f"-- ========== END {name} ==========\n"


def _read(path: Path) -> str:
    """The file as text with LF line endings; a file Lua could not load is refused.

    A byte-order mark is refused, not stripped: in the middle of the bundle it
    would be a syntax error to Lua, and silently changing a module is not this
    script's job.
    """
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleError(f"{path.name} is not UTF-8 ({exc.reason} at byte {exc.start})") from exc
    if text.startswith(_BOM):
        raise BundleError(f"{path.name} starts with a byte-order mark; save it without one")
    return text.replace("\r\n", "\n")


def find_modinfo(mod_dir: Path) -> Path:
    """The single ``.modinfo`` file of ``mod_dir``."""
    found = sorted(mod_dir.glob("*.modinfo"))
    if len(found) != 1:
        raise BundleError(f"expected one .modinfo in {mod_dir}, found {len(found)}")
    return found[0]


def module_order(mod_dir: Path) -> list[str]:
    """The Lua module file names, in the order the modinfo's ``<Files>`` lists them.

    XML comments are removed first, so a commented-out entry is not an entry. The
    bundle itself is listed there too and is left out. Every other ``.lua`` entry
    must be spelled ``Scripts/<name>.lua`` with forward slashes: an entry in any
    other spelling is refused, never skipped, because a skipped module would
    silently not ship. An entry with attributes, CDATA or any markup is refused for
    the same reason. A name listed twice is an error: it would be bundled twice.
    """
    modinfo = find_modinfo(mod_dir)
    blocks = _FILES_BLOCK.findall(_XML_COMMENT.sub("", _read(modinfo)))
    if len(blocks) != 1:
        raise BundleError(
            f"{modinfo.name} must hold exactly one <Files> block, found {len(blocks)}"
        )
    entries = _FILE_ENTRY.findall(blocks[0])
    if len(entries) != len(_FILE_OPENING.findall(blocks[0])):
        raise BundleError(
            f"{modinfo.name} has a <File> entry this script cannot read (attributes, or no "
            "closing tag); write it as <File>Scripts/Name.lua</File>"
        )
    prefix = f"{SCRIPTS_SUBDIR}/"
    names = []
    for raw in entries:
        listed = raw.strip()
        if not _PLAIN_PATH.fullmatch(listed):
            raise BundleError(f"{modinfo.name} lists {listed!r}, which is not a plain path")
        if not listed.lower().endswith(_LUA_SUFFIX):
            continue
        if (
            not listed.startswith(prefix)
            or not listed.endswith(_LUA_SUFFIX)
            or "\\" in listed
            or ".." in listed.split("/")
        ):
            raise BundleError(
                f"{modinfo.name} lists {listed!r}; a Lua entry must be spelled "
                f"{prefix}<name>{_LUA_SUFFIX}"
            )
        name = listed[len(prefix) :]
        if name != BUNDLE_NAME:
            names.append(name)
    if not names:
        raise BundleError(f"{modinfo.name} lists no Lua module")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise BundleError(f"the modinfo lists {', '.join(repeated)} more than once")
    return names


def build_bundle(mod_dir: Path) -> str:
    """The bundle text for ``mod_dir``: header, then each module between its markers."""
    scripts = mod_dir / SCRIPTS_SUBDIR
    sections = []
    for name in module_order(mod_dir):
        path = scripts / name
        if not path.is_file():
            raise BundleError(f"the modinfo lists {name}, which is not in {scripts}")
        sections.append(f"{begin_marker(name)}{_read(path)}\n{end_marker(name)}")
    return HEADER + "\n".join(sections)


def first_difference(expected: str, actual: str) -> str:
    """One line saying where two texts first differ, for the ``--check`` message."""
    expected_lines = expected.split("\n")
    actual_lines = actual.split("\n")
    for number, (want, have) in enumerate(zip(expected_lines, actual_lines, strict=False), 1):
        if want != have:
            return f"line {number}: generated {want!r}, committed {have!r}"
    return (
        f"line count: generated {len(expected_lines)}, committed {len(actual_lines)} "
        "(one is a prefix of the other)"
    )


def write_bundle(path: Path, text: str) -> None:
    """Publish ``text`` at ``path`` through a temporary file, with LF line endings."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(text.encode("utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 when the committed bundle differs from the generated one",
    )
    parser.add_argument(
        "--mod-dir",
        type=Path,
        default=DEFAULT_MOD_DIR,
        help="the mod directory holding the .modinfo and Scripts/ (default: mod/AlphaZeroAI)",
    )
    args = parser.parse_args(argv)
    bundle_path = args.mod_dir / SCRIPTS_SUBDIR / BUNDLE_NAME
    try:
        generated = build_bundle(args.mod_dir)
        committed = _read(bundle_path) if bundle_path.is_file() else None
    except (BundleError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if committed == generated:
        print(f"{bundle_path.name}: up to date")
        return EXIT_OK
    if args.check:
        where = "missing" if committed is None else first_difference(generated, committed)
        print(
            f"{bundle_path.name}: out of date ({where}). Edit the module files, never the "
            "bundle, then run: python scripts/build_lua_bundle.py",
            file=sys.stderr,
        )
        return EXIT_OUT_OF_DATE
    write_bundle(bundle_path, generated)
    print(f"wrote {bundle_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
