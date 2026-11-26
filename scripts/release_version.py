#!/usr/bin/env python3
"""Release tags and the versions they stand for (plan S3.5).

Tags are ``vX.Y.Z`` (final) or ``vX.Y.Z-rc.N`` (release candidate, N >= 1).
The repository always carries the BASE version ``X.Y.Z`` in ``pyproject.toml``,
``alpha_zero/__init__.py`` and ``CITATION.cff``: release prep bumps them once,
before promotion, and the final-tag rule forbids later edits. So:

* ``check`` -- ``${tag%%-*}`` (without the ``v``) must equal pyproject's version;
* ``stamp`` -- an rc build stamps PEP 440 ``X.Y.ZrcN`` into the BUILD COPY of
  ``pyproject.toml`` (and ``__version__``) so the wheel is versioned as the
  pre-release it is. The release workflow runs it in its own checkout; nothing
  is committed;
* ``notes`` -- release notes come from the base version's ``## [X.Y.Z]``
  CHANGELOG section, for rc and final alike.

Used by ``.github/workflows/release.yml`` and ``scripts/maintainer/tag_release.py``.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TAG_RE = re.compile(
    r"^v(?P<base>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))(?:-rc\.(?P<rc>[1-9]\d*))?$"
)


class ReleaseError(ValueError):
    """A tag, version or file does not satisfy the release rules."""


@dataclass(frozen=True)
class ReleaseTag:
    name: str
    base: str
    rc: int | None

    @property
    def is_rc(self) -> bool:
        return self.rc is not None

    @property
    def pep440(self) -> str:
        return self.base if self.rc is None else f"{self.base}rc{self.rc}"


def parse_tag(tag: str) -> ReleaseTag:
    """Parse ``vX.Y.Z`` / ``vX.Y.Z-rc.N`` (a ``refs/tags/`` prefix is accepted)."""
    name = tag.removeprefix("refs/tags/")
    match = TAG_RE.match(name)
    if not match:
        raise ReleaseError(f"{tag!r} is not a release tag (vX.Y.Z or vX.Y.Z-rc.N, N >= 1)")
    rc = match.group("rc")
    return ReleaseTag(name=name, base=match.group("base"), rc=int(rc) if rc else None)


def pyproject_version(text: str) -> str:
    try:
        version = tomllib.loads(text)["project"]["version"]
    except (tomllib.TOMLDecodeError, KeyError) as exc:
        raise ReleaseError(f"pyproject.toml has no readable [project].version: {exc}") from exc
    return str(version)


def check_matches(tag: ReleaseTag, pyproject_text: str) -> None:
    declared = pyproject_version(pyproject_text)
    if declared != tag.base:
        raise ReleaseError(
            f"tag {tag.name} is version {tag.base}, but pyproject.toml says {declared}. "
            "Bump [project].version (and CHANGELOG.md, CITATION.cff, alpha_zero/__init__.py) "
            "in release prep, or tag the right version."
        )


_PROJECT_VERSION_RE = re.compile(r'^(version\s*=\s*)"[^"]*"', re.M)
_INIT_VERSION_RE = re.compile(r'^(__version__\s*=\s*)"[^"]*"', re.M)


def stamp_pyproject(text: str, tag: ReleaseTag) -> str:
    """``text`` with ``[project].version`` set to ``tag.pep440`` (only that line)."""
    check_matches(tag, text)
    header = re.search(r"^\[project\]\s*$", text, re.M)
    if header is None:
        raise ReleaseError("pyproject.toml has no [project] table")
    start = header.end()
    nxt = re.search(r"^\[", text[start:], re.M)
    end = start + nxt.start() if nxt else len(text)
    section, count = _PROJECT_VERSION_RE.subn(rf'\g<1>"{tag.pep440}"', text[start:end], count=1)
    if count != 1:
        raise ReleaseError("no version line in [project]")
    out = text[:start] + section + text[end:]
    if pyproject_version(out) != tag.pep440:
        raise ReleaseError("stamping did not take effect")
    return out


def stamp_init(text: str, tag: ReleaseTag) -> str:
    out, count = _INIT_VERSION_RE.subn(rf'\g<1>"{tag.pep440}"', text, count=1)
    if count != 1:
        raise ReleaseError("no __version__ assignment to stamp")
    return out


def release_notes(changelog: str, tag: ReleaseTag, *, repo: str) -> str | None:
    """The base version's CHANGELOG section as release notes, or None if absent."""
    pattern = re.compile(rf"^## \[{re.escape(tag.base)}\][^\n]*\n(.*?)(?=^## \[|\Z)", re.M | re.S)
    match = pattern.search(changelog)
    body = match.group(1).strip() if match else ""
    if not body:
        return None
    return (
        f"## Changes in {tag.name}\n\n{body}\n\n"
        f"**Full Changelog**: https://github.com/{repo}/blob/{tag.name}/CHANGELOG.md\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="the tag's base version must equal pyproject's")
    c.add_argument("--tag", required=True)
    c.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    s = sub.add_parser("stamp", help="stamp the PEP 440 version into the build copy")
    s.add_argument("--tag", required=True)
    s.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    s.add_argument("--init", type=Path, default=None)
    n = sub.add_parser("notes", help="write release notes from CHANGELOG.md")
    n.add_argument("--tag", required=True)
    n.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    n.add_argument("--repo", required=True)
    n.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        tag = parse_tag(args.tag)
        if args.cmd == "check":
            check_matches(tag, args.pyproject.read_text(encoding="utf-8"))
            print(f"{tag.name}: base version {tag.base} matches pyproject.toml")
        elif args.cmd == "stamp":
            args.pyproject.write_text(
                stamp_pyproject(args.pyproject.read_text(encoding="utf-8"), tag), encoding="utf-8"
            )
            if args.init is not None:
                args.init.write_text(
                    stamp_init(args.init.read_text(encoding="utf-8"), tag), encoding="utf-8"
                )
            print(f"stamped {tag.pep440}")
        else:
            notes = release_notes(args.changelog.read_text(encoding="utf-8"), tag, repo=args.repo)
            if notes is None:
                print(f"no '## [{tag.base}]' section in {args.changelog}", file=sys.stderr)
                return 3
            args.out.write_text(notes, encoding="utf-8")
            print(f"wrote {args.out}")
    except ReleaseError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
