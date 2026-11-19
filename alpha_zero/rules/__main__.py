"""``python -m alpha_zero.rules``: validate a rule set, or print its content hash.

``TARGET`` is a built-in id (``science-v1-draft``) or a path to a rule-set file.

Exit codes: 0 success; 1 the rule set is invalid, malformed or does not match its
pin; 2 usage error (unknown subcommand, or a target that is neither a built-in id
nor an existing file).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from alpha_zero.rules.builtin import BUILTIN, RulesetHashMismatchError, builtin_ids, load_builtin
from alpha_zero.rules.loader import RulesetLoadError, content_sha256, load_ruleset
from alpha_zero.rules.queries import available
from alpha_zero.rules.registry import build_registry
from alpha_zero.rules.spec import RulesetSpec
from alpha_zero.rules.validate import RulesetValidationError, ensure_valid

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2


class _UsageError(Exception):
    """The target names neither a built-in rule set nor a file."""


def _load(target: str) -> RulesetSpec:
    """Load ``target`` strictly; a built-in is also hash-checked and validated."""
    if target in BUILTIN:
        return load_builtin(target)
    path = Path(target)
    if not path.is_file():
        known = ", ".join(builtin_ids())
        raise _UsageError(f"{target!r} is neither a built-in rule set ({known}) nor a file")
    return load_ruleset(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alpha_zero.rules",
        description="Validate and identify game rule sets (see docs/GAME_RULES.md).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("validate", "check a rule set's structure; exit 1 and list every violated rule"),
        ("hash", "print '<id> <content sha256>' of a rule set that loads"),
    ):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("target", help="a built-in id or a path to a rule-set YAML file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    args = _build_parser().parse_args(argv)
    try:
        spec = _load(args.target)
        if args.command == "validate":
            ensure_valid(spec)
            start = ", ".join(available(spec, ())) or "none"
            print(f"{spec.id}: valid")
            print(f"  techs: {len(spec.techs)}; available at the start: {start}")
            print(f"  index fingerprint: {build_registry(spec).fingerprint}")
        else:
            print(f"{spec.id} {content_sha256(spec)}")
    except _UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except RulesetValidationError as exc:
        print(f"{exc.ruleset_id}: invalid", file=sys.stderr)
        for issue in exc.issues:
            print(f"  {issue}", file=sys.stderr)
        return EXIT_INVALID
    except (RulesetLoadError, RulesetHashMismatchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
