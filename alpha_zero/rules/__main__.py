"""``python -m alpha_zero.rules``: validate, identify, pace and import rule sets.

``validate TARGET``            check structure; list every violated rule
``hash TARGET``                print the id and the content sha256
``pace TARGET``                zero-game pacing numbers (tree costs, science rates)
``import-gameinfo IN``         turn a GameInfo export into a rule-set file

``TARGET`` is a built-in id (``science-v1-draft``) or a path to a rule-set file.

Exit codes: 0 success; 1 the rule set is invalid, malformed, unreadable or does not
match its pin, or an export is malformed; 2 usage error (unknown subcommand, a target that is
neither a built-in id nor an existing file) or a refused output path.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from alpha_zero.rules.builtin import (
    BUILTIN,
    BuiltinTableError,
    RulesetHashMismatchError,
    builtin_ids,
    load_builtin,
)
from alpha_zero.rules.gameinfo import (
    DEFAULT_RULESET_ID,
    GameInfoError,
    RefusedPathError,
    import_gameinfo,
)
from alpha_zero.rules.loader import RulesetLoadError, content_sha256, load_ruleset
from alpha_zero.rules.pacing import PacingReport, pacing_report, turns_to_research
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


DEFAULT_PACE_MAX_TURNS = 60
#: Constant science rates the pace report tabulates.
PACE_RATES: tuple[int, ...] = (2, 4, 6, 8, 10, 15, 20)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alpha_zero.rules",
        description="Validate, identify, pace and import game rule sets (docs/GAME_RULES.md).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    target_help = "a built-in id or a path to a rule-set YAML file"
    for name, text in (
        ("validate", "check a rule set's structure; exit 1 and list every violated rule"),
        ("hash", "print '<id> <content sha256>' of a rule set that loads"),
    ):
        commands.add_parser(name, help=text).add_argument("target", help=target_help)
    pace = commands.add_parser("pace", help="zero-game pacing numbers of a valid rule set")
    pace.add_argument("target", help=target_help)
    pace.add_argument(
        "--max_turns",
        type=int,
        default=DEFAULT_PACE_MAX_TURNS,
        help=f"game length the band is read against (default {DEFAULT_PACE_MAX_TURNS})",
    )
    importer = commands.add_parser(
        "import-gameinfo", help="turn a GameInfo table export (JSON) into a rule-set file"
    )
    importer.add_argument("source", help="the export: a JSON object of tables")
    importer.add_argument(
        "--out",
        default=None,
        help="output file, outside the repository (default <CIV6AZ_DATA_ROOT>/gameinfo/<id>.yaml)",
    )
    importer.add_argument(
        "--id", default=DEFAULT_RULESET_ID, help=f"rule-set id (default {DEFAULT_RULESET_ID})"
    )
    return parser


def _print_pace(report: PacingReport) -> None:
    print(f"{report.ruleset_id}: pacing (zero-game model, no simulation)")
    print(f"  full-tree cost: {report.full_tree_cost}")
    if report.capstone_tech is None or report.capstone_cost is None:
        print("  capstone: none declared")
        return
    cost = report.capstone_cost
    print(f"  capstone: {report.capstone_tech}")
    print(f"  capstone path cost: {cost} (the capstone and its whole prerequisite set)")
    low, high = report.band_percent
    if report.band_rates is None:
        rates = "no whole rate lands in the band"
    else:
        rates = f"{report.band_rates[0]} to {report.band_rates[1]} science per turn"
    print(f"  to finish the capstone at {low}-{high}% of {report.max_turns} turns: {rates}")
    for rate in PACE_RATES:
        print(f"  at {rate} science per turn: {turns_to_research(cost, rate)} turns")


def _run_import(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.is_file():
        print(f"error: {args.source!r} is not a file", file=sys.stderr)
        return EXIT_USAGE
    try:
        written, summary = import_gameinfo(source, args.out, args.id)
    except RefusedPathError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except RulesetValidationError as exc:
        print(f"{exc.ruleset_id}: the imported rule set is invalid", file=sys.stderr)
        for issue in exc.issues:
            print(f"  {issue}", file=sys.stderr)
        return EXIT_INVALID
    except (GameInfoError, RulesetLoadError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    print(f"wrote {written}")
    print(
        f"  {summary.ruleset_id}: {summary.techs} techs, {summary.units} units, "
        f"{summary.buildings} buildings"
    )
    if summary.skipped_non_positive_cost:
        print(f"  skipped {summary.skipped_non_positive_cost} rows with a non-positive cost")
    if summary.civics_not_imported:
        print(
            f"  {summary.civics_not_imported} civics not imported: schema version 1 has no civic tree"
        )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "import-gameinfo":
        return _run_import(args)
    try:
        spec = _load(args.target)
        if args.command == "validate":
            ensure_valid(spec)
            start = ", ".join(available(spec, ())) or "none"
            print(f"{spec.id}: valid")
            print(f"  techs: {len(spec.techs)}; available at the start: {start}")
            print(f"  index fingerprint: {build_registry(spec).fingerprint}")
        elif args.command == "pace":
            if args.max_turns <= 0:
                raise _UsageError(f"--max_turns must be positive, got {args.max_turns}")
            _print_pace(pacing_report(ensure_valid(spec), args.max_turns))
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
    except (RulesetLoadError, RulesetHashMismatchError, BuiltinTableError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
