#!/usr/bin/env python3
"""Expressivity census of search-in-the-simulator, at zero games (GM-B11).

How much of what the live game offers can the bridge's search say at all, when it
plays in the MicroCiv stand-in seeded from a live state? E1..E4, the sampling, the
aggregation and the stop rule are fixed in the pre-registration (``SPEC`` below).

It plays **no game**, loads no network and runs no search: per state it converts the
state, seeds the simulator and reads the root mask (``game_budget`` price 0). Only
``Lua.log`` input meeting the registered minimum (40 states, 10 after turn 30, current
mod) is a census result; anything else only shows the computation runs, and says so.
Exit: 0 computed; 1 malformed input; 2 refused or usage; 3 below the minimum, nothing written.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Run as a script, sys.path[0] is scripts/: make this checkout win over an installed copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_zero.bridge.log_tailer import DROPPED_MISSING_PIECE, UNREADABLE_STATE, iter_log_states
from alpha_zero.bridge.protocol import GameStateData
from scripts.census_measures import (
    E1_THRESHOLD,
    E2_THRESHOLD,
    LATE_TURN,
    METRICS,
    MIN_LATE_STATES,
    MIN_STATE_SCHEMA,
    MIN_STATES,
    REGISTERED_BOARDS,
    STOP_BOARD,
    MalformedInputError,
    _share,
    aggregate,
    aggregate_board,
    command_key,
    e1_expressible,
    e2_usable_units,
    e3_impassable_land,
    e4_left_out,
    first_state_per_turn,
    loss_can_fire_by_band,
    malformed_fields,
    measure_state,
    median_interval_ranks,
    offered_commands,
    parser_losses,
    stop_rule_verdict,
)

#: The measures are importable from here too (the tests and the memo read one module).
__all__ = [
    "MalformedInputError",
    "aggregate",
    "aggregate_board",
    "census",
    "command_key",
    "describe_input",
    "e1_expressible",
    "e2_usable_units",
    "e3_impassable_land",
    "e4_left_out",
    "evidence_block",
    "first_state_per_turn",
    "loss_can_fire_by_band",
    "main",
    "malformed_fields",
    "measure_state",
    "median_interval_ranks",
    "not_a_census_reason",
    "offered_commands",
    "parser_losses",
    "registered_stop_rule",
    "render_summary",
    "stop_rule_verdict",
]

RUNNER = "census_search_expressivity"
JSON_NAME = "census.json"
SPEC = "kb/specs/20261123-expressivity-census-preregistration.md"
EXIT_OK, EXIT_MALFORMED, EXIT_REFUSED, EXIT_BELOW_MINIMUM = 0, 1, 2, 3
#: Per-source counts of log states that were never offered for sampling.
LOG_COUNTS = (UNREADABLE_STATE, DROPPED_MISSING_PIECE)


def _void_record(
    index: int, raw: dict[str, Any], boards: Sequence[int], reason: str, dropped: dict[str, int]
) -> dict[str, Any]:
    void = {"void": True, "void_reason": reason[:300]}
    record = {"source": index, "turn": raw.get("turn_id", raw.get("turn"))}
    record["boards"] = {str(b): dict(void) for b in boards}
    return {**record, "parser_dropped": dropped} if dropped else record


def _record(index: int, raw: dict[str, Any], boards: Sequence[int]) -> dict[str, Any]:
    """One sampled state's values on every board.

    A state that cannot be parsed or measured, or that is void on any board, is void on
    every board (both boards then use the same states). So is a state with a field of the
    wrong type (checked before the parser is asked), and one the tolerant parser had to
    repair by dropping part of it: what is left is not the state. It is never replaced.
    """
    per_board: dict[str, dict[str, Any]] = {}
    dropped: dict[str, int] = {}
    if malformed := malformed_fields(raw):  # never parsed: it would read them as empty
        return _void_record(index, raw, boards, f"malformed field(s): {', '.join(malformed)}", {})
    try:
        state = GameStateData.from_dict(raw)
        dropped = parser_losses(raw, state)
        if dropped:
            lost = ", ".join(f"{kind}: {n}" for kind, n in dropped.items())
            reason: str | None = f"the parser dropped part of the state ({lost})"
        else:
            per_board = {str(b): measure_state(state, b) for b in boards}
            bad = [f"board {b}: {v.get('void_reason')}" for b, v in per_board.items() if v["void"]]
            reason = "; ".join(bad) or None
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    if reason is not None:
        return _void_record(index, raw, boards, reason, dropped)
    tiles = state.map_data.tiles
    return {
        "source": index,
        "turn": state.turn_id,
        "map_size": [state.map_data.width, state.map_data.height],
        "civilization": state.current_player_civ,
        "schema_version": state.schema_version,
        "observes_others": state.observes_others,
        "tiles_exported": len(tiles),
        "tiles_revealed_share": _share(sum(t.is_revealed for t in tiles), len(tiles))["value"],
        "boards": per_board,
    }


def census(sources: Sequence[tuple[str, Path]], boards: Sequence[int]) -> dict[str, Any]:
    """Sample every source (``("lua_log" | "states_json", path)``) and compute everything."""
    records: list[dict[str, Any]] = []
    described: list[dict[str, Any]] = []
    for index, (kind, path) in enumerate(sources):
        # States of a log that lost a piece or do not parse: never used, always counted.
        lost = dict.fromkeys(LOG_COUNTS, 0)
        try:
            found: Any
            if kind == "lua_log":
                found = iter_log_states(path, lost)
            else:
                found = json.loads(path.read_bytes())
            if not isinstance(found, Iterator | list):
                raise MalformedInputError("expected a JSON list of state objects")
            sampled, counts = first_state_per_turn(found)
        except (OSError, ValueError) as exc:
            raise MalformedInputError(f"{path}: {exc}") from exc
        described.append({"source": index, "kind": kind, "path": str(path), **counts, **lost})
        records += [_record(index, raw, boards) for raw in sampled]
    aggregates = {str(b): aggregate_board(records, b) for b in boards}
    return {"sources": described, "states": records, "aggregates": aggregates}


def describe_input(
    result: Mapping[str, Any],
    boards: Sequence[int],
    mod_version: str | None = None,
    played_by: str | None = None,
) -> dict[str, Any]:
    """What the input is, and every condition it misses for being a census result.

    A census result needs: only ``Lua.log`` sources, the registered minimum met by
    non-void states, no state from an older mod, boards 8 and 16 both computed, and the
    mod version and who played the turns recorded.
    """
    sources, states = result["sources"], result["states"]
    kinds = {s["kind"] for s in sources}
    kind = "mixed" if len(kinds) > 1 else next(iter(kinds))
    good = [r for r in states if not any(r["boards"][str(b)].get("void") for b in boards)]
    late = sum(r["turn"] > LATE_TURN for r in good)
    old = sum(r["schema_version"] < MIN_STATE_SCHEMA or r["observes_others"] is None for r in good)
    meets = len(good) >= MIN_STATES and late >= MIN_LATE_STATES
    reasons = []
    if kind != "lua_log":
        reasons.append(f"input is {kind}, not only a real game's Lua.log")
    if not meets:
        reasons.append(
            f"{len(good)} states, {late} after turn {LATE_TURN} "
            f"(registered minimum: {MIN_STATES} and {MIN_LATE_STATES})"
        )
    if old:
        reasons.append(f"{old} state(s) from an older mod")
    if not set(REGISTERED_BOARDS) <= set(boards):
        reasons.append("boards 8 and 16 were not both computed")
    if not (mod_version or "").strip():
        reasons.append("--mod-version was not given")
    if not (played_by or "").strip():
        reasons.append("--played-by was not given")
    return {
        "input_kind": kind,
        "sampling": "first complete state of each game turn, per source",
        "states_sampled": len(states),
        "states_void": len(states) - len(good),
        "states_void_parser_dropped": sum("parser_dropped" in r for r in states),
        **{key: sum(s[key] for s in sources) for key in LOG_COUNTS},
        "turn_went_back": sum(s["turn_went_back"] for s in sources),
        "states_measured": len(good),
        "states_after_turn_30": late,
        "states_not_from_the_current_mod": old,
        "tiles_revealed_share": aggregate([r["tiles_revealed_share"] for r in good]),
        "boards": list(boards),
        "registered_minimum": {"states": MIN_STATES, "after_turn_30": MIN_LATE_STATES},
        "meets_registered_minimum": meets,
        "not_a_census_reasons": reasons,
        "is_census_result": not reasons,
    }


def not_a_census_reason(info: Mapping[str, Any]) -> str:
    """Every condition the input misses for being a census result (empty when it is one)."""
    return "; ".join(info["not_a_census_reasons"])


def registered_stop_rule(aggregates: Mapping[str, Any], is_census: bool) -> dict[str, Any]:
    """The stop-rule field of the output.

    The registered words (``stop``, ``no_stop``, ``undetermined``) appear under ``verdict``
    only for a census result. On any other input ``verdict`` is ``not_decided`` and what
    the input would give is kept under a name that says it is unregistered.
    """
    at = aggregates.get(str(STOP_BOARD))
    if at is None:
        stop: dict[str, Any] = {"board": STOP_BOARD, "verdict": None, "borderline": False}
        stop["not_computed"] = f"board {STOP_BOARD} was not requested"
    else:
        stop = stop_rule_verdict(at["E1"]["all"], at["E2"]["all"], at["U"]["all"])
    stop["decided_by_this_input"] = is_census
    if not is_census:
        stop["verdict_on_this_input_unregistered"] = stop["verdict"]
        stop["verdict"] = "not_decided"
    return stop


def evidence_block(info: Mapping[str, Any], argv: Sequence[str], sha: str | None) -> dict[str, Any]:
    """The ``docs/evidence/SCHEMA.md`` block; no game is played, so ``games_total`` is 0."""
    real = bool(info["is_census_result"])
    claim = f"expressivity (E1..E4) of search-in-the-simulator, as registered in {SPEC}"
    return {
        "schema_version": 1,
        "git_sha": sha,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "search_epoch": "post-2026-09-04",
        "seed": 0,
        "command": shlex.join(["python", "scripts/census_search_expressivity.py", *argv]),
        "hardware": {
            "cpu_model": platform.processor() or None,
            "gpu_model": None,
            "logical_cores": os.cpu_count(),
            "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        },
        "python": platform.python_version(),
        "torch": getattr(sys.modules.get("torch"), "__version__", None),
        "numpy": np.__version__,
        "games_total": 0,
        "claims": [claim if real else "the four computations ran on this input, nothing more"],
        "non_claims": [
            "not evidence of playing strength or of learning; no game, network or search",
            "expressivity is necessary, not sufficient: it says nothing about move quality",
            "bound to this commit: a change to the converter, proxy or translator needs a re-run",
            "the exploratory block is not part of the registered census and has no verdict",
            *([] if real else ["NOT a census result: " + not_a_census_reason(info)]),
        ],
    }


def _metric_line(board: str, metric: str, a: Mapping[str, Any]) -> str:
    if a["median"] is None:
        return f"  board {board} {metric}: no value ({a['states_excluded']} excluded)"
    ci = a["median_ci95"]
    ci_text = f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "n/a"
    return (
        f"  board {board} {metric}: median {a['median']:.3f}, IQR {a['q25']:.3f}-{a['q75']:.3f}, "
        f"95% {ci_text}, n={a['states_with_value']}, excluded={a['states_excluded']}"
    )


def render_summary(payload: Mapping[str, Any]) -> str:
    """The short human summary. On input that is not a census, that is said before any number."""
    info, stop = payload["input"], payload["stop_rule"]
    real = info["is_census_result"]
    lines = ["Expressivity census of search-in-the-simulator (plays 0 games)"]
    if not real:
        lines.append(f"NOT a census result: {not_a_census_reason(info)}.")
        lines.append("The stop rule is NOT decided by this input; no number below is an estimate.")
    real_log = "a Lua.log" if info["input_kind"] == "lua_log" else "NOT only a real Lua.log"
    lines += [
        f"Input: {len(payload['sources'])} source(s), kind {info['input_kind']} ({real_log})",
        f"States: {info['states_sampled']} sampled ({info['sampling']}), {info['states_void']} "
        f"void ({info['states_void_parser_dropped']} of them repaired by the parser, which "
        f"dropped part of them), {info['states_after_turn_30']} after turn {LATE_TURN}. Not sampled: "
        f"{info[UNREADABLE_STATE]} unreadable, {info[DROPPED_MISSING_PIECE]} dropped for a "
        f"missing piece. Registered minimum ({MIN_STATES} states, {MIN_LATE_STATES} after turn "
        f"{LATE_TURN}): " + ("met" if info["meets_registered_minimum"] else "NOT met"),
    ]
    for board, agg in payload["aggregates"].items():
        lines += [_metric_line(board, metric, agg[metric]["all"]) for metric in METRICS]
    mark = ", borderline" if stop["borderline"] else ""
    if real:
        rule = f"median E1 < {E1_THRESHOLD} or median E2 < {E2_THRESHOLD}"
        lines.append(f"Stop rule at board {STOP_BOARD} ({rule}): {stop['verdict']}{mark}")
        lines.append("This input meets the registration: the stop rule is decided as shown.")
    else:
        would = stop["verdict_on_this_input_unregistered"]
        shown = f"this input would give {would}{mark}" if would else stop["not_computed"]
        lines.append(f"Stop rule: NOT DECIDED ({shown})")
    bands = (payload["exploratory"] or {}).get("loss_can_fire")
    if bands:
        text = ", ".join(
            f"{b['numerator']}/{b['denominator']} {label.replace('_', ' ')}"
            for label, b in bands.items()
        )
        lines.append(
            f"Exploratory, not registered, no verdict: the losing ending can fire in {text}"
        )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    from alpha_zero.utils.game_budget import add_budget_arguments, enforce_cli

    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--log", type=Path, action="append", default=[], help="a Civ6 Lua.log")
    p.add_argument("--states-json", type=Path, action="append", default=[], help="raw state list")
    p.add_argument("--boards", type=int, nargs="+", default=[8, 16], help="board sizes")
    p.add_argument("--out", type=Path, required=True, help=f"directory for {JSON_NAME}")
    small = "compute below the registered minimum; the output is then not a census result"
    p.add_argument("--allow-small", action="store_true", help=small)
    attest = "; required for the output to count as a census result"
    p.add_argument("--mod-version", default=None, help="the mod version" + attest)
    p.add_argument("--played-by", default=None, help="who played the turns" + attest)
    add_budget_arguments(p)
    args = p.parse_args(argv)
    enforce_cli(RUNNER, {"states": 0}, args)  # priced like every runner; the price is 0
    return args


def _given_twice(paths: Sequence[Path]) -> Path | None:
    """The first file two inputs name, or None.

    Paths are resolved (symbolic links followed); an existing file is also known by its
    device and inode, so two hard links to one file are one file too.
    """
    seen: set[Any] = set()
    for path in paths:
        resolved = path.resolve()
        try:
            info = resolved.stat()
            keys = {resolved, (info.st_dev, info.st_ino)} if info.st_ino else {resolved}
        except OSError:  # missing: refused as missing right after this
            keys = {resolved}
        if keys & seen:
            return resolved
        seen |= keys
    return None


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw)
    from alpha_zero.utils.checkpoint import read_git_sha

    sources = [("lua_log", p) for p in args.log] + [("states_json", p) for p in args.states_json]
    boards = list(dict.fromkeys(args.boards))
    target = args.out / JSON_NAME
    sha = None
    refusal = None
    if not sources:
        refusal = "give at least one --log or --states-json"
    elif min(boards) <= 0:
        refusal = "--boards must be positive"
    elif twice := _given_twice([p for _, p in sources]):  # its turns would count twice
        refusal = f"one file given more than once (--log/--states-json): {twice}"
    elif missing := [str(p) for _, p in sources if not p.is_file()]:
        refusal = f"no such file: {', '.join(missing)}"
    elif target.exists():
        refusal = f"{target} exists and is never overwritten"
    elif (sha := read_git_sha()) is None:  # a census is bound to the commit it ran at
        refusal = "the commit cannot be resolved: set CIV6AZ_GIT_SHA to the full 40-hex sha"
    if refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        result = census(sources, boards)
        if not result["states"]:
            raise MalformedInputError("no complete state with a turn number was found")
    except MalformedInputError as exc:
        print(f"MALFORMED INPUT: {exc}", file=sys.stderr)
        return EXIT_MALFORMED
    info = describe_input(result, boards, args.mod_version, args.played_by)
    if not info["meets_registered_minimum"] and not args.allow_small:
        why = not_a_census_reason(info) + ". Nothing was written (see --allow-small)"
        print(f"BELOW THE REGISTERED MINIMUM: {why}.", file=sys.stderr)
        return EXIT_BELOW_MINIMUM
    measured = [r for r in result["states"] if "map_size" in r]
    exploratory = None
    if STOP_BOARD in boards:
        exploratory = {
            "not_part_of_the_registered_census": True,
            "what": "states where the losing ending can fire at all (the player has no city "
            "and no own unit is left out of the stand-in), by turn band; no verdict",
            "board": STOP_BOARD,
            "loss_can_fire": loss_can_fire_by_band(result["states"]),
        }
    payload: dict[str, Any] = {
        "evidence": evidence_block(info, raw, sha),
        "specification": SPEC,
        "input": info,
        "recorded": {
            "mod_version": args.mod_version,
            "played_by": args.played_by,
            "map_sizes": sorted({tuple(r["map_size"]) for r in measured}),
            "civilizations": sorted({r["civilization"] for r in measured}),
        },
        "stop_rule": registered_stop_rule(result["aggregates"], info["is_census_result"]),
        "exploratory": exploratory,
        **result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8") as handle:  # "x": never overwrite
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(render_summary(payload) + f"\nWrote {target}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
