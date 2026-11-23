"""The measures of the expressivity census (GM-B11): pure functions, no I/O.

E1..E4 for one state, the aggregation and the stop rule, as fixed in
``kb/specs/20261123-expressivity-census-preregistration.md`` and its implementation
notes. ``scripts/census_search_expressivity.py`` is the command line around them and
re-exports every name here. Importing this module does not load torch; measuring a
state does (the simulator's search module is imported then).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from alpha_zero.bridge.microciv_proxy import SIM_MOUNTAIN, simulator_terrain
from alpha_zero.bridge.protocol import GameStateData

MIN_STATES, MIN_LATE_STATES, LATE_TURN, MIN_STATE_SCHEMA = 40, 10, 30, 2
STOP_BOARD, E1_THRESHOLD, E2_THRESHOLD = 8, 0.3, 0.5
REGISTERED_BOARDS = (8, 16)
METRICS = ("E1", "U", "E2", "E3", "E4")
WATER_TYPES = frozenset({"coast", "ocean"})
#: Offer kind -> command type; and the types that move or use a unit (E2: skip does not).
_TARGETED = {"move": "unit_move", "attack": "unit_attack", "ranged_attack": "unit_ranged_attack"}
_UNTARGETED = {"skip": "unit_skip", "found_city": "unit_found_city"}
_USES_UNIT = frozenset({*_TARGETED.values(), "unit_found_city"})
#: Turn bands of the exploratory addendum: label -> (first turn, last turn).
TURN_BANDS = {"up_to_turn_10": (None, 10), "turn_11_to_30": (11, 30), "after_turn_30": (31, None)}

Key = tuple[Any, ...]
Cell = tuple[int, int]
Share = dict[str, Any]  #: ``{"value": float | None, "numerator": int, "denominator": int}``
_Agg = Mapping[str, Any] | None


class MalformedInputError(ValueError):
    """The input cannot be read as states."""


def offered_commands(legal_actions: Iterable[Any]) -> dict[Key, bool]:
    """The offered atomic commands (not ``end_turn``), duplicates removed: ``{key: handled}``.

    ``handled`` says whether the translator handles the command's kind at all (the upper
    bound U): a unit move, attack, ranged attack, skip, founding, or producing a unit.
    """
    offered: dict[Key, bool] = {}
    for offer in legal_actions:
        kind = getattr(offer, "action_type", None)
        unit_id, city_id = getattr(offer, "unit_id", None), getattr(offer, "city_id", None)
        targets = [(int(t[0]), int(t[1])) for t in getattr(offer, "targets", None) or []]
        items = list(getattr(offer, "items", None) or [])
        found: dict[Key, bool] = {}
        if kind in _TARGETED:
            found = {(_TARGETED[kind], unit_id, target): True for target in targets}
        elif kind in _UNTARGETED:
            found = {(_UNTARGETED[kind], unit_id): True}
        elif kind == "produce":
            kinds = getattr(offer, "item_types", None) or {}
            for item in items:  # a unit item as the translator reads it (``_unit_names``)
                is_unit = isinstance(item, str) and kinds.get(item, "unit") == "unit"
                found[("city_produce", city_id, item)] = is_unit
        elif kind != "end_turn":
            actor = unit_id if unit_id is not None else city_id
            found = {(kind, actor, detail): False for detail in [*items, *targets] or [None]}
        for key, handled in found.items():
            offered[key] = handled or offered.get(key, False)
    return offered


def command_key(command: Mapping[str, Any]) -> Key | None:
    """The atomic-command key of a translated live command; None for ``end_turn``."""
    kind = command.get("action_type")
    if kind == "end_turn":
        return None
    if kind in _TARGETED.values():
        target = command.get("target") or {}
        return (kind, command.get("unit_id"), (target.get("x"), target.get("y")))
    if kind in _UNTARGETED.values():
        return (kind, command.get("unit_id"))
    return (kind, command.get("city_id", command.get("unit_id")), command.get("item_id"))


def _share(numerator: int, denominator: int) -> Share:
    """A share with its counts; ``value`` is None when the denominator is zero."""
    value = numerator / denominator if denominator else None
    return {"value": value, "numerator": numerator, "denominator": denominator}


def e1_expressible(offered: Mapping[Key, bool], translated: Iterable[Key]) -> Share:
    """E1: offered atomic commands whose key is in T, with U and the split by kind."""
    in_t = set(translated)
    by_kind: dict[str, list[int]] = {}
    for key in offered:
        counts = by_kind.setdefault(str(key[0]), [0, 0])
        counts[0] += key in in_t
        counts[1] += 1
    result = _share(sum(c[0] for c in by_kind.values()), len(offered))
    result["upper_bound_u"] = _share(sum(offered.values()), len(offered))
    result["by_kind"] = {k: _share(c[0], c[1]) for k, c in sorted(by_kind.items())}
    return result


def e2_usable_units(own: Iterable[int], offered: Iterable[Key], translated: Iterable[Key]) -> Share:
    """E2: own unit ids with an offered move, attack or founding that have one in T."""
    offered_units = {k[1] for k in offered if k[0] in _USES_UNIT} & set(own)
    usable = {k[1] for k in translated if k[0] in _USES_UNIT} & offered_units
    return _share(len(usable), len(offered_units))


def e3_impassable_land(cells: Iterable[Cell], terrain: Mapping[Cell, Any]) -> Share:
    """E3: of the non-water ``cells`` (those a game tile reached), the share the stand-in
    marks impassable. ``terrain`` is the converted state's (cell -> resolved tile dict)."""
    tiles = [terrain[cell] for cell in set(cells) if isinstance(terrain.get(cell), Mapping)]
    land = [tile for tile in tiles if str(tile.get("type")).lower() not in WATER_TYPES]
    return _share(sum(simulator_terrain(tile) == SIM_MOUNTAIN for tile in land), len(land))


def e4_left_out(left_out: int, units: Iterable[Mapping[str, Any]], player: Any) -> Share:
    """E4: of the converted units of ``player`` (the acting one), those the stand-in leaves out."""
    return _share(int(left_out), sum(u.get("player") == player for u in units))


def parser_losses(raw: Mapping[str, Any], state: GameStateData) -> dict[str, int]:
    """What ``GameStateData.from_dict`` dropped from ``raw`` to make ``state``, by kind.

    The live parser is tolerant on purpose: it leaves out an invalid tile, unit, city,
    player, legal action, target or item and returns the rest. A census must not measure
    what is left as if it were the state. Only kinds with a loss are returned; ``{}`` for
    a state that was parsed whole. Targets and items are compared only inside legal
    actions that survive parsing (one dropped whole is counted once, as a legal action).
    """
    from alpha_zero.bridge.protocol import LegalAction

    def count(value: Any) -> int:
        return len(value) if isinstance(value, list | tuple) else 0

    raw_map = raw.get("map")
    raw_tiles = raw_map.get("tiles") if isinstance(raw_map, Mapping) else None
    summary = raw.get("player")  # with no players list the parser makes one from this
    players = count(raw.get("players")) or int(isinstance(summary, Mapping) and bool(summary))
    losses = {
        "tiles": count(raw_tiles) - len(state.map_data.tiles),
        "units": count(raw.get("units")) - len(state.units),
        "cities": count(raw.get("cities")) - len(state.cities),
        "players": players - len(state.players),
        "legal_actions": count(raw.get("legal_actions")) - len(state.legal_actions),
        "legal_action_targets": 0,
        "legal_action_items": 0,
    }
    raw_actions = raw.get("legal_actions")
    for action in raw_actions if isinstance(raw_actions, list | tuple) else []:
        try:
            parsed = LegalAction.from_dict(action)
        except Exception:
            parsed = None
        if parsed is None:  # dropped whole: already counted under legal_actions
            continue
        losses["legal_action_targets"] += count(action.get("targets")) - len(parsed.targets)
        items = action.get("items", action.get("item", []))  # the parser's own lookup
        offered = 0 if items is None else len(items) if isinstance(items, list) else 1
        losses["legal_action_items"] += offered - len(parsed.items)
    return {kind: lost for kind, lost in losses.items() if lost > 0}


def measure_state(state: GameStateData, board: int) -> dict[str, Any]:
    """E1..E4 of one state on one board; void when a translated command was not offered."""
    from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv
    from alpha_zero.bridge.search_translator import SearchTranslator
    from alpha_zero.bridge.state_converter import StateConverter

    converter = StateConverter(board_size=board, num_players=2, research_registry=False)
    cs = converter.convert(state)
    env = BridgeMCTSEnv.from_bridge_state(cs, seed=0)
    translator = SearchTranslator(cs, env.own_unit_ids(), env.own_city_ids(), allowed=None)
    decode = env.env.action_space_handler.decode_action
    _mask, commands, reasons = translator.root_mask(env.env.get_valid_actions(), decode)
    raw = cs["_raw_state"]
    offered = offered_commands(raw.legal_actions)
    translated = {key for key in map(command_key, commands.values()) if key is not None}
    stray = translated - offered.keys()
    if stray:
        return {"void": True, "void_reason": f"{len(stray)} translated command(s) not offered"}
    own_ids = [u.id for u in raw.units if u.owner_id == raw.player_id]
    cells = {converter.cell_of(tile.x, tile.y) for tile in raw.map_data.tiles}
    left_out = env.proxy_stats["own_units_not_represented"]
    has_city = any(c.owner_id == raw.player_id for c in raw.cities)
    return {
        "void": False,
        "E1": e1_expressible(offered, translated),
        "E2": e2_usable_units(own_ids, offered, translated),
        "E3": e3_impassable_land(cells, cs["terrain"]),
        "E4": e4_left_out(left_out, cs["units"], cs["current_player"]),
        "cells_with_a_tile": len(cells),
        # Exploratory addendum, not a registered quantity: can the losing ending fire at all?
        "loss_can_fire": not has_city and left_out == 0,
        "untranslated_reasons": dict(sorted(reasons.items())),
    }


def median_interval_ranks(n: int, confidence: float = 0.95) -> tuple[int, int] | None:
    """1-based ranks ``(k, n + 1 - k)`` of the order-statistic interval for the median: the
    largest ``k`` with ``P(Binomial(n, 1/2) < k) <= (1 - c) / 2``. (14, 27) at n = 40,
    (6, 15) at n = 20, None when there is none (n < 6 at 95%)."""
    tail, cumulative, k = (1.0 - confidence) / 2.0, 0.0, 0
    for i in range(n // 2 + 1):
        cumulative += math.comb(n, i) / 2**n
        if cumulative > tail:
            break
        k = i + 1
    return (k, n + 1 - k) if k else None


def aggregate(values: Sequence[float | None]) -> dict[str, Any]:
    """Median, IQR and the 95% order-statistic interval of the values that exist."""
    have = sorted(v for v in values if v is not None)
    ranks = median_interval_ranks(len(have))
    q25, median, q75 = (float(np.percentile(have, q)) if have else None for q in (25, 50, 75))
    out: dict[str, Any] = {"states_with_value": len(have), "median_ci95_ranks": ranks}
    out |= {"states_excluded": len(values) - len(have), "median": median, "q25": q25, "q75": q75}
    out["median_ci95"] = [have[ranks[0] - 1], have[ranks[1] - 1]] if ranks else None
    return out


def _condition(agg: _Agg, threshold: float) -> dict[str, Any]:
    median = agg.get("median") if agg else None
    ci = agg.get("median_ci95") if agg else None
    fires = None if median is None else median < threshold
    borderline = None if ci is None else ci[0] < threshold <= ci[1]
    out = {"median": median, "threshold": threshold, "fires": fires, "median_ci95": ci}
    return {**out, "borderline": borderline}


def stop_rule_verdict(e1: _Agg, e2: _Agg, u: _Agg = None) -> dict[str, Any]:
    """The registered stop rule from the board-8 aggregates of E1, E2 (and U): ``stop`` when
    median E1 < 0.3 or median E2 < 0.5; ``undetermined`` when neither fires and a median is
    missing; ``borderline`` when a 95% interval for a median holds its threshold."""
    c1, c2 = _condition(e1, E1_THRESHOLD), _condition(e2, E2_THRESHOLD)
    fires = [c1["fires"], c2["fires"]]
    u_median = u.get("median") if u else None
    return {
        "board": STOP_BOARD,
        "E1": c1,
        "E2": c2,
        "verdict": "stop" if any(fires) else "undetermined" if None in fires else "no_stop",
        "borderline": bool(c1["borderline"] or c2["borderline"]),
        "median_u": u_median,
        "e1_fires_from_the_offer_mix_alone": None if u_median is None else u_median < E1_THRESHOLD,
    }


def _value(record: Mapping[str, Any], board: int, metric: str) -> float | None:
    cell = record["boards"][str(board)]  # of a state that is not void
    share = cell["E1"]["upper_bound_u"] if metric == "U" else cell[metric]
    return None if share["value"] is None else float(share["value"])


def aggregate_board(records: Sequence[Mapping[str, Any]], board: int) -> dict[str, Any]:
    """Every registered figure for one board over the sampled states."""
    usable = [r for r in records if not r["boards"][str(board)].get("void")]
    late = [r for r in usable if r["turn"] > LATE_TURN]
    sources = sorted({r["source"] for r in usable})
    out: dict[str, Any] = {"states": len(usable), "states_void": len(records) - len(usable)}
    for m in METRICS:
        per = {s: [_value(r, board, m) for r in usable if r["source"] == s] for s in sources}
        out[m] = {
            "all": aggregate([_value(r, board, m) for r in usable]),
            "after_turn_30": aggregate([_value(r, board, m) for r in late]),
            "per_source_median": {str(s): aggregate(v)["median"] for s, v in per.items()},
        }
    kinds: dict[str, list[int]] = {}
    for record in usable:
        for kind, share in record["boards"][str(board)]["E1"]["by_kind"].items():
            pooled = kinds.setdefault(kind, [0, 0])
            pooled[0] += share["numerator"]
            pooled[1] += share["denominator"]
    out["E1_by_kind_pooled"] = {k: _share(c[0], c[1]) for k, c in sorted(kinds.items())}
    return out


def first_state_per_turn(messages: Iterable[Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The first complete state of each game turn (the fixed sampling), and counts: states
    read, without a turn number (not sampled), and times the turn went back (a reload or a
    second game in one source; no turn is sampled twice)."""
    kept: dict[int, dict[str, Any]] = {}
    counts = {"states_read": 0, "states_without_turn": 0, "turn_went_back": 0}
    last: int | None = None
    for message in messages:
        if not isinstance(message, dict):
            raise MalformedInputError(f"a state is a {type(message).__name__}, not an object")
        state = message["state"] if isinstance(message.get("state"), dict) else message
        counts["states_read"] += 1
        turn = state.get("turn_id", state.get("turn"))
        if not isinstance(turn, int) or isinstance(turn, bool):
            counts["states_without_turn"] += 1
            continue
        counts["turn_went_back"] += last is not None and turn < last
        last = turn
        kept.setdefault(turn, state)
    return list(kept.values()), counts


def loss_can_fire_by_band(records: Sequence[Mapping[str, Any]], board: int = STOP_BOARD) -> Share:
    """Exploratory addendum: states where the losing ending can fire, by turn band.

    It can fire when the player has no city and no own unit is left out of the stand-in.
    Not part of the registered census; there is no verdict. Void states are left out.
    """
    out: Share = {}
    for label, (first, last) in TURN_BANDS.items():
        cells = [
            r["boards"][str(board)]
            for r in records
            if (first is None or r["turn"] >= first) and (last is None or r["turn"] <= last)
        ]
        cells = [c for c in cells if not c.get("void")]
        out[label] = _share(sum(bool(c.get("loss_can_fire")) for c in cells), len(cells))
    return out
