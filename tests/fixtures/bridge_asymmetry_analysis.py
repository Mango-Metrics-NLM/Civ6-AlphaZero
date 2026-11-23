"""Turn the probe's per-search rows into the verdict and every number behind it."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tests.fixtures.bridge_asymmetry_probe import ARMS, INIT_SEEDS, all_roots, measure_chunk
from tests.fixtures.bridge_asymmetry_stats import (
    classify,
    cluster_bootstrap,
    sign_test_p,
)

BOOT_SEED = 20261123
MAIN_FAMILIES = ("R1", "R2", "R3", "R4")


def _key(row: dict[str, Any]) -> tuple[str, int, int]:
    return row["root"], row["init"], row["seed"]


def _by_arm(rows: list[dict[str, Any]]) -> dict[str, dict[tuple[str, int, int], dict[str, Any]]]:
    out: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {a: {} for a in ARMS}
    for row in rows:
        out[row["arm"]][_key(row)] = row
    return out


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(r[field]) for r in rows) / len(rows) if rows else float("nan")


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(r[field]) for r in rows) / len(rows) if rows else float("nan")


def agreement(
    left: dict[Any, dict[str, Any]], right: dict[Any, dict[str, Any]], pick: Any
) -> float:
    """Share of paired searches where both arms chose the same command."""
    keys = [k for k in left if k in right and pick(left[k])]
    return sum(left[k]["command"] == right[k]["command"] for k in keys) / len(keys)


def analyze(rows: list[dict[str, Any]], wall_seconds: float = 0.0) -> dict[str, Any]:
    arms = _by_arm(rows)
    a, b, c = arms["a"], arms["b"], arms["c"]

    def main(row: dict[str, Any]) -> bool:
        return row["family"] in MAIN_FAMILIES and not row["own_city"]

    # Rates and means per arm (main roots), plus the in-tree counts.
    per_arm: dict[str, Any] = {}
    for name, table in arms.items():
        sel = [r for r in table.values() if main(r)]
        per_arm[name] = {
            "n": len(sel),
            "contact_rate": _rate(sel, "contact"),
            "retreat_rate": _rate(sel, "retreat"),
            "neutral_rate": 1.0 - _rate(sel, "contact") - _rate(sel, "retreat"),
            "attack_rate": sum(r["kind"] == "attack" for r in sel) / len(sel),
            "mean_root_value": _mean(sel, "root_value"),
            "enemy_eliminated_lines": sum(r["enemy_eliminated"] for r in sel),
            "own_eliminated_lines": sum(r["own_eliminated"] for r in sel),
            "elimination_suppressed": sum(r["elimination_suppressed"] for r in sel),
        }

    # Main comparison: (a) vs (c), paired by (root, init, seed).
    pairs = [(k, a[k], c[k]) for k in sorted(a) if k in c and main(a[k])]
    diffs = {k: int(rc["contact"]) - int(ra["contact"]) for k, ra, rc in pairs}
    only_c = sum(d > 0 for d in diffs.values())  # contact in (c) only
    only_a = sum(d < 0 for d in diffs.values())
    clusters: dict[Any, list[float]] = {}
    for (root, init, _), d in diffs.items():
        clusters.setdefault((root, init), []).append(float(d))
    mean, lo, hi = cluster_bootstrap(clusters, seed=BOOT_SEED)
    p = sign_test_p(only_c, only_a)
    per_init = {
        str(i): sum(v for (_, ii, _), v in diffs.items() if ii == i)
        / max(1, sum(1 for (_, ii, _) in diffs if ii == i))
        for i in INIT_SEEDS
    }
    value_drop = per_arm["c"]["mean_root_value"] - per_arm["a"]["mean_root_value"]
    own_elim_a = per_arm["a"]["own_eliminated_lines"]
    verdict = classify(
        diff=mean,
        ci=(lo, hi),
        p=p,
        per_init_diffs=list(per_init.values()),
        own_elimination_in_a=own_elim_a,
    )

    # Positive control: (a) vs (b) on R1 and R2.
    def ctl(row: dict[str, Any]) -> bool:
        return row["family"] in ("R1", "R2")

    pa = [r for r in a.values() if ctl(r)]
    pb = [r for r in b.values() if ctl(r)]
    positive = {
        "root_value_b_minus_a": _mean(pb, "root_value") - _mean(pa, "root_value"),
        "attacks_a": sum(r["kind"] == "attack" for r in pa),
        "attacks_b": sum(r["kind"] == "attack" for r in pb),
        "n": len(pa),
    }
    negative = {
        "own_city_a_vs_c_agreement": agreement(a, c, lambda r: r["own_city"]),
        "r5_a_vs_b_agreement": agreement(a, b, lambda r: r["family"] == "R5"),
    }
    n_actions = {r["root"]: r["n_actions"] for r in a.values()}
    return {
        "verdict": verdict,
        "vacuous": own_elim_a <= 0,
        "per_arm": per_arm,
        "paired_diff_c_minus_a": {
            "mean": mean,
            "ci95": [lo, hi],
            "discordant_contact_only_in_c": only_c,
            "discordant_contact_only_in_a": only_a,
            "sign_test_p": p,
            "per_init": per_init,
            "n_pairs": len(pairs),
            "n_clusters": len(clusters),
            "root_value_c_minus_a": value_drop,
        },
        "positive_control": positive,
        "negative_controls": negative,
        "action_counts_per_root": n_actions,
        "wall_seconds": wall_seconds,
    }


def run_probe(out: Path | None = None) -> dict[str, Any]:
    """The whole probe (2,430 searches, zero games); callable from a script."""
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for init in INIT_SEEDS:
            rows += measure_chunk(arm, init, all_roots())
    result = analyze(rows, time.perf_counter() - start)
    if out is not None:
        out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result
