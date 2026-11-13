"""Tests for ``scripts/probe_target_sharpness.py`` (LF-A5/A12 probe).

Tiny settings (board 4, a handful of positions) so the module stays fast: the
point is the output contract and determinism, not the numbers.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_target_sharpness.py"
TINY: dict[str, Any] = {"positions": 3, "board_size": 4, "max_turns": 5, "k": 4, "n": 8}
CELL_KEYS = {
    "search",
    "target",
    "entropy_nats",
    "max_p",
    "support",
    "share_max_p_above_greedy_threshold",
    "share_above_threshold_in_temperature_phase",
    "share_max_p_above_0_5",
    "argmax_agrees_with_current",
}


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("probe_target_sharpness", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod() -> Any:
    return _load()


@pytest.fixture(scope="module")
def result(mod: Any) -> dict[str, Any]:
    return mod.run_probes(seeds=[7, 11], **TINY)


def test_schema(mod: Any, result: dict[str, Any]) -> None:
    assert result["probe"] == "probe_target_sharpness"
    assert result["games_played"] == 0
    assert [run["seed"] for run in result["per_seed"]] == [7, 11]
    assert result["pooled"]["positions"] == 2 * TINY["positions"]
    pooled = result["pooled"]["cells"]
    for cells, keys in [
        (pooled, CELL_KEYS | {"share_across_seeds"}),
        *((run["cells"], CELL_KEYS) for run in result["per_seed"]),
    ]:
        assert set(cells) == {name for name, _, _ in mod.CELLS}
        for cell in cells.values():
            assert set(cell) == keys
            assert 0.0 <= cell["share_max_p_above_greedy_threshold"] <= 1.0
            assert 0.0 < cell["max_p"]["mean"] <= 1.0
            assert cell["entropy_nats"]["mean"] >= 0.0
    assert pooled["current"]["argmax_agrees_with_current"] == 1.0
    spread = pooled["current"]["share_across_seeds"]
    assert spread["min"] <= spread["mean"] <= spread["max"] and spread["se"] >= 0.0
    for run in result["per_seed"]:
        assert len(run["positions"]) == TINY["positions"]
        assert run["current_visited_q_spread"]["mean"] >= 0.0
    assert result["settings"]["current"]["greedy_threshold"] == pytest.approx(0.99)


def test_same_seed_is_identical(mod: Any, result: dict[str, Any]) -> None:
    again = mod.run_probes(seeds=[7, 11], **TINY)
    assert json.dumps(again, sort_keys=True) == json.dumps(result, sort_keys=True)


def test_different_seed_differs(mod: Any, result: dict[str, Any]) -> None:
    other = mod.run_probe(seed=8, **TINY)
    assert json.dumps(other["positions"]) != json.dumps(result["per_seed"][0]["positions"])


def test_duplicate_seeds_are_refused(mod: Any) -> None:
    with pytest.raises(ValueError, match="distinct"):
        mod.run_probes(seeds=[7, 7], **TINY)


def test_paper_target_spans_every_legal_action(mod: Any) -> None:
    """The paper's target is a softmax over all legal actions, not the visited ones."""
    node = mod.GumbelMCTSNode(to_play=0)
    node.expand(np.full(6, 1 / 6, dtype=np.float32), np.array([1, 1, 1, 0, 1, 1]))
    node.children[0].visit_count, node.children[0].value_sum = 2, 1.0
    node.children[0].to_play = 0
    target = mod.paper_target(node, 0.0, 6, c_visit=50.0, c_scale=1.0, normalisation="minmax")
    assert target.sum() == pytest.approx(1.0)
    assert (target > 0).sum() == 5 and target[3] == 0.0
    assert int(target.argmax()) == 0  # the only visited action, with the best Q


def test_sigma_is_zero_when_all_q_equal(mod: Any) -> None:
    """Min-max over a constant vector must not divide by zero or move the prior."""
    sigma = mod.paper_sigma_completed(
        priors=np.array([0.5, 0.5]),
        visits=np.array([1.0, 1.0]),
        q_values=np.array([0.2, 0.2]),
        raw_value=0.2,
        c_visit=50.0,
        c_scale=1.0,
    )
    assert np.allclose(sigma, 0.0)


def test_cli_writes_json_and_report(mod: Any, tmp_path: Path) -> None:
    argv = [
        "--positions",
        "2",
        "--board_size",
        "4",
        "--max_turns",
        "5",
        "--k",
        "4",
        "--n",
        "8",
        "--seeds",
        "3",
        "4",
        "--out_dir",
        str(tmp_path),
    ]
    assert mod.main(argv) == 0
    data = json.loads((tmp_path / mod.JSON_NAME).read_text(encoding="utf-8"))
    evidence = data["evidence"]
    assert evidence["search_epoch"] == "post-2026-09-04"
    assert evidence["games_total"] == 0 and evidence["seed"] == 3
    assert evidence["seeds"] == [3, 4]
    assert evidence["claims"] and evidence["non_claims"]
    report = (tmp_path / mod.REPORT_NAME).read_text(encoding="utf-8")
    assert "current_search_paper_affine_target" in report and "mctx_minmax_c1" in report
    # Per-position rows are left out of the committed JSON unless asked for.
    assert all("positions" not in run for run in data["per_seed"])


def test_priced_at_zero_games() -> None:
    from alpha_zero.utils.game_budget import priced_runners, worst_case_games

    assert "probe_target_sharpness" in priced_runners()
    assert worst_case_games("probe_target_sharpness", {"positions": 10_000}) == 0


# -- the two equivalences the memo rests on (T12 review M3) ------------------------

PRIORS = np.array([0.5, 0.3, 0.2])
VISITS = np.array([2.0, 1.0, 0.0])
QS = np.array([0.4, -0.2, 0.0])  # the third is unvisited; its q is ignored
RAW = 0.1
# v_mix = (raw + sum N * weighted q) / (sum N + 1), weighted over visited by prior:
#   weighted q = (0.5 * 0.4 + 0.3 * -0.2) / 0.8 = 0.175; v_mix = (0.1 + 3 * 0.175) / 4 = 0.15625
COMPLETED = np.array([0.4, -0.2, 0.15625])


def test_sigma_matches_mctx_qtransform_by_hand(mod: Any) -> None:
    """mctx ``qtransform_completed_by_mix_value`` defaults: min-max, value_scale 0.1."""
    sigma = mod.paper_sigma_completed(
        priors=PRIORS, visits=VISITS, q_values=QS, raw_value=RAW, c_visit=50.0, c_scale=0.1
    )
    normalised = (COMPLETED - (-0.2)) / 0.6  # [1, 0, 0.59375]
    assert sigma == pytest.approx((50.0 + 2.0) * 0.1 * normalised)


def test_sigma_affine_reading_by_hand(mod: Any) -> None:
    sigma = mod.paper_sigma_completed(
        priors=PRIORS,
        visits=VISITS,
        q_values=QS,
        raw_value=RAW,
        c_visit=50.0,
        c_scale=1.0,
        normalisation="affine",
    )
    assert sigma == pytest.approx(52.0 * (COMPLETED + 1.0) / 2.0)


def test_unknown_normalisation_is_refused(mod: Any) -> None:
    with pytest.raises(ValueError, match="normalisation"):
        mod.paper_sigma_completed(
            priors=PRIORS,
            visits=VISITS,
            q_values=QS,
            raw_value=RAW,
            c_visit=50,
            c_scale=1,
            normalisation="zscore",
        )


def probe_current_vs_production(mod: Any, positions: int = 4) -> list[float]:
    """Max abs difference between the probe's `current` target and GumbelMCTS.search."""
    from alpha_zero.core.mcts.gumbel import GumbelMCTS

    network = mod.build_network(4, 5, 7, "cpu")
    base = mod.search_config(4, 5, 4, 8, "cpu", 7)
    gaps = []
    for index in range(positions):
        env, _ = mod.sample_position(4, 5, 7, index)
        seed = mod.derive_seed(7, mod.STREAM_SEARCH, index)
        config = mod.dataclasses.replace(base, seed=seed)
        produced, _ = GumbelMCTS(network, config).search(env)
        probe = mod.ProbeGumbelMCTS(network, config, halving=mod.HALVING_CURRENT)
        probe.search(env)
        assert probe.last_root is not None
        mine = mod.repo_target(probe.last_root, int(env.action_space.n), probe.config)
        if mine.sum() <= 0.0:
            mine[probe.last_choice] = 1.0
        gaps.append(float(np.abs(mine - produced).max()))
    return gaps


def test_probe_current_target_is_productions(mod: Any) -> None:
    """Every number in the memo describes `current`; it must be the production search."""
    assert max(probe_current_vs_production(mod)) < 1e-6


def test_equivalence_check_has_teeth(mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: a probe target built with a different c_scale must be caught."""
    real = mod.repo_target

    def drifted(root: Any, n: int, config: Any) -> np.ndarray:
        return real(root, n, mod.dataclasses.replace(config, c_scale=config.c_scale * 2))

    monkeypatch.setattr(mod, "repo_target", drifted)
    assert max(probe_current_vs_production(mod)) > 1e-6
