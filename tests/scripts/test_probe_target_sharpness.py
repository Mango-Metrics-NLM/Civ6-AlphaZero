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
    for cells in [result["pooled"]["cells"], *(run["cells"] for run in result["per_seed"])]:
        assert set(cells) == {name for name, _, _ in mod.CELLS}
        for cell in cells.values():
            assert set(cell) == CELL_KEYS
            assert 0.0 <= cell["share_max_p_above_greedy_threshold"] <= 1.0
            assert 0.0 < cell["max_p"]["mean"] <= 1.0
            assert cell["entropy_nats"]["mean"] >= 0.0
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
    target = mod.paper_target(node, 0.0, 6, c_visit=50.0, c_scale=1.0)
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
    assert "current_search_paper_target" in report


def test_priced_at_zero_games() -> None:
    from alpha_zero.utils.game_budget import priced_runners, worst_case_games

    assert "probe_target_sharpness" in priced_runners()
    assert worst_case_games("probe_target_sharpness", {"positions": 10_000}) == 0
