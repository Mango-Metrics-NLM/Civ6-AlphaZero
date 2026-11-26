"""Tests for ``scripts/calibrate_throughput.py``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "calibrate_throughput.py"


def _load():
    spec = importlib.util.spec_from_file_location("calibrate_throughput", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reports_positive_rates_on_tiny_board() -> None:
    mod = _load()
    res = mod.calibrate(
        board_size=4, max_turns=5, mcts="gumbel", k=4, n=4, device="cpu", games=1, seed=1
    )
    assert res["ms_per_eval"] > 0
    assert res["actions_per_game"] > 0
    assert res["s_per_game"] > 0
    assert res["total_evals"] > 0


def test_rejects_unknown_mcts() -> None:
    mod = _load()
    with pytest.raises(ValueError):
        mod.calibrate(4, 5, "bogus", 4, 4, "cpu")


def test_cli_writes_json(tmp_path: Path) -> None:
    mod = _load()
    out = tmp_path / "c.json"
    rc = mod.main(
        [
            "--board_size",
            "4",
            "--max_turns",
            "5",
            "--k",
            "4",
            "--n",
            "4",
            "--games",
            "1",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert json.loads(out.read_text())["results"]["s_per_game"] > 0
