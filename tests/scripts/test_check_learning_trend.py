"""``scripts/check_learning_trend.py`` (LF-A2): descriptive trend, exit 0 / 1 / 2.

Fixtures are written in the formats the producers write: ``eval_<iter>.json`` as
``training_civ6.write_eval_json`` lays it out, ``iterations.jsonl`` lines as
``train`` appends them, and benchmark envelopes built from the benchmark's own
row builder so they pass :func:`~alpha_zero.evaluation.benchmark.validate_envelope`.

Negative controls: the 2-point fixture must exit 2 (VLP gate 4), a planted
arithmetic error must exit 1, and a planted torn line must exit 1.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha_zero.evaluation import benchmark as bench  # noqa: E402
from alpha_zero.evaluation.scoring import summarize  # noqa: E402
from scripts import check_learning_trend as clt  # noqa: E402


def _eval_file(directory: Path, iteration: int, wins: int, draws: int, losses: int) -> Path:
    result = dict(summarize(wins, draws, losses).to_dict())
    result["win_rate"] = wins / (wins + draws + losses)
    payload = {
        "eval_schema_version": 1,
        "iteration": iteration,
        "opponent": "random",
        "result": result,
    }
    path = directory / f"eval_{iteration}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record(
    iteration: int,
    *,
    share: float | None = 0.01,
    max_p: float | None = 0.3,
    generator: str = "network",
    error: bool = False,
    loss: float = 1.0,
) -> dict[str, Any]:
    prequential: dict[str, Any]
    if error:
        prequential = {"error": "RuntimeError('boom')", "generator": generator}
    else:
        prequential = {
            "generator": generator,
            "value_mse": 0.5,
            "target_share_above_greedy_threshold": share,
            "target_max_p_mean": max_p,
        }
    return {
        "run_id": "fixture",
        "iteration": iteration,
        "train": {"total_loss": loss, "policy_loss": loss / 2, "value_loss": loss / 2},
        "prequential": prequential,
        "eval": None,
        "checkpoint": None,
    }


def _write_records(directory: Path, records: list[dict[str, Any]]) -> Path:
    path = directory / "iterations.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _envelope(directory: Path, name: str, checkpoint: str, outcomes: list[str]) -> Path:
    """A schema-valid benchmark envelope whose candidate is ``checkpoint``."""
    n = len(outcomes)
    games = [{"pair": g // 2, "seat": g % 2, "result": r} for g, r in enumerate(outcomes)]
    protocol = bench.BenchmarkProtocol(n_games=n, num_simulations=24, gumbel_k=8)
    command = shlex.join(
        [
            "python",
            "scripts/benchmark_suite.py",
            "--checkpoint",
            f"E:\\runs\\a1\\ckpt\\{checkpoint}",
            "--opponent-checkpoint",
            "E:\\runs\\a1\\ckpt\\checkpoint_0.pt",
            "--n-games",
            str(n),
        ]
    )
    evidence = bench.collect_evidence(20261111, n, command, claims=["c"], non_claims=["n"])
    evidence["device"] = "cpu"
    evidence["search"] = {"mcts_type": "gumbel"}
    envelope = {
        "benchmark_schema_version": bench.BENCHMARK_SCHEMA_VERSION,
        "protocol": protocol.to_dict(),
        "candidate": {"kind": "checkpoint", "id": f"checkpoint:{name}"},
        "shard": None,
        "evidence": evidence,
        "results": [
            bench._result_record("checkpoint:" + "0" * 64, games, evidence["generated_utc"])
        ],
    }
    bench.validate_envelope(envelope)
    out = directory / name
    out.mkdir()
    (out / "result.json").write_text(json.dumps(envelope), encoding="utf-8")
    return out / "result.json"


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = clt.main(argv)
    out = capsys.readouterr().out
    return code, json.loads(out)


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


def test_two_points_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The VLP's negative control: two evaluation points are not a trend."""
    _eval_file(tmp_path, 5, 10, 5, 5)
    _eval_file(tmp_path, 10, 12, 4, 4)
    code, report = _run(["--run-dir", str(tmp_path)], capsys)
    assert code == clt.EXIT_INSUFFICIENT
    assert report["status"] == "insufficient"
    assert report["score_series"][0]["trend"]["slope_per_iteration"] is None


def test_three_points_exit_0_with_the_least_squares_slope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for it, wins in ((5, 10), (10, 12), (15, 14)):
        _eval_file(tmp_path, it, wins, 0, 20 - wins)
    code, report = _run(["--run-dir", str(tmp_path)], capsys)
    assert code == clt.EXIT_TREND and report["status"] == "trend"
    trend = report["score_series"][0]["trend"]
    # Scores 0.5, 0.6, 0.7 over iterations 5, 10, 15: slope 0.02 per iteration.
    assert trend["slope_per_iteration"] == pytest.approx(0.02)
    assert trend["delta"] == pytest.approx(0.2)
    assert report["descriptive_only"] is True


def test_a_score_that_does_not_match_its_counts_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for it in (5, 10, 15):
        _eval_file(tmp_path, it, 10, 0, 10)
    planted = json.loads((tmp_path / "eval_10.json").read_text(encoding="utf-8"))
    planted["result"]["score_rate"] = 0.9
    (tmp_path / "eval_10.json").write_text(json.dumps(planted), encoding="utf-8")
    code, report = _run(["--run-dir", str(tmp_path)], capsys)
    assert code == clt.EXIT_MALFORMED
    assert any("does not match its counts" in p for p in report["problems"])


def test_a_torn_iterations_line_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for it in (5, 10, 15):
        _eval_file(tmp_path, it, 10, 0, 10)
    path = _write_records(tmp_path, [_record(1), _record(2)])
    path.write_text(path.read_text(encoding="utf-8") + '{"iteration": 3, "tra', encoding="utf-8")
    code, _ = _run(["--run-dir", str(tmp_path)], capsys)
    assert code == clt.EXIT_MALFORMED


def test_malformed_beats_a_computed_trend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for it in (5, 10, 15):
        _eval_file(tmp_path, it, 10, 0, 10)
    bad = tmp_path / "bench"
    bad.mkdir()
    (bad / "result.json").write_text("{not json", encoding="utf-8")
    code, report = _run(["--run-dir", str(tmp_path), "--benchmark", str(bad)], capsys)
    assert code == clt.EXIT_MALFORMED
    assert report["score_series"], "the valid series is still described"


def test_a_missing_input_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, report = _run(["--eval", str(tmp_path / "eval_5.json")], capsys)
    assert code == clt.EXIT_INSUFFICIENT and report["missing"]


def test_no_input_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        clt.main([])
    assert exc.value.code == 2


def test_duplicate_points_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = _eval_file(tmp_path, 5, 10, 0, 10)
    other = tmp_path / "copy"
    other.mkdir()
    second = other / "eval_5.json"
    second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
    code, _ = _run(["--eval", str(first), str(second)], capsys)
    assert code == clt.EXIT_MALFORMED


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def test_iterations_jsonl_eval_fills_iterations_without_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [_record(i) for i in (1, 2, 3)]
    for record, wins in zip(records, (8, 10, 12), strict=True):
        record["eval"] = dict(summarize(wins, 0, 20 - wins).to_dict())
    _write_records(tmp_path, records)
    code, report = _run(["--run-dir", str(tmp_path)], capsys)
    assert code == clt.EXIT_TREND
    points = report["score_series"][0]["points"]
    assert [p["iteration"] for p in points] == [1, 2, 3]


def test_benchmark_series_is_placed_by_the_candidate_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arena = tmp_path / "arena"
    arena.mkdir()
    _envelope(arena, "secondary_5", "checkpoint_5.pt", ["win", "loss", "draw", "draw"])
    _envelope(arena, "secondary_10", "checkpoint_10.pt", ["win", "draw", "win", "loss"])
    _envelope(arena, "primary", "checkpoint_20.pt", ["win", "win", "win", "draw"])
    code, report = _run(["--benchmark", str(arena)], capsys)
    assert code == clt.EXIT_TREND
    (series,) = report["score_series"]
    assert series["label"].startswith("benchmark vs checkpoint_0.pt")
    assert [p["iteration"] for p in series["points"]] == [5, 10, 20]
    assert series["trend"]["slope_per_iteration"] > 0


def test_an_invalid_envelope_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _envelope(tmp_path, "primary", "checkpoint_20.pt", ["win", "loss"])
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["results"][0]["candidate_wins"] = 2  # counts no longer add up
    path.write_text(json.dumps(envelope), encoding="utf-8")
    code, _ = _run(["--benchmark", str(path)], capsys)
    assert code == clt.EXIT_MALFORMED


def test_a_baseline_candidate_is_reported_unplaced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _envelope(tmp_path, "positive_control", "latest.pt", ["win", "draw"])
    code, report = _run(["--benchmark", str(path)], capsys)
    assert code == clt.EXIT_INSUFFICIENT
    assert report["unplaced_benchmark_rows"][0]["candidate_file"] == "latest.pt"


def test_diagnostics_skip_error_records(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_records(tmp_path, [_record(1, loss=3.0), _record(2, error=True), _record(3, loss=1.0)])
    _, report = _run(["--run-dir", str(tmp_path)], capsys)
    by_label = {s["label"]: s for s in report["diagnostic_series"]}
    assert by_label["train.total_loss"]["trend"]["n_points"] == 3
    assert by_label["prequential.value_mse"]["trend"]["n_points"] == 2


# ---------------------------------------------------------------------------
# Option E trigger (docs/EVALUATION.md section 7.1)
# ---------------------------------------------------------------------------


class TestOptionETrigger:
    def test_quiet_run_does_not_fire(self) -> None:
        records = {i: _record(i, share=0.01, max_p=0.32) for i in range(1, 21)}
        trigger = clt.option_e_trigger(records)
        assert trigger is not None
        assert trigger["s0"] == 0.01 and trigger["t1_threshold"] == 0.25
        assert not trigger["fired"]

    def test_t1_fires_above_the_floor_after_iteration_zero(self) -> None:
        records = {1: _record(1, share=0.30), 2: _record(2, share=0.26)}
        trigger = clt.option_e_trigger(records)
        assert trigger is not None
        # s0 = 0.30 makes the threshold 1.5, so neither fires; the floor needs s0 <= 0.05.
        assert trigger["t1_threshold"] == pytest.approx(1.5) and not trigger["t1_fired_at"]
        records[1] = _record(1, share=0.02)
        trigger = clt.option_e_trigger(records)
        assert trigger is not None and trigger["t1_fired_at"] == [2]

    def test_t1_ignores_loop_index_zero(self) -> None:
        trigger = clt.option_e_trigger({1: _record(1, share=0.9, generator="simple_mcts")})
        assert trigger is not None and trigger["t1_fired_at"] == []

    def test_t2_fires_at_any_iteration(self) -> None:
        trigger = clt.option_e_trigger({1: _record(1, max_p=0.81), 2: _record(2, max_p=0.5)})
        assert trigger is not None and trigger["t2_fired_at"] == [1] and trigger["fired"]

    @pytest.mark.parametrize(
        ("first", "reason"),
        [
            (_record(1, share=0.04, generator="simple_mcts"), "not network-generated"),
            (_record(1, error=True), "error-free"),
            (_record(1, share=None), "null"),
        ],
    )
    def test_s0_defaults_to_zero(self, first: dict[str, Any], reason: str) -> None:
        trigger = clt.option_e_trigger({1: first, 2: _record(2, share=0.1)})
        assert trigger is not None and trigger["s0"] == 0.0 and reason in trigger["s0_source"]

    def test_no_records_no_trigger(self) -> None:
        assert clt.option_e_trigger({}) is None


# ---------------------------------------------------------------------------
# --print-protocol-sha
# ---------------------------------------------------------------------------


def test_print_protocol_sha_matches_the_library(capsys: pytest.CaptureFixture[str]) -> None:
    from alpha_zero.evaluation.a1_protocol import protocol_sha256

    assert clt.main(["--print-protocol-sha"]) == 0
    printed = capsys.readouterr().out.strip()
    expected = protocol_sha256((REPO_ROOT / "docs" / "EVALUATION.md").read_text(encoding="utf-8"))
    assert printed == expected and len(printed) == 64


def test_print_protocol_sha_exit_codes(tmp_path: Path) -> None:
    assert clt.main(["--print-protocol-sha", "--evaluation-md", str(tmp_path / "x.md")]) == 2
    bad = tmp_path / "bad.md"
    bad.write_text("no markers here\n", encoding="utf-8")
    assert clt.main(["--print-protocol-sha", "--evaluation-md", str(bad)]) == 1
