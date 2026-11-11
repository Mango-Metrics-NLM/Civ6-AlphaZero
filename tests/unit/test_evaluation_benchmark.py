"""The benchmark protocol: schema, reproducibility, and a non-vacuous sign check.

Three properties, each with a negative control performed when the file was
written (and reverted):

* **Schema.** ``validate_envelope`` rejects a missing key, a wrong schema
  version and counts that do not add up. Control: delete ``"mde_80pct"`` from
  ``_RESULT_EXTRA_KEYS`` -> ``test_envelope_validates`` still passes but
  ``test_result_rows_carry_the_extra_keys`` fails, which is why both exist.
* **Reproducibility.** Two runs at one seed give identical envelopes once
  ``generated_utc`` (and the per-row ``timestamp`` that copies it) is scrubbed.
  Control: drop ``env_seeds=`` from ``run_benchmark``'s arena call ->
  ``test_same_seed_twice_is_identical`` fails.
* **Sign.** The heuristic baseline beats random with ``ci_low > 0.5`` at tiny
  settings (measured 27W/4D/9L, score 0.725, CI [0.578, 0.833], 0.3 s).
  Control: swap the win/loss attribution in ``run_arena`` -> score 0.275,
  ``test_heuristic_beats_random_with_significance`` fails.

Why heuristic and not SimpleMCTS: measured at board 6 / 20 turns, 64 sims,
n=40 paired, **SimpleMCTS scored 0.388 against random** (5W/21D/14L, 146.6 s).
A one-game trace explains it -- at temperature 0 its near-tied visit counts
resolve by ``argmax`` to the lowest index, which is END_TURN, in 87% of moves
(tied in 78%) -- so it mostly passes. That is a property of the baseline, not
of the arena (random-vs-random measured 0.537, CI covering 0.5). A strength
check must use a player that is actually strong, and must stay fast enough to
gate every PR.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.evaluation import benchmark as bench
from alpha_zero.evaluation.baseline_agents import GreedyHeuristicAgent, RandomAgent
from alpha_zero.evaluation.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkProtocol,
    checkpoint_opponent_id,
    render_report,
    run_benchmark,
    validate_envelope,
)
from alpha_zero.utils.seeding import SeedStream, derive_seed

_TINY = BenchmarkProtocol(
    board_size=6, max_turns=20, num_simulations=8, n_games=40, mcts_type="simple"
)
_SEED = 7


@pytest.fixture(autouse=True)
def _quiet_env_warnings() -> Any:
    """MicroCiv logs every rejected move at WARNING; thousands per benchmark."""
    env_logger = logging.getLogger("alpha_zero.envs.microciv")
    previous = env_logger.level
    env_logger.setLevel(logging.ERROR)
    yield
    env_logger.setLevel(previous)


def _random_factory(seed: int | None) -> RandomAgent:
    return RandomAgent(seed=seed)


def _heuristic_factory(seed: int | None) -> GreedyHeuristicAgent:
    return GreedyHeuristicAgent(seed=seed)


def _scrub(envelope: dict[str, Any]) -> dict[str, Any]:
    scrubbed = copy.deepcopy(envelope)
    scrubbed["evidence"]["generated_utc"] = "<scrubbed>"
    for row in scrubbed["results"]:
        row["timestamp"] = "<scrubbed>"
    return scrubbed


@pytest.fixture(scope="module")
def heuristic_envelope() -> dict[str, Any]:
    logging.getLogger("alpha_zero.envs.microciv").setLevel(logging.ERROR)
    return run_benchmark(
        _TINY, None, ["random"], _SEED, candidate_factory=_heuristic_factory, command="test"
    )


class TestProtocol:
    def test_defaults_are_the_documented_recipe(self) -> None:
        p = BenchmarkProtocol()
        assert (p.board_size, p.max_turns, p.num_simulations, p.gumbel_k) == (8, 60, 25, 16)
        assert p.n_games % 2 == 0 and p.paired_seats

    def test_odd_games_are_refused(self) -> None:
        with pytest.raises(ValueError, match="even"):
            BenchmarkProtocol(n_games=39)

    def test_env_seeds_are_named_streams(self) -> None:
        p = BenchmarkProtocol(n_games=6)
        assert p.env_seeds(42) == [derive_seed(42, SeedStream.BENCH_ENV, i) for i in range(3)]
        assert p.env_seeds(None) == [None, None, None]

    def test_round_trips_through_dict(self) -> None:
        assert BenchmarkProtocol.from_dict(_TINY.to_dict()) == _TINY


class TestEnvelope:
    def test_envelope_validates(self, heuristic_envelope: dict[str, Any]) -> None:
        validate_envelope(heuristic_envelope)
        assert heuristic_envelope["benchmark_schema_version"] == BENCHMARK_SCHEMA_VERSION
        assert heuristic_envelope["evidence"]["search_epoch"] == bench.SEARCH_EPOCH
        assert heuristic_envelope["evidence"]["games_total"] == _TINY.n_games

    def test_result_rows_carry_the_extra_keys(self, heuristic_envelope: dict[str, Any]) -> None:
        row = heuristic_envelope["results"][0]
        for key in (
            "opponent_id",
            "draw_rate",
            "decisive_win_rate",
            "confidence",
            "mde_80pct",
            "elo_estimate",
            "elo_ci_low",
            "elo_ci_high",
            "score_rate",
            "ci_low",
            "ci_high",
            "schema_version",
        ):
            assert key in row, key
        assert row["opponent_id"] == "random"
        assert row["candidate_wins"] + row["champion_wins"] + row["draws"] == row["num_games"]

    def test_hardware_block_is_complete(self, heuristic_envelope: dict[str, Any]) -> None:
        hw = heuristic_envelope["evidence"]["hardware"]
        assert set(hw) == {"cpu_model", "gpu_model", "logical_cores", "omp_num_threads"}

    @pytest.mark.parametrize(
        "mutate, message",
        [
            (lambda e: e.pop("results"), "missing top-level key"),
            (lambda e: e.__setitem__("benchmark_schema_version", 99), "benchmark_schema_version"),
            (lambda e: e["evidence"].pop("git_sha"), "evidence missing"),
            (lambda e: e["results"][0].pop("mde_80pct"), "missing 'mde_80pct'"),
            (lambda e: e["results"][0].__setitem__("draws", 999), "counts sum"),
        ],
    )
    def test_rejects_malformed_envelopes(
        self, heuristic_envelope: dict[str, Any], mutate: Any, message: str
    ) -> None:
        broken = copy.deepcopy(heuristic_envelope)
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            validate_envelope(broken)

    def test_report_renders_every_opponent(self, heuristic_envelope: dict[str, Any]) -> None:
        text = render_report(heuristic_envelope)
        assert "`random`" in text and "Score rate" in text and "search epoch" in text


class TestReproducibility:
    def test_maps_come_from_the_named_stream(self) -> None:
        """The arena must receive ``protocol.env_seeds(seed)``, not ``seed + pair``.

        ``test_same_seed_twice_is_identical`` cannot see this: the arena's
        arithmetic fallback is deterministic too, so dropping ``env_seeds=``
        keeps that test green while the benchmark's maps silently collide with
        a training run's streams. Measured when written: planting
        ``env_seeds=None`` left the replay test passing and failed only this one.
        """
        from unittest.mock import patch

        from alpha_zero.envs import env_factory

        seen: list[int | None] = []
        real = env_factory.create_game_env

        def tracing(**kwargs: Any) -> Any:
            seen.append(kwargs["seed"])
            return real(**kwargs)

        small = BenchmarkProtocol(
            board_size=6, max_turns=4, num_simulations=2, n_games=4, mcts_type="simple"
        )
        with patch("alpha_zero.envs.env_factory.create_game_env", side_effect=tracing):
            run_benchmark(
                small, None, ["random"], 11, candidate_factory=_random_factory, command="x"
            )
        expected = small.env_seeds(11)
        assert seen == [expected[0], expected[0], expected[1], expected[1]]
        assert seen[0] != 11, "maps must be named streams, not the base seed itself"

    def test_same_seed_twice_is_identical(self) -> None:
        first = run_benchmark(
            _TINY, None, ["random"], _SEED, candidate_factory=_random_factory, command="x"
        )
        second = run_benchmark(
            _TINY, None, ["random"], _SEED, candidate_factory=_random_factory, command="x"
        )
        assert first["evidence"]["generated_utc"] != "" and _scrub(first) == _scrub(second)

    def test_different_seeds_differ(self) -> None:
        small = BenchmarkProtocol(
            board_size=6, max_turns=20, num_simulations=8, n_games=20, mcts_type="simple"
        )
        first = run_benchmark(
            small, None, ["random"], 1, candidate_factory=_random_factory, command="x"
        )
        second = run_benchmark(
            small, None, ["random"], 2, candidate_factory=_random_factory, command="x"
        )
        # Random vs random on different maps and streams: identical W/D/L is
        # possible by chance, identical per-pair map seeds is not.
        assert small.env_seeds(1) != small.env_seeds(2)
        assert first["evidence"]["seed"] != second["evidence"]["seed"]


class TestStrength:
    def test_heuristic_beats_random_with_significance(
        self, heuristic_envelope: dict[str, Any]
    ) -> None:
        row = heuristic_envelope["results"][0]
        assert row["ci_low"] > 0.5, (
            f"heuristic vs random: {row['candidate_wins']}W/{row['draws']}D/"
            f"{row['champion_wins']}L, score {row['score_rate']:.3f}, "
            f"CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        )
        assert row["elo_ci_low"] > 0
        assert row["decisive_win_rate"] is not None and row["decisive_win_rate"] > 0.5

    def test_random_vs_random_is_not_significant(self) -> None:
        """Guards the sign check against a biased arena: parity must read as parity."""
        envelope = run_benchmark(
            _TINY, None, ["random"], _SEED, candidate_factory=_random_factory, command="x"
        )
        row = envelope["results"][0]
        assert row["ci_low"] <= 0.5 <= row["ci_high"]


class TestOpponents:
    def test_simple_mcts_opponent_runs_through_the_protocol(self) -> None:
        small = BenchmarkProtocol(
            board_size=6, max_turns=4, num_simulations=2, n_games=2, mcts_type="simple"
        )
        envelope = run_benchmark(
            small, None, ["simple_mcts"], _SEED, candidate_factory=_random_factory, command="x"
        )
        assert envelope["results"][0]["opponent_id"] == "simple_mcts"

    def test_unknown_opponent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown opponent"):
            run_benchmark(_TINY, None, ["grandmaster"], _SEED, candidate_factory=_random_factory)

    def test_checkpoint_opponent_needs_a_model(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.pt"
        path.write_bytes(b"not really a checkpoint")
        opponent_id = checkpoint_opponent_id(path)
        assert opponent_id.startswith("checkpoint:") and len(opponent_id) == len("checkpoint:") + 64
        with pytest.raises(ValueError, match="no model"):
            run_benchmark(_TINY, None, [opponent_id], _SEED, candidate_factory=_random_factory)

    def test_candidate_or_factory_required(self) -> None:
        with pytest.raises(ValueError, match="candidate"):
            run_benchmark(_TINY, None, ["random"], _SEED)


class TestCli:
    def test_main_writes_result_and_report(self, tmp_path: Path) -> None:
        out = tmp_path / "bench"
        code = bench.main(
            [
                "--opponents",
                "random",
                "--seed",
                "3",
                "--n-games",
                "2",
                "--board-size",
                "6",
                "--max-turns",
                "3",
                "--sims",
                "2",
                "--gumbel-k",
                "2",
                "--out",
                str(out),
            ]
        )
        assert code == 0
        envelope = json.loads((out / "result.json").read_text(encoding="utf-8"))
        validate_envelope(envelope)
        assert envelope["protocol"]["mcts_type"] == "gumbel"
        assert "# Benchmark report" in (out / "report.md").read_text(encoding="utf-8")
