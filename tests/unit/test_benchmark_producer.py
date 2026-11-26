"""The benchmark as the A1 evidence producer (LF-D6, plan §5.1 / §5.2).

What A1 needs from ``scripts/benchmark_suite.py`` and what each test pins:

* **Evidence contract.** ``result.json`` carries an ``evidence`` block that the
  committed-evidence gate (``tests/regression/test_evidence_contract_regression.py``)
  accepts: ``schema_version``, ``search_epoch: post-2026-09-04``, ``claims`` /
  ``non_claims``, a reproducible ``shlex``-joined command and the git SHA.
* **Statistics.** Each arm reports the iid ternary interval, the pair-clustered
  interval and their hull at 95% and 99%, plus the one-sided 2.5% MDE.
* **Shards.** ``--shard i/N`` plays whole seat-swapped pairs with the seeds the
  unsharded run would use, so the merged shards *are* the unsharded run.
* **Positive control.** ``--candidate heuristic`` runs with no network.
* **Pricing.** The five A1 arms, parsed by the runner's own parser through the
  run-card driver, price at exactly 980 games.

Negative controls performed when written (each reverted; see the T11 report):
splitting a pair across shards fails ``test_union_of_shards_equals_unsharded``;
taking the hull from the iid interval alone fails
``test_hull_is_computed_from_both_intervals``.
"""

from __future__ import annotations

import copy
import json
import logging
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

from alpha_zero.evaluation import benchmark as bench
from alpha_zero.evaluation.baseline_agents import RandomAgent
from alpha_zero.evaluation.benchmark import BenchmarkProtocol, run_benchmark
from alpha_zero.evaluation.scoring import pair_score_interval, score_interval
from alpha_zero.local import run_card as rc
from alpha_zero.utils.checkpoint import read_git_sha
from alpha_zero.utils.game_budget import worst_case_games
from tests.regression.test_evidence_contract_regression import (
    block_problems,
    evidence_blocks,
    report_files,
)

_TINY_ARGS = ["--seed", "5", "--board-size", "5", "--max-turns", "6", "--sims", "2"]
_TINY_ARGS += ["--gumbel-k", "2", "--device", "cpu"]


@pytest.fixture(autouse=True)
def _quiet_env() -> Any:
    env_logger = logging.getLogger("alpha_zero.envs.microciv")
    previous = env_logger.level
    env_logger.setLevel(logging.ERROR)
    yield
    env_logger.setLevel(previous)


def _random(seed: int | None) -> RandomAgent:
    return RandomAgent(seed=seed)


def _main(argv: list[str], out: Path) -> dict[str, Any]:
    assert bench.main([*argv, "--out", str(out)]) == 0
    return json.loads((out / "result.json").read_text(encoding="utf-8"))


def _games(pair_scores: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Per-game records from ``(seat0_result, seat1_result)`` per pair."""
    return [
        {"pair": p, "seat": seat, "result": res}
        for p, pair in enumerate(pair_scores)
        for seat, res in enumerate(pair)
    ]


def _scrub_rows(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    rows = copy.deepcopy(envelope["results"])
    for row in rows:
        row["timestamp"] = "<scrubbed>"
    return rows


# ---------------------------------------------------------------------------
# Evidence contract and the positive control
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def heuristic_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[str], dict[str, Any]]:
    logging.getLogger("alpha_zero.envs.microciv").setLevel(logging.ERROR)
    out = tmp_path_factory.mktemp("heuristic")
    argv = ["--opponents=", "--candidate", "heuristic", "--opponents=random", "--n-games", "8"]
    argv += _TINY_ARGS
    return argv, _main(argv, out)


class TestEvidenceContract:
    def test_evidence_block_passes_the_committed_evidence_gate(
        self, heuristic_run: tuple[list[str], dict[str, Any]]
    ) -> None:
        _, envelope = heuristic_run
        block = envelope["evidence"]
        assert block_problems(block) == []
        assert block["schema_version"] == 1
        assert block["search_epoch"] == "post-2026-09-04"
        assert block["claims"] and block["non_claims"]
        assert block["games_total"] == 8

    def test_a_planted_evidence_dir_passes_the_gate_predicates(
        self, heuristic_run: tuple[list[str], dict[str, Any]], tmp_path: Path
    ) -> None:
        _, envelope = heuristic_run
        d = tmp_path / "planted_bench"
        d.mkdir()
        (d / "result.json").write_text(json.dumps(envelope), encoding="utf-8")
        (d / "report.md").write_text(bench.render_report(envelope), encoding="utf-8")
        assert len(report_files(d)) == 1
        [(_, block)] = evidence_blocks(d)
        assert block_problems(block) == []

    def test_command_reproduces_the_argv_and_sha_is_recorded(
        self, heuristic_run: tuple[list[str], dict[str, Any]]
    ) -> None:
        argv, envelope = heuristic_run
        command = shlex.split(envelope["evidence"]["command"])
        assert command[:3] == ["python", "-m", "alpha_zero.evaluation.benchmark"]
        assert command[3 : 3 + len(argv)] == argv
        assert envelope["evidence"]["git_sha"] == read_git_sha()

    def test_heuristic_candidate_plays_with_no_network(
        self, heuristic_run: tuple[list[str], dict[str, Any]]
    ) -> None:
        _, envelope = heuristic_run
        assert envelope["candidate"] == {"kind": "heuristic", "id": "heuristic"}
        [row] = envelope["results"]
        assert row["opponent_id"] == "random"
        assert len(row["games"]) == row["num_games"] == 8
        assert row["pairs"] == 4

    def test_every_arm_reports_iid_pair_and_hull_at_95_and_99(
        self, heuristic_run: tuple[list[str], dict[str, Any]]
    ) -> None:
        _, envelope = heuristic_run
        [row] = envelope["results"]
        for level in ("0.95", "0.99"):
            block = row["intervals"][level]
            iid, pair, hull = block["iid"], block["pair"], block["hull"]
            assert hull == [min(iid[0], pair[0]), max(iid[1], pair[1])]
        assert row["intervals"]["0.95"]["iid"] == [row["ci_low"], row["ci_high"]]
        assert row["mde_alpha025_80pct"] > row["mde_80pct"]
        assert row["hull_ratio"] >= 1.0

    def test_device_and_search_config_are_recorded(
        self, heuristic_run: tuple[list[str], dict[str, Any]]
    ) -> None:
        _, envelope = heuristic_run
        block = envelope["evidence"]
        assert block["device"] == "cpu"
        assert block["search"] == {
            "mcts_type": "gumbel",
            "num_simulations": 2,
            "gumbel_k": 2,
            "search_semantics": bench.SEARCH_SEMANTICS,
        }
        assert block["torch"] and block["numpy"]

    def test_command_names_the_wrapper_when_the_wrapper_ran(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        argv = ["--candidate", "random", "--opponents=random", "--n-games", "2", *_TINY_ARGS]
        out = tmp_path / "w"
        monkeypatch.setattr("sys.argv", ["scripts/benchmark_suite.py", *argv, "--out", str(out)])
        assert bench.main() == 0
        envelope = json.loads((out / "result.json").read_text(encoding="utf-8"))
        command = shlex.split(envelope["evidence"]["command"])
        assert command[:2] == ["python", "scripts/benchmark_suite.py"]
        assert command[2:] == [*argv, "--out", str(out)]

    def test_an_existing_result_is_never_overwritten(self, tmp_path: Path) -> None:
        argv = ["--candidate", "random", "--opponents=random", "--n-games", "2", *_TINY_ARGS]
        first = _main(argv, tmp_path)
        before = (tmp_path / "result.json").read_bytes()
        with pytest.raises(SystemExit) as excinfo:
            bench.main([*argv, "--seed", "6", "--out", str(tmp_path)])
        assert excinfo.value.code != 0
        assert (tmp_path / "result.json").read_bytes() == before
        assert first["evidence"]["seed"] == 5

    def test_random_candidate_runs(self, tmp_path: Path) -> None:
        argv = ["--candidate", "random", "--opponents=random", "--n-games", "2", *_TINY_ARGS]
        envelope = _main(argv, tmp_path)
        assert envelope["candidate"]["kind"] == "random"

    def test_baseline_candidate_refuses_a_checkpoint(self, tmp_path: Path) -> None:
        argv = ["--candidate", "heuristic", "--checkpoint", str(tmp_path / "x.pt")]
        with pytest.raises(SystemExit):
            bench.main([*argv, "--opponents=random", "--out", str(tmp_path)])


# ---------------------------------------------------------------------------
# Statistics over per-game records
# ---------------------------------------------------------------------------


class TestIntervalsFromGames:
    def test_pair_scores_average_the_two_seats(self) -> None:
        games = _games([("win", "win"), ("win", "draw"), ("loss", "draw"), ("loss", "win")])
        assert bench.pair_scores(games) == [1.0, 0.75, 0.25, 0.5]

    def test_an_incomplete_pair_is_refused(self) -> None:
        games = _games([("win", "loss")])[:1]
        with pytest.raises(ValueError, match="pair 0"):
            bench.pair_scores(games)

    def test_hull_is_computed_from_both_intervals(self) -> None:
        # Fixture A of test_scoring_pair_interval: six pairs won twice, four lost twice.
        games_a = _games([("win", "win")] * 6 + [("loss", "loss")] * 4)
        block = bench.interval_summary(games_a, 0.95)
        pair = pair_score_interval([1.0] * 6 + [0.0] * 4, 0.95)
        assert block["iid"] == pytest.approx(list(score_interval(12, 0, 8, 0.95)))
        assert block["pair"] == pytest.approx(list(pair))
        assert block["hull"] == pytest.approx([0.279939, 0.920061], abs=1e-6)
        # Fixture B at 99%: the low from the iid interval, the high from the pair one.
        games_b = _games(
            [("win", "loss")] * 5 + [("win", "win")] * 2 + [("loss", "draw")],
        )
        block_b = bench.interval_summary(games_b, 0.99)
        assert block_b["hull"] == pytest.approx([0.304054, 0.835234], abs=1e-6)

    def test_hull_ratio_uses_the_half_width_facing_one_half(self) -> None:
        """Fixture A: score 0.6, so the 0.5-facing side is the low one.

        iid low 0.386030, hull low 0.279939: (0.6 - 0.279939) / (0.6 - 0.386030).
        """
        games_a = _games([("win", "win")] * 6 + [("loss", "loss")] * 4)
        row = bench._result_record("random", games_a, "t")
        assert row["hull_ratio"] == pytest.approx((0.6 - 0.279939) / (0.6 - 0.386030), abs=1e-5)
        assert row["mde_alpha025_80pct_hull"] == pytest.approx(
            row["mde_alpha025_80pct"] * row["hull_ratio"]
        )
        # Below one half the facing side is the high one.
        games_low = _games([("loss", "loss")] * 6 + [("win", "win")] * 4)
        low = bench._result_record("random", games_low, "t")
        hi_iid, hi_hull = low["intervals"]["0.95"]["iid"][1], low["intervals"]["0.95"]["hull"][1]
        assert low["hull_ratio"] == pytest.approx((hi_hull - 0.4) / (hi_iid - 0.4))


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------


class TestShards:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [("1/1", (1, 1)), ("2/3", (2, 3)), (" 3/3 ", (3, 3))],
    )
    def test_shard_spec_parses(self, spec: str, expected: tuple[int, int]) -> None:
        assert bench.parse_shard(spec) == expected

    @pytest.mark.parametrize("spec", ["0/2", "3/2", "1/0", "1", "a/b", "1/2/3"])
    def test_bad_shard_spec_is_refused(self, spec: str) -> None:
        with pytest.raises(ValueError):
            bench.parse_shard(spec)

    def test_more_shards_than_pairs_is_a_parser_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            bench._parse_args(["--n-games", "4", "--shard", "3/3"])
        assert excinfo.value.code == 2
        assert "--shard" in capsys.readouterr().err

    def test_shards_partition_the_pairs(self) -> None:
        for num_pairs in (1, 3, 7, 100):
            for count in range(1, min(num_pairs, 5) + 1):
                ranges = [bench.shard_pairs(num_pairs, i, count) for i in range(1, count + 1)]
                assert [p for r in ranges for p in r] == list(range(num_pairs))
                assert all(len(r) > 0 for r in ranges)

    def test_union_of_shards_equals_unsharded(self) -> None:
        protocol = BenchmarkProtocol(
            board_size=5, max_turns=6, num_simulations=2, n_games=10, mcts_type="simple"
        )
        kwargs: dict[str, Any] = {"candidate_factory": _random, "command": "x"}
        full = run_benchmark(protocol, None, ["random", "heuristic"], 21, **kwargs)
        shards = [
            run_benchmark(protocol, None, ["random", "heuristic"], 21, shard=(i, 3), **kwargs)
            for i in (1, 2, 3)
        ]
        for shard in shards:
            lo, hi = shard["shard"]["pairs"]
            for row in shard["results"]:
                seats = {(g["pair"], g["seat"]) for g in row["games"]}
                assert seats == {(p, s) for p in range(lo, hi) for s in (0, 1)}, (
                    "a shard must hold whole seat-swapped pairs, and only its own"
                )
        # The shards ran on a GPU this (merging) host does not have: the merged
        # evidence must name the shards' hardware, not the merger's GPU 0.
        for shard in shards:
            shard["evidence"]["hardware"] = {**shard["evidence"]["hardware"], "gpu_model": "GPU-1"}
        merged = bench.merge_shards(shards, command="merge")
        assert merged["evidence"]["hardware"]["gpu_model"] == "GPU-1"
        assert _scrub_rows(merged) == _scrub_rows(full)
        assert merged["evidence"]["games_total"] == full["evidence"]["games_total"] == 20
        assert merged["shard"] is None
        assert [s["index"] for s in merged["evidence"]["shards"]] == [1, 2, 3]
        assert block_problems(merged["evidence"]) == []
        outcomes = {g["result"] for row in full["results"] for g in row["games"]}
        assert len(outcomes) > 1, "fixture is vacuous: every game had one outcome"

    def test_merge_refuses_an_incomplete_or_mismatched_set(self) -> None:
        protocol = BenchmarkProtocol(
            board_size=5, max_turns=4, num_simulations=2, n_games=4, mcts_type="simple"
        )
        kw: dict[str, Any] = {"candidate_factory": _random, "command": "x"}
        one = run_benchmark(protocol, None, ["random"], 3, shard=(1, 2), **kw)
        two = run_benchmark(protocol, None, ["random"], 3, shard=(2, 2), **kw)
        other_seed = run_benchmark(protocol, None, ["random"], 4, shard=(2, 2), **kw)
        with pytest.raises(ValueError, match="shard"):
            bench.merge_shards([one], command="x")
        with pytest.raises(ValueError, match="shard"):
            bench.merge_shards([one, one], command="x")
        with pytest.raises(ValueError, match="seed"):
            bench.merge_shards([one, other_seed], command="x")
        for key, value in (("device", "cuda"), ("torch", "0.0"), ("numpy", "0.0")):
            odd = copy.deepcopy(two)
            odd["evidence"][key] = value
            with pytest.raises(ValueError, match=key):
                bench.merge_shards([one, odd], command="x")
        odd = copy.deepcopy(two)
        odd["evidence"]["search"]["search_semantics"] = "other"
        with pytest.raises(ValueError, match="search"):
            bench.merge_shards([one, odd], command="x")
        bench.merge_shards([two, one], command="x")  # order does not matter

    def test_network_candidate_shards_and_merges_through_the_cli(self, tmp_path: Path) -> None:
        """The search path: a shard must reproduce the arena's per-game search seeds."""
        base = ["--opponents=random", "--n-games", "6", *_TINY_ARGS]
        full = _main(base, tmp_path / "full")
        for i in (1, 2):
            _main([*base, "--shard", f"{i}/2"], tmp_path / f"s{i}")
        argv = ["--merge", str(tmp_path / "s1"), str(tmp_path / "s2" / "result.json")]
        merged = _main(argv, tmp_path / "merged")
        assert _scrub_rows(merged) == _scrub_rows(full)
        assert merged["candidate"] == full["candidate"]
        assert (tmp_path / "merged" / "report.md").exists()


# ---------------------------------------------------------------------------
# Pricing the A1 arms through the runner's own parser
# ---------------------------------------------------------------------------

_A1_BASE = (
    "python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 8 "
    "--max-turns 60 --sims 24 --gumbel-k 8 --mcts-type gumbel --device cpu"
)
_CKPT = "D:/civ6az-data/runs/a1/ckpt/checkpoint_{}.pt"
_A1_ARMS: list[tuple[str, str, int]] = [
    ("primary", f"--checkpoint {_CKPT.format(20)} --opponent-checkpoint {_CKPT.format(0)} "
     "--n-games 200", 200),
    ("symmetry", f"--checkpoint {_CKPT.format(0)} --opponent-checkpoint {_CKPT.format(0)} "
     "--n-games 40", 40),
    ("positive-control", "--candidate heuristic --opponents=random --n-games 40", 40),
    *[
        (f"secondary-{k}", f"--checkpoint {_CKPT.format(k)} "
         f"--opponent-checkpoint {_CKPT.format(0)} --n-games 60", 60)
        for k in (5, 10, 15)
    ],
    ("final-vs-random", f"--checkpoint {_CKPT.format(20)} --opponents=random --n-games 40", 40),
]  # fmt: skip


#: The registered training command, verbatim (``<D>`` written with forward slashes).
_A1_TRAIN = shlex.split(
    "python -m alpha_zero.training_civ6 --seed 20261111 --board_size 8 --max_turns 60 "
    "--num_iterations 20 --num_games 24 --mcts gumbel --gumbel_k 8 --gumbel_sims 24 "
    "--device cpu --bootstrap none --save_interval 5 --eval_games 0 "
    "--checkpoint_dir D:/civ6az-data/runs/a1/ckpt"
)


def _arm_argv(flags: str) -> list[str]:
    return shlex.split(f"{_A1_BASE} {flags}")[2:]


class TestA1Pricing:
    def test_every_arm_parses_with_the_runner_parser(self) -> None:
        for name, flags, games in _A1_ARMS:
            args = bench._parse_args(_arm_argv(flags))
            assert worst_case_games("benchmark", bench.benchmark_budget_inputs(args)) == games, name
            assert (args.board_size, args.max_turns, args.sims, args.gumbel_k) == (8, 60, 24, 8)
            assert args.seed == 20261111 and args.mcts_type == "gumbel"

    def test_the_card_prices_at_980(self, tmp_path: Path) -> None:
        """Plan §5.2: training (480) + the five arena arms (500) = 980, per invocation."""
        training = {"name": "train", "runner": "training_civ6", "argv": _A1_TRAIN[3:]}
        card = {
            "card_id": "a1-pricing-probe",
            "sha": "0123456789abcdef0123456789abcdef01234567",
            "max_games": 1000,
            "invocations": [training]
            + [
                {"name": name, "runner": "benchmark", "argv": _arm_argv(flags)}
                for name, flags, _ in _A1_ARMS
            ],
        }
        path = tmp_path / "a1.yaml"
        path.write_text(yaml.safe_dump(card), encoding="utf-8")
        priced = rc.price_card(rc.load_card(path))
        arena = [p.games for p in priced.invocations[1:]]
        assert arena == [g for _, _, g in _A1_ARMS] and sum(arena) == 500
        assert priced.invocations[0].games == 480
        assert priced.total == 980

    def test_a_shard_prices_its_own_games(self) -> None:
        flags = _A1_ARMS[0][1]
        prices = [
            worst_case_games(
                "benchmark",
                bench.benchmark_budget_inputs(
                    bench._parse_args([*_arm_argv(flags), "--shard", f"{i}/3"])
                ),
            )
            for i in (1, 2, 3)
        ]
        assert sum(prices) == 200 and all(p % 2 == 0 for p in prices)

    def test_merge_plays_no_games(self, tmp_path: Path) -> None:
        args = bench._parse_args(["--merge", str(tmp_path)])
        assert worst_case_games("benchmark", bench.benchmark_budget_inputs(args)) == 0


# ---------------------------------------------------------------------------
# The T9 horizon check still guards both checkpoint flags
# ---------------------------------------------------------------------------


def _checkpoint(path: Path, max_turns: int) -> Path:
    from alpha_zero.core.civ6_network import MicroCivNetwork
    from alpha_zero.training_civ6 import TrainingConfig
    from alpha_zero.utils.checkpoint import build_provenance

    net = MicroCivNetwork(board_size=5, action_space_size=376, num_res_blocks=1, num_filters=8)
    meta = build_provenance(TrainingConfig(board_size=5, max_turns=max_turns), git_sha=None)
    net.save(str(path), metadata=meta)
    return path


class TestHorizonCheck:
    @pytest.mark.parametrize("flag", ["--checkpoint", "--opponent-checkpoint"])
    def test_a_checkpoint_at_another_horizon_is_refused(self, tmp_path: Path, flag: str) -> None:
        from alpha_zero.utils.checkpoint import HorizonMismatch

        ckpt = _checkpoint(tmp_path / "c.pt", max_turns=30)
        argv = ["--opponents=random", flag, str(ckpt), "--n-games", "2", *_TINY_ARGS]
        with pytest.raises(HorizonMismatch):
            bench.main([*argv, "--out", str(tmp_path / "out")])

    def test_a_matching_checkpoint_pair_runs(self, tmp_path: Path) -> None:
        ckpt = _checkpoint(tmp_path / "c.pt", max_turns=6)
        argv = ["--opponents=", "--checkpoint", str(ckpt), "--opponent-checkpoint", str(ckpt)]
        envelope = _main([*argv, "--n-games", "2", *_TINY_ARGS], tmp_path / "out")
        assert envelope["candidate"]["kind"] == "checkpoint"
        assert envelope["candidate"]["id"] == bench.checkpoint_opponent_id(ckpt)
        assert envelope["results"][0]["opponent_id"] == envelope["candidate"]["id"]
