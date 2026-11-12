"""In-loop evaluation decoupled from checkpointing, and prequential diagnostics (T10).

Three changes to ``alpha_zero.training_civ6``, each pinned here:

* **LF-A14** -- evaluation runs on ``--eval_interval`` with ``--eval_games`` games,
  not on ``--save_interval``; ``--eval_games 0`` turns in-loop evaluation off. The
  game-budget model prices exactly what the loop plays.
* **LF-A3** -- ``evaluate_against_random`` returns the full
  :class:`~alpha_zero.evaluation.scoring.ScoreSummary` (the float it used to return
  scored every draw as a loss). The old float is still available from a shim that
  warns of deprecation. Each evaluation writes ``eval_<iter>.json``.
* **LF-A13** -- prequential value diagnostics: the network that generated an
  iteration's games is scored on those games *before* they enter the buffer, so the
  score is out-of-sample. Only played steps are recorded; search rollouts are not.

Negative controls (recorded in the T10 report): recording rollout steps into the
sink fails ``test_histogram_counts_only_played_steps``; re-coupling evaluation to
``save_interval`` fails ``test_evaluation_follows_eval_interval_not_save_interval``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from alpha_zero import training_civ6
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.evaluation.scoring import ScoreSummary, summarize
from alpha_zero.training.diagnostics import StepRecord, prequential_value_metrics
from alpha_zero.training_civ6 import (
    TrainingConfig,
    TrainingExample,
    build_parser,
    config_from_args,
    evaluate_against_random,
    evaluate_win_rate_against_random,
    self_play_game,
    training_budget_inputs,
    write_eval_json,
)
from alpha_zero.utils.game_budget import worst_case_games

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_config(tmp_path: Path, **overrides: Any) -> TrainingConfig:
    base: dict[str, Any] = {
        "board_size": 4,
        "max_turns": 3,
        "num_simulations": 2,
        "num_res_blocks": 1,
        "num_filters": 8,
        "num_games_per_iteration": 1,
        "num_iterations": 1,
        "min_replay_size": 10**9,
        "save_interval": 10**9,
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "log_dir": str(tmp_path / "log"),
        "seed": 1,
        "device": "cpu",
    }
    base.update(overrides)
    return TrainingConfig(**base)


def _price(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args, checkpoint_dir="unused")
    return worst_case_games("training_civ6", training_budget_inputs(config, args.dry_run))


class _EvalRecorder:
    """Stands in for ``evaluate_against_random``; records the 1-based iterations."""

    def __init__(self) -> None:
        self.iterations: list[int] = []
        self.num_games: list[int] = []

    def __call__(
        self, network: Any, config: Any, num_games: int = 20, iteration: int = 0
    ) -> ScoreSummary:
        self.iterations.append(iteration + 1)
        self.num_games.append(num_games)
        return summarize(1, 1, 0)


def _checkpoints(config: TrainingConfig) -> set[int]:
    ckpt = Path(config.checkpoint_dir)
    return {int(p.stem.split("_")[1]) for p in ckpt.glob("checkpoint_*.pt")}


def _eval_files(config: TrainingConfig) -> set[int]:
    ckpt = Path(config.checkpoint_dir)
    return {int(p.stem.split("_")[1]) for p in ckpt.glob("eval_*.json")}


# ---------------------------------------------------------------------------
# LF-A14: --eval_interval / --eval_games
# ---------------------------------------------------------------------------


class TestEvalSchedule:
    def test_evaluation_follows_eval_interval_not_save_interval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _EvalRecorder()
        monkeypatch.setattr(training_civ6, "run_self_play", lambda *a, **k: [])
        monkeypatch.setattr(training_civ6, "evaluate_against_random", recorder)
        config = _tiny_config(
            tmp_path, num_iterations=6, save_interval=3, eval_interval=2, eval_games=7
        )
        training_civ6.train(config)
        assert recorder.iterations == [2, 4, 6]
        assert recorder.num_games == [7, 7, 7]
        # Checkpoints keep their own cadence; checkpoint_0 is the pinned random-init.
        assert _checkpoints(config) == {0, 3, 6}
        assert _eval_files(config) == {2, 4, 6}

    def test_eval_interval_defaults_to_save_interval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _EvalRecorder()
        monkeypatch.setattr(training_civ6, "run_self_play", lambda *a, **k: [])
        monkeypatch.setattr(training_civ6, "evaluate_against_random", recorder)
        config = _tiny_config(tmp_path, num_iterations=6, save_interval=3)
        assert config.eval_interval is None
        training_civ6.train(config)
        assert recorder.iterations == [3, 6]
        assert recorder.num_games == [training_civ6.DEFAULT_EVAL_GAMES] * 2

    def test_eval_games_zero_runs_no_evaluation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _EvalRecorder()
        monkeypatch.setattr(training_civ6, "run_self_play", lambda *a, **k: [])
        monkeypatch.setattr(training_civ6, "evaluate_against_random", recorder)
        config = _tiny_config(
            tmp_path, num_iterations=4, save_interval=2, eval_interval=1, eval_games=0
        )
        training_civ6.train(config)
        assert recorder.iterations == []
        assert _eval_files(config) == set()
        assert _checkpoints(config) == {0, 2, 4}

    def test_eval_games_zero_under_dry_run_runs_no_evaluation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Real (tiny) self-play: --dry_run trains even on a small buffer.
        recorder = _EvalRecorder()
        monkeypatch.setattr(training_civ6, "evaluate_against_random", recorder)
        training_civ6.train(_tiny_config(tmp_path, eval_games=0), dry_run=True)
        assert recorder.iterations == []

    def test_dry_run_still_evaluates_every_iteration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _EvalRecorder()
        monkeypatch.setattr(training_civ6, "evaluate_against_random", recorder)
        training_civ6.train(_tiny_config(tmp_path), dry_run=True)
        profile = training_civ6.DRY_RUN_PROFILE
        assert recorder.iterations == list(range(1, profile.num_iterations + 1))
        assert recorder.num_games == [profile.num_eval_games] * profile.num_iterations

    def test_flags_parse_and_reach_the_config(self) -> None:
        args = build_parser().parse_args(["--eval_interval", "4", "--eval_games", "0"])
        config = config_from_args(args, checkpoint_dir="unused")
        assert (config.eval_interval, config.eval_games) == (4, 0)
        defaults = config_from_args(build_parser().parse_args([]), checkpoint_dir="unused")
        assert defaults.eval_interval is None
        assert defaults.eval_games == training_civ6.DEFAULT_EVAL_GAMES

    @pytest.mark.parametrize("argv", [["--eval_interval", "0"], ["--eval_games", "-1"]])
    def test_invalid_values_are_rejected(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)


class TestPricing:
    def test_defaults_price_to_10200(self) -> None:
        assert _price([]) == 10_200

    def test_dry_run_prices_to_8(self) -> None:
        assert _price(["--dry_run"]) == 8

    def test_dry_run_with_eval_games_zero_prices_self_play_only(self) -> None:
        assert _price(["--dry_run", "--eval_games", "0"]) == 4

    def test_registered_a1_command_prices_to_480(self) -> None:
        argv = (
            "--seed 20261111 --board_size 8 --max_turns 60 --num_iterations 20 "
            "--num_games 24 --mcts gumbel --gumbel_k 8 --gumbel_sims 24 --device cpu "
            "--bootstrap none --save_interval 5 --eval_games 0"
        ).split()
        assert _price(argv) == 480

    def test_eval_interval_and_games_are_priced(self) -> None:
        # 100 x 100 self-play + 5 games at i+1 in {3, 6, ..., 99} (33 evaluations).
        assert _price(["--eval_interval", "3", "--eval_games", "5"]) == 10_000 + 5 * 33

    def test_eval_interval_does_not_follow_save_interval_once_given(self) -> None:
        assert _price(["--save_interval", "1", "--eval_interval", "50"]) == 10_000 + 20 * 2

    @pytest.mark.parametrize(
        ("start", "iterations", "every"), [(0, 100, 10), (7, 20, 3), (0, 9, 10), (5, 6, 1)]
    )
    def test_price_matches_the_trainers_condition(
        self, start: int, iterations: int, every: int
    ) -> None:
        brute = sum(1 for i in range(start, iterations) if (i + 1) % every == 0)
        resolved = {
            "num_games": 0,
            "num_iterations": iterations,
            "start_iteration": start,
            "save_interval": 1,
            "eval_interval": every,
            "eval_games": 1,
        }
        assert worst_case_games("training_civ6", resolved) == brute


# ---------------------------------------------------------------------------
# LF-A3: ScoreSummary, the float shim, eval_<iter>.json
# ---------------------------------------------------------------------------


def _scripted_env(outcomes: list[int | None]) -> MagicMock:
    """Each game ends on its first step with a scripted winner (random-seat branch)."""
    env = MagicMock()
    env.current_player = 99
    env.num_players = 2
    state = {"game": 0}

    def _step(_action: int) -> tuple[None, float, bool, bool, dict[str, Any]]:
        winner = outcomes[state["game"]]
        state["game"] += 1
        return None, 0.0, True, False, {"winner": winner}

    env.reset.return_value = (None, {})
    env.step.side_effect = _step
    env.get_valid_actions.return_value = np.array([1, 0, 1])
    return env


def _run_scripted(fn: Any, outcomes: list[int | None]) -> Any:
    config = TrainingConfig(board_size=8, max_turns=10, num_simulations=2, device="cpu")
    with (
        patch("alpha_zero.training_civ6.MicroCivEnv", return_value=_scripted_env(outcomes)),
        patch("alpha_zero.training_civ6.Civ6MCTS"),
        patch("alpha_zero.training_civ6._mcts_network"),
    ):
        return fn(object(), config, num_games=len(outcomes))


class TestScoreSummaryReturn:
    # Seats rotate 0,1,0,1, so these winners are 1 win, 2 draws, 1 loss.
    OUTCOMES: list[int | None] = [0, None, None, 0]

    def test_returns_the_full_summary(self) -> None:
        summary = _run_scripted(evaluate_against_random, self.OUTCOMES)
        assert isinstance(summary, ScoreSummary)
        assert (summary.wins, summary.draws, summary.losses) == (1, 2, 1)
        assert summary.score_rate == pytest.approx(0.5)
        assert summary.draw_rate == pytest.approx(0.5)
        assert summary.ci_low < 0.5 < summary.ci_high
        assert summary == summarize(1, 2, 1)

    def test_float_shim_warns_and_returns_the_old_float(self) -> None:
        with pytest.warns(DeprecationWarning, match="evaluate_against_random"):
            rate = _run_scripted(evaluate_win_rate_against_random, self.OUTCOMES)
        # wins / games: the two draws count against it, exactly as before.
        assert isinstance(rate, float)
        assert rate == pytest.approx(0.25)


class TestEvalJson:
    def test_write_eval_json_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "eval_3.json"
        meta = {"iteration": 3, "evidence": {"schema_version": 1, "seed": 7}}
        write_eval_json(path, summarize(3, 1, 0), meta)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["iteration"] == 3
        assert payload["evidence"] == {"schema_version": 1, "seed": 7}
        result = payload["result"]
        assert (result["wins"], result["draws"], result["losses"]) == (3, 1, 0)
        assert result["score_rate"] == pytest.approx(0.875)
        assert result["draw_rate"] == pytest.approx(0.25)
        assert result["win_rate"] == pytest.approx(0.75)
        assert result["ci_low"] <= result["score_rate"] <= result["ci_high"]
        assert not list(tmp_path.glob("*.tmp")), "the write must be atomic (no stray temp)"

    def test_train_writes_a_schema_shaped_eval_file(self, tmp_path: Path) -> None:
        config = _tiny_config(tmp_path, eval_interval=1, eval_games=2)
        training_civ6.train(config)
        payload = json.loads((Path(config.checkpoint_dir) / "eval_1.json").read_text("utf-8"))
        assert payload["iteration"] == 1
        assert payload["opponent"] == "random"
        assert payload["result"]["total_games"] == 2
        evidence = payload["evidence"]
        assert evidence["schema_version"] == 1
        assert evidence["search_epoch"] == "post-2026-09-04"
        assert evidence["seed"] == 1
        assert evidence["games_total"] == 2
        assert evidence["claims"] and evidence["non_claims"]
        assert payload["provenance"]["max_turns"] == config.max_turns
        # M6: a reproducible, shell-quoted module invocation, not the script path.
        assert evidence["command"].startswith("python -m alpha_zero.training_civ6")

    def test_eval_names_its_checkpoint_or_null_when_none_is_saved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # M5: with eval every 2 and save every 3, eval_2 scores weights never saved.
        monkeypatch.setattr(training_civ6, "run_self_play", lambda *a, **k: [])
        monkeypatch.setattr(training_civ6, "evaluate_against_random", _EvalRecorder())
        config = _tiny_config(
            tmp_path, num_iterations=6, save_interval=3, eval_interval=2, eval_games=1
        )
        training_civ6.train(config)
        ckpt = Path(config.checkpoint_dir)
        two = json.loads((ckpt / "eval_2.json").read_text("utf-8"))
        six = json.loads((ckpt / "eval_6.json").read_text("utf-8"))
        assert two["checkpoint"] is None
        assert six["checkpoint"] == "checkpoint_6.pt"
        assert (ckpt / six["checkpoint"]).is_file()
        assert "the network after iteration 2" in two["evidence"]["claims"][0]

    def test_command_is_a_shell_quoted_module_invocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.argv", ["/x/training_civ6.py", "--checkpoint_dir", "a b"])
        block = training_civ6.eval_evidence_block(TrainingConfig(device="cpu"), {}, 2)
        assert block["command"] == "python -m alpha_zero.training_civ6 --checkpoint_dir 'a b'"


# ---------------------------------------------------------------------------
# LF-A13: prequential diagnostics
# ---------------------------------------------------------------------------


class _VectorValueNet(nn.Module):
    """Predicts ``vector[0]`` as the value and uniform logits over the action space."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.num_actions = num_actions

    def forward(
        self, spatial: torch.Tensor, vector: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(vector.shape[0], self.num_actions), vector[:, 0]


def _example(pred: float, target: float, policy: list[float], mask: list[float]) -> TrainingExample:
    return TrainingExample(
        spatial=np.zeros((1, 2, 2), dtype=np.float32),
        vector=np.array([pred, 0.0], dtype=np.float32),
        action_mask=np.array(mask, dtype=np.float32),
        policy=np.array(policy, dtype=np.float32),
        value=target,
    )


class TestPrequentialMetrics:
    MASK = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ONE_HOT = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    HALF = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def _fixture(self) -> list[TrainingExample]:
        # targets z = [1, -1, 1, 1]; predictions v = [0.5, -0.5, 0.0, 1.0].
        return [
            _example(0.5, 1.0, self.ONE_HOT, self.MASK),
            _example(-0.5, -1.0, self.HALF, self.MASK),
            _example(0.0, 1.0, self.ONE_HOT, self.MASK),
            _example(1.0, 1.0, self.HALF, self.MASK),
        ]

    def test_r2_oos_on_a_hand_computed_fixture(self) -> None:
        # SSE = 0.25 + 0.25 + 1 + 0 = 1.5.
        # c = 0:   SST = 1 + 1 + 1 + 1 = 4               -> R2 = 1 - 1.5/4 = 0.625
        # c = 0.5: SST = 0.25 + 2.25 + 0.25 + 0.25 = 3   -> R2 = 1 - 1.5/3 = 0.5
        net = _VectorValueNet(8)
        at_zero = prequential_value_metrics(net, self._fixture(), [], 0.0)
        assert at_zero["r2_oos"] == pytest.approx(0.625)
        assert at_zero["value_mse"] == pytest.approx(1.5 / 4)
        assert at_zero["baseline_mse"] == pytest.approx(1.0)
        at_half = prequential_value_metrics(net, self._fixture(), [], 0.5)
        assert at_half["r2_oos"] == pytest.approx(0.5)
        assert at_half["buffer_mean"] == pytest.approx(0.5)
        assert at_half["n_positions"] == 4

    def test_sign_accuracy_against_the_constant_baseline(self) -> None:
        metrics = prequential_value_metrics(_VectorValueNet(8), self._fixture(), [], 0.0)
        # |z| = 1 everywhere; signs right on 0.5, -0.5, 1.0 and wrong on 0.0 -> 3/4.
        assert metrics["n_decisive"] == 4
        assert metrics["sign_accuracy"] == pytest.approx(0.75)
        # The best constant sign predicts "+" and is right 3 times out of 4.
        assert metrics["sign_accuracy_baseline"] == pytest.approx(0.75)

    @pytest.mark.parametrize(("c", "expected"), [(0.0, None), (0.5, 0.75), (-0.2, 0.25)])
    def test_prequential_sign_baseline_uses_the_prior_constant(
        self, c: float, expected: float | None
    ) -> None:
        # M2: the constant sign(c), known before the games; c == 0 abstains (None).
        metrics = prequential_value_metrics(_VectorValueNet(8), self._fixture(), [], c)
        if expected is None:
            assert metrics["sign_accuracy_baseline_prequential"] is None
        else:
            assert metrics["sign_accuracy_baseline_prequential"] == pytest.approx(expected)
        # Reported at a sane precision, not float32 noise.
        assert metrics["sign_accuracy"] == round(metrics["sign_accuracy"], 6)

    def test_n_games_counts_distinct_games_in_the_step_records(self) -> None:
        # M4: positions within a game share one outcome; n_games is the sample size.
        steps = [StepRecord(0, 1, "move", True), StepRecord(2, 1, "move", True)]
        assert prequential_value_metrics(_VectorValueNet(8), [], steps, 0.0)["n_games"] == 2
        assert prequential_value_metrics(_VectorValueNet(8), [], [], 0.0)["n_games"] == 0

    def test_entropies(self) -> None:
        metrics = prequential_value_metrics(_VectorValueNet(8), self._fixture(), [], 0.0)
        # Targets: two one-hots (0) and two uniform-over-2 (ln 2).
        assert metrics["target_entropy"] == pytest.approx(np.log(2) / 2)
        # Uniform logits over the 4 legal actions.
        assert metrics["policy_entropy"] == pytest.approx(np.log(4))

    def test_target_sharpness(self) -> None:
        # LF-A5 / T12 I1: max-p of the targets [1, .5, 1, .5] -> mean 0.75; above the
        # default 0.99 threshold: the two one-hots -> 0.5. Strictly above, as
        # select_action's `max() > greedy_threshold`.
        metrics = prequential_value_metrics(_VectorValueNet(8), self._fixture(), [], 0.0)
        assert metrics["target_max_p_mean"] == pytest.approx(0.75)
        assert metrics["target_share_above_greedy_threshold"] == pytest.approx(0.5)
        assert metrics["greedy_threshold"] == pytest.approx(0.99)

    @pytest.mark.parametrize(("threshold", "share"), [(0.4, 1.0), (0.5, 0.5), (1.0, 0.0)])
    def test_target_share_uses_the_runs_threshold(self, threshold: float, share: float) -> None:
        metrics = prequential_value_metrics(
            _VectorValueNet(8), self._fixture(), [], 0.0, greedy_threshold=threshold
        )
        assert metrics["target_share_above_greedy_threshold"] == pytest.approx(share)
        assert metrics["greedy_threshold"] == pytest.approx(threshold)

    def test_target_sharpness_is_none_without_positions(self) -> None:
        metrics = prequential_value_metrics(_VectorValueNet(8), [], [], 0.0)
        assert metrics["target_max_p_mean"] is None
        assert metrics["target_share_above_greedy_threshold"] is None

    def test_histogram_and_invalid_counts(self) -> None:
        steps = [
            StepRecord(0, 1, "move", True),
            StepRecord(0, 2, "move", False),
            StepRecord(0, 3, "end_turn", True),
            StepRecord(1, 4, "found_city", False),
        ]
        metrics = prequential_value_metrics(_VectorValueNet(8), [], steps, 0.0)
        assert metrics["action_type_histogram"] == {"move": 2, "end_turn": 1, "found_city": 1}
        assert metrics["num_steps"] == 4
        assert metrics["invalid_action_count"] == 2
        assert metrics["invalid_move_count"] == 1

    def test_empty_iteration_is_well_defined(self) -> None:
        metrics = prequential_value_metrics(_VectorValueNet(8), [], [], 0.0)
        assert metrics["n_positions"] == 0
        assert metrics["r2_oos"] is None
        assert metrics["sign_accuracy"] is None
        json.dumps(metrics)  # serialisable as-is, for the per-iteration record

    def test_restores_the_networks_training_mode(self) -> None:
        net = _VectorValueNet(8)
        net.train()
        prequential_value_metrics(net, self._fixture(), [], 0.0)
        assert net.training


class TestStepSink:
    def test_histogram_counts_only_played_steps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from alpha_zero.core.civ6_mcts import SimpleMCTS

        env = MicroCivEnv(board_size=4, max_turns=3, seed=5)
        played = [0]
        total = [0]
        real_step = MicroCivEnv.step

        def counting_step(self: MicroCivEnv, action: int) -> Any:
            total[0] += 1
            if self is env:
                played[0] += 1
            return real_step(self, action)

        monkeypatch.setattr(MicroCivEnv, "step", counting_step)
        config = TrainingConfig(board_size=4, max_turns=3, num_simulations=3, device="cpu")
        sink: list[StepRecord] = []
        examples = self_play_game(
            env, SimpleMCTS(num_simulations=3, seed=5), config, game_seed=5, step_sink=sink
        )

        assert total[0] > played[0], "the search must have stepped clones, or this proves nothing"
        assert len(sink) == played[0] == len(examples)
        metrics = prequential_value_metrics(_VectorValueNet(8), [], sink, 0.0)
        assert sum(metrics["action_type_histogram"].values()) == played[0]
        assert all(record.game_idx == 0 for record in sink)

    def test_records_decoded_type_and_success(self) -> None:
        from alpha_zero.core.civ6_mcts import SimpleMCTS

        env = MicroCivEnv(board_size=4, max_turns=3, seed=2)
        config = TrainingConfig(board_size=4, max_turns=3, num_simulations=2, device="cpu")
        sink: list[StepRecord] = []
        self_play_game(
            env, SimpleMCTS(num_simulations=2, seed=2), config, step_sink=sink, game_idx=3
        )
        decode = env.action_space_handler.decode_action
        for record in sink:
            assert record.game_idx == 3
            assert record.action_type == decode(record.action)["type"]
            assert isinstance(record.success, bool)

    def test_no_sink_is_the_default_and_changes_nothing(self) -> None:
        from alpha_zero.core.civ6_mcts import SimpleMCTS

        config = TrainingConfig(board_size=4, max_turns=3, num_simulations=2, device="cpu")

        def play(sink: list[StepRecord] | None) -> list[float]:
            env = MicroCivEnv(board_size=4, max_turns=3, seed=9)
            mcts = SimpleMCTS(num_simulations=2, seed=9)
            return [e.value for e in self_play_game(env, mcts, config, 9, step_sink=sink)]

        assert play(None) == play([])

    def test_run_self_play_tags_steps_with_their_game(self) -> None:
        config = TrainingConfig(board_size=4, max_turns=2, num_simulations=2, device="cpu", seed=4)
        sink: list[StepRecord] = []
        examples = training_civ6.run_self_play(
            None, config, num_games=2, use_simple_mcts=True, step_sink=sink
        )
        assert len(sink) == len(examples)
        assert {record.game_idx for record in sink} == {0, 1}


def _fake_self_play(values_per_iteration: list[list[float]]) -> Any:
    """A ``run_self_play`` stand-in returning one example per scripted value target."""
    calls = {"i": 0}
    one_hot, mask = TestPrequentialMetrics.ONE_HOT, TestPrequentialMetrics.MASK

    def fake(*_a: Any, **_k: Any) -> list[TrainingExample]:
        values = values_per_iteration[calls["i"]]
        calls["i"] += 1
        return [_example(0.0, v, one_hot, mask) for v in values]

    return fake


class TestTrainLoopDiagnostics:
    def test_r2_baseline_is_zero_at_iteration_zero_then_the_buffer_mean(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[float, int]] = []

        def capture(
            net: Any, fresh: list[Any], steps: list[Any], buffer_mean: float, **_kw: Any
        ) -> dict:
            seen.append((buffer_mean, len(fresh)))
            return {"n_positions": len(fresh)}

        monkeypatch.setattr(training_civ6, "prequential_value_metrics", capture)
        monkeypatch.setattr(
            training_civ6, "run_self_play", _fake_self_play([[1.0, 1.0, 1.0, -1.0], [-1.0]])
        )
        training_civ6.train(_tiny_config(tmp_path, num_iterations=2))
        # Iteration 0: empty buffer, so c = 0 -- not NaN, not the fresh games' own mean
        # (which would leak the targets into the baseline). Iteration 1: mean of the
        # buffer BEFORE the second game is inserted = mean(1, 1, 1, -1) = 0.5.
        assert seen == [(0.0, 4), (0.5, 1)]

    def test_metrics_land_in_the_per_iteration_record(self, tmp_path: Path) -> None:
        config = _tiny_config(tmp_path, num_iterations=2)
        training_civ6.train(config)
        lines = (Path(config.checkpoint_dir) / "iterations.jsonl").read_text("utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        assert [r["iteration"] for r in records] == [1, 2]
        for record in records:
            pq = record["prequential"]
            assert pq["n_positions"] == record["self_play"]["examples"]
            assert sum(pq["action_type_histogram"].values()) == pq["num_steps"]
            assert pq["num_steps"] == record["self_play"]["examples"]
            assert record["eval"] is None
        assert records[0]["prequential"]["buffer_mean"] == 0.0
        # M3: iteration 0 is generated by the SimpleMCTS bootstrap, not the network.
        assert [r["prequential"]["generator"] for r in records] == ["simple_mcts", "network"]
        assert records[0]["prequential"]["n_games"] == config.num_games_per_iteration
        for record in records:
            pq = record["prequential"]
            assert 0.0 <= pq["target_share_above_greedy_threshold"] <= 1.0
            assert 0.0 < pq["target_max_p_mean"] <= 1.0
            assert pq["greedy_threshold"] == pytest.approx(config.gumbel_greedy_threshold)

    def test_record_eval_entry_and_checkpoint_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # M7: the record's eval carries the legacy win_rate too; checkpoint is relative.
        monkeypatch.setattr(training_civ6, "run_self_play", lambda *a, **k: [])
        monkeypatch.setattr(training_civ6, "evaluate_against_random", _EvalRecorder())
        config = _tiny_config(tmp_path, num_iterations=2, save_interval=2, eval_interval=1)
        training_civ6.train(config)
        lines = (Path(config.checkpoint_dir) / "iterations.jsonl").read_text("utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        assert records[0]["eval"]["win_rate"] == pytest.approx(0.5)
        assert records[0]["eval"]["score_rate"] == pytest.approx(0.75)
        assert [r["checkpoint"] for r in records] == [None, "checkpoint_2.pt"]

    def test_diagnostics_do_not_change_the_trained_weights(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def run(root: Path) -> dict[str, torch.Tensor]:
            config = _tiny_config(root, num_iterations=2, min_replay_size=1, num_epochs=1)
            training_civ6.train(config)
            latest = Path(config.checkpoint_dir) / "latest.pt"
            return torch.load(latest, map_location="cpu", weights_only=True)["state_dict"]

        with_diagnostics = run(tmp_path / "a")
        monkeypatch.setattr(training_civ6, "prequential_value_metrics", lambda *a, **k: {})
        without = run(tmp_path / "b")
        assert with_diagnostics.keys() == without.keys()
        for key, tensor in with_diagnostics.items():
            assert torch.equal(tensor, without[key]), key
