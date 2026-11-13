"""Tests for the game-budget cost models and the 1,000-game cap (LF-G11).

Two layers:

* the pure cost models (``worst_case_games``) against hand-derived numbers, and
* negative controls that drive each runner's real ``main()`` / argument path and
  prove the cap fires (exit status 2, ``REFUSED`` on stderr, work not started).
  A model that is correct but never called would pass the first layer and
  protect nothing, which is why the second layer exists.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from alpha_zero.utils import game_budget
from alpha_zero.utils.game_budget import (
    DEFAULT_MAX_GAMES,
    EXIT_CODE_REFUSED,
    GameBudgetExceeded,
    GameBudgetUnpriceable,
    enforce,
    priced_runners,
    worst_case_games,
)

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _assert_refused(excinfo: pytest.ExceptionInfo[SystemExit], capsys: pytest.CaptureFixture[str]):
    assert excinfo.value.code == EXIT_CODE_REFUSED
    err = capsys.readouterr().err
    assert "REFUSED" in err and "--max_games" in err
    return err


# ---------------------------------------------------------------------------
# Pure cost models
# ---------------------------------------------------------------------------


class TestTrainingCiv6Model:
    DEFAULTS: dict[str, Any] = {"num_games": 100, "num_iterations": 100, "save_interval": 10}

    def test_defaults_price_to_10200(self) -> None:
        # 100 games x 100 iterations + 20 eval games x 10 evaluations.
        assert worst_case_games("training_civ6", self.DEFAULTS) == 10_200

    def test_eval_games_zero_drops_the_evaluation_term(self) -> None:
        assert worst_case_games("training_civ6", {**self.DEFAULTS, "eval_games": 0}) == 10_000

    def test_dry_run_inputs_price_to_8(self) -> None:
        from alpha_zero.training_civ6 import TrainingConfig, training_budget_inputs

        inputs = training_budget_inputs(TrainingConfig(), dry_run=True)
        # 2 iterations x 2 games + 2 evaluations (every iteration) x 2 games.
        assert worst_case_games("training_civ6", inputs) == 8

    @pytest.mark.parametrize(
        ("start", "iterations", "every", "expected_evals"),
        [
            (0, 100, 10, 10),
            (0, 9, 10, 0),  # a run shorter than save_interval never evaluates
            (0, 10, 10, 1),
            (7, 20, 10, 2),  # i+1 in 8..20 -> 10, 20
            (10, 20, 10, 1),  # i+1 in 11..20 -> 20
            (0, 5, 1, 5),
        ],
    )
    def test_evaluation_count_matches_the_trainers_condition(
        self, start: int, iterations: int, every: int, expected_evals: int
    ) -> None:
        # Brute-force the trainer's own condition: (iteration + 1) % save_interval == 0.
        brute = sum(1 for i in range(start, iterations) if (i + 1) % every == 0)
        assert brute == expected_evals
        priced = worst_case_games(
            "training_civ6",
            {
                "num_games": 0,
                "num_iterations": iterations,
                "start_iteration": start,
                "save_interval": every,
                "eval_games": 1,
            },
        )
        assert priced == expected_evals

    def test_eval_interval_overrides_save_interval(self) -> None:
        # LF-A14: evaluation on its own cadence. 100 x 100 + 20 x 4 (i+1 in 25..100).
        resolved = {**self.DEFAULTS, "save_interval": 1, "eval_interval": 25}
        assert worst_case_games("training_civ6", resolved) == 10_080

    def test_eval_interval_none_falls_back_to_save_interval(self) -> None:
        resolved = {**self.DEFAULTS, "eval_interval": None}
        assert worst_case_games("training_civ6", resolved) == 10_200

    def test_eval_games_zero_is_free_even_under_dry_run(self) -> None:
        from alpha_zero.training_civ6 import TrainingConfig, training_budget_inputs

        inputs = training_budget_inputs(TrainingConfig(eval_games=0), dry_run=True)
        assert worst_case_games("training_civ6", inputs) == 4
        assert worst_case_games("training_civ6", {**self.DEFAULTS, "eval_always": True}) > 10_000
        no_eval = {**self.DEFAULTS, "eval_always": True, "eval_games": 0}
        assert worst_case_games("training_civ6", no_eval) == 10_000

    def test_trainer_inputs_carry_the_resolved_eval_schedule(self) -> None:
        from alpha_zero.training_civ6 import TrainingConfig, training_budget_inputs

        inputs = training_budget_inputs(
            TrainingConfig(save_interval=5, eval_interval=2, eval_games=3), dry_run=False
        )
        assert (inputs["eval_interval"], inputs["eval_games"]) == (2, 3)
        # 100 x 100 + 3 x 50.
        assert worst_case_games("training_civ6", inputs) == 10_150
        coupled = training_budget_inputs(TrainingConfig(save_interval=5), dry_run=False)
        assert coupled["eval_interval"] == 5

    def test_resume_prices_only_the_remaining_iterations(self) -> None:
        from alpha_zero.training_civ6 import TrainingConfig, training_budget_inputs

        config = TrainingConfig(resume_from="checkpoints/civ6/checkpoint_90.pt")
        inputs = training_budget_inputs(config, dry_run=False)
        assert inputs["start_iteration"] == 90
        # 10 iterations x 100 games + 1 evaluation (i+1 == 100) x 20 games.
        assert worst_case_games("training_civ6", inputs) == 1_020


class TestOtherModels:
    def test_benchmark_200_games_against_3_default_opponents_is_600(self) -> None:
        assert worst_case_games("benchmark", {"n_games": 200, "num_opponents": 3}) == 600

    def test_benchmark_counts_opponent_checkpoints(self) -> None:
        priced = worst_case_games(
            "benchmark", {"n_games": 200, "num_opponents": 3, "num_opponent_checkpoints": 2}
        )
        assert priced == 1_000

    def test_orchestrator_selfplay_and_evaluation_terms(self) -> None:
        resolved = {
            "num_iterations": 10,
            "games_per_iteration": 5,
            "evaluation_interval": 5,
            "evaluation_games": 20,
            "full_eval_interval": 5,
        }
        # self-play 50; evaluations at iterations 0 and 5 (40); both are also
        # full-eval iterations, each adding two baseline batches (80).
        assert worst_case_games("orchestrator", resolved) == 50 + 40 + 80

    def test_orchestrator_workers_and_retries_multiply_selfplay(self) -> None:
        resolved = {
            "num_iterations": 2,
            "games_per_iteration": 3,
            "workers": 4,
            "retries": 1,
            "evaluation_games": 0,
        }
        assert worst_case_games("orchestrator", resolved) == 4 * 3 * 2 * 2

    def test_orchestrator_dry_run_plays_one_game(self) -> None:
        assert worst_case_games("orchestrator", {"dry_run": True}) == 1

    def test_post_run_evaluation_round_robin(self) -> None:
        resolved = {"num_models": 11, "tournament_games": 10, "baseline_games": 20}
        assert worst_case_games("post_run_evaluation", resolved) == 55 * 10 + 2 * 20

    def test_post_run_evaluation_dry_run_still_prices_the_tournament(self) -> None:
        resolved = {
            "num_models": 11,
            "tournament_games": 10,
            "baseline_games": 20,
            "dry_run": True,
        }
        assert worst_case_games("post_run_evaluation", resolved) == 550

    def test_enhanced_training_includes_in_loop_and_final_evaluations(self) -> None:
        resolved = {"num_iterations": 10, "games_per_iteration": 10, "eval_interval": 5}
        assert worst_case_games("run_enhanced_training", resolved) == 100 + 2 * 10 + 20

    def test_chess_dry_run_plays_nothing(self) -> None:
        resolved = {"iterations": 50, "games_per_iter": 50}
        assert worst_case_games("run_chess_training", {**resolved, "dry_run": True}) == 0
        assert worst_case_games("run_chess_training", resolved) == 2_500

    def test_sweep_multiplies_trials_by_trial_games(self) -> None:
        resolved = {"n_trials": 20, "trial_games": 10}
        assert worst_case_games("run_sweep", resolved) == 200

    def test_sweep_with_unpriceable_trial_is_refused(self) -> None:
        with pytest.raises(GameBudgetUnpriceable):
            enforce("run_sweep", {"n_trials": 1, "trial_games": None})

    def test_unknown_runner_is_an_error_not_zero(self) -> None:
        with pytest.raises(KeyError, match="no game-budget model"):
            worst_case_games("not_a_runner", {})

    def test_every_listed_runner_has_a_model(self) -> None:
        assert set(priced_runners()) >= {
            "training_civ6",
            "orchestrator",
            "benchmark",
            "run_baseline_eval",
            "run_encoder_ab.train",
            "run_encoder_ab.arena",
            "run_chess_training",
            "run_enhanced_training",
            "run_muzero_training",
            "run_sweep",
            "post_run_evaluation",
            "validate_convergence",
            "benchmark_batch_selfplay",
            "local_muzero_launcher",
            "calibrate_throughput",
        }

    def test_calibrate_throughput_prices_one_game_per_game(self) -> None:
        assert worst_case_games("calibrate_throughput", {"games": 2}) == 2
        assert worst_case_games("calibrate_throughput", {"games": 1001}) == 1_001


# ---------------------------------------------------------------------------
# enforce()
# ---------------------------------------------------------------------------


class TestEnforce:
    RESOLVED: dict[str, Any] = {"num_games": 100, "num_iterations": 100, "save_interval": 10}

    def test_default_cap_is_1000(self) -> None:
        assert DEFAULT_MAX_GAMES == 1000

    def test_over_cap_raises_with_the_price_in_the_message(self) -> None:
        with pytest.raises(GameBudgetExceeded, match=r"10,200 games exceeds --max_games 1,000"):
            enforce("training_civ6", self.RESOLVED)

    def test_exactly_at_the_cap_is_allowed(self) -> None:
        assert enforce("benchmark", {"n_games": 250, "num_opponents": 4}) == 1_000

    def test_one_over_the_cap_is_refused(self) -> None:
        with pytest.raises(GameBudgetExceeded):
            enforce("benchmark", {"n_games": 1001, "num_opponents": 1})

    def test_approval_ref_lets_the_run_through_and_logs_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=game_budget.logger.name):
            games = enforce("training_civ6", self.RESOLVED, approval_ref="#142")
        assert games == 10_200
        assert "#142" in caplog.text and "10,200" in caplog.text

    def test_blank_approval_ref_does_not_count(self) -> None:
        with pytest.raises(GameBudgetExceeded):
            enforce("training_civ6", self.RESOLVED, approval_ref="   ")

    def test_raised_cap_alone_never_authorises(self) -> None:
        # Not even a small run: the flag above the default cap is itself refused.
        for cap in (1_001, 20_000, 10**12):
            with pytest.raises(GameBudgetExceeded, match="--approval_ref"):
                enforce("training_civ6", self.RESOLVED, max_games=cap)
        with pytest.raises(GameBudgetExceeded, match="--approval_ref"):
            enforce("benchmark", {"n_games": 2, "num_opponents": 1}, max_games=5_000)

    def test_raised_cap_with_approval_is_allowed_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=game_budget.logger.name):
            games = enforce("training_civ6", self.RESOLVED, max_games=20_000, approval_ref="#9")
        assert games == 10_200
        assert "#9" in caplog.text and "20,000" in caplog.text

    def test_unpriceable_run_proceeds_only_on_approval(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        resolved = {"n_trials": 3, "trial_games": None}
        with pytest.raises(GameBudgetUnpriceable):
            enforce("run_sweep", resolved)
        with caplog.at_level(logging.WARNING, logger=game_budget.logger.name):
            assert enforce("run_sweep", resolved, approval_ref="#7") == -1
        assert "#7" in caplog.text


# ---------------------------------------------------------------------------
# Negative controls through the real entry points
# ---------------------------------------------------------------------------


class TestTrainingCiv6Main:
    def _run(self, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> mock.MagicMock:
        from alpha_zero import training_civ6

        train = mock.MagicMock()
        monkeypatch.setattr(training_civ6, "train", train)
        monkeypatch.setattr(sys, "argv", ["training_civ6", *argv])
        training_civ6.main()
        return train

    def test_defaults_are_refused_before_any_work(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from alpha_zero import training_civ6

        train = mock.MagicMock()
        monkeypatch.setattr(training_civ6, "train", train)
        monkeypatch.setattr(sys, "argv", ["training_civ6"])
        with pytest.raises(SystemExit) as excinfo:
            training_civ6.main()
        err = _assert_refused(excinfo, capsys)
        assert "10,200" in err
        train.assert_not_called()

    def test_dry_run_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        train = self._run(monkeypatch, ["--dry_run"])
        train.assert_called_once()
        assert train.call_args.kwargs["dry_run"] is True

    def test_small_explicit_run_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        train = self._run(monkeypatch, ["--num_games", "10", "--num_iterations", "10"])
        train.assert_called_once()

    def test_approval_ref_passes_and_logs(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=game_budget.logger.name):
            train = self._run(monkeypatch, ["--approval_ref", "#142"])
        train.assert_called_once()
        assert "#142" in caplog.text

    @pytest.mark.parametrize("cap", ["10200", "20000", str(10**12)])
    def test_max_games_flag_alone_never_raises_the_cap(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cap: str
    ) -> None:
        from alpha_zero import training_civ6

        train = mock.MagicMock()
        monkeypatch.setattr(training_civ6, "train", train)
        monkeypatch.setattr(sys, "argv", ["training_civ6", "--max_games", cap])
        with pytest.raises(SystemExit) as excinfo:
            training_civ6.main()
        err = _assert_refused(excinfo, capsys)
        assert "--approval_ref" in err
        train.assert_not_called()

    def test_small_run_with_raised_max_games_is_still_refused(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            self._run(monkeypatch, ["--dry_run", "--max_games", "20000"])
        _assert_refused(excinfo, capsys)

    def test_raised_max_games_with_approval_ref_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        train = self._run(monkeypatch, ["--max_games", "20000", "--approval_ref", "#142"])
        train.assert_called_once()

    def test_max_games_flag_can_lower_the_cap(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            self._run(
                monkeypatch, ["--num_games", "10", "--num_iterations", "10", "--max_games", "50"]
            )
        _assert_refused(excinfo, capsys)


class TestBenchmarkMain:
    def _main(self, argv: list[str], tmp_path: Path) -> tuple[int, mock.MagicMock]:
        from alpha_zero.evaluation import benchmark

        run = mock.MagicMock(return_value={})
        with (
            mock.patch.object(benchmark, "run_benchmark", run),
            mock.patch.object(benchmark, "render_report", return_value=""),
        ):
            code = benchmark.main([*argv, "--out", str(tmp_path / "out")])
        return code, run

    def test_200_games_against_three_default_opponents_is_allowed(self, tmp_path: Path) -> None:
        code, run = self._main(["--n-games", "200"], tmp_path)
        assert code == 0
        run.assert_called_once()

    def test_over_cap_variant_is_refused_before_running(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from alpha_zero.evaluation import benchmark

        run = mock.MagicMock()
        with mock.patch.object(benchmark, "run_benchmark", run):
            with pytest.raises(SystemExit) as excinfo:
                benchmark.main(["--n-games", "400", "--out", str(tmp_path / "out")])
        err = _assert_refused(excinfo, capsys)
        assert "1,200" in err
        run.assert_not_called()

    def test_opponent_checkpoints_count_toward_the_price(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from alpha_zero.evaluation import benchmark

        # 3 default opponents + 1 checkpoint = 4 x 252 = 1,008 > 1,000. The
        # checkpoint is never loaded: the refusal comes first.
        with pytest.raises(SystemExit) as excinfo:
            benchmark.main(
                [
                    "--n-games",
                    "252",
                    "--opponent-checkpoint",
                    str(tmp_path / "missing.pt"),
                    "--out",
                    str(tmp_path / "out"),
                ]
            )
        _assert_refused(excinfo, capsys)

    def test_approval_ref_passes_and_logs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=game_budget.logger.name):
            code, run = self._main(["--n-games", "400", "--approval_ref", "#200"], tmp_path)
        assert code == 0
        run.assert_called_once()
        assert "#200" in caplog.text


class TestOrchestratorMain:
    def _main(self, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> tuple[int, mock.MagicMock]:
        from alpha_zero.orchestrator import __main__ as cli
        from alpha_zero.orchestrator import orchestrator as orch_module

        orchestrator = mock.MagicMock()
        orchestrator.dry_run.return_value = True
        cls = mock.MagicMock(return_value=orchestrator)
        monkeypatch.setattr(orch_module, "TrainingOrchestrator", cls)
        monkeypatch.setattr(sys, "argv", ["orchestrator", *argv])
        return cli.main(), cls

    def test_production_preset_is_refused_before_building_anything(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from alpha_zero.orchestrator import __main__ as cli
        from alpha_zero.orchestrator import orchestrator as orch_module

        cls = mock.MagicMock()
        monkeypatch.setattr(orch_module, "TrainingOrchestrator", cls)
        monkeypatch.setattr(sys, "argv", ["orchestrator", "--preset", "production"])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        err = _assert_refused(excinfo, capsys)
        assert "100,060,000" in err
        cls.assert_not_called()

    def test_default_config_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            self._main(monkeypatch, [])
        _assert_refused(excinfo, capsys)

    def test_fast_preset_dry_run_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        code, cls = self._main(monkeypatch, ["--preset", "fast", "--dry_run"])
        assert code == 0
        cls.assert_called_once()

    def test_override_that_shrinks_a_preset_below_the_cap_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, cls = self._main(
            monkeypatch,
            ["--preset", "production", "--iterations", "2", "--games-per-iteration", "5"],
        )
        assert code == 0
        cls.return_value.run.assert_called_once()

    def test_approval_ref_passes_and_logs(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=game_budget.logger.name):
            code, cls = self._main(
                monkeypatch, ["--preset", "production", "--approval_ref", "#300"]
            )
        assert code == 0
        cls.return_value.run.assert_called_once()
        assert "#300" in caplog.text


# ---------------------------------------------------------------------------
# The remaining runners: each refuses through its real main()
# ---------------------------------------------------------------------------


def _import_script(name: str) -> Any:
    sys.path.insert(0, str(_ROOT / "scripts"))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(_ROOT / "scripts"))


class TestScriptRunnersRefuse:
    def test_run_baseline_eval(self, capsys: pytest.CaptureFixture[str]) -> None:
        script = _import_script("run_baseline_eval")
        with pytest.raises(SystemExit) as excinfo:
            script.main(["--model_version", "1", "--n_games", "501"])  # 2 x 501 = 1,002
        err = _assert_refused(excinfo, capsys)
        assert "1,002" in err

    def test_run_encoder_ab_train(self, capsys: pytest.CaptureFixture[str]) -> None:
        script = _import_script("run_encoder_ab")
        with pytest.raises(SystemExit) as excinfo:
            script.main(["train", "--iterations", "100", "--games-per-iter", "14"])
        err = _assert_refused(excinfo, capsys)
        assert "2,800" in err  # 2 arms x 100 x 14

    def test_run_encoder_ab_arena(self, capsys: pytest.CaptureFixture[str]) -> None:
        script = _import_script("run_encoder_ab")
        with pytest.raises(SystemExit) as excinfo:
            script.main(["arena", "--num_games", "1500"])
        _assert_refused(excinfo, capsys)

    def test_run_encoder_ab_arena_default_is_allowed(self, tmp_path: Path) -> None:
        script = _import_script("run_encoder_ab")
        # Passes the budget (800 <= 1,000) and then stops at the first missing
        # checkpoint -- proof the check let the run proceed.
        code = script.main(["arena", "--checkpoint-dir", str(tmp_path)])
        assert code == 1

    def test_run_chess_training(self, capsys: pytest.CaptureFixture[str]) -> None:
        script = _import_script("run_chess_training")
        with pytest.raises(SystemExit) as excinfo:
            script.main(["--iterations", "50", "--games-per-iter", "50"])
        _assert_refused(excinfo, capsys)

    def test_run_enhanced_training(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script = _import_script("run_enhanced_training")
        run = mock.MagicMock()
        monkeypatch.setattr(script, "run_enhanced_training", run)
        monkeypatch.setattr(script, "setup_logging", mock.MagicMock())
        monkeypatch.setattr(sys, "argv", ["run_enhanced_training", "--preset", "full"])
        with pytest.raises(SystemExit) as excinfo:
            script.main()
        _assert_refused(excinfo, capsys)
        run.assert_not_called()

    def test_run_enhanced_training_fast_preset_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = _import_script("run_enhanced_training")
        run = mock.MagicMock()
        monkeypatch.setattr(script, "run_enhanced_training", run)
        monkeypatch.setattr(script, "setup_logging", mock.MagicMock())
        monkeypatch.setattr(sys, "argv", ["run_enhanced_training"])
        script.main()
        run.assert_called_once()

    def test_run_muzero_training(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script = _import_script("run_muzero_training")
        train = mock.MagicMock()
        monkeypatch.setattr(script, "train", train)
        monkeypatch.setattr(sys, "argv", ["run_muzero_training", "--iterations", "60"])
        with pytest.raises(SystemExit) as excinfo:
            script.main()  # 60 x 20 = 1,200
        _assert_refused(excinfo, capsys)
        train.assert_not_called()

    def test_run_sweep(self, capsys: pytest.CaptureFixture[str]) -> None:
        script = _import_script("run_sweep")
        with pytest.raises(SystemExit) as excinfo:
            script.main(["--n-trials", "101"])  # 101 trials x 10 games
        _assert_refused(excinfo, capsys)

    def test_run_sweep_prices_with_the_runners_own_parser(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--gam`` is an abbreviation the child accepts: 20 x 30 x 2 trials = 1,200."""
        config = tmp_path / "sweep.yaml"
        lines = [
            "study_name: s",
            "n_trials: 2",
            "metric: m",
            "command:",
            '  - "{python}"',
            "  - scripts/run_muzero_training.py",
            "  - --iterations",
            '  - "20"',
            "  - --gam",
            '  - "30"',
            "search_space:",
            "  lr: {type: float, low: 0.001, high: 0.01}",
        ]
        config.write_text(os.linesep.join(lines) + os.linesep, encoding="utf-8")
        script = _import_script("run_sweep")
        cmd = ["py", "scripts/run_muzero_training.py", "--iterations", "20", "--gam", "30"]
        assert script.trial_games(cmd, {}) == 600
        with pytest.raises(SystemExit) as excinfo:
            script.main(["--config", str(config)])
        err = _assert_refused(excinfo, capsys)
        assert "1,200" in err

    def test_run_sweep_ambiguous_abbreviation_is_unpriceable(self) -> None:
        script = _import_script("run_sweep")
        # "--w" matches --workers, --wandb ...: the child would die, so refuse.
        assert script.trial_games(["py", "run_muzero_training.py", "--w", "1"], {}) is None

    def test_run_sweep_prices_presets_and_placeholders(self) -> None:
        script = _import_script("run_sweep")
        cmd = ["{python}", "scripts/run_muzero_training.py", "--quick", "--lr={lr}"]
        space = {"lr": {"type": "float", "low": 0.001, "high": 0.01}}
        assert script.trial_games(cmd, space) == 10
        assert script.trial_games(["py", "something_else.py"], {}) is None
        assert script.trial_games(["py", "run_muzero_training.py", "--lr={nope}"], {}) is None

    def test_run_sweep_dry_run_is_allowed(self) -> None:
        script = _import_script("run_sweep")
        assert script.main(["--dry-run", "--n-trials", "1000"]) == 0

    def test_post_run_evaluation(self, capsys: pytest.CaptureFixture[str]) -> None:
        script = _import_script("post_run_evaluation")
        with pytest.raises(SystemExit) as excinfo:
            script.main(
                [
                    "--gcs_bucket",
                    "b",
                    "--gcp_project",
                    "p",
                    "--model_start",
                    "1",
                    "--model_end",
                    "60",
                    "--model_stride",
                    "1",
                ]
            )  # C(60, 2) x 10 = 17,700
        err = _assert_refused(excinfo, capsys)
        assert "17,740" in err

    def test_validate_convergence(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script = _import_script("validate_convergence")
        run = mock.MagicMock(return_value=0)
        monkeypatch.setattr(script, "run", run)
        monkeypatch.setattr(sys, "argv", ["validate_convergence", "--iterations", "100"])
        with pytest.raises(SystemExit) as excinfo:
            script.main()  # 100 x (10 + 10)
        _assert_refused(excinfo, capsys)
        run.assert_not_called()

    def test_benchmark_batch_selfplay(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script = _import_script("benchmark_batch_selfplay")
        create_model = mock.MagicMock()
        monkeypatch.setattr(script, "create_model", create_model)
        monkeypatch.setattr(sys, "argv", ["benchmark_batch_selfplay", "--games", "600"])
        with pytest.raises(SystemExit) as excinfo:
            script.main()  # 2 x (600 + 2)
        _assert_refused(excinfo, capsys)
        create_model.assert_not_called()

    def test_local_muzero_launcher(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script = _import_script("local_muzero_launcher")
        storage = mock.MagicMock()
        monkeypatch.setattr(script, "GCSModelStorage", storage)
        monkeypatch.setattr(
            sys, "argv", ["local_muzero_launcher", "--iterations", "50", "--games_per_worker", "11"]
        )
        with pytest.raises(SystemExit) as excinfo:
            script.main()  # 50 x 2 x 11 = 1,100
        _assert_refused(excinfo, capsys)
        storage.assert_not_called()


class TestLocalMuzeroLauncherGcs:
    """A script named "local" that writes through GCS must say so (final review M-6)."""

    def test_refused_without_allow_gcs_before_touching_storage(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script = _import_script("local_muzero_launcher")
        storage = mock.MagicMock()
        monkeypatch.setattr(script, "GCSModelStorage", storage)
        monkeypatch.setattr(sys, "argv", ["local_muzero_launcher", "--iterations", "1"])
        with pytest.raises(SystemExit) as excinfo:
            script.main()
        assert excinfo.value.code == EXIT_CODE_REFUSED
        assert "--allow-gcs" in capsys.readouterr().err
        storage.assert_not_called()

    def test_allow_gcs_reaches_storage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        script = _import_script("local_muzero_launcher")
        storage = mock.MagicMock(side_effect=RuntimeError("stop here"))
        monkeypatch.setattr(script, "GCSModelStorage", storage)
        monkeypatch.setattr(
            sys, "argv", ["local_muzero_launcher", "--iterations", "1", "--allow-gcs"]
        )
        with mock.patch.dict(os.environ), pytest.raises(RuntimeError, match="stop here"):
            script.main()
        storage.assert_called_once()


class TestCalibrateThroughput:
    """``--games`` reached ``run_self_play`` unchecked before it was priced."""

    @staticmethod
    def _script_and_spy(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, mock.MagicMock]:
        script = _import_script("calibrate_throughput")
        calibrate = mock.MagicMock(return_value={"ms_per_eval": 1.0})
        monkeypatch.setattr(script, "calibrate", calibrate)
        return script, calibrate

    def test_over_the_cap_is_refused_before_anything_runs(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script, calibrate = self._script_and_spy(monkeypatch)
        with pytest.raises(SystemExit) as excinfo:
            script.main(["--games", "1001"])
        _assert_refused(excinfo, capsys)
        calibrate.assert_not_called()

    def test_a_small_run_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script, calibrate = self._script_and_spy(monkeypatch)
        assert script.main(["--games", "2"]) == 0
        calibrate.assert_called_once()
        capsys.readouterr()

    def test_approval_ref_raises_the_cap(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script, calibrate = self._script_and_spy(monkeypatch)
        assert script.main(["--games", "1001", "--approval_ref", "#142"]) == 0
        calibrate.assert_called_once()
        capsys.readouterr()

    def test_max_games_can_lower_the_cap(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script, calibrate = self._script_and_spy(monkeypatch)
        with pytest.raises(SystemExit) as excinfo:
            script.main(["--games", "5", "--max_games", "4"])
        _assert_refused(excinfo, capsys)
        calibrate.assert_not_called()


class TestRunCiv6Ai:
    def test_hard_coded_budget_is_10_by_10(self, monkeypatch: pytest.MonkeyPatch) -> None:
        script = _import_script("run_civ6_ai")
        run = mock.MagicMock()
        monkeypatch.setattr(script.subprocess, "run", run)
        monkeypatch.setattr(sys, "argv", ["run_civ6_ai"])
        script.main()
        cmd = run.call_args.args[0]
        assert "--num_games=10" in cmd and "--num_iterations=10" in cmd
        # 10 x 10 + 20 x 1 evaluation: comfortably under the cap.
        assert (
            worst_case_games(
                "training_civ6", {"num_games": 10, "num_iterations": 10, "save_interval": 10}
            )
            <= DEFAULT_MAX_GAMES
        )


class TestProductionLauncher:
    """``launch_50iter_production_monitored`` built a production orchestrator unpriced (I-2)."""

    @staticmethod
    def _main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> tuple[int, mock.MagicMock]:
        from alpha_zero.orchestrator import orchestrator as orch_module

        script = _import_script("launch_50iter_production_monitored")
        orchestrator = mock.MagicMock()
        orchestrator.initialize.return_value = True
        cls = mock.MagicMock(return_value=orchestrator)
        monkeypatch.setattr(orch_module, "TrainingOrchestrator", cls)
        # The launcher writes its overrides straight into os.environ; keep them out
        # of every later test.
        with mock.patch.dict(os.environ):
            for key in [k for k in os.environ if k.startswith("CIV6AZ_TRAINING_")]:
                del os.environ[key]
            return script.main(argv), cls

    def test_defaults_are_refused_before_building_the_orchestrator(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            self._main(monkeypatch, [])
        err = _assert_refused(excinfo, capsys)
        # 50 iterations x 200 games + 50 games x (1 evaluation + 2 x 1 full evaluation).
        assert "10,150" in err

    def test_defaults_price_through_the_orchestrator_model(self) -> None:
        resolved = {
            "dry_run": False,
            "num_iterations": 50,
            "games_per_iteration": 200,
            "evaluation_interval": 2500,
            "evaluation_games": 50,
            "full_eval_interval": 5,
        }
        assert worst_case_games("orchestrator", resolved) == 10_150

    def test_refusal_builds_nothing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from alpha_zero.orchestrator import orchestrator as orch_module

        cls = mock.MagicMock()
        monkeypatch.setattr(orch_module, "TrainingOrchestrator", cls)
        script = _import_script("launch_50iter_production_monitored")
        with mock.patch.dict(os.environ), pytest.raises(SystemExit) as excinfo:
            script.main([])
        _assert_refused(excinfo, capsys)
        cls.assert_not_called()

    def test_dry_run_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        code, cls = self._main(monkeypatch, ["--dry-run"])
        assert code == 0
        cls.assert_not_called()

    def test_approval_ref_passes_and_logs(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=game_budget.logger.name):
            code, cls = self._main(monkeypatch, ["--approval_ref", "#301"])
        assert code == 0
        cls.return_value.run.assert_called_once()
        assert "#301" in caplog.text

    def test_max_games_flag_alone_never_raises_the_cap(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            self._main(monkeypatch, ["--max_games", "20000"])
        # Not argparse's own exit 2 for an unknown flag: the refusal message.
        _assert_refused(excinfo, capsys)


class TestEnhancedEnvSmokeTest:
    """``--num-selfplay-games`` reached self-play unchecked (I-2)."""

    @staticmethod
    def _script_and_spy(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, mock.MagicMock]:
        script = _import_script("enhanced_env_smoke_test")
        selfplay = mock.MagicMock(
            return_value=script.SmokeResult("local_selfplay", True, 0.0, "stub")
        )
        monkeypatch.setattr(script, "test_local_selfplay", selfplay)
        for name in (
            "test_env_creation",
            "test_observation_shapes",
            "test_enhanced_systems",
            "test_vectorized_env",
            "test_gcs_path_isolation",
        ):
            monkeypatch.setattr(
                script,
                name,
                mock.MagicMock(return_value=script.SmokeResult(name, True, 0.0, "stub")),
            )
        return script, selfplay

    def test_model_prices_one_game_per_selfplay_game(self) -> None:
        assert worst_case_games("enhanced_env_smoke_test", {"num_selfplay_games": 2}) == 2
        assert worst_case_games("enhanced_env_smoke_test", {"num_selfplay_games": 1001}) == 1001

    def test_default_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script, selfplay = self._script_and_spy(monkeypatch)
        assert script.main(["--skip-container"]) == 0
        selfplay.assert_called_once()
        capsys.readouterr()

    def test_over_the_cap_is_refused_before_anything_runs(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script, selfplay = self._script_and_spy(monkeypatch)
        with pytest.raises(SystemExit) as excinfo:
            script.main(["--skip-container", "--num-selfplay-games", "1001"])
        err = _assert_refused(excinfo, capsys)
        assert "1,001" in err
        selfplay.assert_not_called()

    def test_approval_ref_raises_the_cap(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script, selfplay = self._script_and_spy(monkeypatch)
        argv = ["--skip-container", "--num-selfplay-games", "1001", "--approval_ref", "#302"]
        assert script.main(argv) == 0
        selfplay.assert_called_once()
        capsys.readouterr()
