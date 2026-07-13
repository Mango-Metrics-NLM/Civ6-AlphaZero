"""Tests for the Optuna-backed hyperparameter search component.

Covers the declarative search-space contract, SweepConfig env-override
precedence (including the legacy W&B-era SWEEP_TIMEOUT_SECONDS), the
search loop with an injected deterministic objective and mocked tracker,
the subprocess objective (metric extraction, failure pruning, timeout
pruning), and the run_sweep.py CLI.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alpha_zero.optimizations.hyperparameter_search import (
    ENV_PREFIX,
    LEGACY_TIMEOUT_ENV,
    METRICS_MARKER,
    HyperparameterSearch,
    SearchSpaceError,
    SubprocessTrainingObjective,
    SweepConfig,
    suggest_params,
    validate_search_space,
)

optuna = pytest.importorskip("optuna")

_SPACE = {
    "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [32, 64]},
    "blocks": {"type": "int", "low": 2, "high": 6, "step": 2},
}


# ---------------------------------------------------------------------------
# Search-space validation
# ---------------------------------------------------------------------------


class TestValidateSearchSpace:
    def test_valid_space_passes(self):
        validate_search_space(_SPACE)

    def test_empty_space_rejected(self):
        with pytest.raises(SearchSpaceError, match="at least one parameter"):
            validate_search_space({})

    def test_unknown_type_rejected(self):
        with pytest.raises(SearchSpaceError, match=r"lr.*type must be one of"):
            validate_search_space({"lr": {"type": "gaussian"}})

    def test_missing_bounds_rejected(self):
        with pytest.raises(SearchSpaceError, match=r"requires 'low' and 'high'"):
            validate_search_space({"lr": {"type": "float", "low": 0.1}})

    def test_inverted_range_rejected(self):
        with pytest.raises(SearchSpaceError, match=r"low.*must be <= high"):
            validate_search_space({"n": {"type": "int", "low": 9, "high": 1}})

    def test_empty_choices_rejected(self):
        with pytest.raises(SearchSpaceError, match="non-empty 'choices'"):
            validate_search_space({"c": {"type": "categorical", "choices": []}})

    def test_non_mapping_spec_rejected(self):
        with pytest.raises(SearchSpaceError, match="must be a mapping"):
            validate_search_space({"c": [1, 2, 3]})


class TestSuggestParams:
    def test_suggests_within_declared_domain(self):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7))
        collected: list[dict] = []

        def objective(trial):
            params = suggest_params(trial, _SPACE)
            collected.append(params)
            return 0.0

        study.optimize(objective, n_trials=5)
        for params in collected:
            assert 1e-4 <= params["lr"] <= 1e-2
            assert params["batch_size"] in (32, 64)
            assert params["blocks"] in (2, 4, 6)


# ---------------------------------------------------------------------------
# SweepConfig
# ---------------------------------------------------------------------------


class TestSweepConfig:
    def test_defaults_are_valid(self):
        cfg = SweepConfig()
        assert cfg.direction == "maximize"
        assert cfg.n_trials >= 1

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("direction", "sideways", "direction must be one of"),
            ("sampler", "quantum", "sampler must be one of"),
            ("n_trials", 0, "n_trials must be >= 1"),
        ],
    )
    def test_invalid_values_rejected(self, field, value, match):
        with pytest.raises(ValueError, match=match):
            SweepConfig(**{field: value})

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match=r"Unknown SweepConfig keys.*typo"):
            SweepConfig.from_dict({"typo": 1})

    def test_invalid_search_space_rejected_at_construction(self):
        with pytest.raises(SearchSpaceError):
            SweepConfig(search_space={"x": {"type": "nope"}})

    def test_env_overrides_precedence(self, monkeypatch):
        monkeypatch.setenv(f"{ENV_PREFIX}N_TRIALS", "3")
        monkeypatch.setenv(f"{ENV_PREFIX}STUDY_NAME", "env-study")
        monkeypatch.setenv(f"{ENV_PREFIX}TRIAL_TIMEOUT_SECONDS", "60")
        monkeypatch.setenv(LEGACY_TIMEOUT_ENV, "999")  # loses to canonical
        cfg = SweepConfig().apply_env_overrides()
        assert cfg.n_trials == 3
        assert cfg.study_name == "env-study"
        assert cfg.trial_timeout_seconds == 60

    def test_legacy_timeout_honored_when_canonical_unset(self, monkeypatch):
        monkeypatch.delenv(f"{ENV_PREFIX}TRIAL_TIMEOUT_SECONDS", raising=False)
        monkeypatch.setenv(LEGACY_TIMEOUT_ENV, "120")
        cfg = SweepConfig().apply_env_overrides()
        assert cfg.trial_timeout_seconds == 120

    def test_garbage_env_int_warns_and_keeps_default(self, monkeypatch, caplog):
        monkeypatch.setenv(f"{ENV_PREFIX}N_TRIALS", "lots")
        with caplog.at_level("WARNING"):
            cfg = SweepConfig(n_trials=7).apply_env_overrides()
        assert cfg.n_trials == 7
        assert any("Invalid" in rec.message for rec in caplog.records)

    def test_from_yaml_roundtrip(self, tmp_path):
        yaml_text = textwrap.dedent(
            """
            study_name: yaml-study
            direction: minimize
            metric: loss
            n_trials: 2
            search_space:
              lr: {type: float, low: 0.001, high: 0.1}
            command: ["{python}", "train.py", "--lr={lr}"]
            """
        )
        path = tmp_path / "sweep.yaml"
        path.write_text(yaml_text)
        cfg = SweepConfig.from_yaml(path)
        assert cfg.study_name == "yaml-study"
        assert cfg.direction == "minimize"
        assert cfg.search_space["lr"]["type"] == "float"


# ---------------------------------------------------------------------------
# HyperparameterSearch with injected objective + tracker
# ---------------------------------------------------------------------------


def _quadratic_config(n_trials: int = 12) -> SweepConfig:
    return SweepConfig(
        study_name="unit-quadratic",
        direction="minimize",
        metric="loss",
        n_trials=n_trials,
        sampler="random",
        seed=42,
        search_space={"x": {"type": "float", "low": -10.0, "high": 10.0}},
        tracking_enabled=False,
    )


class TestHyperparameterSearch:
    def test_finds_minimum_of_quadratic(self):
        search = HyperparameterSearch(
            _quadratic_config(), objective=lambda p: (p["x"] - 3.0) ** 2
        )
        result = search.run()
        assert result.n_trials == 12
        assert result.best_value == pytest.approx(
            (result.best_params["x"] - 3.0) ** 2
        )
        # Random sampling over [-10, 10] with 12 seeded trials lands near 3.
        assert abs(result.best_params["x"] - 3.0) < 5.0

    def test_tracker_receives_per_trial_and_best_logs(self):
        tracker = MagicMock()
        search = HyperparameterSearch(
            _quadratic_config(n_trials=3),
            objective=lambda p: p["x"] ** 2,
            tracker=tracker,
        )
        result = search.run()
        # One log per trial + one best-value log at the end.
        assert tracker.log.call_count == 4
        per_trial_payload = tracker.log.call_args_list[0].args[0]
        assert "sweep/loss" in per_trial_payload
        tracker.set_tag.assert_called_once()
        tracker.finish.assert_called_once()
        assert result.n_pruned == 0

    def test_pruned_trials_counted_not_fatal(self):
        calls = {"n": 0}

        def flaky(params):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise optuna.TrialPruned("simulated failure")
            return params["x"] ** 2

        search = HyperparameterSearch(_quadratic_config(n_trials=6), objective=flaky)
        result = search.run()
        assert result.n_trials == 6
        assert result.n_pruned == 3

    def test_seeded_runs_are_reproducible(self):
        objective = lambda p: (p["x"] + 1.0) ** 2  # noqa: E731
        r1 = HyperparameterSearch(_quadratic_config(), objective=objective).run()
        r2 = HyperparameterSearch(_quadratic_config(), objective=objective).run()
        assert r1.best_params == r2.best_params
        assert r1.best_value == r2.best_value


# ---------------------------------------------------------------------------
# SubprocessTrainingObjective
# ---------------------------------------------------------------------------


def _stub_training_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "stub_training.py"
    script.write_text(textwrap.dedent(body))
    return script


class TestSubprocessTrainingObjective:
    def test_empty_command_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SubprocessTrainingObjective(command=[], metric="loss")

    def test_build_command_expands_placeholders(self):
        obj = SubprocessTrainingObjective(
            command=["{python}", "train.py", "--lr={lr}"], metric="loss"
        )
        cmd = obj.build_command({"lr": 0.01})
        assert cmd == [sys.executable, "train.py", "--lr=0.01"]

    def test_build_command_unknown_placeholder_raises(self):
        obj = SubprocessTrainingObjective(command=["train", "--x={missing}"], metric="m")
        with pytest.raises(ValueError, match="unknown placeholder"):
            obj.build_command({"lr": 0.01})

    def test_extract_metric_uses_last_marker_line(self):
        stdout = (
            f"progress...\n{METRICS_MARKER}{{\"loss\": 9.0}}\n"
            f"more...\n{METRICS_MARKER}{{\"loss\": 1.5}}\n"
        )
        obj = SubprocessTrainingObjective(command=["x"], metric="loss")
        assert obj.extract_metric(stdout) == 1.5

    def test_extract_metric_missing_marker_raises(self):
        obj = SubprocessTrainingObjective(command=["x"], metric="loss")
        with pytest.raises(ValueError, match=r"no .* line found"):
            obj.extract_metric("no marker here")

    def test_extract_metric_missing_key_raises(self):
        obj = SubprocessTrainingObjective(command=["x"], metric="loss")
        with pytest.raises(ValueError, match="not in reported metrics"):
            obj.extract_metric(f"{METRICS_MARKER}{{\"other\": 1.0}}")

    def test_end_to_end_with_stub_script(self, tmp_path):
        script = _stub_training_script(
            tmp_path,
            """
            import json, sys
            lr = float(sys.argv[1].split("=")[1])
            print("training...")
            print("SWEEP_METRICS_JSON=" + json.dumps({"loss": lr * 10}))
            """,
        )
        obj = SubprocessTrainingObjective(
            command=["{python}", str(script), "--lr={lr}"],
            metric="loss",
            timeout_seconds=30,
        )
        assert obj({"lr": 0.5}) == pytest.approx(5.0)

    def test_nonzero_exit_prunes(self, tmp_path):
        script = _stub_training_script(tmp_path, "import sys; sys.exit(3)")
        obj = SubprocessTrainingObjective(
            command=["{python}", str(script)], metric="loss", timeout_seconds=30
        )
        with pytest.raises(optuna.TrialPruned, match="exited with code 3"):
            obj({})

    def test_timeout_prunes(self, tmp_path):
        script = _stub_training_script(tmp_path, "import time; time.sleep(30)")
        obj = SubprocessTrainingObjective(
            command=["{python}", str(script)], metric="loss", timeout_seconds=1
        )
        with pytest.raises(optuna.TrialPruned, match="timed out"):
            obj({})

    def test_missing_metric_output_prunes(self, tmp_path):
        script = _stub_training_script(tmp_path, "print('done, no marker')")
        obj = SubprocessTrainingObjective(
            command=["{python}", str(script)], metric="loss", timeout_seconds=30
        )
        with pytest.raises(optuna.TrialPruned):
            obj({})


# ---------------------------------------------------------------------------
# run_sweep.py CLI
# ---------------------------------------------------------------------------


class TestRunSweepCli:
    def test_dry_run_with_repo_default_config(self):
        import run_sweep

        assert run_sweep.main(["--dry-run"]) == 0

    def test_missing_config_returns_2(self, tmp_path):
        import run_sweep

        assert run_sweep.main(["--config", str(tmp_path / "nope.yaml")]) == 2

    def test_cli_overrides_apply(self, tmp_path, monkeypatch):
        import run_sweep

        captured: dict = {}

        class FakeSearch:
            def __init__(self, config):
                captured["config"] = config

            def run(self):
                from alpha_zero.optimizations.hyperparameter_search import SweepResult

                return SweepResult(
                    best_params={}, best_value=0.0, n_trials=1, n_pruned=0, study_name="x"
                )

        monkeypatch.setattr(run_sweep, "HyperparameterSearch", FakeSearch)
        assert run_sweep.main(["--n-trials", "2", "--study-name", "cli-study"]) == 0
        assert captured["config"].n_trials == 2
        assert captured["config"].study_name == "cli-study"
