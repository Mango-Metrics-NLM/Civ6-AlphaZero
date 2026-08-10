"""Unit tests for the encoder A/B driver's pure logic.

The driver's expensive parts (self-play, training, arena) are exercised by a
smoke run, not here. What these tests cover is everything that decides whether
the *evidence* is trustworthy: the statistics, the self-consistency gate, the
checkpoint payload contract, and the reporting rules that keep the output from
over-claiming.

The consistency gate matters most. A malformed arena result — counts that do not
add up, a rate outside its own interval — must never reach a committed report,
because a plausible-looking table is harder to catch than a crash.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

_DRIVER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_encoder_ab.py"

_spec = importlib.util.spec_from_file_location("run_encoder_ab", _DRIVER_PATH)
assert _spec is not None and _spec.loader is not None
driver = importlib.util.module_from_spec(_spec)
sys.modules["run_encoder_ab"] = driver
_spec.loader.exec_module(driver)


def _arena(wins: int = 60, draws: int = 20, losses: int = 20, **overrides: Any) -> dict[str, Any]:
    """Build a self-consistent arena result for use as a test fixture."""
    total = wins + draws + losses
    score = wins + 0.5 * draws
    rate = score / total
    low, high = driver.wilson_interval(score, total)
    result = {
        "model_a": {"encoder_type": "hex_crystalline", "checkpoint": "a.pt"},
        "model_b": {"encoder_type": "patch_embed", "checkpoint": "b.pt"},
        "num_games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "score_rate": rate,
        "score_rate_ci95": [low, high],
        "ci95_half_width": (high - low) / 2,
        "elo_difference": driver.elo_difference(rate),
        "significant_at_95pct": low > 0.5 or high < 0.5,
        "wall_clock_seconds": 1.0,
        "provenance": {"config": {"board_size": 8}, "device": "cpu"},
    }
    result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestWilsonInterval:
    """The interval is what stops the report from over-claiming."""

    def test_interval_brackets_the_point_estimate(self) -> None:
        low, high = driver.wilson_interval(60, 100)

        assert low < 0.6 < high

    def test_interval_stays_within_zero_and_one(self) -> None:
        """Unlike the normal approximation, Wilson cannot escape [0, 1]."""
        for successes, total in [(0, 10), (10, 10), (1, 1000), (999, 1000)]:
            low, high = driver.wilson_interval(successes, total)

            assert 0.0 <= low <= high <= 1.0, f"{successes}/{total} produced [{low}, {high}]"

    def test_interval_narrows_as_the_sample_grows(self) -> None:
        """More games must buy more precision, or the sample size is pointless."""
        widths = []
        for total in (100, 400, 1600):
            low, high = driver.wilson_interval(total * 0.55, total)
            widths.append(high - low)

        assert widths == sorted(widths, reverse=True)

    def test_a_small_sample_cannot_resolve_a_small_edge(self) -> None:
        """At 200 games a 55% score rate still spans 0.5 — the honest answer."""
        low, high = driver.wilson_interval(110, 200)

        assert low < 0.5 < high

    def test_a_large_sample_can_resolve_the_same_edge(self) -> None:
        """At 1600 games the same rate separates from chance."""
        low, _high = driver.wilson_interval(880, 1600)

        assert low > 0.5

    def test_zero_games_yields_the_uninformative_interval(self) -> None:
        assert driver.wilson_interval(0, 0) == (0.0, 1.0)


class TestEloDifference:
    """Elo is a presentation of the score rate, and must stay consistent with it."""

    def test_even_score_is_zero_elo(self) -> None:
        assert driver.elo_difference(0.5) == pytest.approx(0.0, abs=1e-9)

    def test_winning_is_positive_and_losing_is_negative(self) -> None:
        assert driver.elo_difference(0.75) > 0
        assert driver.elo_difference(0.25) < 0

    def test_is_antisymmetric_about_one_half(self) -> None:
        assert driver.elo_difference(0.7) == pytest.approx(-driver.elo_difference(0.3))

    def test_degenerate_rates_stay_finite(self) -> None:
        """A clean sweep must not produce an infinity in the report."""
        for rate in (0.0, 1.0):
            assert math.isfinite(driver.elo_difference(rate))


# ---------------------------------------------------------------------------
# Consistency gate
# ---------------------------------------------------------------------------


class TestConsistencyGate:
    """Nothing internally contradictory may reach a committed report."""

    def test_a_well_formed_result_passes(self) -> None:
        assert driver.check_consistency(_arena()) == []

    def test_counts_that_do_not_sum_are_rejected(self) -> None:
        problems = driver.check_consistency(_arena(num_games=999))

        assert any("num_games" in p for p in problems)

    def test_a_mismatched_score_is_rejected(self) -> None:
        problems = driver.check_consistency(_arena(score=12.5))

        assert any("score" in p for p in problems)

    def test_a_rate_outside_its_interval_is_rejected(self) -> None:
        problems = driver.check_consistency(_arena(score_rate_ci95=[0.9, 0.95]))

        assert any("outside its own confidence interval" in p for p in problems)

    def test_an_interval_outside_zero_one_is_rejected(self) -> None:
        problems = driver.check_consistency(_arena(score_rate_ci95=[-0.1, 1.4]))

        assert any("escapes" in p for p in problems)

    def test_an_overstated_significance_claim_is_rejected(self) -> None:
        """Claiming significance the interval does not support is the key failure."""
        problems = driver.check_consistency(
            _arena(wins=30, draws=40, losses=30, significant_at_95pct=True)
        )

        assert any("significance" in p for p in problems)


# ---------------------------------------------------------------------------
# Checkpoint payload
# ---------------------------------------------------------------------------


class TestCheckpointPayload:
    """The payload must match the one format that records the encoder type."""

    @pytest.fixture
    def model(self) -> Any:
        from alpha_zero.cloud.gcp.training_worker import create_model

        return create_model(
            board_size=8, encoder_type="hex_crystalline", num_res_blocks=2, num_heads=4
        )

    def test_payload_has_the_expected_top_level_keys(self, model: Any) -> None:
        payload = driver.build_checkpoint_payload(model, "hex_crystalline")

        assert set(payload) == {"state_dict", "model_config"}

    def test_payload_records_the_encoder_type(self, model: Any) -> None:
        """Without this, a reloaded checkpoint silently becomes patch_embed."""
        payload = driver.build_checkpoint_payload(model, "hex_crystalline")

        assert payload["model_config"]["encoder_type"] == "hex_crystalline"

    def test_model_config_is_json_serializable(self, model: Any) -> None:
        """Enums must be flattened, or the config cannot become a config.json."""
        payload = driver.build_checkpoint_payload(model, "hex_crystalline")

        restored = json.loads(json.dumps(payload["model_config"]))

        assert restored["encoder_type"] == "hex_crystalline"

    def test_state_dict_keeps_the_historical_key_namespace(self, model: Any) -> None:
        payload = driver.build_checkpoint_payload(model, "hex_crystalline")

        assert any(k.startswith("encoder.patch_embed.") for k in payload["state_dict"])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestSeeding:
    """Seeding covers model construction; game streams need the env seed."""

    def test_same_seed_produces_identical_initial_weights(self) -> None:
        import torch

        from alpha_zero.cloud.gcp.training_worker import create_model

        driver.set_global_seed(1234)
        first = create_model(
            board_size=8, encoder_type="hex_crystalline", num_res_blocks=2, num_heads=4
        ).state_dict()
        driver.set_global_seed(1234)
        second = create_model(
            board_size=8, encoder_type="hex_crystalline", num_res_blocks=2, num_heads=4
        ).state_dict()

        assert first.keys() == second.keys()
        for key, value in first.items():
            assert torch.equal(value, second[key]), f"{key} differs under a fixed seed"

    def test_different_seeds_produce_different_weights(self) -> None:
        """Confirms the previous test is actually constraining something."""
        import torch

        from alpha_zero.cloud.gcp.training_worker import create_model

        driver.set_global_seed(1)
        first = create_model(
            board_size=8, encoder_type="patch_embed", num_res_blocks=2, num_heads=4
        ).state_dict()
        driver.set_global_seed(2)
        second = create_model(
            board_size=8, encoder_type="patch_embed", num_res_blocks=2, num_heads=4
        ).state_dict()

        assert any(not torch.equal(v, second[k]) for k, v in first.items())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestRunConfig:
    """Configuration is frozen, fully defaulted, and device-agnostic."""

    def test_config_is_immutable(self) -> None:
        config = driver.ABRunConfig()

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.iterations = 99  # type: ignore[misc]

    def test_config_is_json_serializable(self) -> None:
        json.dumps(driver.ABRunConfig().to_dict())

    def test_auto_device_resolves_to_something_concrete(self) -> None:
        resolved = driver.ABRunConfig(device="auto").resolved_device()

        assert resolved and resolved != "auto"

    def test_explicit_device_is_respected(self) -> None:
        assert driver.ABRunConfig(device="cpu").resolved_device() == "cpu"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    """The report must not claim more than the numbers support."""

    def test_report_includes_counts_and_interval(self) -> None:
        text = driver.render_report(_arena(), None)

        assert "Wilson 95% CI" in text
        assert "| Games | 100 |" in text

    def test_non_significant_result_carries_an_explicit_warning(self) -> None:
        """A wide interval must be called out, not left for the reader to notice."""
        text = driver.render_report(_arena(wins=30, draws=40, losses=30), None)

        assert "does not distinguish" in text

    def test_significant_result_omits_the_warning(self) -> None:
        text = driver.render_report(_arena(wins=900, draws=50, losses=50), None)

        assert "does not distinguish" not in text

    def test_report_always_states_single_run_variance(self) -> None:
        """The one-training-run caveat is unconditional."""
        text = driver.render_report(_arena(wins=900, draws=50, losses=50), None)

        assert "trained **once**" in text

    def test_budget_comes_from_training_not_the_arena(self) -> None:
        """The arena is a separate invocation with its own defaults.

        Reporting the arena's config would misstate how the arms were trained.
        """
        arena = _arena()
        arena["provenance"]["config"] = {"iterations": 999, "board_size": 8}
        training = {"config": {"iterations": 7, "board_size": 8}, "device": "cpu"}

        text = driver.render_report(arena, None, training)

        assert "| `iterations` | 7 |" in text
        assert "999" not in text

    def test_summary_table_is_compact_and_quantified(self) -> None:
        table = driver.render_summary_table(_arena())

        assert "Wilson 95% CI" in table
        assert "W/D/L" in table


# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------


class TestEnvironmentCoercion:
    """Env overrides are cast from the field's runtime type.

    Annotation-based dispatch does not work here: ``from __future__ import
    annotations`` turns every field type into a string, so a naive
    implementation silently casts everything one way.
    """

    def test_int_field_is_cast(self) -> None:
        assert driver._coerce_env_value("50", 20) == 50

    def test_float_field_is_cast(self) -> None:
        assert driver._coerce_env_value("0.5", 1e-3) == pytest.approx(0.5)

    def test_string_field_is_left_alone(self) -> None:
        assert driver._coerce_env_value("cuda", "auto") == "cuda"

    @pytest.mark.parametrize("raw", ["true", "TRUE", "1", "yes", "on"])
    def test_truthy_bools(self, raw: str) -> None:
        assert driver._coerce_env_value(raw, False) is True

    @pytest.mark.parametrize("raw", ["false", "0", "no", "off"])
    def test_falsey_bools(self, raw: str) -> None:
        assert driver._coerce_env_value(raw, True) is False

    def test_bool_is_not_parsed_as_int(self) -> None:
        """``bool`` subclasses ``int``; the bool branch must come first."""
        assert driver._coerce_env_value("true", True) is True

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            driver._coerce_env_value("not-a-number", 20)

    def test_env_override_applies_to_any_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fields with no CLI flag are still reachable, and correctly typed."""
        monkeypatch.setenv("CIV6AZ_AB_MCTS_SIMS", "77")
        args = driver.build_parser().parse_args(["train"])

        assert driver._config_from_args(args).mcts_sims == 77

    def test_cli_flag_beats_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6AZ_AB_ITERATIONS", "99")
        args = driver.build_parser().parse_args(["train", "--iterations", "3"])

        assert driver._config_from_args(args).iterations == 3

    def test_bad_env_value_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6AZ_AB_MCTS_SIMS", "12.5")
        args = driver.build_parser().parse_args(["train"])

        with pytest.raises(ValueError, match="CIV6AZ_AB_MCTS_SIMS"):
            driver._config_from_args(args)


# ---------------------------------------------------------------------------
# Controlled-comparison verification
# ---------------------------------------------------------------------------


class TestArmProvenanceComparison:
    """The report may not claim a controlled comparison without checking."""

    def _provenance(self, **config: Any) -> dict[str, Any]:
        base = {"iterations": 6, "seed": 42, "mcts_sims": 25}
        return {"git_revision": "abc123", "config": {**base, **config}}

    def test_identical_arms_report_no_divergence(self) -> None:
        provenances = {"patch_embed": self._provenance(), "hex_crystalline": self._provenance()}

        assert driver.compare_arm_provenance(provenances) == []

    def test_differing_budget_is_reported(self) -> None:
        provenances = {
            "patch_embed": self._provenance(iterations=5),
            "hex_crystalline": self._provenance(iterations=50),
        }

        divergence = driver.compare_arm_provenance(provenances)

        assert any("iterations" in d for d in divergence)

    def test_differing_code_revision_is_reported(self) -> None:
        """The historical A/B failed exactly this way — mismatched images."""
        provenances = {
            "patch_embed": self._provenance(),
            "hex_crystalline": {"git_revision": "def456", "config": self._provenance()["config"]},
        }

        assert "code revision" in driver.compare_arm_provenance(provenances)

    def test_a_single_arm_cannot_diverge(self) -> None:
        assert driver.compare_arm_provenance({"patch_embed": self._provenance()}) == []

    def test_report_warns_when_arms_diverged(self) -> None:
        """A divergent run must not be described as a controlled comparison."""
        text = driver.render_report(_arena(), None, None, ["iterations (a=5, b=50)"])

        assert "NOT trained under identical settings" in text
        assert "differ only in encoder type" not in text

    def test_report_claims_control_only_when_verified(self) -> None:
        text = driver.render_report(_arena(), None, None, [])

        assert "differ only in encoder type" in text
        assert "Verified by comparing" in text
