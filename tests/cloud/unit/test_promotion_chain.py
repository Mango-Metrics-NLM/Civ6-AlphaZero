"""End-to-end tests for the deployed promotion decision chain.

Covers the seam that had no tests at all: ``run_arena`` -> ``ArenaResult`` ->
``save_results`` -> GCS -> ``load_results`` -> promotion verdict -> gate.

Three gaps this closes, all noted during the audit:

* ``load_results`` had zero test coverage in any form.
* No test exercised ``save_results`` -> ``load_results`` as a round trip.
* No test asserted a *promotion*. Every existing gate test asserted rejection,
  so a gate that could never return ``True`` would have passed all of them --
  which is exactly the state the deployed gate was in.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.arena import ArenaResult
from alpha_zero.cloud.gcp.evaluation_worker import load_results, save_results
from alpha_zero.cloud.gcp.vertex_orchestrator import (
    parse_args,
    resolve_evaluation_threshold,
    resolve_promotion,
)
from alpha_zero.evaluation.promotion import PromotionConfig, verdict_from_results
from alpha_zero.evaluation.scoring import PromotionDecision

# The measured record from docs/evidence/hexcrystal_ab/arena_result.json.
DRAW_HEAVY = ArenaResult(wins=142, losses=70, draws=588, total_games=800)
# A clearly stronger candidate.
STRONG = ArenaResult(wins=300, losses=100, draws=0, total_games=400)
# A clearly weaker one.
WEAK = ArenaResult(wins=100, losses=300, draws=0, total_games=400)


class _FakeGCS:
    """In-memory stand-in that exercises the real serialization code."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def upload_blob(self, bucket: str, path: str, data: bytes) -> str:
        self.blobs[path] = data
        return f"gs://{bucket}/{path}"

    def download_blob(self, bucket: str, path: str) -> bytes:
        if path not in self.blobs:
            from alpha_zero.cloud.gcp.storage import GCSStorageError

            raise GCSStorageError(f"missing blob {path}")
        return self.blobs[path]


@pytest.fixture
def gcs() -> _FakeGCS:
    return _FakeGCS()


@pytest.fixture
def config() -> MagicMock:
    cfg = MagicMock()
    cfg.project_id = "test-project"
    cfg.bucket = "test-bucket"
    return cfg


def _round_trip(gcs: _FakeGCS, config: MagicMock, result: ArenaResult) -> dict[str, Any]:
    """Persist an arena result and read it back through the real code paths."""
    backend_cls = MagicMock(return_value=gcs)
    with patch("alpha_zero.cloud.gcp.storage.GCSBackend", backend_cls):
        save_results(
            config=config,
            results=result.to_dict(),
            iteration=7,
            model_version_new=8,
            model_version_old=7,
        )
        loaded = load_results(config=config, iteration=7)
    assert loaded is not None, "load_results returned None for a blob just written"
    return loaded


class TestArtifactRoundTrip:
    def test_counts_survive_the_round_trip(self, gcs: _FakeGCS, config: MagicMock) -> None:
        loaded = _round_trip(gcs, config, DRAW_HEAVY)
        assert loaded["wins"] == 142
        assert loaded["draws"] == 588
        assert loaded["losses"] == 70
        assert loaded["total_games"] == 800

    def test_both_statistics_are_persisted(self, gcs: _FakeGCS, config: MagicMock) -> None:
        loaded = _round_trip(gcs, config, DRAW_HEAVY)
        assert loaded["win_rate"] == pytest.approx(142 / 800)
        assert loaded["score_rate"] == pytest.approx(0.545)

    def test_metadata_is_attached(self, gcs: _FakeGCS, config: MagicMock) -> None:
        loaded = _round_trip(gcs, config, DRAW_HEAVY)
        assert loaded["iteration"] == 7
        assert loaded["model_version_new"] == 8
        assert loaded["model_version_old"] == 7

    def test_missing_artifact_returns_none(self, gcs: _FakeGCS, config: MagicMock) -> None:
        """The deserialize side had no coverage at all before this."""
        with patch("alpha_zero.cloud.gcp.storage.GCSBackend", MagicMock(return_value=gcs)):
            assert load_results(config=config, iteration=999) is None


class TestDrawHandlingThroughTheChain:
    def test_a_draw_is_a_loss_under_the_legacy_policy(
        self, gcs: _FakeGCS, config: MagicMock
    ) -> None:
        loaded = _round_trip(gcs, config, ArenaResult(wins=0, losses=0, draws=10, total_games=10))
        verdict = verdict_from_results(loaded, PromotionConfig(policy="legacy_win_rate"))
        assert verdict is not None
        assert verdict.statistic == 0.0

    def test_a_draw_is_half_a_point_under_score_rate(
        self, gcs: _FakeGCS, config: MagicMock
    ) -> None:
        loaded = _round_trip(gcs, config, ArenaResult(wins=0, losses=0, draws=10, total_games=10))
        verdict = verdict_from_results(loaded, PromotionConfig(policy="score_rate"))
        assert verdict is not None
        assert verdict.statistic == pytest.approx(0.5)

    def test_draw_heavy_record_is_unreachable_under_legacy_but_not_score_rate(
        self, gcs: _FakeGCS, config: MagicMock
    ) -> None:
        """The defect, demonstrated end to end on the measured record."""
        loaded = _round_trip(gcs, config, DRAW_HEAVY)

        legacy = verdict_from_results(loaded, PromotionConfig(policy="legacy_win_rate"))
        score = verdict_from_results(loaded, PromotionConfig(policy="score_rate", threshold=0.5))
        assert legacy is not None and score is not None

        # 17.75% against a 55% bar: the ceiling is 1 - 0.735 = 26.5%.
        assert legacy.statistic == pytest.approx(0.1775)
        assert legacy.promoted is False
        # The same games, scored draw-aware, clear a 50% bar.
        assert score.statistic == pytest.approx(0.545)
        assert score.promoted is True


class TestGateReturnsBothAnswers:
    """No pre-existing test asserted a promotion. These do."""

    def test_strong_candidate_is_promoted(self, gcs: _FakeGCS, config: MagicMock) -> None:
        loaded = _round_trip(gcs, config, STRONG)
        verdict = verdict_from_results(loaded, PromotionConfig(threshold=0.55))
        assert verdict is not None
        assert verdict.promoted is True
        assert resolve_promotion(verdict, loaded["win_rate"], 0.55) is True

    def test_weak_candidate_is_rejected(self, gcs: _FakeGCS, config: MagicMock) -> None:
        loaded = _round_trip(gcs, config, WEAK)
        verdict = verdict_from_results(loaded, PromotionConfig(threshold=0.55))
        assert verdict is not None
        assert verdict.promoted is False
        assert resolve_promotion(verdict, loaded["win_rate"], 0.55) is False

    def test_gate_returns_both_answers_on_the_same_threshold(
        self, gcs: _FakeGCS, config: MagicMock
    ) -> None:
        strong = verdict_from_results(_round_trip(gcs, config, STRONG))
        weak = verdict_from_results(_round_trip(gcs, config, WEAK))
        assert {strong.decision, weak.decision} == {
            PromotionDecision.PROMOTE,
            PromotionDecision.REJECT,
        }


class TestResolverPrefersTheVerdict:
    """The verdict must win over the raw scalar when they disagree.

    Every other test here uses records where both paths agree, so none of them
    can tell whether the resolver actually consults the verdict. A mutation
    that deleted the verdict branch entirely passed all of them. These do not.
    """

    def test_verdict_overrides_a_scalar_that_would_reject(
        self, gcs: _FakeGCS, config: MagicMock
    ) -> None:
        loaded = _round_trip(gcs, config, DRAW_HEAVY)
        win_rate = loaded["win_rate"]
        verdict = verdict_from_results(loaded, PromotionConfig(policy="score_rate", threshold=0.5))
        assert verdict is not None

        # The two paths genuinely disagree on this record.
        assert win_rate < 0.5, "scalar path would reject"
        assert verdict.promoted is True, "policy path would promote"

        assert resolve_promotion(verdict, win_rate, 0.5) is True

    def test_verdict_overrides_a_scalar_that_would_promote(
        self, gcs: _FakeGCS, config: MagicMock
    ) -> None:
        loaded = _round_trip(gcs, config, ArenaResult(wins=6, losses=4, draws=0, total_games=10))
        win_rate = loaded["win_rate"]
        verdict = verdict_from_results(
            loaded, PromotionConfig(policy="score_rate_lower_bound", threshold=0.5)
        )
        assert verdict is not None

        # 6W/4L clears a 0.55 bar on the point estimate but is far too noisy
        # for the interval policy to act on.
        assert win_rate >= 0.55, "scalar path would promote"
        assert verdict.promoted is False, "interval policy would not"

        assert resolve_promotion(verdict, win_rate, 0.55) is False


class TestBackwardsCompatibility:
    def test_stale_artifact_without_counts_falls_back_to_the_legacy_scalar(self) -> None:
        """A pre-change artifact must not read as a rejection."""
        stale = {"win_rate": 0.62, "iteration": 3}
        assert verdict_from_results(stale) is None
        # Falls through to the historical comparison, which promotes at 0.62.
        assert resolve_promotion(None, stale["win_rate"], 0.55) is True
        assert resolve_promotion(None, 0.40, 0.55) is False

    def test_resolver_reproduces_the_historical_comparison_exactly(self) -> None:
        for win_rate in (0.0, 0.3, 0.5499, 0.55, 0.5501, 1.0):
            assert resolve_promotion(None, win_rate, 0.55) is (win_rate >= 0.55)

    def test_default_policy_on_a_real_artifact_matches_the_old_gate(
        self, gcs: _FakeGCS, config: MagicMock
    ) -> None:
        """With no configuration, the new path decides exactly as the old one."""
        for arena in (DRAW_HEAVY, STRONG, WEAK):
            loaded = _round_trip(gcs, config, arena)
            verdict = verdict_from_results(loaded, PromotionConfig(threshold=0.55))
            legacy_answer = loaded["win_rate"] >= 0.55
            assert verdict is not None
            assert verdict.promoted is legacy_answer

    def test_artifact_json_is_plain_and_readable_by_an_older_reader(
        self, gcs: _FakeGCS, config: MagicMock
    ) -> None:
        """An older binary reads win_rate and ignores keys it does not know."""
        _round_trip(gcs, config, DRAW_HEAVY)
        raw = json.loads(next(iter(gcs.blobs.values())).decode("utf-8"))
        assert "win_rate" in raw
        assert isinstance(raw["win_rate"], float)


class TestThresholdPrecedence:
    """--evaluation_win_threshold: explicit CLI > CIV6AZ_ env > default.

    The flag defaults to None precisely so that omitting it leaves the
    documented CIV6AZ_PROMOTION_THRESHOLD opt-in alive; a concrete argparse
    default would silently override the env var on every invocation.
    """

    def test_cli_flag_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6AZ_PROMOTION_THRESHOLD", "0.52")
        monkeypatch.setattr("sys.argv", ["prog", "--evaluation_win_threshold", "0.60"])
        args = parse_args()
        assert resolve_evaluation_threshold(args) == 0.60

    def test_env_applies_when_flag_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6AZ_PROMOTION_THRESHOLD", "0.52")
        monkeypatch.setattr("sys.argv", ["prog"])
        args = parse_args()
        assert args.evaluation_win_threshold is None
        assert resolve_evaluation_threshold(args) == 0.52

    def test_default_when_neither_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CIV6AZ_PROMOTION_THRESHOLD", raising=False)
        monkeypatch.delenv("CIV6_PROMOTION_THRESHOLD", raising=False)
        monkeypatch.setattr("sys.argv", ["prog"])
        args = parse_args()
        assert resolve_evaluation_threshold(args) == 0.55

    def test_verdict_and_scalar_fallback_gate_on_the_same_number(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defect the first fix introduced: env-set threshold dead on the
        verdict path while the scalar fallback still honoured it."""
        monkeypatch.setenv("CIV6AZ_PROMOTION_THRESHOLD", "0.52")
        monkeypatch.setattr("sys.argv", ["prog"])
        threshold = resolve_evaluation_threshold(parse_args())
        config = PromotionConfig(threshold=threshold)
        assert config.threshold == 0.52
