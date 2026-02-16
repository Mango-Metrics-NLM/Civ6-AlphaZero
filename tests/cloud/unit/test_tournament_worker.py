"""Unit tests for standalone tournament worker and cloud tournament infrastructure.

Tests:
- resolve_model_versions(): model selection modes (explicit, stride, range)
- _apply_model_subsampling(): stride/range filtering in orchestrator
- VertexTournamentJob: job launch, wait, cancel
- run_tournament(): round-robin matchup logic, ELO computation, dry run
- save_tournament_results(): GCS save / dry run skip
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.arena import ArenaResult
from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.vertex_jobs import (
    JobResult,
    JobSpec,
    JobState,
    VertexAIClient,
    VertexTournamentJob,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test GCPConfig."""
    return GCPConfig(
        project_id="test-project",
        region="us-central1",
        bucket="test-bucket",
        container_image_uri="gcr.io/test/image:latest",
    )


@pytest.fixture
def args():
    """Create test CLI arguments matching orchestrator defaults."""
    return argparse.Namespace(
        tournament_games=4,
        tournament_mcts_sims=8,
        mcts_type="gumbel",
        gumbel_num_actions=16,
        board_size=8,
        action_space_size=376,
        max_turns=15,
        tournament_model_stride=1,
        tournament_model_range=None,
        tournament_cloud=False,
        tournament_machine_type="n1-standard-8",
        tournament_accelerator_type="NVIDIA_TESLA_T4",
        tournament_accelerator_count=1,
    )


@pytest.fixture
def mock_arena():
    """Mock run_arena to return predictable results."""
    with patch("alpha_zero.cloud.gcp.arena.run_arena") as mock:
        mock.return_value = ArenaResult(
            wins=2,
            losses=1,
            draws=1,
            total_games=4,
        )
        yield mock


# =============================================================================
# Mock Vertex AI Client
# =============================================================================


class MockVertexAIClient(VertexAIClient):
    """Mock Vertex AI client for testing tournament jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._job_counter = 0

    def _ensure_initialized(self) -> None:
        pass

    def submit_custom_job(
        self,
        spec: JobSpec,
        staging_bucket: str | None = None,
    ) -> str:
        self._job_counter += 1
        job_name = f"projects/test/locations/us-central1/customJobs/{self._job_counter}"
        self._jobs[job_name] = {
            "display_name": spec.display_name,
            "state": JobState.RUNNING.value,
            "spec": spec,
        }
        return job_name

    def get_job_state(self, job_name: str) -> tuple[str, dict[str, Any]]:
        if job_name not in self._jobs:
            from alpha_zero.cloud.gcp.exceptions import VertexJobError

            raise VertexJobError(f"Job not found: {job_name}")
        job = self._jobs[job_name]
        return job["state"], job

    def cancel_job(self, job_name: str) -> None:
        if job_name in self._jobs:
            self._jobs[job_name]["state"] = JobState.CANCELLED.value

    def wait_for_completion(self, job_name: str, timeout_seconds: int = 3600) -> JobResult:
        if job_name not in self._jobs:
            from alpha_zero.cloud.gcp.exceptions import VertexJobError

            raise VertexJobError(f"Job not found: {job_name}")
        self._jobs[job_name]["state"] = JobState.SUCCEEDED.value
        return JobResult(
            job_id="mock-job-id",
            job_name=job_name,
            state=JobState.SUCCEEDED.value,
            success=True,
            error_message=None,
        )


# =============================================================================
# TestResolveModelVersions
# =============================================================================


class TestResolveModelVersions:
    """Tests for resolve_model_versions() in tournament_worker."""

    def test_explicit_list(self, config):
        """Explicit --model_versions takes priority."""
        from alpha_zero.cloud.gcp.tournament_worker import resolve_model_versions

        args = argparse.Namespace(
            model_versions="5,10,15,20",
            model_stride=1,
            model_range=None,
        )
        result = resolve_model_versions(args, config, dry_run=True)
        assert result == [5, 10, 15, 20]

    def test_explicit_list_deduplicates_and_sorts(self, config):
        """Explicit list is sorted and deduplicated."""
        from alpha_zero.cloud.gcp.tournament_worker import resolve_model_versions

        args = argparse.Namespace(
            model_versions="20,5,10,5",
            model_stride=1,
            model_range=None,
        )
        result = resolve_model_versions(args, config, dry_run=True)
        assert result == [5, 10, 20]

    def test_stride_subsamples(self, config):
        """Stride=3 selects every 3rd model from dry_run list [0..9]."""
        from alpha_zero.cloud.gcp.tournament_worker import resolve_model_versions

        args = argparse.Namespace(
            model_versions=None,
            model_stride=3,
            model_range=None,
        )
        result = resolve_model_versions(args, config, dry_run=True)
        # [0,1,2,3,4,5,6,7,8,9] stride=3 -> [0,3,6,9], first=0, last=9 both included
        assert 0 in result
        assert 9 in result
        assert len(result) == 4

    def test_range_filters(self, config):
        """Range filter limits models to start:end."""
        from alpha_zero.cloud.gcp.tournament_worker import resolve_model_versions

        args = argparse.Namespace(
            model_versions=None,
            model_stride=1,
            model_range="3:7",
        )
        result = resolve_model_versions(args, config, dry_run=True)
        # dry_run gives [0..9], range 3:7 -> [3,4,5,6,7]
        assert result == [3, 4, 5, 6, 7]

    def test_stride_plus_range(self, config):
        """Stride + range work together."""
        from alpha_zero.cloud.gcp.tournament_worker import resolve_model_versions

        args = argparse.Namespace(
            model_versions=None,
            model_stride=2,
            model_range="2:8",
        )
        result = resolve_model_versions(args, config, dry_run=True)
        # range [2,3,4,5,6,7,8], stride=2 -> [2,4,6] + first=2(already in), last=8(added)
        assert 2 in result
        assert 8 in result

    def test_all_models_when_no_filters(self, config):
        """No filters returns all models."""
        from alpha_zero.cloud.gcp.tournament_worker import resolve_model_versions

        args = argparse.Namespace(
            model_versions=None,
            model_stride=1,
            model_range=None,
        )
        result = resolve_model_versions(args, config, dry_run=True)
        assert result == list(range(10))

    def test_empty_explicit_list_falls_through(self, config):
        """Empty model_versions string falls through to all-models mode."""
        from alpha_zero.cloud.gcp.tournament_worker import resolve_model_versions

        args = argparse.Namespace(
            model_versions="",
            model_stride=1,
            model_range=None,
        )
        result = resolve_model_versions(args, config, dry_run=True)
        assert result == list(range(10))

    def test_first_and_last_always_included(self, config):
        """First and last model are always included when subsampling."""
        from alpha_zero.cloud.gcp.tournament_worker import resolve_model_versions

        args = argparse.Namespace(
            model_versions=None,
            model_stride=4,
            model_range=None,
        )
        result = resolve_model_versions(args, config, dry_run=True)
        # [0..9] stride=4 -> [0,4,8] + last=9
        assert result[0] == 0
        assert result[-1] == 9


# =============================================================================
# TestApplyModelSubsampling
# =============================================================================


class TestApplyModelSubsampling:
    """Tests for _apply_model_subsampling() in vertex_orchestrator."""

    def test_no_op_with_stride_1(self, args):
        """Stride=1 and no range returns all models unchanged."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import _apply_model_subsampling

        versions = [0, 1, 2, 3, 4, 5]
        result = _apply_model_subsampling(versions, args)
        assert result == versions

    def test_stride_reduces_count(self, args):
        """Stride > 1 reduces the number of models."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import _apply_model_subsampling

        args.tournament_model_stride = 3
        versions = list(range(10))
        result = _apply_model_subsampling(versions, args)
        assert len(result) < len(versions)
        assert 0 in result
        assert 9 in result

    def test_range_filters_versions(self, args):
        """Range filter limits to start:end range."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import _apply_model_subsampling

        args.tournament_model_range = "3:7"
        versions = list(range(10))
        result = _apply_model_subsampling(versions, args)
        assert all(3 <= v <= 7 for v in result)

    def test_empty_range_returns_empty(self, args):
        """Range that excludes all versions returns empty list."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import _apply_model_subsampling

        args.tournament_model_range = "100:200"
        versions = list(range(10))
        result = _apply_model_subsampling(versions, args)
        assert result == []

    def test_backwards_compatible_no_attrs(self):
        """Works with old args that don't have tournament_model_* fields."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import _apply_model_subsampling

        old_args = argparse.Namespace(
            tournament_games=10,
            mcts_type="gumbel",
        )
        versions = [0, 1, 2, 3, 4]
        result = _apply_model_subsampling(versions, old_args)
        assert result == versions


# =============================================================================
# TestVertexTournamentJob
# =============================================================================


class TestVertexTournamentJob:
    """Tests for VertexTournamentJob in vertex_jobs."""

    def test_launch_returns_job_name(self, config):
        """Launch returns a valid job resource name."""
        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        job_name = job.launch(model_versions=[0, 1, 2, 3])
        assert "customJobs" in job_name

    def test_launch_passes_model_versions(self, config):
        """Model versions are passed as comma-separated arg."""
        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        job.launch(model_versions=[10, 20, 30])
        # Check the spec that was submitted
        submitted_spec = next(iter(client._jobs.values()))["spec"]
        versions_arg = [a for a in submitted_spec.args if a.startswith("--model_versions=")]
        assert len(versions_arg) == 1
        assert "10,20,30" in versions_arg[0]

    def test_launch_configurable_machine(self, config):
        """Machine type and GPU are configurable."""
        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        job.launch(
            model_versions=[0, 1],
            machine_type="a2-highgpu-1g",
            accelerator_type="NVIDIA_TESLA_A100",
            accelerator_count=2,
        )
        submitted_spec = next(iter(client._jobs.values()))["spec"]
        assert submitted_spec.machine_type == "a2-highgpu-1g"
        assert submitted_spec.accelerator_type == "NVIDIA_TESLA_A100"
        assert submitted_spec.accelerator_count == 2

    def test_launch_display_name_includes_model_count(self, config):
        """Display name reflects number of models."""
        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        job.launch(model_versions=[0, 1, 2, 3, 4])
        submitted_spec = next(iter(client._jobs.values()))["spec"]
        assert "5models" in submitted_spec.display_name

    def test_launch_spot_is_false(self, config):
        """Tournament jobs use on-demand VMs (not spot)."""
        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        job.launch(model_versions=[0, 1])
        submitted_spec = next(iter(client._jobs.values()))["spec"]
        assert submitted_spec.spot is False

    def test_launch_timeout_24h(self, config):
        """Tournament job timeout is 24 hours."""
        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        job.launch(model_versions=[0, 1])
        submitted_spec = next(iter(client._jobs.values()))["spec"]
        assert submitted_spec.timeout_seconds == 86400

    def test_wait_for_completion(self, config):
        """Wait returns JobResult with succeeded state."""
        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        job_name = job.launch(model_versions=[0, 1])
        result = job.wait_for_completion(job_name)
        assert result.success is True
        # After wait, _current_job should be cleared
        assert job._current_job is None

    def test_cancel(self, config):
        """Cancel stops the running job."""
        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        job_name = job.launch(model_versions=[0, 1])
        job.cancel()
        state, _ = client.get_job_state(job_name)
        assert state == JobState.CANCELLED.value

    def test_cancel_noop_when_no_job(self, config):
        """Cancel is safe when no job is running."""
        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        job.cancel()  # Should not raise

    def test_launch_requires_container_image(self):
        """Launch raises error when container_image_uri is not configured."""
        from alpha_zero.cloud.gcp.exceptions import VertexJobError

        config = GCPConfig(
            project_id="test",
            region="us-central1",
            bucket="test-bucket",
            container_image_uri=None,
        )
        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        with pytest.raises(VertexJobError, match="container_image_uri"):
            job.launch(model_versions=[0, 1])

    def test_launch_includes_wandb_env_vars(self, config):
        """W&B env vars are included in job env when set."""
        import os

        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        with patch.dict(os.environ, {"WANDB_ENABLED": "true", "WANDB_API_KEY": "test-key"}):
            job.launch(model_versions=[0, 1])
        submitted_spec = next(iter(client._jobs.values()))["spec"]
        assert "WANDB_ENABLED" in submitted_spec.env_vars


# =============================================================================
# TestRunTournament
# =============================================================================


class TestRunTournament:
    """Tests for run_tournament() in tournament_worker."""

    def test_returns_expected_keys(self, config, mock_arena):
        """Tournament results contain all expected keys."""
        from alpha_zero.cloud.gcp.tournament_worker import run_tournament

        result = run_tournament(
            config=config,
            model_versions=[0, 1, 2],
            tournament_games=4,
            dry_run=True,
        )

        assert "model_versions" in result
        assert "elo_ratings" in result
        assert "matches" in result
        assert "best_model" in result
        assert "best_elo" in result
        assert "metadata" in result

    def test_matchup_count_3_models(self, config, mock_arena):
        """3 models produce 3 matchups (3 choose 2)."""
        from alpha_zero.cloud.gcp.tournament_worker import run_tournament

        result = run_tournament(
            config=config,
            model_versions=[0, 1, 2],
            tournament_games=4,
            dry_run=True,
        )
        assert len(result["matches"]) == 3

    def test_matchup_count_5_models(self, config, mock_arena):
        """5 models produce 10 matchups (5 choose 2)."""
        from alpha_zero.cloud.gcp.tournament_worker import run_tournament

        result = run_tournament(
            config=config,
            model_versions=[0, 1, 2, 3, 4],
            tournament_games=4,
            dry_run=True,
        )
        assert len(result["matches"]) == 10

    def test_elo_ratings_for_all_models(self, config, mock_arena):
        """ELO ratings are computed for all models."""
        from alpha_zero.cloud.gcp.tournament_worker import run_tournament

        result = run_tournament(
            config=config,
            model_versions=[0, 1, 2],
            tournament_games=4,
            dry_run=True,
        )
        assert len(result["elo_ratings"]) == 3
        expected_ids = {"v000", "v001", "v002"}
        assert set(result["elo_ratings"].keys()) == expected_ids

    def test_best_model_is_highest_elo(self, config, mock_arena):
        """best_model corresponds to highest ELO rating."""
        from alpha_zero.cloud.gcp.tournament_worker import run_tournament

        result = run_tournament(
            config=config,
            model_versions=[0, 1, 2],
            tournament_games=4,
            dry_run=True,
        )
        max_player = max(result["elo_ratings"].items(), key=lambda x: x[1])
        assert result["best_model"] == max_player[0]
        assert result["best_elo"] == max_player[1]

    def test_elo_ranking_sorted_descending(self, config, mock_arena):
        """ELO ratings dict is sorted descending."""
        from alpha_zero.cloud.gcp.tournament_worker import run_tournament

        result = run_tournament(
            config=config,
            model_versions=[0, 1, 2],
            tournament_games=4,
            dry_run=True,
        )
        elo_values = list(result["elo_ratings"].values())
        assert elo_values == sorted(elo_values, reverse=True)

    def test_skips_with_less_than_2_models(self, config):
        """Returns empty dict with fewer than 2 models."""
        from alpha_zero.cloud.gcp.tournament_worker import run_tournament

        result = run_tournament(
            config=config,
            model_versions=[0],
            tournament_games=4,
            dry_run=True,
        )
        assert result == {}

    def test_metadata_includes_duration(self, config, mock_arena):
        """Metadata includes duration and configuration."""
        from alpha_zero.cloud.gcp.tournament_worker import run_tournament

        result = run_tournament(
            config=config,
            model_versions=[0, 1],
            tournament_games=6,
            mcts_type="gumbel",
            mcts_sims=25,
            board_size=8,
            max_turns=30,
            dry_run=True,
        )
        metadata = result["metadata"]
        assert metadata["total_matchups"] == 1
        assert metadata["total_games"] == 6
        assert metadata["games_per_matchup"] == 6
        assert metadata["mcts_type"] == "gumbel"
        assert metadata["mcts_sims"] == 25
        assert "duration_seconds" in metadata

    def test_elo_conserved_2_players(self, config):
        """Total ELO is conserved (sum = N * 1500)."""
        with patch("alpha_zero.cloud.gcp.arena.run_arena") as mock:
            mock.return_value = ArenaResult(wins=4, losses=0, draws=0, total_games=4)

            from alpha_zero.cloud.gcp.tournament_worker import run_tournament

            result = run_tournament(
                config=config,
                model_versions=[0, 1],
                tournament_games=4,
                dry_run=True,
            )

            total_elo = sum(result["elo_ratings"].values())
            assert abs(total_elo - 3000.0) < 0.01


# =============================================================================
# TestSaveTournamentResults
# =============================================================================


class TestSaveTournamentResults:
    """Tests for save_tournament_results() in tournament_worker."""

    def test_dry_run_returns_none(self, config):
        """Dry run skips GCS save and returns None."""
        from alpha_zero.cloud.gcp.tournament_worker import save_tournament_results

        result = save_tournament_results(config, {"test": True}, dry_run=True)
        assert result is None

    def test_saves_to_correct_path(self, config):
        """Results are saved to tournament/post_run_results.json."""
        with patch("alpha_zero.cloud.gcp.storage.GCSBackend") as mock_gcs:
            mock_backend = MagicMock()
            mock_backend.upload_blob.return_value = "gs://test-bucket/tournament/post_run_results.json"
            mock_gcs.return_value = mock_backend

            from alpha_zero.cloud.gcp.tournament_worker import save_tournament_results

            uri = save_tournament_results(config, {"test": True}, dry_run=False)

            mock_gcs.assert_called_once_with(config.project_id)
            call_args = mock_backend.upload_blob.call_args[0]
            assert call_args[0] == config.bucket
            assert call_args[1] == "tournament/post_run_results.json"
            assert isinstance(call_args[2], bytes)
            assert uri is not None


# =============================================================================
# TestLogTournamentToWandb
# =============================================================================


class TestLogTournamentToWandb:
    """Tests for _log_tournament_to_wandb() in vertex_orchestrator."""

    def test_logs_when_enabled(self):
        """Logs ELO ratings and summary metrics when enabled."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import _log_tournament_to_wandb

        mock_logger = MagicMock()
        mock_logger.is_enabled = True

        results = {
            "model_versions": [0, 1, 2],
            "elo_ratings": {"v000": 1516.0, "v001": 1500.0, "v002": 1484.0},
            "matches": [{"a": "b"}],
            "best_model": "v000",
            "best_elo": 1516.0,
        }

        _log_tournament_to_wandb(mock_logger, results)

        assert mock_logger.log.call_count > 0
        all_metrics = {}
        for call in mock_logger.log.call_args_list:
            all_metrics.update(call[0][0])
        assert "tournament/elo_v000" in all_metrics
        assert "tournament/best_elo" in all_metrics

    def test_noop_when_none(self):
        """No error when wandb_logger is None."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import _log_tournament_to_wandb

        _log_tournament_to_wandb(None, {})  # Should not raise

    def test_noop_when_disabled(self):
        """No logging when is_enabled is False."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import _log_tournament_to_wandb

        mock_logger = MagicMock()
        mock_logger.is_enabled = False

        _log_tournament_to_wandb(mock_logger, {"elo_ratings": {"v000": 1500}})

        mock_logger.log.assert_not_called()


# =============================================================================
# TestOrchestratorIntegration
# =============================================================================


class TestOrchestratorIntegration:
    """Tests for refactored run_post_run_evaluation() in vertex_orchestrator."""

    def test_local_mode_delegates_to_tournament_worker(self, config, args):
        """Local mode (default) delegates to tournament_worker.run_tournament()."""
        with patch("alpha_zero.cloud.gcp.tournament_worker.run_tournament") as mock_run:
            mock_run.return_value = {
                "model_versions": [0, 1],
                "elo_ratings": {"v000": 1516, "v001": 1484},
                "matches": [],
                "best_model": "v000",
                "best_elo": 1516,
            }
            with patch("alpha_zero.cloud.gcp.tournament_worker.save_tournament_results"):
                from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

                result = run_post_run_evaluation(
                    config=config,
                    model_versions=[0, 1],
                    args=args,
                    dry_run=True,
                )

                mock_run.assert_called_once()
                assert result["best_model"] == "v000"

    def test_subsampling_applied(self, config, args):
        """Model subsampling is applied before tournament."""
        args.tournament_model_stride = 3

        with patch("alpha_zero.cloud.gcp.tournament_worker.run_tournament") as mock_run:
            mock_run.return_value = {
                "model_versions": [0, 3, 6, 9],
                "elo_ratings": {"v000": 1500, "v003": 1500, "v006": 1500, "v009": 1500},
                "matches": [],
                "best_model": "v000",
                "best_elo": 1500,
            }
            with patch("alpha_zero.cloud.gcp.tournament_worker.save_tournament_results"):
                from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

                run_post_run_evaluation(
                    config=config,
                    model_versions=list(range(10)),
                    args=args,
                    dry_run=True,
                )

                # Check that run_tournament received subsampled versions
                call_kwargs = mock_run.call_args[1]
                versions = call_kwargs["model_versions"]
                assert len(versions) < 10
                assert 0 in versions
                assert 9 in versions

    def test_skips_with_insufficient_models_after_subsampling(self, config, args):
        """Skips tournament if subsampling leaves < 2 models."""
        args.tournament_model_range = "100:200"

        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        result = run_post_run_evaluation(
            config=config,
            model_versions=list(range(10)),
            args=args,
            dry_run=True,
        )
        assert result == {}

    def test_backwards_compatible_with_old_args(self, config):
        """Works with args that don't have new tournament_* fields."""
        old_args = argparse.Namespace(
            tournament_games=4,
            tournament_mcts_sims=8,
            mcts_type="gumbel",
            gumbel_num_actions=16,
            board_size=8,
            max_turns=15,
        )

        with patch("alpha_zero.cloud.gcp.tournament_worker.run_tournament") as mock_run:
            mock_run.return_value = {
                "model_versions": [0, 1],
                "elo_ratings": {"v000": 1516, "v001": 1484},
                "matches": [],
                "best_model": "v000",
                "best_elo": 1516,
            }
            with patch("alpha_zero.cloud.gcp.tournament_worker.save_tournament_results"):
                from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

                result = run_post_run_evaluation(
                    config=config,
                    model_versions=[0, 1],
                    args=old_args,
                    dry_run=True,
                )

                assert result["best_model"] == "v000"

    def test_explicit_model_versions_override_subsampling(self, config, args):
        """--tournament_model_versions overrides stride/range subsampling."""
        args.tournament_model_versions = "5,10,15"
        args.tournament_model_stride = 3  # Should be ignored

        with patch("alpha_zero.cloud.gcp.tournament_worker.run_tournament") as mock_run:
            mock_run.return_value = {
                "model_versions": [5, 10, 15],
                "elo_ratings": {"v005": 1516, "v010": 1500, "v015": 1484},
                "matches": [],
                "best_model": "v005",
                "best_elo": 1516,
            }
            with patch("alpha_zero.cloud.gcp.tournament_worker.save_tournament_results"):
                from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

                run_post_run_evaluation(
                    config=config,
                    model_versions=list(range(20)),
                    args=args,
                    dry_run=True,
                )

                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["model_versions"] == [5, 10, 15]

    def test_cloud_mode_delegates_to_vertex_job(self, config, args):
        """Cloud mode submits a Vertex AI job instead of running locally."""
        args.tournament_cloud = True

        with patch(
            "alpha_zero.cloud.gcp.vertex_orchestrator._run_tournament_cloud"
        ) as mock_cloud:
            mock_cloud.return_value = {
                "model_versions": [0, 1],
                "elo_ratings": {"v000": 1516, "v001": 1484},
                "matches": [],
                "best_model": "v000",
                "best_elo": 1516,
            }

            from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

            result = run_post_run_evaluation(
                config=config,
                model_versions=[0, 1],
                args=args,
                dry_run=False,
            )

            mock_cloud.assert_called_once()
            assert result["best_model"] == "v000"


# =============================================================================
# TestEntrypointTournamentMode
# =============================================================================


class TestEntrypointTournamentMode:
    """Tests for tournament mode in entrypoint.py."""

    def test_tournament_mode_routes_to_worker(self):
        """--mode=tournament routes to tournament_worker.main()."""
        import sys

        with patch.object(sys, "argv", ["entrypoint", "--mode=tournament", "--dry_run"]):
            with patch(
                "alpha_zero.cloud.gcp.tournament_worker.main", return_value=0
            ) as mock_main:
                from alpha_zero.cloud.gcp.entrypoint import main

                result = main()

                mock_main.assert_called_once()
                assert result == 0

    def test_unknown_mode_returns_1(self):
        """Unknown mode prints error and returns 1."""
        import sys

        with patch.object(sys, "argv", ["entrypoint", "--mode=bogus"]):
            from alpha_zero.cloud.gcp.entrypoint import main

            result = main()
            assert result == 1


# =============================================================================
# TestVertexTournamentJobMode
# =============================================================================


class TestVertexTournamentJobMode:
    """Tests for VertexTournamentJob using --mode=tournament."""

    def test_uses_mode_tournament_not_python_m(self, config):
        """Job args start with --mode=tournament, not python -m."""
        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        job.launch(model_versions=[0, 1, 2])
        submitted_spec = next(iter(client._jobs.values()))["spec"]
        assert submitted_spec.args[0] == "--mode=tournament"
        assert "python" not in submitted_spec.args
        assert "-m" not in submitted_spec.args

    def test_no_command_field_set(self, config):
        """Tournament job uses args only (no command), relying on container ENTRYPOINT."""
        client = MockVertexAIClient()
        job = VertexTournamentJob(config, client)
        job.launch(model_versions=[0, 1])
        submitted_spec = next(iter(client._jobs.values()))["spec"]
        assert submitted_spec.command == []


# =============================================================================
# TestRunTournamentCloud
# =============================================================================


class TestRunTournamentCloud:
    """Tests for _run_tournament_cloud() in vertex_orchestrator."""

    def test_returns_results_on_success(self, config, args):
        """Returns parsed results from GCS on success."""
        import json

        args.tournament_cloud = True

        mock_results = {
            "model_versions": [0, 1],
            "elo_ratings": {"v000": 1516, "v001": 1484},
            "matches": [],
            "best_model": "v000",
            "best_elo": 1516,
        }

        with patch(
            "alpha_zero.cloud.gcp.vertex_jobs.VertexAIClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.submit_custom_job.return_value = "mock-job"
            mock_client.wait_for_completion.return_value = JobResult(
                job_id="mock-id",
                job_name="mock-job",
                state=JobState.SUCCEEDED.value,
                success=True,
            )

            with patch(
                "alpha_zero.cloud.gcp.storage.GCSBackend"
            ) as mock_gcs:
                mock_backend = mock_gcs.return_value
                mock_backend.download_blob.return_value = json.dumps(
                    mock_results
                ).encode()

                from alpha_zero.cloud.gcp.vertex_orchestrator import (
                    _run_tournament_cloud,
                )

                result = _run_tournament_cloud(config, [0, 1], args)
                assert result["best_model"] == "v000"

    def test_returns_empty_on_failure(self, config, args):
        """Returns empty dict when job fails."""
        args.tournament_cloud = True

        with patch(
            "alpha_zero.cloud.gcp.vertex_jobs.VertexAIClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.submit_custom_job.return_value = "mock-job"
            mock_client.wait_for_completion.return_value = JobResult(
                job_id="mock-id",
                job_name="mock-job",
                state=JobState.FAILED.value,
                success=False,
                error_message="GPU OOM",
            )

            from alpha_zero.cloud.gcp.vertex_orchestrator import (
                _run_tournament_cloud,
            )

            result = _run_tournament_cloud(config, [0, 1], args)
            assert result == {}
