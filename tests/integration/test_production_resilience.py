"""Integration tests for PBTOrchestrator production resilience features.

Tests checkpoint rotation, quota guarding, evaluation strategy switching,
and baseline benchmarking to ensure production hardening works correctly.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.pbt_orchestrator import PBTOrchestrator
from alpha_zero.pbt.config import AgentState, PBTConfig
from alpha_zero.pbt.evaluator import EvaluationReport, MatchResult, PBTEvaluator
from alpha_zero.utils.wandb_logger import WandbLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pbt_config() -> PBTConfig:
    """Minimal PBT configuration for testing."""
    return PBTConfig(
        population_size=3,
        min_population_size=3,
        evaluation_games_per_match=2,
        evaluation_strategy="round_robin",
        elo_initial_rating=1200.0,
        elo_k_factor=32.0,
        checkpoint_retention_generations=3,
        evaluation_strategy_switch_gen=3,
        baseline_interval=5,
    )


@pytest.fixture
def gcp_config() -> GCPConfig:
    """Mock GCP configuration."""
    config = MagicMock(spec=GCPConfig)
    config.project_id = "test-project"
    config.region = "us-central1"
    config.bucket = "test-bucket"
    config.service_account = "test@test-project.iam.gserviceaccount.com"

    # Mock paths
    paths = MagicMock()
    paths.logs_prefix = "logs"
    paths.checkpoints_prefix = "checkpoints"
    config.paths = paths

    return config


@pytest.fixture
def agents() -> list[AgentState]:
    """Sample agent population."""
    return [
        AgentState(agent_id="agent_0", elo_rating=1200.0),
        AgentState(agent_id="agent_1", elo_rating=1250.0),
        AgentState(agent_id="agent_2", elo_rating=1150.0),
    ]


@pytest.fixture
def baseline_agents() -> list[AgentState]:
    """Sample baseline agents for benchmarking."""
    return [
        AgentState(agent_id="baseline_random", elo_rating=1000.0),
        AgentState(agent_id="baseline_heuristic", elo_rating=1100.0),
    ]


def _mock_arena_runner(agent_new: AgentState, agent_old: AgentState, num_games: int) -> MatchResult:
    """Mock arena runner that returns balanced results."""
    return MatchResult(
        agent_id_new=agent_new.agent_id,
        agent_id_old=agent_old.agent_id,
        win_rate=0.55,
        draw_rate=0.10,
        loss_rate=0.35,
        total_games=num_games,
    )


# ---------------------------------------------------------------------------
# Checkpoint Rotation Tests
# ---------------------------------------------------------------------------


class TestCheckpointRotation:
    """Tests for checkpoint rotation feature."""

    def test_checkpoint_rotation_keeps_recent_gens(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """Checkpoint rotation should keep only the most recent N generations."""
        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend") as mock_backend_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage"),
        ):
            mock_backend = MagicMock()
            mock_backend_cls.return_value = mock_backend

            # Simulate 10 generations of checkpoints
            manifest_data = {
                str(i): [f"checkpoints/gen_{i}/agent_0.pt", f"checkpoints/gen_{i}/agent_1.pt"]
                for i in range(1, 11)
            }
            manifest_json = json.dumps(manifest_data).encode()

            # Setup mock responses
            mock_backend.blob_exists.return_value = True
            mock_backend.download_blob.return_value = manifest_json
            mock_backend.list_blobs.return_value = [
                f"checkpoints/gen_10/agent_0.pt",
                f"checkpoints/gen_10/agent_1.pt",
            ]

            orchestrator = PBTOrchestrator(pbt_config, gcp_config, checkpoint_retention=3)

            # Run rotation for generation 10
            orchestrator._rotate_checkpoints(gen=10)

            # Should delete generations 1-7 (keep 8, 9, 10)
            delete_calls = mock_backend.delete_blob.call_args_list
            deleted_gens = set()
            for call in delete_calls:
                blob_path = call[0][1]  # Second arg is blob path
                # Extract generation from path like "checkpoints/gen_5/..."
                if "/gen_" in blob_path:
                    gen_str = blob_path.split("/gen_")[1].split("/")[0]
                    deleted_gens.add(int(gen_str))

            # Generations 1-7 should be deleted (cutoff = 10 - 3 = 7)
            expected_deleted = set(range(1, 8))
            assert deleted_gens == expected_deleted

    def test_checkpoint_rotation_unlimited(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """When checkpoint_retention=0, no deletions should occur."""
        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend") as mock_backend_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage"),
        ):
            mock_backend = MagicMock()
            mock_backend_cls.return_value = mock_backend

            orchestrator = PBTOrchestrator(pbt_config, gcp_config, checkpoint_retention=0)

            # Run rotation
            orchestrator._rotate_checkpoints(gen=10)

            # Should not attempt any operations (early return)
            mock_backend.blob_exists.assert_not_called()
            mock_backend.delete_blob.assert_not_called()

    def test_checkpoint_rotation_early_gen(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """Early generations should not trigger deletions."""
        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend") as mock_backend_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage"),
        ):
            mock_backend = MagicMock()
            mock_backend_cls.return_value = mock_backend

            # Setup manifest with only 2 generations
            manifest_data = {
                "1": ["checkpoints/gen_1/agent_0.pt"],
                "2": ["checkpoints/gen_2/agent_0.pt"],
            }
            manifest_json = json.dumps(manifest_data).encode()

            mock_backend.blob_exists.return_value = True
            mock_backend.download_blob.return_value = manifest_json
            mock_backend.list_blobs.return_value = ["checkpoints/gen_2/agent_0.pt"]

            orchestrator = PBTOrchestrator(pbt_config, gcp_config, checkpoint_retention=5)

            # Run rotation for generation 2 (cutoff = 2 - 5 = -3, no deletions)
            orchestrator._rotate_checkpoints(gen=2)

            # No deletions should occur (cutoff is negative)
            mock_backend.delete_blob.assert_not_called()

    def test_checkpoint_rotation_backend_failure(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """Backend failures should be caught and logged, not crash the loop."""
        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend") as mock_backend_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage"),
        ):
            mock_backend = MagicMock()
            mock_backend_cls.return_value = mock_backend

            # Simulate backend failure
            mock_backend.blob_exists.side_effect = Exception("GCS connection error")

            orchestrator = PBTOrchestrator(pbt_config, gcp_config, checkpoint_retention=3)

            # Should not raise, just log warning
            orchestrator._rotate_checkpoints(gen=10)

            # Verify it attempted the operation
            mock_backend.blob_exists.assert_called_once()


# ---------------------------------------------------------------------------
# Quota Guard Tests
# ---------------------------------------------------------------------------


class TestQuotaGuard:
    """Tests for GPU quota checking feature."""

    def test_quota_guard_passes_available(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """When no quota limit is set, should return True."""
        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage"),
        ):
            orchestrator = PBTOrchestrator(pbt_config, gcp_config)

            # No environment variable set
            result = orchestrator._check_quota(gen=1)
            assert result is True

    def test_quota_guard_passes_under_limit(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """When GPU usage is under limit, should return True."""
        # Mock aiplatform module (imported inside _check_quota)
        mock_aiplatform = MagicMock()
        mock_job1 = MagicMock()
        mock_job1.gpu_count = 2
        mock_job2 = MagicMock()
        mock_job2.gpu_count = 3

        mock_aiplatform.init.return_value = None
        mock_aiplatform.CustomJob.list.return_value = [mock_job1, mock_job2]

        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage"),
            patch.dict(os.environ, {"CIV6AZ_GPU_QUOTA_LIMIT": "10"}),
            patch.dict(sys.modules, {"google.cloud.aiplatform": mock_aiplatform}),
        ):
            orchestrator = PBTOrchestrator(pbt_config, gcp_config)

            # 2 + 3 = 5 GPUs active < 10 limit
            result = orchestrator._check_quota(gen=1)
            assert result is True

    def test_quota_guard_fails_over_limit(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """When GPU usage exceeds limit, should return False.

        Note: This test mocks the google.cloud.aiplatform module that is
        imported dynamically inside the _check_quota method. The key is to
        create a mock module object and insert it into sys.modules BEFORE
        the import statement executes.
        """
        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage"),
            patch.dict(os.environ, {"CIV6AZ_GPU_QUOTA_LIMIT": "5"}),
        ):
            orchestrator = PBTOrchestrator(pbt_config, gcp_config)

            # Create mocks for the aiplatform module structure
            # Mock 6 GPUs active (4 + 2 = 6 >= 5 limit)
            mock_job1 = MagicMock()
            mock_job1.gpu_count = 4
            mock_job2 = MagicMock()
            mock_job2.gpu_count = 2

            # Create a mock module that mimics google.cloud.aiplatform
            mock_aiplatform = MagicMock()
            mock_aiplatform.init = MagicMock(return_value=None)
            mock_aiplatform.CustomJob = MagicMock()
            mock_aiplatform.CustomJob.list = MagicMock(return_value=[mock_job1, mock_job2])

            # Create parent modules for proper import chain
            mock_google = MagicMock()
            mock_google_cloud = MagicMock()
            mock_google.cloud = mock_google_cloud
            mock_google_cloud.aiplatform = mock_aiplatform

            # Patch sys.modules with the complete chain
            with patch.dict(
                sys.modules,
                {
                    "google": mock_google,
                    "google.cloud": mock_google_cloud,
                    "google.cloud.aiplatform": mock_aiplatform,
                },
            ):
                # Now when _check_quota does "from google.cloud import aiplatform",
                # it will get our mock
                result = orchestrator._check_quota(gen=1)
                assert result is False

    def test_quota_guard_optimistic_on_error(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """Exception during quota check should return True (optimistic)."""
        # Simulate API failure
        mock_aiplatform = MagicMock()
        mock_aiplatform.init.side_effect = Exception("API connection error")

        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage"),
            patch.dict(os.environ, {"CIV6AZ_GPU_QUOTA_LIMIT": "10"}),
            patch.dict(sys.modules, {"google.cloud.aiplatform": mock_aiplatform}),
        ):
            orchestrator = PBTOrchestrator(pbt_config, gcp_config)

            # Should return True on error (optimistic fallback)
            result = orchestrator._check_quota(gen=1)
            assert result is True


# ---------------------------------------------------------------------------
# Strategy Switch Tests
# ---------------------------------------------------------------------------


class TestStrategySwitch:
    """Tests for evaluation strategy switching feature."""

    def test_strategy_switch_at_threshold(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """Strategy should switch from round_robin to top_k_vs_best at threshold."""
        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage"),
        ):
            orchestrator = PBTOrchestrator(pbt_config, gcp_config, evaluation_strategy_switch_gen=3)

            # Before threshold (gen=2)
            assert orchestrator.pbt_config.evaluation_strategy == "round_robin"
            orchestrator._maybe_switch_strategy(gen=2)
            assert orchestrator.pbt_config.evaluation_strategy == "round_robin"

            # At threshold (gen=3)
            orchestrator._maybe_switch_strategy(gen=3)
            assert orchestrator.pbt_config.evaluation_strategy == "top_k_vs_best"

            # After threshold (gen=4) - should remain switched
            orchestrator._maybe_switch_strategy(gen=4)
            assert orchestrator.pbt_config.evaluation_strategy == "top_k_vs_best"

    def test_strategy_switch_zero_disabled(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """When switch_gen=0, strategy should never switch."""
        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage"),
        ):
            orchestrator = PBTOrchestrator(pbt_config, gcp_config, evaluation_strategy_switch_gen=0)

            # Test across multiple generations
            for gen in range(1, 10):
                orchestrator._maybe_switch_strategy(gen=gen)
                assert orchestrator.pbt_config.evaluation_strategy == "round_robin"

    def test_strategy_switch_already_switched(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """If already on top_k_vs_best, should remain unchanged."""
        pbt_config.evaluation_strategy = "top_k_vs_best"

        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend"),
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage"),
        ):
            orchestrator = PBTOrchestrator(pbt_config, gcp_config, evaluation_strategy_switch_gen=3)

            # Already switched, should do nothing
            orchestrator._maybe_switch_strategy(gen=3)
            assert orchestrator.pbt_config.evaluation_strategy == "top_k_vs_best"


# ---------------------------------------------------------------------------
# Baseline Benchmark Tests
# ---------------------------------------------------------------------------


class TestBaselineBenchmark:
    """Tests for baseline benchmarking feature."""

    def test_baseline_benchmark_interval(
        self, pbt_config: PBTConfig, agents: list[AgentState], baseline_agents: list[AgentState]
    ) -> None:
        """Baseline benchmarks should run only at specified intervals."""
        pbt_config.baseline_interval = 5

        evaluator = PBTEvaluator(
            pbt_config,
            arena_runner=_mock_arena_runner,
            baseline_agents=baseline_agents,
            baseline_interval=5,
        )

        # Gen 1: not divisible by 5, no baselines
        results_g1 = evaluator.run_baseline_benchmarks(
            agents, generation=1, arena_runner=_mock_arena_runner
        )
        assert len(results_g1) == 0

        # Gen 5: divisible by 5, should run baselines
        results_g5 = evaluator.run_baseline_benchmarks(
            agents, generation=5, arena_runner=_mock_arena_runner
        )
        # 3 agents × 2 baselines = 6 matches
        assert len(results_g5) == 6

        # Gen 10: divisible by 5, should run baselines
        results_g10 = evaluator.run_baseline_benchmarks(
            agents, generation=10, arena_runner=_mock_arena_runner
        )
        assert len(results_g10) == 6

    def test_baseline_benchmark_no_baselines(
        self, pbt_config: PBTConfig, agents: list[AgentState]
    ) -> None:
        """When baseline list is empty, should return empty results."""
        pbt_config.baseline_interval = 5

        evaluator = PBTEvaluator(
            pbt_config,
            arena_runner=_mock_arena_runner,
            baseline_agents=[],  # No baselines
            baseline_interval=5,
        )

        # Even at interval generation, no baselines configured
        results = evaluator.run_baseline_benchmarks(
            agents, generation=5, arena_runner=_mock_arena_runner
        )
        assert len(results) == 0

    def test_baseline_benchmark_interval_zero(
        self, pbt_config: PBTConfig, agents: list[AgentState], baseline_agents: list[AgentState]
    ) -> None:
        """When baseline_interval=0, baselines should never run."""
        pbt_config.baseline_interval = 0

        evaluator = PBTEvaluator(
            pbt_config,
            arena_runner=_mock_arena_runner,
            baseline_agents=baseline_agents,
            baseline_interval=0,
        )

        # Test across multiple generations
        for gen in [1, 5, 10, 15]:
            results = evaluator.run_baseline_benchmarks(
                agents, generation=gen, arena_runner=_mock_arena_runner
            )
            assert len(results) == 0

    def test_baseline_benchmark_no_elo_update(
        self, pbt_config: PBTConfig, agents: list[AgentState], baseline_agents: list[AgentState]
    ) -> None:
        """Baseline results should not affect agent ELO ratings."""
        pbt_config.baseline_interval = 5

        evaluator = PBTEvaluator(
            pbt_config,
            arena_runner=_mock_arena_runner,
            baseline_agents=baseline_agents,
            baseline_interval=5,
        )

        # Record initial ELO ratings
        initial_elos = {a.agent_id: a.elo_rating for a in agents}

        # Run baseline benchmarks (not full evaluation)
        baseline_results = evaluator.run_baseline_benchmarks(
            agents, generation=5, arena_runner=_mock_arena_runner
        )

        # ELO ratings should remain unchanged (baselines don't update ELO)
        for agent in agents:
            assert agent.elo_rating == initial_elos[agent.agent_id]

        # Results should be returned for reporting
        assert len(baseline_results) == 6


# ---------------------------------------------------------------------------
# Full Generation Loop Tests
# ---------------------------------------------------------------------------


class TestFullGenerationLoop:
    """Tests for complete generation loop with all features enabled."""

    def test_full_generation_loop_all_features(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig, baseline_agents: list[AgentState]
    ) -> None:
        """Full generation loop with W&B, rotation, quota, strategy switch."""
        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient") as mock_client_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob") as mock_training_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob") as mock_selfplay_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend") as mock_backend_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage") as mock_storage_cls,
        ):
            # Setup mocks
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_training = MagicMock()
            mock_training_cls.return_value = mock_training
            mock_training.launch.return_value = "training-job-1"

            mock_selfplay = MagicMock()
            mock_selfplay_cls.return_value = mock_selfplay
            mock_selfplay.launch.return_value = ["selfplay-job-1"]
            mock_selfplay.wait_for_completion.return_value = None

            mock_backend = MagicMock()
            mock_backend_cls.return_value = mock_backend
            mock_backend.blob_exists.return_value = False  # No prior state
            mock_backend.list_blobs.return_value = []

            mock_storage = MagicMock()
            mock_storage_cls.return_value = mock_storage
            mock_storage.get_latest_version.return_value = 1

            # Mock job polling
            mock_client.get_job_state.return_value = ("JOB_STATE_SUCCEEDED", None)

            # Mock W&B logger
            mock_wandb = MagicMock(spec=WandbLogger)

            # Create orchestrator with all features
            orchestrator = PBTOrchestrator(
                pbt_config,
                gcp_config,
                wandb_logger=mock_wandb,
                checkpoint_retention=3,
                evaluation_strategy_switch_gen=2,
                baseline_agents=baseline_agents,
            )

            # Mock evaluator to avoid complex setup
            mock_evaluator = MagicMock(spec=PBTEvaluator)
            mock_report = EvaluationReport(
                generation=1,
                elo_ratings={"agent_0": 1220.0, "agent_1": 1200.0, "agent_2": 1180.0},
                fitness_scores={"agent_0": 0.8, "agent_1": 0.7, "agent_2": 0.6},
            )
            mock_evaluator.evaluate_population.return_value = mock_report
            orchestrator.evaluator = mock_evaluator

            # Run 2 generations
            orchestrator.run(num_generations=2)

            # Verify checkpoint rotation was called
            assert mock_backend.upload_blob.call_count >= 2  # State saves

            # Verify W&B logging (if enabled)
            if mock_wandb:
                assert mock_wandb.log.call_count >= 2

            # Verify strategy switch occurred at gen 2
            assert orchestrator.pbt_config.evaluation_strategy == "top_k_vs_best"

    def test_orchestrator_handles_phase_failures_gracefully(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """Orchestrator should handle phase failures without crashing."""
        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient") as mock_client_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob") as mock_training_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob") as mock_selfplay_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend") as mock_backend_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage") as mock_storage_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_job_state.return_value = ("JOB_STATE_SUCCEEDED", None)

            mock_training = MagicMock()
            mock_training_cls.return_value = mock_training
            mock_training.launch.return_value = "training-job-1"

            # Selfplay fails
            mock_selfplay = MagicMock()
            mock_selfplay_cls.return_value = mock_selfplay
            mock_selfplay.launch.side_effect = Exception("Selfplay launch failed")

            mock_backend = MagicMock()
            mock_backend_cls.return_value = mock_backend
            mock_backend.blob_exists.return_value = False
            mock_backend.list_blobs.return_value = []

            mock_storage = MagicMock()
            mock_storage_cls.return_value = mock_storage
            mock_storage.get_latest_version.return_value = 1

            orchestrator = PBTOrchestrator(pbt_config, gcp_config)

            # Mock evaluator
            mock_evaluator = MagicMock(spec=PBTEvaluator)
            mock_report = EvaluationReport(
                generation=1,
                elo_ratings={"agent_0": 1220.0, "agent_1": 1200.0, "agent_2": 1180.0},
                fitness_scores={"agent_0": 0.8, "agent_1": 0.7, "agent_2": 0.6},
            )
            mock_evaluator.evaluate_population.return_value = mock_report
            orchestrator.evaluator = mock_evaluator

            # Mock PopulationManager's exploit_and_explore
            orchestrator.manager.exploit_and_explore = MagicMock(return_value=[])

            # Should not crash, just log errors
            try:
                orchestrator.run(num_generations=1)
            except Exception as e:
                pytest.fail(f"Orchestrator crashed on phase failure: {e}")

    def test_orchestrator_saves_state_after_each_generation(
        self, pbt_config: PBTConfig, gcp_config: GCPConfig
    ) -> None:
        """Orchestrator should save state after each generation completes."""
        with (
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient") as mock_client_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob") as mock_training_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob") as mock_selfplay_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend") as mock_backend_cls,
            patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage") as mock_storage_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_job_state.return_value = ("JOB_STATE_SUCCEEDED", None)

            mock_training = MagicMock()
            mock_training_cls.return_value = mock_training
            mock_training.launch.return_value = "training-job-1"

            mock_selfplay = MagicMock()
            mock_selfplay_cls.return_value = mock_selfplay
            mock_selfplay.launch.return_value = ["selfplay-job-1"]
            mock_selfplay.wait_for_completion.return_value = None

            mock_backend = MagicMock()
            mock_backend_cls.return_value = mock_backend
            mock_backend.blob_exists.return_value = False
            mock_backend.list_blobs.return_value = []

            mock_storage = MagicMock()
            mock_storage_cls.return_value = mock_storage
            mock_storage.get_latest_version.return_value = 1

            orchestrator = PBTOrchestrator(pbt_config, gcp_config)

            # Mock evaluator
            mock_evaluator = MagicMock(spec=PBTEvaluator)
            mock_report = EvaluationReport(
                generation=1,
                elo_ratings={"agent_0": 1220.0, "agent_1": 1200.0, "agent_2": 1180.0},
                fitness_scores={"agent_0": 0.8, "agent_1": 0.7, "agent_2": 0.6},
            )
            mock_evaluator.evaluate_population.return_value = mock_report
            orchestrator.evaluator = mock_evaluator

            # Run 3 generations
            orchestrator.run(num_generations=3)

            # Should have saved state 3 times (once per generation)
            # Plus initial save when initializing population
            upload_calls = [
                call
                for call in mock_backend.upload_blob.call_args_list
                if "pbt_state.json" in str(call)
            ]
            assert len(upload_calls) >= 3
