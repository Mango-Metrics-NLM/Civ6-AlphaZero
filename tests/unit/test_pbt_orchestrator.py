import unittest
from unittest.mock import MagicMock, patch
from alpha_zero.cloud.gcp.pbt_orchestrator import PBTOrchestrator
from alpha_zero.pbt.config import PBTConfig
from alpha_zero.pbt.evaluator import MatchResult
from alpha_zero.cloud.gcp.config import GCPConfig


def _mock_arena_runner(agent_new, agent_old, num_games):
    """Return a fixed MatchResult for testing."""
    return MatchResult(
        agent_id_new=agent_new.agent_id,
        agent_id_old=agent_old.agent_id,
        win_rate=0.5,
        draw_rate=0.0,
        loss_rate=0.5,
        total_games=num_games,
    )


class TestPBTOrchestrator(unittest.TestCase):
    def setUp(self):
        self.pbt_config = PBTConfig(population_size=4, num_generations=2, evaluation_games=10)
        self.gcp_config = MagicMock(spec=GCPConfig)
        self.gcp_config.project_id = "test-project"
        self.gcp_config.region = "us-central1"
        self.gcp_config.bucket = "test-bucket"
        self.gcp_config.paths = MagicMock()
        self.gcp_config.paths.logs_prefix = "logs"
        self.gcp_config.paths.checkpoints_prefix = "checkpoints"
        self.gcp_config.service_account = "sa@example.com"

    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage")
    def test_run_orchestrator(
        self, MockStorage, MockBackend, MockSelfPlay, MockTraining, MockClient
    ):
        # Setup Mocks
        orchestrator = PBTOrchestrator(self.pbt_config, self.gcp_config)

        # Provide a mock arena runner so evaluation doesn't hit real code
        orchestrator.evaluator._arena_runner = _mock_arena_runner

        # Mock Job Launches
        orchestrator.selfplay_job_manager.launch.return_value = ["job_sp_1"]
        orchestrator.training_job_manager.launch.side_effect = [f"job_train_{i}" for i in range(8)]

        # Mock Job Completion
        orchestrator.client.get_job_state.return_value = ("JOB_STATE_SUCCEEDED", {})

        # Mock Storage
        mock_backend_instance = MockBackend.return_value
        mock_backend_instance.blob_exists.return_value = False  # First run
        mock_backend_instance.list_blobs.return_value = []

        mock_storage_instance = MockStorage.return_value
        mock_storage_instance.get_latest_version.return_value = 2  # Incremented version

        # Run
        orchestrator.run()

        # Verify
        # Initialization
        self.assertEqual(len(orchestrator.manager.agents), 4)

        # Generation 1 & 2
        # Self Play
        self.assertEqual(
            orchestrator.selfplay_job_manager.launch.call_count, 8
        )  # 4 agents * 2 gens
        # Training
        self.assertEqual(
            orchestrator.training_job_manager.launch.call_count, 8
        )  # 4 agents * 2 gens

        # State Saved
        self.assertTrue(mock_backend_instance.upload_blob.called)
