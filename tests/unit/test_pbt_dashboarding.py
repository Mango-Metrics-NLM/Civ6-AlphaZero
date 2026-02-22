"""Tests for PBT population dashboard features in WandbLogger and PBTOrchestrator.

This module tests the W&B integration for Population-Based Training (PBT),
including population-level statistics, per-agent ELO tracking, and mutation
event logging.

Test Coverage:
    - WandbLogger.log_population_stats()
    - WandbLogger.define_pbt_charts()
    - PBTOrchestrator._log_population_dashboard()
    - PBTOrchestrator._log_mutation_events()
"""

import unittest
from unittest.mock import MagicMock, call, patch

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.pbt_orchestrator import PBTOrchestrator
from alpha_zero.pbt.config import AgentState, PBTConfig
from alpha_zero.pbt.evaluator import EvaluationReport
from alpha_zero.utils.wandb_logger import WandbConfig, WandbLogger


class TestWandbLoggerPopulationStats(unittest.TestCase):
    """Tests for WandbLogger.log_population_stats()."""

    def setUp(self):
        """Create a mock-initialized WandbLogger for testing."""
        self.wandb_config = WandbConfig(enabled=True)
        self.logger = WandbLogger(self.wandb_config)
        # Simulate initialized state
        self.logger._initialized = True
        self.logger._run = MagicMock()

    @patch("alpha_zero.utils.wandb_logger.wandb")
    def test_log_population_stats_payload_keys(self, mock_wandb):
        """Verify all expected pbt/* keys are present in wandb.log() call."""
        self.logger.log_population_stats(
            generation=5,
            mean_fitness=0.65,
            max_fitness=0.85,
            min_fitness=0.45,
            mean_elo=1500.0,
            max_elo=1650.0,
            min_elo=1350.0,
            population_size=8,
            mutations_count=3,
        )

        # Extract the data dict from wandb.log call
        mock_wandb.log.assert_called_once()
        call_args = mock_wandb.log.call_args
        data = call_args[0][0]

        # Verify all required keys
        expected_keys = {
            "pbt/generation",
            "pbt/fitness_mean",
            "pbt/fitness_max",
            "pbt/fitness_min",
            "pbt/elo_mean",
            "pbt/elo_max",
            "pbt/elo_min",
            "pbt/population_size",
            "pbt/mutations_count",
        }
        self.assertEqual(set(data.keys()), expected_keys)

        # Verify values
        self.assertEqual(data["pbt/generation"], 5)
        self.assertEqual(data["pbt/fitness_mean"], 0.65)
        self.assertEqual(data["pbt/fitness_max"], 0.85)
        self.assertEqual(data["pbt/fitness_min"], 0.45)
        self.assertEqual(data["pbt/elo_mean"], 1500.0)
        self.assertEqual(data["pbt/elo_max"], 1650.0)
        self.assertEqual(data["pbt/elo_min"], 1350.0)
        self.assertEqual(data["pbt/population_size"], 8)
        self.assertEqual(data["pbt/mutations_count"], 3)

    @patch("alpha_zero.utils.wandb_logger.wandb")
    def test_log_population_stats_extra_kwargs_prefixed(self, mock_wandb):
        """Verify custom kwargs are auto-prefixed with pbt/."""
        self.logger.log_population_stats(
            generation=3,
            mean_fitness=0.5,
            max_fitness=0.7,
            min_fitness=0.3,
            mean_elo=1400.0,
            max_elo=1500.0,
            min_elo=1300.0,
            population_size=4,
            custom_metric=42.0,
            another_metric="test",
        )

        mock_wandb.log.assert_called_once()
        data = mock_wandb.log.call_args[0][0]

        # Check custom metrics got prefixed
        self.assertIn("pbt/custom_metric", data)
        self.assertIn("pbt/another_metric", data)
        self.assertEqual(data["pbt/custom_metric"], 42.0)
        self.assertEqual(data["pbt/another_metric"], "test")

    @patch("alpha_zero.utils.wandb_logger.wandb")
    def test_log_population_stats_kwargs_already_prefixed(self, mock_wandb):
        """Verify kwargs already prefixed with pbt/ are not double-prefixed."""
        self.logger.log_population_stats(
            generation=2,
            mean_fitness=0.6,
            max_fitness=0.8,
            min_fitness=0.4,
            mean_elo=1450.0,
            max_elo=1550.0,
            min_elo=1350.0,
            population_size=6,
            **{"pbt/already_prefixed": 99},
        )

        mock_wandb.log.assert_called_once()
        data = mock_wandb.log.call_args[0][0]

        # Should not be "pbt/pbt/already_prefixed"
        self.assertIn("pbt/already_prefixed", data)
        self.assertEqual(data["pbt/already_prefixed"], 99)
        self.assertNotIn("pbt/pbt/already_prefixed", data)

    @patch("alpha_zero.utils.wandb_logger.wandb")
    def test_log_population_stats_noop_when_uninitialized(self, mock_wandb):
        """Verify method returns without error when logger not initialized."""
        uninitialized_logger = WandbLogger(WandbConfig(enabled=True))
        # Do NOT set _initialized or _run

        # Should not raise
        uninitialized_logger.log_population_stats(
            generation=1,
            mean_fitness=0.5,
            max_fitness=0.7,
            min_fitness=0.3,
            mean_elo=1400.0,
            max_elo=1500.0,
            min_elo=1300.0,
            population_size=4,
        )

        # wandb.log should not have been called
        mock_wandb.log.assert_not_called()


class TestWandbLoggerDefinePBTCharts(unittest.TestCase):
    """Tests for WandbLogger.define_pbt_charts()."""

    def setUp(self):
        """Create a mock-initialized WandbLogger for testing."""
        self.wandb_config = WandbConfig(enabled=True)
        self.logger = WandbLogger(self.wandb_config)
        self.logger._initialized = True
        self.logger._run = MagicMock()

    @patch("alpha_zero.utils.wandb_logger.wandb")
    def test_define_pbt_charts_registers_metrics(self, mock_wandb):
        """Verify wandb.define_metric() is called for each PBT metric."""
        self.logger.define_pbt_charts()

        # Should call define_metric multiple times
        self.assertGreater(mock_wandb.define_metric.call_count, 0)

        # Verify key metrics are registered
        expected_calls = [
            call("pbt/generation"),
            call("pbt/elo_max", step_metric="pbt/generation", summary="max"),
            call("pbt/elo_mean", step_metric="pbt/generation", summary="last"),
            call("pbt/elo_min", step_metric="pbt/generation", summary="min"),
            call("pbt/fitness_max", step_metric="pbt/generation", summary="max"),
            call("pbt/fitness_mean", step_metric="pbt/generation", summary="last"),
            call("pbt/fitness_min", step_metric="pbt/generation", summary="min"),
            call("pbt/population_size", step_metric="pbt/generation", summary="last"),
            call("pbt/mutations_count", step_metric="pbt/generation", summary="sum"),
        ]

        for expected_call in expected_calls:
            self.assertIn(expected_call, mock_wandb.define_metric.call_args_list)

    @patch("alpha_zero.utils.wandb_logger.wandb")
    def test_define_pbt_charts_noop_when_uninitialized(self, mock_wandb):
        """Verify method returns without error when logger not initialized."""
        uninitialized_logger = WandbLogger(WandbConfig(enabled=True))
        # Do NOT set _initialized or _run

        # Should not raise
        uninitialized_logger.define_pbt_charts()

        # define_metric should not have been called
        mock_wandb.define_metric.assert_not_called()

    @patch("alpha_zero.utils.wandb_logger.wandb")
    def test_define_pbt_charts_handles_exception(self, mock_wandb):
        """Verify exceptions in define_metric are caught and logged."""
        mock_wandb.define_metric.side_effect = Exception("Test exception")

        # Should not raise
        self.logger.define_pbt_charts()

        # Should have attempted to call define_metric at least once
        self.assertGreater(mock_wandb.define_metric.call_count, 0)


class TestPBTOrchestratorLogPopulationDashboard(unittest.TestCase):
    """Tests for PBTOrchestrator._log_population_dashboard()."""

    def setUp(self):
        """Set up PBTOrchestrator with mocked dependencies."""
        self.pbt_config = PBTConfig(
            population_size=4,
            num_generations=2,
            evaluation_games=10,
        )
        self.gcp_config = MagicMock(spec=GCPConfig)
        self.gcp_config.project_id = "test-project"
        self.gcp_config.region = "us-central1"
        self.gcp_config.bucket = "test-bucket"
        self.gcp_config.paths = MagicMock()
        self.gcp_config.paths.logs_prefix = "logs"
        self.gcp_config.service_account = "sa@example.com"

        # Create mock agents with ELO and fitness
        self.mock_agents = [
            AgentState(
                agent_id="agent_0",
                hyperparameters={"learning_rate": 0.001},
                elo_rating=1500.0,
                fitness_score=0.6,
            ),
            AgentState(
                agent_id="agent_1",
                hyperparameters={"learning_rate": 0.002},
                elo_rating=1550.0,
                fitness_score=0.7,
            ),
            AgentState(
                agent_id="agent_2",
                hyperparameters={"learning_rate": 0.0015},
                elo_rating=1450.0,
                fitness_score=0.5,
            ),
            AgentState(
                agent_id="agent_3",
                hyperparameters={"learning_rate": 0.0018},
                elo_rating=1600.0,
                fitness_score=0.8,
            ),
        ]

        # Create mock evaluation report
        self.mock_report = EvaluationReport(
            generation=5,
            matchups=[],
            elo_ratings={
                "agent_0": 1500.0,
                "agent_1": 1550.0,
                "agent_2": 1450.0,
                "agent_3": 1600.0,
            },
        )

    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage")
    def test_orchestrator_log_population_dashboard(
        self, MockStorage, MockBackend, MockSelfPlay, MockTraining, MockClient
    ):
        """Verify _log_population_dashboard calls wandb_logger with correct data."""
        # Create mock WandbLogger
        mock_wandb_logger = MagicMock(spec=WandbLogger)

        # Create orchestrator with mock logger
        orchestrator = PBTOrchestrator(
            self.pbt_config,
            self.gcp_config,
            wandb_logger=mock_wandb_logger,
        )

        # Populate manager with mock agents (agents is a dict)
        orchestrator.manager.agents = {a.agent_id: a for a in self.mock_agents}

        # Call _log_population_dashboard
        orchestrator._log_population_dashboard(
            gen=5,
            report=self.mock_report,
            elapsed=123.45,
        )

        # Verify log_population_stats was called with correct values
        mock_wandb_logger.log_population_stats.assert_called_once()
        call_kwargs = mock_wandb_logger.log_population_stats.call_args[1]

        self.assertEqual(call_kwargs["generation"], 5)
        self.assertAlmostEqual(call_kwargs["mean_fitness"], 0.65, places=2)
        self.assertEqual(call_kwargs["max_fitness"], 0.8)
        self.assertEqual(call_kwargs["min_fitness"], 0.5)
        self.assertAlmostEqual(call_kwargs["mean_elo"], 1525.0, places=1)
        self.assertEqual(call_kwargs["max_elo"], 1600.0)
        self.assertEqual(call_kwargs["min_elo"], 1450.0)
        self.assertEqual(call_kwargs["population_size"], 4)
        self.assertAlmostEqual(call_kwargs["elapsed_seconds"], 123.45, places=2)

        # Verify per-agent ELO logging
        mock_wandb_logger.log.assert_called()
        per_agent_call = mock_wandb_logger.log.call_args
        per_agent_data = per_agent_call[0][0]

        self.assertIn("pbt/agent/agent_0/elo", per_agent_data)
        self.assertIn("pbt/agent/agent_1/elo", per_agent_data)
        self.assertIn("pbt/agent/agent_2/elo", per_agent_data)
        self.assertIn("pbt/agent/agent_3/elo", per_agent_data)
        self.assertEqual(per_agent_data["pbt/agent/agent_0/elo"], 1500.0)
        self.assertEqual(per_agent_data["pbt/agent/agent_3/elo"], 1600.0)

    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage")
    def test_dashboard_disabled_when_no_logger(
        self, MockStorage, MockBackend, MockSelfPlay, MockTraining, MockClient
    ):
        """Verify _log_population_dashboard is no-op when wandb_logger is None."""
        # Create orchestrator WITHOUT wandb_logger
        orchestrator = PBTOrchestrator(
            self.pbt_config,
            self.gcp_config,
            wandb_logger=None,
        )

        orchestrator.manager.agents = self.mock_agents

        # Should not raise
        orchestrator._log_population_dashboard(
            gen=5,
            report=self.mock_report,
            elapsed=123.45,
        )

    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage")
    def test_dashboard_with_empty_agents(
        self, MockStorage, MockBackend, MockSelfPlay, MockTraining, MockClient
    ):
        """Verify _log_population_dashboard handles empty agent list."""
        mock_wandb_logger = MagicMock(spec=WandbLogger)

        orchestrator = PBTOrchestrator(
            self.pbt_config,
            self.gcp_config,
            wandb_logger=mock_wandb_logger,
        )

        # Empty agent dict
        orchestrator.manager.agents = {}

        # Should not raise or call logger
        orchestrator._log_population_dashboard(
            gen=5,
            report=self.mock_report,
            elapsed=123.45,
        )

        mock_wandb_logger.log_population_stats.assert_not_called()
        mock_wandb_logger.log.assert_not_called()


class TestPBTOrchestratorLogMutationEvents(unittest.TestCase):
    """Tests for PBTOrchestrator._log_mutation_events()."""

    def setUp(self):
        """Set up PBTOrchestrator with mocked dependencies."""
        self.pbt_config = PBTConfig(
            population_size=4,
            num_generations=2,
            evaluation_games=10,
        )
        self.gcp_config = MagicMock(spec=GCPConfig)
        self.gcp_config.project_id = "test-project"
        self.gcp_config.region = "us-central1"
        self.gcp_config.bucket = "test-bucket"
        self.gcp_config.paths = MagicMock()
        self.gcp_config.paths.logs_prefix = "logs"
        self.gcp_config.service_account = "sa@example.com"

    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage")
    def test_log_mutation_events_with_changes(
        self, MockStorage, MockBackend, MockSelfPlay, MockTraining, MockClient
    ):
        """Verify mutation events are logged as a W&B table when changes exist."""
        mock_wandb_logger = MagicMock(spec=WandbLogger)

        orchestrator = PBTOrchestrator(
            self.pbt_config,
            self.gcp_config,
            wandb_logger=mock_wandb_logger,
        )

        # Create mutation changes
        changes = [
            ("agent_0", "agent_3", {"learning_rate": 0.0018, "c_puct": 1.5}),
            ("agent_2", "agent_1", {"learning_rate": 0.002, "c_puct": 1.3}),
        ]

        orchestrator._log_mutation_events(gen=5, changes=changes)

        # Verify mutations_count logged
        mock_wandb_logger.log.assert_called()
        count_call = mock_wandb_logger.log.call_args_list[0]
        count_data = count_call[0][0]
        self.assertEqual(count_data["pbt/mutations_count"], 2)
        self.assertEqual(count_call[1]["step"], 5)

        # Verify table logged
        mock_wandb_logger.log_table.assert_called_once()
        table_call = mock_wandb_logger.log_table.call_args
        table_name = table_call[0][0]
        columns = table_call[0][1]
        rows = table_call[0][2]

        self.assertEqual(table_name, "pbt/mutations_gen_5")
        self.assertEqual(columns, ["replaced_id", "source_id", "learning_rate", "c_puct"])
        self.assertEqual(len(rows), 2)

        # Check first row
        self.assertEqual(rows[0][0], "agent_0")
        self.assertEqual(rows[0][1], "agent_3")
        self.assertEqual(rows[0][2], 0.0018)
        self.assertEqual(rows[0][3], 1.5)

        # Check second row
        self.assertEqual(rows[1][0], "agent_2")
        self.assertEqual(rows[1][1], "agent_1")
        self.assertEqual(rows[1][2], 0.002)
        self.assertEqual(rows[1][3], 1.3)

    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage")
    def test_log_mutation_events_empty_list(
        self, MockStorage, MockBackend, MockSelfPlay, MockTraining, MockClient
    ):
        """Verify no table is logged for empty mutation changes."""
        mock_wandb_logger = MagicMock(spec=WandbLogger)

        orchestrator = PBTOrchestrator(
            self.pbt_config,
            self.gcp_config,
            wandb_logger=mock_wandb_logger,
        )

        # Empty changes
        orchestrator._log_mutation_events(gen=5, changes=[])

        # Verify mutations_count logged as 0
        mock_wandb_logger.log.assert_called_once()
        count_call = mock_wandb_logger.log.call_args
        count_data = count_call[0][0]
        self.assertEqual(count_data["pbt/mutations_count"], 0)

        # Table should NOT be logged
        mock_wandb_logger.log_table.assert_not_called()

    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage")
    def test_log_mutation_events_disabled_when_no_logger(
        self, MockStorage, MockBackend, MockSelfPlay, MockTraining, MockClient
    ):
        """Verify _log_mutation_events is no-op when wandb_logger is None."""
        orchestrator = PBTOrchestrator(
            self.pbt_config,
            self.gcp_config,
            wandb_logger=None,
        )

        changes = [
            ("agent_0", "agent_3", {"learning_rate": 0.0018}),
        ]

        # Should not raise
        orchestrator._log_mutation_events(gen=5, changes=changes)

    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage")
    def test_log_mutation_events_with_missing_hyperparameters(
        self, MockStorage, MockBackend, MockSelfPlay, MockTraining, MockClient
    ):
        """Verify table rows handle missing hyperparameter keys gracefully."""
        mock_wandb_logger = MagicMock(spec=WandbLogger)

        orchestrator = PBTOrchestrator(
            self.pbt_config,
            self.gcp_config,
            wandb_logger=mock_wandb_logger,
        )

        # Changes with missing keys
        changes = [
            ("agent_0", "agent_3", {"learning_rate": 0.0018}),  # Missing c_puct
            ("agent_2", "agent_1", {"c_puct": 1.3}),  # Missing learning_rate
        ]

        orchestrator._log_mutation_events(gen=5, changes=changes)

        # Verify table logged with empty strings for missing keys
        mock_wandb_logger.log_table.assert_called_once()
        table_call = mock_wandb_logger.log_table.call_args
        rows = table_call[0][2]

        # First row: has learning_rate, missing c_puct
        self.assertEqual(rows[0][2], 0.0018)
        self.assertEqual(rows[0][3], "")

        # Second row: missing learning_rate, has c_puct
        self.assertEqual(rows[1][2], "")
        self.assertEqual(rows[1][3], 1.3)


if __name__ == "__main__":
    unittest.main()
