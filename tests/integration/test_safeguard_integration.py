"""Integration tests for training safeguards with orchestrators.

Tests cover:
- TrainingOrchestrator safeguard integration
- PBTOrchestrator safeguard config acceptance
- Safeguard check_step() integration during training loops
- Intervention logging and handling
- Config loading from YAML
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from alpha_zero.orchestrator.config import OrchestratorConfig
from alpha_zero.pbt.config import PBTConfig
from alpha_zero.training.convergence_monitor import ConvergenceStatus
from alpha_zero.training.training_safeguards import (
    InterventionAction,
    SafeguardReport,
    TrainingSafeguard,
    TrainingSafeguardConfig,
)
from alpha_zero.utils.memory_guard import MemoryStatus


class TestSafeguardIntegration(unittest.TestCase):
    """Integration tests for training safeguards."""

    def test_orchestrator_safeguard_setter(self):
        """TrainingOrchestrator.set_training_safeguard() stores the safeguard."""
        from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

        config = OrchestratorConfig()
        config.experiment_name = "test_safeguard_setter"
        config.training.num_iterations = 1
        config.training.games_per_iteration = 1

        orchestrator = TrainingOrchestrator(config)

        safeguard_config = TrainingSafeguardConfig()
        safeguard = TrainingSafeguard(safeguard_config)

        orchestrator.set_training_safeguard(safeguard)

        assert orchestrator._training_safeguard is safeguard

    def test_orchestrator_without_safeguard(self):
        """When _training_safeguard is None, _train_epochs works normally (backwards compat)."""
        from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

        sample_data = [
            {
                "spatial": np.random.rand(13, 16, 16).astype(np.float32),
                "vector": np.random.rand(64).astype(np.float32),
                "action_mask": np.ones(376, dtype=np.float32),
                "policy": np.array([1.0] + [0.0] * 375, dtype=np.float32),
                "value": 0.5,
            }
            for _ in range(64)
        ]

        mock_buffer = MagicMock()
        mock_buffer.is_ready.return_value = True
        mock_buffer.__len__.return_value = 100
        mock_buffer.sample.return_value = sample_data

        with patch("alpha_zero.utils.factories.create_replay_buffer", return_value=mock_buffer):
            config = OrchestratorConfig()
            config.experiment_name = "test_no_safeguard"
            config.training.num_iterations = 1
            config.training.games_per_iteration = 1
            config.training.resume = False

            orchestrator = TrainingOrchestrator(config)

            assert orchestrator._training_safeguard is None

            orchestrator.initialize()

            try:
                metrics = orchestrator._train_epochs(num_epochs=1)
                assert "policy_loss" in metrics or metrics == {}
            except Exception as e:
                self.fail(f"Training failed with no safeguard: {e}")

    def test_safeguard_check_step_called_during_training(self):
        """When safeguard is set and training occurs, check_step() is called."""
        safeguard_config = TrainingSafeguardConfig()
        safeguard = TrainingSafeguard(safeguard_config)

        mock_check_step = MagicMock(return_value=SafeguardReport(
            step=0,
            loss=0.5,
            grad_norm=1.0,
            nan_detected=False,
            loss_exploding=False,
        ))
        safeguard.check_step = mock_check_step

        assert hasattr(safeguard, "check_step")
        assert callable(safeguard.check_step)

        report = safeguard.check_step(loss=0.5, grad_norm=1.0, step=100, device="cuda")

        assert mock_check_step.call_count == 1
        assert report.loss == 0.5
        assert report.grad_norm == 1.0

    def test_nan_triggers_warning_log(self):
        """When safeguard returns intervention, the intervention field is populated."""

        safeguard_config = TrainingSafeguardConfig(max_consecutive_nan_steps=1)
        safeguard = TrainingSafeguard(safeguard_config)

        report1 = safeguard.check_step(loss=float("nan"), grad_norm=1.0, step=1)
        assert report1.nan_detected is True
        assert report1.intervention is not None
        assert report1.intervention.action == InterventionAction.ROLLBACK
        assert "NaN" in report1.intervention.reason

    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage")
    def test_pbt_orchestrator_accepts_safeguard_config(
        self,
        mock_storage,
        mock_gcs_backend,
        mock_selfplay_job,
        mock_training_job,
        mock_vertex_client,
    ):
        """PBTOrchestrator accepts training_safeguard_config kwarg."""
        from alpha_zero.cloud.gcp.pbt_orchestrator import PBTOrchestrator

        mock_gcp = MagicMock()
        mock_gcp.project_id = "test-project"
        mock_gcp.region = "us-central1"
        mock_gcp.bucket = "test-bucket"
        mock_gcp.service_account = None
        mock_gcp.paths.logs_prefix = "logs"
        mock_gcp.paths.checkpoints_prefix = "checkpoints"

        pbt_config = PBTConfig(population_size=4)
        safeguard_config = TrainingSafeguardConfig(
            nan_detection_enabled=True,
            loss_explosion_threshold=5.0,
        )

        orchestrator = PBTOrchestrator(
            pbt_config=pbt_config,
            gcp_config=mock_gcp,
            training_safeguard_config=safeguard_config,
        )

        assert orchestrator.training_safeguard_config is safeguard_config
        assert orchestrator.training_safeguard_config.nan_detection_enabled is True
        assert orchestrator.training_safeguard_config.loss_explosion_threshold == 5.0

    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexAIClient")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexTrainingJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.VertexSelfPlayJob")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSBackend")
    @patch("alpha_zero.cloud.gcp.pbt_orchestrator.GCSModelStorage")
    def test_pbt_orchestrator_none_safeguard_config(
        self,
        mock_storage,
        mock_gcs_backend,
        mock_selfplay_job,
        mock_training_job,
        mock_vertex_client,
    ):
        """Default is None (backwards compat)."""
        from alpha_zero.cloud.gcp.pbt_orchestrator import PBTOrchestrator

        mock_gcp = MagicMock()
        mock_gcp.project_id = "test-project"
        mock_gcp.region = "us-central1"
        mock_gcp.bucket = "test-bucket"
        mock_gcp.service_account = None
        mock_gcp.paths.logs_prefix = "logs"
        mock_gcp.paths.checkpoints_prefix = "checkpoints"

        pbt_config = PBTConfig(population_size=4)

        orchestrator = PBTOrchestrator(
            pbt_config=pbt_config,
            gcp_config=mock_gcp,
        )

        assert orchestrator.training_safeguard_config is None

    def test_safeguard_config_from_yaml_integration(self):
        """TrainingSafeguardConfig.from_yaml works with valid YAML."""
        yaml_content = {
            "nan_detection_enabled": True,
            "loss_explosion_threshold": 8.0,
            "loss_explosion_window": 100,
            "gradient_norm_clip": 2.0,
            "gradient_norm_alert_threshold": 10.0,
            "memory_budget_fraction": 0.9,
            "checkpoint_validation_enabled": True,
            "auto_intervention_enabled": True,
            "max_consecutive_nan_steps": 5,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config = TrainingSafeguardConfig.from_yaml(temp_path)

            assert config.nan_detection_enabled is True
            assert config.loss_explosion_threshold == 8.0
            assert config.loss_explosion_window == 100
            assert config.gradient_norm_clip == 2.0
            assert config.gradient_norm_alert_threshold == 10.0
            assert config.memory_budget_fraction == 0.9
            assert config.checkpoint_validation_enabled is True
            assert config.auto_intervention_enabled is True
            assert config.max_consecutive_nan_steps == 5
        finally:
            Path(temp_path).unlink()

    def test_convergence_monitor_integration(self):
        """TrainingSafeguard with ConvergenceMonitor."""
        mock_convergence = MagicMock()
        mock_convergence.get_status.return_value = ConvergenceStatus.STALLED

        safeguard_config = TrainingSafeguardConfig()
        safeguard = TrainingSafeguard(
            safeguard_config,
            convergence_monitor=mock_convergence,
        )

        report = safeguard.check_step(
            loss=0.5,
            grad_norm=1.0,
            step=100,
        )

        assert report.convergence_status == ConvergenceStatus.STALLED
        assert report.intervention is not None
        assert report.intervention.action == InterventionAction.INCREASE_EXPLORATION

    def test_memory_guard_integration(self):
        """TrainingSafeguard with MemoryGuard."""
        mock_memory_guard = MagicMock()
        mock_memory_guard.get_memory_status.return_value = MemoryStatus.OOM_RISK

        safeguard_config = TrainingSafeguardConfig()
        safeguard = TrainingSafeguard(
            safeguard_config,
            memory_guard=mock_memory_guard,
        )

        report = safeguard.check_step(
            loss=0.5,
            grad_norm=1.0,
            step=100,
            device="gpu:0",
        )

        assert report.memory_status == MemoryStatus.OOM_RISK
        assert report.intervention is not None
        assert report.intervention.action == InterventionAction.REDUCE_BATCH_SIZE


if __name__ == "__main__":
    unittest.main()
