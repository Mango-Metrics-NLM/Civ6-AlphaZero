"""Unit tests for training safeguards module."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from alpha_zero.training.convergence_monitor import ConvergenceStatus
from alpha_zero.training.training_safeguards import (
    CheckpointValidation,
    GradientStatus,
    Intervention,
    InterventionAction,
    SafeguardReport,
    TrainingSafeguard,
    TrainingSafeguardConfig,
)
from alpha_zero.utils.memory_guard import MemoryStatus


class TestTrainingSafeguardConfig(unittest.TestCase):

    def test_config_defaults(self):
        config = TrainingSafeguardConfig()
        self.assertTrue(config.nan_detection_enabled)
        self.assertEqual(config.loss_explosion_threshold, 10.0)
        self.assertEqual(config.loss_explosion_window, 50)
        self.assertEqual(config.gradient_norm_clip, 1.0)
        self.assertEqual(config.gradient_norm_alert_threshold, 5.0)
        self.assertEqual(config.memory_budget_fraction, 0.85)
        self.assertTrue(config.checkpoint_validation_enabled)
        self.assertTrue(config.auto_intervention_enabled)
        self.assertEqual(config.max_consecutive_nan_steps, 3)

    def test_config_custom_values(self):
        config = TrainingSafeguardConfig(
            nan_detection_enabled=False,
            loss_explosion_threshold=5.0,
            loss_explosion_window=100,
            gradient_norm_clip=2.0,
            gradient_norm_alert_threshold=10.0,
            memory_budget_fraction=0.9,
            checkpoint_validation_enabled=False,
            auto_intervention_enabled=False,
            max_consecutive_nan_steps=5,
        )
        self.assertFalse(config.nan_detection_enabled)
        self.assertEqual(config.loss_explosion_threshold, 5.0)
        self.assertEqual(config.loss_explosion_window, 100)
        self.assertEqual(config.gradient_norm_clip, 2.0)
        self.assertEqual(config.gradient_norm_alert_threshold, 10.0)
        self.assertEqual(config.memory_budget_fraction, 0.9)
        self.assertFalse(config.checkpoint_validation_enabled)
        self.assertFalse(config.auto_intervention_enabled)
        self.assertEqual(config.max_consecutive_nan_steps, 5)

    def test_config_validation_loss_explosion_threshold(self):
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(loss_explosion_threshold=1.0)
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(loss_explosion_threshold=0.5)

    def test_config_validation_loss_explosion_window(self):
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(loss_explosion_window=0)
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(loss_explosion_window=-1)

    def test_config_validation_gradient_norm_clip(self):
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(gradient_norm_clip=0)
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(gradient_norm_clip=-1.0)

    def test_config_validation_gradient_norm_alert(self):
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(gradient_norm_alert_threshold=0)
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(gradient_norm_alert_threshold=-1.0)

    def test_config_validation_memory_budget(self):
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(memory_budget_fraction=0.0)
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(memory_budget_fraction=-0.1)
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(memory_budget_fraction=1.1)

    def test_config_validation_max_nan_steps(self):
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(max_consecutive_nan_steps=0)
        with self.assertRaises(ValueError):
            TrainingSafeguardConfig(max_consecutive_nan_steps=-1)

    def test_config_serialization(self):
        config = TrainingSafeguardConfig(
            nan_detection_enabled=False,
            loss_explosion_threshold=5.0,
            max_consecutive_nan_steps=5,
        )
        data = config.to_dict()
        restored = TrainingSafeguardConfig.from_dict(data)
        self.assertEqual(config.nan_detection_enabled, restored.nan_detection_enabled)
        self.assertEqual(config.loss_explosion_threshold, restored.loss_explosion_threshold)
        self.assertEqual(config.max_consecutive_nan_steps, restored.max_consecutive_nan_steps)

    def test_config_from_env(self):
        env_vars = {
            "CIV6AZ_SAFEGUARD_NAN_DETECTION": "false",
            "CIV6AZ_SAFEGUARD_LOSS_EXPLOSION_THRESHOLD": "5.0",
            "CIV6AZ_SAFEGUARD_LOSS_EXPLOSION_WINDOW": "100",
            "CIV6AZ_SAFEGUARD_GRAD_CLIP": "2.0",
            "CIV6AZ_SAFEGUARD_GRAD_ALERT": "10.0",
            "CIV6AZ_SAFEGUARD_MEMORY_BUDGET": "0.9",
            "CIV6AZ_SAFEGUARD_CHECKPOINT_VALIDATION": "false",
            "CIV6AZ_SAFEGUARD_AUTO_INTERVENTION": "false",
            "CIV6AZ_SAFEGUARD_MAX_NAN_STEPS": "5",
        }
        with patch.dict(os.environ, env_vars):
            config = TrainingSafeguardConfig.from_env()
            self.assertFalse(config.nan_detection_enabled)
            self.assertEqual(config.loss_explosion_threshold, 5.0)
            self.assertEqual(config.loss_explosion_window, 100)
            self.assertEqual(config.gradient_norm_clip, 2.0)
            self.assertEqual(config.gradient_norm_alert_threshold, 10.0)
            self.assertEqual(config.memory_budget_fraction, 0.9)
            self.assertFalse(config.checkpoint_validation_enabled)
            self.assertFalse(config.auto_intervention_enabled)
            self.assertEqual(config.max_consecutive_nan_steps, 5)

    def test_config_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("nan_detection_enabled: false\n")
            f.write("loss_explosion_threshold: 5.0\n")
            f.write("max_consecutive_nan_steps: 5\n")
            temp_path = f.name

        try:
            config = TrainingSafeguardConfig.from_yaml(temp_path)
            self.assertFalse(config.nan_detection_enabled)
            self.assertEqual(config.loss_explosion_threshold, 5.0)
            self.assertEqual(config.max_consecutive_nan_steps, 5)
        finally:
            os.unlink(temp_path)


class TestIntervention(unittest.TestCase):

    def test_intervention_creation(self):
        intervention = Intervention(
            action=InterventionAction.REDUCE_LR,
            reason="Loss exploding",
            severity=4,
            metadata={"loss": 100.0},
        )
        self.assertEqual(intervention.action, InterventionAction.REDUCE_LR)
        self.assertEqual(intervention.reason, "Loss exploding")
        self.assertEqual(intervention.severity, 4)
        self.assertEqual(intervention.metadata["loss"], 100.0)

    def test_intervention_severity_validation(self):
        with self.assertRaises(ValueError):
            Intervention(
                action=InterventionAction.REDUCE_LR, reason="test", severity=0
            )
        with self.assertRaises(ValueError):
            Intervention(
                action=InterventionAction.REDUCE_LR, reason="test", severity=6
            )

    def test_intervention_to_dict(self):
        intervention = Intervention(
            action=InterventionAction.ROLLBACK,
            reason="NaN detected",
            severity=5,
            metadata={"step": 100},
        )
        data = intervention.to_dict()
        self.assertEqual(data["action"], InterventionAction.ROLLBACK)
        self.assertEqual(data["reason"], "NaN detected")
        self.assertEqual(data["severity"], 5)
        self.assertEqual(data["metadata"]["step"], 100)


class TestCheckpointValidation(unittest.TestCase):

    def test_checkpoint_validation_valid(self):
        validation = CheckpointValidation(
            valid=True,
            path="/path/to/checkpoint.pt",
            num_parameters=1000000,
            has_nan=False,
            has_inf=False,
        )
        self.assertTrue(validation.valid)
        self.assertEqual(validation.path, "/path/to/checkpoint.pt")
        self.assertEqual(validation.num_parameters, 1000000)
        self.assertFalse(validation.has_nan)
        self.assertFalse(validation.has_inf)
        self.assertEqual(validation.dtype_issues, [])
        self.assertIsNone(validation.error)

    def test_checkpoint_validation_with_nan(self):
        validation = CheckpointValidation(
            valid=False,
            path="/path/to/checkpoint.pt",
            has_nan=True,
        )
        self.assertFalse(validation.valid)
        self.assertTrue(validation.has_nan)

    def test_checkpoint_validation_to_dict(self):
        validation = CheckpointValidation(
            valid=False,
            path="/path/to/checkpoint.pt",
            has_nan=True,
            dtype_issues=["param1: int64"],
            error="NaN detected",
        )
        data = validation.to_dict()
        self.assertFalse(data["valid"])
        self.assertEqual(data["path"], "/path/to/checkpoint.pt")
        self.assertTrue(data["has_nan"])
        self.assertEqual(data["dtype_issues"], ["param1: int64"])
        self.assertEqual(data["error"], "NaN detected")


class TestSafeguardReport(unittest.TestCase):

    def test_safeguard_report_creation(self):
        report = SafeguardReport(
            step=100,
            loss=0.5,
            grad_norm=0.3,
            nan_detected=False,
            loss_exploding=False,
            gradient_status=GradientStatus.NORMAL,
        )
        self.assertEqual(report.step, 100)
        self.assertEqual(report.loss, 0.5)
        self.assertEqual(report.grad_norm, 0.3)
        self.assertFalse(report.nan_detected)
        self.assertFalse(report.loss_exploding)
        self.assertEqual(report.gradient_status, GradientStatus.NORMAL)
        self.assertIsNone(report.memory_status)
        self.assertIsNone(report.convergence_status)
        self.assertIsNone(report.intervention)

    def test_safeguard_report_with_intervention(self):
        intervention = Intervention(
            action=InterventionAction.REDUCE_LR,
            reason="Loss exploding",
            severity=4,
        )
        report = SafeguardReport(
            step=100,
            loss=100.0,
            grad_norm=50.0,
            loss_exploding=True,
            gradient_status=GradientStatus.EXPLODING,
            intervention=intervention,
        )
        self.assertEqual(report.intervention.action, InterventionAction.REDUCE_LR)
        self.assertTrue(report.loss_exploding)

    def test_safeguard_report_to_dict(self):
        intervention = Intervention(
            action=InterventionAction.ROLLBACK,
            reason="NaN detected",
            severity=5,
        )
        report = SafeguardReport(
            step=100,
            loss=math.nan,
            grad_norm=1.0,
            nan_detected=True,
            intervention=intervention,
        )
        data = report.to_dict()
        self.assertEqual(data["step"], 100)
        self.assertTrue(data["nan_detected"])
        self.assertIsNotNone(data["intervention"])
        self.assertEqual(data["intervention"]["action"], InterventionAction.ROLLBACK)


class TestTrainingSafeguardNanDetection(unittest.TestCase):

    def test_nan_detection_loss(self):
        safeguard = TrainingSafeguard()
        result = safeguard.check_nan(math.nan, 1.0)
        self.assertTrue(result)
        self.assertEqual(safeguard.consecutive_nan_count, 1)

    def test_nan_detection_gradient(self):
        safeguard = TrainingSafeguard()
        result = safeguard.check_nan(1.0, math.nan)
        self.assertTrue(result)
        self.assertEqual(safeguard.consecutive_nan_count, 1)

    def test_nan_detection_both(self):
        safeguard = TrainingSafeguard()
        result = safeguard.check_nan(math.nan, math.nan)
        self.assertTrue(result)
        self.assertEqual(safeguard.consecutive_nan_count, 1)

    def test_nan_detection_disabled(self):
        config = TrainingSafeguardConfig(nan_detection_enabled=False)
        safeguard = TrainingSafeguard(config)
        report = safeguard.check_step(math.nan, 1.0, step=1)
        self.assertFalse(report.nan_detected)

    def test_consecutive_nan_threshold(self):
        config = TrainingSafeguardConfig(max_consecutive_nan_steps=3)
        safeguard = TrainingSafeguard(config)

        report1 = safeguard.check_step(math.nan, 1.0, step=1)
        self.assertTrue(report1.nan_detected)
        self.assertIsNone(report1.intervention)

        report2 = safeguard.check_step(math.nan, 1.0, step=2)
        self.assertTrue(report2.nan_detected)
        self.assertIsNone(report2.intervention)

        report3 = safeguard.check_step(math.nan, 1.0, step=3)
        self.assertTrue(report3.nan_detected)
        self.assertIsNotNone(report3.intervention)
        self.assertEqual(report3.intervention.action, InterventionAction.ROLLBACK)
        self.assertEqual(report3.intervention.severity, 5)

    def test_consecutive_nan_resets_on_valid(self):
        safeguard = TrainingSafeguard()

        safeguard.check_step(math.nan, 1.0, step=1)
        self.assertEqual(safeguard.consecutive_nan_count, 1)

        safeguard.check_step(math.nan, 1.0, step=2)
        self.assertEqual(safeguard.consecutive_nan_count, 2)

        safeguard.check_step(0.5, 1.0, step=3)
        self.assertEqual(safeguard.consecutive_nan_count, 0)


class TestTrainingSafeguardLossExplosion(unittest.TestCase):

    def test_loss_explosion_above_threshold(self):
        config = TrainingSafeguardConfig(loss_explosion_threshold=2.0)
        safeguard = TrainingSafeguard(config)

        for i in range(20):
            safeguard.check_step(1.0, 0.5, step=i)

        report = safeguard.check_step(10.0, 0.5, step=20)
        self.assertTrue(report.loss_exploding)

    def test_loss_explosion_within_normal(self):
        safeguard = TrainingSafeguard()

        for i in range(20):
            safeguard.check_step(1.0, 0.5, step=i)

        report = safeguard.check_step(1.5, 0.5, step=20)
        self.assertFalse(report.loss_exploding)

    def test_loss_explosion_rolling_baseline(self):
        config = TrainingSafeguardConfig(
            loss_explosion_window=10, loss_explosion_threshold=2.0
        )
        safeguard = TrainingSafeguard(config)

        for i in range(20):
            safeguard.check_step(1.0, 0.5, step=i)

        self.assertEqual(len(safeguard._loss_history), 10)

        report = safeguard.check_step(5.0, 0.5, step=20)
        self.assertTrue(report.loss_exploding)

    def test_loss_explosion_insufficient_history(self):
        safeguard = TrainingSafeguard()

        for i in range(5):
            report = safeguard.check_step(1.0, 0.5, step=i)
            self.assertFalse(report.loss_exploding)

        report = safeguard.check_step(100.0, 0.5, step=5)
        self.assertFalse(report.loss_exploding)

    def test_loss_explosion_ignores_nan(self):
        safeguard = TrainingSafeguard()

        for i in range(20):
            safeguard.check_step(1.0, 0.5, step=i)

        report = safeguard.check_step(math.nan, 0.5, step=20)
        self.assertFalse(report.loss_exploding)


class TestTrainingSafeguardGradientHealth(unittest.TestCase):

    def test_gradient_normal(self):
        safeguard = TrainingSafeguard()
        status = safeguard.check_gradient_health(0.5)
        self.assertEqual(status, GradientStatus.NORMAL)

    def test_gradient_high(self):
        config = TrainingSafeguardConfig(gradient_norm_alert_threshold=5.0)
        safeguard = TrainingSafeguard(config)
        status = safeguard.check_gradient_health(6.0)
        self.assertEqual(status, GradientStatus.HIGH)

    def test_gradient_exploding(self):
        config = TrainingSafeguardConfig(gradient_norm_alert_threshold=5.0)
        safeguard = TrainingSafeguard(config)
        status = safeguard.check_gradient_health(60.0)
        self.assertEqual(status, GradientStatus.EXPLODING)

    def test_gradient_exploding_nan(self):
        safeguard = TrainingSafeguard()
        status = safeguard.check_gradient_health(math.nan)
        self.assertEqual(status, GradientStatus.EXPLODING)


class TestTrainingSafeguardMemoryCheck(unittest.TestCase):

    def test_memory_check_delegates(self):
        mock_guard = MagicMock()
        mock_guard.get_memory_status.return_value = MemoryStatus.NORMAL
        safeguard = TrainingSafeguard(memory_guard=mock_guard)

        status = safeguard.check_memory(device="gpu:0")
        self.assertEqual(status, MemoryStatus.NORMAL)
        mock_guard.get_memory_status.assert_called_once_with(device="gpu:0")

    def test_memory_check_no_guard(self):
        safeguard = TrainingSafeguard()
        status = safeguard.check_memory(device="gpu:0")
        self.assertIsNone(status)

    def test_memory_check_exception_handling(self):
        mock_guard = MagicMock()
        mock_guard.get_memory_status.side_effect = RuntimeError("GPU not available")
        safeguard = TrainingSafeguard(memory_guard=mock_guard)

        status = safeguard.check_memory(device="gpu:0")
        self.assertIsNone(status)


class TestTrainingSafeguardCheckpointValidation(unittest.TestCase):

    @patch("alpha_zero.training.training_safeguards.torch")
    def test_checkpoint_validation_valid(self, mock_torch):
        # Create a mock tensor class that passes isinstance checks
        class MockTensor:
            def __init__(self):
                self.dtype = "float32"

            def numel(self):
                return 1000

        # Set up torch types as actual types (not strings)
        mock_torch.float32 = type("float32", (), {})()
        mock_torch.float16 = type("float16", (), {})()
        mock_torch.bfloat16 = type("bfloat16", (), {})()

        # Create mock tensor instance
        mock_tensor = MockTensor()
        mock_tensor.dtype = mock_torch.float32

        # Make isinstance check pass by setting Tensor
        mock_torch.Tensor = MockTensor

        # Set up the mock functions
        mock_torch.isnan.return_value.any.return_value = False
        mock_torch.isinf.return_value.any.return_value = False
        mock_torch.load.return_value = {"param1": mock_tensor}

        safeguard = TrainingSafeguard()
        validation = safeguard.validate_checkpoint("/path/to/checkpoint.pt")

        self.assertTrue(validation.valid)
        self.assertEqual(validation.num_parameters, 1000)
        self.assertFalse(validation.has_nan)
        self.assertFalse(validation.has_inf)

    @patch("alpha_zero.training.training_safeguards.torch")
    def test_checkpoint_validation_nan_weights(self, mock_torch):
        # Create a mock tensor class that passes isinstance checks
        class MockTensor:
            def __init__(self):
                self.dtype = "float32"

            def numel(self):
                return 1000

        # Set up torch types as actual types (not strings)
        mock_torch.float32 = type("float32", (), {})()
        mock_torch.float16 = type("float16", (), {})()
        mock_torch.bfloat16 = type("bfloat16", (), {})()

        # Create mock tensor instance
        mock_tensor = MockTensor()
        mock_tensor.dtype = mock_torch.float32

        # Make isinstance check pass by setting Tensor
        mock_torch.Tensor = MockTensor

        # Set up the mock functions (NaN detected)
        mock_torch.isnan.return_value.any.return_value = True
        mock_torch.isinf.return_value.any.return_value = False
        mock_torch.load.return_value = {"param1": mock_tensor}

        safeguard = TrainingSafeguard()
        validation = safeguard.validate_checkpoint("/path/to/checkpoint.pt")

        self.assertFalse(validation.valid)
        self.assertTrue(validation.has_nan)

    @patch("alpha_zero.training.training_safeguards.torch")
    def test_checkpoint_validation_file_not_found(self, mock_torch):
        mock_torch.load.side_effect = FileNotFoundError("File not found")

        safeguard = TrainingSafeguard()
        validation = safeguard.validate_checkpoint("/nonexistent/checkpoint.pt")

        self.assertFalse(validation.valid)
        self.assertIsNotNone(validation.error)
        self.assertIn("File not found", validation.error)

    def test_checkpoint_validation_disabled(self):
        config = TrainingSafeguardConfig(checkpoint_validation_enabled=False)
        safeguard = TrainingSafeguard(config)
        validation = safeguard.validate_checkpoint("/path/to/checkpoint.pt")

        self.assertTrue(validation.valid)
        self.assertIn("disabled", validation.error.lower())


class TestTrainingSafeguardInterventions(unittest.TestCase):

    def test_intervention_reduce_lr_loss_explosion(self):
        config = TrainingSafeguardConfig(loss_explosion_threshold=2.0)
        safeguard = TrainingSafeguard(config)

        for i in range(20):
            safeguard.check_step(1.0, 0.5, step=i)

        report = safeguard.check_step(10.0, 0.5, step=20)

        self.assertIsNotNone(report.intervention)
        self.assertEqual(report.intervention.action, InterventionAction.REDUCE_LR)
        self.assertEqual(report.intervention.severity, 4)

    def test_intervention_reduce_lr_gradient_exploding(self):
        config = TrainingSafeguardConfig(gradient_norm_alert_threshold=5.0)
        safeguard = TrainingSafeguard(config)

        report = safeguard.check_step(1.0, 60.0, step=1)

        self.assertIsNotNone(report.intervention)
        self.assertEqual(report.intervention.action, InterventionAction.REDUCE_LR)
        self.assertEqual(report.intervention.severity, 4)

    def test_intervention_rollback_consecutive_nan(self):
        config = TrainingSafeguardConfig(max_consecutive_nan_steps=2)
        safeguard = TrainingSafeguard(config)

        safeguard.check_step(math.nan, 1.0, step=1)
        report = safeguard.check_step(math.nan, 1.0, step=2)

        self.assertIsNotNone(report.intervention)
        self.assertEqual(report.intervention.action, InterventionAction.ROLLBACK)
        self.assertEqual(report.intervention.severity, 5)

    def test_intervention_reduce_batch_memory_oom(self):
        mock_guard = MagicMock()
        mock_guard.get_memory_status.return_value = MemoryStatus.OOM_RISK
        safeguard = TrainingSafeguard(memory_guard=mock_guard)

        report = safeguard.check_step(1.0, 0.5, step=1, device="gpu:0")

        self.assertIsNotNone(report.intervention)
        self.assertEqual(report.intervention.action, InterventionAction.REDUCE_BATCH_SIZE)
        self.assertEqual(report.intervention.severity, 4)

    def test_intervention_reduce_batch_memory_critical(self):
        mock_guard = MagicMock()
        mock_guard.get_memory_status.return_value = MemoryStatus.CRITICAL
        safeguard = TrainingSafeguard(memory_guard=mock_guard)

        report = safeguard.check_step(1.0, 0.5, step=1, device="gpu:0")

        self.assertIsNotNone(report.intervention)
        self.assertEqual(report.intervention.action, InterventionAction.REDUCE_BATCH_SIZE)
        self.assertEqual(report.intervention.severity, 3)

    def test_intervention_increase_exploration_stalled(self):
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = ConvergenceStatus.STALLED
        safeguard = TrainingSafeguard(convergence_monitor=mock_monitor)

        report = safeguard.check_step(1.0, 0.5, step=1)

        self.assertIsNotNone(report.intervention)
        self.assertEqual(report.intervention.action, InterventionAction.INCREASE_EXPLORATION)
        self.assertEqual(report.intervention.severity, 2)

    def test_intervention_reduce_lr_diverging(self):
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = ConvergenceStatus.DIVERGING
        safeguard = TrainingSafeguard(convergence_monitor=mock_monitor)

        report = safeguard.check_step(1.0, 0.5, step=1)

        self.assertIsNotNone(report.intervention)
        self.assertEqual(report.intervention.action, InterventionAction.REDUCE_LR)
        self.assertEqual(report.intervention.severity, 4)

    def test_intervention_none_when_healthy(self):
        safeguard = TrainingSafeguard()
        report = safeguard.check_step(1.0, 0.5, step=1)
        self.assertIsNone(report.intervention)

    def test_auto_intervention_disabled(self):
        config = TrainingSafeguardConfig(
            auto_intervention_enabled=False,
            max_consecutive_nan_steps=1,
        )
        safeguard = TrainingSafeguard(config)

        report = safeguard.check_step(math.nan, 1.0, step=1)

        self.assertTrue(report.nan_detected)
        self.assertIsNone(report.intervention)


class TestTrainingSafeguardIntegration(unittest.TestCase):

    def test_check_step_full_pipeline(self):
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = ConvergenceStatus.IMPROVING
        mock_guard = MagicMock()
        mock_guard.get_memory_status.return_value = MemoryStatus.NORMAL

        safeguard = TrainingSafeguard(
            convergence_monitor=mock_monitor,
            memory_guard=mock_guard,
        )

        report = safeguard.check_step(1.0, 0.5, step=100, device="gpu:0", lr=0.001)

        self.assertEqual(report.step, 100)
        self.assertEqual(report.loss, 1.0)
        self.assertEqual(report.grad_norm, 0.5)
        self.assertFalse(report.nan_detected)
        self.assertFalse(report.loss_exploding)
        self.assertEqual(report.gradient_status, GradientStatus.NORMAL)
        self.assertEqual(report.memory_status, MemoryStatus.NORMAL)
        self.assertEqual(report.convergence_status, ConvergenceStatus.IMPROVING)
        self.assertIsNone(report.intervention)
        self.assertIsNotNone(report.timestamp)

        mock_monitor.get_status.assert_called_once()
        mock_guard.get_memory_status.assert_called_once_with(device="gpu:0")

    def test_safeguard_report_fields(self):
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = ConvergenceStatus.STABLE
        mock_guard = MagicMock()
        mock_guard.get_memory_status.return_value = MemoryStatus.WARNING

        safeguard = TrainingSafeguard(
            convergence_monitor=mock_monitor,
            memory_guard=mock_guard,
        )

        report = safeguard.check_step(2.0, 1.5, step=50)

        self.assertEqual(report.step, 50)
        self.assertEqual(report.loss, 2.0)
        self.assertEqual(report.grad_norm, 1.5)
        self.assertEqual(report.memory_status, MemoryStatus.WARNING)
        self.assertEqual(report.convergence_status, ConvergenceStatus.STABLE)

        data = report.to_dict()
        self.assertIn("step", data)
        self.assertIn("loss", data)
        self.assertIn("grad_norm", data)
        self.assertIn("timestamp", data)

    def test_reset_clears_state(self):
        safeguard = TrainingSafeguard()

        for i in range(20):
            safeguard.check_step(1.0, 0.5, step=i)

        safeguard.check_step(math.nan, 1.0, step=20)

        self.assertGreater(len(safeguard._loss_history), 0)
        self.assertGreater(safeguard.consecutive_nan_count, 0)
        self.assertGreater(safeguard.step_count, 0)

        safeguard.reset()

        self.assertEqual(len(safeguard._loss_history), 0)
        self.assertEqual(safeguard.consecutive_nan_count, 0)
        self.assertEqual(safeguard.step_count, 0)
        self.assertEqual(safeguard.intervention_count, 0)
        self.assertIsNone(safeguard._last_valid_loss)

    def test_step_count_increments(self):
        safeguard = TrainingSafeguard()

        self.assertEqual(safeguard.step_count, 0)

        safeguard.check_step(1.0, 0.5, step=10)
        self.assertEqual(safeguard.step_count, 10)

        safeguard.check_step(1.0, 0.5, step=20)
        self.assertEqual(safeguard.step_count, 20)

    def test_intervention_count_increments(self):
        config = TrainingSafeguardConfig(
            loss_explosion_threshold=2.0,
            gradient_norm_alert_threshold=5.0,
        )
        safeguard = TrainingSafeguard(config)

        self.assertEqual(safeguard.intervention_count, 0)

        for i in range(20):
            safeguard.check_step(1.0, 0.5, step=i)

        safeguard.check_step(10.0, 0.5, step=20)
        self.assertEqual(safeguard.intervention_count, 1)

        safeguard.check_step(1.0, 60.0, step=21)
        self.assertEqual(safeguard.intervention_count, 2)


if __name__ == "__main__":
    unittest.main()
