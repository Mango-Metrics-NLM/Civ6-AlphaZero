"""
Unit Tests for Phase 14 Training Worker Enhancements.

Tests new features in training_worker.py:
- Learning rate scheduling integration
- Training diagnostics tracking
- Training report generation
- Performance profiling
- Backwards-compatible behavior when new features are disabled
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from alpha_zero.cloud.gcp.training_worker import parse_args, prepare_dataset, train_epoch


@pytest.fixture
def training_setup():
    """Create minimal model, optimizer, and dataloader for testing.

    Returns:
        Tuple of (model, optimizer, dataloader, device).
    """
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    config = TransformerConfig(
        spatial_channels=13,
        spatial_size=8,
        vector_dim=64,
        action_space_size=376,
        embed_dim=64,
        num_layers=2,
        num_heads=8,
    )
    model = Civ6TransformerNetwork(config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Create tiny dataset: 4 examples (2 batches of 2)
    batch_size = 2
    num_examples = 4
    spatial = torch.randn(num_examples, 13 * 8 * 8)
    vector = torch.randn(num_examples, 64)
    policies = torch.softmax(torch.randn(num_examples, 376), dim=-1)
    values = torch.randn(num_examples)

    dataset = TensorDataset(spatial, vector, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")

    return model, optimizer, dataloader, device


class TestTrainEpochBackwardsCompatibility:
    """Test train_epoch with new params disabled (backwards compatible)."""

    def test_train_epoch_constant_lr(self, training_setup):
        """Test train_epoch without lr_scheduler (constant LR)."""
        model, optimizer, dataloader, device = training_setup

        # Call train_epoch with all new params as None
        total_loss, policy_loss, value_loss, num_batches = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            global_step=0,
            wandb_logger=None,
            spatial_channels=13,
            spatial_size=8,
            wandb_log_interval=10,
            lr_scheduler=None,
            diagnostics=None,
            report=None,
            profiler=None,
        )

        # Verify returns
        assert isinstance(total_loss, float)
        assert isinstance(policy_loss, float)
        assert isinstance(value_loss, float)
        assert isinstance(num_batches, int)
        assert num_batches == 2  # 4 examples / batch_size 2
        assert total_loss > 0
        assert policy_loss > 0
        assert value_loss > 0

    def test_train_epoch_no_errors_without_new_features(self, training_setup):
        """Test that train_epoch works correctly without diagnostics/report/profiler."""
        model, optimizer, dataloader, device = training_setup

        # Run multiple times to ensure stability
        for _ in range(3):
            total_loss, _policy_loss, _value_loss, num_batches = train_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                global_step=0,
                wandb_logger=None,
                spatial_channels=13,
                spatial_size=8,
            )

            assert total_loss > 0
            assert num_batches == 2


class TestTrainEpochWithDiagnostics:
    """Test train_epoch with TrainingDiagnostics enabled."""

    def test_train_epoch_with_diagnostics(self, training_setup):
        """Test train_epoch with real TrainingDiagnostics."""
        model, optimizer, dataloader, device = training_setup

        try:
            from alpha_zero.training.diagnostics import TrainingDiagnostics
        except ImportError:
            pytest.skip("TrainingDiagnostics not available")

        diagnostics = TrainingDiagnostics(window_size=10)

        total_loss, policy_loss, value_loss, num_batches = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            global_step=0,
            wandb_logger=None,
            spatial_channels=13,
            spatial_size=8,
            diagnostics=diagnostics,
        )

        # Verify returns are correct
        assert isinstance(total_loss, float)
        assert isinstance(policy_loss, float)
        assert isinstance(value_loss, float)
        assert num_batches == 2

        # Diagnostics should have computed metrics (we can't verify exact values,
        # but we can check it didn't crash)
        assert total_loss > 0


class TestTrainEpochWithReport:
    """Test train_epoch with TrainingReport enabled."""

    def test_train_epoch_with_report(self, training_setup):
        """Test train_epoch with TrainingReport and verify record_step was called."""
        model, optimizer, dataloader, device = training_setup

        try:
            from alpha_zero.training.training_report import TrainingReport, TrainingReportConfig
        except ImportError:
            pytest.skip("TrainingReport not available")

        # Create report with interval=1 so it reports every step
        report = TrainingReport(TrainingReportConfig(report_interval=1))

        _total_loss, _policy_loss, _value_loss, num_batches = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            global_step=0,
            wandb_logger=None,
            spatial_channels=13,
            spatial_size=8,
            report=report,
        )

        # Verify returns
        assert num_batches == 2

        # Verify report recorded steps (generate_report clears _losses, so check report_count)
        assert report.report_count > 0, "Report should have generated at least one report"


class TestTrainEpochWithProfiler:
    """Test train_epoch with PerformanceProfiler enabled."""

    def test_train_epoch_with_profiler(self, training_setup):
        """Test train_epoch with PerformanceProfiler and verify timing sections."""
        model, optimizer, dataloader, device = training_setup

        try:
            from alpha_zero.utils.logging_config import PerformanceProfiler
        except ImportError:
            pytest.skip("PerformanceProfiler not available")

        profiler = PerformanceProfiler()

        _total_loss, _policy_loss, _value_loss, num_batches = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            global_step=0,
            wandb_logger=None,
            spatial_channels=13,
            spatial_size=8,
            profiler=profiler,
        )

        # Verify returns
        assert num_batches == 2

        # Verify profiler recorded sections
        forward_summary = profiler.get_summary("forward_pass")
        backward_summary = profiler.get_summary("backward_pass")

        assert forward_summary is not None, "Profiler should have forward_pass timings"
        assert backward_summary is not None, "Profiler should have backward_pass timings"

        # Check that timings are reasonable (> 0)
        assert forward_summary["count"] > 0
        assert forward_summary["total"] > 0
        assert backward_summary["count"] > 0
        assert backward_summary["total"] > 0


class TestTrainEpochWithLRScheduler:
    """Test train_epoch with learning rate scheduler."""

    def test_train_epoch_with_lr_scheduler(self, training_setup):
        """Test train_epoch with lr_scheduler updates optimizer LR."""
        model, optimizer, dataloader, device = training_setup

        # Create a mock lr_scheduler that returns a fixed LR
        mock_scheduler = MagicMock()
        fixed_lr = 0.00042
        mock_scheduler.get_learning_rate.return_value = fixed_lr

        # Initial LR is different
        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr != fixed_lr

        _total_loss, _policy_loss, _value_loss, num_batches = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            global_step=0,
            wandb_logger=None,
            spatial_channels=13,
            spatial_size=8,
            lr_scheduler=mock_scheduler,
        )

        # Verify scheduler was called
        assert mock_scheduler.get_learning_rate.call_count == num_batches

        # Verify optimizer LR was updated
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr == fixed_lr


class TestParseArgs:
    """Test parse_args function."""

    def test_parse_args_lr_args(self):
        """Test that parse_args includes new LR schedule arguments."""
        test_args = [
            "training_worker.py",
            "--iteration=5",
            "--trajectory_iterations=3,4,5",
            "--lr_schedule_type=warmup_cosine",
            "--lr_initial=0.002",
            "--lr_final=0.0001",
            "--lr_warmup_steps=1000",
            "--lr_decay_steps=50000",
            "--weight_decay=0.02",
            "--diagnostic_window=200",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        # Verify new args are present
        assert args.lr_schedule_type == "warmup_cosine"
        assert args.lr_initial == 0.002
        assert args.lr_final == 0.0001
        assert args.lr_warmup_steps == 1000
        assert args.lr_decay_steps == 50000
        assert args.weight_decay == 0.02
        assert args.diagnostic_window == 200

    def test_parse_args_defaults(self):
        """Test that parse_args has correct defaults for new arguments."""
        test_args = [
            "training_worker.py",
            "--iteration=1",
            "--trajectory_iterations=1",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        # Verify defaults
        assert args.lr_schedule_type == "constant"
        assert args.lr_initial == 0.001
        assert args.lr_final == 0.0001
        assert args.lr_warmup_steps == 0
        assert args.lr_decay_steps == 0
        assert args.weight_decay == 0.01
        assert args.diagnostic_window == 100


class TestPrepareDataset:
    """Test prepare_dataset function."""

    def test_prepare_dataset_normal(self):
        """Test prepare_dataset with normal trajectories."""
        # Create mock trajectories (new format with spatial/vector)
        trajectories = [
            {
                "spatial": [[0.1] * (13 * 8 * 8) for _ in range(3)],
                "vector": [[0.5] * 64 for _ in range(3)],
                "policies": [[1.0 / 376] * 376 for _ in range(3)],
                "rewards": [0.0, 0.0, 1.0],
            },
            {
                "spatial": [[0.2] * (13 * 8 * 8) for _ in range(2)],
                "vector": [[0.6] * 64 for _ in range(2)],
                "policies": [[1.0 / 376] * 376 for _ in range(2)],
                "rewards": [0.0, -1.0],
            },
        ]

        dataset = prepare_dataset(trajectories)

        # Verify dataset size
        assert len(dataset) == 5  # 3 + 2 examples

        # Verify tensor shapes
        spatial, vector, policy, value = dataset[0]
        assert spatial.shape == (13 * 8 * 8,)
        assert vector.shape == (64,)
        assert policy.shape == (376,)
        assert value.shape == ()

    def test_prepare_dataset_legacy_format(self):
        """Test prepare_dataset with legacy format (states instead of spatial)."""
        # Legacy format uses "states" key instead of "spatial"
        trajectories = [
            {
                "states": [[0.1] * (13 * 8 * 8) for _ in range(2)],
                "policies": [[1.0 / 376] * 376 for _ in range(2)],
                "rewards": [0.0, 1.0],
            },
        ]

        dataset = prepare_dataset(trajectories, vector_dim=64)

        # Verify dataset size
        assert len(dataset) == 2

        # Verify tensors
        spatial, vector, policy, value = dataset[0]
        assert spatial.shape == (13 * 8 * 8,)
        assert vector.shape == (64,)
        assert policy.shape == (376,)
        assert value.shape == ()

        # Vector should be zero-filled for legacy format
        assert torch.allclose(vector, torch.zeros(64))


class TestTrainEpochAllFeatures:
    """Integration test with all new features enabled."""

    def test_train_epoch_with_all_features(self, training_setup):
        """Test train_epoch with lr_scheduler, diagnostics, report, and profiler."""
        model, optimizer, dataloader, device = training_setup

        # Try to import all optional components
        try:
            from alpha_zero.training.diagnostics import TrainingDiagnostics
            from alpha_zero.training.training_report import TrainingReport, TrainingReportConfig
            from alpha_zero.utils.logging_config import PerformanceProfiler
        except ImportError:
            pytest.skip("Required modules not available")

        # Create all components
        diagnostics = TrainingDiagnostics(window_size=10)
        report = TrainingReport(TrainingReportConfig(report_interval=1))
        profiler = PerformanceProfiler()

        # Mock LR scheduler
        mock_scheduler = MagicMock()
        mock_scheduler.get_learning_rate.return_value = 0.0005

        # Run train_epoch with all features
        total_loss, _policy_loss, _value_loss, num_batches = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            global_step=0,
            wandb_logger=None,
            spatial_channels=13,
            spatial_size=8,
            lr_scheduler=mock_scheduler,
            diagnostics=diagnostics,
            report=report,
            profiler=profiler,
        )

        # Verify basic returns
        assert num_batches == 2
        assert total_loss > 0

        # Verify scheduler was called
        assert mock_scheduler.get_learning_rate.call_count == num_batches

        # Verify report generated (generate_report clears _losses, so check report_count)
        assert report.report_count > 0

        # Verify profiler recorded timings
        assert profiler.get_summary("forward_pass") is not None
        assert profiler.get_summary("backward_pass") is not None


class TestTrainEpochEdgeCases:
    """Test edge cases and error conditions."""

    def test_train_epoch_empty_dataloader(self, training_setup):
        """Test train_epoch with empty dataloader."""
        model, optimizer, _, device = training_setup

        # Create empty dataloader
        empty_dataset = TensorDataset(
            torch.empty(0, 13 * 8 * 8),
            torch.empty(0, 64),
            torch.empty(0, 376),
            torch.empty(0),
        )
        empty_dataloader = DataLoader(empty_dataset, batch_size=2)

        total_loss, policy_loss, value_loss, num_batches = train_epoch(
            model=model,
            dataloader=empty_dataloader,
            optimizer=optimizer,
            device=device,
            global_step=0,
            wandb_logger=None,
            spatial_channels=13,
            spatial_size=8,
        )

        # Should handle gracefully
        assert num_batches == 0
        assert total_loss == 0.0
        assert policy_loss == 0.0
        assert value_loss == 0.0

    def test_train_epoch_spatial_size_mismatch(self, training_setup):
        """Test train_epoch raises error on spatial dimension mismatch."""
        model, optimizer, dataloader, device = training_setup

        # Call with wrong spatial_size
        with pytest.raises(ValueError, match="Spatial tensor flat dim"):
            train_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                global_step=0,
                wandb_logger=None,
                spatial_channels=13,
                spatial_size=16,  # Wrong! Dataset has size 8
            )
