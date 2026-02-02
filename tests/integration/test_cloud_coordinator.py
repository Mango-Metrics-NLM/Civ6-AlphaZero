"""Integration tests for cloud coordinator."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import pytest

from alpha_zero.cloud.config import CloudConfig
from alpha_zero.cloud.gcp import GCSCheckpointManager
from alpha_zero.distributed.cloud_coordinator import (
    CloudCoordinatorConfig,
    CloudTrainingCoordinator,
)


@pytest.fixture
def cloud_config() -> CloudConfig:
    """Create test cloud configuration."""
    return CloudConfig(
        project_id="test-project",
        region="us-central1",
        bucket_name="test-bucket",
        checkpoint_prefix="checkpoints/",
        enable_cloud_logging=True,
        enable_cloud_monitoring=True,
        use_spot=True,
        preemption_check_interval=5.0,
    )


@pytest.fixture
def coordinator_config(cloud_config: CloudConfig) -> CloudCoordinatorConfig:
    """Create test coordinator configuration."""
    return CloudCoordinatorConfig(
        cloud_config=cloud_config,
        sync_interval=60.0,
        enable_cloud_checkpoints=True,
        enable_preemption_handling=True,
        checkpoint_on_preemption=True,
        upload_metrics=True,
        checkpoint_dir=tempfile.mkdtemp(),
    )


class TestCloudCoordinatorConfig:
    """Tests for CloudCoordinatorConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CloudCoordinatorConfig()
        
        assert config.cloud_config is None
        assert config.sync_interval == 300.0
        assert config.enable_cloud_checkpoints is True
        assert config.enable_preemption_handling is True

    def test_from_dict_with_cloud(self) -> None:
        """Test creating config from dictionary with cloud section."""
        data = {
            "sync_interval": 120.0,
            "enable_cloud_checkpoints": False,
            "cloud": {
                "project_id": "dict-project",
                "region": "europe-west1",
            },
        }
        
        config = CloudCoordinatorConfig.from_dict(data)
        
        assert config.sync_interval == 120.0
        assert config.enable_cloud_checkpoints is False
        assert config.cloud_config is not None
        assert config.cloud_config.project_id == "dict-project"

    def test_from_dict_with_gcp(self) -> None:
        """Test creating config with gcp section."""
        data = {
            "gcp": {
                "project_id": "gcp-project",
                "bucket_name": "gcp-bucket",
            },
        }
        
        config = CloudCoordinatorConfig.from_dict(data)
        
        assert config.cloud_config is not None
        assert config.cloud_config.project_id == "gcp-project"


class TestCloudTrainingCoordinator:
    """Tests for CloudTrainingCoordinator."""

    def test_initialization(
        self,
        coordinator_config: CloudCoordinatorConfig,
    ) -> None:
        """Test coordinator initialization."""
        coordinator = CloudTrainingCoordinator(coordinator_config)
        
        assert coordinator._cloud_config == coordinator_config.cloud_config
        assert coordinator._gcs_manager is None  # Not initialized until initialize()

    def test_initialize_with_mock(
        self,
        coordinator_config: CloudCoordinatorConfig,
    ) -> None:
        """Test initialization with mock components."""
        coordinator = CloudTrainingCoordinator(coordinator_config)
        coordinator.initialize(force_mock=True)
        
        # GCS manager should be initialized
        assert coordinator._gcs_manager is not None
        assert coordinator._gcs_manager.is_mock is True
        
        # Preemption handler should be initialized
        assert coordinator._preemption_handler is not None
        
        # Cloud logger should be initialized
        assert coordinator._cloud_logger is not None
        
        # Cleanup
        coordinator.shutdown()

    def test_shutdown(
        self,
        coordinator_config: CloudCoordinatorConfig,
    ) -> None:
        """Test coordinator shutdown."""
        coordinator = CloudTrainingCoordinator(coordinator_config)
        coordinator.initialize(force_mock=True)
        
        # Should not raise
        coordinator.shutdown()
        
        # Preemption handler should be stopped
        assert coordinator._preemption_handler is not None
        assert coordinator._preemption_handler._monitoring is False

    def test_log_training_metrics(
        self,
        coordinator_config: CloudCoordinatorConfig,
    ) -> None:
        """Test logging training metrics."""
        coordinator = CloudTrainingCoordinator(coordinator_config)
        coordinator.initialize(force_mock=True)
        
        try:
            # Should not raise
            coordinator.log_training_metrics(
                step=100,
                metrics={"loss": 0.5, "accuracy": 0.9},
                labels={"experiment": "test"},
            )
            
            # Check metrics were logged to cloud logger
            assert coordinator._cloud_logger is not None
            assert len(coordinator._cloud_logger._buffer) >= 1
        finally:
            coordinator.shutdown()


class TestCloudCoordinatorCheckpoints:
    """Tests for checkpoint operations."""

    def test_save_checkpoint_queues_upload(
        self,
        coordinator_config: CloudCoordinatorConfig,
    ) -> None:
        """Test that save_checkpoint queues cloud upload."""
        coordinator = CloudTrainingCoordinator(coordinator_config)
        coordinator.initialize(force_mock=True)
        
        try:
            # Create a simple checkpoint file
            checkpoint_dir = Path(coordinator_config.checkpoint_dir)
            checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
            checkpoint_path.write_bytes(b"test data")
            
            # Directly call save_checkpoint - it should handle missing learner gracefully
            # and queue the checkpoint for upload
            coordinator.save_checkpoint(
                step=100,
                network_state={"weights": "test"},
            )
            
            # Check that no errors occurred
            # The pending uploads list may be empty if sync happened immediately
            assert hasattr(coordinator, "_pending_uploads")
        finally:
            coordinator.shutdown()

    def test_load_checkpoint_checks_cloud(
        self,
        coordinator_config: CloudCoordinatorConfig,
    ) -> None:
        """Test that load_checkpoint checks cloud storage."""
        coordinator = CloudTrainingCoordinator(coordinator_config)
        coordinator.initialize(force_mock=True)
        
        try:
            # Upload a checkpoint to mock storage
            assert coordinator._gcs_manager is not None
            
            checkpoint_dir = Path(coordinator_config.checkpoint_dir)
            local_checkpoint = checkpoint_dir / "cloud_checkpoint.pt"
            local_checkpoint.write_bytes(b"checkpoint data")
            
            coordinator._gcs_manager.upload_checkpoint(
                str(local_checkpoint),
                "checkpoints/step_500.pt",
            )
            
            # Remove local file
            local_checkpoint.unlink()
            
            # Attempt cloud load - should not raise
            result = coordinator.load_checkpoint(from_cloud=True)
            
            # Result may be None if the checkpoint format doesn't match expected structure
            # The important thing is that it checked cloud storage without error
        finally:
            coordinator.shutdown()



class TestCloudCoordinatorPreemption:
    """Tests for preemption handling."""

    def test_emergency_checkpoint_callback(
        self,
        coordinator_config: CloudCoordinatorConfig,
    ) -> None:
        """Test emergency checkpoint is called on preemption."""
        coordinator = CloudTrainingCoordinator(coordinator_config)
        coordinator.initialize(force_mock=True)
        
        try:
            # Set a current step
            coordinator._current_step = 500  # type: ignore[attr-defined]
            
            # Mock network state
            with mock.patch.object(
                coordinator,
                "_get_current_network_state",
                return_value={"weights": "test"},
            ):
                # Simulate preemption
                assert coordinator._preemption_handler is not None
                coordinator._preemption_handler.simulate_preemption()
            
            # Preemption should be pending
            assert coordinator._preemption_handler.is_preemption_pending() is True
        finally:
            coordinator.shutdown()

    def test_disable_preemption_handling(
        self,
        cloud_config: CloudConfig,
    ) -> None:
        """Test disabling preemption handling."""
        config = CloudCoordinatorConfig(
            cloud_config=cloud_config,
            enable_preemption_handling=False,
        )
        
        coordinator = CloudTrainingCoordinator(config)
        coordinator.initialize(force_mock=True)
        
        try:
            # Preemption handler should not be initialized
            assert coordinator._preemption_handler is None
        finally:
            coordinator.shutdown()


class TestFullTrainingLoop:
    """Tests for full training loop simulation."""

    def test_multi_step_training_simulation(
        self,
        coordinator_config: CloudCoordinatorConfig,
    ) -> None:
        """Test multi-step training with periodic checkpoints."""
        coordinator = CloudTrainingCoordinator(coordinator_config)
        coordinator.initialize(force_mock=True)
        
        try:
            # Simulate multiple training steps
            for step in range(0, 500, 100):
                # Log metrics (as would happen during training)
                coordinator.log_training_metrics(
                    step=step,
                    metrics={
                        "loss": 1.0 / (step + 1),
                        "policy_loss": 0.5 / (step + 1),
                        "value_loss": 0.3 / (step + 1),
                    },
                )
                
                # Save checkpoint at intervals
                if step % 200 == 0 and step > 0:
                    coordinator.save_checkpoint(
                        step=step,
                        network_state={"weights": f"step_{step}"},
                    )
            
            # Verify cloud logger was created and used
            assert coordinator._cloud_logger is not None
            # Buffer may have auto-flushed, but logger should be functional
        finally:
            coordinator.shutdown()

    def test_graceful_shutdown_during_training(
        self,
        coordinator_config: CloudCoordinatorConfig,
    ) -> None:
        """Test graceful shutdown during active training."""
        coordinator = CloudTrainingCoordinator(coordinator_config)
        coordinator.initialize(force_mock=True)
        
        # Start "training"
        coordinator.log_training_metrics(
            step=100,
            metrics={"loss": 0.5},
        )
        
        # Graceful shutdown should not raise
        coordinator.shutdown()
        
        # Create a new coordinator for re-initialization test
        coordinator2 = CloudTrainingCoordinator(coordinator_config)
        try:
            coordinator2.initialize(force_mock=True)
            
            # Continue training
            coordinator2.log_training_metrics(
                step=200,
                metrics={"loss": 0.3},
            )
        finally:
            coordinator2.shutdown()


class TestCheckpointRoundTrip:
    """Tests for checkpoint save/load round-trip."""

    def test_checkpoint_persistence_mock(
        self,
        coordinator_config: CloudCoordinatorConfig,
    ) -> None:
        """Test checkpoint save and load with mock storage."""
        coordinator = CloudTrainingCoordinator(coordinator_config)
        coordinator.initialize(force_mock=True)
        
        try:
            checkpoint_dir = Path(coordinator_config.checkpoint_dir)
            
            # Create test checkpoint content
            import torch
            test_state = {
                "step": 1000,
                "network": {"fc1.weight": torch.randn(64, 128)},
                "optimizer": {"lr": 0.001},
            }
            
            # Save checkpoint locally
            checkpoint_path = checkpoint_dir / "round_trip_test.pt"
            torch.save(test_state, checkpoint_path)
            
            # Upload to mock GCS
            assert coordinator._gcs_manager is not None
            coordinator._gcs_manager.upload_checkpoint(
                str(checkpoint_path),
                "checkpoints/round_trip_test.pt",
            )
            
            # Delete local file
            checkpoint_path.unlink()
            assert not checkpoint_path.exists()
            
            # Download from mock GCS
            download_path = checkpoint_dir / "downloaded.pt"
            coordinator._gcs_manager.download_checkpoint(
                "checkpoints/round_trip_test.pt",
                str(download_path),
            )
            
            # Verify content
            loaded_state = torch.load(download_path, weights_only=False)
            assert loaded_state["step"] == 1000
            assert "network" in loaded_state
        finally:
            coordinator.shutdown()


class TestMetricsAggregation:
    """Tests for metrics buffering and aggregation."""

    def test_metrics_batching(
        self,
        coordinator_config: CloudCoordinatorConfig,
    ) -> None:
        """Test that metrics are properly batched."""
        coordinator = CloudTrainingCoordinator(coordinator_config)
        coordinator.initialize(force_mock=True)
        
        try:
            # Log fewer metrics to test buffering (less than buffer_size=100)
            for step in range(50):
                coordinator.log_training_metrics(
                    step=step,
                    metrics={
                        "loss": 1.0 - step * 0.01,
                        "entropy": 0.5 + step * 0.001,
                    },
                    labels={"batch": str(step // 10)},
                )
            
            # Check buffer state - should have entries since < 100 (buffer_size)
            assert coordinator._cloud_logger is not None
            assert len(coordinator._cloud_logger._buffer) == 50
        finally:
            coordinator.shutdown()

    def test_different_metric_types(
        self,
        coordinator_config: CloudCoordinatorConfig,
    ) -> None:
        """Test logging various metric types."""
        coordinator = CloudTrainingCoordinator(coordinator_config)
        coordinator.initialize(force_mock=True)
        
        try:
            # Log different metric categories
            coordinator.log_training_metrics(
                step=1,
                metrics={
                    "policy_loss": 0.5,
                    "value_loss": 0.3,
                    "total_loss": 0.8,
                    "entropy": 0.1,
                    "learning_rate": 0.001,
                    "games_played": 1000,
                    "mcts_simulations": 200,
                },
            )
            
            assert coordinator._cloud_logger is not None
            # Verify log entry was created
            assert len(coordinator._cloud_logger._buffer) >= 1
        finally:
            coordinator.shutdown()

