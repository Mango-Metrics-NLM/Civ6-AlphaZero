"""Cloud-aware training coordinator.

Extends TrainingCoordinator with cloud integration:
- GCS checkpoint storage
- Preemption handling
- Cloud logging and monitoring
- Graceful shutdown for spot instances
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from alpha_zero.cloud.config import CloudConfig
from alpha_zero.cloud.exceptions import CloudStorageError, PreemptionError
from alpha_zero.cloud.gcp import (
    GCS_AVAILABLE,
    GCSCheckpointManager,
    PreemptionHandler,
    CloudLogger,
    CloudMetricsExporter,
)
from alpha_zero.distributed.coordinator import CoordinatorConfig, TrainingCoordinator

logger = logging.getLogger(__name__)


@dataclass
class CloudCoordinatorConfig(CoordinatorConfig):
    """Configuration for cloud-aware training coordinator.

    Extends CoordinatorConfig with cloud-specific settings.

    Attributes:
        cloud_config: Cloud infrastructure configuration.
        sync_interval: Interval for syncing checkpoints to cloud (seconds).
        enable_cloud_checkpoints: Whether to sync checkpoints to GCS.
        enable_preemption_handling: Whether to handle spot instance preemption.
        checkpoint_on_preemption: Whether to save checkpoint on preemption.
        upload_metrics: Whether to upload metrics to Cloud Monitoring.
    """

    cloud_config: CloudConfig | None = None
    sync_interval: float = 300.0  # 5 minutes
    enable_cloud_checkpoints: bool = True
    enable_preemption_handling: bool = True
    checkpoint_on_preemption: bool = True
    upload_metrics: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CloudCoordinatorConfig:
        """Create config from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            CloudCoordinatorConfig instance.
        """
        # Extract cloud config if present
        cloud_data = data.pop("cloud", None) or data.pop("gcp", None)
        cloud_config = None
        if cloud_data:
            cloud_config = CloudConfig(**cloud_data)
        elif "cloud_config" in data and isinstance(data["cloud_config"], dict):
            cloud_config = CloudConfig(**data.pop("cloud_config"))

        # Get cloud coordinator specific fields
        sync_interval = data.pop("sync_interval", 300.0)
        enable_cloud_checkpoints = data.pop("enable_cloud_checkpoints", True)
        enable_preemption_handling = data.pop("enable_preemption_handling", True)
        checkpoint_on_preemption = data.pop("checkpoint_on_preemption", True)
        upload_metrics = data.pop("upload_metrics", True)

        return cls(
            cloud_config=cloud_config,
            sync_interval=sync_interval,
            enable_cloud_checkpoints=enable_cloud_checkpoints,
            enable_preemption_handling=enable_preemption_handling,
            checkpoint_on_preemption=checkpoint_on_preemption,
            upload_metrics=upload_metrics,
            **data,
        )


class CloudTrainingCoordinator(TrainingCoordinator):
    """Cloud-aware distributed training coordinator.

    Extends TrainingCoordinator with cloud integration for:
    - Automatic checkpoint upload to GCS
    - Preemption detection and handling
    - Cloud logging and monitoring
    - Resume from cloud checkpoints

    Example:
        >>> cloud_config = CloudConfig.from_env()
        >>> config = CloudCoordinatorConfig(
        ...     cloud_config=cloud_config,
        ...     checkpoint_dir="./checkpoints",
        ... )
        >>> coordinator = CloudTrainingCoordinator(config)
        >>> coordinator.run()
    """

    def __init__(self, config: CloudCoordinatorConfig) -> None:
        """Initialize cloud training coordinator.

        Args:
            config: Cloud coordinator configuration.
        """
        # Call parent constructor
        super().__init__(config)
        self._cloud_config = config.cloud_config
        self._cloud_coordinator_config = config

        # Initialize cloud components (lazy to allow testing)
        self._gcs_manager: GCSCheckpointManager | None = None
        self._preemption_handler: PreemptionHandler | None = None
        self._cloud_logger: CloudLogger | None = None
        self._metrics_exporter: CloudMetricsExporter | None = None

        # Sync state
        self._last_cloud_sync: float = 0.0
        self._pending_uploads: list[str] = []
        self._upload_lock = threading.Lock()
        self._upload_thread: threading.Thread | None = None

        self._force_mock = False

    def initialize(self, force_mock: bool = False) -> None:
        """Initialize coordinator with cloud components.

        Args:
            force_mock: Force mock implementations for testing.
        """
        self._force_mock = force_mock

        if self._cloud_config:
            # Initialize GCS checkpoint manager
            if self._cloud_coordinator_config.enable_cloud_checkpoints:
                try:
                    self._gcs_manager = GCSCheckpointManager(
                        self._cloud_config,
                        force_mock=force_mock,
                    )
                    logger.info("GCS checkpoint manager initialized")
                except Exception as e:
                    logger.warning("Failed to initialize GCS manager: %s", e)

            # Initialize preemption handler
            if self._cloud_coordinator_config.enable_preemption_handling:
                self._preemption_handler = PreemptionHandler(
                    checkpoint_callback=self._emergency_checkpoint,
                    cleanup_callbacks=[self._cleanup],
                    check_interval=self._cloud_config.preemption_check_interval,
                    force_mock=force_mock,
                )
                self._preemption_handler.start_monitoring()
                logger.info("Preemption handler initialized")

            # Initialize cloud logging
            if self._cloud_config.enable_cloud_logging:
                self._cloud_logger = CloudLogger(
                    self._cloud_config,
                    force_mock=force_mock,
                )
                logger.info("Cloud logger initialized")

            # Initialize metrics exporter
            if (
                self._cloud_coordinator_config.upload_metrics
                and self._cloud_config.enable_cloud_monitoring
            ):
                self._metrics_exporter = CloudMetricsExporter(
                    self._cloud_config,
                    force_mock=force_mock,
                )
                logger.info("Metrics exporter initialized")

        logger.info("CloudTrainingCoordinator initialized")

    def shutdown(self) -> None:
        """Shutdown coordinator with cloud cleanup."""
        logger.info("Shutting down CloudTrainingCoordinator")

        # Stop preemption monitoring
        if self._preemption_handler:
            self._preemption_handler.stop_monitoring()

        # Flush cloud logger
        if self._cloud_logger:
            self._cloud_logger.flush()

        # Upload any pending checkpoints
        self._sync_pending_uploads()

        # Call parent shutdown
        super().shutdown()

        logger.info("CloudTrainingCoordinator shutdown complete")

    def save_checkpoint(
        self,
        step: int,
        network_state: dict[str, Any],
        optimizer_state: dict[str, Any] | None = None,
        additional_state: dict[str, Any] | None = None,
    ) -> str:
        """Save checkpoint locally and optionally sync to cloud.

        Args:
            step: Training step number.
            network_state: Network state dict.
            optimizer_state: Optional optimizer state.
            additional_state: Optional additional state.

        Returns:
            Local path to checkpoint.
        """
        # Save locally first
        local_path = super().save_checkpoint(
            step, network_state, optimizer_state, additional_state
        )

        # Queue for cloud upload
        if self._gcs_manager and self._cloud_coordinator_config.enable_cloud_checkpoints:
            with self._upload_lock:
                self._pending_uploads.append(local_path)

            # Check if it's time for a sync
            if time.time() - self._last_cloud_sync > self._cloud_coordinator_config.sync_interval:
                self._sync_pending_uploads()

        # Log metrics if available
        if self._cloud_logger and additional_state:
            metrics = additional_state.get("metrics", {})
            if metrics:
                self._cloud_logger.log_metrics(step, metrics)

        return local_path

    def load_checkpoint(
        self,
        path: str | None = None,
        from_cloud: bool = True,
    ) -> dict[str, Any] | None:
        """Load checkpoint from local or cloud storage.

        Args:
            path: Specific checkpoint path/key to load.
            from_cloud: Whether to check cloud storage for checkpoints.

        Returns:
            Checkpoint state dict, or None if not found.
        """
        # Try local first
        local_checkpoint = super().load_checkpoint(path)
        if local_checkpoint:
            return local_checkpoint

        # Try cloud if enabled
        if from_cloud and self._gcs_manager:
            try:
                if path:
                    remote_key = path
                else:
                    remote_key = self._gcs_manager.get_latest_checkpoint()

                if remote_key:
                    local_path = Path(self.config.checkpoint_dir) / Path(remote_key).name
                    downloaded = self._gcs_manager.download_checkpoint(
                        remote_key, local_path
                    )
                    if downloaded.exists():
                        return super().load_checkpoint(str(downloaded))

            except CloudStorageError as e:
                logger.warning("Failed to load checkpoint from cloud: %s", e)

        return None

    def log_training_metrics(
        self,
        step: int,
        metrics: dict[str, float],
        labels: dict[str, str] | None = None,
    ) -> None:
        """Log training metrics to cloud and local loggers.

        Args:
            step: Training step number.
            metrics: Metric name to value mapping.
            labels: Optional labels for filtering.
        """
        # Log to cloud logger
        if self._cloud_logger:
            self._cloud_logger.log_metrics(step, metrics, labels)

        # Export to Cloud Monitoring
        if self._metrics_exporter:
            self._metrics_exporter.export_training_metrics(step, metrics)

        # Also log locally
        logger.info("Step %d metrics: %s", step, metrics)

    def _emergency_checkpoint(self) -> None:
        """Save emergency checkpoint during preemption.

        Called by preemption handler when spot instance is about to be terminated.
        """
        if not self._cloud_coordinator_config.checkpoint_on_preemption:
            return

        logger.warning("Saving emergency checkpoint due to preemption")

        try:
            # Get current training state
            step = self._current_step if hasattr(self, "_current_step") else 0
            network_state = self._get_current_network_state()

            if network_state:
                checkpoint_path = self.save_checkpoint(
                    step,
                    network_state,
                    additional_state={"preemption": True},
                )

                # Force immediate upload
                if self._gcs_manager:
                    remote_key = f"{self._cloud_config.checkpoint_prefix}emergency_step_{step}.pt"  # type: ignore[union-attr]
                    self._gcs_manager.upload_checkpoint(
                        checkpoint_path,
                        remote_key,
                        metadata={"preemption": "true", "step": str(step)},
                    )
                    logger.info("Emergency checkpoint uploaded: %s", remote_key)

        except Exception as e:
            logger.error("Failed to save emergency checkpoint: %s", e)
            raise PreemptionError(
                "Emergency checkpoint failed",
                checkpoint_saved=False,
                cause=e,
            ) from e

    def _get_current_network_state(self) -> dict[str, Any] | None:
        """Get current network state for checkpoint.

        Override in subclass to provide actual network state.

        Returns:
            Network state dict or None.
        """
        # Placeholder - actual implementation would get from shared storage
        return None

    def _sync_pending_uploads(self) -> None:
        """Sync pending checkpoints to cloud storage."""
        if not self._gcs_manager:
            return

        with self._upload_lock:
            uploads = list(self._pending_uploads)
            self._pending_uploads.clear()

        if not uploads:
            return

        for local_path in uploads:
            try:
                path = Path(local_path)
                if path.exists():
                    remote_key = f"{self._cloud_config.checkpoint_prefix}{path.name}"  # type: ignore[union-attr]
                    self._gcs_manager.upload_checkpoint(local_path, remote_key)
                    logger.debug("Uploaded checkpoint: %s", remote_key)
            except Exception as e:
                logger.warning("Failed to upload checkpoint %s: %s", local_path, e)

        self._last_cloud_sync = time.time()

    def _cleanup(self) -> None:
        """Cleanup callback for preemption."""
        logger.info("Running preemption cleanup")

        # Flush logs
        if self._cloud_logger:
            self._cloud_logger.flush()

        # Log preemption event
        if self._cloud_logger:
            self._cloud_logger.log_event(
                "WARNING",
                "Training preempted",
                payload={"step": getattr(self, "_current_step", 0)},
            )
