"""
Orchestrator State Persistence for Reboot-Resilient Training.

Provides GCS-backed state checkpointing and heartbeat monitoring so
the vertex_orchestrator can resume from its last completed iteration
after a VM reboot, crash, or preemption.

Usage::

    from alpha_zero.cloud.gcp.orchestrator_state import (
        HeartbeatThread,
        OrchestratorState,
        OrchestratorStateManager,
    )

    state_mgr = OrchestratorStateManager(config)
    state = state_mgr.load_state()  # Returns None if no prior state
    if state is None:
        state = OrchestratorState(...)

    # After each iteration:
    state.current_iteration = i + 1
    state_mgr.save_state(state)

Sprint 28 Phase 1: GCP orchestrator migration for reboot resilience.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from alpha_zero.cloud.gcp.exceptions import GCSStorageError
from alpha_zero.utils.compat import UTC

if TYPE_CHECKING:
    from alpha_zero.cloud.gcp.config import GCPConfig
    from alpha_zero.cloud.gcp.storage import StorageBackend

logger = logging.getLogger(__name__)

# GCS blob paths for orchestrator state and heartbeat.
_STATE_BLOB = "orchestrator/state.json"
_HEARTBEAT_BLOB = "orchestrator/heartbeat.json"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class OrchestratorState:
    """Persistent state for orchestrator resume capability.

    Serialized to GCS as JSON after each completed iteration so the
    orchestrator can resume from the exact point of interruption.

    Attributes:
        current_iteration: Next iteration index to execute (0-based).
        start_iteration: The CLI ``--start_iteration`` value.
        total_iterations: Total iterations requested (CLI ``--iterations``).
        model_version: Current best model version in GCS.
        successful_iterations: Count of iterations that succeeded.
        failed_iterations: Count of iterations that failed.
        consecutive_failures: Current consecutive failure streak.
        win_rate_history: Win rates from evaluation phases.
        total_estimated_cost: Accumulated estimated cost (USD).
        initial_model_version: Model version at run start (for tournament).
        start_time_utc: ISO-8601 timestamp of the first run start.
        last_checkpoint_utc: ISO-8601 timestamp of last state save.
        cli_args_hash: Hash of CLI args to detect config changes across resumes.
    """

    current_iteration: int = 0
    start_iteration: int = 0
    total_iterations: int = 0
    model_version: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    consecutive_failures: int = 0
    win_rate_history: list[float] = field(default_factory=list)
    total_estimated_cost: float = 0.0
    initial_model_version: int = 0
    start_time_utc: str = ""
    last_checkpoint_utc: str = ""
    cli_args_hash: str = ""

    def to_json(self) -> str:
        """Serialize to JSON string.

        Returns:
            Pretty-printed JSON representation.
        """
        data = asdict(self)
        return json.dumps(data, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> OrchestratorState:
        """Deserialize from JSON string.

        Args:
            raw: JSON string (as produced by :meth:`to_json`).

        Returns:
            Populated ``OrchestratorState`` instance.

        Raises:
            ValueError: If JSON is malformed or missing required fields.
        """
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in orchestrator state: {exc}") from exc

        # Filter to known fields for forward-compatibility (ignore unknown keys)
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def mark_checkpoint(self) -> None:
        """Update ``last_checkpoint_utc`` to now."""
        self.last_checkpoint_utc = datetime.now(UTC).isoformat()


# =============================================================================
# State Manager
# =============================================================================


class OrchestratorStateManager:
    """Manages orchestrator state persistence in GCS.

    Reads/writes ``OrchestratorState`` JSON blobs and heartbeat files
    to a configured GCS bucket, using the project's ``StorageBackend``
    protocol for easy testing with ``LocalBackend``.

    Args:
        config: GCP configuration (provides bucket name and project ID).
        backend: Optional storage backend override (defaults to
            :func:`~alpha_zero.cloud.gcp.storage.get_backend`).
    """

    def __init__(
        self,
        config: GCPConfig,
        backend: StorageBackend | None = None,
    ) -> None:
        from alpha_zero.cloud.gcp.storage import get_backend

        self._bucket = config.bucket
        self._backend = backend or get_backend(config.project_id)
        self._agent_id = config.paths.agent_id

    @property
    def _state_blob(self) -> str:
        """Agent-scoped state blob path."""
        if self._agent_id:
            return f"orchestrator/{self._agent_id}/state.json"
        return _STATE_BLOB

    @property
    def _heartbeat_blob(self) -> str:
        """Agent-scoped heartbeat blob path."""
        if self._agent_id:
            return f"orchestrator/{self._agent_id}/heartbeat.json"
        return _HEARTBEAT_BLOB

    # -- State persistence ----------------------------------------------------

    def save_state(self, state: OrchestratorState) -> str:
        """Save orchestrator state to GCS.

        Args:
            state: Current orchestrator state.

        Returns:
            GCS URI of the saved state blob.

        Raises:
            GCSStorageError: If the upload fails.
        """
        state.mark_checkpoint()
        payload = state.to_json().encode("utf-8")
        uri = self._backend.upload_blob(self._bucket, self._state_blob, payload)
        logger.info(
            "Saved orchestrator state: iteration=%d, model=v%d → %s",
            state.current_iteration,
            state.model_version,
            uri,
        )
        return uri

    def load_state(self) -> OrchestratorState | None:
        """Load orchestrator state from GCS.

        Returns:
            Previously saved state, or ``None`` if no state file exists.

        Raises:
            ValueError: If the state file contains invalid JSON.
        """
        try:
            if not self._backend.blob_exists(self._bucket, self._state_blob):
                logger.info("No prior orchestrator state found in GCS")
                return None

            data = self._backend.download_blob(self._bucket, self._state_blob)
            state = OrchestratorState.from_json(data.decode("utf-8"))
            logger.info(
                "Loaded orchestrator state: iteration=%d, model=v%d, "
                "success=%d, failures=%d",
                state.current_iteration,
                state.model_version,
                state.successful_iterations,
                state.failed_iterations,
            )
            return state
        except GCSStorageError:
            logger.warning("Failed to load orchestrator state from GCS", exc_info=True)
            return None

    def clear_state(self) -> None:
        """Delete the orchestrator state blob (for fresh starts).

        Silently succeeds if no state exists.
        """
        try:
            self._backend.delete_blob(self._bucket, self._state_blob)
            logger.info("Cleared orchestrator state from GCS")
        except GCSStorageError:
            logger.debug("No state to clear (blob may not exist)")

    # -- Heartbeat ------------------------------------------------------------

    def write_heartbeat(
        self,
        *,
        status: str,
        iteration: int,
        model_version: int,
        uptime_seconds: float = 0.0,
    ) -> None:
        """Write a heartbeat JSON blob to GCS.

        Args:
            status: Current orchestrator status (e.g. "running", "selfplay").
            iteration: Current iteration number.
            model_version: Current model version.
            uptime_seconds: Seconds since orchestrator started.
        """
        heartbeat = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "iteration": iteration,
            "model_version": model_version,
            "uptime_seconds": round(uptime_seconds, 1),
        }
        payload = json.dumps(heartbeat, indent=2).encode("utf-8")
        try:
            self._backend.upload_blob(self._bucket, self._heartbeat_blob, payload)
            logger.debug("Heartbeat written: status=%s, iter=%d", status, iteration)
        except GCSStorageError:
            logger.warning("Failed to write heartbeat to GCS", exc_info=True)

    def read_heartbeat(self) -> dict[str, Any] | None:
        """Read the latest heartbeat from GCS.

        Returns:
            Heartbeat dict, or ``None`` if not found.
        """
        try:
            data = self._backend.download_blob(self._bucket, self._heartbeat_blob)
            return json.loads(data.decode("utf-8"))  # type: ignore[no-any-return]
        except (GCSStorageError, json.JSONDecodeError):
            return None


# =============================================================================
# Heartbeat Background Thread
# =============================================================================


class HeartbeatThread(threading.Thread):
    """Daemon thread that periodically writes heartbeat to GCS.

    Starts automatically when the orchestrator begins a training run.
    Stops when the main process exits (daemon thread) or when
    :meth:`stop` is called explicitly.

    Args:
        state_manager: State manager instance for GCS writes.
        interval_seconds: Seconds between heartbeat writes.
    """

    def __init__(
        self,
        state_manager: OrchestratorStateManager,
        interval_seconds: float = 60.0,
    ) -> None:
        super().__init__(daemon=True, name="heartbeat-thread")
        self._state_manager = state_manager
        self._interval = interval_seconds
        self._stop_event = threading.Event()

        # Mutable state accessed from the main thread via setters
        self._lock = threading.Lock()
        self._status = "starting"
        self._iteration = 0
        self._model_version = 0
        self._start_time = time.monotonic()

    # -- Public API (called from main thread) ---------------------------------

    def update(
        self,
        *,
        status: str | None = None,
        iteration: int | None = None,
        model_version: int | None = None,
    ) -> None:
        """Update heartbeat metadata (thread-safe).

        Args:
            status: New status string.
            iteration: New iteration number.
            model_version: New model version.
        """
        with self._lock:
            if status is not None:
                self._status = status
            if iteration is not None:
                self._iteration = iteration
            if model_version is not None:
                self._model_version = model_version

    def stop(self) -> None:
        """Signal the heartbeat thread to stop."""
        self._stop_event.set()

    # -- Thread body ----------------------------------------------------------

    def run(self) -> None:
        """Periodically write heartbeat to GCS until stopped."""
        logger.info(
            "Heartbeat thread started (interval=%.0fs)", self._interval
        )
        while not self._stop_event.is_set():
            with self._lock:
                status = self._status
                iteration = self._iteration
                model_version = self._model_version

            uptime = time.monotonic() - self._start_time
            try:
                self._state_manager.write_heartbeat(
                    status=status,
                    iteration=iteration,
                    model_version=model_version,
                    uptime_seconds=uptime,
                )
            except Exception:
                logger.warning("Heartbeat write failed", exc_info=True)

            # Sleep in short increments so stop() is responsive
            for _ in range(int(self._interval)):
                if self._stop_event.is_set():
                    break
                time.sleep(1.0)

        # Final heartbeat with "stopped" status
        try:
            self._state_manager.write_heartbeat(
                status="stopped",
                iteration=self._iteration,
                model_version=self._model_version,
                uptime_seconds=time.monotonic() - self._start_time,
            )
        except Exception:
            logger.debug("Final heartbeat write failed", exc_info=True)

        logger.info("Heartbeat thread stopped")
