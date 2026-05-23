"""Orchestrator state persistence for preemption-resilient training.

Provides a GCS-backed state checkpoint so the orchestrator can resume
from exactly where it left off after preemption or crash.

Example::

    state = OrchestratorState.load(config) or OrchestratorState(...)
    for iteration in state.remaining_iterations():
        ...
        state.advance(new_model_version, win_rate, cost)
        state.save(config)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# GCS blob name for orchestrator state (relative to bucket root).
_STATE_BLOB_PREFIX = "orchestrator"
_STATE_BLOB_NAME = "state.json"


@dataclass
class IterationResult:
    """Result of a single training iteration.

    Attributes:
        iteration: The iteration number.
        model_version: Model version produced.
        win_rate: Evaluation win rate (0.0 if skipped).
        cost_usd: Estimated cost in USD.
        duration_seconds: Wall-clock time for the iteration.
        rejected: True if the model was rejected by evaluation gate.
    """

    iteration: int
    model_version: int
    win_rate: float = 0.0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    rejected: bool = False


@dataclass
class OrchestratorState:
    """Persistent orchestrator state for resumable training runs.

    Attributes:
        start_iteration: First iteration number in this run.
        end_iteration: Last iteration number (exclusive).
        current_iteration: Next iteration to execute.
        current_model_version: Latest accepted model version.
        initial_model_version: Model version at run start.
        successful_iterations: Count of successful iterations.
        failed_iterations: Count of failed iterations.
        consecutive_failures: Current streak of consecutive failures.
        total_cost_usd: Cumulative estimated cost.
        win_rate_history: Win rates from evaluation phases.
        results: Per-iteration results log.
        started_at: Unix timestamp when the run started.
        updated_at: Unix timestamp of last state update.
        agent_id: Optional agent ID for GCS namespace isolation.
        run_id: Unique run identifier for state file scoping.
    """

    start_iteration: int
    end_iteration: int
    current_iteration: int
    current_model_version: int
    initial_model_version: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    consecutive_failures: int = 0
    total_cost_usd: float = 0.0
    win_rate_history: list[float] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    agent_id: str | None = None
    run_id: str = ""

    # --- Iteration helpers ---

    def remaining_iterations(self) -> range:
        """Return range of iterations still to execute."""
        return range(self.current_iteration, self.end_iteration)

    @property
    def is_complete(self) -> bool:
        """True when all iterations have been processed."""
        return self.current_iteration >= self.end_iteration

    @property
    def total_iterations(self) -> int:
        """Total number of planned iterations."""
        return self.end_iteration - self.start_iteration

    # --- Mutation ---

    def record_success(
        self,
        iteration: int,
        model_version: int,
        win_rate: float = 0.0,
        cost_usd: float = 0.0,
        duration_seconds: float = 0.0,
        rejected: bool = False,
    ) -> None:
        """Record a successful iteration and advance state."""
        result = IterationResult(
            iteration=iteration,
            model_version=model_version,
            win_rate=win_rate,
            cost_usd=cost_usd,
            duration_seconds=duration_seconds,
            rejected=rejected,
        )
        self.results.append(asdict(result))
        self.total_cost_usd += cost_usd

        if not rejected:
            self.current_model_version = model_version

        if win_rate > 0:
            self.win_rate_history.append(win_rate)

        self.successful_iterations += 1
        self.consecutive_failures = 0
        self.current_iteration = iteration + 1
        self.updated_at = time.time()

    def record_failure(self, iteration: int, cost_usd: float = 0.0) -> None:
        """Record a failed iteration."""
        self.failed_iterations += 1
        self.consecutive_failures += 1
        self.total_cost_usd += cost_usd
        self.current_iteration = iteration + 1
        self.updated_at = time.time()

    # --- GCS persistence ---

    def _blob_path(self) -> str:
        """Build the GCS blob path for this state."""
        parts = [_STATE_BLOB_PREFIX]
        if self.agent_id:
            parts.append(self.agent_id)
        if self.run_id:
            parts.append(self.run_id)
        parts.append(_STATE_BLOB_NAME)
        return "/".join(parts)

    def save(self, config: Any) -> str | None:
        """Persist state to GCS.

        Args:
            config: GCPConfig instance with bucket and project_id.

        Returns:
            GCS URI of saved state, or None on failure.
        """
        try:
            from alpha_zero.cloud.gcp.storage import GCSBackend

            backend = GCSBackend(config.project_id)
            blob_path = self._blob_path()
            data = json.dumps(asdict(self), indent=2).encode("utf-8")
            backend.upload_blob(config.bucket, blob_path, data)
            uri = f"gs://{config.bucket}/{blob_path}"
            logger.info("Saved orchestrator state to %s", uri)
            return uri
        except Exception as e:
            logger.warning("Failed to save orchestrator state: %s", e)
            return None

    @classmethod
    def load(
        cls,
        config: Any,
        agent_id: str | None = None,
        run_id: str = "",
    ) -> OrchestratorState | None:
        """Load state from GCS.

        Args:
            config: GCPConfig instance with bucket and project_id.
            agent_id: Optional agent ID for namespace scoping.
            run_id: Run ID to scope the state file.

        Returns:
            OrchestratorState instance, or None if no state exists.
        """
        try:
            from alpha_zero.cloud.gcp.storage import GCSBackend

            # Build blob path
            parts = [_STATE_BLOB_PREFIX]
            if agent_id:
                parts.append(agent_id)
            if run_id:
                parts.append(run_id)
            parts.append(_STATE_BLOB_NAME)
            blob_path = "/".join(parts)

            backend = GCSBackend(config.project_id)
            data = backend.download_blob(config.bucket, blob_path)
            state_dict = json.loads(data.decode("utf-8"))

            # Reconstruct dataclass, ignoring unknown fields for
            # backwards compatibility
            import dataclasses

            known_fields = {f.name for f in dataclasses.fields(cls)}
            filtered = {k: v for k, v in state_dict.items() if k in known_fields}
            state = cls(**filtered)
            logger.info(
                "Loaded orchestrator state: iter %d/%d, model v%d",
                state.current_iteration,
                state.end_iteration,
                state.current_model_version,
            )
            return state
        except Exception as e:
            logger.debug("No existing orchestrator state found: %s", e)
            return None

    def to_summary(self) -> dict[str, Any]:
        """Return a human-readable summary dict for logging."""
        elapsed = self.updated_at - self.started_at
        return {
            "iterations_completed": f"{self.current_iteration - self.start_iteration}"
            f"/{self.total_iterations}",
            "successful": self.successful_iterations,
            "failed": self.failed_iterations,
            "model_version": self.current_model_version,
            "total_cost_usd": f"${self.total_cost_usd:.2f}",
            "elapsed_minutes": f"{elapsed / 60:.1f}",
        }
