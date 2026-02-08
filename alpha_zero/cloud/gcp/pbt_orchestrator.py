"""Population Based Training orchestrator for Vertex AI.

Manages the PBT lifecycle: initialize population → self-play → train →
evaluate → evolve → repeat.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.storage import GCSBackend, GCSModelStorage
from alpha_zero.cloud.gcp.vertex_jobs import (
    VertexAIClient,
    VertexEvaluationJob,
    VertexSelfPlayJob,
    VertexTrainingJob,
)
from alpha_zero.pbt.config import AgentState, PBTConfig
from alpha_zero.pbt.population import PopulationManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_VERSION = 1
_JOB_POLL_INTERVAL_SECONDS = 30
_TERMINAL_JOB_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
})


class PBTOrchestrator:
    """Orchestrates Population Based Training on Vertex AI."""

    def __init__(
        self,
        pbt_config: PBTConfig,
        gcp_config: GCPConfig | None = None,
    ) -> None:
        self.pbt_config = pbt_config
        self.gcp_config = gcp_config or GCPConfig()

        # Vertex AI client
        self.client = VertexAIClient(
            project=self.gcp_config.project_id,
            region=self.gcp_config.region,
            service_account=self.gcp_config.service_account,
        )

        self.manager = PopulationManager(self.pbt_config)

        # Job managers
        self.training_job_manager = VertexTrainingJob(self.gcp_config, self.client)
        self.selfplay_job_manager = VertexSelfPlayJob(self.gcp_config, self.client)
        self.eval_job_manager = VertexEvaluationJob(self.gcp_config, self.client)

        # State storage
        self.storage_backend = GCSBackend(self.gcp_config.project_id)
        self.population_state_path = (
            f"{self.gcp_config.paths.logs_prefix}/pbt_state.json"
        )

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def load_state(self) -> None:
        """Load population state from GCS."""
        try:
            if self.storage_backend.blob_exists(
                self.gcp_config.bucket, self.population_state_path
            ):
                data = self.storage_backend.download_blob(
                    self.gcp_config.bucket, self.population_state_path
                )
                state_dict = json.loads(data.decode())

                self.manager.generation = state_dict.get("generation", 0)
                agents_data: list[dict[str, Any]] = state_dict.get("agents", [])
                existing_agents = [AgentState.from_dict(ad) for ad in agents_data]
                self.manager.initialize(existing_agents=existing_agents)

                logger.info(
                    "Loaded PBT state from GCS: gen=%d, agents=%d",
                    self.manager.generation,
                    len(self.manager.agents),
                )
            else:
                logger.info("No PBT state found in GCS, starting fresh.")
        except Exception:
            logger.exception("Failed to load PBT state")

    def save_state(self) -> None:
        """Save population state to GCS."""
        try:
            state_dict = {
                "generation": self.manager.generation,
                "agents": [a.to_dict() for a in self.manager.get_all_agents()],
            }
            data = json.dumps(state_dict, indent=2).encode()
            self.storage_backend.upload_blob(
                self.gcp_config.bucket, self.population_state_path, data
            )
            logger.info("Saved PBT state to GCS.")
        except Exception:
            logger.exception("Failed to save PBT state")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the PBT training loop."""
        self.load_state()

        logger.info("Starting PBT loop. Config: %s", self.pbt_config)

        # Initialize if empty
        if not self.manager.agents:
            logger.info("Initializing new population...")
            self.manager.initialize()
            for agent in self.manager.get_all_agents():
                agent.model_path = str(_DEFAULT_MODEL_VERSION)
            self.save_state()

        start_gen = self.manager.generation
        for gen in range(start_gen, self.pbt_config.num_generations):
            logger.info("=== Starting Generation %d ===", gen + 1)

            self._run_self_play_phase(gen + 1)
            self._run_training_phase(gen + 1)
            self._run_evaluation_phase(gen + 1)

            self.manager.exploit_and_explore()
            self.save_state()

    # ------------------------------------------------------------------
    # Phase 1: Self-play
    # ------------------------------------------------------------------

    def _run_self_play_phase(self, gen: int) -> None:
        """Launch and wait for self-play jobs."""
        logger.info("Phase 1: Self-Play")
        job_names: list[str] = []

        for agent in self.manager.get_all_agents():
            model_version = _safe_int(agent.model_path, _DEFAULT_MODEL_VERSION)

            extra_args = [
                f"--c_puct={agent.hyperparameters.get('c_puct', 1.5)}",
                f"--agent_id={agent.agent_id}",
            ]

            logger.info(
                "Launching self-play for agent %s (v%d)...",
                agent.agent_id,
                model_version,
            )
            jobs = self.selfplay_job_manager.launch(
                num_workers=1,
                model_version=model_version,
                iteration=gen,
                extra_args=extra_args,
            )
            job_names.extend(jobs)

        if job_names:
            self.selfplay_job_manager.wait_for_completion(job_names)
        else:
            logger.warning("No self-play jobs launched.")

    # ------------------------------------------------------------------
    # Phase 2: Training
    # ------------------------------------------------------------------

    def _run_training_phase(self, gen: int) -> None:
        """Launch and wait for training jobs."""
        logger.info("Phase 2: Training")
        job_map: Dict[str, AgentState] = {}

        for agent in self.manager.get_all_agents():
            extra_args = [
                f"--learning_rate={agent.hyperparameters.get('learning_rate', 0.01)}",
                f"--agent_id={agent.agent_id}",
            ]

            logger.info("Launching training for agent %s...", agent.agent_id)
            job_name = self.training_job_manager.launch(
                iteration=gen,
                trajectory_iterations=[gen],
                resume_from_checkpoint="latest",
                extra_args=extra_args,
            )
            job_map[job_name] = agent

        self._poll_jobs(job_map)

    def _poll_jobs(self, job_map: Dict[str, AgentState]) -> None:
        """Poll until all jobs in *job_map* reach a terminal state."""
        active_jobs = set(job_map.keys())

        while active_jobs:
            for job_name in list(active_jobs):
                state, _ = self.client.get_job_state(job_name)
                if state not in _TERMINAL_JOB_STATES:
                    continue

                active_jobs.discard(job_name)
                agent = job_map[job_name]

                if state == "JOB_STATE_SUCCEEDED":
                    logger.info("Training succeeded for agent %s", agent.agent_id)
                    self._update_agent_model_version(agent)
                else:
                    logger.error(
                        "Training failed for agent %s (state: %s)",
                        agent.agent_id,
                        state,
                    )

            if active_jobs:
                time.sleep(_JOB_POLL_INTERVAL_SECONDS)

    def _update_agent_model_version(self, agent: AgentState) -> None:
        """Query GCS for the latest model version and update the agent."""
        try:
            agent_config = GCPConfig()
            agent_config.paths.agent_id = agent.agent_id
            storage = GCSModelStorage(agent_config)
            new_version = storage.get_latest_version()
            agent.model_path = str(new_version)
            logger.info(
                "Updated agent %s to model v%d", agent.agent_id, new_version
            )
        except Exception:
            logger.exception(
                "Failed to update version for agent %s", agent.agent_id
            )

    # ------------------------------------------------------------------
    # Phase 3: Evaluation
    # ------------------------------------------------------------------

    def _run_evaluation_phase(self, gen: int) -> None:
        """Assign fitness scores to all agents.

        TODO: Implement real cross-agent evaluation via VertexEvaluationJob
        once the evaluation worker supports ``--agent_id_new`` /
        ``--agent_id_old`` for cross-prefix model loading.

        Current MVP uses a constant fitness of 0.5 so that evolution mechanics
        can be validated end-to-end without cross-agent storage wiring.
        """
        logger.info("Phase 3: Evaluation")

        for agent in self.manager.get_all_agents():
            # Placeholder: neutral fitness until real eval is wired up.
            agent.fitness_score = 0.5

        logger.warning(
            "Evaluation phase using placeholder fitness (gen=%d). "
            "TODO: implement cross-agent evaluation.",
            gen,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_int(value: str | None, default: int) -> int:
    """Parse *value* as ``int``, returning *default* on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
