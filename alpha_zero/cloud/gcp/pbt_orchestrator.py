"""Population Based Training orchestrator for Vertex AI.

Manages the PBT lifecycle: initialize population -> self-play -> train ->
evaluate -> evolve -> repeat.  The evaluation phase uses :class:`PBTEvaluator`
to run cross-agent arena matches and compute ELO-based fitness.
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
from alpha_zero.pbt.evaluator import EvaluationReport, MatchResult, PBTEvaluator
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
    """Orchestrates Population-Based Training on Vertex AI.

    Coordinates the self-play, training, evaluation, and evolution
    phases of PBT.  The evaluation phase uses :class:`PBTEvaluator` to
    run cross-agent arena matches and produce ELO-derived fitness scores.

    Args:
        pbt_config: PBT hyperparameters and evaluation settings.
        gcp_config: GCP project / bucket / region configuration.
        evaluator: Optional pre-built evaluator -- if *None*, one is
            created automatically.
    """

    def __init__(
        self,
        pbt_config: PBTConfig,
        gcp_config: GCPConfig | None = None,
        evaluator: PBTEvaluator | None = None,
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
        self.evaluator = evaluator or PBTEvaluator(self.pbt_config)

        # Job managers
        self.training_job_manager = VertexTrainingJob(self.gcp_config, self.client)
        self.selfplay_job_manager = VertexSelfPlayJob(self.gcp_config, self.client)
        self.eval_job_manager = VertexEvaluationJob(self.gcp_config, self.client)

        # State storage
        self.storage_backend = GCSBackend(self.gcp_config.project_id)
        self.population_state_path = (
            f"{self.gcp_config.paths.logs_prefix}/pbt_state.json"
        )

        logger.info(
            "PBTOrchestrator initialised: pop=%d, strategy=%s",
            pbt_config.population_size,
            pbt_config.evaluation_strategy,
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

    def run(self, num_generations: int | None = None) -> None:
        """Run the complete PBT loop.

        Args:
            num_generations: Number of generations to run.  Falls back
                to ``self.pbt_config.num_generations`` when *None*.
        """
        self.load_state()

        total = num_generations or getattr(self.pbt_config, "num_generations", 10)
        logger.info("Starting PBT loop for %d generations", total)

        # Initialize if empty
        if not self.manager.agents:
            logger.info("Initializing new population...")
            self.manager.initialize()
            for agent in self.manager.get_all_agents():
                agent.model_path = str(_DEFAULT_MODEL_VERSION)
            self.save_state()

        start_gen = self.manager.generation
        for gen in range(start_gen, start_gen + total):
            gen_num = gen + 1
            logger.info("=" * 60)
            logger.info("Generation %d / %d", gen_num, start_gen + total)
            logger.info("=" * 60)

            t0 = time.monotonic()

            self._run_selfplay_phase(gen_num)
            self._run_training_phase(gen_num)
            report = self._run_evaluation_phase(gen_num)
            self._run_evolution_phase(gen_num)
            self.save_state()

            elapsed = time.monotonic() - t0
            logger.info(
                "Generation %d complete in %.1fs | "
                "matchups=%d | top_elo=%.0f",
                gen_num,
                elapsed,
                len(report.matchups),
                max(report.elo_ratings.values())
                if report.elo_ratings
                else 0,
            )

    # ------------------------------------------------------------------
    # Phase 1: Self-play
    # ------------------------------------------------------------------

    def _run_selfplay_phase(self, gen: int) -> None:
        """Launch self-play jobs for each agent in the population."""
        logger.info("Phase 1: Self-play (gen=%d)", gen)
        job_names: list[str] = []

        for agent in self.manager.get_all_agents():
            version = _safe_int(agent.model_path, _DEFAULT_MODEL_VERSION)

            extra_args = [
                f"--c_puct={agent.hyperparameters.get('c_puct', 1.5)}",
                f"--agent_id={agent.agent_id}",
            ]

            try:
                jobs = self.selfplay_job_manager.launch(
                    num_workers=1,
                    model_version=version,
                    iteration=gen,
                    extra_args=extra_args,
                )
                job_names.extend(jobs)
            except Exception:
                logger.exception(
                    "Self-play error for agent %s", agent.agent_id
                )

        if job_names:
            self.selfplay_job_manager.wait_for_completion(job_names)
        else:
            logger.warning("No self-play jobs launched.")

    # ------------------------------------------------------------------
    # Phase 2: Training
    # ------------------------------------------------------------------

    def _run_training_phase(self, gen: int) -> None:
        """Launch and wait for training jobs."""
        logger.info("Phase 2: Training (gen=%d)", gen)
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

    def _run_evaluation_phase(self, gen: int) -> EvaluationReport:
        """Assign fitness scores via cross-agent arena evaluation.

        Uses :class:`PBTEvaluator` to run round-robin (or top-k) arena
        matches, then converts match results into ELO-derived fitness.
        """
        logger.info("Phase 3: Evaluation (gen=%d)", gen)

        agents = self.manager.get_all_agents()
        report = self.evaluator.evaluate_population(
            agents,
            generation=gen,
            arena_runner=_local_arena_runner,
        )

        for agent in agents:
            logger.info(
                "Agent %s: elo=%.0f fitness=%.3f",
                agent.agent_id,
                agent.elo_rating,
                agent.fitness_score,
            )

        return report

    # ------------------------------------------------------------------
    # Phase 4: Evolution
    # ------------------------------------------------------------------

    def _run_evolution_phase(self, gen: int) -> None:
        """Run PBT exploit-and-explore on the population."""
        logger.info("Phase 4: Evolution (gen=%d)", gen)
        self.manager.exploit_and_explore()


# ---------------------------------------------------------------------------
# Default arena runner (local mode)
# ---------------------------------------------------------------------------


def _local_arena_runner(
    agent_new: AgentState,
    agent_old: AgentState,
    num_games: int,
) -> MatchResult:
    """Local arena runner using lazy torch import.

    Falls back to a balanced placeholder result if torch/networks
    are not importable (e.g. in CI or CPU-only environments).
    """
    try:
        import torch  # type: ignore[import-untyped]

        from alpha_zero.cloud.gcp.evaluation_worker import run_arena

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from alpha_zero.core.networks import create_network  # type: ignore[import-untyped]

        model_new = create_network("transformer", device=str(device))
        model_old = create_network("transformer", device=str(device))

        results = run_arena(
            model_new=model_new,
            model_old=model_old,
            num_games=num_games,
            mcts_type="gumbel",
            mcts_sims=25,
            gumbel_num_actions=8,
            board_size=6,
            max_turns=100,
            device=device,
        )

        return MatchResult(
            agent_id_new=agent_new.agent_id,
            agent_id_old=agent_old.agent_id,
            win_rate=results["win_rate"],
            draw_rate=results["draw_rate"],
            loss_rate=results["loss_rate"],
            total_games=int(results["total_games"]),
        )

    except ImportError:
        logger.warning(
            "torch/networks not available; returning placeholder result"
        )
        return MatchResult(
            agent_id_new=agent_new.agent_id,
            agent_id_old=agent_old.agent_id,
            win_rate=0.5,
            draw_rate=0.0,
            loss_rate=0.5,
            total_games=num_games,
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
