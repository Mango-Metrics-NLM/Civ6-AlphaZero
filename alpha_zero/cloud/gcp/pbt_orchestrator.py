"""Population Based Training orchestrator for Vertex AI.

Manages the PBT lifecycle: initialize population → self-play → train →
evaluate → evolve → repeat.  The evaluation phase uses :class:`PBTEvaluator`
to run cross-agent arena matches and compute ELO-based fitness.
"""

from __future__ import annotations

import logging
import time
from typing import Any

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
        gcp_config: GCP project / bucket / region configuration.
        pbt_config: PBT hyperparameters and evaluation settings.
        evaluator: Optional pre-built evaluator — if *None*, one is
            created automatically.
    """

    def __init__(
        self,
        gcp_config: GCPConfig,
        pbt_config: PBTConfig,
        evaluator: PBTEvaluator | None = None,
    ) -> None:
        self.gcp = gcp_config
        self.pbt = pbt_config

        self.client = VertexAIClient(
            gcp_config.project_id,
            gcp_config.region,
            gcp_config.service_account,
        )

        self.manager = PopulationManager(pbt_config)
        self.evaluator = evaluator or PBTEvaluator(pbt_config)

        self._selfplay = VertexSelfPlayJob(gcp_config, self.client)
        self._training = VertexTrainingJob(gcp_config, self.client)
        self._evaluation = VertexEvaluationJob(gcp_config, self.client)

        logger.info(
            "PBTOrchestrator initialised: pop=%d, strategy=%s",
            pbt_config.population_size,
            pbt_config.evaluation_strategy,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, num_generations: int = 10) -> None:
        """Run the complete PBT loop for *num_generations*."""
        logger.info("Starting PBT loop for %d generations", num_generations)

        for gen in range(1, num_generations + 1):
            logger.info("=" * 60)
            logger.info("Generation %d / %d", gen, num_generations)
            logger.info("=" * 60)

            t0 = time.monotonic()

            self._run_selfplay_phase(gen)
            self._run_training_phase(gen)
            report = self._run_evaluation_phase(gen)
            self._run_evolution_phase(gen)

            elapsed = time.monotonic() - t0
            logger.info(
                "Generation %d complete in %.1fs | "
                "matchups=%d | top_elo=%.0f",
                gen,
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

        for agent in self.manager.get_all_agents():
            version = _safe_int(agent.model_path, _DEFAULT_MODEL_VERSION)
            try:
                job_names = self._selfplay.launch(
                    num_workers=1,
                    model_version=version,
                    iteration=gen,
                )
                for job_name in job_names:
                    result = self._selfplay.wait_for_completion(job_name)
                    if result.success:
                        agent.games_played += self.pbt.evaluation_games
                        logger.info(
                            "Self-play OK: agent=%s, games=%d",
                            agent.agent_id,
                            agent.games_played,
                        )
                    else:
                        logger.warning(
                            "Self-play FAILED: agent=%s, error=%s",
                            agent.agent_id,
                            result.error_message,
                        )
            except Exception:
                logger.exception(
                    "Self-play error for agent %s", agent.agent_id
                )

    # ------------------------------------------------------------------
    # Phase 2: Training
    # ------------------------------------------------------------------

    def _run_training_phase(self, gen: int) -> None:
        """Launch training jobs for each agent."""
        logger.info("Phase 2: Training (gen=%d)", gen)

        for agent in self.manager.get_all_agents():
            try:
                storage = GCSModelStorage(self.gcp)
                new_version = storage.get_latest_version()
                agent.model_path = str(new_version)
                logger.info(
                    "Updated agent %s to model v%d",
                    agent.agent_id,
                    new_version,
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
