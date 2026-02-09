"""Population Based Training orchestrator for Vertex AI.

Manages the PBT lifecycle: initialize population -> self-play -> train ->
evaluate -> evolve -> repeat.  The evaluation phase uses :class:`PBTEvaluator`
to run cross-agent arena matches and compute ELO-based fitness.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.storage import GCSBackend, GCSModelStorage
from alpha_zero.cloud.gcp.vertex_jobs import (
    VertexAIClient,
    VertexSelfPlayJob,
    VertexTrainingJob,
)
from alpha_zero.pbt.config import AgentState, PBTConfig
from alpha_zero.pbt.evaluator import EvaluationReport, MatchResult, PBTEvaluator
from alpha_zero.pbt.population import PopulationManager
from alpha_zero.utils.wandb_logger import WandbLogger

# Training safeguards (optional, graceful degradation)
try:
    from alpha_zero.training.training_safeguards import TrainingSafeguardConfig

    _SAFEGUARDS_AVAILABLE = True
except ImportError:
    _SAFEGUARDS_AVAILABLE = False
    TrainingSafeguardConfig = None  # type: ignore[assignment, misc]

# MuZero integration (optional, graceful degradation)
try:
    from alpha_zero.training.muzero_integration import MuZeroIntegrationConfig

    _MUZERO_AVAILABLE = True
except ImportError:
    _MUZERO_AVAILABLE = False
    MuZeroIntegrationConfig = None  # type: ignore[assignment, misc]

# Tournament integration (optional, graceful degradation)
try:
    from alpha_zero.tournament.orchestrator import (
        TournamentOrchestrator,
    )

    _TOURNAMENT_AVAILABLE = True
except ImportError:
    _TOURNAMENT_AVAILABLE = False
    TournamentOrchestrator = None  # type: ignore[assignment, misc]

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
        *,
        wandb_logger: WandbLogger | None = None,
        checkpoint_retention: int | None = None,
        evaluation_strategy_switch_gen: int | None = None,
        baseline_agents: list[AgentState] | None = None,
        training_safeguard_config: TrainingSafeguardConfig | None = None,
        muzero_config: MuZeroIntegrationConfig | None = None,
        tournament_interval: int = 0,
        tournament_games_per_match: int = 10,
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

        # State storage
        self.storage_backend = GCSBackend(self.gcp_config.project_id)
        self.population_state_path = (
            f"{self.gcp_config.paths.logs_prefix}/pbt_state.json"
        )

        # Production features
        self.wandb_logger = wandb_logger
        self.checkpoint_retention = (
            checkpoint_retention if checkpoint_retention is not None
            else pbt_config.checkpoint_retention_generations
        )
        self.evaluation_strategy_switch_gen = (
            evaluation_strategy_switch_gen if evaluation_strategy_switch_gen is not None
            else pbt_config.evaluation_strategy_switch_gen
        )
        self.baseline_agents = baseline_agents or []
        self.training_safeguard_config = training_safeguard_config
        self.muzero_config = muzero_config
        self.tournament_interval = tournament_interval
        self.tournament_games_per_match = tournament_games_per_match

        # Define PBT W&B charts if logger is provided
        if self.wandb_logger is not None and pbt_config.wandb_enabled:
            self.wandb_logger.define_pbt_charts()

        logger.info(
            "PBTOrchestrator initialised: pop=%d, strategy=%s, "
            "checkpoint_retention=%d, strategy_switch_gen=%d",
            pbt_config.population_size,
            pbt_config.evaluation_strategy,
            self.checkpoint_retention,
            self.evaluation_strategy_switch_gen,
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
    # Production features
    # ------------------------------------------------------------------

    def _log_population_dashboard(
        self,
        gen: int,
        report: EvaluationReport,
        elapsed: float,
    ) -> None:
        """Log population-level metrics to W&B."""
        if self.wandb_logger is None:
            return

        agents = self.manager.get_all_agents()
        if not agents:
            return

        elo_values = [a.elo_rating for a in agents]
        fitness_values = [a.fitness_score for a in agents]

        self.wandb_logger.log_population_stats(
            generation=gen,
            mean_fitness=sum(fitness_values) / len(fitness_values),
            max_fitness=max(fitness_values),
            min_fitness=min(fitness_values),
            mean_elo=sum(elo_values) / len(elo_values),
            max_elo=max(elo_values),
            min_elo=min(elo_values),
            population_size=len(agents),
            elapsed_seconds=elapsed,
        )

        # Per-agent ELO
        per_agent_data = {
            f"pbt/agent/{a.agent_id}/elo": a.elo_rating for a in agents
        }
        self.wandb_logger.log(per_agent_data, step=gen)

    def _log_mutation_events(
        self,
        gen: int,
        changes: list[tuple[str, str, dict[str, Any]]],
    ) -> None:
        """Log mutation events to W&B."""
        if self.wandb_logger is None:
            return

        self.wandb_logger.log({"pbt/mutations_count": len(changes)}, step=gen)

        if changes:
            columns = ["replaced_id", "source_id", "learning_rate", "c_puct"]
            rows = [
                [
                    replaced_id,
                    source_id,
                    new_hp.get("learning_rate", ""),
                    new_hp.get("c_puct", ""),
                ]
                for replaced_id, source_id, new_hp in changes
            ]
            self.wandb_logger.log_table(
                f"pbt/mutations_gen_{gen}", columns, rows
            )

    def _rotate_checkpoints(self, gen: int) -> None:
        """Delete old checkpoint generations beyond retention window."""
        if self.checkpoint_retention <= 0:
            return

        try:
            bucket = self.gcp_config.bucket
            manifest_path = f"{self.gcp_config.paths.checkpoints_prefix}/checkpoint_manifest.json"

            # Load or create manifest
            manifest: dict[str, list[str]] = {}
            try:
                if self.storage_backend.blob_exists(bucket, manifest_path):
                    data = self.storage_backend.download_blob(bucket, manifest_path)
                    manifest = json.loads(data.decode())
            except Exception:
                logger.debug("No existing checkpoint manifest; creating new one.")

            # Record current generation's checkpoint paths
            gen_key = str(gen)
            prefix = f"{self.gcp_config.paths.checkpoints_prefix}/gen_{gen}/"
            try:
                current_blobs = self.storage_backend.list_blobs(bucket, prefix)
                manifest[gen_key] = current_blobs
            except Exception:
                logger.debug("No checkpoint blobs found for gen %d", gen)
                manifest[gen_key] = []

            # Determine which generations to prune
            all_gens = sorted(int(k) for k in manifest)
            cutoff = gen - self.checkpoint_retention
            to_prune = [g for g in all_gens if g <= cutoff]

            deleted_count = 0
            for old_gen in to_prune:
                old_key = str(old_gen)
                for blob_path in manifest.get(old_key, []):
                    try:
                        self.storage_backend.delete_blob(bucket, blob_path)
                        deleted_count += 1
                    except Exception:
                        logger.debug("Failed to delete blob: %s", blob_path)
                manifest.pop(old_key, None)

            if deleted_count > 0:
                logger.info(
                    "Checkpoint rotation: deleted %d blobs from %d old generations",
                    deleted_count,
                    len(to_prune),
                )

            # Save updated manifest
            manifest_data = json.dumps(manifest, indent=2).encode()
            self.storage_backend.upload_blob(bucket, manifest_path, manifest_data)

        except Exception:
            logger.warning("Checkpoint rotation failed (non-fatal)", exc_info=True)

    def _check_quota(self, gen: int) -> bool:
        """Check whether GPU quota is available for the next phase.

        Returns ``True`` if quota is sufficient or on any error
        (optimistic fallback).
        """
        quota_limit_str = os.environ.get("CIV6AZ_GPU_QUOTA_LIMIT", "")
        if not quota_limit_str:
            return True

        try:
            from google.cloud import aiplatform  # type: ignore[import-untyped]

            quota_limit = int(quota_limit_str)

            aiplatform.init(
                project=self.gcp_config.project_id,
                location=self.gcp_config.region,
            )
            active_jobs = aiplatform.CustomJob.list(
                filter='state="JOB_STATE_RUNNING"',
                project=self.gcp_config.project_id,
                location=self.gcp_config.region,
            )

            gpu_count = sum(
                int(getattr(job, "gpu_count", 1) or 1)
                for job in active_jobs
            )

            if gpu_count >= quota_limit:
                logger.warning(
                    "GPU quota exceeded: %d active >= %d limit (gen=%d)",
                    gpu_count,
                    quota_limit,
                    gen,
                )
                return False

            return True

        except Exception:
            logger.debug("Quota check failed (optimistic fallback)", exc_info=True)
            return True

    def _wait_for_quota(self, gen: int) -> None:
        """Poll until GPU quota becomes available."""
        max_wait = int(os.environ.get("CIV6AZ_QUOTA_WAIT_MAX_SECONDS", "1800"))
        wait_time = 60
        max_interval = 300
        elapsed = 0

        while elapsed < max_wait:
            logger.warning(
                "Waiting for GPU quota (gen=%d, waited=%ds/%ds)...",
                gen,
                elapsed,
                max_wait,
            )
            time.sleep(wait_time)
            elapsed += wait_time

            if self._check_quota(gen):
                logger.info("GPU quota available after %ds", elapsed)
                return

            wait_time = min(wait_time * 2, max_interval)

        logger.warning(
            "Quota wait timeout after %ds; proceeding anyway (gen=%d)",
            max_wait,
            gen,
        )

    def _maybe_switch_strategy(self, gen: int) -> None:
        """Switch evaluation strategy at the configured generation."""
        switch_gen = self.evaluation_strategy_switch_gen
        if switch_gen <= 0 or gen < switch_gen:
            return

        if self.pbt_config.evaluation_strategy == "top_k_vs_best":
            return  # Already switched

        old_strategy = self.pbt_config.evaluation_strategy
        self.pbt_config.evaluation_strategy = "top_k_vs_best"
        logger.info(
            "Evaluation strategy switched: %s -> top_k_vs_best at gen %d",
            old_strategy,
            gen,
        )

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

        total_gen_val = num_generations if num_generations is not None else getattr(self.pbt_config, "num_generations", 10)
        total: int = int(total_gen_val) if total_gen_val is not None else 10
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

            # Dynamic strategy switch
            self._maybe_switch_strategy(gen_num)

            # Quota guard
            if not self._check_quota(gen_num):
                self._wait_for_quota(gen_num)

            self._run_selfplay_phase(gen_num)
            self._run_training_phase(gen_num)
            report = self._run_evaluation_phase(gen_num)
            changes = self._run_evolution_phase(gen_num)
            self.save_state()
            self._rotate_checkpoints(gen_num)

            # Tournament evaluation (Phase 10)
            self._run_tournament_evaluation(gen_num)

            elapsed = time.monotonic() - t0

            # W&B dashboards
            self._log_population_dashboard(gen_num, report, elapsed)
            self._log_mutation_events(gen_num, changes)

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

            # Pass MuZero config if configured
            if self.muzero_config is not None:
                extra_args.append("--training_mode=muzero")
                extra_args.append(
                    f"--muzero_unroll_steps={self.muzero_config.unroll_steps}"
                )
                extra_args.append(
                    f"--muzero_td_steps={self.muzero_config.td_steps}"
                )
                extra_args.append(
                    f"--muzero_discount={self.muzero_config.discount}"
                )

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
        job_map: dict[str, AgentState] = {}

        for agent in self.manager.get_all_agents():
            extra_args = [
                f"--learning_rate={agent.hyperparameters.get('learning_rate', 0.01)}",
                f"--agent_id={agent.agent_id}",
            ]

            # Pass MuZero config as extra args if configured
            if self.muzero_config is not None:
                extra_args.append("--training_mode=muzero")
                extra_args.append(
                    f"--muzero_unroll_steps={self.muzero_config.unroll_steps}"
                )
                extra_args.append(
                    f"--muzero_td_steps={self.muzero_config.td_steps}"
                )
                extra_args.append(
                    f"--muzero_consistency_weight="
                    f"{self.muzero_config.consistency_loss_weight}"
                )
                extra_args.append(
                    f"--muzero_reward_weight="
                    f"{self.muzero_config.reward_loss_weight}"
                )

            # Pass safeguard config as env-var overrides if configured
            if self.training_safeguard_config is not None:
                cfg = self.training_safeguard_config
                extra_args.append(
                    f"--safeguard_loss_threshold={cfg.loss_explosion_threshold}"
                )
                extra_args.append(
                    f"--safeguard_max_nan={cfg.max_consecutive_nan_steps}"
                )

            logger.info("Launching training for agent %s...", agent.agent_id)
            job_name = self.training_job_manager.launch(
                iteration=gen,
                trajectory_iterations=[gen],
                resume_from_checkpoint="latest",
                extra_args=extra_args,
            )
            job_map[job_name] = agent

        self._poll_jobs(job_map)

    def _poll_jobs(self, job_map: dict[str, AgentState]) -> None:
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

        # Prepare evaluation arguments
        eval_kwargs: dict[str, Any] = {"generation": gen}
        if self.evaluator.arena_runner is None:
            logger.info("No arena_runner configured; falling back to local runner")
            eval_kwargs["arena_runner"] = lambda agent_new, agent_old, num_games: _local_arena_runner(
                agent_new, agent_old, num_games, gcp_config=self.gcp_config
            )

        report = self.evaluator.evaluate_population(agents, **eval_kwargs)

        for agent in agents:
            logger.info(
                "Agent %s: elo=%.0f fitness=%.3f",
                agent.agent_id,
                agent.elo_rating,
                agent.fitness_score,
            )

        return report

    # ------------------------------------------------------------------
    # Phase 3b: Tournament evaluation (optional)
    # ------------------------------------------------------------------

    def _run_tournament_evaluation(self, gen: int) -> None:
        """Run tournament evaluation if configured.

        Runs a mini-tournament among population agents at the configured
        interval.  Results are logged to W&B if available.

        Args:
            gen: Current generation number.
        """
        if self.tournament_interval <= 0:
            return
        if gen % self.tournament_interval != 0:
            return
        if not _TOURNAMENT_AVAILABLE:
            logger.debug(
                "Tournament module not available; skipping tournament "
                "evaluation at gen %d",
                gen,
            )
            return

        logger.info("Running tournament evaluation at gen %d", gen)

        try:
            agents = self.manager.get_all_agents()
            player_ids = [a.agent_id for a in agents]

            if len(player_ids) < 2:
                logger.warning(
                    "Need at least 2 agents for tournament; skipping"
                )
                return

            # Create tournament orchestrator with existing evaluator's
            # arena runner if available
            arena_runner = getattr(
                self.evaluator, "arena_runner", None
            )
            tournament = TournamentOrchestrator(
                arena_runner=arena_runner,
                games_per_match=self.tournament_games_per_match,
            )

            # Create and run bracket
            bracket = tournament.create_bracket(player_ids)
            result = tournament.run_tournament(bracket)

            logger.info(
                "Tournament at gen %d complete: winner=%s, "
                "games=%d, standings=%d",
                gen,
                result.winner,
                result.total_games,
                len(result.final_standings),
            )

            # Log to W&B if available
            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    {
                        "tournament/generation": gen,
                        "tournament/total_games": result.total_games,
                        "tournament/winner": result.winner,
                        "tournament/duration_seconds": (
                            result.duration_seconds
                        ),
                    },
                    step=gen,
                )

        except Exception:
            logger.warning(
                "Tournament evaluation failed at gen %d (non-fatal)",
                gen,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Phase 4: Evolution
    # ------------------------------------------------------------------

    def _run_evolution_phase(
        self, gen: int
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Run PBT exploit-and-explore on the population.

        Returns:
            List of ``(replaced_id, source_id, new_hyperparameters)`` tuples.
        """
        logger.info("Phase 4: Evolution (gen=%d)", gen)
        return self.manager.exploit_and_explore()


# ---------------------------------------------------------------------------
# Default arena runner (local mode)
# ---------------------------------------------------------------------------


def _local_arena_runner(
    agent_new: AgentState,
    agent_old: AgentState,
    num_games: int,
    gcp_config: GCPConfig | None = None,
) -> MatchResult:
    """Local arena runner using lazy torch import.

    Attempts to load agent checkpoints if *gcp_config* is provided.
    Falls back to random initialization or placeholder results if
    torch/networks are not importable.
    """
    try:
        import torch  # type: ignore[import-untyped]

        from alpha_zero.cloud.gcp.evaluation_worker import run_arena
        from alpha_zero.utils.config import NetworkConfig
        from alpha_zero.utils.factories import create_network

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Derive board_size/action_space from environment or fallback to defaults
        # In a real run, these should come from Global Config
        board_size = int(os.environ.get("CIV6AZ_BOARD_SIZE", 6))
        action_space_size = int(os.environ.get("CIV6AZ_ACTION_SPACE", 158))

        net_config = NetworkConfig()

        def _prepare_model(agent: AgentState) -> torch.nn.Module:
            model = create_network(
                config=net_config,
                board_size=board_size,
                action_space_size=action_space_size,
                device=str(device),
            )
            if agent.model_path and gcp_config:
                try:
                    # Load weights from GCS using GCSModelStorage
                    from alpha_zero.cloud.gcp.storage import GCSModelStorage
                    storage = GCSModelStorage(gcp_config)
                    # Use provided version/path. If it's a version number, convert to int.
                    try:
                        version = int(agent.model_path)
                        state_dict, _, _ = storage.load_model(version)
                        model.load_state_dict(state_dict)
                        logger.info("Loaded weights for agent %s (v%d) from GCS", agent.agent_id, version)
                    except ValueError:
                        # Handle as a path if not an integer version
                        logger.warning(
                            "Agent %s model_path '%s' is not a version number; skipping weight load.",
                            agent.agent_id, agent.model_path,
                        )
                except Exception as e:
                    logger.warning("Failed to load weights for agent %s: %r. Using random init.", agent.agent_id, e)
            return model

        model_new = _prepare_model(agent_new)
        model_old = _prepare_model(agent_old)

        results = run_arena(
            model_new=model_new,
            model_old=model_old,
            num_games=num_games,
            mcts_type="gumbel",
            mcts_sims=25,
            gumbel_num_actions=8,
            board_size=board_size,
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
