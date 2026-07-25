"""Vertex AI job launcher for Civ6 AlphaZero training.

Provides utilities to submit training jobs to Vertex AI:
- Self-play worker fleet
- Training jobs
- Evaluation jobs
- Full training iterations

Usage:
    # Submit self-play fleet
    python -m infra.vertex_launcher --mode=selfplay --iteration=5 --num-workers=3

    # Submit training job
    python -m infra.vertex_launcher --mode=train --iteration=5

    # Submit evaluation job
    python -m infra.vertex_launcher --mode=eval --iteration=5

    # Run full iteration
    python -m infra.vertex_launcher --mode=iteration --iteration=5

Environment Variables:
    GOOGLE_CLOUD_PROJECT: GCP project ID
    CIV6_GCS_BUCKET: GCS bucket name
    CIV6_REGION: GCP region (default: us-central1)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Optional imports
try:
    from google.cloud import aiplatform

    HAS_VERTEX = True
except ImportError:
    HAS_VERTEX = False
    aiplatform = None  # type: ignore


@dataclass
class JobConfig:
    """Configuration for a Vertex AI job.

    Attributes:
        display_name: Job display name.
        container_uri: Container image URI.
        machine_type: GCP machine type.
        accelerator_type: GPU type (or empty for CPU).
        accelerator_count: Number of GPUs.
        replica_count: Number of replicas.
        spot: Whether to use Spot VMs.
        args: Command-line arguments for container.
        env_vars: Environment variables.
        timeout_hours: Job timeout in hours.
    """

    display_name: str
    container_uri: str
    machine_type: str = "n1-standard-8"
    accelerator_type: str = ""
    accelerator_count: int = 0
    replica_count: int = 1
    spot: bool = False
    args: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    timeout_hours: float = 6.0

    def to_worker_pool_spec(self) -> dict[str, Any]:
        """Convert to Vertex AI worker pool spec format."""
        machine_spec: dict[str, Any] = {
            "machine_type": self.machine_type,
        }

        if self.accelerator_type and self.accelerator_count > 0:
            machine_spec["accelerator_type"] = self.accelerator_type
            machine_spec["accelerator_count"] = self.accelerator_count

        container_spec: dict[str, Any] = {
            "image_uri": self.container_uri,
        }

        if self.args:
            container_spec["args"] = self.args

        if self.env_vars:
            container_spec["env"] = [{"name": k, "value": v} for k, v in self.env_vars.items()]

        return {
            "machine_spec": machine_spec,
            "replica_count": self.replica_count,
            "container_spec": container_spec,
        }


class VertexAILauncher:
    """Launcher for Vertex AI training jobs.

    Attributes:
        project_id: GCP project ID.
        region: GCP region.
        bucket: GCS bucket for storage.
        artifact_registry: Artifact Registry URI.
    """

    def __init__(
        self,
        project_id: str | None = None,
        region: str = "us-central1",
        bucket: str | None = None,
        artifact_registry: str | None = None,
    ) -> None:
        """Initialize Vertex AI launcher.

        Args:
            project_id: GCP project ID. Defaults to environment.
            region: GCP region.
            bucket: GCS bucket name.
            artifact_registry: Artifact Registry URI.
        """
        if not HAS_VERTEX:
            raise ImportError(
                "google-cloud-aiplatform is required. "
                "Install with: pip install google-cloud-aiplatform"
            )

        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.region = region or os.environ.get("CIV6_REGION", "us-central1")
        self.bucket = bucket or os.environ.get("CIV6_GCS_BUCKET", "")
        self.artifact_registry = artifact_registry or os.environ.get(
            "CIV6_ARTIFACT_REGISTRY",
            f"{self.region}-docker.pkg.dev/{self.project_id}/civ6-alphazero",
        )

        if not self.project_id:
            raise ValueError("project_id is required")

        if not self.bucket:
            raise ValueError("bucket is required")

        # Initialize Vertex AI SDK
        aiplatform.init(
            project=self.project_id,
            location=self.region,
        )

        logger.info(
            f"VertexAILauncher initialized (project={self.project_id}, region={self.region})"
        )

    def _get_container_uri(self, component: str) -> str:
        """Get container image URI for a component."""
        return f"{self.artifact_registry}/civ6-{component}:latest"

    def _get_base_env_vars(self, iteration: int) -> dict[str, str]:
        """Get common environment variables."""
        return {
            "CIV6_STORAGE_BACKEND": "gcs",
            "CIV6_GCS_BUCKET": self.bucket,
            "CIV6_GCS_PREFIX": "civ6-alphazero",
            "CIV6_ITERATION": str(iteration),
            "GOOGLE_CLOUD_PROJECT": self.project_id,
        }

    def submit_selfplay_job(
        self,
        iteration: int,
        worker_id: int,
        num_games: int = 100,
        num_simulations: int = 800,
        profile: str = "selfplay_t4_spot",
        board_size: int = 8,
    ) -> aiplatform.CustomJob:
        """Submit a self-play worker job.

        Args:
            iteration: Training iteration.
            worker_id: Worker identifier.
            num_games: Games to play.
            num_simulations: MCTS simulations per move.
            profile: Machine profile name.
            board_size: Game board size.

        Returns:
            Submitted CustomJob.
        """
        from alpha_zero.cloud.config import get_machine_profile

        machine = get_machine_profile(profile)

        env_vars = self._get_base_env_vars(iteration)
        env_vars.update(
            {
                "CIV6_WORKER_ID": str(worker_id),
                "CIV6_NUM_GAMES": str(num_games),
                "CIV6_NUM_SIMULATIONS": str(num_simulations),
                "CIV6_BOARD_SIZE": str(board_size),
                "CIV6_DEVICE": "cuda" if machine.accelerator_count > 0 else "cpu",
            }
        )

        job_config = JobConfig(
            display_name=f"civ6-selfplay-iter{iteration}-worker{worker_id}",
            container_uri=self._get_container_uri("selfplay"),
            machine_type=machine.machine_type,
            accelerator_type=machine.accelerator_type,
            accelerator_count=machine.accelerator_count,
            spot=machine.spot,
            args=[
                f"--num-games={num_games}",
                f"--num-simulations={num_simulations}",
                f"--iteration={iteration}",
                f"--worker-id={worker_id}",
            ],
            env_vars=env_vars,
            timeout_hours=3.0,
        )

        job = aiplatform.CustomJob(
            display_name=job_config.display_name,
            worker_pool_specs=[job_config.to_worker_pool_spec()],
        )

        logger.info(f"Submitting self-play job: {job_config.display_name}")
        job.run(sync=False)
        return job

    def submit_selfplay_fleet(
        self,
        iteration: int,
        num_workers: int = 3,
        games_per_worker: int = 100,
        num_simulations: int = 800,
        profile: str = "selfplay_t4_spot",
        board_size: int = 8,
    ) -> list[aiplatform.CustomJob]:
        """Submit a fleet of self-play workers.

        Args:
            iteration: Training iteration.
            num_workers: Number of workers.
            games_per_worker: Games per worker.
            num_simulations: MCTS simulations per move.
            profile: Machine profile name.
            board_size: Game board size.

        Returns:
            List of submitted jobs.
        """
        jobs = []
        for worker_id in range(num_workers):
            job = self.submit_selfplay_job(
                iteration=iteration,
                worker_id=worker_id,
                num_games=games_per_worker,
                num_simulations=num_simulations,
                profile=profile,
                board_size=board_size,
            )
            jobs.append(job)
            time.sleep(1)  # Small delay to avoid rate limiting

        logger.info(f"Submitted {len(jobs)} self-play workers")
        return jobs

    def submit_training_job(
        self,
        iteration: int,
        training_steps: int = 1000,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        replay_window: int = 10,
        profile: str = "train_l4_single",
        board_size: int = 8,
    ) -> aiplatform.CustomJob:
        """Submit a training job.

        Args:
            iteration: Training iteration.
            training_steps: Number of training steps.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            replay_window: Iterations to include in replay buffer.
            profile: Machine profile name.
            board_size: Game board size.

        Returns:
            Submitted CustomJob.
        """
        from alpha_zero.cloud.config import get_machine_profile

        machine = get_machine_profile(profile)

        env_vars = self._get_base_env_vars(iteration)
        env_vars.update(
            {
                "CIV6_TRAINING_STEPS": str(training_steps),
                "CIV6_BATCH_SIZE": str(batch_size),
                "CIV6_LEARNING_RATE": str(learning_rate),
                "CIV6_REPLAY_WINDOW": str(replay_window),
                "CIV6_BOARD_SIZE": str(board_size),
                "CIV6_DEVICE": "cuda" if machine.accelerator_count > 0 else "cpu",
                "CIV6_USE_AMP": "true" if machine.accelerator_count > 0 else "false",
            }
        )

        job_config = JobConfig(
            display_name=f"civ6-trainer-iter{iteration}",
            container_uri=self._get_container_uri("trainer"),
            machine_type=machine.machine_type,
            accelerator_type=machine.accelerator_type,
            accelerator_count=machine.accelerator_count,
            spot=machine.spot,
            args=[
                f"--training-steps={training_steps}",
                f"--batch-size={batch_size}",
                f"--iteration={iteration}",
            ],
            env_vars=env_vars,
            timeout_hours=2.0,
        )

        job = aiplatform.CustomJob(
            display_name=job_config.display_name,
            worker_pool_specs=[job_config.to_worker_pool_spec()],
        )

        logger.info(f"Submitting training job: {job_config.display_name}")
        job.run(sync=False)
        return job

    def submit_evaluation_job(
        self,
        iteration: int,
        num_games: int = 400,
        num_simulations: int = 200,
        promotion_threshold: float = 0.55,
        profile: str = "eval_l4_spot",
        board_size: int = 8,
    ) -> aiplatform.CustomJob:
        """Submit an evaluation job.

        Args:
            iteration: Training iteration.
            num_games: Number of evaluation games.
            num_simulations: MCTS simulations per move.
            promotion_threshold: Win rate for promotion.
            profile: Machine profile name.
            board_size: Game board size.

        Returns:
            Submitted CustomJob.
        """
        from alpha_zero.cloud.config import get_machine_profile

        machine = get_machine_profile(profile)

        env_vars = self._get_base_env_vars(iteration)
        env_vars.update(
            {
                "CIV6_EVAL_GAMES": str(num_games),
                "CIV6_NUM_SIMULATIONS": str(num_simulations),
                "CIV6_PROMOTION_THRESHOLD": str(promotion_threshold),
                "CIV6_BOARD_SIZE": str(board_size),
                "CIV6_DEVICE": "cuda" if machine.accelerator_count > 0 else "cpu",
            }
        )

        job_config = JobConfig(
            display_name=f"civ6-evaluator-iter{iteration}",
            container_uri=self._get_container_uri("evaluator"),
            machine_type=machine.machine_type,
            accelerator_type=machine.accelerator_type,
            accelerator_count=machine.accelerator_count,
            spot=machine.spot,
            args=[
                f"--num-games={num_games}",
                f"--num-simulations={num_simulations}",
                f"--iteration={iteration}",
            ],
            env_vars=env_vars,
            timeout_hours=1.0,
        )

        job = aiplatform.CustomJob(
            display_name=job_config.display_name,
            worker_pool_specs=[job_config.to_worker_pool_spec()],
        )

        logger.info(f"Submitting evaluation job: {job_config.display_name}")
        job.run(sync=False)
        return job

    def wait_for_jobs(
        self,
        jobs: list[aiplatform.CustomJob],
        poll_interval: float = 60.0,
        timeout_hours: float = 6.0,
    ) -> list[bool]:
        """Wait for jobs to complete.

        Args:
            jobs: List of jobs to wait for.
            poll_interval: Seconds between status checks.
            timeout_hours: Maximum wait time in hours.

        Returns:
            List of success flags (True if completed successfully).
        """
        start_time = time.time()
        timeout_seconds = timeout_hours * 3600
        results: list[bool | None] = [None] * len(jobs)

        while None in results:
            if time.time() - start_time > timeout_seconds:
                logger.error("Timeout waiting for jobs")
                break

            for i, job in enumerate(jobs):
                if results[i] is not None:
                    continue

                job.refresh()
                state = job.state

                if state == aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED:
                    logger.info(f"Job {job.display_name} succeeded")
                    results[i] = True
                elif state in [
                    aiplatform.gapic.JobState.JOB_STATE_FAILED,
                    aiplatform.gapic.JobState.JOB_STATE_CANCELLED,
                ]:
                    logger.error(f"Job {job.display_name} failed/cancelled: {state}")
                    results[i] = False

            if None in results:
                time.sleep(poll_interval)

        return [r if r is not None else False for r in results]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Vertex AI Job Launcher for Civ6 AlphaZero",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["selfplay", "train", "eval", "iteration"],
        help="Job type to submit",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Training iteration",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="Number of self-play workers",
    )
    parser.add_argument(
        "--games-per-worker",
        type=int,
        default=100,
        help="Games per self-play worker",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=1000,
        help="Training steps per iteration",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=400,
        help="Evaluation games",
    )
    parser.add_argument(
        "--selfplay-profile",
        type=str,
        default="selfplay_t4_spot",
        help="Machine profile for self-play",
    )
    parser.add_argument(
        "--train-profile",
        type=str,
        default="train_l4_single",
        help="Machine profile for training",
    )
    parser.add_argument(
        "--eval-profile",
        type=str,
        default="eval_l4_spot",
        help="Machine profile for evaluation",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=8,
        help="Game board size",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for jobs to complete",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without submitting",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    if args.dry_run:
        logger.info(f"Dry run - would submit {args.mode} job(s)")
        logger.info(f"  Iteration: {args.iteration}")
        logger.info(f"  Workers: {args.num_workers}")
        logger.info(f"  Games/worker: {args.games_per_worker}")
        return

    launcher = VertexAILauncher()

    if args.mode == "selfplay":
        jobs = launcher.submit_selfplay_fleet(
            iteration=args.iteration,
            num_workers=args.num_workers,
            games_per_worker=args.games_per_worker,
            profile=args.selfplay_profile,
            board_size=args.board_size,
        )
        if args.wait:
            launcher.wait_for_jobs(jobs)

    elif args.mode == "train":
        job = launcher.submit_training_job(
            iteration=args.iteration,
            training_steps=args.training_steps,
            profile=args.train_profile,
            board_size=args.board_size,
        )
        if args.wait:
            launcher.wait_for_jobs([job])

    elif args.mode == "eval":
        job = launcher.submit_evaluation_job(
            iteration=args.iteration,
            num_games=args.eval_games,
            profile=args.eval_profile,
            board_size=args.board_size,
        )
        if args.wait:
            launcher.wait_for_jobs([job])

    elif args.mode == "iteration":
        logger.info(f"Running full iteration {args.iteration}")

        # 1. Self-play
        selfplay_jobs = launcher.submit_selfplay_fleet(
            iteration=args.iteration,
            num_workers=args.num_workers,
            games_per_worker=args.games_per_worker,
            profile=args.selfplay_profile,
            board_size=args.board_size,
        )
        launcher.wait_for_jobs(selfplay_jobs)

        # 2. Training
        train_job = launcher.submit_training_job(
            iteration=args.iteration,
            training_steps=args.training_steps,
            profile=args.train_profile,
            board_size=args.board_size,
        )
        launcher.wait_for_jobs([train_job])

        # 3. Evaluation
        eval_job = launcher.submit_evaluation_job(
            iteration=args.iteration,
            num_games=args.eval_games,
            profile=args.eval_profile,
            board_size=args.board_size,
        )
        launcher.wait_for_jobs([eval_job])

        logger.info(f"Iteration {args.iteration} complete")


if __name__ == "__main__":
    main()
