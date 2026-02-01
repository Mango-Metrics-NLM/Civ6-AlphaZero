"""Cloud-native self-play worker for Vertex AI.

Generates game trajectories using MCTS-guided self-play and
stores them in GCS for training.

Usage:
    python -m alpha_zero.cloud.selfplay_worker \\
        --num-games=100 \\
        --iteration=5 \\
        --worker-id=0

Environment Variables:
    CIV6_GCS_BUCKET: GCS bucket for storage
    CIV6_GCS_PREFIX: Path prefix within bucket
    CIV6_ITERATION: Training iteration
    CIV6_WORKER_ID: Worker identifier
    CIV6_NUM_GAMES: Number of games to play
    CIV6_NUM_SIMULATIONS: MCTS simulations per move
    CIV6_DEVICE: PyTorch device (cuda/cpu)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import torch

from alpha_zero.cloud.preemption import PreemptionHandler, register_preemption_handler
from alpha_zero.storage import StorageConfig, create_storage_backend
from alpha_zero.storage.models import ModelMetadata

if TYPE_CHECKING:
    from alpha_zero.core.civ6_network import MicroCivNetwork
    from alpha_zero.distributed.shared_storage import GameTrajectory
    from alpha_zero.storage.backend import StorageBackend

logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play worker.

    Attributes:
        num_games: Number of games to play.
        iteration: Current training iteration.
        worker_id: Unique worker identifier.
        num_simulations: MCTS simulations per move.
        c_puct: MCTS exploration constant.
        dirichlet_alpha: Dirichlet noise alpha parameter.
        dirichlet_epsilon: Dirichlet noise weight.
        temperature: Move selection temperature.
        temperature_drop_move: Move to drop temperature to near-zero.
        max_moves: Maximum moves per game.
        board_size: Game board size.
        device: PyTorch device.
        max_staleness: Maximum model staleness (iterations behind).
    """

    num_games: int = 100
    iteration: int = 0
    worker_id: int = 0
    num_simulations: int = 800
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_drop_move: int = 30
    max_moves: int = 200
    board_size: int = 8
    device: str = "cuda"
    max_staleness: int = 2

    @classmethod
    def from_env(cls) -> SelfPlayConfig:
        """Create configuration from environment variables."""
        return cls(
            num_games=int(os.environ.get("CIV6_NUM_GAMES", "100")),
            iteration=int(os.environ.get("CIV6_ITERATION", "0")),
            worker_id=int(os.environ.get("CIV6_WORKER_ID", "0")),
            num_simulations=int(os.environ.get("CIV6_NUM_SIMULATIONS", "800")),
            c_puct=float(os.environ.get("CIV6_MCTS_C_PUCT", "1.5")),
            dirichlet_alpha=float(os.environ.get("CIV6_MCTS_DIRICHLET_ALPHA", "0.3")),
            dirichlet_epsilon=float(os.environ.get("CIV6_MCTS_DIRICHLET_EPSILON", "0.25")),
            temperature=float(os.environ.get("CIV6_TEMPERATURE", "1.0")),
            temperature_drop_move=int(os.environ.get("CIV6_TEMPERATURE_DROP", "30")),
            max_moves=int(os.environ.get("CIV6_MAX_MOVES", "200")),
            board_size=int(os.environ.get("CIV6_BOARD_SIZE", "8")),
            device=os.environ.get("CIV6_DEVICE", "cuda"),
        )


class CloudSelfPlayWorker:
    """Self-play worker for cloud deployment.

    Generates game trajectories using MCTS-guided self-play.
    Handles:
    - Loading champion model from storage
    - Running self-play games
    - Storing trajectories to GCS
    - Graceful preemption handling

    Attributes:
        config: Self-play configuration.
        storage: Storage backend for models and trajectories.
        preemption_handler: Handler for VM preemption.
    """

    def __init__(
        self,
        config: SelfPlayConfig,
        storage: StorageBackend,
        preemption_handler: PreemptionHandler | None = None,
    ) -> None:
        """Initialize self-play worker.

        Args:
            config: Self-play configuration.
            storage: Storage backend.
            preemption_handler: Optional preemption handler.
        """
        self.config = config
        self.storage = storage
        self.preemption_handler = preemption_handler

        # State
        self._model: MicroCivNetwork | None = None
        self._model_version: int = 0
        self._games_completed: int = 0
        self._current_trajectory: GameTrajectory | None = None

        # Register cleanup callback
        if self.preemption_handler:
            self.preemption_handler.add_cleanup_callback(self._cleanup_on_preemption)

        logger.info(
            f"CloudSelfPlayWorker initialized "
            f"(worker={config.worker_id}, iteration={config.iteration})"
        )

    def _cleanup_on_preemption(self) -> None:
        """Cleanup callback for preemption."""
        logger.info("Running preemption cleanup...")

        # Flush current trajectory if in progress
        if self._current_trajectory and len(self._current_trajectory.examples) > 0:
            logger.info(
                f"Flushing partial trajectory "
                f"({len(self._current_trajectory.examples)} examples)"
            )
            try:
                self._save_trajectory(self._current_trajectory, partial=True)
            except Exception as e:
                logger.error(f"Failed to save partial trajectory: {e}")

        logger.info(
            f"Preemption cleanup complete. "
            f"Games completed: {self._games_completed}/{self.config.num_games}"
        )

    def _load_model(self) -> None:
        """Load the champion model from storage."""
        from alpha_zero.core.civ6_network import MicroCivNetwork

        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        try:
            state_dict, metadata = self.storage.load_champion(map_location=device)
            self._model_version = metadata.version

            # Create network and load weights
            self._model = MicroCivNetwork(
                board_size=self.config.board_size,
                action_space_size=356,  # MicroCiv action space
            ).to(device)
            self._model.load_state_dict(state_dict)
            self._model.eval()

            logger.info(f"Loaded champion model v{self._model_version}")

        except FileNotFoundError:
            # No champion yet - create fresh model
            logger.warning("No champion model found, using random initialization")
            self._model = MicroCivNetwork(
                board_size=self.config.board_size,
                action_space_size=356,
            ).to(device)
            self._model.eval()
            self._model_version = 0

    def _check_model_staleness(self) -> bool:
        """Check if model is too stale and needs refresh.

        Returns:
            True if model should be refreshed.
        """
        try:
            current_version = self.storage.get_champion_version()
            staleness = current_version - self._model_version

            if staleness > self.config.max_staleness:
                logger.warning(
                    f"Model is {staleness} versions behind "
                    f"(threshold: {self.config.max_staleness})"
                )
                return True
            return False

        except Exception as e:
            logger.warning(f"Failed to check model staleness: {e}")
            return False

    def _create_environment(self):
        """Create game environment."""
        from alpha_zero.envs.microciv import MicroCivEnv

        return MicroCivEnv(
            board_size=self.config.board_size,
            max_turns=self.config.max_moves,
        )

    def _create_mcts(self):
        """Create MCTS instance."""
        from alpha_zero.core.civ6_mcts import SimpleMCTS

        return SimpleMCTS(
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
        )

    def _play_game(self) -> GameTrajectory:
        """Play a single self-play game.

        Returns:
            Game trajectory with training examples.
        """
        from alpha_zero.distributed.shared_storage import GameTrajectory, TrainingExample

        env = self._create_environment()
        mcts = self._create_mcts()

        obs, _ = env.reset()
        game_history: list[TrainingExample] = []
        move_count = 0
        done = False

        while not done and move_count < self.config.max_moves:
            # Check for preemption
            if self.preemption_handler and self.preemption_handler.is_preempted:
                logger.info("Preemption detected, stopping game early")
                break

            # Get MCTS policy
            with torch.no_grad():
                policy = mcts.search(
                    env,
                    self._model,
                    obs,
                )

            # Apply temperature
            if move_count < self.config.temperature_drop_move:
                temp = self.config.temperature
            else:
                temp = 0.1  # Near-deterministic

            if temp > 0:
                policy_temp = np.power(policy, 1.0 / temp)
                policy_temp = policy_temp / policy_temp.sum()
            else:
                policy_temp = policy

            # Select action
            action_mask = obs.get("action_mask", np.ones(len(policy)))
            policy_masked = policy_temp * action_mask
            if policy_masked.sum() > 0:
                policy_masked = policy_masked / policy_masked.sum()
                action = np.random.choice(len(policy_masked), p=policy_masked)
            else:
                # Fallback to random valid action
                valid_actions = np.where(action_mask > 0)[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0

            # Store training example
            example = TrainingExample(
                spatial=obs["spatial"].copy(),
                vector=obs["vector"].copy(),
                action_mask=action_mask.copy(),
                policy=policy.copy(),
                value=0.0,  # Filled in after game ends
            )
            game_history.append(example)

            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            move_count += 1

        # Determine winner and assign values
        winner = info.get("winner")
        for i, example in enumerate(game_history):
            player_at_move = i % 2
            if winner is None:
                example.value = 0.0  # Draw
            elif player_at_move == winner:
                example.value = 1.0  # Win
            else:
                example.value = -1.0  # Loss

        trajectory = GameTrajectory(
            examples=game_history,
            game_id=move_count,  # Use move count as game ID
            actor_id=self.config.worker_id,
            game_length=move_count,
            winner=winner,
        )

        return trajectory

    def _save_trajectory(
        self,
        trajectory: GameTrajectory,
        partial: bool = False,
    ) -> str:
        """Save trajectory to storage.

        Args:
            trajectory: Game trajectory to save.
            partial: Whether this is a partial trajectory (from preemption).

        Returns:
            Storage path where trajectory was saved.
        """
        game_id = f"game_{uuid.uuid4().hex[:8]}"
        if partial:
            game_id = f"partial_{game_id}"

        path = self.storage.save_trajectory(
            trajectory=trajectory,
            iteration=self.config.iteration,
            worker_id=self.config.worker_id,
            game_id=game_id,
        )

        logger.debug(f"Saved trajectory {game_id} ({len(trajectory.examples)} examples)")
        return path

    def run(self) -> dict:
        """Run self-play and generate trajectories.

        Returns:
            Statistics dictionary with run results.
        """
        start_time = datetime.now(UTC)
        logger.info(
            f"Starting self-play: {self.config.num_games} games, "
            f"iteration {self.config.iteration}"
        )

        # Load model
        self._load_model()

        trajectories_saved = 0
        total_examples = 0
        total_moves = 0

        for game_num in range(self.config.num_games):
            # Check for preemption
            if self.preemption_handler and self.preemption_handler.is_preempted:
                logger.info("Preemption detected, stopping self-play")
                break

            # Check model staleness periodically
            if game_num > 0 and game_num % 10 == 0:
                if self._check_model_staleness():
                    logger.info("Refreshing stale model")
                    self._load_model()

            # Play game
            try:
                self._current_trajectory = self._play_game()
                self._save_trajectory(self._current_trajectory)

                trajectories_saved += 1
                total_examples += len(self._current_trajectory.examples)
                total_moves += self._current_trajectory.game_length

                self._games_completed += 1
                self._current_trajectory = None

                if (game_num + 1) % 10 == 0:
                    logger.info(
                        f"Progress: {game_num + 1}/{self.config.num_games} games, "
                        f"{total_examples} examples"
                    )

            except Exception as e:
                logger.error(f"Error in game {game_num}: {e}")
                continue

        # Calculate statistics
        elapsed = (datetime.now(UTC) - start_time).total_seconds()
        stats = {
            "worker_id": self.config.worker_id,
            "iteration": self.config.iteration,
            "games_completed": self._games_completed,
            "games_requested": self.config.num_games,
            "trajectories_saved": trajectories_saved,
            "total_examples": total_examples,
            "total_moves": total_moves,
            "avg_game_length": total_moves / max(trajectories_saved, 1),
            "elapsed_seconds": elapsed,
            "games_per_hour": self._games_completed / max(elapsed / 3600, 0.001),
            "model_version": self._model_version,
            "preempted": (
                self.preemption_handler.is_preempted
                if self.preemption_handler
                else False
            ),
        }

        logger.info(
            f"Self-play complete: {stats['games_completed']}/{stats['games_requested']} games, "
            f"{stats['total_examples']} examples, "
            f"{stats['games_per_hour']:.1f} games/hr"
        )

        return stats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Civ6 AlphaZero Self-Play Worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-games",
        type=int,
        default=int(os.environ.get("CIV6_NUM_GAMES", "100")),
        help="Number of games to play",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=int(os.environ.get("CIV6_ITERATION", "0")),
        help="Current training iteration",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=int(os.environ.get("CIV6_WORKER_ID", "0")),
        help="Worker identifier",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=int(os.environ.get("CIV6_NUM_SIMULATIONS", "800")),
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=int(os.environ.get("CIV6_BOARD_SIZE", "8")),
        help="Game board size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("CIV6_DEVICE", "cuda"),
        help="PyTorch device (cuda/cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for self-play worker."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    # Create configuration
    config = SelfPlayConfig(
        num_games=args.num_games,
        iteration=args.iteration,
        worker_id=args.worker_id,
        num_simulations=args.num_simulations,
        board_size=args.board_size,
        device=args.device,
    )

    if args.dry_run:
        logger.info("Dry run - configuration validated")
        logger.info(f"Config: {config}")
        return

    # Create storage backend
    storage_config = StorageConfig.from_env()
    storage = create_storage_backend(storage_config)

    # Register preemption handler
    preemption_handler = register_preemption_handler(grace_period=25.0)

    # Create and run worker
    worker = CloudSelfPlayWorker(
        config=config,
        storage=storage,
        preemption_handler=preemption_handler,
    )

    try:
        stats = worker.run()
        logger.info(f"Worker completed with stats: {stats}")
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        storage.close()


if __name__ == "__main__":
    main()
