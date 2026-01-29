"""Training coordinator for distributed AlphaZero.

Manages the overall training process, including:
- Spawning and monitoring actors and learner
- Checkpointing and evaluation
- Progress tracking and statistics
- Graceful shutdown handling
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.distributed.actor import ActorConfig, ActorPool
from alpha_zero.distributed.learner import Learner, LearnerConfig
from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    ReplayBufferConfig,
    SharedStorage,
    SharedStorageConfig,
)
from alpha_zero.envs.microciv import MicroCivEnv

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorConfig:
    """Configuration for the training coordinator."""

    # Environment
    board_size: int = 16
    max_turns: int = 200

    # Network
    num_res_blocks: int = 6
    num_filters: int = 64

    # MCTS (for actors)
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30

    # Distributed settings
    num_actors: int = 64
    games_per_checkpoint: int = 500

    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    lr_decay_steps: list[int] = field(default_factory=lambda: [100000, 300000, 500000])
    lr_decay_rate: float = 0.1

    # Replay buffer
    replay_buffer_size: int = 500_000
    min_replay_size: int = 10_000

    # Training limits
    max_games: int = 0  # 0 = unlimited
    max_training_steps: int = 0  # 0 = unlimited

    # Checkpointing
    checkpoint_dir: str = "checkpoints/civ6"
    log_dir: str = "logs/civ6"
    save_interval: int = 1000

    # Evaluation
    eval_interval: int = 500  # Games between evaluations
    eval_games: int = 20  # Games per evaluation

    # Monitoring
    monitor_interval: float = 10.0  # Seconds between status updates
    restart_dead_actors: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_yaml(cls, path: str) -> CoordinatorConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            CoordinatorConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        # Flatten nested config
        flat_config = {}

        # Environment
        env_cfg = data.get('environment', {})
        flat_config['board_size'] = env_cfg.get('board_size', 16)
        flat_config['max_turns'] = env_cfg.get('max_turns', 200)

        # Network
        net_cfg = data.get('network', {})
        flat_config['num_res_blocks'] = net_cfg.get('num_res_blocks', 6)
        flat_config['num_filters'] = net_cfg.get('num_filters', 64)

        # MCTS
        mcts_cfg = data.get('mcts', {})
        flat_config['num_simulations'] = mcts_cfg.get('num_simulations', 50)
        flat_config['c_puct'] = mcts_cfg.get('c_puct', 1.5)
        flat_config['dirichlet_alpha'] = mcts_cfg.get('dirichlet_alpha', 0.3)
        flat_config['dirichlet_epsilon'] = mcts_cfg.get('dirichlet_epsilon', 0.25)
        flat_config['temperature'] = mcts_cfg.get('temperature', 1.0)
        flat_config['temperature_threshold'] = mcts_cfg.get('temperature_threshold', 30)

        # Training
        train_cfg = data.get('training', {})
        flat_config['batch_size'] = train_cfg.get('batch_size', 256)
        flat_config['learning_rate'] = train_cfg.get('learning_rate', 0.001)
        flat_config['weight_decay'] = train_cfg.get('weight_decay', 0.0001)
        flat_config['gradient_clip'] = train_cfg.get('gradient_clip', 1.0)
        flat_config['lr_decay_steps'] = train_cfg.get('lr_decay_steps', [100000, 300000, 500000])
        flat_config['lr_decay_rate'] = train_cfg.get('lr_decay_rate', 0.1)
        flat_config['replay_buffer_size'] = train_cfg.get('replay_buffer_size', 500000)
        flat_config['min_replay_size'] = train_cfg.get('min_replay_size', 10000)

        # Self-play
        selfplay_cfg = data.get('selfplay', {})
        flat_config['num_actors'] = selfplay_cfg.get('num_actors', 64)
        flat_config['games_per_checkpoint'] = selfplay_cfg.get('games_per_checkpoint', 500)
        flat_config['save_interval'] = selfplay_cfg.get('save_interval', 100)

        # MicroCiv overrides (Phase 1)
        microciv_cfg = data.get('microciv', {})
        if microciv_cfg:
            flat_config['board_size'] = microciv_cfg.get('board_size', flat_config['board_size'])
            flat_config['num_res_blocks'] = microciv_cfg.get('num_res_blocks', flat_config['num_res_blocks'])
            flat_config['num_filters'] = microciv_cfg.get('num_filters', flat_config['num_filters'])
            flat_config['num_simulations'] = microciv_cfg.get('num_simulations', flat_config['num_simulations'])
            flat_config['batch_size'] = microciv_cfg.get('batch_size', flat_config['batch_size'])

        return cls(**flat_config)


class TrainingCoordinator:
    """Coordinates the distributed training process.

    Manages:
    - Shared storage for model weights
    - Distributed replay buffer
    - Actor pool for self-play
    - Learner for network training
    - Evaluation and checkpointing
    """

    def __init__(self, config: CoordinatorConfig):
        """Initialize coordinator.

        Args:
            config: Coordinator configuration.
        """
        self.config = config

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        # Initialize shared storage
        storage_config = SharedStorageConfig(
            compress_weights=True,
            track_statistics=True,
        )
        self.shared_storage = SharedStorage(config=storage_config)

        # Initialize replay buffer
        buffer_config = ReplayBufferConfig(
            max_size=config.replay_buffer_size,
            min_size=config.min_replay_size,
            sample_batch_size=config.batch_size,
        )
        self.replay_buffer = DistributedReplayBuffer(config=buffer_config)

        # Create actor config
        self.actor_config = ActorConfig(
            board_size=config.board_size,
            max_turns=config.max_turns,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            temperature=config.temperature,
            temperature_threshold=config.temperature_threshold,
            games_per_update=1,
            max_games=0,  # Unlimited
            device='cpu',  # Actors on CPU
        )

        # Create learner config
        self.learner_config = LearnerConfig(
            board_size=config.board_size,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_clip=config.gradient_clip,
            lr_decay_steps=config.lr_decay_steps,
            lr_decay_rate=config.lr_decay_rate,
            min_buffer_size=config.min_replay_size,
            max_training_steps=config.max_training_steps,
            checkpoint_dir=config.checkpoint_dir,
            checkpoint_interval=config.save_interval,
            device=config.device,
        )

        # Actor pool and learner
        self.actor_pool: ActorPool | None = None
        self.learner: Learner | None = None

        # Evaluation network
        self.eval_network: MicroCivNetwork | None = None

        # Statistics
        self.start_time = 0.0
        self.last_eval_games = 0
        self._running = False
        self._shutdown_requested = False

        logger.info(
            f"TrainingCoordinator initialized: "
            f"{config.num_actors} actors, "
            f"buffer size {config.replay_buffer_size}"
        )

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_requested = True
            self.shared_storage.request_stop()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _init_evaluation_network(self) -> None:
        """Initialize network for evaluation."""
        env = MicroCivEnv(board_size=self.config.board_size)
        self.eval_network = MicroCivNetwork(
            board_size=self.config.board_size,
            action_space_size=env.action_space.n,
            num_res_blocks=self.config.num_res_blocks,
            num_filters=self.config.num_filters,
        ).to(self.config.device)
        self.eval_network.eval()

    def _update_eval_network(self) -> bool:
        """Update evaluation network with latest weights.

        Returns:
            True if weights were updated.
        """
        if self.eval_network is None:
            return False

        weights = self.shared_storage.get_model_weights()
        if weights is None:
            return False

        try:
            self.eval_network.load_state_dict(weights)
            return True
        except Exception as e:
            logger.error(f"Failed to load eval weights: {e}")
            return False

    def evaluate(self, num_games: int = 20) -> dict[str, float]:
        """Evaluate current network against random player.

        Args:
            num_games: Number of evaluation games.

        Returns:
            Evaluation results.
        """
        if self.eval_network is None:
            self._init_evaluation_network()

        self._update_eval_network()

        env = MicroCivEnv(
            board_size=self.config.board_size,
            max_turns=self.config.max_turns,
        )

        mcts_config = MCTSConfig(
            num_simulations=self.config.num_simulations // 2,
            c_puct=self.config.c_puct,
        )
        mcts = Civ6MCTS(self.eval_network, mcts_config, device=self.config.device)

        wins = 0
        rng = np.random.default_rng()

        for game_idx in range(num_games):
            obs, _ = env.reset()
            network_player = game_idx % 2

            while True:
                if env.current_player == network_player:
                    action_probs, _ = mcts.search(env, add_noise=False)
                    action = mcts.select_action(action_probs, temperature=0.0)
                else:
                    mask = env.get_valid_actions()
                    valid = np.where(mask > 0)[0]
                    action = rng.choice(valid) if len(valid) > 0 else 0

                obs, _, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    winner = info.get('winner')
                    if winner == network_player:
                        wins += 1
                    break

        win_rate = wins / num_games

        # Store evaluation result
        self.shared_storage.add_eval_entry({
            'games_played': self.shared_storage.get_games_played(),
            'training_step': self.shared_storage.get_training_step(),
            'win_rate': win_rate,
        })

        return {
            'win_rate': win_rate,
            'games': num_games,
        }

    def _log_status(self) -> None:
        """Log current training status."""
        storage_stats = self.shared_storage.get_statistics()
        buffer_stats = self.replay_buffer.get_statistics()

        elapsed = time.time() - self.start_time
        games = storage_stats['games_played']
        games_per_hour = (games / elapsed) * 3600 if elapsed > 0 else 0

        status_lines = [
            "=== Training Status ===",
            f"Elapsed: {elapsed / 60:.1f} min",
            f"Games: {games} ({games_per_hour:.0f}/hour)",
            f"Training step: {storage_stats['training_step']}",
            f"Model version: {storage_stats['model_version']}",
            f"Buffer: {buffer_stats['buffer_size']} examples",
            f"Actors running: {self.actor_pool.get_running_count() if self.actor_pool else 0}",
        ]

        # Add latest loss if available
        loss_history = storage_stats.get('loss_history', [])
        if loss_history:
            latest = loss_history[-1]
            status_lines.append(
                f"Loss: {latest.get('total_loss', 0):.4f} "
                f"(p={latest.get('policy_loss', 0):.4f}, "
                f"v={latest.get('value_loss', 0):.4f})"
            )

        # Add latest eval if available
        eval_history = storage_stats.get('eval_history', [])
        if eval_history:
            latest = eval_history[-1]
            status_lines.append(f"Win rate vs random: {latest.get('win_rate', 0):.1%}")

        logger.info("\n".join(status_lines))

    def run(self) -> None:
        """Run the distributed training process.

        This is the main entry point that:
        1. Starts the actor pool
        2. Runs the learner (in the main process)
        3. Monitors progress and performs evaluation
        4. Handles graceful shutdown
        """
        self._running = True
        self.start_time = time.time()
        self._setup_signal_handlers()

        logger.info("Starting distributed training...")

        try:
            # Initialize actor pool
            self.actor_pool = ActorPool(
                num_actors=self.config.num_actors,
                actor_config=self.actor_config,
                storage=self.shared_storage,
                replay_buffer=self.replay_buffer,
            )

            # Initialize learner
            self.learner = Learner(
                config=self.learner_config,
                shared_storage=self.shared_storage,
                replay_buffer=self.replay_buffer,
            )

            # Initialize evaluation network
            self._init_evaluation_network()

            # Start actors
            self.actor_pool.start()
            logger.info(f"Started {self.config.num_actors} actors")

            # Main training loop
            last_status_time = time.time()
            last_eval_games = 0

            while self._running and not self._shutdown_requested:
                # Process replay buffer trajectories
                self.replay_buffer.process_pending_trajectories()

                # Check if buffer is ready
                if not self.replay_buffer.is_ready():
                    logger.info(
                        f"Waiting for buffer: "
                        f"{self.replay_buffer.size()}/{self.config.min_replay_size}"
                    )
                    time.sleep(1.0)
                    continue

                # Training steps
                for _ in range(100):  # Batch of training steps
                    if self._shutdown_requested:
                        break

                    losses = self.learner.train_step()
                    if losses is None:
                        break

                # Status logging
                current_time = time.time()
                if current_time - last_status_time >= self.config.monitor_interval:
                    self._log_status()
                    last_status_time = current_time

                    # Restart dead actors
                    if self.config.restart_dead_actors and self.actor_pool:
                        restarted = self.actor_pool.restart_dead_actors()
                        if restarted > 0:
                            logger.info(f"Restarted {restarted} dead actors")

                # Evaluation
                current_games = self.shared_storage.get_games_played()
                if current_games - last_eval_games >= self.config.eval_interval:
                    logger.info("Running evaluation...")
                    eval_result = self.evaluate(self.config.eval_games)
                    logger.info(f"Evaluation: win rate = {eval_result['win_rate']:.1%}")
                    last_eval_games = current_games

                # Check termination conditions
                if self.config.max_games > 0:
                    if current_games >= self.config.max_games:
                        logger.info(f"Reached max games ({self.config.max_games})")
                        break

                if self.config.max_training_steps > 0:
                    if self.learner.training_step >= self.config.max_training_steps:
                        logger.info(
                            f"Reached max training steps "
                            f"({self.config.max_training_steps})"
                        )
                        break

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        logger.info("Shutting down coordinator...")

        self._running = False
        self.shared_storage.request_stop()

        # Stop actors
        if self.actor_pool:
            logger.info("Stopping actors...")
            self.actor_pool.stop(timeout=10.0)

        # Final checkpoint
        if self.learner and self.learner.training_step > 0:
            logger.info("Saving final checkpoint...")
            self.learner._save_checkpoint(self.learner.training_step)

        # Cleanup shared resources
        self.replay_buffer.shutdown()
        self.shared_storage.shutdown()

        # Log final statistics
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        games = self.shared_storage.get_games_played()

        logger.info(
            f"Training complete: "
            f"{games} games, "
            f"{self.learner.training_step if self.learner else 0} steps, "
            f"{elapsed / 60:.1f} minutes"
        )


def main() -> None:
    """Main entry point for distributed training."""
    parser = argparse.ArgumentParser(description="Distributed Civ6 AlphaZero Training")

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/civ6_config.yaml",
        help="Path to configuration file",
    )

    # Overrides
    parser.add_argument("--num_actors", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_simulations", type=int, default=None)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--max_training_steps", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    # Quick test mode
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Quick test with minimal settings",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Load config
    if os.path.exists(args.config):
        config = CoordinatorConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = CoordinatorConfig()
        logger.info("Using default configuration")

    # Apply overrides
    if args.num_actors is not None:
        config.num_actors = args.num_actors
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.num_simulations is not None:
        config.num_simulations = args.num_simulations
    if args.max_games is not None:
        config.max_games = args.max_games
    if args.max_training_steps is not None:
        config.max_training_steps = args.max_training_steps
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.device is not None:
        config.device = args.device

    # Dry run settings
    if args.dry_run:
        config.num_actors = 2
        config.min_replay_size = 100
        config.max_games = 10
        config.max_training_steps = 50
        config.eval_interval = 5
        config.eval_games = 2
        config.monitor_interval = 5.0
        config.save_interval = 25
        logger.info("Dry run mode: using minimal settings")

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Actors: {config.num_actors}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  MCTS simulations: {config.num_simulations}")
    logger.info(f"  Buffer size: {config.replay_buffer_size}")
    logger.info(f"  Device: {config.device}")

    # Run training
    coordinator = TrainingCoordinator(config)
    coordinator.run()


if __name__ == "__main__":
    main()
