"""Self-play actor for distributed AlphaZero training.

Actors run MCTS self-play games in separate processes, fetching the latest
network weights from shared storage and sending game trajectories to the
replay buffer.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch.multiprocessing as mp

from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig, SimpleMCTS
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    GameTrajectory,
    SharedStorage,
    TrainingExample,
    WeightSynchronizer,
)
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.utils.constants import DISTRIBUTED, GAME, REWARDS
from alpha_zero.utils.reward_shaping import RewardShaper, RewardShapingConfig

logger = logging.getLogger(__name__)


@dataclass
class ActorConfig:
    """Configuration for self-play actor."""

    # Actor identity
    actor_id: int = 0

    # Environment settings
    board_size: int = GAME.DEFAULT_BOARD_SIZE
    max_turns: int = GAME.DEFAULT_MAX_TURNS

    # Network settings
    num_res_blocks: int = 6
    num_filters: int = 64

    # MCTS settings
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30

    # Self-play settings
    games_per_update: int = 1  # Games to play before checking for new weights
    max_games: int = 0  # 0 = unlimited

    # Performance settings
    device: str = "cpu"
    weight_update_interval: float = 1.0  # Seconds between weight checks

    # Timeout settings
    trajectory_submission_timeout: float = DISTRIBUTED.TRAJECTORY_SUBMISSION_TIMEOUT

    # Logging
    log_interval: int = 10  # Log every N games

    # Reward shaping settings
    use_reward_shaping: bool = True
    reward_shaping_weight: float = 0.5  # How much to weight shaped rewards vs outcome
    n_step_returns: int = 5  # N-step return for value targets (0 = full game)


@dataclass
class GameHistoryItem:
    """Single state in game history."""

    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    policy: np.ndarray
    player: int
    shaped_reward: float = 0.0  # Intermediate reward from reward shaping


class SelfPlayActor:
    """Self-play actor that generates training data.

    Runs MCTS games using the current network weights, sending
    completed game trajectories to the replay buffer.
    """

    def __init__(
        self,
        config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Initialize actor.

        Args:
            config: Actor configuration.
            shared_storage: Shared storage for model weights.
            replay_buffer: Replay buffer for trajectories.
        """
        self.config = config
        self.storage = shared_storage
        self.replay_buffer = replay_buffer

        # Create environment
        self.env = MicroCivEnv(
            board_size=config.board_size,
            max_turns=config.max_turns,
        )
        self.action_space_size = self.env.action_space.n

        # Create network (will load weights from storage)
        self.network: MicroCivNetwork | None = None
        self.mcts: Civ6MCTS | SimpleMCTS | None = None

        # Weight synchronizer
        self.weight_sync: WeightSynchronizer | None = None

        # Statistics
        self.games_played = 0
        self.total_examples = 0
        self.start_time = 0.0

        # Control flags
        self._running = False
        self._game_id_counter = 0

        logger.info(f"Actor {config.actor_id} initialized")

    def _init_network(self) -> None:
        """Initialize network and MCTS."""
        self.network = MicroCivNetwork(
            board_size=self.config.board_size,
            action_space_size=self.action_space_size,
            num_res_blocks=self.config.num_res_blocks,
            num_filters=self.config.num_filters,
        ).to(self.config.device)
        self.network.eval()

        # Initialize weight synchronizer
        self.weight_sync = WeightSynchronizer(
            self.storage,
            update_interval=self.config.weight_update_interval,
        )

        # Try to load initial weights
        self._update_weights()

        # Initialize MCTS
        self._update_mcts()

    def _update_weights(self) -> bool:
        """Update network weights from shared storage.

        Returns:
            True if weights were updated.
        """
        if self.weight_sync is None or self.network is None:
            return False

        if self.weight_sync.should_update():
            if self.weight_sync.update_model(self.network):
                logger.debug(
                    f"Actor {self.config.actor_id}: Updated to model "
                    f"v{self.weight_sync.get_current_version()}",
                )
                return True
        return False

    def _update_mcts(self) -> None:
        """Update MCTS with current network."""
        if self.network is None:
            return

        version = self.weight_sync.get_current_version() if self.weight_sync else 0

        if version == 0:
            # No trained weights yet, use SimpleMCTS
            self.mcts = SimpleMCTS(
                num_simulations=self.config.num_simulations,
                c_puct=self.config.c_puct,
            )
            logger.debug(f"Actor {self.config.actor_id}: Using SimpleMCTS (no weights)")
        else:
            # Use neural network MCTS
            mcts_config = MCTSConfig(
                num_simulations=self.config.num_simulations,
                c_puct=self.config.c_puct,
                dirichlet_alpha=self.config.dirichlet_alpha,
                dirichlet_epsilon=self.config.dirichlet_epsilon,
                temperature=self.config.temperature,
                temperature_threshold=self.config.temperature_threshold,
            )
            self.mcts = Civ6MCTS(
                self.network,
                mcts_config,
                device=self.config.device,
            )

    def play_game(self) -> GameTrajectory:
        """Play a single self-play game.

        Returns:
            Game trajectory with training examples.

        Enhanced with reward shaping:
        - Tracks intermediate rewards for better credit assignment
        - Computes n-step returns for value targets
        - Blends shaped rewards with final outcome
        """
        if self.mcts is None:
            raise RuntimeError("MCTS not initialized")

        game_history: list[GameHistoryItem] = []
        obs, _ = self.env.reset()

        # Initialize reward shaper if enabled
        reward_shaper = None
        if self.config.use_reward_shaping:
            shaping_config = RewardShapingConfig(
                enabled=True,
                gamma=REWARDS.GAMMA,
            )
            reward_shaper = RewardShaper(shaping_config, self.config.board_size)

        # Play until game ends
        while True:
            # Get MCTS policy
            action_probs, _ = self.mcts.search(self.env)

            # Compute shaped reward for this state
            shaped_reward = 0.0
            if reward_shaper is not None:
                shaped_reward = reward_shaper.shape_reward(
                    obs=obs,
                    reward=0.0,  # No terminal reward yet
                    terminated=False,
                )

            # Store state and policy
            game_history.append(
                GameHistoryItem(
                    spatial=obs["spatial"].copy(),
                    vector=obs["vector"].copy(),
                    action_mask=obs["action_mask"].copy(),
                    policy=action_probs.copy(),
                    player=self.env.current_player,
                    shaped_reward=shaped_reward,
                ),
            )

            # Select action with temperature
            temperature = (
                self.config.temperature
                if self.env.turn < self.config.temperature_threshold
                else 0.0
            )
            action = self.mcts.select_action(action_probs, temperature)

            # Take action
            obs, _, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                break

        # Create training examples with value targets
        winner = info.get("winner")
        examples = self._create_training_examples(game_history, winner)

        # Create trajectory
        self._game_id_counter += 1
        trajectory = GameTrajectory(
            examples=examples,
            game_id=self._game_id_counter,
            actor_id=self.config.actor_id,
            game_length=len(game_history),
            winner=winner,
        )

        return trajectory

    def _create_training_examples(
        self,
        game_history: list[GameHistoryItem],
        winner: int | None,
    ) -> list[TrainingExample]:
        """Create training examples with enhanced value targets.

        Args:
            game_history: List of game history items.
            winner: Winner of the game (0, 1, or None for draw).

        Returns:
            List of training examples with computed value targets.
        """
        examples = []
        _n_steps = len(game_history)  # Reserved for future n-step return
        gamma = REWARDS.GAMMA

        for i, hist in enumerate(game_history):
            # Base value from game outcome
            if winner is None:
                outcome_value = 0.0  # Draw
            elif winner == hist.player:
                outcome_value = 1.0  # Win
            else:
                outcome_value = -1.0  # Loss

            # Compute value target
            if self.config.use_reward_shaping and self.config.n_step_returns > 0:
                # N-step return: sum of discounted shaped rewards + bootstrapped value
                value = self._compute_n_step_value(
                    game_history,
                    i,
                    outcome_value,
                    gamma,
                    self.config.n_step_returns,
                )
                # Blend with outcome
                value = (
                    self.config.reward_shaping_weight * value
                    + (1 - self.config.reward_shaping_weight) * outcome_value
                )
            else:
                # Traditional: just use game outcome
                value = outcome_value

            examples.append(
                TrainingExample(
                    spatial=hist.spatial,
                    vector=hist.vector,
                    action_mask=hist.action_mask,
                    policy=hist.policy,
                    value=value,
                ),
            )

        return examples

    def _compute_n_step_value(
        self,
        game_history: list[GameHistoryItem],
        start_idx: int,
        final_value: float,
        gamma: float,
        n_steps: int,
    ) -> float:
        """Compute n-step return value for a state.

        Args:
            game_history: Full game history.
            start_idx: Index of the state.
            final_value: Final game outcome value.
            gamma: Discount factor.
            n_steps: Number of steps to look ahead.

        Returns:
            N-step return value.
        """
        total_return = 0.0
        discount = 1.0
        end_idx = min(start_idx + n_steps, len(game_history))

        # Sum discounted shaped rewards
        for i in range(start_idx, end_idx):
            total_return += discount * game_history[i].shaped_reward
            discount *= gamma

        # Bootstrap with final value
        total_return += discount * final_value

        return total_return

    def run(self) -> None:
        """Main actor loop.

        Plays games and submits trajectories until stopped.
        """
        self._running = True
        self.start_time = time.time()

        # Initialize network
        self._init_network()

        logger.info(f"Actor {self.config.actor_id}: Starting self-play")

        games_since_update = 0

        while self._running and not self.storage.should_stop():
            # Check for max games limit
            if self.config.max_games > 0 and self.games_played >= self.config.max_games:
                logger.info(
                    f"Actor {self.config.actor_id}: Reached max games ({self.config.max_games})",
                )
                break

            # Play a game
            try:
                trajectory = self.play_game()

                # Submit to replay buffer
                if self.replay_buffer.submit_trajectory(
                    trajectory, timeout=self.config.trajectory_submission_timeout
                ):
                    self.games_played += 1
                    self.total_examples += len(trajectory.examples)
                    games_since_update += 1

                    # Update shared storage statistics
                    self.storage.increment_games_played(1)
                    self.storage.increment_examples(len(trajectory.examples))

                # Check for weight updates
                if games_since_update >= self.config.games_per_update:
                    if self._update_weights():
                        self._update_mcts()
                    games_since_update = 0

                # Logging
                if self.games_played % self.config.log_interval == 0:
                    elapsed = time.time() - self.start_time
                    games_per_sec = self.games_played / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Actor {self.config.actor_id}: "
                        f"{self.games_played} games, "
                        f"{self.total_examples} examples, "
                        f"{games_per_sec:.2f} games/sec, "
                        f"model v{self.weight_sync.get_current_version() if self.weight_sync else 0}",
                    )

            except Exception as e:
                logger.error(f"Actor {self.config.actor_id}: Error in game: {e}")
                import traceback

                traceback.print_exc()
                time.sleep(1.0)  # Brief pause before retry

        self._running = False
        logger.info(
            f"Actor {self.config.actor_id}: Stopped after {self.games_played} games",
        )

    def stop(self) -> None:
        """Signal the actor to stop."""
        self._running = False

    def is_running(self) -> bool:
        """Check if actor is running."""
        return self._running

    def get_statistics(self) -> dict:
        """Get actor statistics.

        Returns:
            Dictionary with statistics.
        """
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        return {
            "actor_id": self.config.actor_id,
            "games_played": self.games_played,
            "total_examples": self.total_examples,
            "elapsed_time": elapsed,
            "games_per_second": self.games_played / elapsed if elapsed > 0 else 0,
            "model_version": (self.weight_sync.get_current_version() if self.weight_sync else 0),
            "is_running": self._running,
        }


def actor_process(
    actor_id: int,
    config_dict: dict[str, Any],
    storage: SharedStorage,
    replay_buffer: DistributedReplayBuffer,
) -> None:
    """Entry point for actor subprocess.

    Args:
        actor_id: Actor ID.
        config_dict: Configuration dictionary.
        storage: Shared storage instance.
        replay_buffer: Replay buffer instance.
    """
    # Set up logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [Actor-{actor_id}] [%(levelname)s] %(message)s",
    )

    # Handle signals for graceful shutdown
    def signal_handler(signum: int, frame: object | None) -> None:
        logger.info(f"Actor {actor_id}: Received signal {signum}")
        storage.request_stop()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Create config
    config = ActorConfig(actor_id=actor_id, **config_dict)

    # Create and run actor
    actor = SelfPlayActor(config, storage, replay_buffer)

    try:
        actor.run()
    except Exception as e:
        logger.error(f"Actor {actor_id} crashed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        logger.info(f"Actor {actor_id}: Process exiting")


class ActorPool:
    """Manages a pool of actor processes.

    Spawns and monitors multiple actor processes for parallel self-play.
    """

    def __init__(
        self,
        num_actors: int,
        actor_config: ActorConfig,
        storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ) -> None:
        """Initialize actor pool.

        Args:
            num_actors: Number of actors to spawn.
            actor_config: Base configuration for actors.
            storage: Shared storage instance.
            replay_buffer: Replay buffer instance.
        """
        self.num_actors = num_actors
        self.base_config = actor_config
        self.storage = storage
        self.replay_buffer = replay_buffer

        self.processes: list[mp.Process] = []
        self._started = False

        logger.info(f"ActorPool initialized with {num_actors} actors")

    def start(self) -> None:
        """Start all actor processes."""
        if self._started:
            logger.warning("ActorPool already started")
            return

        # Create config dict (excluding actor_id which we set per-actor)
        config_dict = {
            "board_size": self.base_config.board_size,
            "max_turns": self.base_config.max_turns,
            "num_res_blocks": self.base_config.num_res_blocks,
            "num_filters": self.base_config.num_filters,
            "num_simulations": self.base_config.num_simulations,
            "c_puct": self.base_config.c_puct,
            "dirichlet_alpha": self.base_config.dirichlet_alpha,
            "dirichlet_epsilon": self.base_config.dirichlet_epsilon,
            "temperature": self.base_config.temperature,
            "temperature_threshold": self.base_config.temperature_threshold,
            "games_per_update": self.base_config.games_per_update,
            "max_games": self.base_config.max_games,
            "device": self.base_config.device,
            "weight_update_interval": self.base_config.weight_update_interval,
            "log_interval": self.base_config.log_interval,
        }

        # Spawn actor processes
        for i in range(self.num_actors):
            process = mp.Process(
                target=actor_process,
                args=(i, config_dict, self.storage, self.replay_buffer),
                name=f"Actor-{i}",
            )
            process.start()
            self.processes.append(process)
            logger.info(f"Started Actor-{i} (pid={process.pid})")

        self._started = True
        logger.info(f"Started {self.num_actors} actor processes")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop all actor processes.

        Args:
            timeout: Maximum time to wait for graceful shutdown.
        """
        if not self._started:
            return

        logger.info("Stopping actor pool...")

        # Signal stop
        self.storage.request_stop()

        # Wait for processes to finish
        deadline = time.time() + timeout

        for process in self.processes:
            remaining = max(0, deadline - time.time())
            process.join(timeout=remaining)

            if process.is_alive():
                logger.warning(f"Actor {process.name} did not stop, terminating")
                process.terminate()
                process.join(timeout=1.0)

                if process.is_alive():
                    logger.error(f"Actor {process.name} did not terminate, killing")
                    process.kill()

        self.processes.clear()
        self._started = False
        logger.info("Actor pool stopped")

    def is_running(self) -> bool:
        """Check if any actors are still running."""
        return any(p.is_alive() for p in self.processes)

    def get_running_count(self) -> int:
        """Get number of running actors."""
        return sum(1 for p in self.processes if p.is_alive())

    def restart_dead_actors(self) -> int:
        """Restart any actors that have crashed.

        Returns:
            Number of actors restarted.
        """
        if not self._started or self.storage.should_stop():
            return 0

        restarted = 0
        config_dict = {
            "board_size": self.base_config.board_size,
            "max_turns": self.base_config.max_turns,
            "num_res_blocks": self.base_config.num_res_blocks,
            "num_filters": self.base_config.num_filters,
            "num_simulations": self.base_config.num_simulations,
            "c_puct": self.base_config.c_puct,
            "dirichlet_alpha": self.base_config.dirichlet_alpha,
            "dirichlet_epsilon": self.base_config.dirichlet_epsilon,
            "temperature": self.base_config.temperature,
            "temperature_threshold": self.base_config.temperature_threshold,
            "games_per_update": self.base_config.games_per_update,
            "max_games": self.base_config.max_games,
            "device": self.base_config.device,
            "weight_update_interval": self.base_config.weight_update_interval,
            "log_interval": self.base_config.log_interval,
        }

        for i, process in enumerate(self.processes):
            if not process.is_alive():
                logger.warning(f"Actor-{i} died, restarting")
                new_process = mp.Process(
                    target=actor_process,
                    args=(i, config_dict, self.storage, self.replay_buffer),
                    name=f"Actor-{i}",
                )
                new_process.start()
                self.processes[i] = new_process
                restarted += 1

        return restarted
