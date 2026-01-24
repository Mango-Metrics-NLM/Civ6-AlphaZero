"""Main Training Orchestrator for Civ6 AlphaZero.

This module provides the unified training orchestrator that integrates all
components for AlphaZero-style training:
- Neural network training
- MCTS-guided self-play
- Curriculum learning
- Enhanced reward shaping
- RAG-augmented strategy analysis
- Checkpoint management
- W&B and TensorBoard logging

Usage:
    from alpha_zero.orchestrator import TrainingOrchestrator, OrchestratorConfig

    config = OrchestratorConfig.from_yaml("configs/orchestrator.yaml")
    orchestrator = TrainingOrchestrator(config)
    orchestrator.run()
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from alpha_zero.orchestrator.checkpointer import CheckpointManager
from alpha_zero.orchestrator.config import OrchestratorConfig
from alpha_zero.orchestrator.rag_integration import RAGIntegration
from alpha_zero.orchestrator.scheduler import HyperparameterScheduler
from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor
from alpha_zero.utils.factories import (
    create_curriculum_manager,
    create_environment,
    create_llm_client,
    create_mcts,
    create_network,
    create_replay_buffer,
    create_reward_shaper,
    create_wandb_logger,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Current state of training."""

    iteration: int = 0
    total_games: int = 0
    total_steps: int = 0
    best_win_rate: float = 0.0
    current_win_rate: float = 0.0
    recent_losses: list[float] = field(default_factory=list)
    curriculum_stage: str = "beginner"
    is_running: bool = False
    start_time: float = 0.0


@dataclass
class GameResult:
    """Result of a self-play game."""

    trajectory: list[dict[str, Any]]
    winner: int  # 0 = draw, 1 = player 1, -1 = player 2
    total_reward: float
    turns: int
    metrics: dict[str, float] = field(default_factory=dict)


class PaddedEnvironmentWrapper:
    """Wrapper that pads environment observations to expected network sizes.

    When curriculum learning uses smaller boards than the network was trained for,
    this wrapper pads observations and action spaces to match expected sizes.
    """

    def __init__(
        self,
        env: Any,
        expected_board_size: int,
        expected_action_size: int,
    ):
        """Initialize wrapper.

        Args:
            env: Base environment to wrap.
            expected_board_size: Board size the network expects.
            expected_action_size: Action space size the network expects.
        """
        self._env = env
        self._expected_board_size = expected_board_size
        self._expected_action_size = expected_action_size

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped environment."""
        return getattr(self._env, name)

    def _pad_spatial(self, spatial: np.ndarray) -> np.ndarray:
        """Pad spatial features to expected board size."""
        current_size = spatial.shape[-1]
        if current_size >= self._expected_board_size:
            return spatial

        pad_amount = self._expected_board_size - current_size
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        return np.pad(
            spatial,
            ((0, 0), (pad_left, pad_right), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )

    def _pad_actions(self, arr: np.ndarray) -> np.ndarray:
        """Pad action array to expected action space size."""
        current_size = arr.shape[0]
        if current_size >= self._expected_action_size:
            return arr

        padded = np.zeros(self._expected_action_size, dtype=arr.dtype)
        padded[:current_size] = arr
        return padded

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Get padded observation from environment."""
        obs = self._env._get_observation()
        if isinstance(obs, dict):
            padded_obs = dict(obs)
            if "spatial" in padded_obs:
                padded_obs["spatial"] = self._pad_spatial(padded_obs["spatial"])
            if "action_mask" in padded_obs:
                padded_obs["action_mask"] = self._pad_actions(padded_obs["action_mask"])
            return padded_obs
        return obs  # type: ignore[no-any-return]

    def get_valid_actions(self) -> np.ndarray:
        """Get padded action mask."""
        mask = self._env.get_valid_actions()
        return self._pad_actions(mask)

    def reset(self, **kwargs) -> tuple[Any, dict]:
        """Reset environment and return padded observation.

        Returns:
            Tuple of (padded_observation, info).
        """
        result = self._env.reset(**kwargs)
        # Handle gymnasium API: reset returns (obs, info)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        if isinstance(obs, dict):
            padded_obs = dict(obs)
            if "spatial" in padded_obs:
                padded_obs["spatial"] = self._pad_spatial(padded_obs["spatial"])
            if "action_mask" in padded_obs:
                padded_obs["action_mask"] = self._pad_actions(padded_obs["action_mask"])
            return padded_obs, info
        return obs, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Take step and return padded observation."""
        obs, reward, terminated, truncated, info = self._env.step(action)
        if isinstance(obs, dict):
            padded_obs = dict(obs)
            if "spatial" in padded_obs:
                padded_obs["spatial"] = self._pad_spatial(padded_obs["spatial"])
            return padded_obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info

    def clone(self) -> PaddedEnvironmentWrapper:
        """Clone the wrapper and underlying environment.

        This ensures MCTS simulations also use padded observations.
        """
        cloned_env = self._env.clone()
        return PaddedEnvironmentWrapper(
            cloned_env,
            self._expected_board_size,
            self._expected_action_size,
        )


class TrainingOrchestrator:
    """Unified training orchestrator for AlphaZero-style training.

    Coordinates all training components including:
    - Neural network and MCTS
    - Self-play game generation
    - Replay buffer and training loop
    - Curriculum learning
    - Reward shaping
    - RAG/LLM strategy analysis
    - Logging and checkpointing

    Attributes:
        config: Training configuration.
        state: Current training state.
    """

    def __init__(self, config: OrchestratorConfig):
        """Initialize training orchestrator.

        Args:
            config: Orchestrator configuration.
        """
        self.config = config
        self.state = TrainingState()
        self._setup_logging()

        # Will be initialized in initialize()
        self._network: torch.nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._mcts: Any = None
        self._env: Any = None
        self._replay_buffer: Any = None
        self._scheduler: HyperparameterScheduler | None = None
        self._checkpointer: CheckpointManager | None = None
        self._curriculum_manager: Any = None
        self._reward_shaper: Any = None
        self._rag: RAGIntegration | None = None
        self._strategy_advisor: StrategyAdvisor | None = None
        self._wandb_logger: Any = None
        self._device = "cpu"

        # Signal handling
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
            ],
        )

        # Add file handler if log_dir specified
        if self.config.log_dir:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / f"training_{self.config.experiment_name}.log"
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            logging.getLogger().addHandler(file_handler)

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received, finishing current iteration...")
        self._shutdown_requested = True

    def initialize(self) -> bool:
        """Initialize all training components.

        Returns:
            True if initialization successful.
        """
        logger.info(f"Initializing orchestrator: {self.config.experiment_name}")

        try:
            # Set device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self._device}")

            # Set random seed
            if self.config.random_seed is not None:
                torch.manual_seed(self.config.random_seed)
                np.random.seed(self.config.random_seed)
                logger.info(f"Random seed set to {self.config.random_seed}")

            # Initialize environment
            self._env = create_environment(self.config.environment)
            board_size = self.config.environment.board_size
            action_space_size = self._env.action_space.n

            # Store expected sizes for padding when curriculum uses smaller boards
            self._expected_board_size = board_size
            self._expected_action_size = action_space_size

            # Initialize network
            self._network = create_network(
                self.config.network,
                board_size=board_size,
                action_space_size=action_space_size,
                device=self._device,
            )

            # Initialize optimizer (learning rate comes from scheduler)
            self._optimizer = torch.optim.Adam(
                self._network.parameters(),
                lr=self.config.lr_schedule.initial_value,
                weight_decay=self.config.training.weight_decay,
            )

            # Initialize MCTS
            self._mcts = create_mcts(
                self.config.mcts,
                self._network,
                device=self._device,
            )

            # Initialize replay buffer
            self._replay_buffer = create_replay_buffer(
                max_size=self.config.training.replay_buffer_size,
                min_size=self.config.training.min_replay_size,
            )

            # Initialize scheduler
            self._scheduler = HyperparameterScheduler(
                lr_schedule=self.config.lr_schedule,
                batch_size_schedule=self.config.batch_size_schedule,
                temperature_schedule=self.config.temperature_schedule,
                exploration_schedule=self.config.exploration_schedule,
            )

            # Initialize checkpointer
            self._checkpointer = CheckpointManager(
                checkpoint_dir=self.config.checkpoint_dir,
                max_checkpoints=self.config.training.max_checkpoints,
            )

            # Initialize optional components
            self._curriculum_manager = create_curriculum_manager(
                self.config.curriculum,
                save_path=str(Path(self.config.checkpoint_dir) / "curriculum_state.json"),
            )

            self._reward_shaper = create_reward_shaper(
                self.config.rewards,
                board_size=board_size,
            )

            # Initialize RAG
            if self.config.rag.enabled:
                self._rag = RAGIntegration(self.config.rag)
                self._rag.initialize()

            # Initialize LLM/Strategy Advisor
            if self.config.llm.enabled and self.config.strategy_advisor.enabled:
                llm_client = create_llm_client(self.config.llm)
                if llm_client:
                    self._strategy_advisor = StrategyAdvisor(
                        llm_client,
                        self._rag,
                        self.config.strategy_advisor,
                    )

            # Initialize W&B
            if self.config.wandb.enabled:
                self._wandb_logger = create_wandb_logger(
                    self.config.wandb,
                    experiment_name=self.config.experiment_name,
                )

            # Resume from checkpoint if available
            if self.config.training.resume and self._checkpointer.has_checkpoint():
                self._resume_from_checkpoint()

            logger.info("Orchestrator initialization complete")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _resume_from_checkpoint(self) -> None:
        """Resume training from latest checkpoint."""
        assert self._checkpointer is not None
        assert self._network is not None
        assert self._optimizer is not None
        assert self._scheduler is not None

        checkpoint = self._checkpointer.load_latest()

        self._network.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state" in checkpoint:
            self._scheduler.set_step(checkpoint["scheduler_state"]["step"])

        if "training_state" in checkpoint:
            state_dict = checkpoint["training_state"]
            self.state.iteration = state_dict.get("iteration", 0)
            self.state.total_games = state_dict.get("total_games", 0)
            self.state.best_win_rate = state_dict.get("best_win_rate", 0.0)

        logger.info(f"Resumed from iteration {self.state.iteration}")

    def run(self) -> None:
        """Run the main training loop.

        Executes the full training process:
        1. Self-play to generate training data
        2. Train network on replay buffer
        3. Evaluate periodically
        4. Checkpoint periodically
        """
        if not self.initialize():
            logger.error("Failed to initialize, aborting training")
            return

        self.state.is_running = True
        self.state.start_time = time.time()

        logger.info(f"Starting training for {self.config.training.num_iterations} iterations")

        try:
            while self.state.iteration < self.config.training.num_iterations:
                if self._shutdown_requested:
                    logger.info("Shutdown requested, stopping training")
                    break

                self._training_iteration()
                self.state.iteration += 1

        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

        finally:
            self._cleanup()

    def _training_iteration(self) -> None:
        """Execute a single training iteration."""
        assert self._scheduler is not None
        assert self._network is not None

        iteration = self.state.iteration
        start_time = time.time()

        logger.info(f"=== Iteration {iteration} ===")

        # Update hyperparameters
        self._scheduler.set_step(iteration)
        self._update_optimizer_lr()

        # Get curriculum parameters if enabled
        env_params = None
        if self._curriculum_manager:
            env_params = self._curriculum_manager.get_environment_params()
            self.state.curriculum_stage = self._curriculum_manager.get_current_stage().name

        # Run self-play games
        games_this_iteration = self.config.training.games_per_iteration
        game_results = self._run_self_play(games_this_iteration, env_params)

        # Add to replay buffer
        for result in game_results:
            self._add_to_buffer(result)

        self.state.total_games += len(game_results)
        logger.info(f"Buffer size: {len(self._replay_buffer)}, min required: {self._replay_buffer.min_size}")

        # Train if buffer is ready
        train_metrics = {}
        if self._replay_buffer.is_ready():
            logger.info(f"Training for {self.config.training.num_epochs} epochs...")
            train_metrics = self._train_epochs(self.config.training.num_epochs)
            logger.info(f"Training done: loss={train_metrics.get('total_loss', 0):.4f}")
        else:
            logger.info("Buffer not ready, skipping training")

        # Update curriculum based on performance
        if self._curriculum_manager and game_results:
            win_rate = sum(1 for r in game_results if r.winner == 1) / len(game_results)
            self._curriculum_manager.record_game(won=win_rate > 0.5)

        # Evaluate periodically
        if iteration % self.config.training.evaluation_interval == 0:
            self._evaluate()

        # Checkpoint periodically
        if iteration % self.config.training.checkpoint_interval == 0:
            self._save_checkpoint()

        # Strategy analysis periodically
        if self._strategy_advisor and self._strategy_advisor.should_analyze(iteration):
            self._run_strategy_analysis(game_results)
            self._strategy_advisor.mark_analysis_done(iteration)

        # Log metrics
        iteration_time = time.time() - start_time
        self._log_iteration(iteration, game_results, train_metrics, iteration_time)

    def _run_self_play(
        self,
        num_games: int,
        env_params: dict[str, Any] | None = None,
    ) -> list[GameResult]:
        """Run self-play games to generate training data.

        Args:
            num_games: Number of games to play.
            env_params: Optional environment parameter overrides.

        Returns:
            List of game results.
        """
        assert self._network is not None

        results = []
        self._network.eval()

        logger.info(f"Starting self-play: {num_games} games")
        for game_idx in range(num_games):
            game_start = time.time()
            result = self._play_single_game(env_params)
            game_time = time.time() - game_start
            results.append(result)
            logger.info(
                f"Game {game_idx + 1}/{num_games}: "
                f"turns={result.turns}, winner={result.winner}, "
                f"reward={result.total_reward:.2f}, time={game_time:.1f}s"
            )

        return results

    def _play_single_game(
        self,
        env_params: dict[str, Any] | None = None,
    ) -> GameResult:
        """Play a single self-play game.

        Args:
            env_params: Optional environment parameter overrides.

        Returns:
            Game result with trajectory.
        """
        assert self._scheduler is not None

        # Create environment with optional overrides
        # Always create a fresh environment for each game to ensure clean state
        if env_params:
            base_env = create_environment(self.config.environment, env_params)
        else:
            # For evaluation: use default environment config
            base_env = create_environment(self.config.environment)

        # Always wrap environment to ensure consistent interface
        env = PaddedEnvironmentWrapper(
            base_env,
            self._expected_board_size,
            self._expected_action_size,
        )

        trajectory = []
        state, _ = env.reset()  # Unpack (obs, info) tuple
        done = False
        total_reward = 0.0
        turn = 0

        temperature = self._scheduler.get_temperature()

        while not done:
            # Get action probabilities from MCTS
            mcts_start = time.time()
            mcts_policy, _ = self._mcts.search(env)
            mcts_time = time.time() - mcts_start

            # Select action with temperature
            temp = temperature if turn < self.config.mcts.temperature_threshold else 0.1
            action = self._mcts.select_action(mcts_policy, temp)

            # Log every 50 steps to track progress
            if turn % 50 == 0:
                # Get turn info from wrapped env if available, otherwise from env directly
                base_env = getattr(env, '_env', env)
                env_turn = getattr(base_env, 'turn', '?')
                env_max = getattr(base_env, 'max_turns', '?')
                logger.info(f"Step {turn}: env_turn={env_turn}/{env_max}, mcts_time={mcts_time:.2f}s")

            # Store transition (pad to expected sizes for network compatibility)
            spatial, vector = self._encode_state(state)
            action_mask = self._pad_action_array(env.get_valid_actions())
            padded_policy = self._pad_action_array(mcts_policy)

            trajectory.append({
                "spatial": spatial,
                "vector": vector,
                "action_mask": action_mask,
                "policy": padded_policy,
                "action": action,
            })

            # Take action (gymnasium API returns 5 values)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Apply reward shaping
            if self._reward_shaper:
                reward = self._reward_shaper.shape_reward(
                    next_state, reward, info, done
                )

            total_reward += reward
            state = next_state
            turn += 1

        # Determine winner and assign values
        winner = info.get("winner", 0)
        self._assign_values(trajectory, winner, total_reward)

        return GameResult(
            trajectory=trajectory,
            winner=winner,
            total_reward=total_reward,
            turns=turn,
            metrics={"avg_reward": total_reward / max(turn, 1)},
        )

    def _encode_state(self, state: Any) -> tuple[np.ndarray, np.ndarray]:
        """Encode game state for network input.

        Pads spatial features to match network's expected board size when
        curriculum uses a smaller board.

        Args:
            state: Game state from environment.

        Returns:
            Tuple of (spatial_features, vector_features).
        """
        # Network's expected board size (max size from config)
        target_size = self.config.environment.board_size

        # Default encoding - override for specific environments
        if hasattr(state, "spatial") and hasattr(state, "vector"):
            spatial = state.spatial
            vector = state.vector
        # Handle dict states
        elif isinstance(state, dict):
            spatial = state.get("spatial", np.zeros((13, target_size, target_size)))
            vector = state.get("vector", np.zeros(10))
        else:
            # Fallback
            return np.zeros((13, target_size, target_size)), np.zeros(10)

        # Pad spatial features if smaller than target size
        current_size = spatial.shape[-1]
        if current_size < target_size:
            pad_amount = target_size - current_size
            # Pad equally on both sides (or extra on right/bottom if odd)
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left
            spatial = np.pad(
                spatial,
                ((0, 0), (pad_left, pad_right), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )

        return spatial, vector

    def _pad_action_array(self, arr: np.ndarray) -> np.ndarray:
        """Pad action array to network's expected action space size.

        When curriculum uses smaller boards, the action space is smaller.
        This pads the array with zeros to match the network's expected size.

        Args:
            arr: Action array (mask or policy) from smaller environment.

        Returns:
            Padded array matching expected action space size.
        """
        current_size = arr.shape[0]
        if current_size >= self._expected_action_size:
            return arr

        padded = np.zeros(self._expected_action_size, dtype=arr.dtype)
        padded[:current_size] = arr
        return padded

    def _assign_values(
        self,
        trajectory: list[dict[str, Any]],
        winner: int,
        total_reward: float,
    ) -> None:
        """Assign value targets to trajectory positions.

        Uses final game outcome as value target for all positions.

        Args:
            trajectory: Game trajectory.
            winner: Game winner (1, -1, or 0 for draw).
            total_reward: Total reward from game.
        """
        # Simple approach: all positions get final outcome as value
        value = 1.0 if winner == 1 else (-1.0 if winner == -1 else 0.0)

        for i, step in enumerate(trajectory):
            # Alternate perspective for player 2
            step["value"] = value if i % 2 == 0 else -value

    def _add_to_buffer(self, result: GameResult) -> None:
        """Add game result to replay buffer.

        Args:
            result: Game result to add.
        """
        for step in result.trajectory:
            self._replay_buffer.buffer.append({
                "spatial": step["spatial"],
                "vector": step["vector"],
                "action_mask": step["action_mask"],
                "policy": step["policy"],
                "value": step["value"],
            })

    def _train_epochs(self, num_epochs: int) -> dict[str, float]:
        """Train network for specified epochs.

        Args:
            num_epochs: Number of training epochs.

        Returns:
            Training metrics.
        """
        assert self._network is not None
        assert self._optimizer is not None
        assert self._scheduler is not None

        self._network.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        batch_size = self._scheduler.get_batch_size()

        for _ in range(num_epochs):
            batch = self._replay_buffer.sample(batch_size)
            if not batch:
                continue

            # Prepare batch tensors
            spatial = torch.FloatTensor(
                np.array([b["spatial"] for b in batch])
            ).to(self._device)
            vector = torch.FloatTensor(
                np.array([b["vector"] for b in batch])
            ).to(self._device)
            action_mask = torch.FloatTensor(
                np.array([b["action_mask"] for b in batch])
            ).to(self._device)
            target_policy = torch.FloatTensor(
                np.array([b["policy"] for b in batch])
            ).to(self._device)
            target_value = torch.FloatTensor(
                np.array([b["value"] for b in batch])
            ).to(self._device).unsqueeze(1)

            # Forward pass
            self._optimizer.zero_grad()
            policy_logits, value = self._network(spatial, vector)

            # Apply action mask
            policy_logits = policy_logits - (1 - action_mask) * 1e8

            # Compute losses
            policy_loss = F.cross_entropy(
                policy_logits,
                target_policy.argmax(dim=1),
            )
            # Ensure value has same shape as target_value [batch_size, 1]
            if value.dim() == 1:
                value = value.unsqueeze(1)
            value_loss = F.mse_loss(value, target_value)
            loss = policy_loss + value_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._network.parameters(),
                self.config.training.gradient_clip,
            )
            self._optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1

        if num_batches > 0:
            avg_policy = total_policy_loss / num_batches
            avg_value = total_value_loss / num_batches
            avg_total = total_loss / num_batches

            self.state.recent_losses.append(avg_total)
            if len(self.state.recent_losses) > 100:
                self.state.recent_losses.pop(0)

            return {
                "policy_loss": avg_policy,
                "value_loss": avg_value,
                "total_loss": avg_total,
            }

        return {}

    def _update_optimizer_lr(self) -> None:
        """Update optimizer learning rate from scheduler."""
        assert self._scheduler is not None
        assert self._optimizer is not None

        lr = self._scheduler.get_learning_rate()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def _evaluate(self) -> dict[str, float]:
        """Evaluate current model performance.

        Returns:
            Evaluation metrics.
        """
        assert self._network is not None
        assert self._checkpointer is not None

        logger.info("Running evaluation...")
        self._network.eval()

        # Play evaluation games
        num_eval_games = self.config.training.evaluation_games
        wins = 0
        total_reward = 0.0

        for _ in range(num_eval_games):
            result = self._play_single_game()
            if result.winner == 1:
                wins += 1
            total_reward += result.total_reward

        win_rate = wins / max(num_eval_games, 1)
        avg_reward = total_reward / max(num_eval_games, 1)

        self.state.current_win_rate = win_rate

        # Update best model if improved
        if win_rate > self.state.best_win_rate:
            self.state.best_win_rate = win_rate
            self._checkpointer.save_best(
                self._network,
                metrics={"win_rate": win_rate, "avg_reward": avg_reward},
            )
            logger.info(f"New best model saved with win_rate={win_rate:.3f}")

        logger.info(f"Evaluation: win_rate={win_rate:.3f}, avg_reward={avg_reward:.3f}")

        return {"win_rate": win_rate, "avg_reward": avg_reward}

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        assert self._network is not None
        assert self._optimizer is not None
        assert self._scheduler is not None
        assert self._checkpointer is not None

        checkpoint_data = {
            "model_state_dict": self._network.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state": {
                "step": self._scheduler.current_step,
            },
            "training_state": {
                "iteration": self.state.iteration,
                "total_games": self.state.total_games,
                "best_win_rate": self.state.best_win_rate,
                "current_win_rate": self.state.current_win_rate,
            },
        }

        self._checkpointer.save(
            checkpoint_data,
            iteration=self.state.iteration,
            metrics={
                "win_rate": self.state.current_win_rate,
                "loss": np.mean(self.state.recent_losses) if self.state.recent_losses else 0.0,
            },
        )

        logger.info(f"Checkpoint saved at iteration {self.state.iteration}")

    def _run_strategy_analysis(self, game_results: list[GameResult]) -> None:
        """Run LLM strategy analysis on recent games.

        Args:
            game_results: Recent game results to analyze.
        """
        if not self._strategy_advisor or not game_results:
            return

        # Analyze most recent game
        result = game_results[-1]
        report = self._strategy_advisor.generate_game_report(
            trajectory=result.trajectory,
            outcome="win" if result.winner == 1 else ("loss" if result.winner == -1 else "draw"),
        )

        # Store insights if RAG is enabled
        if self._rag and report.learning_points:
            for point in report.learning_points:
                self._rag.store_training_insight(point)

        logger.info(f"Strategy analysis complete: {report.strategy_assessment[:100]}...")

    def _log_iteration(
        self,
        iteration: int,
        game_results: list[GameResult],
        train_metrics: dict[str, float],
        iteration_time: float,
    ) -> None:
        """Log iteration metrics.

        Args:
            iteration: Current iteration number.
            game_results: Game results from this iteration.
            train_metrics: Training metrics.
            iteration_time: Time taken for iteration.
        """
        # Compute game metrics
        wins = sum(1 for r in game_results if r.winner == 1)
        win_rate = wins / max(len(game_results), 1)
        avg_turns = np.mean([r.turns for r in game_results]) if game_results else 0

        # Log to console
        if iteration % 10 == 0:
            logger.info(
                f"Iter {iteration}: games={len(game_results)}, "
                f"win_rate={win_rate:.3f}, "
                f"loss={train_metrics.get('total_loss', 0):.4f}, "
                f"stage={self.state.curriculum_stage}, "
                f"time={iteration_time:.1f}s"
            )

        # Log to W&B
        if self._wandb_logger and self._scheduler is not None:
            self._wandb_logger.log({
                "iteration": iteration,
                "games": len(game_results),
                "win_rate": win_rate,
                "avg_turns": avg_turns,
                "curriculum_stage": self.state.curriculum_stage,
                "learning_rate": self._scheduler.get_learning_rate(),
                "batch_size": self._scheduler.get_batch_size(),
                "buffer_size": len(self._replay_buffer),
                **train_metrics,
            })

    def _cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up...")

        # Save final checkpoint
        if self._checkpointer and self._network:
            self._checkpointer.save_final(
                self._network,
                metrics={
                    "win_rate": self.state.current_win_rate,
                    "total_games": self.state.total_games,
                    "total_iterations": self.state.iteration,
                },
            )

        # Close RAG
        if self._rag:
            self._rag.close()

        # Close W&B
        if self._wandb_logger and hasattr(self._wandb_logger, "finish"):
            self._wandb_logger.finish()

        self.state.is_running = False
        elapsed = time.time() - self.state.start_time
        logger.info(
            f"Training complete: {self.state.iteration} iterations, "
            f"{self.state.total_games} games, "
            f"{elapsed / 3600:.1f} hours"
        )

    def dry_run(self) -> bool:
        """Perform a dry run to validate configuration.

        Returns:
            True if validation successful.
        """
        logger.info("Performing dry run...")

        # Validate config
        warnings = self.config.validate()
        for warning in warnings:
            logger.warning(f"Config warning: {warning}")

        # Try to initialize components
        if not self.initialize():
            return False

        # Run a single mini-iteration
        logger.info("Running single test iteration...")
        result = self._play_single_game()
        logger.info(f"Test game completed: {result.turns} turns, winner={result.winner}")

        # Cleanup
        self._cleanup()

        logger.info("Dry run successful!")
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get training statistics.

        Returns:
            Dictionary of training statistics.
        """
        return {
            "iteration": self.state.iteration,
            "total_games": self.state.total_games,
            "best_win_rate": self.state.best_win_rate,
            "current_win_rate": self.state.current_win_rate,
            "curriculum_stage": self.state.curriculum_stage,
            "buffer_size": len(self._replay_buffer) if self._replay_buffer else 0,
            "recent_avg_loss": np.mean(self.state.recent_losses) if self.state.recent_losses else 0,
            "is_running": self.state.is_running,
            "elapsed_hours": (time.time() - self.state.start_time) / 3600 if self.state.is_running else 0,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrainingOrchestrator("
            f"experiment={self.config.experiment_name}, "
            f"iteration={self.state.iteration}, "
            f"games={self.state.total_games}, "
            f"running={self.state.is_running})"
        )
