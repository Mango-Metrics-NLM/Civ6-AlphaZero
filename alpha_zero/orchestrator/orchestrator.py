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
from typing import Any, TypeVar

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from alpha_zero.orchestrator.checkpointer import CheckpointManager
from alpha_zero.orchestrator.config import OrchestratorConfig
from alpha_zero.orchestrator.rag_integration import RAGIntegration
from alpha_zero.orchestrator.scheduler import HyperparameterScheduler

# Reflection system imports (optional, graceful degradation if not available)
try:
    from alpha_zero.core.reflection_generator import (
        ReflectionGenerator,
        ReflectionGeneratorConfig,
    )
    from alpha_zero.core.reflection_memory import ReflectionConfig, ReflectionMemory

    REFLECTION_AVAILABLE = True
except ImportError:
    REFLECTION_AVAILABLE = False
    ReflectionMemory = None  # type: ignore
    ReflectionGenerator = None  # type: ignore
from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

# Hierarchical system imports (optional, graceful degradation if not available)
try:
    from alpha_zero.hierarchical.config import HierarchicalConfig as HierarchicalSystemConfig
    from alpha_zero.hierarchical.training_integration import (
        HierarchicalTrainingConfig,
        HierarchicalTrainingIntegration,
    )

    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    HierarchicalTrainingIntegration = None  # type: ignore
    HierarchicalTrainingConfig = None  # type: ignore
from alpha_zero.utils.factories import (
    create_curriculum_manager,
    create_environment,
    create_llm_client,
    create_mcts,
    create_mlflow_logger,
    create_network,
    create_replay_buffer,
    create_reward_shaper,
    create_wandb_logger,
)

# Training safeguards (optional, graceful degradation)
try:
    import alpha_zero.training.training_safeguards  # noqa: F401

    SAFEGUARDS_AVAILABLE = True
except ImportError:
    SAFEGUARDS_AVAILABLE = False

# MuZero integration (optional, graceful degradation)
try:
    from alpha_zero.training.muzero_integration import MuZeroTrainingBridge

    MUZERO_AVAILABLE = True
except ImportError:
    MUZERO_AVAILABLE = False
    MuZeroTrainingBridge = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

# Type variable for the _require() initialization guard (see TrainingOrchestrator).
_T = TypeVar("_T")


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
    ) -> None:
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

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get padded observation from environment."""
        obs = self._env.get_observation()
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

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
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

    def __init__(self, config: OrchestratorConfig) -> None:
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
        self._mlflow_logger: Any = None
        self._elo_system: Any = None
        self._eval_pipeline: Any = None
        self._elo_ratings: dict[str, float] = {}
        self._device = "cpu"

        # Reflection system (LATS-style learning from failures)
        self._reflection_memory: Any = None
        self._reflection_generator: Any = None

        # Hierarchical decision-making system
        self._hierarchical_integration: Any = None

        # Training safeguards (Phase 8)
        self._training_safeguard: Any = None

        # Training-stagnation detector (win-rate / ELO plateau). Opt-in via
        # config.stagnation; mirrors the safeguard hook but fires at eval time
        # because the signal (win-rate vs incumbent / ELO) lives there.
        self._stagnation_detector: Any = None
        self._stagnation_stopped: bool = False

        # MuZero integration (Phase 9)
        self._muzero_bridge: Any = None
        self._training_mode: str = "alphazero"

        # Signal handling
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _require(self, value: _T | None, name: str) -> _T:
        """Return ``value`` if it is initialized, else raise a clear error.

        Replaces bare ``assert self._x is not None`` guards, which are stripped under
        ``python -O`` and would surface as opaque ``AttributeError``s on ``None``.
        Re-binding the attribute to the return value (``self._x = self._require(...)``)
        also narrows the type for static analysis exactly as the old ``assert`` did.

        Args:
            value: The lazily-initialized component (set during :meth:`initialize`).
            name: Human-readable component name for the error message.

        Returns:
            The non-``None`` ``value``.

        Raises:
            RuntimeError: If ``value`` is ``None`` (i.e. ``initialize()`` has not run).
        """
        if value is None:
            raise RuntimeError(
                f"Orchestrator component '{name}' is not initialized; call initialize() first."
            )
        return value

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
                log_dir / f"training_{self.config.experiment_name}.log",
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
            )
            logging.getLogger().addHandler(file_handler)

    def _handle_shutdown(self, signum: int, frame: object | None) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received, finishing current iteration...")
        self._shutdown_requested = True

    def set_training_safeguard(self, safeguard: Any) -> None:
        """Set optional training safeguard for health monitoring.

        Args:
            safeguard: A :class:`TrainingSafeguard` instance, or *None*
                to disable safeguards.
        """
        self._training_safeguard = safeguard

    def set_muzero_bridge(
        self,
        bridge: Any,
        *,
        training_mode: str = "muzero",
    ) -> None:
        """Set optional MuZero training bridge.

        When set with ``training_mode="muzero"``, self-play and training
        will use MuZero components (learned dynamics model, MuZeroMCTS)
        instead of standard AlphaZero MCTS.

        Args:
            bridge: A :class:`MuZeroTrainingBridge` instance, or *None*
                to disable MuZero mode.
            training_mode: Training mode to use — ``"alphazero"`` (default)
                or ``"muzero"``.
        """
        self._muzero_bridge = bridge
        self._training_mode = training_mode
        if bridge is not None:
            logger.info("MuZero bridge set, training_mode=%s", training_mode)

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

            # Initialize MCTS (reflection_memory will be set after reflection system init)
            self._mcts = create_mcts(
                self.config.mcts,
                self._network,
                device=self._device,
                reflection_memory=None,  # Will be updated after reflection init
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

            # Initialize MLflow (runs in parallel with W&B; either or both can be enabled).
            # The experiment name comes from config.mlflow.experiment_name (set via
            # YAML or CIV6AZ_MLFLOW_EXPERIMENT_NAME); the top-level
            # config.experiment_name is used only as a fallback when the mlflow
            # section did not set its own. This keeps CIV6AZ_MLFLOW_EXPERIMENT_NAME
            # authoritative over CIV6AZ_EXPERIMENT_NAME for MLflow specifically.
            if self.config.mlflow.enabled:
                mlflow_experiment_name = (
                    self.config.mlflow.experiment_name or self.config.experiment_name
                )
                self._mlflow_logger = create_mlflow_logger(
                    self.config.mlflow,
                    experiment_name=mlflow_experiment_name,
                )
                if self._mlflow_logger is not None:
                    hparams: dict[str, Any] = {
                        "training/batch_size": self.config.training.batch_size,
                        "training/num_epochs": self.config.training.num_epochs,
                        "training/games_per_iteration": (self.config.training.games_per_iteration),
                        "training/full_eval_interval": (self.config.training.full_eval_interval),
                        "training/num_iterations": self.config.training.num_iterations,
                        "mcts/num_simulations": self.config.mcts.num_simulations,
                        "mcts/c_puct": self.config.mcts.c_puct,
                        "network/num_res_blocks": self.config.network.num_res_blocks,
                        "network/num_filters": self.config.network.num_filters,
                        "env/board_size": self.config.environment.board_size,
                        "env/max_turns": self.config.environment.max_turns,
                        "lr_schedule/initial": self.config.lr_schedule.initial_value,
                        "experiment_name": self.config.experiment_name,
                    }
                    self._mlflow_logger.init(config_dict=hparams)

            # Initialize EvaluationPipeline + ELO for value_correlation and elo_spread.
            # These are independent of the MLflow flag so other consumers (e.g.
            # in-memory introspection) can use them.
            try:
                from alpha_zero.evaluation.evaluation_pipeline import EvaluationPipeline
                from alpha_zero.tournament.config import ELOConfig
                from alpha_zero.tournament.elo import ELORatingSystem

                self._eval_pipeline = EvaluationPipeline(
                    board_size=self.config.environment.board_size,
                    max_turns=self.config.environment.max_turns,
                    seed=self.config.random_seed,
                )
                self._elo_system = ELORatingSystem(ELOConfig())
                # Seed both rolling identities to the initial rating.
                initial_rating = float(self._elo_system.config.initial_rating)
                self._elo_ratings = {
                    "current": initial_rating,
                    "previous_best": initial_rating,
                }
            except ImportError as exc:
                logger.warning(
                    "EvaluationPipeline / ELO unavailable; skipping rich eval: %s",
                    exc,
                )
                self._eval_pipeline = None
                self._elo_system = None

            # Training-stagnation detector (opt-in). Reuses the standalone,
            # already-tested EloPlateauDetector; keyed on ELO/win-rate (NOT loss,
            # which can fall while the champion never improves — the v1.2.0 trap).
            if getattr(self.config, "stagnation", None) and self.config.stagnation.enabled:
                try:
                    from alpha_zero.training.elo_plateau_detector import (
                        EloPlateauDetector,
                        EloPlateauDetectorConfig,
                    )

                    self._stagnation_detector = EloPlateauDetector(
                        EloPlateauDetectorConfig.from_env(),
                        alert_manager=None,
                    )
                    logger.info("Training-stagnation detector enabled (ELO plateau).")
                except Exception as exc:  # graceful degradation, like safeguards
                    logger.warning("Stagnation detector unavailable: %s", exc)
                    self._stagnation_detector = None

            # Initialize reflection system (LATS-style learning from failures)
            if self.config.reflection.enabled and REFLECTION_AVAILABLE:
                try:
                    # Create reflection config from orchestrator config
                    ref_config = ReflectionConfig(
                        storage_backend=self.config.reflection.storage_backend,
                        collection_name=self.config.reflection.collection_name,
                        embedding_dimension=self.config.reflection.embedding_dimension,
                        similarity_threshold=self.config.reflection.similarity_threshold,
                        max_retrievals=self.config.reflection.max_retrievals,
                        warning_penalty=self.config.reflection.warning_penalty,
                        max_entries=self.config.reflection.max_entries,
                    )
                    self._reflection_memory = ReflectionMemory(ref_config)

                    # Initialize reflection memory asynchronously
                    import asyncio

                    success = asyncio.run(self._reflection_memory.initialize())
                    if success:
                        logger.info("ReflectionMemory initialized successfully")
                    else:
                        logger.warning("ReflectionMemory initialization failed, continuing without")
                        self._reflection_memory = None

                    # Create reflection generator if critiques are enabled
                    if self.config.reflection.generate_critiques and self._reflection_memory:
                        gen_config = ReflectionGeneratorConfig(
                            temperature=self.config.reflection.llm_temperature,
                            max_tokens=self.config.reflection.llm_max_tokens,
                            cache_critiques=self.config.reflection.cache_critiques,
                            use_fallback_on_failure=self.config.reflection.use_fallback_on_failure,
                        )
                        self._reflection_generator = ReflectionGenerator(gen_config)
                        logger.info("ReflectionGenerator initialized")

                except Exception as e:
                    logger.warning(f"Failed to initialize reflection system: {e}")
                    self._reflection_memory = None
                    self._reflection_generator = None
            elif self.config.reflection.enabled and not REFLECTION_AVAILABLE:
                logger.warning("Reflection enabled but reflection module not available")

            # Update MCTS with reflection memory if available
            if self._reflection_memory is not None and hasattr(self._mcts, "_reflection_memory"):
                self._mcts._reflection_memory = self._reflection_memory
                logger.info("MCTS updated with reflection memory")

            # Initialize hierarchical decision-making system
            if self.config.hierarchical.enabled and HIERARCHICAL_AVAILABLE:
                try:
                    # Create hierarchical config from orchestrator config
                    h_config = HierarchicalSystemConfig.default()
                    h_config = h_config.apply_env_overrides()

                    training_config = HierarchicalTrainingConfig(
                        enabled=True,
                        hierarchical_config=h_config,
                        train_hierarchical_networks=True,
                        use_goal_shaped_rewards=True,
                    )

                    self._hierarchical_integration = HierarchicalTrainingIntegration(
                        config=training_config,
                        orchestrator=self,
                        llm_advisor=self._strategy_advisor,
                    )
                    logger.info("Hierarchical decision-making system initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize hierarchical system: {e}")
                    self._hierarchical_integration = None
            elif self.config.hierarchical.enabled and not HIERARCHICAL_AVAILABLE:
                logger.warning("Hierarchical enabled but hierarchical module not available")

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
        self._checkpointer = self._require(self._checkpointer, "checkpointer")
        self._network = self._require(self._network, "network")
        self._optimizer = self._require(self._optimizer, "optimizer")
        self._scheduler = self._require(self._scheduler, "scheduler")

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
        self._scheduler = self._require(self._scheduler, "scheduler")
        self._network = self._require(self._network, "network")

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
        logger.info(
            f"Buffer size: {len(self._replay_buffer)}, min required: {self._replay_buffer.min_size}"
        )

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

        When ``training_mode == "muzero"`` and a :class:`MuZeroTrainingBridge`
        is set, self-play uses :meth:`MuZeroTrainingBridge.selfplay_step`
        (MuZeroMCTS, no environment cloning) instead of standard MCTS.

        Args:
            num_games: Number of games to play.
            env_params: Optional environment parameter overrides.

        Returns:
            List of game results.
        """
        self._network = self._require(self._network, "network")

        # MuZero self-play path
        if self._training_mode == "muzero" and self._muzero_bridge is not None:
            return self._run_muzero_self_play(num_games, env_params)

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
                f"reward={result.total_reward:.2f}, time={game_time:.1f}s",
            )

            # Process game result for reflection (on losses)
            if self._reflection_memory is not None:
                self._process_game_result(result)

        return results

    def _run_muzero_self_play(
        self,
        num_games: int,
        env_params: dict[str, Any] | None = None,
    ) -> list[GameResult]:
        """Run MuZero self-play games using the MuZero training bridge.

        MuZero MCTS searches over the learned dynamics model rather than
        cloning the environment, which is faster for complex environments.

        Args:
            num_games: Number of games to play.
            env_params: Optional environment parameter overrides.

        Returns:
            List of game results.
        """
        self._network = self._require(self._network, "network")
        self._muzero_bridge = self._require(self._muzero_bridge, "muzero_bridge")

        results = []
        self._network.eval()

        # Create MuZero MCTS instance on the orchestrator's device
        mcts = self._muzero_bridge.create_mcts(self._network, device=self._device)

        logger.info("Starting MuZero self-play: %d games", num_games)
        for game_idx in range(num_games):
            game_start = time.time()

            # Create fresh environment
            if env_params:
                base_env = create_environment(self.config.environment, env_params)
            else:
                base_env = create_environment(self.config.environment)

            env = PaddedEnvironmentWrapper(
                base_env,
                self._expected_board_size,
                self._expected_action_size,
            )

            # Run self-play via bridge (MuZeroMCTS, no env cloning)
            trajectory = self._muzero_bridge.selfplay_step(
                network=self._network,
                env=env,
                mcts=mcts,
                max_moves=self.config.environment.max_turns,
            )

            # Convert MuZero trajectory to GameResult
            total_reward = sum(trajectory.rewards) if trajectory.rewards else 0.0
            winner = 1 if total_reward > 0 else (-1 if total_reward < 0 else 0)

            result = GameResult(
                trajectory=[],  # Raw trajectory stored separately
                winner=winner,
                total_reward=total_reward,
                turns=len(trajectory.actions),
                metrics={"muzero_mode": 1.0},
            )

            # Store trajectory data in replay buffer
            if self._replay_buffer is not None:
                for i, obs in enumerate(trajectory.observations):
                    if isinstance(obs, dict):
                        self._replay_buffer.add(
                            {
                                "spatial": obs.get("spatial", obs.get("observation")),
                                "vector": obs.get(
                                    "vector", np.zeros(self.config.network.vector_dim)
                                ),
                                "action_mask": obs.get(
                                    "action_mask",
                                    np.ones(self._expected_action_size),
                                ),
                                "policy": trajectory.policies[i],
                                "value": trajectory.values[i],
                            }
                        )

            game_time = time.time() - game_start
            results.append(result)
            logger.info(
                "MuZero game %d/%d: turns=%d, winner=%d, reward=%.2f, time=%.1fs",
                game_idx + 1,
                num_games,
                result.turns,
                result.winner,
                result.total_reward,
                game_time,
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
        self._scheduler = self._require(self._scheduler, "scheduler")

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

        # Check if hierarchical mode is enabled
        use_hierarchical = (
            self._hierarchical_integration is not None and self.config.hierarchical.enabled
        )

        # Notify hierarchical integration of game start
        if use_hierarchical:
            self._hierarchical_integration.on_game_start()

        while not done:
            if use_hierarchical:
                # Use hierarchical decision-making
                game_state = self._build_game_state_dict(state, env, turn)
                valid_actions = env.get_valid_actions()
                action, _hierarchical_info = self._hierarchical_integration.select_action(
                    game_state, valid_actions
                )
                # Create a uniform policy for training (hierarchical provides action directly)
                mcts_policy = np.zeros(self._expected_action_size, dtype=np.float32)
                mcts_policy[action] = 1.0
                mcts_time = 0.0  # No MCTS time in hierarchical mode
            else:
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
                base_env_ref = getattr(env, "_env", env)
                env_turn = getattr(base_env_ref, "turn", "?")
                env_max = getattr(base_env_ref, "max_turns", "?")
                mode_str = "hierarchical" if use_hierarchical else "mcts"
                logger.info(
                    f"Step {turn}: env_turn={env_turn}/{env_max}, mode={mode_str}, time={mcts_time:.2f}s"
                )

            # Store transition (pad to expected sizes for network compatibility)
            spatial, vector = self._encode_state(state)
            action_mask = self._pad_action_array(env.get_valid_actions())
            padded_policy = self._pad_action_array(mcts_policy)

            trajectory.append(
                {
                    "spatial": spatial,
                    "vector": vector,
                    "action_mask": action_mask,
                    "policy": padded_policy,
                    "action": action,
                }
            )

            # Take action (gymnasium API returns 5 values)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Apply reward shaping (hierarchical or standard)
            if use_hierarchical:
                game_state = self._build_game_state_dict(state, env, turn)
                next_game_state = self._build_game_state_dict(next_state, env, turn + 1)
                reward = self._hierarchical_integration.compute_shaped_reward(
                    game_state, action, reward, next_game_state, done
                )
            elif self._reward_shaper:
                reward = self._reward_shaper.shape_reward(
                    next_state,
                    reward,
                    info,
                    done,
                )

            total_reward += reward
            state = next_state
            turn += 1

        # Determine winner and assign values
        winner = info.get("winner", 0)
        self._assign_values(trajectory, winner, total_reward)

        # Notify hierarchical integration of game end
        if use_hierarchical:
            self._hierarchical_integration.on_game_end(winner, total_reward)

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

    def _build_game_state_dict(
        self,
        state: Any,
        env: Any,
        turn: int,
    ) -> dict[str, Any]:
        """Build a game state dictionary for hierarchical controller.

        Args:
            state: Current game state from environment.
            env: Environment instance.
            turn: Current turn number.

        Returns:
            Dictionary with game state information for hierarchical decisions.
        """
        spatial, vector = self._encode_state(state)

        game_state = {
            "turn": turn,
            "spatial": spatial,
            "vector": vector,
        }

        # Extract additional info from environment if available
        base_env = getattr(env, "_env", env)

        if hasattr(base_env, "get_game_info"):
            game_state.update(base_env.get_game_info())

        # Add basic metrics from vector features if available
        if len(vector) >= 10:
            game_state["turn_progress"] = float(vector[0])
            game_state["resources"] = float(vector[1]) if len(vector) > 1 else 0.0
            game_state["own_units"] = float(vector[2]) if len(vector) > 2 else 0.0
            game_state["enemy_units"] = float(vector[3]) if len(vector) > 3 else 0.0
            game_state["own_cities"] = float(vector[4]) if len(vector) > 4 else 0.0
            game_state["enemy_cities"] = float(vector[5]) if len(vector) > 5 else 0.0

        return game_state

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

    async def _process_game_result_async(
        self,
        result: GameResult,
    ) -> None:
        """Process completed game, adding reflection for losses.

        This method generates verbal self-critiques for lost games and stores
        them in the reflection memory for future MCTS guidance.

        Args:
            result: The completed game result.
        """
        # Only reflect on losses (winner != 1 means we lost)
        if result.winner == 1:
            return  # We won, no reflection needed

        if not self._reflection_memory or not self._reflection_generator:
            return  # Reflection system not available

        try:
            # Create a simplified trajectory for critique generation
            from alpha_zero.distributed.shared_storage import GameTrajectory, TrainingExample

            examples = []
            for step in result.trajectory:
                examples.append(
                    TrainingExample(
                        spatial=step["spatial"],
                        vector=step["vector"],
                        action_mask=step["action_mask"],
                        policy=step["policy"],
                        value=step.get("value", 0.0),
                    )
                )

            trajectory = GameTrajectory(
                examples=examples,
                game_id=self.state.total_games,
                game_length=result.turns,
                winner=result.winner,
            )

            # Generate critique
            critique = await self._reflection_generator.generate_critique(trajectory)

            # Add to reflection memory
            await self._reflection_memory.add_failed_game(
                trajectory,
                critique=critique.full_critique,
                action_taken=critique.mistake_action,
            )

            logger.debug(f"Reflection added: {critique.summary[:50]}...")

        except Exception as e:
            logger.warning(f"Failed to generate reflection: {e}")

    def _process_game_result(self, result: GameResult) -> None:
        """Synchronous wrapper for processing game results.

        Args:
            result: The completed game result.
        """
        import asyncio

        try:
            asyncio.run(self._process_game_result_async(result))
        except Exception as e:
            logger.warning(f"Error in game result processing: {e}")

    def _add_to_buffer(self, result: GameResult) -> None:
        """Add game result to replay buffer.

        Args:
            result: Game result to add.
        """
        for step in result.trajectory:
            self._replay_buffer.buffer.append(
                {
                    "spatial": step["spatial"],
                    "vector": step["vector"],
                    "action_mask": step["action_mask"],
                    "policy": step["policy"],
                    "value": step["value"],
                }
            )

    def _train_epochs(self, num_epochs: int) -> dict[str, float]:
        """Train network for specified epochs.

        When ``training_mode == "muzero"`` and a MuZero training bridge is
        set, training uses :meth:`MuZeroTrainingBridge.training_step`
        instead of the standard AlphaZero loss computation.

        Args:
            num_epochs: Number of training epochs.

        Returns:
            Training metrics.
        """
        self._network = self._require(self._network, "network")
        self._optimizer = self._require(self._optimizer, "optimizer")
        self._scheduler = self._require(self._scheduler, "scheduler")

        # MuZero training path
        if self._training_mode == "muzero" and self._muzero_bridge is not None:
            return self._train_muzero_epochs(num_epochs)

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
                np.array([b["spatial"] for b in batch]),
            ).to(self._device)
            vector = torch.FloatTensor(
                np.array([b["vector"] for b in batch]),
            ).to(self._device)
            action_mask = torch.FloatTensor(
                np.array([b["action_mask"] for b in batch]),
            ).to(self._device)
            target_policy = torch.FloatTensor(
                np.array([b["policy"] for b in batch]),
            ).to(self._device)
            target_value = (
                torch.FloatTensor(
                    np.array([b["value"] for b in batch]),
                )
                .to(self._device)
                .unsqueeze(1)
            )

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

            # Training safeguard check (Phase 8)
            if self._training_safeguard is not None:
                grad_norm = (
                    sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in self._network.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                report = self._training_safeguard.check_step(
                    loss=loss.item(),
                    grad_norm=grad_norm,
                    step=self.state.total_steps + num_batches,
                    device=self._device,
                )
                if report.intervention is not None:
                    logger.warning(
                        "Safeguard intervention: %s (reason: %s)",
                        report.intervention.action.value,
                        report.intervention.reason,
                    )

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

    def _train_muzero_epochs(self, num_epochs: int) -> dict[str, float]:
        """Train using MuZero trainer for the specified number of epochs.

        Delegates batch preparation and training to the MuZero bridge,
        which uses unrolled dynamics loss (policy + value + reward +
        consistency).

        Args:
            num_epochs: Number of training epochs.

        Returns:
            Training metrics dictionary.
        """
        self._network = self._require(self._network, "network")
        self._muzero_bridge = self._require(self._muzero_bridge, "muzero_bridge")
        self._scheduler = self._require(self._scheduler, "scheduler")

        self._network.train()

        # Create or reuse MuZero trainer
        trainer = self._muzero_bridge.create_trainer(self._network, device=self._device)

        batch_size = self._scheduler.get_batch_size()
        total_loss = 0.0
        num_batches = 0

        for _epoch_idx in range(num_epochs):
            batch = self._replay_buffer.sample(batch_size)
            if not batch:
                continue

            # Training step via bridge
            metrics = self._muzero_bridge.training_step(
                trainer=trainer,
                batch=batch,
                step=self.state.total_steps + num_batches,
            )

            step_loss = metrics.get("total_loss", 0.0)
            total_loss += step_loss
            num_batches += 1

            # Training safeguard check
            if self._training_safeguard is not None:
                grad_norm = metrics.get("grad_norm", 0.0)
                report = self._training_safeguard.check_step(
                    loss=step_loss,
                    grad_norm=grad_norm,
                    step=self.state.total_steps + num_batches,
                    device=self._device,
                )
                if report.intervention is not None:
                    logger.warning(
                        "Safeguard intervention (MuZero): %s (reason: %s)",
                        report.intervention.action.value,
                        report.intervention.reason,
                    )

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            self.state.recent_losses.append(avg_loss)
            if len(self.state.recent_losses) > 100:
                self.state.recent_losses.pop(0)

            return {"total_loss": avg_loss, "training_mode": 1.0}

        return {}

    def _update_optimizer_lr(self) -> None:
        """Update optimizer learning rate from scheduler."""
        self._scheduler = self._require(self._scheduler, "scheduler")
        self._optimizer = self._require(self._optimizer, "optimizer")

        lr = self._scheduler.get_learning_rate()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def _evaluate(self) -> dict[str, float]:
        """Evaluate current model performance.

        Returns:
            Evaluation metrics.
        """
        self._network = self._require(self._network, "network")
        self._checkpointer = self._require(self._checkpointer, "checkpointer")

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

        # Log the fast champion-vs-self signal to MLflow every evaluation step.
        if self._mlflow_logger is not None:
            self._mlflow_logger.log_metrics(
                {
                    "eval/champion/win_rate": float(win_rate),
                    "eval/champion/avg_reward": float(avg_reward),
                },
                step=self.state.iteration,
            )

        # Periodically run the richer EvaluationPipeline (value_correlation +
        # vs-random / vs-heuristic) and update the ELO leaderboard.
        self._run_periodic_full_evaluation(win_rate)

        # Stagnation check (after eval/ELO are fresh). Alerts and, if configured,
        # requests a graceful stop.
        self._check_training_stagnation(win_rate)

        return {"win_rate": win_rate, "avg_reward": avg_reward}

    def _run_periodic_full_evaluation(self, win_rate: float) -> None:
        """Run EvaluationPipeline + ELO update every ``full_eval_interval`` iters.

        Logs ``eval/<opponent>/*`` and ``elo/*`` metrics to MLflow. No-op if
        prerequisites are missing or the iteration is off-cadence.
        """
        if self._eval_pipeline is None or self._elo_system is None:
            return
        interval = max(int(self.config.training.full_eval_interval), 1)
        if self.state.iteration % interval != 0:
            return

        try:
            random_result = self._eval_pipeline.evaluate_vs_random(
                self._network,
                n_games=self.config.training.evaluation_games,
            )
            heuristic_result = self._eval_pipeline.evaluate_vs_heuristic(
                self._network,
                n_games=self.config.training.evaluation_games,
            )
        except Exception as exc:
            logger.warning("Periodic full evaluation failed: %s", exc)
            return

        eval_results = {"random": random_result, "heuristic": heuristic_result}

        # ELO update vs the rolling "previous_best" identity.
        elo_updates: list[Any] = []
        previous_rating = self._elo_ratings.get("previous_best", 0.0)
        current_rating = self._elo_ratings.get("current", 0.0)
        if win_rate > self.state.best_win_rate:
            score1, score2 = 1.0, 0.0
        elif win_rate < self.state.best_win_rate:
            score1, score2 = 0.0, 1.0
        else:
            score1, score2 = 0.5, 0.5
        try:
            update1, update2 = self._elo_system.process_match(
                player1_id="current",
                player2_id="previous_best",
                player1_rating=current_rating,
                player2_rating=previous_rating,
                score1=score1,
                score2=score2,
            )
            self._elo_ratings["current"] = float(update1.rating_after)
            self._elo_ratings["previous_best"] = float(update2.rating_after)
            elo_updates = [update1, update2]
        except Exception as exc:
            logger.warning("ELO update failed: %s", exc)
            elo_updates = []

        if self._mlflow_logger is not None:
            self._mlflow_logger.log_evaluation(eval_results, self.state.iteration)
            if elo_updates:
                leaderboard = self._build_elo_leaderboard()
                self._mlflow_logger.log_elo_update(
                    elo_updates,
                    leaderboard,
                    self.state.iteration,
                )

    def _check_training_stagnation(self, win_rate: float) -> None:
        """Feed the stagnation detector and act on a detected plateau.

        No-op unless ``config.stagnation.enabled`` created a detector. The signal
        is ELO when available, else a win-rate-scaled pseudo-rating so the same
        plateau logic works without the (optional) ELO system. On plateau we log
        an MLflow alert and, if ``config.stagnation.auto_stop`` is set, request a
        graceful shutdown (the run loop checkpoints on exit, preserving best).
        """
        detector = self._stagnation_detector
        if detector is None:
            return

        # Prefer real ELO; otherwise map win-rate into an ELO-like scale so the
        # detector's point-based improvement threshold stays meaningful
        # (0.01 win-rate ~= 10 ELO points, the default improvement threshold).
        if self._elo_system is not None and "current" in self._elo_ratings:
            rating = float(self._elo_ratings["current"])
        else:
            rating = float(win_rate) * 1000.0

        try:
            detector.record_evaluation(
                iteration=self.state.iteration,
                elo_rating=rating,
                win_rate=float(win_rate),
                games_played=int(self.config.training.evaluation_games),
            )
            plateau = bool(detector.is_plateau())
            stats = detector.get_statistics()
        except Exception as exc:  # detector must never crash training
            logger.warning("Stagnation check failed: %s", exc)
            return

        stalled_for = int(stats.get("iterations_since_improvement", 0))

        if self.config.stagnation.alert_to_mlflow and self._mlflow_logger is not None:
            try:
                self._mlflow_logger.log_metrics(
                    {
                        "stagnation/elo_plateau": float(plateau),
                        "stagnation/iterations_since_improvement": float(stalled_for),
                    },
                    step=self.state.iteration,
                )
                if plateau and hasattr(self._mlflow_logger, "set_tag"):
                    self._mlflow_logger.set_tag(
                        "alert/WARNING/stagnation",
                        f"ELO/win-rate plateau for {stalled_for} evals at "
                        f"iteration {self.state.iteration}",
                    )
            except Exception as exc:
                logger.warning("Failed to log stagnation metrics: %s", exc)

        if plateau:
            logger.warning(
                "Training stagnation: no win-rate/ELO improvement for %d evaluations "
                "(iteration %d).",
                stalled_for,
                self.state.iteration,
            )
            if self.config.stagnation.auto_stop:
                logger.warning("auto_stop enabled - requesting graceful shutdown.")
                self._stagnation_stopped = True
                self._shutdown_requested = True

    def _build_elo_leaderboard(self) -> Any:
        """Build a ``RatingLeaderboard`` snapshot from ``self._elo_ratings``."""
        from alpha_zero.tournament.elo import RatingLeaderboard

        board = RatingLeaderboard()
        for player_id, rating in self._elo_ratings.items():
            board.update(player_id, rating)
        return board

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        self._network = self._require(self._network, "network")
        self._optimizer = self._require(self._optimizer, "optimizer")
        self._scheduler = self._require(self._scheduler, "scheduler")
        self._checkpointer = self._require(self._checkpointer, "checkpointer")

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
                "loss": float(np.mean(self.state.recent_losses))
                if self.state.recent_losses
                else 0.0,
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
                f"time={iteration_time:.1f}s",
            )

        # Log to W&B
        if self._wandb_logger and self._scheduler is not None:
            self._wandb_logger.log(
                {
                    "iteration": iteration,
                    "games": len(game_results),
                    "win_rate": win_rate,
                    "avg_turns": avg_turns,
                    "curriculum_stage": self.state.curriculum_stage,
                    "learning_rate": self._scheduler.get_learning_rate(),
                    "batch_size": self._scheduler.get_batch_size(),
                    "buffer_size": len(self._replay_buffer),
                    **train_metrics,
                }
            )

        # Log to MLflow (parallel to W&B; both can be enabled simultaneously)
        if self._mlflow_logger is not None:
            game_metrics: dict[str, float] = {
                "game/games": float(len(game_results)),
                "game/win_rate": float(win_rate),
                "game/avg_game_length": float(avg_turns),
            }
            train_metrics_ns: dict[str, float] = {
                f"train/{k}" if not k.startswith("train/") else k: float(v)
                for k, v in train_metrics.items()
            }
            scheduler_metrics: dict[str, float] = {}
            if self._scheduler is not None:
                scheduler_metrics = {
                    "train/learning_rate": float(self._scheduler.get_learning_rate()),
                    "train/batch_size": float(self._scheduler.get_batch_size()),
                    "train/buffer_size": float(len(self._replay_buffer)),
                    "train/curriculum_stage": float(self.state.curriculum_stage),
                    "train/iteration_time_seconds": float(iteration_time),
                }
            self._mlflow_logger.log_iteration(
                iteration=iteration,
                train_metrics=train_metrics_ns,
                game_metrics=game_metrics,
                scheduler_metrics=scheduler_metrics,
            )

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

        # Close MLflow
        if self._mlflow_logger is not None and hasattr(self._mlflow_logger, "finish"):
            self._mlflow_logger.finish()

        self.state.is_running = False
        elapsed = time.time() - self.state.start_time
        logger.info(
            f"Training complete: {self.state.iteration} iterations, "
            f"{self.state.total_games} games, "
            f"{elapsed / 3600:.1f} hours",
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
            "elapsed_hours": (time.time() - self.state.start_time) / 3600
            if self.state.is_running
            else 0,
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
