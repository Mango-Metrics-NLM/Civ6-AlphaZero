"""Factory functions for creating training components.

This module provides a unified interface for component instantiation with
configuration-driven creation. All factories handle:
- Configuration validation
- Dependency injection
- Graceful degradation when optional dependencies are missing

Usage:
    from alpha_zero.utils.factories import (
        create_network,
        create_mcts,
        create_environment,
        create_reward_shaper,
        create_curriculum_manager,
        create_rag_client,
        create_llm_client,
    )

    network = create_network(config.network, board_size=16, action_space_size=356)
    mcts = create_mcts(config.mcts, network, device="cuda")
    env = create_environment(config.environment)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from alpha_zero.orchestrator.config import (
        CurriculumConfig,
        EnvironmentConfig,
        LLMConfig,
        MCTSConfig,
        NetworkConfig,
        RAGConfig,
        RewardsConfig,
        StrategyAdvisorConfig,
    )

logger = logging.getLogger(__name__)


def create_network(
    config: NetworkConfig,
    board_size: int,
    action_space_size: int,
    device: str = "cpu",
    framework: str | None = None,
) -> Any:
    """Create neural network from configuration.

    Automatically selects between MicroCivNetwork (for smaller boards)
    and Civ6Network (for larger boards or when attention is needed).
    Supports both PyTorch and JAX backends via the framework parameter.

    Args:
        config: Network configuration.
        board_size: Game board size.
        action_space_size: Size of action space.
        device: Compute device ("cpu" or "cuda").
        framework: ML framework ("pytorch" or "jax"). If None, reads from
            CIV6AZ_FRAMEWORK env var, defaulting to "pytorch".

    Returns:
        Configured neural network on specified device.
    """
    effective_framework = (
        framework
        or getattr(config, "framework", None)
        or os.environ.get("CIV6AZ_FRAMEWORK", "pytorch")
    ).lower().strip()

    if effective_framework == "jax":
        try:
            from alpha_zero.core.jax_network import create_jax_network

            logger.info("Creating JAX network via factory")
            return create_jax_network(
                config,
                board_size=board_size,
                action_space_size=action_space_size,
            )
        except ImportError as e:
            logger.warning(
                "JAX not available, falling back to PyTorch: %s", e,
            )
            # Fall through to PyTorch

    from alpha_zero.core.civ6_network import Civ6Network, MicroCivNetwork

    # Use MicroCivNetwork for smaller boards (faster training)
    if board_size <= 16 and config.spatial_channels <= 13:
        logger.info(
            f"Creating MicroCivNetwork: board={board_size}, "
            f"actions={action_space_size}, filters={config.num_filters}",
        )
        network: torch.nn.Module = MicroCivNetwork(
            board_size=board_size,
            action_space_size=action_space_size,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
        )
    else:
        logger.info(
            f"Creating Civ6Network: board={board_size}, "
            f"channels={config.spatial_channels}, filters={config.num_filters}",
        )
        network = Civ6Network(
            board_size=board_size,
            action_space_size=action_space_size,
            spatial_channels=config.spatial_channels,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            num_heads=config.num_heads,
        )

    return network.to(device)


def create_mcts(
    config: MCTSConfig,
    network: torch.nn.Module | None = None,
    device: str = "cpu",
    use_simple: bool = False,
    reflection_memory: Any = None,
) -> Any:
    """Create MCTS instance from configuration.

    Can create either a neural network-guided MCTS (Civ6MCTS) or a simpler
    rollout-based MCTS (SimpleMCTS) for testing without a trained network.

    Args:
        config: MCTS configuration.
        network: Neural network for evaluation. Required unless use_simple=True.
        device: Compute device.
        use_simple: Use SimpleMCTS (random rollouts) instead of NN-guided.
        reflection_memory: Optional ReflectionMemory for LATS-style learning.
            When provided, MCTS will penalize actions that led to failures
            in similar states.

    Returns:
        Configured MCTS instance.

    Raises:
        ValueError: If network is None and use_simple is False.
    """
    from alpha_zero.core.civ6_mcts import (
        Civ6MCTS,
        SimpleMCTS,
    )
    from alpha_zero.core.civ6_mcts import (
        MCTSConfig as MCTSConfigClass,
    )

    if use_simple:
        logger.info(
            f"Creating SimpleMCTS: simulations={config.num_simulations}, c_puct={config.c_puct}",
        )
        return SimpleMCTS(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
        )

    if network is None:
        raise ValueError("Network required for Civ6MCTS. Use use_simple=True for testing.")

    mcts_config = MCTSConfigClass(
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        temperature=config.temperature,
        temperature_threshold=config.temperature_threshold,
    )

    logger.info(
        f"Creating Civ6MCTS: simulations={config.num_simulations}, "
        f"c_puct={config.c_puct}, device={device}, "
        f"reflection={'enabled' if reflection_memory else 'disabled'}",
    )
    return Civ6MCTS(network, mcts_config, device=device, reflection_memory=reflection_memory)


def create_environment(
    config: EnvironmentConfig,
    override_params: dict[str, Any] | None = None,
) -> Any:
    """Create environment with tuning and overrides.

    Supports creating either a basic MicroCivEnv or a tuned environment
    with early termination and production acceleration.

    Args:
        config: Environment configuration.
        override_params: Parameters to override (e.g., from curriculum).

    Returns:
        Configured environment instance.
    """
    from alpha_zero.envs.microciv import MicroCivEnv

    # Apply overrides (e.g., from curriculum)
    board_size = config.board_size
    max_turns = config.max_turns

    if override_params:
        board_size = override_params.get("board_size", board_size)
        max_turns = override_params.get("max_turns", max_turns)

    logger.info(
        f"Creating environment: type={config.type}, "
        f"board={board_size}x{board_size}, max_turns={max_turns}"
    )

    if config.type == "microciv":
        env = MicroCivEnv(
            board_size=board_size,
            max_turns=max_turns,
        )

        # Note: Environment tuning wrapper is disabled because MCTS
        # requires access to private methods (_get_observation) which
        # gymnasium wrappers block. The base MicroCivEnv is sufficient.
        if config.early_termination_enabled:
            logger.info("Early termination enabled (built into MicroCivEnv)")

        return env

    raise ValueError(f"Unknown environment type: {config.type}")


def create_reward_shaper(
    config: RewardsConfig,
    board_size: int,
) -> Any | None:
    """Create reward shaper from configuration.

    Args:
        config: Enhanced rewards configuration.
        board_size: Game board size for potential calculation.

    Returns:
        Configured reward shaper, or None if rewards disabled.
    """
    if not config.enabled:
        logger.info("Reward shaping disabled")
        return None

    try:
        from alpha_zero.utils.enhanced_rewards import (
            create_enhanced_reward_shaper,
        )

        logger.info(f"Creating reward shaper: scale={config.scale_factor}")
        return create_enhanced_reward_shaper(
            scale_factor=config.scale_factor,
            gamma=config.gamma,
        )
    except ImportError:
        logger.warning("Enhanced rewards not available")
        return None


def create_curriculum_manager(
    config: CurriculumConfig,
    save_path: str | None = None,
) -> Any | None:
    """Create curriculum manager from configuration.

    Args:
        config: Curriculum configuration.
        save_path: Path for curriculum state persistence.

    Returns:
        Configured curriculum manager, or None if curriculum disabled.
    """
    if not config.enabled:
        logger.info("Curriculum learning disabled")
        return None

    try:
        from alpha_zero.utils.curriculum_learning import (
            CurriculumConfig as CurriculumConfigClass,
        )
        from alpha_zero.utils.curriculum_learning import (
            CurriculumManager,
            CurriculumStage,
        )

        # Convert stages to CurriculumStage objects
        stages = []
        for stage in config.stages:
            stages.append(
                CurriculumStage(
                    name=stage.name,
                    level=stage.level,
                    board_size=stage.board_size,
                    max_turns=stage.max_turns,
                    promotion_threshold=stage.promotion_threshold,
                    min_games=stage.min_games,
                )
            )

        curriculum_config = CurriculumConfigClass(
            stages=stages,
            allow_demotion=config.allow_demotion,
            metrics_window=config.metrics_window,
        )

        logger.info(f"Creating curriculum manager: {len(stages)} stages")
        manager = CurriculumManager(curriculum_config)

        if save_path:
            manager.config.save_path = save_path

        return manager
    except ImportError:
        logger.warning("Curriculum learning not available")
        return None


def create_rag_client(
    config: RAGConfig,
) -> Any | None:
    """Create RAG client from configuration.

    Args:
        config: RAG configuration.

    Returns:
        Configured RAG client, or None if disabled or dependencies missing.
    """
    if not config.enabled:
        logger.info("RAG system disabled")
        return None

    try:
        from alpha_zero.utils.rag_client import RAGClient
        from alpha_zero.utils.rag_client import RAGConfig as RAGConfigClass

        rag_config = RAGConfigClass(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
            top_k=config.top_k,
        )

        logger.info(
            f"Creating RAG client: collection={config.collection_name}, "
            f"embedding={config.embedding_model}",
        )
        return RAGClient(rag_config)
    except ImportError as e:
        logger.warning(f"RAG dependencies not available: {e}")
        return None


def create_knowledge_base(
    rag_client: Any,
    config: RAGConfig,
) -> Any | None:
    """Create knowledge base for document ingestion.

    Args:
        rag_client: Initialized RAG client.
        config: RAG configuration.

    Returns:
        Configured knowledge base, or None if RAG client is None.
    """
    if rag_client is None:
        return None

    try:
        from alpha_zero.utils.knowledge_base import KnowledgeBase

        logger.info("Creating knowledge base")
        return KnowledgeBase(rag_client)
    except ImportError:
        logger.warning("Knowledge base not available")
        return None


def create_llm_client(
    config: LLMConfig,
) -> Any | None:
    """Create LLM client with fallback support.

    Args:
        config: LLM configuration.

    Returns:
        Configured LLM client, or None if disabled or dependencies missing.
    """
    if not config.enabled:
        logger.info("LLM integration disabled")
        return None

    try:
        from alpha_zero.utils.llm_client import (
            LLMClient,
            LLMProvider,
        )
        from alpha_zero.utils.llm_client import (
            LLMConfig as LLMConfigClass,
        )

        # Map provider string to enum
        provider_map = {
            "auto": LLMProvider.AUTO,
            "openai": LLMProvider.OPENAI,
            "anthropic": LLMProvider.ANTHROPIC,
        }
        provider = provider_map.get(config.provider.lower(), LLMProvider.AUTO)

        llm_config = LLMConfigClass(
            provider=provider,
            openai_model=config.openai_model,
            anthropic_model=config.anthropic_model,
            openai_max_tokens=config.max_tokens,
            anthropic_max_tokens=config.max_tokens,
            openai_temperature=config.temperature,
            anthropic_temperature=config.temperature,
            max_retries=config.max_retries,
            timeout=config.timeout,
        )

        logger.info(f"Creating LLM client: provider={config.provider}")
        return LLMClient(llm_config)
    except ImportError as e:
        logger.warning(f"LLM dependencies not available: {e}")
        return None
    except Exception as e:
        # Catch initialization errors (e.g., no API keys configured)
        logger.warning(f"LLM client initialization failed: {e}")
        return None


def create_strategy_advisor(
    llm_client: Any,
    rag_client: Any,
    config: StrategyAdvisorConfig,
) -> Any | None:
    """Create strategy advisor with RAG-augmented LLM.

    Args:
        llm_client: Initialized LLM client.
        rag_client: Initialized RAG client (can be None).
        config: Strategy advisor configuration.

    Returns:
        Configured strategy advisor, or None if disabled or LLM not available.
    """
    if not config.enabled:
        logger.info("Strategy advisor disabled")
        return None

    if llm_client is None:
        logger.warning("Strategy advisor requires LLM client")
        return None

    # Strategy advisor will be implemented in rag_integration.py
    # For now, return a placeholder that can be replaced later
    logger.info(
        f"Strategy advisor configured: analysis_interval={config.analysis_interval}",
    )

    return {
        "llm_client": llm_client,
        "rag_client": rag_client,
        "config": config,
        "_placeholder": True,
    }


def create_wandb_logger(
    config: Any,
    experiment_name: str | None = None,
) -> Any | None:
    """Create W&B logger from configuration.

    Args:
        config: W&B configuration.
        experiment_name: Override experiment name.

    Returns:
        Configured W&B logger, or None if disabled.
    """
    if not config.enabled:
        logger.info("W&B logging disabled")
        return None

    try:
        from alpha_zero.utils.wandb_logger import WandbConfig, WandbLogger

        wandb_config = WandbConfig(
            project=config.project,
            entity=config.entity,
            tags=config.tags,
            mode=config.mode,
        )

        logger.info(f"Creating W&B logger: project={config.project}")
        return WandbLogger(wandb_config)
    except ImportError:
        logger.warning("W&B not available")
        return None


def create_replay_buffer(
    max_size: int = 500000,
    min_size: int = 10000,
) -> Any:
    """Create replay buffer for training.

    Args:
        max_size: Maximum buffer size.
        min_size: Minimum samples before training starts.

    Returns:
        Replay buffer instance.
    """
    from collections import deque
    from dataclasses import dataclass

    import numpy as np

    @dataclass
    class TrainingExample:
        """Single training example."""

        spatial: np.ndarray
        vector: np.ndarray
        action_mask: np.ndarray
        policy: np.ndarray
        value: float

    class ReplayBuffer:
        """Simple replay buffer using deque."""

        def __init__(self, max_size: int) -> None:
            self.buffer: deque[TrainingExample] = deque(maxlen=max_size)
            self.min_size = min_size

        def add(self, example: TrainingExample) -> None:
            self.buffer.append(example)

        def add_game(self, examples: list[TrainingExample]) -> None:
            for example in examples:
                self.add(example)

        def sample(self, batch_size: int) -> list[TrainingExample]:
            indices = np.random.choice(
                len(self.buffer),
                size=min(batch_size, len(self.buffer)),
                replace=False,
            )
            return [self.buffer[i] for i in indices]

        def is_ready(self) -> bool:
            return len(self.buffer) >= self.min_size

        def __len__(self) -> int:
            return len(self.buffer)

    logger.info(f"Creating replay buffer: max_size={max_size}, min_size={min_size}")
    return ReplayBuffer(max_size)
