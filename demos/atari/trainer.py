"""Single-player training loop for Atari AlphaZero demo.

Adapts the self-play training pipeline for single-player Atari games
where value targets are derived from discounted returns rather than
win/loss outcomes.

Example:
    >>> from demos.atari import AtariTrainer, AtariConfig
    >>> config = AtariConfig.for_game("Breakout")
    >>> trainer = AtariTrainer(config)
    >>> trainer.train(dry_run=True)
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from demos.atari.atari_env import AtariEnv
from demos.atari.atari_network import AtariNetwork
from demos.atari.config import AtariConfig, AtariTrainingConfig
from demos.atari.hierarchical import AtariHierarchicalAdapter
from demos.atari.novel_features import AtariRAGAdvisor, AtariReflectionMemory

logger = logging.getLogger(__name__)


# ============================================================================
# Training example & replay buffer
# ============================================================================


@dataclass
class TrainingExample:
    """A single training sample."""

    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    """Simple deque-backed replay buffer."""

    def __init__(self, max_size: int, min_size: int) -> None:
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)
        self.min_size = min_size

    def add(self, example: TrainingExample) -> None:
        self.buffer.append(example)

    def add_episode(self, examples: list[TrainingExample]) -> None:
        for ex in examples:
            self.add(ex)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        indices = np.random.choice(
            len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False
        )
        return [self.buffer[i] for i in indices]

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.min_size

    def __len__(self) -> int:
        return len(self.buffer)


class ReplayDataset(Dataset):
    """Torch dataset wrapping a list of ``TrainingExample`` objects."""

    def __init__(self, examples: list[TrainingExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "spatial": torch.from_numpy(ex.spatial).float(),
            "vector": torch.from_numpy(ex.vector).float(),
            "action_mask": torch.from_numpy(ex.action_mask).float(),
            "policy": torch.from_numpy(ex.policy).float(),
            "value": torch.tensor(ex.value, dtype=torch.float32),
        }


# ============================================================================
# Loss function
# ============================================================================


def compute_atari_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute AlphaZero-style loss for Atari.

    Args:
        policy_logits: Predicted policy logits (B, A).
        value: Predicted value (B,).
        target_policy: Target policy distribution (B, A).
        target_value: Target value (B,).
        action_mask: Valid action mask (B, A).

    Returns:
        ``(total_loss, policy_loss, value_loss)``
    """
    value_loss = F.mse_loss(value, target_value)

    if action_mask is not None:
        target_policy = target_policy * action_mask
        normaliser = target_policy.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target_policy = target_policy / normaliser

    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -torch.sum(target_policy * log_probs, dim=-1).mean()

    total_loss = policy_loss + value_loss
    return total_loss, policy_loss, value_loss


# ============================================================================
# Self-play episode
# ============================================================================


def _discount_returns(
    rewards: list[float],
    gamma: float,
) -> list[float]:
    """Compute discounted returns from a list of per-step rewards.

    Args:
        rewards: Per-step rewards.
        gamma: Discount factor.

    Returns:
        List of discounted returns (one per step), normalised to [-1, 1].
    """
    returns: list[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()

    # Normalise to [-1, 1] for the value head
    arr = np.array(returns, dtype=np.float32)
    max_abs = max(np.abs(arr).max(), 1e-8)
    arr = arr / max_abs
    return list(arr.tolist())


def run_episode(
    env: AtariEnv,
    network: AtariNetwork,
    mcts: Any,
    config: AtariTrainingConfig,
    hierarchical: AtariHierarchicalAdapter | None = None,
    rag_advisor: AtariRAGAdvisor | None = None,
    game_name: str = "",
) -> tuple[list[TrainingExample], list[dict[str, Any]]]:
    """Play one self-play episode and collect training data.

    Args:
        env: Atari environment.
        network: Neural network.
        mcts: MCTS instance (``Civ6MCTS`` or ``SimpleMCTS``).
        config: Training configuration.
        hierarchical: Optional hierarchical reasoning adapter.
        rag_advisor: Optional RAG advisor for policy boosting.
        game_name: Game name for RAG queries.

    Returns:
        Tuple of ``(training_examples, trajectory)`` where trajectory
        is the raw per-step data used for reflection memory.
    """
    obs, _info = env.reset()
    episode_history: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    rewards: list[float] = []
    step = 0

    logger.debug(
        "run_episode: starting (spatial=%s, actions=%d, game=%s)",
        obs["spatial"].shape,
        len(env.get_valid_actions()),
        game_name or "unknown",
    )

    while True:
        # MCTS search
        action_probs, _ = mcts.search(env)

        # Optional RAG policy boost
        if rag_advisor is not None:
            boost = rag_advisor.get_policy_boost(
                obs, num_actions=len(action_probs), game_name=game_name
            )
            if boost.any():
                action_probs = action_probs + boost
                # Re-normalise
                total = action_probs.sum()
                if total > 0:
                    action_probs = action_probs / total

        # Optional hierarchical reasoning overlay
        if hierarchical is not None and hierarchical.config.enabled:
            hier_result = hierarchical.decide(obs, env.get_valid_actions(), action_probs)
            if hier_result["policy"] is not None:
                action_probs = hier_result["policy"]

        # Record state
        episode_history.append(
            {
                "spatial": obs["spatial"].copy(),
                "vector": obs["vector"].copy(),
                "action_mask": obs["action_mask"].copy(),
                "policy": action_probs.copy(),
            }
        )

        # Action selection with temperature from MCTS config if available
        mcts_config = getattr(mcts, "config", None)
        if mcts_config is not None:
            temperature = getattr(mcts_config, "temperature", 1.0)
            temp_threshold = getattr(mcts_config, "temperature_threshold", 500)
        else:
            temperature = 1.0
            temp_threshold = 500
        if step >= temp_threshold:
            temperature = 0.0

        action = mcts.select_action(action_probs, temperature)

        # Step
        obs, reward, terminated, truncated, _info = env.step(action)
        rewards.append(float(reward))

        # Record trajectory step for reflection memory
        trajectory.append(
            {
                "obs": episode_history[-1],
                "action": action,
                "reward": float(reward),
            }
        )

        if hierarchical is not None:
            hierarchical.update(float(reward), terminated or truncated)

        step += 1

        if terminated or truncated:
            break

    total_reward = sum(rewards)
    logger.debug(
        "run_episode: finished after %d steps (total_reward=%.2f, "
        "policy_entropy_mean=%.3f)",
        step,
        total_reward,
        float(-np.mean([
            np.sum(h["policy"] * np.log(h["policy"] + 1e-10))
            for h in episode_history
        ])) if episode_history else 0.0,
    )

    # Compute discounted returns as value targets
    values = _discount_returns(rewards, config.discount_factor)

    examples: list[TrainingExample] = []
    for hist, val in zip(episode_history, values, strict=True):
        examples.append(
            TrainingExample(
                spatial=hist["spatial"],
                vector=hist["vector"],
                action_mask=hist["action_mask"],
                policy=hist["policy"],
                value=val,
            )
        )

    return examples, trajectory


# ============================================================================
# Training epoch
# ============================================================================


def train_epoch(
    network: AtariNetwork,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: str,
    gradient_clip: float,
) -> dict[str, float]:
    """Train network for one epoch over the given dataloader.

    Args:
        network: The neural network.
        optimizer: Optimiser instance.
        dataloader: Batched training data.
        device: PyTorch device string.
        gradient_clip: Maximum gradient norm.

    Returns:
        Dict of ``{total_loss, policy_loss, value_loss}`` (averaged).
    """
    network.train()
    total_loss = 0.0
    policy_loss = 0.0
    value_loss = 0.0
    num_batches = 0

    logger.debug("train_epoch: %d batches, device=%s", len(dataloader), device)

    for batch in dataloader:
        spatial = batch["spatial"].to(device)
        vector = batch["vector"].to(device)
        action_mask = batch["action_mask"].to(device)
        target_policy = batch["policy"].to(device)
        target_value = batch["value"].to(device)

        optimizer.zero_grad()

        logits, value = network(spatial, vector, action_mask)
        loss, p_loss, v_loss = compute_atari_loss(
            logits, value, target_policy, target_value, action_mask
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        policy_loss += p_loss.item()
        value_loss += v_loss.item()
        num_batches += 1

    if num_batches > 0:
        grad_norm = sum(
            p.grad.norm().item() for p in network.parameters() if p.grad is not None
        )
        logger.debug(
            "train_epoch: %d batches, avg_loss=%.4f, grad_norm=%.4f",
            num_batches,
            total_loss / num_batches,
            grad_norm,
        )

    denom = max(num_batches, 1)
    return {
        "total_loss": total_loss / denom,
        "policy_loss": policy_loss / denom,
        "value_loss": value_loss / denom,
    }


# ============================================================================
# Evaluation
# ============================================================================


def evaluate(
    env: AtariEnv,
    network: AtariNetwork,
    mcts: Any,
    num_episodes: int,
) -> dict[str, float]:
    """Evaluate the agent over several episodes (greedy action selection).

    Args:
        env: Atari environment.
        network: Neural network.
        mcts: MCTS instance.
        num_episodes: Number of episodes.

    Returns:
        Dict with ``mean_reward``, ``max_reward``, ``min_reward``.
    """
    total_rewards: list[float] = []

    for _ in range(num_episodes):
        _obs, _info = env.reset()
        episode_reward = 0.0

        while True:
            try:
                action_probs, _ = mcts.search(env, add_noise=False)
            except TypeError:
                action_probs, _ = mcts.search(env)
            action = mcts.select_action(action_probs, temperature=0.0)
            _obs, reward, terminated, truncated, _info = env.step(action)
            episode_reward += float(reward)
            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

    rewards_arr = np.array(total_rewards)
    return {
        "mean_reward": float(rewards_arr.mean()),
        "max_reward": float(rewards_arr.max()),
        "min_reward": float(rewards_arr.min()),
    }


# ============================================================================
# Main Trainer
# ============================================================================


class AtariTrainer:
    """Orchestrates the full Atari AlphaZero training pipeline.

    Args:
        config: Master configuration.
    """

    def __init__(self, config: AtariConfig | None = None) -> None:
        self.config = config or AtariConfig()
        self._network: AtariNetwork | None = None
        self._env: AtariEnv | None = None

    def train(self, dry_run: bool = False) -> dict[str, Any]:
        """Run the training loop.

        Args:
            dry_run: If ``True``, run only 1 iteration with 1 episode for
                     smoke-testing.

        Returns:
            Summary dict with final metrics.
        """
        cfg = self.config
        t_cfg = cfg.training

        # Directories
        os.makedirs(t_cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(t_cfg.log_dir, exist_ok=True)

        # Environment (used to query action/observation spaces)
        env = AtariEnv(cfg.environment)
        self._env = env
        num_actions = env.action_space.n

        logger.info("Environment: %s (actions=%d)", env, num_actions)

        # Network
        network = AtariNetwork(
            config=cfg.network,
            num_actions=num_actions,
            spatial_size=cfg.environment.spatial_size,
            spatial_channels=cfg.environment.spatial_channels,
            vector_dim=cfg.environment.vector_dim,
        ).to(t_cfg.device)
        self._network = network

        logger.info("Parameters: %s", f"{sum(p.numel() for p in network.parameters()):,}")
        logger.info("Device: %s", t_cfg.device)

        # MCTS — reuse core Civ6MCTS which is protocol-compatible
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig, SimpleMCTS

        mcts_cfg = MCTSConfig(
            num_simulations=cfg.mcts.num_simulations,
            c_puct=cfg.mcts.c_puct,
            dirichlet_alpha=cfg.mcts.dirichlet_alpha,
            dirichlet_epsilon=cfg.mcts.dirichlet_epsilon,
            temperature=cfg.mcts.temperature,
            temperature_threshold=cfg.mcts.temperature_threshold,
        )

        # Optimizer
        optimizer = optim.Adam(
            network.parameters(),
            lr=t_cfg.learning_rate,
            weight_decay=t_cfg.weight_decay,
        )

        # Replay buffer
        replay = ReplayBuffer(t_cfg.replay_buffer_size, t_cfg.min_replay_size)

        # Optional hierarchical reasoning
        hierarchical: AtariHierarchicalAdapter | None = None
        if cfg.hierarchical.enabled:
            hierarchical = AtariHierarchicalAdapter(cfg.hierarchical, num_actions)
            logger.info("Hierarchical reasoning ENABLED")
        else:
            logger.info("Hierarchical reasoning DISABLED")

        # Optional novel features (reflection memory + RAG advisor)
        reflection: AtariReflectionMemory | None = None
        rag_advisor: AtariRAGAdvisor | None = None
        if cfg.novel_features is not None:
            if cfg.novel_features.reflection_enabled:
                reflection = AtariReflectionMemory(cfg.novel_features)
                logger.info("Reflection Memory ENABLED (max=%d)", cfg.novel_features.reflection_max_entries)
            if cfg.novel_features.rag_enabled:
                rag_advisor = AtariRAGAdvisor(cfg.novel_features)
                rag_advisor.ingest_default_strategies(cfg.environment.game_name)
                logger.info("RAG Advisor ENABLED (%d strategies)", rag_advisor.size)

        # Training loop
        num_iterations = 1 if dry_run else t_cfg.num_iterations
        episodes_per = 1 if dry_run else t_cfg.episodes_per_iteration
        metrics_history: list[dict[str, Any]] = []

        for iteration in range(num_iterations):
            iter_start = time.time()
            logger.info("=" * 60)
            logger.info("Iteration %d / %d", iteration + 1, num_iterations)
            logger.info("=" * 60)

            # --- Self-play ---
            use_simple = iteration == 0 and not dry_run
            if use_simple:
                mcts_inst: Civ6MCTS | SimpleMCTS = SimpleMCTS(
                    num_simulations=cfg.mcts.num_simulations,
                    c_puct=cfg.mcts.c_puct,
                )
            else:
                mcts_inst = Civ6MCTS(network, mcts_cfg, device=t_cfg.device)

            sp_start = time.time()
            all_examples: list[TrainingExample] = []
            for _ep in range(episodes_per):
                ep_env = AtariEnv(cfg.environment)
                try:
                    examples, trajectory = run_episode(
                        ep_env, network, mcts_inst, t_cfg, hierarchical,
                        rag_advisor=rag_advisor,
                        game_name=cfg.environment.game_name,
                    )
                    all_examples.extend(examples)

                    # Store failed episodes in reflection memory
                    if reflection is not None and trajectory:
                        episode_reward = sum(s["reward"] for s in trajectory)
                        if episode_reward < 0:
                            reflection.add_failed_episode(
                                trajectory, episode_reward, cfg.environment.game_name,
                            )
                finally:
                    ep_env.close()

            replay.add_episode(all_examples)
            sp_time = time.time() - sp_start
            logger.info(
                "Self-play: %d examples from %d episodes (%.1fs)",
                len(all_examples),
                episodes_per,
                sp_time,
            )

            # Log novel features stats
            if reflection is not None:
                logger.info("Reflection memory: %d entries", reflection.size)
            if rag_advisor is not None:
                logger.info("RAG knowledge base: %d entries", rag_advisor.size)

            # --- Training ---
            if replay.is_ready() or dry_run:
                tr_start = time.time()
                sample_size = min(len(replay), t_cfg.batch_size * t_cfg.training_sample_multiplier)
                train_examples = replay.sample(sample_size)

                dataset = ReplayDataset(train_examples)
                dataloader = DataLoader(
                    dataset,
                    batch_size=t_cfg.batch_size,
                    shuffle=True,
                    num_workers=0,
                )

                num_epochs = 1 if dry_run else t_cfg.num_epochs
                for epoch in range(num_epochs):
                    losses = train_epoch(
                        network, optimizer, dataloader, t_cfg.device, t_cfg.gradient_clip
                    )
                    logger.info(
                        "  Epoch %d: loss=%.4f  policy=%.4f  value=%.4f",
                        epoch + 1,
                        losses["total_loss"],
                        losses["policy_loss"],
                        losses["value_loss"],
                    )

                tr_time = time.time() - tr_start
                logger.info("Training: %.1fs", tr_time)
            else:
                losses = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
                logger.info(
                    "Skipping training (buffer %d / %d)",
                    len(replay),
                    t_cfg.min_replay_size,
                )

            # --- Evaluation ---
            eval_metrics: dict[str, float] = {}
            should_eval = (
                (iteration + 1) % t_cfg.eval_interval == 0
                or iteration == num_iterations - 1
                or dry_run
            )
            if should_eval:
                eval_env = AtariEnv(cfg.environment)
                eval_mcts = Civ6MCTS(network, mcts_cfg, device=t_cfg.device)
                n_eval = 1 if dry_run else t_cfg.eval_episodes
                eval_metrics = evaluate(eval_env, network, eval_mcts, n_eval)
                eval_env.close()
                logger.info(
                    "Eval: mean=%.2f  max=%.2f  min=%.2f",
                    eval_metrics["mean_reward"],
                    eval_metrics["max_reward"],
                    eval_metrics["min_reward"],
                )

            # --- Checkpoint ---
            should_save = (iteration + 1) % t_cfg.save_interval == 0 or dry_run
            if should_save:
                ckpt_path = os.path.join(
                    t_cfg.checkpoint_dir,
                    f"atari_{cfg.environment.game_name}_iter{iteration + 1}.pt",
                )
                network.save(ckpt_path)

            iter_time = time.time() - iter_start
            iter_metrics: dict[str, Any] = {
                "iteration": iteration + 1,
                "examples": len(all_examples),
                "buffer_size": len(replay),
                "iter_time": iter_time,
                **losses,
                **eval_metrics,
            }
            if reflection is not None:
                iter_metrics["reflection_entries"] = reflection.size
            if rag_advisor is not None:
                iter_metrics["rag_entries"] = rag_advisor.size
            metrics_history.append(iter_metrics)

            if hierarchical is not None:
                goals = hierarchical.get_active_goals()
                logger.info("Active goals: %s", goals)

        # Final save
        final_path = os.path.join(t_cfg.checkpoint_dir, "latest.pt")
        network.save(final_path)

        env.close()
        logger.info("Training complete.  Final model: %s", final_path)

        return {
            "final_model": final_path,
            "iterations": len(metrics_history),
            "metrics": metrics_history,
        }
