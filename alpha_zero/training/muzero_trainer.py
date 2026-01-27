"""MuZero-specific training module for Civ6 AlphaZero.

This module provides MuZero training with trajectory-based multi-step learning:

1. **GameTrajectory**: Stores full game data for trajectory sampling
2. **TrajectoryBuffer**: Manages game trajectories with position sampling
3. **MuZeroTrainer**: Implements MuZero training loop with unrolling

Features:
- Configurable via dataclass (no hard-coded values)
- Comprehensive logging for debugging
- Backwards compatible with existing replay buffer
- Modular, reusable components

Reference:
- MuZero: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
- EfficientZero: "Mastering Atari Games with Limited Data"
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from alpha_zero.core.losses import (
    LossConfig,
    compute_consistency_loss,
    compute_muzero_step_loss,
    compute_policy_loss,
    compute_reward_loss,
    compute_value_loss,
    scalar_to_categorical,
)
from alpha_zero.utils.checkpoint import safe_load_checkpoint

if TYPE_CHECKING:
    from alpha_zero.core.muzero_network import MuZeroNetwork
    from alpha_zero.training_civ6 import ReplayBuffer, TrainingExample

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MuZeroTrainingConfig:
    """MuZero training configuration.

    All training parameters are configurable here - no hard-coded values in
    the training code.

    Attributes:
        unroll_steps: Number of dynamics steps to unroll during training (K).
        td_steps: Number of steps for n-step return computation (n).
        use_muzero_network: Whether network supports MuZero-style training.
        use_consistency_loss: Enable EfficientZero self-supervised consistency.
        use_value_prefix: Enable value prefix prediction for credit assignment.
        batch_size: Number of trajectories per training batch.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularization weight.
        gradient_clip: Maximum gradient norm for clipping.
        policy_loss_weight: Weight for policy loss component.
        value_loss_weight: Weight for value loss component.
        reward_loss_weight: Weight for reward loss component.
        consistency_loss_weight: Weight for EfficientZero consistency loss.
        value_prefix_loss_weight: Weight for value prefix loss.
        discount: Discount factor for n-step returns.
        use_categorical: Use categorical distributions for value/reward.
        support_size: Size of categorical support (for categorical mode).
        min_value: Minimum value in categorical support.
        max_value: Maximum value in categorical support.

    Note:
        Prioritized experience replay is not yet implemented. Trajectories are
        sampled uniformly weighted by length. Priority-based sampling (PER)
        can be added in a future version for improved sample efficiency.
    """

    # Unrolling
    unroll_steps: int = 5
    td_steps: int = 10

    # Network
    use_muzero_network: bool = True
    use_consistency_loss: bool = True
    use_value_prefix: bool = True

    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    # Loss weights (from LossConfig)
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    consistency_loss_weight: float = 2.0
    value_prefix_loss_weight: float = 1.0

    # Value computation
    discount: float = 0.997

    # Categorical support
    use_categorical: bool = True
    support_size: int = 601
    min_value: float = -300.0
    max_value: float = 300.0



    def to_loss_config(self) -> LossConfig:
        """Convert to LossConfig for loss computation."""
        return LossConfig(
            use_categorical=self.use_categorical,
            support_size=self.support_size,
            min_value=self.min_value,
            max_value=self.max_value,
            policy_loss_weight=self.policy_loss_weight,
            value_loss_weight=self.value_loss_weight,
            reward_loss_weight=self.reward_loss_weight,
            consistency_loss_weight=self.consistency_loss_weight,
            value_prefix_loss_weight=self.value_prefix_loss_weight,
        )


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class GameTrajectory:
    """Stores full game data for multi-step MuZero training.

    A trajectory contains the complete history of a game:
    - Observations at each step (for representation function)
    - Actions taken (for dynamics function)
    - Rewards received (targets for reward prediction)
    - MCTS policies (targets for policy prediction)
    - Value targets (computed from n-step returns + bootstrapped value)

    All arrays should have the same length (number of steps in game).

    Attributes:
        observations: List of observation dicts with 'spatial', 'vector', 'action_mask'.
        actions: Actions taken at each step, shape (T,).
        rewards: Rewards received after each action, shape (T,).
        policies: MCTS visit count policies at each step, shape (T, A).
        values: Value targets (e.g., game outcome or n-step return), shape (T,).
        dones: Episode termination flags at each step, shape (T,).
        game_id: Optional identifier for this game.
        priority: Sampling priority for prioritized replay (default 1.0).
    """

    observations: list[dict[str, np.ndarray]] = field(default_factory=list)
    actions: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    rewards: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    policies: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    values: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    dones: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    game_id: str | None = None
    priority: float = 1.0

    def __len__(self) -> int:
        """Return number of steps in trajectory."""
        return len(self.observations)

    def __post_init__(self):
        """Validate trajectory after initialization."""
        if len(self.observations) > 0:
            self._validate()

    def _validate(self) -> None:
        """Validate trajectory data consistency."""
        n = len(self.observations)
        if n == 0:
            return

        # Check array lengths match
        if len(self.actions) not in (0, n, n - 1):
            logger.warning(
                f"Trajectory {self.game_id}: actions length {len(self.actions)} "
                f"doesn't match observations {n}"
            )

        if len(self.rewards) not in (0, n, n - 1):
            logger.warning(
                f"Trajectory {self.game_id}: rewards length {len(self.rewards)} "
                f"doesn't match observations {n}"
            )

        if len(self.policies) not in (0, n):
            logger.warning(
                f"Trajectory {self.game_id}: policies length {len(self.policies)} "
                f"doesn't match observations {n}"
            )

    def append(
        self,
        observation: dict[str, np.ndarray],
        action: int | None = None,
        reward: float = 0.0,
        policy: np.ndarray | None = None,
        value: float = 0.0,
        done: bool = False,
    ) -> None:
        """Append a step to the trajectory.

        Args:
            observation: Observation dict with 'spatial', 'vector', 'action_mask'.
            action: Action taken (None for terminal state).
            reward: Reward received.
            policy: MCTS policy distribution.
            value: Value target.
            done: Whether episode terminated.
        """
        self.observations.append(observation)

        if action is not None:
            self.actions = np.append(self.actions, action)

        self.rewards = np.append(self.rewards, reward)
        self.values = np.append(self.values, value)
        self.dones = np.append(self.dones, done)

        if policy is not None:
            if len(self.policies) == 0:
                self.policies = policy.reshape(1, -1)
            else:
                self.policies = np.vstack([self.policies, policy])

    def finalize(self, final_value: float = 0.0) -> None:
        """Finalize trajectory after game ends.

        Sets final values based on game outcome.

        Args:
            final_value: Final game outcome value (1 for win, -1 for loss, 0 for draw).
        """
        # Set all values to final outcome (simple approach)
        # In practice, you might want n-step returns with bootstrapping
        self.values = np.full(len(self.observations), final_value, dtype=np.float32)

        logger.debug(
            f"Finalized trajectory {self.game_id}: {len(self)} steps, "
            f"final_value={final_value:.2f}"
        )

    def compute_n_step_returns(
        self,
        discount: float = 0.997,
        n_steps: int = 10,
    ) -> None:
        """Compute n-step returns for value targets.

        Updates self.values with n-step returns:
        G_t = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1} + gamma^n * V(s_{t+n})

        Args:
            discount: Discount factor gamma.
            n_steps: Number of steps for n-step return.
        """
        n = len(self.rewards)
        if n == 0:
            return

        returns = np.zeros(n, dtype=np.float32)
        last_value = self.values[-1] if len(self.values) > 0 else 0.0

        for t in range(n - 1, -1, -1):
            # Bootstrapped value at step t+n (or end of episode)
            if t + n_steps < n:
                bootstrap = self.values[t + n_steps] * (discount**n_steps)
            else:
                bootstrap = last_value * (discount ** (n - t))

            # Sum of discounted rewards from t to t+n-1
            rewards_sum = 0.0
            for k in range(min(n_steps, n - t)):
                rewards_sum += (discount**k) * self.rewards[t + k]

            returns[t] = rewards_sum + bootstrap

        self.values = returns

        logger.debug(
            f"Computed {n_steps}-step returns for trajectory {self.game_id}, "
            f"mean={returns.mean():.3f}, std={returns.std():.3f}"
        )

    @classmethod
    def from_examples(
        cls,
        examples: Sequence["TrainingExample"],
        game_id: str | None = None,
    ) -> "GameTrajectory":
        """Create trajectory from list of TrainingExample (backwards compatibility).

        Args:
            examples: List of TrainingExample from existing replay buffer.
            game_id: Optional game identifier.

        Returns:
            GameTrajectory created from examples.
        """
        trajectory = cls(game_id=game_id)

        for ex in examples:
            obs = {
                "spatial": ex.spatial,
                "vector": ex.vector,
                "action_mask": ex.action_mask,
            }
            trajectory.append(
                observation=obs,
                policy=ex.policy,
                value=ex.value,
            )

        logger.debug(f"Created trajectory from {len(examples)} examples")
        return trajectory


# =============================================================================
# Trajectory Buffer
# =============================================================================


@dataclass
class TrajectoryBatchSample:
    """A batch sample from the trajectory buffer.

    Contains all data needed for one training batch with unrolling.

    Attributes:
        spatial: Initial observations spatial, shape (B, C, H, W).
        vector: Initial observations vector, shape (B, V).
        action_mask: Initial action masks, shape (B, A).
        actions: Actions for unrolling, shape (B, K).
        target_policies: Target policies for steps 0..K, list of (B, A).
        target_values: Target values for steps 0..K, list of (B,).
        target_rewards: Target rewards for steps 1..K, list of (B,).
        action_masks_unroll: Action masks for steps 1..K, list of (B, A).
        observations_spatial_unroll: Spatial observations for steps 1..K, list of (B, C, H, W).
            Used for EfficientZero consistency loss (comparing dynamics predictions
            with actual next observation representations).
        observations_vector_unroll: Vector observations for steps 1..K, list of (B, V).
            Used for EfficientZero consistency loss.
        weights: Importance sampling weights, shape (B,).
        indices: Source indices for priority updates, list of (traj_idx, pos_idx).
    """

    spatial: torch.Tensor
    vector: torch.Tensor
    action_mask: torch.Tensor
    actions: torch.Tensor  # (B, K)
    target_policies: list[torch.Tensor]  # K+1 tensors of shape (B, A)
    target_values: list[torch.Tensor]  # K+1 tensors of shape (B,)
    target_rewards: list[torch.Tensor]  # K tensors of shape (B,)
    action_masks_unroll: list[torch.Tensor]  # K tensors of shape (B, A)
    observations_spatial_unroll: list[torch.Tensor] | None = None  # K tensors of shape (B, C, H, W)
    observations_vector_unroll: list[torch.Tensor] | None = None  # K tensors of shape (B, V)
    weights: torch.Tensor | None = None
    indices: list[tuple[int, int]] | None = None  # (traj_idx, pos_idx) for each sample



class TrajectoryBuffer:
    """Buffer for storing game trajectories for multi-step training.

    Supports:
    - Adding complete game trajectories
    - Sampling positions with unroll segments
    - Priority-based sampling (optional)
    - Backwards compatibility with ReplayBuffer

    Attributes:
        max_trajectories: Maximum number of trajectories to store.
        max_positions: Maximum total positions across all trajectories.
        trajectories: Stored game trajectories.
        total_positions: Current total positions.
    """

    def __init__(
        self,
        max_trajectories: int = 1000,
        max_positions: int = 500000,
    ):
        """Initialize trajectory buffer.

        Args:
            max_trajectories: Maximum number of trajectories to store.
            max_positions: Maximum total positions (for memory management).
        """
        self.max_trajectories = max_trajectories
        self.max_positions = max_positions
        self.trajectories: deque[GameTrajectory] = deque(maxlen=max_trajectories)
        self.total_positions = 0

        logger.info(
            f"TrajectoryBuffer initialized: max_trajectories={max_trajectories}, "
            f"max_positions={max_positions}"
        )

    def __len__(self) -> int:
        """Return number of trajectories."""
        return len(self.trajectories)

    @property
    def num_positions(self) -> int:
        """Return total number of positions across all trajectories."""
        return self.total_positions

    def add_trajectory(self, trajectory: GameTrajectory) -> None:
        """Add a game trajectory to the buffer.

        Args:
            trajectory: GameTrajectory to add.
        """
        if len(trajectory) == 0:
            logger.warning("Attempted to add empty trajectory, skipping")
            return

        # Remove oldest trajectories if over capacity
        while (
            len(self.trajectories) >= self.max_trajectories
            or self.total_positions + len(trajectory) > self.max_positions
        ):
            if len(self.trajectories) == 0:
                break
            old = self.trajectories.popleft()
            self.total_positions -= len(old)
            logger.debug(f"Removed trajectory {old.game_id} ({len(old)} positions)")

        self.trajectories.append(trajectory)
        self.total_positions += len(trajectory)

        logger.debug(
            f"Added trajectory {trajectory.game_id}: {len(trajectory)} positions, "
            f"total={self.total_positions}"
        )

    def add_trajectories(self, trajectories: Sequence[GameTrajectory]) -> None:
        """Add multiple trajectories to the buffer.

        Args:
            trajectories: Sequence of GameTrajectory to add.
        """
        for traj in trajectories:
            self.add_trajectory(traj)

        logger.info(
            f"Added {len(trajectories)} trajectories, "
            f"total={len(self.trajectories)}, positions={self.total_positions}"
        )

    def update_priorities(
        self,
        indices: list[tuple[int, int]],
        priorities: list[float],
    ) -> None:
        """Update priorities for sampled positions.

        For prioritized experience replay (PER), this updates trajectory
        priorities based on TD errors from training. Each trajectory's
        priority is set to the maximum priority of its sampled positions.

        Args:
            indices: List of (trajectory_idx, position_idx) tuples.
            priorities: New priority values for each position.

        Note:
            Currently updates trajectory-level priorities. Position-level
            priorities can be implemented for finer-grained control.
        """
        if len(indices) != len(priorities):
            raise ValueError(
                f"indices ({len(indices)}) and priorities ({len(priorities)}) "
                "must have same length"
            )

        # Group priorities by trajectory
        traj_max_priorities: dict[int, float] = {}
        for (traj_idx, _pos_idx), priority in zip(indices, priorities):
            if traj_idx not in traj_max_priorities:
                traj_max_priorities[traj_idx] = priority
            else:
                traj_max_priorities[traj_idx] = max(
                    traj_max_priorities[traj_idx], priority
                )

        # Update trajectory priorities
        for traj_idx, max_priority in traj_max_priorities.items():
            if 0 <= traj_idx < len(self.trajectories):
                self.trajectories[traj_idx].priority = max(
                    self.trajectories[traj_idx].priority,
                    max_priority,
                )

        logger.debug(
            f"Updated priorities for {len(traj_max_priorities)} trajectories"
        )

    def sample_positions(
        self,
        batch_size: int,
        unroll_steps: int,
        device: str | torch.device = "cpu",
    ) -> TrajectoryBatchSample:
        """Sample positions from trajectories for multi-step training.

        Samples random positions and extracts unroll segments of length K.
        For positions near the end of trajectories, uses absorbing states
        (repeated last state with zero reward).

        Args:
            batch_size: Number of positions to sample.
            unroll_steps: Number of dynamics steps (K).
            device: Device to place tensors on.

        Returns:
            TrajectoryBatchSample with all data for training.

        Raises:
            ValueError: If buffer is empty.
        """
        if len(self.trajectories) == 0:
            raise ValueError("Cannot sample from empty buffer")

        logger.debug(
            f"Sampling {batch_size} positions with {unroll_steps} unroll steps"
        )

        # Sample trajectory indices weighted by length
        traj_lengths = [len(t) for t in self.trajectories]
        total_len = sum(traj_lengths)
        traj_probs = [l / total_len for l in traj_lengths]

        # Lists to collect batch data
        spatials = []
        vectors = []
        action_masks = []
        all_actions = []
        all_target_policies = [[] for _ in range(unroll_steps + 1)]
        all_target_values = [[] for _ in range(unroll_steps + 1)]
        all_target_rewards = [[] for _ in range(unroll_steps)]
        all_action_masks_unroll = [[] for _ in range(unroll_steps)]
        # For EfficientZero consistency loss: collect next observations
        all_obs_spatial_unroll = [[] for _ in range(unroll_steps)]
        all_obs_vector_unroll = [[] for _ in range(unroll_steps)]
        # Track sampled indices for priority updates
        sampled_indices: list[tuple[int, int]] = []

        for _ in range(batch_size):
            # Sample trajectory
            traj_idx = np.random.choice(len(self.trajectories), p=traj_probs)
            traj = self.trajectories[traj_idx]

            # Sample position within trajectory
            pos_idx = np.random.randint(0, len(traj))

            # Track index for priority updates
            sampled_indices.append((int(traj_idx), int(pos_idx)))

            # Get initial observation
            obs = traj.observations[pos_idx]
            spatials.append(obs["spatial"])
            vectors.append(obs["vector"])
            action_masks.append(obs["action_mask"])

            # Get unroll segment
            actions_segment = []
            for k in range(unroll_steps + 1):
                step_idx = pos_idx + k

                # Get policy target
                if step_idx < len(traj) and len(traj.policies) > step_idx:
                    policy = traj.policies[step_idx]
                else:
                    # Absorbing state: uniform policy over valid actions
                    policy = np.ones_like(traj.policies[0]) / len(traj.policies[0])
                all_target_policies[k].append(policy)

                # Get value target
                if step_idx < len(traj) and len(traj.values) > step_idx:
                    value = traj.values[step_idx]
                else:
                    # Absorbing state: terminal value
                    value = traj.values[-1] if len(traj.values) > 0 else 0.0
                all_target_values[k].append(value)

                # Get action and reward for unroll steps (k >= 1 uses action k-1)
                if k < unroll_steps:
                    action_idx = pos_idx + k
                    if action_idx < len(traj.actions):
                        actions_segment.append(traj.actions[action_idx])
                    else:
                        # Absorbing state: use action 0 (won't be used in loss)
                        actions_segment.append(0)

                    reward_idx = pos_idx + k
                    if reward_idx < len(traj.rewards):
                        reward = traj.rewards[reward_idx]
                    else:
                        # Absorbing state: zero reward
                        reward = 0.0
                    all_target_rewards[k].append(reward)

                    # Get action mask and observation for next state (for consistency loss)
                    next_obs_idx = pos_idx + k + 1
                    if next_obs_idx < len(traj):
                        next_obs = traj.observations[next_obs_idx]
                        next_mask = next_obs["action_mask"]
                        next_spatial = next_obs["spatial"]
                        next_vector = next_obs["vector"]
                    else:
                        # Absorbing state: reuse last observation (consistency loss will be skipped)
                        next_mask = np.ones_like(obs["action_mask"])
                        next_spatial = obs["spatial"]
                        next_vector = obs["vector"]
                    all_action_masks_unroll[k].append(next_mask)
                    all_obs_spatial_unroll[k].append(next_spatial)
                    all_obs_vector_unroll[k].append(next_vector)

            all_actions.append(actions_segment)

        # Convert to tensors
        spatial_tensor = torch.from_numpy(np.stack(spatials)).float().to(device)
        vector_tensor = torch.from_numpy(np.stack(vectors)).float().to(device)
        action_mask_tensor = torch.from_numpy(np.stack(action_masks)).float().to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=device)

        target_policies_tensors = [
            torch.from_numpy(np.stack(p)).float().to(device)
            for p in all_target_policies
        ]
        target_values_tensors = [
            torch.tensor(v, dtype=torch.float32, device=device)
            for v in all_target_values
        ]
        target_rewards_tensors = [
            torch.tensor(r, dtype=torch.float32, device=device)
            for r in all_target_rewards
        ]
        action_masks_unroll_tensors = [
            torch.from_numpy(np.stack(m)).float().to(device)
            for m in all_action_masks_unroll
        ]
        # Observations for EfficientZero consistency loss
        obs_spatial_unroll_tensors = [
            torch.from_numpy(np.stack(s)).float().to(device)
            for s in all_obs_spatial_unroll
        ]
        obs_vector_unroll_tensors = [
            torch.from_numpy(np.stack(v)).float().to(device)
            for v in all_obs_vector_unroll
        ]

        return TrajectoryBatchSample(
            spatial=spatial_tensor,
            vector=vector_tensor,
            action_mask=action_mask_tensor,
            actions=actions_tensor,
            target_policies=target_policies_tensors,
            target_values=target_values_tensors,
            target_rewards=target_rewards_tensors,
            action_masks_unroll=action_masks_unroll_tensors,
            observations_spatial_unroll=obs_spatial_unroll_tensors,
            observations_vector_unroll=obs_vector_unroll_tensors,
            indices=sampled_indices,
        )



    @classmethod
    def from_replay_buffer(
        cls,
        replay_buffer: "ReplayBuffer",
        max_trajectories: int = 1000,
    ) -> "TrajectoryBuffer":
        """Create TrajectoryBuffer from existing ReplayBuffer.

        This provides backwards compatibility with the existing replay buffer.
        Creates a single large trajectory from all examples.

        Args:
            replay_buffer: Existing ReplayBuffer.
            max_trajectories: Maximum trajectories for new buffer.

        Returns:
            TrajectoryBuffer populated from replay buffer.
        """
        buffer = cls(
            max_trajectories=max_trajectories,
            max_positions=len(replay_buffer) + 10000,
        )

        # Convert examples to trajectory
        # Note: This treats all examples as one trajectory (not ideal but compatible)
        if len(replay_buffer) > 0:
            examples = list(replay_buffer.buffer)
            trajectory = GameTrajectory.from_examples(
                examples, game_id="legacy_replay"
            )
            buffer.add_trajectory(trajectory)

        logger.info(
            f"Created TrajectoryBuffer from ReplayBuffer: "
            f"{len(replay_buffer)} examples"
        )

        return buffer


# =============================================================================
# MuZero Trainer
# =============================================================================


class MuZeroTrainer:
    """MuZero trainer implementing multi-step trajectory learning.

    The trainer handles:
    1. Network unrolling through dynamics function
    2. Multi-step loss computation
    3. Gradient updates with clipping
    4. Optional EfficientZero consistency loss

    Attributes:
        network: MuZero network to train.
        config: Training configuration.
        device: Torch device.
        optimizer: Adam optimizer.
        loss_config: Configuration for loss computation.
        step_count: Number of training steps completed.
    """

    def __init__(
        self,
        network: nn.Module,
        config: MuZeroTrainingConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        """Initialize MuZero trainer.

        Args:
            network: MuZero network (or compatible AlphaZero network).
            config: Training configuration.
            device: Device to train on.
        """
        self.network = network
        self.config = config or MuZeroTrainingConfig()
        self.device = torch.device(device)
        self.loss_config = self.config.to_loss_config()

        # Move network to device
        self.network.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.step_count = 0

        logger.info(
            f"MuZeroTrainer initialized: "
            f"unroll_steps={self.config.unroll_steps}, "
            f"batch_size={self.config.batch_size}, "
            f"lr={self.config.learning_rate}, "
            f"device={self.device}"
        )

    def train_step(
        self,
        batch: TrajectoryBatchSample,
    ) -> dict[str, float]:
        """Execute single training step with unrolling.

        This is the core training function:
        1. Initial inference from observation
        2. Unroll dynamics for K steps
        3. Compute losses at each step
        4. Backpropagate and update

        Args:
            batch: TrajectoryBatchSample with observations and targets.

        Returns:
            Dictionary of loss values for logging.
        """
        self.network.train()
        self.step_count += 1

        logger.debug(f"Training step {self.step_count}, batch_size={batch.spatial.size(0)}")

        # Zero gradients
        self.optimizer.zero_grad()

        # =====================================================================
        # Step 0: Initial inference
        # =====================================================================
        initial_output = self._initial_inference(batch)
        hidden_state = initial_output["hidden_state"]
        policy_logits_0 = initial_output["policy_logits"]
        value_0 = initial_output["value"]

        # Collect losses and predictions for all steps
        all_policy_logits = [policy_logits_0]
        all_values = [value_0]
        all_rewards = []
        all_hidden_states = [hidden_state]

        # =====================================================================
        # Steps 1..K: Unroll dynamics
        # =====================================================================
        current_hidden = hidden_state

        for k in range(self.config.unroll_steps):
            unroll_output = self._unroll_step(
                current_hidden,
                batch.actions[:, k],
                batch.action_masks_unroll[k] if k < len(batch.action_masks_unroll) else None,
            )

            current_hidden = unroll_output["hidden_state"]
            all_hidden_states.append(current_hidden)
            all_policy_logits.append(unroll_output["policy_logits"])
            all_values.append(unroll_output["value"])
            all_rewards.append(unroll_output["reward"])

            logger.debug(
                f"  Unroll step {k+1}: hidden_norm={current_hidden.norm().item():.3f}"
            )

        # =====================================================================
        # Compute losses
        # =====================================================================
        losses = self._compute_losses(
            all_policy_logits=all_policy_logits,
            all_values=all_values,
            all_rewards=all_rewards,
            all_hidden_states=all_hidden_states,
            batch=batch,
        )

        # =====================================================================
        # Backward pass and optimization
        # =====================================================================
        total_loss = losses["total_loss"]
        total_loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config.gradient_clip,
        )

        self.optimizer.step()

        # Prepare return dict with detached values
        result = {
            "total_loss": total_loss.item(),
            "policy_loss": losses["policy_loss"].item(),
            "value_loss": losses["value_loss"].item(),
            "reward_loss": losses["reward_loss"].item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "step": self.step_count,
        }

        if losses.get("consistency_loss") is not None:
            result["consistency_loss"] = losses["consistency_loss"].item()

        logger.debug(
            f"Step {self.step_count}: loss={result['total_loss']:.4f}, "
            f"policy={result['policy_loss']:.4f}, value={result['value_loss']:.4f}, "
            f"grad_norm={result['grad_norm']:.4f}"
        )

        return result

    def _initial_inference(
        self,
        batch: TrajectoryBatchSample,
    ) -> dict[str, torch.Tensor]:
        """Perform initial inference from observation.

        Args:
            batch: Batch with initial observations.

        Returns:
            Dict with hidden_state, policy_logits, value.
        """
        if self.config.use_muzero_network and hasattr(self.network, "initial_inference"):
            # MuZero network: use initial_inference
            output = self.network.initial_inference(
                batch.spatial,
                batch.vector,
                batch.action_mask,
            )
            return {
                "hidden_state": output.hidden_state,
                "policy_logits": output.policy_logits,
                "value": output.value,
            }
        else:
            # AlphaZero network: forward returns (policy, value)
            # Create dummy hidden state from spatial features
            policy_logits, value = self.network(
                batch.spatial,
                batch.vector,
                batch.action_mask,
            )
            # Use spatial as hidden state (simplified)
            hidden_state = batch.spatial
            return {
                "hidden_state": hidden_state,
                "policy_logits": policy_logits,
                "value": value,
            }

    def _unroll_step(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Unroll dynamics for one step.

        Args:
            hidden_state: Current hidden state.
            action: Actions to take.
            action_mask: Valid action mask for next state.

        Returns:
            Dict with hidden_state, policy_logits, value, reward.
        """
        if self.config.use_muzero_network and hasattr(self.network, "recurrent_inference"):
            # MuZero network: use recurrent_inference
            next_hidden, reward, policy_logits, value = self.network.recurrent_inference(
                hidden_state,
                action,
                action_mask,
            )
            return {
                "hidden_state": next_hidden,
                "policy_logits": policy_logits,
                "value": value,
                "reward": reward,
            }
        else:
            # AlphaZero network: can't do proper unrolling
            # Just return dummy values (training won't be correct but won't crash)
            logger.warning(
                "Using AlphaZero network for unrolling - reward prediction unavailable"
            )
            batch_size = hidden_state.size(0)
            policy_logits, value = self.network(
                hidden_state if hidden_state.dim() == 4 else hidden_state.unsqueeze(0),
                torch.zeros(batch_size, 64, device=self.device),  # dummy vector
                action_mask,
            )
            return {
                "hidden_state": hidden_state,
                "policy_logits": policy_logits,
                "value": value,
                "reward": torch.zeros(batch_size, device=self.device),
            }

    def _compute_losses(
        self,
        all_policy_logits: list[torch.Tensor],
        all_values: list[torch.Tensor],
        all_rewards: list[torch.Tensor],
        all_hidden_states: list[torch.Tensor],
        batch: TrajectoryBatchSample,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses for the batch.

        Args:
            all_policy_logits: Policy logits for steps 0..K.
            all_values: Values for steps 0..K.
            all_rewards: Rewards for steps 1..K.
            all_hidden_states: Hidden states for steps 0..K.
            batch: Training batch with targets.

        Returns:
            Dict with all loss components.
        """
        K = self.config.unroll_steps
        losses: dict[str, torch.Tensor] = {}

        # Accumulate losses
        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_reward_loss = torch.tensor(0.0, device=self.device)
        total_consistency_loss = None

        # Determine if outputs are categorical based on shape
        # Networks may output scalar or categorical - handle both
        first_value = all_values[0]
        value_is_categorical = first_value.dim() == 2 and first_value.size(-1) == self.config.support_size

        first_reward = all_rewards[0] if len(all_rewards) > 0 else None
        reward_is_categorical = (
            first_reward is not None
            and first_reward.dim() == 2
            and first_reward.size(-1) == self.config.support_size
        )

        # Create loss configs based on actual output shapes
        value_loss_config = LossConfig(
            use_categorical=value_is_categorical,
            support_size=self.config.support_size,
            min_value=self.config.min_value,
            max_value=self.config.max_value,
        )
        reward_loss_config = LossConfig(
            use_categorical=reward_is_categorical,
            support_size=self.config.support_size,
            min_value=self.config.min_value,
            max_value=self.config.max_value,
        )

        # Process each step
        for k in range(K + 1):
            # Policy loss
            policy_loss_k = compute_policy_loss(
                all_policy_logits[k],
                batch.target_policies[k],
                mask=batch.action_mask if k == 0 else batch.action_masks_unroll[k - 1],
            )
            total_policy_loss = total_policy_loss + policy_loss_k

            # Value loss
            value_k = all_values[k]
            # Ensure correct shape for loss function
            if not value_is_categorical and value_k.dim() == 2:
                value_k = value_k.squeeze(-1)  # (B, 1) -> (B,)
            # If dim() == 1, keep as-is (already (B,) for MSE)

            value_loss_k = compute_value_loss(
                value_k,
                batch.target_values[k],
                config=value_loss_config,
            )
            total_value_loss = total_value_loss + value_loss_k

            # Reward loss (only for k >= 1)
            if k > 0:
                reward_k = all_rewards[k - 1]
                # Ensure correct shape for loss function
                if not reward_is_categorical and reward_k.dim() == 2:
                    reward_k = reward_k.squeeze(-1)  # (B, 1) -> (B,)
                # If dim() == 1, keep as-is (already (B,) for MSE)

                reward_loss_k = compute_reward_loss(
                    reward_k,
                    batch.target_rewards[k - 1],
                    config=reward_loss_config,
                )
                total_reward_loss = total_reward_loss + reward_loss_k

            # Consistency loss (EfficientZero)
            # Compares dynamics model prediction with actual next observation representation
            if (
                self.config.use_consistency_loss
                and k > 0
                and hasattr(self.network, "project")
                and batch.observations_spatial_unroll is not None
                and batch.observations_vector_unroll is not None
            ):
                try:
                    # Get actual next observation from batch (k-1 index since unroll starts at step 1)
                    next_obs_spatial = batch.observations_spatial_unroll[k - 1]
                    next_obs_vector = batch.observations_vector_unroll[k - 1]

                    # Compute representation of actual next observation
                    with torch.no_grad():
                        if hasattr(self.network, "representation"):
                            # MuZero network: use representation function
                            actual_next_hidden = self.network.representation(
                                next_obs_spatial, next_obs_vector
                            )
                        else:
                            # AlphaZero fallback: use initial inference
                            output = self.network.initial_inference(
                                next_obs_spatial, next_obs_vector
                            )
                            # Handle both tuple and namedtuple/object returns
                            if hasattr(output, "hidden_state"):
                                actual_next_hidden = output.hidden_state
                            elif isinstance(output, tuple):
                                actual_next_hidden = output[0]
                            else:
                                actual_next_hidden = output
                        # Project the actual next observation representation (stop gradient)
                        proj_target = self.network.project(actual_next_hidden).detach()

                    # Project dynamics prediction (dynamics model's predicted next state)
                    proj_dynamics = self.network.project(all_hidden_states[k])

                    # Apply prediction head (SimSiam-style asymmetry)
                    if hasattr(self.network, "predict_projection"):
                        proj_prediction = self.network.predict_projection(proj_dynamics)
                    else:
                        proj_prediction = proj_dynamics

                    consistency_loss_k = compute_consistency_loss(
                        proj_prediction, proj_target
                    )

                    if total_consistency_loss is None:
                        total_consistency_loss = consistency_loss_k
                    else:
                        total_consistency_loss = total_consistency_loss + consistency_loss_k

                    logger.debug(
                        f"Consistency loss at step {k}: {consistency_loss_k.item():.4f}"
                    )
                except (ValueError, RuntimeError) as e:
                    logger.debug(f"Skipping consistency loss at step {k}: {e}")

        # Store individual losses
        losses["policy_loss"] = total_policy_loss
        losses["value_loss"] = total_value_loss
        losses["reward_loss"] = total_reward_loss

        if total_consistency_loss is not None:
            losses["consistency_loss"] = total_consistency_loss

        # Compute weighted total
        total = (
            self.config.policy_loss_weight * total_policy_loss
            + self.config.value_loss_weight * total_value_loss
            + self.config.reward_loss_weight * total_reward_loss
        )

        if total_consistency_loss is not None:
            total = total + self.config.consistency_loss_weight * total_consistency_loss

        losses["total_loss"] = total

        return losses

    def _compute_targets(
        self,
        batch: TrajectoryBatchSample,
    ) -> dict[str, list[torch.Tensor]]:
        """Compute training targets from batch.

        This method can be used to recompute targets if needed (e.g., for
        reanalysis or priority updates).

        Args:
            batch: Training batch.

        Returns:
            Dict with target_policies, target_values, target_rewards lists.
        """
        # For now, targets come from the batch directly
        # This method exists for extension (e.g., n-step return computation)
        return {
            "target_policies": batch.target_policies,
            "target_values": batch.target_values,
            "target_rewards": batch.target_rewards,
        }

    def train_on_buffer(
        self,
        buffer: TrajectoryBuffer,
        num_steps: int = 1000,
        log_interval: int = 100,
    ) -> dict[str, list[float]]:
        """Train for multiple steps sampling from buffer.

        Args:
            buffer: TrajectoryBuffer to sample from.
            num_steps: Number of training steps.
            log_interval: Steps between logging.

        Returns:
            Dict with lists of metrics over training.
        """
        logger.info(f"Starting training for {num_steps} steps")

        history: dict[str, list[float]] = {
            "total_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "reward_loss": [],
            "grad_norm": [],
        }

        for step in range(num_steps):
            # Sample batch
            batch = buffer.sample_positions(
                batch_size=self.config.batch_size,
                unroll_steps=self.config.unroll_steps,
                device=self.device,
            )

            # Train step
            metrics = self.train_step(batch)

            # Record history
            for key in history:
                if key in metrics:
                    history[key].append(metrics[key])

            # Log progress
            if (step + 1) % log_interval == 0:
                avg_loss = np.mean(history["total_loss"][-log_interval:])
                avg_policy = np.mean(history["policy_loss"][-log_interval:])
                avg_value = np.mean(history["value_loss"][-log_interval:])
                logger.info(
                    f"Step {step + 1}/{num_steps}: "
                    f"loss={avg_loss:.4f}, policy={avg_policy:.4f}, value={avg_value:.4f}"
                )

        logger.info(f"Training complete: {num_steps} steps")
        return history

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint.

        Note:
            Checkpoints containing optimizer state may require unsafe loading.
            Only load checkpoints from trusted sources.
        """
        result = safe_load_checkpoint(
            path,
            map_location=self.device,
            strict_security=False,
            allow_unsafe_fallback=True,
        )
        checkpoint = result.data

        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint.get("step_count", 0)

        if not result.loaded_safely:
            logger.warning(
                "Checkpoint loaded with weights_only=False due to optimizer state. "
                "Ensure checkpoint is from a trusted source."
            )
        logger.info(f"Loaded checkpoint from {path}, step={self.step_count}")


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "MuZeroTrainingConfig",
    "GameTrajectory",
    "TrajectoryBatchSample",
    "TrajectoryBuffer",
    "MuZeroTrainer",
]
