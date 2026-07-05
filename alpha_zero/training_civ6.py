"""Training pipeline for Civ6 AlphaZero.

Implements the AlphaZero self-play training loop:
1. Self-play: Generate games using MCTS + neural network
2. Training: Update network on generated data
3. Evaluation: Compare new network against previous version
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from alpha_zero.core.civ6_mcts import (
    Civ6MCTS,
    GumbelMCTS,
    GumbelMCTSConfig,
    MCTSConfig,
    SimpleMCTS,
)
from alpha_zero.core.civ6_network import MicroCivNetwork, compute_loss
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.multiplayer.network_heads import compute_nplayer_value_loss
from alpha_zero.multiplayer.scoring import (
    ScoringMethod,
    compute_player_values,
    compute_rankings,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Environment
    board_size: int = 16
    max_turns: int = 200
    # Number of players. 2 keeps the classic scalar-value pipeline; > 2 enables
    # the N-player value pipeline (per-player value head + per-player targets).
    num_players: int = 2
    # How terminal game outcomes are converted to per-player training values.
    # MicroCiv only exposes a winner (no per-player scores), so the supported
    # methods are the winner-derivable ones in SUPPORTED_SCORING_METHODS:
    # "winner_take_all" (classic +1/-1/0) and "linear_ranking" (rank-spread).
    # "proportional" needs a per-player score signal this pipeline does not
    # provide and would silently fall back to linear_ranking, so it is not
    # offered here.
    scoring_method: str = "winner_take_all"

    # Network
    num_res_blocks: int = 6
    num_filters: int = 64

    # MCTS
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30

    # Gumbel MuZero MCTS (alternative to standard MCTS)
    use_gumbel_mcts: bool = False
    gumbel_num_simulations: int = 16
    gumbel_num_sampled_actions: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_use_sequential_halving: bool = True

    # Training
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    num_epochs: int = 10

    # Self-play
    num_games_per_iteration: int = 100
    num_iterations: int = 100
    replay_buffer_size: int = 50000
    min_replay_size: int = 1000

    # Checkpointing
    checkpoint_dir: str = "checkpoints/civ6"
    log_dir: str = "logs/civ6"
    save_interval: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingExample:
    """Single training example.

    ``value`` is the scalar outcome from the acting player's perspective
    (kept for backward compatibility and 2-player training). ``value_vector``
    is the optional per-player outcome vector in *relative* player order
    (index 0 = acting player), populated only for N-player (> 2) games so the
    N-player value head can be trained. When ``value_vector`` is ``None`` the
    pipeline uses the scalar ``value`` exactly as before.
    """

    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    policy: np.ndarray
    value: float
    value_vector: np.ndarray | None = None


class GameHistoryItem(TypedDict):
    """Type for game history items."""

    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    policy: np.ndarray
    player: int


class ReplayBuffer:
    """Replay buffer for storing training examples."""

    def __init__(self, max_size: int = 50000) -> None:
        """Initialize replay buffer.

        Args:
            max_size: Maximum number of examples to store.
        """
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add(self, example: TrainingExample) -> None:
        """Add example to buffer."""
        self.buffer.append(example)

    def add_game(self, examples: list[TrainingExample]) -> None:
        """Add all examples from a game."""
        for example in examples:
            self.add(example)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        """Sample batch from buffer.

        Args:
            batch_size: Number of examples to sample.

        Returns:
            List of sampled examples.
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class ReplayDataset(Dataset):
    """PyTorch dataset for replay buffer."""

    def __init__(self, examples: list[TrainingExample]) -> None:
        """Initialize dataset.

        Args:
            examples: List of training examples.
        """
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        # Per-player value vector (N-player) when present; scalar otherwise.
        # Both collate cleanly: scalars stack to (B,), vectors to (B, P).
        if ex.value_vector is not None:
            value = torch.from_numpy(np.asarray(ex.value_vector)).float()
        else:
            value = torch.tensor(ex.value).float()
        return {
            "spatial": torch.from_numpy(ex.spatial).float(),
            "vector": torch.from_numpy(ex.vector).float(),
            "action_mask": torch.from_numpy(ex.action_mask).float(),
            "policy": torch.from_numpy(ex.policy).float(),
            "value": value,
        }


# Scoring methods this pipeline can honour. MicroCiv exposes only a winner
# (no per-player scores), so PROPORTIONAL — which needs a score signal — is
# excluded: it would silently fall back to LINEAR_RANKING inside
# multiplayer.scoring. Wire a real per-player score signal to add it.
SUPPORTED_SCORING_METHODS: tuple[ScoringMethod, ...] = (
    ScoringMethod.WINNER_TAKE_ALL,
    ScoringMethod.LINEAR_RANKING,
)


def _scoring_method_from_str(name: str) -> ScoringMethod:
    """Resolve a scoring-method name to a supported :class:`ScoringMethod`.

    Names that are unknown *or* unsupported by this winner-only pipeline (e.g.
    ``proportional``) fall back to ``winner_take_all`` (the classic AlphaZero
    +1/-1/0 targets) with a warning, so a bad config never crashes training and
    never silently trains on a degenerate score-less proportional signal.
    """
    try:
        method = ScoringMethod(name)
    except ValueError:
        logger.warning("Unknown scoring_method %r; falling back to 'winner_take_all'", name)
        return ScoringMethod.WINNER_TAKE_ALL
    if method not in SUPPORTED_SCORING_METHODS:
        logger.warning(
            "scoring_method %r needs a per-player score signal this pipeline does "
            "not provide; falling back to 'winner_take_all'",
            name,
        )
        return ScoringMethod.WINNER_TAKE_ALL
    return method


def compute_absolute_player_values(
    winner: int | None,
    num_players: int,
    method: ScoringMethod = ScoringMethod.WINNER_TAKE_ALL,
) -> np.ndarray:
    """Per-player training values in ABSOLUTE player order (index p = player p).

    Reuses :mod:`alpha_zero.multiplayer.scoring`. With ``winner_take_all`` this
    reproduces the classic +1 (winner) / -1 (loser) / 0 (draw) targets exactly,
    so the 2-player pipeline is unchanged.

    Args:
        winner: Winning player index, or ``None`` for a draw.
        num_players: Total number of players (>= 2).
        method: Outcome -> value conversion strategy.

    Returns:
        Array of shape ``(num_players,)`` with each value in ``[-1, 1]``.
    """
    rankings = compute_rankings(winner=winner, num_players=num_players)
    values = compute_player_values(rankings, num_players=num_players, method=method)
    return np.array([values[p] for p in range(num_players)], dtype=np.float32)


def relative_value_vector(
    absolute_values: np.ndarray, acting_player: int, num_players: int
) -> np.ndarray:
    """Reorder absolute per-player values into the acting player's frame.

    The MicroCiv encoder is player-relative (index 0 == current player), so the
    N-player value head predicts values in relative order. This maps absolute
    outcomes into that frame: ``out[r] = absolute_values[(acting_player + r) % N]``.
    ``out[0]`` therefore equals the acting player's own outcome.
    """
    return np.array(
        [absolute_values[(acting_player + r) % num_players] for r in range(num_players)],
        dtype=np.float32,
    )


class _PredictNetwork(Protocol):
    """Structural type for a network exposing MCTS's ``predict`` interface."""

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class CurrentPlayerValueView(nn.Module):
    """Adapt an N-player network to scalar-value MCTS.

    N-player networks emit a per-player value vector in *relative* order
    (index 0 = acting player). Scalar MCTS variants (``Civ6MCTS`` /
    ``GumbelMCTS``) expect one value per state, so this thin wrapper reduces the
    vector to its current-player (index 0) component. It holds a reference to
    the wrapped network as a submodule, so parameters are shared and self-play
    runs on the exact weights being trained. A already-scalar value passes
    through unchanged, keeping the wrapper safe for 2-player networks too.
    """

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    @staticmethod
    def _current_player_value(value: torch.Tensor) -> torch.Tensor:
        if value.dim() >= 2 and value.shape[-1] > 1:
            return value[..., 0]
        return value.reshape(value.shape[0]) if value.dim() >= 1 else value

    def forward(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits, value = self.network(spatial, vector, action_mask)
        return policy_logits, self._current_player_value(value)

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # nn.Module has no typed ``predict``; the wrapped network does.
        network = cast("_PredictNetwork", self.network)
        policy_probs, value = network.predict(spatial, vector, action_mask)
        return policy_probs, self._current_player_value(value)


def _mcts_network(network: torch.nn.Module, config: TrainingConfig) -> torch.nn.Module:
    """Return the network view MCTS should use for self-play / evaluation.

    For N-player training the network emits a value vector, so it is wrapped in
    :class:`CurrentPlayerValueView`; for 2-player training the network is
    returned unchanged (byte-identical to the legacy path).
    """
    if getattr(config, "num_players", 2) > 2:
        view = CurrentPlayerValueView(network)
        return view.to(config.device)
    return network


def self_play_game(
    env: MicroCivEnv,
    mcts: Civ6MCTS | SimpleMCTS | GumbelMCTS,
    config: TrainingConfig,
) -> list[TrainingExample]:
    """Play a single self-play game.

    Args:
        env: Game environment.
        mcts: MCTS instance.
        config: Training configuration.

    Returns:
        List of training examples from the game.
    """
    examples = []
    obs, _ = env.reset()

    game_history: list[GameHistoryItem] = []

    while True:
        # Get MCTS policy
        action_probs, _ = mcts.search(env)  # type: ignore[arg-type]

        # Store state and policy
        game_history.append(
            {
                "spatial": obs["spatial"].copy(),
                "vector": obs["vector"].copy(),
                "action_mask": obs["action_mask"].copy(),
                "policy": action_probs.copy(),
                "player": env.current_player,
            },
        )

        # Select action
        temperature = config.temperature if env.turn < config.temperature_threshold else 0.0
        action = mcts.select_action(action_probs, temperature)

        # Take action
        obs, _, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Assign values based on game outcome. Derive player count from the env
    # (authoritative for the game just played) so this works when called
    # directly on an N-player env, not only via train().
    winner = info.get("winner")
    num_players = int(getattr(env, "num_players", 2))
    method = _scoring_method_from_str(getattr(config, "scoring_method", "winner_take_all"))
    absolute_values = compute_absolute_player_values(winner, num_players, method)

    for hist in game_history:
        player = int(hist["player"])
        # Scalar target from the acting player's perspective. For 2-player
        # winner-take-all this is exactly the legacy +1 / -1 / 0 signal.
        value = float(absolute_values[player])
        # Per-player relative target vector only for N-player games.
        value_vector = (
            relative_value_vector(absolute_values, player, num_players) if num_players > 2 else None
        )

        examples.append(
            TrainingExample(
                spatial=hist["spatial"],
                vector=hist["vector"],
                action_mask=hist["action_mask"],
                policy=hist["policy"],
                value=value,
                value_vector=value_vector,
            ),
        )

    return examples


def run_self_play(
    network: torch.nn.Module,
    config: TrainingConfig,
    num_games: int,
    use_simple_mcts: bool = False,
) -> list[TrainingExample]:
    """Run multiple self-play games.

    Args:
        network: Neural network.
        config: Training configuration.
        num_games: Number of games to play.
        use_simple_mcts: If True, use SimpleMCTS instead of network-guided MCTS.

    Returns:
        List of training examples from all games.
    """
    examples = []
    env = MicroCivEnv(
        board_size=config.board_size,
        max_turns=config.max_turns,
        num_players=config.num_players,
    )

    if use_simple_mcts:
        # SimpleMCTS ignores the network entirely (random rollouts), so skip
        # building the scalar-view wrapper.
        mcts: Civ6MCTS | SimpleMCTS | GumbelMCTS = SimpleMCTS(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
        )
    elif config.use_gumbel_mcts:
        # Use Gumbel MuZero MCTS for efficient search with few simulations.
        # N-player networks emit value vectors; wrap so scalar MCTS gets a scalar.
        gumbel_config = GumbelMCTSConfig(
            num_simulations=config.gumbel_num_simulations,
            num_sampled_actions=config.gumbel_num_sampled_actions,
            c_visit=config.gumbel_c_visit,
            use_sequential_halving=config.gumbel_use_sequential_halving,
            temperature=config.temperature,
            temperature_threshold=config.temperature_threshold,
        )
        mcts = GumbelMCTS(_mcts_network(network, config), gumbel_config, device=config.device)  # type: ignore[arg-type]
        logger.info(f"Using Gumbel MuZero MCTS with {config.gumbel_num_simulations} simulations")
    else:
        # N-player networks emit value vectors; wrap so scalar MCTS gets a scalar.
        mcts_config = MCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            temperature=config.temperature,
            temperature_threshold=config.temperature_threshold,
        )
        mcts = Civ6MCTS(_mcts_network(network, config), mcts_config, device=config.device)  # type: ignore[arg-type]

    for _ in tqdm(range(num_games), desc="Self-play"):
        game_examples = self_play_game(env, mcts, config)
        examples.extend(game_examples)

    return examples


def _masked_policy_loss(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked cross-entropy policy loss.

    Identical to the policy term in :func:`compute_loss`; factored out so the
    N-player loss path can reuse it without the deprecated scalar-value bundle.
    """
    if action_mask is not None:
        target_policy = target_policy * action_mask
        target_policy = target_policy / (target_policy.sum(dim=-1, keepdim=True) + 1e-8)
    log_probs = F.log_softmax(policy_logits, dim=-1)
    return -torch.sum(target_policy * log_probs, dim=-1).mean()


def train_epoch(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> dict[str, float]:
    """Train network for one epoch.

    Args:
        network: Neural network.
        optimizer: Optimizer.
        dataloader: Training data loader.
        config: Training configuration.

    Returns:
        Dictionary of loss values.
    """
    network.train()
    total_loss = 0.0
    policy_loss = 0.0
    value_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        spatial = batch["spatial"].to(config.device)
        vector = batch["vector"].to(config.device)
        action_mask = batch["action_mask"].to(config.device)
        target_policy = batch["policy"].to(config.device)
        target_value = batch["value"].to(config.device)

        optimizer.zero_grad()

        # Forward pass
        policy_logits, value = network(spatial, vector, action_mask)

        # Compute loss. For N-player training the value tensor is per-player
        # (B, P), so use the N-player value loss; the 2-player path is
        # byte-identical to the legacy scalar loss.
        if getattr(config, "num_players", 2) > 2:
            # Guard against a scalar value head slipping through with a
            # per-player config (e.g. an NPlayerValueHead import fallback):
            # compute_nplayer_value_loss would broadcast (B,1) vs (B,P)
            # silently and train on a wrong signal. Fail loudly instead.
            if value.ndim != 2 or value.shape[-1] != config.num_players:
                raise RuntimeError(
                    f"Expected per-player value of shape (B, {config.num_players}) for "
                    f"num_players={config.num_players}, got {tuple(value.shape)}. "
                    "The network is not emitting per-player values (scalar value head?)."
                )
            p_loss = _masked_policy_loss(policy_logits, target_policy, action_mask)
            v_loss = compute_nplayer_value_loss(value, target_value, config.num_players)
            loss = p_loss + v_loss
        else:
            loss, p_loss, v_loss = compute_loss(
                policy_logits,
                value,
                target_policy,
                target_value,
                action_mask,
            )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        policy_loss += p_loss.item()
        value_loss += v_loss.item()
        num_batches += 1

    return {
        "total_loss": total_loss / num_batches,
        "policy_loss": policy_loss / num_batches,
        "value_loss": value_loss / num_batches,
    }


def evaluate_against_random(
    network: torch.nn.Module,
    config: TrainingConfig,
    num_games: int = 20,
) -> float:
    """Evaluate network against random player.

    Args:
        network: Neural network.
        config: Training configuration.
        num_games: Number of games to play.

    Returns:
        Win rate against random player.
    """
    env = MicroCivEnv(
        board_size=config.board_size,
        max_turns=config.max_turns,
        num_players=config.num_players,
    )
    mcts_config = MCTSConfig(num_simulations=config.num_simulations // 2)
    # N-player networks emit value vectors; wrap so scalar MCTS gets a scalar.
    mcts = Civ6MCTS(_mcts_network(network, config), mcts_config, device=config.device)  # type: ignore[arg-type]

    wins = 0
    rng = np.random.default_rng()

    for game_idx in range(num_games):
        _obs, _info = env.reset()
        # Rotate which seat the network occupies across all players.
        network_player = game_idx % config.num_players

        while True:
            if env.current_player == network_player:
                # Network's turn
                action_probs, _ = mcts.search(env, add_noise=False)  # type: ignore[arg-type]
                action = mcts.select_action(action_probs, temperature=0.0)
            else:
                # Random player's turn
                mask = env.get_valid_actions()
                valid = np.where(mask > 0)[0]
                action = rng.choice(valid) if len(valid) > 0 else 0

            _obs, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                winner = info.get("winner")
                if winner == network_player:
                    wins += 1
                break

    return wins / num_games


def train(config: TrainingConfig, dry_run: bool = False) -> None:
    """Main training loop.

    Args:
        config: Training configuration.
        dry_run: If True, run minimal training for testing.
    """
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Initialize environment to get action space size and (crucially) the
    # spatial-channel count, which is dynamic for N-player games
    # (3 + 4 * num_players + 2) — never hardcode it here.
    # Match the env config used in self-play/eval (incl. max_turns) so the
    # inferred action space and spatial-channel count can never desync.
    env = MicroCivEnv(
        board_size=config.board_size,
        max_turns=config.max_turns,
        num_players=config.num_players,
    )
    action_space_size = env.action_space.n
    sample_obs, _ = env.reset()
    spatial_channels = int(sample_obs["spatial"].shape[0])

    # Initialize network. When num_players > 2, MicroCivNetwork swaps in an
    # NPlayerValueHead that outputs a per-player value vector.
    network = MicroCivNetwork(
        board_size=config.board_size,
        action_space_size=int(action_space_size),
        num_res_blocks=config.num_res_blocks,
        num_filters=config.num_filters,
        spatial_channels=spatial_channels,
        num_players=config.num_players,
    ).to(config.device)

    logger.info(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    logger.info(f"Action space size: {action_space_size}")
    logger.info(
        "Players: %d | spatial channels: %d | value head: %s | scoring: %s",
        config.num_players,
        spatial_channels,
        "per-player" if config.num_players > 2 else "scalar",
        config.scoring_method,
    )
    logger.info(f"Device: {config.device}")

    # Initialize optimizer
    optimizer = optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)

    # Training loop
    num_iterations = 2 if dry_run else config.num_iterations
    num_games = 2 if dry_run else config.num_games_per_iteration

    for iteration in range(num_iterations):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Iteration {iteration + 1}/{num_iterations}")
        logger.info(f"{'=' * 50}")

        # Self-play
        logger.info("Running self-play...")
        start_time = time.time()

        # Use simple MCTS for first iteration (no trained network)
        use_simple = iteration == 0
        examples = run_self_play(network, config, num_games, use_simple_mcts=use_simple)
        replay_buffer.add_game(examples)

        self_play_time = time.time() - start_time
        logger.info(
            f"Self-play: {len(examples)} examples from {num_games} games ({self_play_time:.1f}s)",
        )

        # Training
        if len(replay_buffer) >= config.min_replay_size or dry_run:
            logger.info("Training network...")
            start_time = time.time()

            # Sample from replay buffer
            sample_size = min(len(replay_buffer), config.batch_size * 100)
            train_examples = replay_buffer.sample(sample_size)

            # Create dataloader
            dataset = ReplayDataset(train_examples)
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
            )

            # Train for multiple epochs
            num_epochs = 1 if dry_run else config.num_epochs
            for epoch in range(num_epochs):
                losses = train_epoch(network, optimizer, dataloader, config)
                logger.info(
                    f"Epoch {epoch + 1}: loss={losses['total_loss']:.4f}, "
                    f"policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f}",
                )

            train_time = time.time() - start_time
            logger.info(f"Training complete ({train_time:.1f}s)")

        # Evaluation
        if (iteration + 1) % config.save_interval == 0 or dry_run:
            logger.info("Evaluating against random...")
            win_rate = evaluate_against_random(
                network,
                config,
                num_games=10 if dry_run else 20,
            )
            logger.info(f"Win rate vs random: {win_rate:.1%}")

            # Save checkpoint
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"checkpoint_{iteration + 1}.pt",
            )
            network.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(config.checkpoint_dir, "latest.pt")
    network.save(final_path)
    logger.info(f"Training complete. Final model: {final_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Civ6 AlphaZero")
    parser.add_argument("--board_size", type=int, default=16)
    parser.add_argument(
        "--num_players",
        type=int,
        default=2,
        help="Number of players (2 = classic scalar pipeline; >2 = N-player "
        "per-player value pipeline).",
    )
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="winner_take_all",
        choices=[m.value for m in SUPPORTED_SCORING_METHODS],
        help="How terminal outcomes become per-player training values "
        "(winner-derivable methods only; 'proportional' needs per-player scores).",
    )
    parser.add_argument("--num_simulations", type=int, default=50)
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_res_blocks", type=int, default=6)
    parser.add_argument("--num_filters", type=int, default=64)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/civ6")
    parser.add_argument("--dry_run", action="store_true", help="Quick test run")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = TrainingConfig(
        board_size=args.board_size,
        num_players=args.num_players,
        scoring_method=args.scoring_method,
        num_simulations=args.num_simulations,
        num_games_per_iteration=args.num_games,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_res_blocks=args.num_res_blocks,
        num_filters=args.num_filters,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    train(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
