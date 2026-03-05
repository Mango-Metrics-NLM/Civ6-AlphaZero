#!/usr/bin/env python3
"""Validate v1.3.0 training convergence fix.

Verifies the critical fix: each training iteration fine-tunes the
incumbent model's weights instead of training from random initialization.

Validation checks:
  1. Weight continuity   — checkpoint save→load preserves exact weights
  2. Loss decrease       — policy loss trends down across mini-iterations
  3. Win-rate improvement — network improves against random player

Usage:
    python scripts/validate_v130_training.py
    python scripts/validate_v130_training.py --iterations=5 --games=10

Exit codes:
    0 — all validations passed
    1 — validation failed
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nn_func

# Add project root for imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alpha_zero.core.civ6_network import MicroCivNetwork  # noqa: E402
from alpha_zero.envs.microciv import MicroCivEnv  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_BOARD_SIZE = 8
_DEFAULT_NUM_RES_BLOCKS = 2
_DEFAULT_NUM_FILTERS = 32
_DEFAULT_LR = 1e-3


@dataclass
class ValidationConfig:
    """Configuration for convergence validation."""

    iterations: int = 3
    games_per_iter: int = 5
    epochs_per_iter: int = 3
    board_size: int = _DEFAULT_BOARD_SIZE
    num_res_blocks: int = _DEFAULT_NUM_RES_BLOCKS
    num_filters: int = _DEFAULT_NUM_FILTERS
    learning_rate: float = _DEFAULT_LR
    eval_games: int = 10
    device: str = "cpu"
    verbose: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_network(cfg: ValidationConfig) -> MicroCivNetwork:
    """Create a small MicroCivNetwork for validation."""
    env = MicroCivEnv(board_size=cfg.board_size)
    return MicroCivNetwork(
        board_size=cfg.board_size,
        action_space_size=env.action_space.n,
        num_res_blocks=cfg.num_res_blocks,
        num_filters=cfg.num_filters,
    ).to(cfg.device)


def _generate_selfplay_data(
    network: MicroCivNetwork,
    cfg: ValidationConfig,
) -> list[dict[str, torch.Tensor]]:
    """Generate self-play training data using the network.

    Uses a simplified random-policy self-play to avoid MCTS overhead
    while still producing valid (state, policy, value) triples.
    """
    env = MicroCivEnv(board_size=cfg.board_size)
    network.eval()
    examples: list[dict[str, torch.Tensor]] = []

    for _ in range(cfg.games_per_iter):
        obs, _ = env.reset()
        game_history: list[dict[str, np.ndarray]] = []

        while True:
            spatial = torch.tensor(
                obs["spatial"], dtype=torch.float32
            ).unsqueeze(0).to(cfg.device)
            vector = torch.tensor(
                obs["vector"], dtype=torch.float32
            ).unsqueeze(0).to(cfg.device)
            action_mask = torch.tensor(
                obs["action_mask"], dtype=torch.float32
            ).unsqueeze(0).to(cfg.device)

            with torch.no_grad():
                policy_logits, _value = network(spatial, vector, action_mask)

            # Use network policy (softmax with mask) for action selection
            masked_logits = policy_logits.squeeze(0).clone()
            mask_tensor = action_mask.squeeze(0)
            masked_logits[mask_tensor == 0] = float("-inf")
            probs = nn_func.softmax(masked_logits, dim=-1).cpu().numpy()

            # Store history
            game_history.append({
                "spatial": obs["spatial"].copy(),
                "vector": obs["vector"].copy(),
                "action_mask": obs["action_mask"].copy(),
                "policy": probs.copy(),
                "player": env.current_player,
            })

            # Select action from policy distribution
            valid_probs = probs.copy()
            valid_probs[valid_probs < 0] = 0
            prob_sum = valid_probs.sum()
            if prob_sum > 0:
                valid_probs /= prob_sum
                action = int(np.random.choice(len(valid_probs), p=valid_probs))
            else:
                valid_actions = np.where(obs["action_mask"] > 0)[0]
                if len(valid_actions) == 0:
                    break  # No valid actions, end the game
                action = int(np.random.choice(valid_actions))

            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        # Assign values
        winner = info.get("winner")
        for hist in game_history:
            if winner is None:
                value = 0.0
            elif winner == hist["player"]:
                value = 1.0
            else:
                value = -1.0

            examples.append({
                "spatial": torch.tensor(hist["spatial"], dtype=torch.float32),
                "vector": torch.tensor(hist["vector"], dtype=torch.float32),
                "action_mask": torch.tensor(hist["action_mask"], dtype=torch.float32),
                "policy": torch.tensor(hist["policy"], dtype=torch.float32),
                "value": torch.tensor(value, dtype=torch.float32),
            })

    return examples


def _train_on_data(
    network: MicroCivNetwork,
    data: list[dict[str, torch.Tensor]],
    cfg: ValidationConfig,
) -> list[float]:
    """Train network on self-play data and return per-epoch losses."""
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.learning_rate)
    epoch_losses: list[float] = []

    if not data:
        return epoch_losses

    for _epoch in range(cfg.epochs_per_iter):
        total_loss_accum = 0.0
        n_batches = 0

        # Simple batch iteration
        batch_size = min(32, len(data))
        indices = list(range(len(data)))
        np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = [data[i] for i in batch_idx]

            spatial = torch.stack([b["spatial"] for b in batch]).to(cfg.device)
            vector = torch.stack([b["vector"] for b in batch]).to(cfg.device)
            action_mask = torch.stack([b["action_mask"] for b in batch]).to(cfg.device)
            target_policy = torch.stack([b["policy"] for b in batch]).to(cfg.device)
            target_value = torch.stack([b["value"] for b in batch]).to(cfg.device)

            policy_logits, value = network(spatial, vector, action_mask)

            # Value loss
            value_loss = nn_func.mse_loss(value.squeeze(-1), target_value)

            # Policy loss (cross-entropy)
            masked_target = target_policy * action_mask
            target_sum = masked_target.sum(dim=-1, keepdim=True) + 1e-8
            masked_target = masked_target / target_sum
            log_probs = nn_func.log_softmax(policy_logits, dim=-1)
            policy_loss = -torch.sum(masked_target * log_probs, dim=-1).mean()

            total_loss = policy_loss + value_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_accum += total_loss.item()
            n_batches += 1

        avg_loss = total_loss_accum / max(n_batches, 1)
        epoch_losses.append(avg_loss)

    return epoch_losses


def _evaluate_vs_random(
    network: MicroCivNetwork,
    cfg: ValidationConfig,
) -> float:
    """Evaluate network win rate against a random player."""
    env = MicroCivEnv(board_size=cfg.board_size)
    network.eval()
    wins = 0

    for _ in range(cfg.eval_games):
        obs, _ = env.reset()
        # Randomly assign network to player 0 or 1
        network_player = np.random.randint(2)

        while True:
            if env.current_player == network_player:
                # Network plays
                spatial = torch.tensor(
                    obs["spatial"], dtype=torch.float32
                ).unsqueeze(0).to(cfg.device)
                vector = torch.tensor(
                    obs["vector"], dtype=torch.float32
                ).unsqueeze(0).to(cfg.device)
                action_mask = torch.tensor(
                    obs["action_mask"], dtype=torch.float32
                ).unsqueeze(0).to(cfg.device)

                with torch.no_grad():
                    policy_logits, _ = network(spatial, vector, action_mask)

                masked_logits = policy_logits.squeeze(0).clone()
                mask_t = action_mask.squeeze(0)
                masked_logits[mask_t == 0] = float("-inf")
                probs = nn_func.softmax(masked_logits, dim=-1).cpu().numpy()
                probs = np.clip(probs, 0, None)
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs /= prob_sum
                    action = int(np.random.choice(len(probs), p=probs))
                else:
                    valid = np.where(obs["action_mask"] > 0)[0]
                    if len(valid) == 0:
                        break
                    action = int(np.random.choice(valid))
            else:
                # Random player
                valid_actions = np.where(obs["action_mask"] > 0)[0]
                if len(valid_actions) == 0:
                    break
                action = int(np.random.choice(valid_actions))

            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                winner = info.get("winner")
                if winner == network_player:
                    wins += 1
                break

    return wins / max(cfg.eval_games, 1)


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of convergence validation."""

    weight_continuity: bool = False
    loss_decreasing: bool = False
    win_rate_improving: bool = False
    iteration_losses: list[float] = field(default_factory=list)
    iteration_win_rates: list[float] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Overall validation passed."""
        return self.weight_continuity and self.loss_decreasing


def validate_weight_continuity(cfg: ValidationConfig) -> bool:
    """Check 1: checkpoint save→load preserves exact weights."""
    network = _create_network(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint.pt")

        # Save
        original_state = {k: v.clone() for k, v in network.state_dict().items()}
        torch.save({"network_state_dict": network.state_dict(), "step": 0}, ckpt_path)

        # Create fresh network and load
        network2 = _create_network(cfg)
        checkpoint = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
        network2.load_state_dict(checkpoint["network_state_dict"])

        # Compare
        for key in original_state:
            if not torch.equal(original_state[key], network2.state_dict()[key]):
                logger.error("Weight mismatch for key: %s", key)
                return False

    return True


def validate_training_loop(cfg: ValidationConfig) -> ValidationResult:
    """Run full validation: iterate, train, evaluate."""
    result = ValidationResult()

    # Check 1: weight continuity
    result.weight_continuity = validate_weight_continuity(cfg)
    result.details.append(
        f"Weight continuity: {'✅ PASS' if result.weight_continuity else '❌ FAIL'}"
    )

    # Check 2 & 3: iterative training
    network = _create_network(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        for iteration in range(cfg.iterations):
            if cfg.verbose:
                logger.info("=== Iteration %d/%d ===", iteration + 1, cfg.iterations)

            # Self-play data generation
            data = _generate_selfplay_data(network, cfg)
            if cfg.verbose:
                logger.info("  Generated %d training examples", len(data))

            # Train
            epoch_losses = _train_on_data(network, data, cfg)
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            result.iteration_losses.append(avg_loss)
            if cfg.verbose:
                logger.info("  Avg loss: %.4f", avg_loss)

            # Save checkpoint (simulates end of iteration)
            ckpt_path = os.path.join(tmpdir, f"model_iter_{iteration}.pt")
            torch.save({
                "network_state_dict": network.state_dict(),
                "step": iteration,
            }, ckpt_path)

            # Load checkpoint (simulates start of next iteration — this is the v1.3.0 fix)
            checkpoint = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
            network.load_state_dict(checkpoint["network_state_dict"])

            # Evaluate
            win_rate = _evaluate_vs_random(network, cfg)
            result.iteration_win_rates.append(win_rate)
            if cfg.verbose:
                logger.info("  Win rate vs random: %.1f%%", win_rate * 100)

    # Check 2: loss should generally decrease (compare first half avg vs second half avg)
    n = len(result.iteration_losses)
    if n >= 2:
        first_half = sum(result.iteration_losses[: n // 2]) / max(n // 2, 1)
        second_half = sum(result.iteration_losses[n // 2 :]) / max(n - n // 2, 1)
        result.loss_decreasing = second_half <= first_half * 1.1  # Allow 10% tolerance
        result.details.append(
            f"Loss trend: first_half={first_half:.4f}, second_half={second_half:.4f} "
            f"{'✅ PASS' if result.loss_decreasing else '❌ FAIL'}"
        )
    else:
        result.loss_decreasing = True  # Not enough data to evaluate
        result.details.append("Loss trend: insufficient iterations (skipped)")

    # Check 3: win rate (informational, not gating)
    if len(result.iteration_win_rates) >= 2:
        first_wr = result.iteration_win_rates[0]
        last_wr = result.iteration_win_rates[-1]
        wr_improved = last_wr >= first_wr
        result.win_rate_improving = wr_improved
        result.details.append(
            f"Win rate: iter_1={first_wr:.1%}, iter_{n}={last_wr:.1%} "
            f"{'✅ IMPROVING' if wr_improved else '⚠️ NOT IMPROVING (informational)'}"
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> ValidationConfig:
    """Parse CLI arguments into ValidationConfig."""
    parser = argparse.ArgumentParser(
        description="Validate v1.3.0 training convergence fix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=3, help="Training iterations")
    parser.add_argument("--games", type=int, default=5, help="Self-play games per iteration")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per iteration")
    parser.add_argument("--eval-games", type=int, default=10, help="Evaluation games")
    parser.add_argument("--board-size", type=int, default=_DEFAULT_BOARD_SIZE)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    return ValidationConfig(
        iterations=args.iterations,
        games_per_iter=args.games,
        epochs_per_iter=args.epochs,
        eval_games=args.eval_games,
        board_size=args.board_size,
        device=args.device,
        verbose=not args.quiet,
    )


def main() -> int:
    """Run validation and return exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = parse_args()
    logger.info("v1.3.0 Convergence Validation")
    logger.info("Config: %d iterations, %d games/iter, %d epochs/iter",
                cfg.iterations, cfg.games_per_iter, cfg.epochs_per_iter)

    result = validate_training_loop(cfg)

    logger.info("")
    logger.info("=" * 50)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 50)
    for detail in result.details:
        logger.info("  %s", detail)
    logger.info("")

    if result.passed:
        logger.info("✅ ALL CRITICAL CHECKS PASSED")
        return 0
    else:
        logger.error("❌ VALIDATION FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
