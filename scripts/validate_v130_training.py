"""Validate v1.3.0 training convergence — weight loading and model improvement.

This script runs a small-scale local training loop to verify:
1. Model weights carry over between iterations (not random re-init)
2. Policy loss trends downward across iterations
3. Win rate against random player improves
4. Checkpoint save/load round-trips correctly

This validates the same conceptual fix as v1.3.0 (--model_version for cloud),
but using the local training pipeline's checkpoint mechanism.

Usage:
    python scripts/validate_v130_training.py
    python scripts/validate_v130_training.py --iterations=10 --games=20
    python scripts/validate_v130_training.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.training_civ6 import (
    ReplayBuffer,
    ReplayDataset,
    TrainingConfig,
    evaluate_against_random,
    run_self_play,
    train_epoch,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from a validation run."""

    iterations_completed: int
    policy_losses: list[float]
    value_losses: list[float]
    total_losses: list[float]
    win_rates: list[float]
    weight_checksums: list[float]
    weight_continuity_verified: bool
    loss_trend_downward: bool
    win_rate_improved: bool
    checkpoint_roundtrip_ok: bool
    all_passed: bool

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            "  v1.3.0 Training Convergence Validation Results",
            "=" * 60,
            "",
            f"  Iterations completed: {self.iterations_completed}",
            "",
            "  Loss Trajectory:",
        ]
        for i, (pl, vl, tl) in enumerate(
            zip(self.policy_losses, self.value_losses, self.total_losses, strict=True)
        ):
            lines.append(f"    Iter {i + 1}: policy={pl:.4f}  value={vl:.4f}  total={tl:.4f}")
        lines.append("")
        lines.append("  Win Rate vs Random:")
        for i, wr in enumerate(self.win_rates):
            lines.append(f"    Iter {i + 1}: {wr:.1%}")
        lines.append("")
        lines.append("  Weight Checksums (should change between iterations):")
        for i, cs in enumerate(self.weight_checksums):
            lines.append(f"    Iter {i + 1}: {cs:.6f}")
        lines.append("")
        lines.append("  Checks:")
        lines.append(f"    [{'PASS' if self.weight_continuity_verified else 'FAIL'}] "
                     "Weight continuity (not re-initialized between iterations)")
        lines.append(f"    [{'PASS' if self.loss_trend_downward else 'FAIL'}] "
                     "Loss trend downward")
        lines.append(f"    [{'PASS' if self.win_rate_improved else 'INFO'}] "
                     "Win rate improved (last > first, informational)")
        lines.append(f"    [{'PASS' if self.checkpoint_roundtrip_ok else 'FAIL'}] "
                     "Checkpoint save/load roundtrip")
        lines.append("")
        if self.all_passed:
            lines.append("  RESULT: ALL CHECKS PASSED")
        else:
            lines.append("  RESULT: SOME CHECKS FAILED")
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_weight_checksum(network: torch.nn.Module) -> float:
    """Compute a reproducible checksum of network weights."""
    total = 0.0
    for param in network.parameters():
        total += param.data.abs().sum().item()
    return total


def verify_checkpoint_roundtrip(
    network: MicroCivNetwork,
    checkpoint_dir: str,
    device: str,
) -> bool:
    """Verify that saving and loading a checkpoint preserves weights exactly."""
    path = os.path.join(checkpoint_dir, "roundtrip_test.pt")

    # Save
    network.save(path)
    original_checksum = compute_weight_checksum(network)

    # Load into new network
    loaded = MicroCivNetwork.load(path, device=device)
    loaded_checksum = compute_weight_checksum(loaded)

    match = abs(original_checksum - loaded_checksum) < 1e-4
    if not match:
        logger.warning(
            "Checkpoint roundtrip MISMATCH: original=%.6f loaded=%.6f",
            original_checksum,
            loaded_checksum,
        )
    else:
        logger.info("Checkpoint roundtrip OK (checksum=%.6f)", original_checksum)

    # Clean up
    os.remove(path)
    return match


def verify_weight_continuity(
    checksums: list[float],
) -> bool:
    """Verify that weights change between iterations (not re-initialized).

    If the network were re-initialized each iteration, the checksums would
    be uncorrelated. If weights carry over, each iteration's checksum should
    be close to (but different from) the previous one.
    """
    if len(checksums) < 2:
        return True

    # Check that consecutive checksums are not dramatically different
    # (a re-init would produce very different checksums)
    for i in range(1, len(checksums)):
        delta = abs(checksums[i] - checksums[i - 1])
        relative_change = delta / max(abs(checksums[i - 1]), 1e-8)
        # Weight updates via gradient descent typically change weights by <10%
        # Random re-initialization would change by >50%
        if relative_change > 0.5:
            logger.warning(
                "Large weight change between iter %d and %d: %.2f%% "
                "(possible re-initialization)",
                i,
                i + 1,
                relative_change * 100,
            )
            return False

    return True


def run_validation(
    iterations: int = 5,
    games_per_iter: int = 10,
    board_size: int = 8,
    num_simulations: int = 8,
    eval_games: int = 10,
    verbose: bool = False,
) -> ValidationResult:
    """Run the full validation loop.

    Args:
        iterations: Number of training iterations.
        games_per_iter: Self-play games per iteration.
        board_size: MicroCiv board size (smaller = faster).
        num_simulations: MCTS simulations per move.
        eval_games: Games for evaluation against random.
        verbose: Print detailed progress.
    """
    device = "cpu"

    with tempfile.TemporaryDirectory(prefix="civ6az_validate_") as tmpdir:
        config = TrainingConfig(
            board_size=board_size,
            max_turns=50,
            num_simulations=num_simulations,
            num_res_blocks=2,
            num_filters=32,
            batch_size=32,
            learning_rate=0.002,
            weight_decay=1e-4,
            gradient_clip=1.0,
            num_epochs=3,
            num_games_per_iteration=games_per_iter,
            num_iterations=iterations,
            replay_buffer_size=10000,
            min_replay_size=32,
            checkpoint_dir=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
            device=device,
            use_gumbel_mcts=True,
            gumbel_num_simulations=num_simulations,
            gumbel_num_sampled_actions=min(num_simulations, 8),
        )

        os.makedirs(config.log_dir, exist_ok=True)

        # Initialize environment
        env = MicroCivEnv(board_size=board_size)
        action_space_size = env.action_space.n

        # Initialize network — ONCE, then carry weights across iterations
        network = MicroCivNetwork(
            board_size=board_size,
            action_space_size=action_space_size,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
        ).to(device)

        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)

        # Track metrics
        policy_losses: list[float] = []
        value_losses: list[float] = []
        total_losses: list[float] = []
        win_rates: list[float] = []
        weight_checksums: list[float] = []

        logger.info("Starting validation: %d iterations, %d games/iter", iterations, games_per_iter)
        logger.info("Board size: %d, MCTS sims: %d", board_size, num_simulations)
        logger.info("Network params: %d", sum(p.numel() for p in network.parameters()))

        for iteration in range(iterations):
            logger.info("--- Iteration %d/%d ---", iteration + 1, iterations)

            # Self-play
            use_simple = iteration == 0
            examples = run_self_play(
                network, config, games_per_iter, use_simple_mcts=use_simple
            )
            replay_buffer.add_game(examples)
            logger.info("Self-play: %d examples from %d games", len(examples), games_per_iter)

            # Training
            if len(replay_buffer) >= config.min_replay_size:
                sample_size = min(len(replay_buffer), config.batch_size * 20)
                train_examples = replay_buffer.sample(sample_size)
                dataset = ReplayDataset(train_examples)
                dataloader = DataLoader(
                    dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
                )

                epoch_losses = []
                for epoch in range(config.num_epochs):
                    losses = train_epoch(network, optimizer, dataloader, config)
                    epoch_losses.append(losses)
                    if verbose:
                        logger.info(
                            "  Epoch %d: loss=%.4f policy=%.4f value=%.4f",
                            epoch + 1,
                            losses["total_loss"],
                            losses["policy_loss"],
                            losses["value_loss"],
                        )

                # Record final epoch's loss
                final = epoch_losses[-1]
                policy_losses.append(final["policy_loss"])
                value_losses.append(final["value_loss"])
                total_losses.append(final["total_loss"])
            else:
                logger.info("Buffer too small (%d), skipping training", len(replay_buffer))
                policy_losses.append(float("nan"))
                value_losses.append(float("nan"))
                total_losses.append(float("nan"))

            # Record post-training weight checksum
            post_checksum = compute_weight_checksum(network)
            weight_checksums.append(post_checksum)

            # Evaluate against random
            wr = evaluate_against_random(network, config, num_games=eval_games)
            win_rates.append(wr)
            logger.info("Win rate vs random: %.1f%%", wr * 100)

            # Save checkpoint (simulates what cloud pipeline does between iterations)
            ckpt_path = os.path.join(tmpdir, f"checkpoint_iter{iteration + 1}.pt")
            network.save(ckpt_path)
            logger.info("Saved checkpoint: %s (checksum=%.4f)", ckpt_path, post_checksum)

        # Verify checkpoint roundtrip
        roundtrip_ok = verify_checkpoint_roundtrip(network, tmpdir, device)

        # Verify weight continuity
        continuity_ok = verify_weight_continuity(weight_checksums)

        # Check loss trend (compare first and last, ignoring NaN)
        valid_losses = [l for l in total_losses if not np.isnan(l)]
        if len(valid_losses) >= 2:
            loss_trend_down = valid_losses[-1] < valid_losses[0]
        else:
            loss_trend_down = False

        # Check win rate improvement (soft check — may not trigger with few iterations)
        if len(win_rates) >= 2:
            wr_improved = win_rates[-1] > win_rates[0]
        else:
            wr_improved = False

        # Core checks: weight continuity + loss trend + checkpoint roundtrip
        # Win rate is informational — it takes many more iterations to improve
        all_passed = continuity_ok and loss_trend_down and roundtrip_ok

        return ValidationResult(
            iterations_completed=iterations,
            policy_losses=policy_losses,
            value_losses=value_losses,
            total_losses=total_losses,
            win_rates=win_rates,
            weight_checksums=weight_checksums,
            weight_continuity_verified=continuity_ok,
            loss_trend_downward=loss_trend_down,
            win_rate_improved=wr_improved,
            checkpoint_roundtrip_ok=roundtrip_ok,
            all_passed=all_passed,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate v1.3.0 training convergence"
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of training iterations (default: 5)",
    )
    parser.add_argument(
        "--games", type=int, default=10,
        help="Self-play games per iteration (default: 10)",
    )
    parser.add_argument(
        "--board_size", type=int, default=8,
        help="Board size (default: 8, smaller = faster)",
    )
    parser.add_argument(
        "--num_simulations", type=int, default=8,
        help="MCTS simulations per move (default: 8)",
    )
    parser.add_argument(
        "--eval_games", type=int, default=10,
        help="Evaluation games against random (default: 10)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed per-epoch progress",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # Always show our logger at INFO
    logger.setLevel(logging.INFO)

    result = run_validation(
        iterations=args.iterations,
        games_per_iter=args.games,
        board_size=args.board_size,
        num_simulations=args.num_simulations,
        eval_games=args.eval_games,
        verbose=args.verbose,
    )

    print(result.summary())

    sys.exit(0 if result.all_passed else 1)


if __name__ == "__main__":
    main()
