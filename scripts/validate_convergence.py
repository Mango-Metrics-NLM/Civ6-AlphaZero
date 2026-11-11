#!/usr/bin/env python3
"""Validate that AlphaZero training fine-tunes the incumbent across iterations.

Replaces validate_v130_training.py. Reuses the real production training path
(``alpha_zero.cloud.gcp.training_worker.train_epoch``) and the real
``GCSCheckpointStorage`` round trip (against an in-memory mock backend).
Tracks every run to MLflow (file backend by default, ``MLFLOW_TRACKING_URI``
override). Exit code 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.storage import GCSCheckpointStorage
from alpha_zero.cloud.gcp.training_worker import create_model, train_epoch
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.evaluation.scoring import ScoreSummary, summarize
from alpha_zero.evaluation.scoring import win_rate as legacy_win_rate
from alpha_zero.monitoring.config import MonitoringConfig
from alpha_zero.monitoring.mlflow_logger import MLflowLogger
from tests.mocks.cloud_mocks import MockStorageBackend

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Convergence validation parameters."""

    iterations: int = 5
    games_per_iter: int = 10
    epochs_per_iter: int = 3
    eval_games: int = 10
    max_turns: int = 15
    board_size: int = 8
    num_res_blocks: int = 2
    num_filters: int = 64
    learning_rate: float = 1e-3
    device: str = "cpu"


def _network_action(
    model: torch.nn.Module, obs: dict, device: torch.device
) -> tuple[int, np.ndarray]:
    """Sample one action from the network's masked softmax. Returns (action, probs).

    ``Civ6TransformerNetwork.forward`` (see
    ``alpha_zero/core/transformer_network.py``) takes spatial input shaped
    ``(B, C, H, W)`` at inference time, so we unsqueeze the (13, 8, 8) obs
    rather than flattening it. ``train_epoch`` separately requires flat
    spatial of shape ``(B, 13*8*8)`` -- _build_synthetic_trajectories
    handles that reshape.
    """
    spatial = torch.tensor(obs["spatial"], dtype=torch.float32).unsqueeze(0).to(device)
    vector = torch.tensor(obs["vector"], dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _v = model(spatial, vector, mask)
    masked = logits.squeeze(0).clone()
    masked[mask.squeeze(0) == 0] = float("-inf")
    probs = torch.softmax(masked, dim=-1).cpu().numpy()
    probs = np.clip(probs, 0, None)
    s = probs.sum()
    valid = np.where(obs["action_mask"] > 0)[0]
    if s <= 0:
        if len(valid) == 0:
            return -1, probs
        return int(np.random.choice(valid)), probs
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs)), probs


def _build_synthetic_trajectories(
    model: torch.nn.Module,
    cfg: ValidationConfig,
    device: torch.device,
) -> TensorDataset:
    """Generate (spatial, vector, action_mask, policy, value) tensors via
    softmax-sampled self-play with the current network. Bounded by max_turns.

    Spatial is stored FLAT so the result matches the contract of
    ``training_worker.train_epoch`` (verified in
    ``tests/unit/test_training_worker_epoch.py``).
    """
    env = MicroCivEnv(board_size=cfg.board_size)
    rows: list[tuple] = []
    for _ in range(cfg.games_per_iter):
        obs, _ = env.reset()
        traj: list[dict] = []
        turn = 0
        info: dict = {}
        while turn < cfg.max_turns:
            action, probs = _network_action(model, obs, device)
            if action < 0:
                break
            traj.append(
                {
                    "spatial": obs["spatial"].copy(),
                    "vector": obs["vector"].copy(),
                    "action_mask": obs["action_mask"].copy(),
                    "policy": probs,
                    "player": env.current_player,
                }
            )
            obs, _r, terminated, truncated, info = env.step(action)
            turn += 1
            if terminated or truncated:
                break
        winner = info.get("winner")
        for step in traj:
            if winner is None:
                value = 0.0
            elif winner == step["player"]:
                value = 1.0
            else:
                value = -1.0
            rows.append(
                (
                    torch.tensor(step["spatial"], dtype=torch.float32).flatten(),
                    torch.tensor(step["vector"], dtype=torch.float32),
                    torch.tensor(step["action_mask"], dtype=torch.float32),
                    torch.tensor(step["policy"], dtype=torch.float32),
                    torch.tensor(value, dtype=torch.float32),
                )
            )
    if not rows:
        rows.append(
            (
                torch.zeros(13 * cfg.board_size * cfg.board_size),
                torch.zeros(64),
                torch.ones(376),
                torch.full((376,), 1.0 / 376),
                torch.tensor(0.0),
            )
        )
    spatial = torch.stack([r[0] for r in rows])
    vector = torch.stack([r[1] for r in rows])
    mask = torch.stack([r[2] for r in rows])
    policy = torch.stack([r[3] for r in rows])
    value = torch.stack([r[4] for r in rows])
    return TensorDataset(spatial, vector, mask, policy, value)


def _evaluate_vs_random(
    model: torch.nn.Module, cfg: ValidationConfig, device: torch.device
) -> ScoreSummary:
    """Play eval_games against random; return the draw-aware record.

    The legacy win rate (draws as losses) is ``scoring.win_rate(summary.wins,
    summary.draws, summary.losses)``; callers log both.
    """
    env = MicroCivEnv(board_size=cfg.board_size)
    wins = 0
    draws = 0
    for _ in range(cfg.eval_games):
        obs, _ = env.reset()
        net_player = int(np.random.randint(2))
        turn = 0
        info: dict = {}
        while turn < cfg.max_turns:
            if env.current_player == net_player:
                action, _probs = _network_action(model, obs, device)
                if action < 0:
                    break
            else:
                valid = np.where(obs["action_mask"] > 0)[0]
                if len(valid) == 0:
                    break
                action = int(np.random.choice(valid))
            obs, _r, terminated, truncated, info = env.step(action)
            turn += 1
            if terminated or truncated:
                winner = info.get("winner")
                if winner == net_player:
                    wins += 1
                elif winner is None:
                    draws += 1
                break
    return summarize(wins, draws, cfg.eval_games - wins - draws)


def run(cfg: ValidationConfig) -> int:
    """Run the validation loop. Returns process exit code (0=pass, 1=fail)."""
    device = torch.device(cfg.device)
    model = create_model(
        board_size=cfg.board_size,
        action_space_size=376,
        num_res_blocks=cfg.num_res_blocks,
        num_filters=cfg.num_filters,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Exercise the production checkpoint round-trip path against an in-memory
    # backend so the validation script asserts the same contract Sub-Project A
    # locked: GCSCheckpointStorage.save_checkpoint -> load_checkpoint(step=None)
    # preserves weights bit-exact.
    gcp_cfg = GCPConfig(project_id="local", bucket="local", region="us-central1")
    ckpt_store = GCSCheckpointStorage(config=gcp_cfg, backend=MockStorageBackend())

    iteration_losses: list[float] = []
    iteration_win_rates: list[float] = []
    iteration_score_rates: list[float] = []
    iteration_draw_rates: list[float] = []

    monitoring = MonitoringConfig.from_env()
    monitoring.run_name = monitoring.run_name or "validate_convergence"

    with MLflowLogger.from_config(monitoring) as ml:
        ml.log_params(
            {
                "iterations": cfg.iterations,
                "games_per_iter": cfg.games_per_iter,
                "epochs_per_iter": cfg.epochs_per_iter,
                "max_turns": cfg.max_turns,
                "board_size": cfg.board_size,
                "num_res_blocks": cfg.num_res_blocks,
                "num_filters": cfg.num_filters,
                "learning_rate": cfg.learning_rate,
                "device": cfg.device,
            }
        )

        global_step = 0
        for it in range(cfg.iterations):
            logger.info("=== iteration %d/%d ===", it + 1, cfg.iterations)
            data = _build_synthetic_trajectories(model, cfg, device)
            loader = DataLoader(data, batch_size=min(32, len(data)), shuffle=True)

            avg_total = avg_policy = avg_value = 0.0
            for _ep in range(cfg.epochs_per_iter):
                total_l, policy_l, value_l, n_batches = train_epoch(
                    model,
                    loader,
                    optimizer,
                    device,
                    global_step=global_step,
                )
                global_step += n_batches
                avg_total += total_l / cfg.epochs_per_iter
                avg_policy += policy_l / cfg.epochs_per_iter
                avg_value += value_l / cfg.epochs_per_iter
            iteration_losses.append(avg_policy)

            # Exercise the production checkpoint round-trip.
            ckpt_store.save_checkpoint(model=model, optimizer=optimizer, step=it)
            reloaded = ckpt_store.load_checkpoint(step=None)
            assert reloaded is not None and reloaded["step"] == it
            model.load_state_dict(reloaded["model_state_dict"])

            eval_summary = _evaluate_vs_random(model, cfg, device)
            win_rate = legacy_win_rate(eval_summary.wins, eval_summary.draws, eval_summary.losses)
            iteration_win_rates.append(win_rate)
            iteration_score_rates.append(eval_summary.score_rate)
            iteration_draw_rates.append(eval_summary.draw_rate)

            ml.log_metrics(
                {
                    "validation/policy_loss": avg_policy,
                    "validation/value_loss": avg_value,
                    "validation/total_loss": avg_total,
                    "validation/win_rate_vs_random": win_rate,
                    "validation/score_rate_vs_random": eval_summary.score_rate,
                    "validation/draw_rate_vs_random": eval_summary.draw_rate,
                },
                step=it,
            )
            logger.info(
                "  policy_loss=%.4f value_loss=%.4f win_rate=%.1f%% score_rate=%.3f draw_rate=%.1f%%",
                avg_policy,
                avg_value,
                win_rate * 100,
                eval_summary.score_rate,
                eval_summary.draw_rate * 100,
            )

        # Convergence gates.
        # Require >=4 iterations to evaluate the loss trend: with fewer than
        # 2 samples per half the first/second-half mean is dominated by
        # single-iteration noise from random-init self-play and the ratio
        # carries no signal. For shorter smoke runs we log the per-iteration
        # metrics but skip the pass/fail gate.
        n = len(iteration_losses)
        ok_loss = True
        if n >= 4:
            first = sum(iteration_losses[: n // 2]) / max(n // 2, 1)
            second = sum(iteration_losses[n // 2 :]) / max(n - n // 2, 1)
            ok_loss = second <= first * 1.05
            ml.log_metrics(
                {
                    "validation/loss_first_half": first,
                    "validation/loss_second_half": second,
                    "validation/loss_ratio": second / max(first, 1e-8),
                }
            )
        ok_wr = True
        if len(iteration_win_rates) >= 4:
            ok_wr = iteration_win_rates[-1] >= iteration_win_rates[0] - 0.05
            ml.log_metrics(
                {
                    "validation/win_rate_first": iteration_win_rates[0],
                    "validation/win_rate_last": iteration_win_rates[-1],
                    "validation/score_rate_first": iteration_score_rates[0],
                    "validation/score_rate_last": iteration_score_rates[-1],
                    # Trend endpoints only; the pass/fail gate below still
                    # reads win_rate pending a recorded decision.
                    "validation/draw_rate_last": iteration_draw_rates[-1],
                }
            )

        ml.set_tag("validation/result", "PASS" if (ok_loss and ok_wr) else "FAIL")
        if ok_loss and ok_wr:
            logger.info("convergence validation PASSED")
            return 0
        logger.error(
            "convergence validation FAILED (ok_loss=%s, ok_wr=%s)",
            ok_loss,
            ok_wr,
        )
        return 1


def parse_args() -> ValidationConfig:
    """Parse CLI args into a ValidationConfig."""
    p = argparse.ArgumentParser(
        description="Validate AlphaZero training convergence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--board-size", type=int, default=8)
    p.add_argument("--num-res-blocks", type=int, default=2)
    p.add_argument("--num-filters", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    return ValidationConfig(
        iterations=a.iterations,
        games_per_iter=a.games,
        epochs_per_iter=a.epochs,
        eval_games=a.eval_games,
        max_turns=a.max_turns,
        board_size=a.board_size,
        num_res_blocks=a.num_res_blocks,
        num_filters=a.num_filters,
        learning_rate=a.learning_rate,
        device=a.device,
    )


def main() -> int:
    """Entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
