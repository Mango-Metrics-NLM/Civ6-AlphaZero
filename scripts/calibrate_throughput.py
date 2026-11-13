#!/usr/bin/env python3
"""Measure self-play throughput on this machine (calibration harness).

Plays a few complete seeded self-play games through the same code path as
``alpha_zero.training_civ6`` (``run_self_play``) and reports network-evaluation
latency, actions per game and seconds per game. This is a *harness*: its output
is a measurement of the machine and the current code, never a wall-clock
estimate for a training run. Estimates come from a calibration on post-fix code
(plan task T0b).

Usage:
    python scripts/calibrate_throughput.py --board_size 8 --max_turns 60 \
        --mcts gumbel --k 8 --n 24 --device cpu --games 2 --out calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Run as a script, sys.path[0] is scripts/, so make this checkout win over any
# separately installed copy of the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.training_civ6 import TrainingConfig, run_self_play
from alpha_zero.utils.device import resolve_device_str
from alpha_zero.utils.game_budget import add_budget_arguments, enforce_cli
from alpha_zero.utils.seeding import derive_seed

logger = logging.getLogger("calibrate_throughput")

MCTS_CHOICES = ("civ6", "gumbel")
CALIBRATION_STREAM = "calibrate.throughput"


def calibrate(
    board_size: int,
    max_turns: int,
    mcts: str,
    k: int,
    n: int,
    device: str,
    games: int = 2,
    seed: int | None = 0,
) -> dict[str, float]:
    """Play ``games`` self-play games and return throughput figures.

    Args:
        board_size: MicroCiv board edge length.
        max_turns: Game-turn limit.
        mcts: ``civ6`` (PUCT) or ``gumbel``.
        k: Gumbel sampled actions (ignored for ``civ6``).
        n: Simulations per move.
        device: Torch device string.
        games: Number of complete games to play.
        seed: Base seed; ``None`` leaves the run unseeded.

    Returns:
        ``ms_per_eval`` (mean wall-clock per network forward, including the
        search around it), ``actions_per_game``, ``s_per_game``, plus raw
        ``total_evals``, ``total_actions`` and ``total_seconds``.
    """
    if mcts not in MCTS_CHOICES:
        raise ValueError(f"mcts must be one of {MCTS_CHOICES}, got {mcts!r}")
    if games < 1:
        raise ValueError("games must be >= 1")
    device = resolve_device_str(device)
    run_seed = derive_seed(seed, CALIBRATION_STREAM)
    config = TrainingConfig(
        board_size=board_size,
        max_turns=max_turns,
        num_simulations=n,
        use_gumbel_mcts=(mcts == "gumbel"),
        gumbel_num_simulations=n,
        gumbel_num_sampled_actions=k,
        device=device,
        seed=run_seed,
    )

    env = MicroCivEnv(board_size=board_size, max_turns=max_turns, seed=run_seed)
    obs, _ = env.reset(seed=run_seed)
    network = _build_network(config, env, obs)

    evals = 0
    original_predict = network.predict  # type: ignore[operator]

    def _counting_predict(*args: Any, **kwargs: Any) -> Any:
        nonlocal evals
        evals += 1
        return original_predict(*args, **kwargs)

    network.predict = _counting_predict  # type: ignore[method-assign]
    network.eval()
    try:
        start = time.perf_counter()
        examples = run_self_play(network, config, games)
        elapsed = time.perf_counter() - start
    finally:
        network.predict = original_predict  # type: ignore[method-assign]

    actions = len(examples)
    return {
        "ms_per_eval": 1000.0 * elapsed / max(evals, 1),
        "actions_per_game": actions / games,
        "s_per_game": elapsed / games,
        "total_evals": float(evals),
        "total_actions": float(actions),
        "total_seconds": elapsed,
    }


def _build_network(config: TrainingConfig, env: MicroCivEnv, obs: Any) -> torch.nn.Module:
    from alpha_zero.core.civ6_network import MicroCivNetwork

    if config.seed is not None:
        torch.manual_seed(config.seed)
    return MicroCivNetwork(
        board_size=config.board_size,
        action_space_size=int(env.action_space.n),
        num_res_blocks=config.num_res_blocks,
        num_filters=config.num_filters,
        spatial_channels=int(obs["spatial"].shape[0]),
        num_players=config.num_players,
    ).to(config.device)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("--board_size", type=int, default=8)
    p.add_argument("--max_turns", type=int, default=60)
    p.add_argument("--mcts", choices=MCTS_CHOICES, default="gumbel")
    p.add_argument("--k", type=int, default=8, help="Gumbel sampled actions")
    p.add_argument("--n", type=int, default=24, help="simulations per move")
    p.add_argument("--device", default="cpu")
    p.add_argument("--games", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="write JSON here")
    add_budget_arguments(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    # Game budget: refuse before the network is built or any game is played.
    enforce_cli("calibrate_throughput", {"games": args.games}, args)
    result = calibrate(
        args.board_size,
        args.max_turns,
        args.mcts,
        args.k,
        args.n,
        args.device,
        games=args.games,
        seed=args.seed,
    )
    payload: dict[str, Any] = {
        "harness": "calibrate_throughput",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "results": result,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        logger.info("wrote %s", args.out)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
