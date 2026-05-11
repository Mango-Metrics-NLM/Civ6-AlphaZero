#!/usr/bin/env python3
"""Thin chess-vs-Stockfish entry point.

Loads ``configs/chess_config.yaml`` (or a user-supplied path), wires the
``ChessEnv`` + ``StockfishOpponent`` + ``StockfishCurriculum`` + optional
``ChessAdvisor`` together, and runs N iterations.

This is intentionally minimal — it exists to make the end-to-end demo
runnable with a single command. It does not implement the full
self-play/learner loop; that lives in the orchestrator and will be wired
in a follow-up. ``--dry-run`` validates the configuration and exits
without playing, suitable for CI smoke tests.

Usage::

    python scripts/run_chess_training.py --config configs/chess_config.yaml
    python scripts/run_chess_training.py --dry-run
    python scripts/run_chess_training.py --iterations 1 --games-per-iter 2 \\
        --skill 0 --no-advisor
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _build_opponent(cfg: dict[str, Any]):  # type: ignore[no-untyped-def]
    from alpha_zero.opponents.stockfish_opponent import (
        StockfishOpponent,
        StockfishOpponentConfig,
    )

    opp_cfg = StockfishOpponentConfig(
        path=cfg.get("path") or os.getenv("STOCKFISH_PATH"),
        skill=int(cfg.get("skill", 5)),
        elo=int(cfg.get("elo", 1500)),
        use_elo=bool(cfg.get("use_elo", False)),
        depth_limit=cfg.get("depth_limit"),
        time_limit_s=cfg.get("time_limit_s"),
        node_limit=cfg.get("node_limit"),
        threads=int(cfg.get("threads", 1)),
        hash_mb=int(cfg.get("hash_mb", 16)),
        name=str(cfg.get("name", "stockfish")),
    )
    return StockfishOpponent(opp_cfg)


def _build_curriculum(cfg: dict[str, Any]):  # type: ignore[no-untyped-def]
    from alpha_zero.orchestrator.curriculum import (
        StockfishCurriculum,
        StockfishCurriculumConfig,
    )

    curr_cfg = StockfishCurriculumConfig(
        target=cfg.get("target", "skill"),
        mode=cfg.get("mode", "fixed"),
        initial_value=int(cfg.get("initial_value", 5)),
        step_size=int(cfg.get("step_size", 1)),
        step_every=int(cfg.get("step_every", 1)),
        window_size=int(cfg.get("window_size", 5)),
        promote_threshold=float(cfg.get("promote_threshold", 0.6)),
        demote_threshold=float(cfg.get("demote_threshold", 0.3)),
        adaptive_step=int(cfg.get("adaptive_step", 1)),
    )
    return StockfishCurriculum(curr_cfg)


def _build_advisor(advisor_cfg: dict[str, Any], llm_cfg: dict[str, Any]):  # type: ignore[no-untyped-def]
    from alpha_zero.orchestrator.chess_advisor import (
        ChessAdvisor,
        ChessAdvisorConfig,
    )
    from alpha_zero.utils.llm_client import LLMClient, LLMConfig, LLMProvider

    if not advisor_cfg.get("enabled"):
        return None

    base_url = (
        os.getenv("LM_STUDIO_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or llm_cfg.get("openai_base_url")
    )
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or llm_cfg.get("openai_api_key")
        or ("lm-studio" if base_url else None)
    )
    llm_config = LLMConfig(
        provider=LLMProvider(llm_cfg.get("provider", "openai")),
        openai_api_key=api_key,
        openai_base_url=base_url,
        openai_model=llm_cfg.get("openai_model", "qwen2.5-14b-instruct"),
        openai_max_tokens=int(llm_cfg.get("openai_max_tokens", 512)),
        openai_temperature=float(llm_cfg.get("openai_temperature", 0.2)),
    )
    advisor_config = ChessAdvisorConfig(
        enabled=True,
        prior_weight=float(advisor_cfg.get("prior_weight", 0.3)),
        max_ply_for_prior=int(advisor_cfg.get("max_ply_for_prior", 24)),
        cache_size=int(advisor_cfg.get("cache_size", 4096)),
        max_tokens=int(advisor_cfg.get("max_tokens", 512)),
        temperature=float(advisor_cfg.get("temperature", 0.2)),
        log_commentary=bool(advisor_cfg.get("log_commentary", False)),
    )
    return ChessAdvisor(advisor_config, llm_client=LLMClient(llm_config))


def _play_one_game(env, opponent, max_plies: int) -> float:  # type: ignore[no-untyped-def]
    """Play a game with the agent picking random legal moves (placeholder).

    Returns the reward from the *agent*'s perspective: +1 win, -1 loss, 0 draw.
    The agent always plays White in this thin entrypoint; production training
    alternates colours.
    """
    obs, _ = env.reset()
    agent_player = env.current_player
    reward = 0.0
    rng = np.random.default_rng()
    while True:
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask > 0)
        if legal.size == 0:
            break
        if env.current_player == agent_player:
            action = int(rng.choice(legal))
        else:
            action = opponent.get_action(env, mask)
        obs, _step_reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            winner = info.get("winner")
            if winner is None:
                reward = 0.0
            elif winner == agent_player:
                reward = 1.0
            else:
                reward = -1.0
            break
    return reward


def _run(config: dict[str, Any], iterations: int, games_per_iter: int, dry_run: bool) -> int:
    from alpha_zero.envs.env_factory import create_game_env

    env_cfg = config.get("environment", {})
    chess_cfg = env_cfg.get("chess", {})
    env = create_game_env(
        gameplay_preset="chess",
        chess_history_depth=int(chess_cfg.get("history_depth", 1)),
        chess_max_plies=int(chess_cfg.get("max_plies", 512)),
        chess_starting_fen=chess_cfg.get("starting_fen"),
    )

    opponent_cfg = config.get("opponent", {})
    curriculum_cfg = config.get("curriculum", {})
    advisor_cfg = config.get("advisor", {})
    llm_cfg = config.get("llm", {})

    curriculum = _build_curriculum(curriculum_cfg)
    # Advisor optional; constructed even in dry-run for validation.
    advisor = _build_advisor(advisor_cfg, llm_cfg)
    logger.info("Curriculum: %s mode, initial=%d", curriculum.config.mode, curriculum.current_value)
    if advisor is not None:
        logger.info("ChessAdvisor: enabled, weight=%s", advisor.config.prior_weight)

    if dry_run:
        logger.info(
            "Dry run OK: env=%s, iterations=%d, games_per_iter=%d",
            type(env).__name__,
            iterations,
            games_per_iter,
        )
        return 0

    # Build opponent only when actually running games (needs real stockfish binary).
    opponent = _build_opponent(opponent_cfg)
    try:
        for it in range(iterations):
            wins = 0
            for _ in range(games_per_iter):
                if curriculum.target == "skill":
                    opponent.set_skill(curriculum.current_value)
                else:
                    opponent.set_elo(curriculum.current_value)
                reward = _play_one_game(env, opponent, max_plies=env.max_plies)
                wins += 1 if reward > 0 else 0
            win_rate = wins / max(games_per_iter, 1)
            new_strength = curriculum.update(win_rate=win_rate)
            logger.info("Iteration %d: win_rate=%.3f, curriculum=%d", it, win_rate, new_strength)
    finally:
        opponent.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "chess_config.yaml",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--games-per-iter", type=int, default=None)
    parser.add_argument("--skill", type=int, default=None, help="Override curriculum.initial_value")
    parser.add_argument("--no-advisor", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = _load_config(args.config)
    training_cfg = config.setdefault("training", {})
    iterations = (
        args.iterations if args.iterations is not None else int(training_cfg.get("iterations", 1))
    )
    games_per_iter = (
        args.games_per_iter
        if args.games_per_iter is not None
        else int(training_cfg.get("games_per_iteration", 1))
    )
    if args.skill is not None:
        config.setdefault("curriculum", {})["initial_value"] = args.skill
    if args.no_advisor:
        config.setdefault("advisor", {})["enabled"] = False
    random.seed(0)
    np.random.seed(0)
    return _run(config, iterations, games_per_iter, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
