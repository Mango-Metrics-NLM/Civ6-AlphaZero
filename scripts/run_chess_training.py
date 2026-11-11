#!/usr/bin/env python3
"""Thin chess-vs-Stockfish entry point.

Loads ``configs/chess_config.yaml`` (or a user-supplied path), wires the
``ChessEnv`` + ``StockfishOpponent`` + ``StockfishCurriculum`` + optional
``ChessAdvisor`` + ``Civ6MCTS`` together, and runs N iterations.

This is intentionally minimal — it exists to make the end-to-end demo
runnable with a single command. It does *not* implement the full
self-play/learner loop (no replay buffer, no gradient updates, no trained
network); that lives in the orchestrator. The MCTS here uses a uniform
stub network so ``Civ6MCTS.search()`` runs and exercises the LLM-prior
callback — sufficient to prove Qwen actually shapes play but not a strong
chess agent. ``--dry-run`` validates the configuration and exits without
playing, suitable for CI smoke tests.

Each finished game is written as a PGN file under ``--pgn-dir``
(default ``replays/chess/``) so you can paste it into Lichess Analysis
or open it in any chess GUI to watch the moves. Use ``--no-pgn`` to
disable.

Usage::

    python scripts/run_chess_training.py --config configs/chess_config.yaml
    python scripts/run_chess_training.py --dry-run
    python scripts/run_chess_training.py --iterations 1 --games-per-iter 2 \\
        --skill 0 --no-advisor
"""

from __future__ import annotations

import argparse
import datetime as _dt
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


class _UniformChessNetwork:
    """Minimal MCTSNetwork stub for the wiring smoke test."""

    def __init__(self, action_space_size: int) -> None:
        self._n = int(action_space_size)

    def predict(self, spatial, vector, action_mask):  # type: ignore[no-untyped-def]
        import torch

        batch = spatial.shape[0]
        mask = action_mask.float()
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        policy = mask / denom
        value = torch.zeros((batch, 1), dtype=torch.float32, device=spatial.device)
        return policy, value

    def __call__(self, spatial, vector, action_mask=None):  # type: ignore[no-untyped-def]
        if action_mask is None:
            import torch

            action_mask = torch.ones((spatial.shape[0], self._n), dtype=torch.float32)
        return self.predict(spatial, vector, action_mask)

    def eval(self):  # type: ignore[no-untyped-def]
        return self


def _resolve_device(requested: str) -> str:
    """Resolve the requested torch device, falling back to cpu when needed.

    ``requested`` may be ``"auto"`` (pick cuda when available, else cpu),
    ``"cuda"`` / ``"cuda:N"`` (use that GPU; fall back to cpu with a warning
    if CUDA isn't available), or ``"cpu"``. We keep this dependency on torch
    inside the function so importing this module doesn't pull torch in just
    to print --help.
    """
    try:
        import torch
    except Exception:
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to cpu")
        return "cpu"
    return requested


def _build_mcts(  # type: ignore[no-untyped-def]
    advisor,
    training_cfg: dict[str, Any],
    action_space_size: int,
    device: str = "auto",
):
    """Construct a Civ6MCTS instance wired to the LLM advisor (if any).

    ``device`` controls only the *stub network* device. The Qwen LLM runs in
    LM Studio (a separate process) and its GPU usage is configured there,
    not here. Keeping this knob is still worth it for when a real chess
    network replaces the stub.
    """
    try:
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig
    except Exception as exc:
        logger.warning("Civ6MCTS unavailable (%s); falling back to random play", exc)
        return None

    resolved_device = _resolve_device(device)
    mcts_cfg = MCTSConfig(
        num_simulations=int(training_cfg.get("num_simulations", 64)),
        c_puct=float(training_cfg.get("c_puct", 1.5)),
    )
    network = _UniformChessNetwork(action_space_size=action_space_size)
    llm_prior_fn = advisor.mcts_callback() if advisor is not None else None
    llm_prior_weight = float(advisor.config.prior_weight) if advisor is not None else 0.5
    mcts = Civ6MCTS(
        network=network,
        config=mcts_cfg,
        device=resolved_device,
        llm_prior_fn=llm_prior_fn,
        llm_prior_weight=llm_prior_weight,
    )
    logger.info(
        "Civ6MCTS constructed: simulations=%d, device=%s, llm_prior=%s, weight=%s",
        mcts_cfg.num_simulations,
        resolved_device,
        llm_prior_fn is not None,
        llm_prior_weight,
    )
    return mcts


def _dump_pgn(
    env,  # type: ignore[no-untyped-def]
    pgn_dir: Path,
    game_idx: int,
    iteration: int,
    agent_player: int,
    reward: float,
    advisor_enabled: bool,
    opponent_label: str,
) -> Path | None:
    """Serialize the just-finished game's board history to a PGN file.

    Returns the written path, or ``None`` if PGN dumping is unavailable
    (e.g. python-chess.pgn not importable, or env doesn't expose ``_board``).
    """
    try:
        import chess
        import chess.pgn
    except Exception as exc:
        logger.debug("PGN dump skipped (chess.pgn unavailable: %s)", exc)
        return None
    board = getattr(env, "_board", None)
    if not isinstance(board, chess.Board):
        logger.debug("PGN dump skipped: env has no chess.Board attribute")
        return None
    try:
        pgn_dir.mkdir(parents=True, exist_ok=True)
        game = chess.pgn.Game.from_board(board)
        agent_label = "MCTS+LLM" if advisor_enabled else "MCTS"
        if agent_player == 0:  # agent plays White
            game.headers["White"] = agent_label
            game.headers["Black"] = opponent_label
        else:
            game.headers["White"] = opponent_label
            game.headers["Black"] = agent_label
        game.headers["Event"] = "Civ6-AlphaZero chess vs Stockfish"
        game.headers["Site"] = "local"
        game.headers["Date"] = _dt.datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = f"{iteration}.{game_idx}"
        # Use the board's actual outcome string when present; fall back to
        # synthesising one from the agent reward so a truncated game still
        # gets a well-formed PGN result tag.
        outcome = board.outcome(claim_draw=True)
        if outcome is not None and outcome.result() in {"1-0", "0-1", "1/2-1/2"}:
            result = outcome.result()
        else:
            if reward > 0:
                result = "1-0" if agent_player == 0 else "0-1"
            elif reward < 0:
                result = "0-1" if agent_player == 0 else "1-0"
            else:
                result = "1/2-1/2"
        game.headers["Result"] = result
        path = pgn_dir / f"iter{iteration:03d}_game{game_idx:03d}.pgn"
        with path.open("w", encoding="utf-8") as fh:
            print(game, file=fh)
        logger.info("PGN written: %s", path)
        return path
    except Exception as exc:
        logger.warning("PGN dump failed: %s", exc)
        return None


def _play_one_game(
    env,  # type: ignore[no-untyped-def]
    opponent,  # type: ignore[no-untyped-def]
    max_plies: int,
    mcts=None,  # type: ignore[no-untyped-def]
    rng: np.random.Generator | None = None,
) -> tuple[float, int]:
    """Play one chess game; return (agent reward, agent player index).

    The returned ``agent_player`` is the colour the agent played
    (``0`` for White, ``1`` for Black) — needed by the PGN writer to
    populate the White/Black tags correctly.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs, _ = env.reset()
    agent_player = env.current_player
    reward = 0.0
    for _ply in range(max_plies):
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask > 0)
        if legal.size == 0:
            break
        if env.current_player == agent_player:
            if mcts is not None:
                action_probs, _root_value = mcts.search(env, add_noise=False)
                masked_probs = action_probs * mask
                if masked_probs.sum() <= 0:
                    action = int(rng.choice(legal))
                else:
                    action = int(np.argmax(masked_probs))
            else:
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
    return reward, agent_player


def _run(
    config: dict[str, Any],
    iterations: int,
    games_per_iter: int,
    dry_run: bool,
    pgn_dir: Path | None,
    device: str = "auto",
) -> int:
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
    training_cfg = config.get("training", {})

    curriculum = _build_curriculum(curriculum_cfg)
    advisor = _build_advisor(advisor_cfg, llm_cfg)
    logger.info("Curriculum: %s mode, initial=%d", curriculum.config.mode, curriculum.current_value)
    if advisor is not None:
        logger.info("ChessAdvisor: enabled, weight=%s", advisor.config.prior_weight)

    mcts = _build_mcts(
        advisor,
        training_cfg,
        action_space_size=env.action_space.n,
        device=device,
    )

    if dry_run:
        logger.info(
            "Dry run OK: env=%s, iterations=%d, games_per_iter=%d, mcts=%s, pgn_dir=%s",
            type(env).__name__,
            iterations,
            games_per_iter,
            mcts is not None,
            pgn_dir,
        )
        return 0

    opponent = _build_opponent(opponent_cfg)
    advisor_enabled = bool(advisor_cfg.get("enabled"))
    try:
        for it in range(iterations):
            wins = 0
            for game_idx in range(games_per_iter):
                if curriculum.target == "skill":
                    opponent.set_skill(curriculum.current_value)
                else:
                    opponent.set_elo(curriculum.current_value)
                reward, agent_player = _play_one_game(
                    env, opponent, max_plies=env.max_plies, mcts=mcts
                )
                wins += 1 if reward > 0 else 0
                if pgn_dir is not None:
                    label = f"Stockfish-{curriculum.target}-{curriculum.current_value}"
                    _dump_pgn(
                        env,
                        pgn_dir=pgn_dir,
                        game_idx=game_idx,
                        iteration=it,
                        agent_player=agent_player,
                        reward=reward,
                        advisor_enabled=advisor_enabled,
                        opponent_label=label,
                    )
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
    parser.add_argument(
        "--pgn-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "replays" / "chess",
        help="Directory to write per-game PGN files (one per finished game).",
    )
    parser.add_argument("--no-pgn", action="store_true", help="Disable PGN writing entirely.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the MCTS stub network: 'auto' (default), 'cpu', "
        "'cuda', or 'cuda:N'. Note: this does NOT affect LM Studio/Qwen, "
        "which is configured inside LM Studio itself.",
    )
    from alpha_zero.utils.game_budget import add_budget_arguments

    add_budget_arguments(parser)
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
    from alpha_zero.utils.game_budget import enforce_cli

    enforce_cli(
        "run_chess_training",
        {"iterations": iterations, "games_per_iter": games_per_iter, "dry_run": args.dry_run},
        args,
    )
    if args.skill is not None:
        config.setdefault("curriculum", {})["initial_value"] = args.skill
    if args.no_advisor:
        config.setdefault("advisor", {})["enabled"] = False
    pgn_dir = None if args.no_pgn else args.pgn_dir
    random.seed(0)
    np.random.seed(0)
    return _run(
        config,
        iterations,
        games_per_iter,
        dry_run=args.dry_run,
        pgn_dir=pgn_dir,
        device=args.device,
    )


if __name__ == "__main__":
    sys.exit(main())
