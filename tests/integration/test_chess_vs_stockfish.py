"""Integration smoke test: agent plays a short game against Stockfish.

Skipped unless ``STOCKFISH_PATH`` is set and points at a working binary.
This test is intentionally short — its job is to verify wiring, not measure
strength.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("chess", reason="requires the 'chess' extra")

from alpha_zero.envs.chess_env import ChessEnv
from alpha_zero.opponents.stockfish_opponent import (
    StockfishOpponent,
    StockfishOpponentConfig,
)


@pytest.mark.integration
def test_short_game_vs_stockfish_completes() -> None:
    path = os.getenv("STOCKFISH_PATH")
    if not path or not os.path.exists(path):
        pytest.skip("STOCKFISH_PATH not set or binary missing")

    env = ChessEnv(history_depth=1, max_plies=20)
    obs, _ = env.reset()
    agent_player = env.current_player
    cfg = StockfishOpponentConfig(path=path, skill=0, depth_limit=1)
    rng = np.random.default_rng(0)
    with StockfishOpponent(cfg) as opponent:
        for _ in range(40):
            mask = obs["action_mask"]
            legal = np.flatnonzero(mask > 0)
            if legal.size == 0:
                break
            if env.current_player == agent_player:
                action = int(rng.choice(legal))
            else:
                action = opponent.get_action(env, mask)
            obs, _, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

    final_info = env.get_info()
    assert final_info["terminated"] or final_info["truncated"]
    if final_info["winner"] is not None:
        assert final_info["winner"] in (0, 1)
