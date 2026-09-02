"""Random-play draw-rate sweep over board_size x max_turns.

Tests the hypothesis that MicroCiv draws at 8x8/30 are a 1-1 city tie
produced by geometry (MIN_CITY_DISTANCE) and economics (settler cost), not
by game length. If true, draw rate should NOT fall as max_turns rises at
board_size=8, and the terminal city-count mass should sit on (1, 1).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter

import numpy as np

from alpha_zero.envs.microciv import MicroCivEnv

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
BOARDS = [8, 10, 12, 16]
TURNS = [30, 60, 100]


def play(board: int, turns: int, seed: int) -> tuple[int | None, tuple[int, ...], int]:
    env = MicroCivEnv(board_size=board, max_turns=turns, seed=seed)
    env.reset()
    rng = np.random.default_rng(seed + 7919)
    steps = 0
    done = False
    while not done:
        mask = env.get_valid_actions()
        legal = np.flatnonzero(mask)
        action = int(rng.choice(legal))
        _, _, term, trunc, _ = env.step(action)
        done = term or trunc
        steps += 1
        if steps > turns * 64 * 2:  # runaway guard
            break
    cities = tuple(sum(1 for c in env.cities if c.player == p) for p in range(env.num_players))
    return env.winner, cities, steps


def unordered(cities: tuple[int, ...]) -> tuple[int, ...]:
    """Drop player identity: (2, 1) and (1, 2) are the same outcome for a draw analysis."""
    return tuple(sorted(cities))


results = []
t0 = time.time()
for board in BOARDS:
    for turns in TURNS:
        outcomes = Counter()
        city_pairs = Counter()
        steps_total = 0
        for g in range(GAMES):
            winner, cities, steps = play(board, turns, seed=10_000 * board + 100 * turns + g)
            outcomes["draw" if winner is None else f"p{winner}"] += 1
            city_pairs[unordered(cities)] += 1  # symmetric multiset, not player-indexed
            steps_total += steps
        draw_rate = outcomes["draw"] / GAMES
        row = {
            "board_size": board,
            "max_turns": turns,
            "games": GAMES,
            "draw_rate": round(draw_rate, 3),
            "outcomes": dict(outcomes),
            "city_pairs": {str(k): v for k, v in city_pairs.most_common(4)},
            "mean_steps": round(steps_total / GAMES, 1),
        }
        results.append(row)
        print(
            f"board={board:2d} turns={turns:3d} draw={draw_rate:.2f} "
            f"cities={dict(city_pairs.most_common(3))} steps={row['mean_steps']}",
            flush=True,
        )

out = {
    "method": "uniform random legal actions, seeded MicroCivEnv, winner from env.winner",
    "elapsed_s": round(time.time() - t0, 1),
    "rows": results,
}
out_path = sys.argv[2] if len(sys.argv) > 2 else "draw_sweep.json"
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)
