"""Probe a trained checkpoint against a random opponent, counting draws.

Committed alongside its results so the numbers in
``final_checkpoint_vs_random.json`` can be reproduced. Note that they cannot be
reproduced *exactly*: ``MicroCivEnv.reset()`` is unseeded here, which is why the
artifact records five samples rather than one.

Usage: ``CKPT=path/to/latest.pt python final_checkpoint_vs_random.py``
"""

import json
import os

import numpy as np
import torch

from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.training_civ6 import TrainingConfig, _mcts_network
from alpha_zero.utils.device import inference_thread_limit

ckpt = os.environ["CKPT"]
cfg = TrainingConfig(board_size=8, max_turns=20, num_simulations=12, device="cpu")
env = MicroCivEnv(board_size=8, max_turns=20, num_players=2)
obs, _ = env.reset()
net = MicroCivNetwork(
    board_size=8, action_space_size=env.action_space.n, input_channels=obs["spatial"].shape[0]
)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)
net.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
net.eval()
m = Civ6MCTS(_mcts_network(net, cfg), MCTSConfig(num_simulations=6), device="cpu")
rng = np.random.default_rng(0)
res = {"win": 0, "draw": 0, "loss": 0}
N = 20
with inference_thread_limit():
    for g in range(N):
        env.reset()
        me = g % 2
        while True:
            if env.current_player == me:
                p, _ = m.search(env, add_noise=False)
                a = m.select_action(p, temperature=0.0)
            else:
                mask = env.get_valid_actions()
                v = np.where(mask > 0)[0]
                a = rng.choice(v) if len(v) else 0
            _, _, t, tr, info = env.step(a)
            if t or tr:
                w = info.get("winner")
                res["win" if w == me else ("draw" if w is None else "loss")] += 1
                break
print(
    json.dumps(
        {
            **res,
            "games": N,
            "legacy_win_rate": res["win"] / N,
            "score_rate": (res["win"] + 0.5 * res["draw"]) / N,
            "draw_rate": res["draw"] / N,
        },
        indent=2,
    )
)
