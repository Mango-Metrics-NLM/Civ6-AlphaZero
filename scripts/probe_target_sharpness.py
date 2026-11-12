#!/usr/bin/env python3
"""Probe how sharp the Gumbel policy target is, at zero games of training (LF-A5/A12).

What it measures
----------------
A randomly initialised, seeded ``MicroCivNetwork`` searches ``--positions``
positions drawn from seeded *random* play. For each position it runs several
variants of the Gumbel search and records, for each variant's policy target:
the entropy (nats), the largest probability (max-p), and whether max-p exceeds
``greedy_threshold`` -- the point above which ``GumbelMCTS.select_action``
ignores temperature, so the self-play temperature schedule is inert on that
move.

It plays **no games**: positions come from random play, and the search only
steps *clones* of them. The ``game_budget`` model for this runner is therefore
0 games. Nothing here trains a network or changes production code.

Variants (one search per row, then a target computed from that search's root)
-----------------------------------------------------------------------------
``current``
    ``GumbelMCTS`` exactly as A1 runs it (``gumbel_search_config``: k, n,
    c_visit 50, c_scale 1.0, ``completed_q`` target, ``greedy_threshold``).
    Halving ranks survivors by ``g + log pi + completedQ`` with raw Q and no
    sigma; the target is ``build_completed_q_policy``.
``current_search_paper_target``
    The *same* tree as ``current``, with the paper's target formula -- isolates
    the target formula from the search path.
``sigma_halving``
    The A12 change alone: halving ranks by ``g + log pi + sigma(completedQ)``,
    ``sigma = (c_visit + max_b N(b)) * c_scale``, on this repository's own
    completion (raw Q, no normalisation). Target unchanged (repository formula).
``paper``
    Danihelka et al. (2022), Go/chess settings: completion of unvisited actions
    by ``v_mix``, min-max normalisation to [0, 1], ``sigma`` with c_visit 50 and
    c_scale 1.0 in the halving score **and** in the target, target a softmax
    over every legal action. As implemented by ``mctx``'s
    ``qtransform_completed_by_mix_value`` and ``gumbel_muzero_policy``.
``paper_mctx_scale``
    ``paper`` with c_scale 0.1 (``mctx``'s default ``value_scale``; the paper's
    Atari setting).

Deliberately *not* varied: interior (non-root) selection stays this
repository's PUCT (``c_puct = c_visit / 50``), where ``mctx`` uses the
deterministic ``argmax(pi' - N / (1 + sum N))``; and the halving schedule, which
already matches ``mctx`` (``gumbel_ops.halving_schedule``).

Usage::

    python scripts/probe_target_sharpness.py --seed 42 \\
        --out_dir docs/evidence/target_sharpness_probe
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import platform
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Run as a script, sys.path[0] is scripts/, so make this checkout win over any
# separately installed copy of the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_zero.core.mcts.backup import perspective_sign
from alpha_zero.core.mcts.gumbel import GumbelMCTS, GumbelMCTSConfig, GumbelMCTSNode
from alpha_zero.core.mcts.gumbel_ops import build_completed_q_policy, compute_completed_q
from alpha_zero.core.mcts_constants import EPSILON
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.training_civ6 import TrainingConfig, gumbel_search_config
from alpha_zero.utils.device import inference_thread_limit
from alpha_zero.utils.game_budget import add_budget_arguments, enforce_cli
from alpha_zero.utils.seeding import derive_seed

logger = logging.getLogger("probe_target_sharpness")

RUNNER = "probe_target_sharpness"
SCHEMA_VERSION = 1
SEARCH_EPOCH = "post-2026-09-04"
JSON_NAME = "probe_target_sharpness.json"
REPORT_NAME = "report.md"

#: Seed streams of this probe. Named, so they cannot collide with any training stream.
STREAM_ENV = "probe.sharpness.env"
STREAM_PLAY = "probe.sharpness.play"
STREAM_SEARCH = "probe.sharpness.search"

#: ``mctx``'s ``qtransform_completed_by_mix_value`` epsilon for the min-max rescale.
MCTX_RESCALE_EPSILON = 1e-8
#: ``mctx``'s default ``value_scale`` (the paper's Atari ``c_scale``).
MCTX_DEFAULT_C_SCALE = 0.1
#: The paper's Go/chess ``c_visit`` and ``c_scale`` (Danihelka et al. 2022, Appendix F).
PAPER_C_VISIT = 50.0
PAPER_C_SCALE = 1.0

HALVING_CURRENT = "current"
HALVING_SIGMA_RAW = "sigma_raw"
HALVING_PAPER = "paper"

#: (cell name, search key, target key). Search keys index ``_SEARCHES``.
CELLS: tuple[tuple[str, str, str], ...] = (
    ("current", "current", "repo"),
    ("current_search_paper_target", "current", "paper"),
    ("sigma_halving", "sigma_halving", "repo"),
    ("paper", "paper", "paper"),
    ("paper_mctx_scale", "paper_mctx_scale", "paper"),
)

#: search key -> (halving mode, c_scale override or None for the run's own).
_SEARCHES: dict[str, tuple[str, float | None]] = {
    "current": (HALVING_CURRENT, None),
    "sigma_halving": (HALVING_SIGMA_RAW, None),
    "paper": (HALVING_PAPER, PAPER_C_SCALE),
    "paper_mctx_scale": (HALVING_PAPER, MCTX_DEFAULT_C_SCALE),
}


# ---------------------------------------------------------------------------
# Paper / mctx completed-Q transform
# ---------------------------------------------------------------------------


def _root_child_q(root: GumbelMCTSNode, child: GumbelMCTSNode) -> float:
    """A root child's Q from the root mover's perspective (0.0 when unvisited)."""
    if child.visit_count == 0:
        return 0.0
    return perspective_sign(root.to_play, child.to_play) * child.value


def paper_sigma_completed(
    *,
    priors: np.ndarray,
    visits: np.ndarray,
    q_values: np.ndarray,
    raw_value: float,
    c_visit: float,
    c_scale: float,
) -> np.ndarray:
    """``sigma(completedQ)`` as ``mctx``'s ``qtransform_completed_by_mix_value``.

    Args are aligned arrays over the *legal* actions only. Visited actions keep
    their own Q; unvisited ones take ``v_mix``; the vector is min-max rescaled to
    [0, 1] and multiplied by ``(c_visit + max N) * c_scale``.
    """
    visited = visits > 0
    total = float(visits.sum())
    if visited.any():
        probs = np.maximum(priors, np.finfo(np.float64).tiny)
        weights = np.where(visited, probs, 0.0)
        weighted_q = float((weights * q_values).sum() / weights.sum())
        v_mix = (raw_value + total * weighted_q) / (total + 1.0)
    else:
        v_mix = raw_value
    completed = np.where(visited, q_values, v_mix)
    low, high = float(completed.min()), float(completed.max())
    normalised = (completed - low) / max(high - low, MCTX_RESCALE_EPSILON)
    max_visits = float(visits.max()) if visits.size else 0.0
    return (c_visit + max_visits) * c_scale * normalised


def paper_target(
    root: GumbelMCTSNode, raw_value: float, action_space: int, *, c_visit: float, c_scale: float
) -> np.ndarray:
    """``softmax(logits + sigma(completedQ))`` over every legal root action."""
    actions = np.array(sorted(root.children), dtype=np.int64)
    out = np.zeros(action_space, dtype=np.float64)
    if actions.size == 0:
        return out
    children = [root.children[int(a)] for a in actions]
    priors = np.array([c.prior for c in children], dtype=np.float64)
    visits = np.array([c.visit_count for c in children], dtype=np.float64)
    qs = np.array([_root_child_q(root, c) for c in children], dtype=np.float64)
    sigma = paper_sigma_completed(
        priors=priors,
        visits=visits,
        q_values=qs,
        raw_value=raw_value,
        c_visit=c_visit,
        c_scale=c_scale,
    )
    scores = np.log(np.maximum(priors, EPSILON)) + sigma
    scores -= scores.max()
    exp = np.exp(scores)
    out[actions] = exp / exp.sum()
    return out


def repo_target(root: GumbelMCTSNode, action_space: int, config: GumbelMCTSConfig) -> np.ndarray:
    """This repository's target: ``build_completed_q_policy`` on the root."""
    priors = np.zeros(action_space, dtype=np.float32)
    visits = np.zeros(action_space, dtype=np.float32)
    qs = np.zeros(action_space, dtype=np.float32)
    for action, child in root.children.items():
        priors[action] = child.prior
        visits[action] = child.visit_count
        qs[action] = _root_child_q(root, child)
    target = build_completed_q_policy(
        priors=priors,
        visit_counts=visits,
        q_values=qs,
        root_value=root.value,
        c_visit=config.c_visit,
        c_scale=config.c_scale,
    ).astype(np.float64)
    return target


# ---------------------------------------------------------------------------
# The search under probe
# ---------------------------------------------------------------------------


class ProbeGumbelMCTS(GumbelMCTS):
    """``GumbelMCTS`` with a selectable halving score, exposing its root.

    ``current`` leaves ranking untouched. The other modes override only
    ``_rank_by_completed_q`` -- the schedule, the dealing of visits and the
    final ``choose_among_most_visited`` are the production ones.
    """

    def __init__(
        self, network: Any, config: GumbelMCTSConfig, *, halving: str, device: str = "cpu"
    ) -> None:
        super().__init__(network, config, device=device)
        if halving not in (HALVING_CURRENT, HALVING_SIGMA_RAW, HALVING_PAPER):
            raise ValueError(f"unknown halving mode {halving!r}")
        self.halving = halving
        self.last_root: GumbelMCTSNode | None = None
        self.last_raw_value = 0.0
        self.last_choice = -1

    def _sequential_halving(
        self,
        env: Any,
        root: GumbelMCTSNode,
        candidate_actions: np.ndarray,
        gumbel_scores: np.ndarray,
        root_value: float,
    ) -> int:
        self.last_root = root
        self.last_raw_value = float(root_value)
        choice = super()._sequential_halving(
            env, root, candidate_actions, gumbel_scores, root_value
        )
        self.last_choice = int(choice)
        return choice

    def _rank_by_completed_q(
        self,
        root: GumbelMCTSNode,
        actions: list[int],
        scores: dict[int, float],
        root_value: float,
    ) -> list[int]:
        if self.halving == HALVING_CURRENT:
            return super()._rank_by_completed_q(root, actions, scores, root_value)
        max_all = max((c.visit_count for c in root.children.values()), default=0)
        if self.halving == HALVING_SIGMA_RAW:
            max_visits = max(
                (root.children[a].visit_count for a in actions if a in root.children), default=0
            )
            coef = (self.config.c_visit + max_all) * self.config.c_scale
            combined: dict[int, float] = {}
            for action in actions:
                completed = 0.0
                child = root.children.get(action)
                if child is not None:
                    q = _root_child_q(root, child)
                    completed = compute_completed_q(q, child.visit_count, max_visits, root_value)
                combined[action] = coef * completed + scores.get(action, 0.0)
            return sorted(actions, key=lambda a: -combined[a])
        # Paper: sigma over the mctx completion of *every* legal action.
        legal = sorted(root.children)
        children = [root.children[a] for a in legal]
        sigma = paper_sigma_completed(
            priors=np.array([c.prior for c in children], dtype=np.float64),
            visits=np.array([c.visit_count for c in children], dtype=np.float64),
            q_values=np.array([_root_child_q(root, c) for c in children], dtype=np.float64),
            raw_value=root_value,
            c_visit=self.config.c_visit,
            c_scale=self.config.c_scale,
        )
        by_action = dict(zip(legal, sigma.tolist(), strict=True))
        return sorted(actions, key=lambda a: -(by_action.get(a, 0.0) + scores.get(a, 0.0)))


# ---------------------------------------------------------------------------
# Positions, network, statistics
# ---------------------------------------------------------------------------


def _new_env(board_size: int, max_turns: int, env_seed: int | None) -> MicroCivEnv:
    env = MicroCivEnv(board_size=board_size, max_turns=max_turns, seed=env_seed)
    env.reset(seed=env_seed)
    return env


def sample_position(
    board_size: int, max_turns: int, seed: int, index: int
) -> tuple[MicroCivEnv, int]:
    """One non-terminal position from seeded uniform-random play.

    A target turn is drawn uniformly from ``[0, max_turns)``; random legal
    actions are played until the game reaches it. If the game ends first, the
    position just before the terminal action is used (rebuilt by replay, which
    is exact because the environment is seeded).

    Returns:
        The environment at the position, and the number of actions played.
    """
    env_seed = derive_seed(seed, STREAM_ENV, index)
    rng = np.random.default_rng(derive_seed(seed, STREAM_PLAY, index))
    target_turn = int(rng.integers(0, max_turns))
    env = _new_env(board_size, max_turns, env_seed)
    played: list[int] = []
    while env.turn < target_turn:
        legal = np.flatnonzero(env.get_valid_actions() > 0)
        action = int(rng.choice(legal))
        _obs, _r, terminated, truncated, _info = env.step(action)
        played.append(action)
        if terminated or truncated:
            played.pop()
            env = _new_env(board_size, max_turns, env_seed)
            for a in played:
                env.step(a)
            break
    return env, len(played)


def build_network(board_size: int, max_turns: int, seed: int, device: str) -> torch.nn.Module:
    """A randomly initialised network, built exactly as ``training_civ6.train`` builds it."""
    from alpha_zero.core.civ6_network import MicroCivNetwork

    env = MicroCivEnv(board_size=board_size, max_turns=max_turns, seed=seed)
    obs, _ = env.reset(seed=seed)
    defaults = TrainingConfig()
    torch.manual_seed(seed)
    network = MicroCivNetwork(
        board_size=board_size,
        action_space_size=int(env.action_space.n),
        num_res_blocks=defaults.num_res_blocks,
        num_filters=defaults.num_filters,
        spatial_channels=int(obs["spatial"].shape[0]),
        num_players=2,
    ).to(device)
    network.eval()
    return network


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats over the support of ``p``."""
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
    }


def search_config(
    board_size: int, max_turns: int, k: int, n: int, device: str, seed: int | None
) -> GumbelMCTSConfig:
    """The self-play ``GumbelMCTSConfig`` A1's command line resolves to."""
    config = TrainingConfig(
        board_size=board_size,
        max_turns=max_turns,
        use_gumbel_mcts=True,
        gumbel_num_simulations=n,
        gumbel_num_sampled_actions=k,
        num_simulations=n,
        device=device,
        seed=seed,
    )
    return gumbel_search_config(
        config,
        num_simulations=config.gumbel_num_simulations,
        seed=seed,
        temperature=config.temperature,
        temperature_threshold=config.temperature_threshold,
    )


def run_probe(
    *,
    positions: int,
    board_size: int,
    max_turns: int,
    k: int,
    n: int,
    seed: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run every variant on ``positions`` positions; return the deterministic result.

    The result holds no timestamps or timings, so two calls with the same
    arguments return equal dictionaries.
    """
    if positions < 1:
        raise ValueError("positions must be >= 1")
    base = search_config(board_size, max_turns, k, n, device, seed)
    network = build_network(board_size, max_turns, seed, device)

    per_cell: dict[str, dict[str, list[float]]] = {
        name: {"entropy": [], "max_p": [], "support": []} for name, _, _ in CELLS
    }
    prior_stats: dict[str, list[float]] = {"entropy": [], "max_p": []}
    agree: dict[str, int] = {key: 0 for key in _SEARCHES if key != "current"}
    rows: list[dict[str, Any]] = []
    forced = 0

    with inference_thread_limit():
        for index in range(positions):
            env, played = sample_position(board_size, max_turns, seed, index)
            legal = int((env.get_valid_actions() > 0).sum())
            forced += int(legal <= 1)
            search_seed = derive_seed(seed, STREAM_SEARCH, index)
            roots: dict[str, ProbeGumbelMCTS] = {}
            for key, (halving, c_scale) in _SEARCHES.items():
                config = dataclasses.replace(base, seed=search_seed)
                if c_scale is not None:
                    config = dataclasses.replace(config, c_scale=c_scale, c_visit=PAPER_C_VISIT)
                mcts = ProbeGumbelMCTS(network, config, halving=halving, device=device)
                mcts.search(env)
                roots[key] = mcts
            current = roots["current"]
            for key in agree:
                agree[key] += int(roots[key].last_choice == current.last_choice)

            row: dict[str, Any] = {"index": index, "turn": int(env.turn), "legal": legal}
            row["actions_played"] = played
            if current.last_root is not None:
                priors = np.array([c.prior for c in current.last_root.children.values()])
                prior_stats["entropy"].append(entropy(priors))
                prior_stats["max_p"].append(float(priors.max()))
            for name, search_key, target_key in CELLS:
                mcts = roots[search_key]
                root = mcts.last_root
                if root is None:  # no halving ran (no legal action): nothing to target
                    target = np.zeros(int(env.action_space.n))
                elif target_key == "repo":
                    target = repo_target(root, int(env.action_space.n), mcts.config)
                else:
                    target = paper_target(
                        root,
                        mcts.last_raw_value,
                        int(env.action_space.n),
                        c_visit=mcts.config.c_visit,
                        c_scale=mcts.config.c_scale,
                    )
                if target.sum() <= 0.0 and mcts.last_choice >= 0:
                    target[mcts.last_choice] = 1.0  # GumbelMCTS's own fallback
                h, mp = entropy(target), float(target.max())
                per_cell[name]["entropy"].append(h)
                per_cell[name]["max_p"].append(mp)
                per_cell[name]["support"].append(float((target > 0).sum()))
                row[name] = {"entropy": round(h, 6), "max_p": round(mp, 6)}
            rows.append(row)

    threshold = float(base.greedy_threshold)
    early = [r["turn"] < base.temperature_threshold for r in rows]
    cells: dict[str, Any] = {}
    for name, search_key, target_key in CELLS:
        stats = per_cell[name]
        above = [mp > threshold for mp in stats["max_p"]]
        early_above = [a for a, e in zip(above, early, strict=True) if e]
        cells[name] = {
            "search": search_key,
            "target": target_key,
            "entropy_nats": _summary(stats["entropy"]),
            "max_p": _summary(stats["max_p"]),
            "support": _summary(stats["support"]),
            "share_max_p_above_greedy_threshold": round(sum(above) / len(above), 6),
            "share_above_threshold_in_temperature_phase": (
                round(sum(early_above) / len(early_above), 6) if early_above else None
            ),
            "share_max_p_above_0_5": round(
                sum(mp > 0.5 for mp in stats["max_p"]) / len(stats["max_p"]), 6
            ),
        }

    return {
        "probe": RUNNER,
        "probe_schema_version": SCHEMA_VERSION,
        "settings": {
            "positions": positions,
            "board_size": board_size,
            "max_turns": max_turns,
            "gumbel_k": k,
            "gumbel_sims": n,
            "seed": seed,
            "device": device,
            "network": "MicroCivNetwork, random init (torch.manual_seed(seed)), 0 games trained",
            "position_source": "seeded uniform-random play to a uniform target turn",
            "current": {
                "c_visit": base.c_visit,
                "c_scale": base.c_scale,
                "policy_target": base.policy_target,
                "greedy_threshold": base.greedy_threshold,
                "temperature": base.temperature,
                "temperature_threshold": base.temperature_threshold,
                "interior_c_puct": base.c_visit / 50.0,
                "halving_score": "g + log pi + completedQ_raw (no sigma)",
            },
            "paper": {
                "c_visit": PAPER_C_VISIT,
                "c_scale": PAPER_C_SCALE,
                "mctx_default_c_scale": MCTX_DEFAULT_C_SCALE,
                "completion": "visited: own Q; unvisited: v_mix",
                "normalisation": "min-max over completed Q of all legal actions",
                "halving_score": "g + log pi + sigma(completedQ_normalised)",
                "target": "softmax(log pi + sigma) over all legal actions",
            },
        },
        "games_played": 0,
        "positions_forced": forced,
        "temperature_phase_positions": int(sum(early)),
        "prior": {
            "entropy_nats": _summary(prior_stats["entropy"]),
            "max_p": _summary(prior_stats["max_p"]),
        },
        "legal_actions": _summary([float(r["legal"]) for r in rows]),
        "chosen_move_agreement_with_current": {
            key: round(count / positions, 6) for key, count in agree.items()
        },
        "cells": cells,
        "positions": rows,
    }


# ---------------------------------------------------------------------------
# Evidence and report
# ---------------------------------------------------------------------------


def evidence_block(result: dict[str, Any], argv: list[str]) -> dict[str, Any]:
    """The ``docs/evidence/SCHEMA.md`` block for one probe result."""
    from alpha_zero.utils.checkpoint import read_git_sha

    settings = result["settings"]
    sha = read_git_sha()
    block: dict[str, Any] = {
        "schema_version": 1,
        "git_sha": sha,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "search_epoch": SEARCH_EPOCH,
        "seed": settings["seed"],
        "command": shlex.join(["python", "scripts/probe_target_sharpness.py", *argv]),
        "hardware": {
            "cpu_model": platform.processor() or None,
            "gpu_model": None,
            "logical_cores": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "games_total": 0,
        "claims": [
            f"sharpness (entropy, max-p, share above greedy_threshold) of the Gumbel policy "
            f"target at k={settings['gumbel_k']}, n={settings['gumbel_sims']} on "
            f"{settings['positions']} random-play positions, for an untrained network, "
            "under the current settings and the paper's",
        ],
        "non_claims": [
            "not evidence of playing strength or of learning",
            "untrained network only; a trained network's targets may be sharper or flatter",
            "positions come from random play, not from self-play",
            "interior (non-root) selection is this repository's PUCT in every variant",
        ],
    }
    if sha is None:
        block["git_sha_null_reason"] = "no CIV6AZ_GIT_SHA and no resolvable .git"
    return block


def render_report(result: dict[str, Any], evidence: dict[str, Any]) -> str:
    """A markdown report that a reader can follow without the JSON."""
    s = result["settings"]
    cur = s["current"]
    lines = [
        "# Gumbel policy-target sharpness probe (LF-A5 / LF-A12)",
        "",
        f"**Date:** {evidence['generated_utc'][:10]} · **Search epoch:** {SEARCH_EPOCH} · "
        f"**Games played:** 0 · **Raw data:** [`{JSON_NAME}`]({JSON_NAME})",
        "",
        f"{s['positions']} positions from seeded uniform-random play at board "
        f"{s['board_size']} / max_turns {s['max_turns']}; a randomly initialised "
        f"`MicroCivNetwork` (seed {s['seed']}, 0 games of training) searches each with "
        f"k={s['gumbel_k']}, n={s['gumbel_sims']}. `greedy_threshold` = "
        f"{cur['greedy_threshold']}: above it `select_action` ignores temperature.",
        "",
        "```",
        evidence["command"],
        "```",
        "",
        "| Variant | Search (halving score) | Target | Entropy mean / median (nats) "
        "| max-p mean / median | Share max-p > threshold | Same in temperature phase "
        "| Support (median) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, cell in result["cells"].items():
        tphase = cell["share_above_threshold_in_temperature_phase"]
        lines.append(
            f"| `{name}` | {cell['search']} | {cell['target']} | "
            f"{cell['entropy_nats']['mean']:.3f} / {cell['entropy_nats']['median']:.3f} | "
            f"{cell['max_p']['mean']:.3f} / {cell['max_p']['median']:.3f} | "
            f"{cell['share_max_p_above_greedy_threshold']:.3f} | "
            f"{'n/a' if tphase is None else f'{tphase:.3f}'} | "
            f"{cell['support']['median']:.0f} |"
        )
    prior = result["prior"]
    agree = result["chosen_move_agreement_with_current"]
    lines += [
        "",
        f"Network prior over legal actions (reference): entropy mean "
        f"{prior['entropy_nats']['mean']:.3f} nats, max-p mean {prior['max_p']['mean']:.3f}. "
        f"Legal actions per position: median {result['legal_actions']['median']:.0f}. "
        f"Forced positions: {result['positions_forced']}. Positions in the temperature phase "
        f"(turn < {cur['temperature_threshold']}): {result['temperature_phase_positions']}.",
        "",
        "Chosen move agrees with `current`: "
        + ", ".join(f"`{k}` {v:.3f}" for k, v in agree.items())
        + ".",
        "",
        "## Variants",
        "",
        f"- `current` — A1's settings: c_visit {cur['c_visit']}, c_scale {cur['c_scale']}, "
        f"target `{cur['policy_target']}`; halving ranks by {cur['halving_score']}; target "
        "restricted to visited actions; completion `q + ((maxN-N)/maxN)·v_root`, raw Q.",
        "- `current_search_paper_target` — the same tree, scored with the paper's target.",
        "- `sigma_halving` — LF-A12 alone: halving ranks by `g + log pi + (c_visit + max N)"
        "·c_scale·completedQ` on this repository's completion; repository target.",
        "- `paper` — Danihelka et al. (2022), Go/chess: c_visit 50, c_scale 1.0, `v_mix` "
        "completion, min-max normalised Q, sigma in the halving score and in a softmax target "
        "over all legal actions (as `mctx`'s `qtransform_completed_by_mix_value`).",
        f"- `paper_mctx_scale` — `paper` with c_scale {s['paper']['mctx_default_c_scale']} "
        "(`mctx`'s default; the paper's Atari setting).",
        "",
        "Not varied: interior selection (this repository's PUCT, c_puct "
        f"{cur['interior_c_puct']}) and the halving schedule (already `mctx`'s).",
        "",
        "## Non-claims",
        "",
        *[f"- {nc}" for nc in evidence["non_claims"]],
        "",
        "Interpretation and the decision this feeds: "
        "`kb/specs/20261111-completed-q-alignment-memo.md`.",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--positions", type=int, default=200)
    p.add_argument("--board_size", type=int, default=8)
    p.add_argument("--max_turns", type=int, default=60)
    p.add_argument("--k", type=int, default=8, help="Gumbel sampled actions (A1: 8)")
    p.add_argument("--n", type=int, default=24, help="Gumbel simulations (A1: 24)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", type=Path, default=None, help="write JSON + report.md here")
    add_budget_arguments(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw)
    # Priced like every runner, though the price is 0: it plays no games.
    enforce_cli(RUNNER, {"positions": args.positions}, args)
    started = time.perf_counter()
    result = run_probe(
        positions=args.positions,
        board_size=args.board_size,
        max_turns=args.max_turns,
        k=args.k,
        n=args.n,
        seed=args.seed,
        device=args.device,
    )
    logger.info("probe finished in %.1f s", time.perf_counter() - started)
    evidence = evidence_block(result, raw)
    payload = {"evidence": evidence, **result}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / JSON_NAME).write_text(text + "\n", encoding="utf-8")
        (args.out_dir / REPORT_NAME).write_text(render_report(result, evidence), encoding="utf-8")
        logger.info("wrote %s", args.out_dir)
    for name, cell in result["cells"].items():
        print(
            f"{name:30s} H={cell['entropy_nats']['mean']:.3f} "
            f"maxp={cell['max_p']['mean']:.3f} "
            f">thr={cell['share_max_p_above_greedy_threshold']:.3f}"
        )
    if not math.isfinite(sum(c["max_p"]["mean"] for c in result["cells"].values())):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
