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
``sigma_halving``
    The A12 change alone: halving ranks by ``g + log pi + sigma(completedQ)``,
    ``sigma = (c_visit + max_b N(b)) * c_scale``, on this repository's own
    completion (raw Q, no normalisation). Target unchanged (repository formula).
``paper_affine``
    Danihelka et al. (2022), Go/chess constants (c_visit 50, c_scale 1.0,
    Appendix F), with Q mapped to [0, 1] by the fixed affine map ``(q + 1) / 2``.
    The paper says only that Go/chess Q values "are normalized to be from the
    [0, 1] interval"; for a value in [-1, 1] the fixed map is the conventional
    reading (MuZero's pseudocode uses known bounds (-1, 1) for board games).
    ``v_mix`` completion, sigma in the halving score and in a softmax target
    over every legal action.
``mctx_minmax_c1``
    As ``paper_affine`` but with ``mctx``'s per-node min-max rescale
    (``qtransform_completed_by_mix_value``, ``rescale_values=True``) and the
    paper's board-game c_scale 1.0. **No published configuration pairs these**:
    ``mctx`` pairs min-max with ``value_scale = 0.1``; the paper documents a
    min-max form only for Atari (section 7).
``mctx_default``
    ``mctx``'s defaults: per-node min-max, ``value_scale`` (c_scale) 0.1,
    ``maxvisit_init`` (c_visit) 50.
``current_search_paper_affine_target`` / ``current_search_mctx_minmax_c1_target``
    The *same* tree as ``current``, scored with that target formula -- isolates
    the target formula from the search path.

Deliberately *not* varied: interior (non-root) selection stays this
repository's PUCT (``c_puct = c_visit / 50``), where ``mctx`` uses the
deterministic ``argmax(pi' - N / (1 + sum N))``; and the halving schedule, which
already matches ``mctx`` (``gumbel_ops.halving_schedule``).

Usage::

    python scripts/probe_target_sharpness.py --seeds 42 \\
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

#: Ways to map completed Q onto [0, 1] before sigma.
NORM_AFFINE = "affine"  # (q + 1) / 2: fixed, for values known to lie in [-1, 1]
NORM_MINMAX = "minmax"  # mctx: per node, over the completed-Q vector
NORMALISATIONS = (NORM_AFFINE, NORM_MINMAX)

#: (cell name, search key, target key). Search keys index ``_SEARCHES``; a target
#: key is ``"repo"`` (``build_completed_q_policy``) or a normalisation, scored with
#: the search's own c_visit / c_scale.
CELLS: tuple[tuple[str, str, str], ...] = (
    ("current", "current", "repo"),
    ("sigma_halving", "sigma_halving", "repo"),
    ("paper_affine", "paper_affine", NORM_AFFINE),
    ("current_search_paper_affine_target", "current", NORM_AFFINE),
    ("mctx_minmax_c1", "mctx_minmax_c1", NORM_MINMAX),
    ("current_search_mctx_minmax_c1_target", "current", NORM_MINMAX),
    ("mctx_default", "mctx_default", NORM_MINMAX),
)

#: search key -> (halving mode, normalisation or None, c_scale override or None).
_SEARCHES: dict[str, tuple[str, str | None, float | None]] = {
    "current": (HALVING_CURRENT, None, None),
    "sigma_halving": (HALVING_SIGMA_RAW, None, None),
    "paper_affine": (HALVING_PAPER, NORM_AFFINE, PAPER_C_SCALE),
    "mctx_minmax_c1": (HALVING_PAPER, NORM_MINMAX, PAPER_C_SCALE),
    "mctx_default": (HALVING_PAPER, NORM_MINMAX, MCTX_DEFAULT_C_SCALE),
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
    normalisation: str = NORM_MINMAX,
) -> np.ndarray:
    """``sigma(completedQ)`` with ``v_mix`` completion and a chosen [0, 1] map.

    Args are aligned arrays over the *legal* actions only. Visited actions keep
    their own Q; unvisited ones take ``v_mix`` (``mctx``'s
    ``_compute_mixed_value``); the vector is mapped to [0, 1] -- ``"minmax"``
    per node as ``mctx``'s ``_rescale_qvalues``, or ``"affine"`` ``(q + 1) / 2``
    -- and multiplied by ``(c_visit + max N) * c_scale``. With ``"minmax"`` this
    is ``qtransform_completed_by_mix_value``.
    """
    if normalisation not in NORMALISATIONS:
        raise ValueError(f"unknown normalisation {normalisation!r}")
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
    if normalisation == NORM_AFFINE:
        normalised = (completed + 1.0) / 2.0
    else:
        low, high = float(completed.min()), float(completed.max())
        normalised = (completed - low) / max(high - low, MCTX_RESCALE_EPSILON)
    max_visits = float(visits.max()) if visits.size else 0.0
    return (c_visit + max_visits) * c_scale * normalised


def paper_target(
    root: GumbelMCTSNode,
    raw_value: float,
    action_space: int,
    *,
    c_visit: float,
    c_scale: float,
    normalisation: str = NORM_MINMAX,
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
        normalisation=normalisation,
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
        self,
        network: Any,
        config: GumbelMCTSConfig,
        *,
        halving: str,
        normalisation: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(network, config, device=device)
        if halving not in (HALVING_CURRENT, HALVING_SIGMA_RAW, HALVING_PAPER):
            raise ValueError(f"unknown halving mode {halving!r}")
        if (halving == HALVING_PAPER) != (normalisation is not None):
            raise ValueError("a normalisation is required by, and only by, the paper halving")
        self.halving = halving
        self.normalisation = normalisation
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
            normalisation=self.normalisation or NORM_MINMAX,
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

    prior_stats: dict[str, list[float]] = {"entropy": [], "max_p": []}
    # Spread of visited root-child Q under the current search. The current
    # target's sharpness scales with this absolute spread (raw Q, no
    # normalisation); the paper's does not (min-max normalised).
    q_spreads: list[float] = []
    root_values: list[float] = []
    agree: dict[str, int] = {key: 0 for key in _SEARCHES if key != "current"}
    rows: list[dict[str, Any]] = []
    forced = 0

    with inference_thread_limit():
        for index in range(positions):
            env, _played = sample_position(board_size, max_turns, seed, index)
            legal = int((env.get_valid_actions() > 0).sum())
            forced += int(legal <= 1)
            search_seed = derive_seed(seed, STREAM_SEARCH, index)
            roots: dict[str, ProbeGumbelMCTS] = {}
            for key, (halving, normalisation, c_scale) in _SEARCHES.items():
                config = dataclasses.replace(base, seed=search_seed)
                if c_scale is not None:
                    config = dataclasses.replace(config, c_scale=c_scale, c_visit=PAPER_C_VISIT)
                mcts = ProbeGumbelMCTS(
                    network, config, halving=halving, normalisation=normalisation, device=device
                )
                mcts.search(env)
                roots[key] = mcts
            current = roots["current"]
            for key in agree:
                agree[key] += int(roots[key].last_choice == current.last_choice)

            row: dict[str, Any] = {"index": index, "turn": int(env.turn), "legal": legal}
            if current.last_root is not None:
                priors = np.array([c.prior for c in current.last_root.children.values()])
                prior_stats["entropy"].append(entropy(priors))
                prior_stats["max_p"].append(float(priors.max()))
                visited_q = [
                    _root_child_q(current.last_root, c)
                    for c in current.last_root.children.values()
                    if c.visit_count > 0
                ]
                if visited_q:
                    q_spreads.append(max(visited_q) - min(visited_q))
                root_values.append(abs(current.last_raw_value))
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
                        normalisation=target_key,
                    )
                if target.sum() <= 0.0 and mcts.last_choice >= 0:
                    target[mcts.last_choice] = 1.0  # GumbelMCTS's own fallback
                h, mp = entropy(target), float(target.max())
                row[name] = {
                    "entropy": round(h, 6),
                    "max_p": round(mp, 6),
                    "support": int((target > 0).sum()),
                    # The move the arena (temperature 0) and late self-play play.
                    "argmax": int(target.argmax()),
                }
            rows.append(row)

    threshold = float(base.greedy_threshold)
    return {
        "seed": seed,
        "positions_forced": forced,
        "temperature_phase_positions": sum(r["turn"] < base.temperature_threshold for r in rows),
        "prior": {
            "entropy_nats": _summary(prior_stats["entropy"]),
            "max_p": _summary(prior_stats["max_p"]),
        },
        "legal_actions": _summary([float(r["legal"]) for r in rows]),
        "current_visited_q_spread": _summary(q_spreads or [0.0]),
        "abs_root_network_value": _summary(root_values or [0.0]),
        "chosen_move_agreement_with_current": {
            key: round(count / positions, 6) for key, count in agree.items()
        },
        "cells": cell_summaries(rows, threshold, base.temperature_threshold),
        "positions": rows,
    }


def cell_summaries(
    rows: list[dict[str, Any]], threshold: float, temperature_threshold: int
) -> dict[str, Any]:
    """Per-variant statistics over ``rows`` (one seed's positions, or several pooled)."""
    early = [r["turn"] < temperature_threshold for r in rows]
    cells: dict[str, Any] = {}
    for name, search_key, target_key in CELLS:
        max_p = [float(r[name]["max_p"]) for r in rows]
        above = [mp > threshold for mp in max_p]
        early_above = [a for a, e in zip(above, early, strict=True) if e]
        cells[name] = {
            "search": search_key,
            "target": target_key,
            "entropy_nats": _summary([float(r[name]["entropy"]) for r in rows]),
            "max_p": _summary(max_p),
            "support": _summary([float(r[name]["support"]) for r in rows]),
            "share_max_p_above_greedy_threshold": round(sum(above) / len(above), 6),
            "share_above_threshold_in_temperature_phase": (
                round(sum(early_above) / len(early_above), 6) if early_above else None
            ),
            "share_max_p_above_0_5": round(sum(mp > 0.5 for mp in max_p) / len(max_p), 6),
            # The target's argmax is the move played at temperature 0 (the arena,
            # and self-play after the temperature threshold).
            "argmax_agrees_with_current": round(
                sum(r[name]["argmax"] == r["current"]["argmax"] for r in rows) / len(rows), 6
            ),
        }
    return cells


def run_probes(
    *,
    seeds: list[int],
    positions: int,
    board_size: int,
    max_turns: int,
    k: int,
    n: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """:func:`run_probe` for each seed, plus statistics pooled over every seed.

    Each seed is a different random-init network *and* a different set of
    positions. Several are needed: the untrained value head differs a great deal
    between initialisations, and the targets with it.
    """
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct; got {seeds}")
    base = search_config(board_size, max_turns, k, n, device, seeds[0])
    per_seed = [
        run_probe(
            positions=positions,
            board_size=board_size,
            max_turns=max_turns,
            k=k,
            n=n,
            seed=seed,
            device=device,
        )
        for seed in seeds
    ]
    pooled_rows = [row for run in per_seed for row in run["positions"]]
    pooled_cells = cell_summaries(
        pooled_rows, float(base.greedy_threshold), base.temperature_threshold
    )
    for name, cell in pooled_cells.items():
        shares = np.array(
            [run["cells"][name]["share_max_p_above_greedy_threshold"] for run in per_seed]
        )
        se = float(shares.std(ddof=1) / math.sqrt(len(shares))) if len(shares) > 1 else None
        cell["share_across_seeds"] = {
            "mean": round(float(shares.mean()), 6),
            "se": None if se is None else round(se, 6),
            "min": round(float(shares.min()), 6),
            "max": round(float(shares.max()), 6),
        }
    return {
        "probe": RUNNER,
        "probe_schema_version": SCHEMA_VERSION,
        "settings": {
            "seeds": list(seeds),
            "positions_per_seed": positions,
            "board_size": board_size,
            "max_turns": max_turns,
            "gumbel_k": k,
            "gumbel_sims": n,
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
                "normalisation": {
                    "paper_affine": "(q + 1) / 2 (paper: 'normalized to [0, 1]', method unstated)",
                    "mctx_minmax_c1": "per-node min-max (mctx) with the paper's c_scale 1.0",
                    "mctx_default": "per-node min-max with mctx's value_scale 0.1",
                },
                "halving_score": "g + log pi + sigma(completedQ_normalised)",
                "target": "softmax(log pi + sigma) over all legal actions",
            },
        },
        "games_played": 0,
        "pooled": {
            "positions": len(pooled_rows),
            "temperature_phase_positions": sum(r["temperature_phase_positions"] for r in per_seed),
            "positions_forced": sum(r["positions_forced"] for r in per_seed),
            "cells": pooled_cells,
        },
        "per_seed": per_seed,
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
        "seed": settings["seeds"][0],
        "seeds": settings["seeds"],
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
            "sharpness (entropy, max-p, share above greedy_threshold) of the Gumbel policy "
            f"target at k={settings['gumbel_k']}, n={settings['gumbel_sims']} on "
            f"{settings['positions_per_seed']} random-play positions per seed, for "
            f"{len(settings['seeds'])} untrained networks (one per seed), under the current "
            "settings and the paper's",
        ],
        "non_claims": [
            "not evidence of playing strength or of learning",
            "untrained networks only; a trained network's targets may be sharper or flatter",
            "positions come from random play, not from self-play",
            "interior (non-root) selection is this repository's PUCT in every variant",
            "the paper does not state how Go/chess Q is normalised; paper_affine is one "
            "reading of it, mctx_minmax_c1 is mctx's rescale at the paper's c_scale",
            "the seed field is the first of several; every seed is listed under seeds",
            "per-position rows are omitted; they are regenerated from the seeds",
        ],
    }
    if sha is None:
        block["git_sha_null_reason"] = "no CIV6AZ_GIT_SHA and no resolvable .git"
    return block


def _cell_row(name: str, cell: dict[str, Any]) -> str:
    tphase = cell["share_above_threshold_in_temperature_phase"]
    return (
        f"| `{name}` | {cell['search']} | {cell['target']} | "
        f"{cell['entropy_nats']['mean']:.3f} / {cell['entropy_nats']['median']:.3f} | "
        f"{cell['max_p']['mean']:.3f} / {cell['max_p']['median']:.3f} | "
        f"{cell['share_max_p_above_greedy_threshold']:.3f} | "
        f"{'n/a' if tphase is None else f'{tphase:.3f}'} | "
        f"{cell['support']['median']:.0f} | {cell['argmax_agrees_with_current']:.3f} |"
        + (
            f" {spread['se']:.3f} [{spread['min']:.3f}, {spread['max']:.3f}] |"
            if (spread := cell.get("share_across_seeds")) and spread["se"] is not None
            else " n/a |"
        )
    )


def render_report(result: dict[str, Any], evidence: dict[str, Any]) -> str:
    """A markdown report that a reader can follow without the JSON."""
    s = result["settings"]
    cur = s["current"]
    pooled = result["pooled"]
    names = [name for name, _, _ in CELLS]
    header = (
        "| Variant | Search (halving score) | Target | Entropy mean / median (nats) "
        "| max-p mean / median | Share max-p > threshold | Same in temperature phase "
        "| Support (median) | Argmax = `current` argmax "
        "| Share across seeds: SE [min, max] |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [
        "# Gumbel policy-target sharpness probe (LF-A5 / LF-A12)",
        "",
        f"**Date:** {evidence['generated_utc'][:10]} · **Search epoch:** {SEARCH_EPOCH} · "
        f"**Games played:** 0 · **Raw data:** [`{JSON_NAME}`]({JSON_NAME})",
        "",
        f"For each of {len(s['seeds'])} seeds ({', '.join(str(x) for x in s['seeds'])}), a "
        f"randomly initialised `MicroCivNetwork` (`torch.manual_seed(seed)`, 0 games of "
        f"training) searches {s['positions_per_seed']} positions from seeded uniform-random "
        f"play at board {s['board_size']} / max_turns {s['max_turns']}, with k={s['gumbel_k']}, "
        f"n={s['gumbel_sims']} (A1's budget). `greedy_threshold` = {cur['greedy_threshold']}: "
        "above it `select_action` ignores temperature, so on that move the self-play "
        f"temperature schedule (temperature {cur['temperature']} before turn "
        f"{cur['temperature_threshold']}) is inert.",
        "",
        "```",
        evidence["command"],
        "```",
        "",
        f"## Pooled over every seed ({pooled['positions']} positions, "
        f"{pooled['temperature_phase_positions']} in the temperature phase, "
        f"{pooled['positions_forced']} forced)",
        "",
        header,
        *[_cell_row(name, cell) for name, cell in pooled["cells"].items()],
        "",
        "## Per seed: share of positions with max-p above the threshold (mean max-p)",
        "",
        "| Seed | abs root value, median | Visited-Q spread, median | "
        + " | ".join(f"`{n}`" for n in names)
        + " | Halving winner agrees with `current` (sigma_halving / paper_affine / "
        "mctx_minmax_c1) |",
        "|---|---|---|" + "---|" * len(names) + "---|",
    ]
    for run in result["per_seed"]:
        agree = run["chosen_move_agreement_with_current"]
        cells = " | ".join(
            f"{run['cells'][n]['share_max_p_above_greedy_threshold']:.3f} "
            f"({run['cells'][n]['max_p']['mean']:.3f})"
            for n in names
        )
        lines.append(
            f"| {run['seed']} | {run['abs_root_network_value']['median']:.3f} | "
            f"{run['current_visited_q_spread']['median']:.3f} | {cells} | "
            f"{agree['sigma_halving']:.2f} / {agree['paper_affine']:.2f} / "
            f"{agree['mctx_minmax_c1']:.2f} |"
        )
    lines += [
        "",
        "The current target's sharpness scales with the *absolute* spread of the visited "
        "root-child Q values: raw Q in [-1, 1] times `(c_visit + max N) * c_scale`, i.e. 57 "
        "per unit of Q at n=24. Under the affine reading the paper's coefficient per unit of "
        "raw Q is half that (28.5). Under per-node min-max any spread at all is stretched to "
        "the full 57-logit range.",
        "",
        "## Variants",
        "",
        f"- `current` — A1's settings: c_visit {cur['c_visit']}, c_scale {cur['c_scale']}, "
        f"target `{cur['policy_target']}`; halving ranks by {cur['halving_score']}; target "
        "restricted to visited actions; completion `q + ((maxN-N)/maxN)·v_root`, raw Q.",
        "- `sigma_halving` — LF-A12 alone: halving ranks by `g + log pi + (c_visit + max N)"
        "·c_scale·completedQ` on this repository's completion; repository target.",
        "- `paper_affine` — Danihelka et al. (2022) Go/chess constants (c_visit 50, c_scale 1.0, "
        "Appendix F) with Q mapped to [0, 1] by `(q + 1) / 2`; `v_mix` completion; sigma in the "
        "halving score and in a softmax target over all legal actions. The paper says only that "
        "Go/chess Q values are normalised to [0, 1]; this is the fixed-bounds reading.",
        "- `mctx_minmax_c1` — the same, but with `mctx`'s per-node min-max rescale "
        "(`qtransform_completed_by_mix_value`) at c_scale 1.0. No published configuration pairs "
        "the two: `mctx` uses min-max with value_scale 0.1, and the paper shows a min-max form "
        "only for Atari (section 7).",
        f"- `mctx_default` — `mctx`'s defaults: per-node min-max, c_scale "
        f"{s['paper']['mctx_default_c_scale']}, c_visit 50.",
        "- `current_search_*_target` — the `current` tree, scored with that target formula.",
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
    p.add_argument("--positions", type=int, default=200, help="positions per seed")
    p.add_argument("--board_size", type=int, default=8)
    p.add_argument("--max_turns", type=int, default=60)
    p.add_argument("--k", type=int, default=8, help="Gumbel sampled actions (A1: 8)")
    p.add_argument("--n", type=int, default=24, help="Gumbel simulations (A1: 24)")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="one random-init network and position set per seed",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", type=Path, default=None, help="write JSON + report.md here")
    p.add_argument(
        "--per_position",
        action="store_true",
        help="keep the per-position rows in the JSON (omitted by default; seeds regenerate them)",
    )
    add_budget_arguments(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw)
    # Priced like every runner, though the price is 0: it plays no games.
    enforce_cli(RUNNER, {"positions": args.positions * len(args.seeds)}, args)
    started = time.perf_counter()
    result = run_probes(
        seeds=list(args.seeds),
        positions=args.positions,
        board_size=args.board_size,
        max_turns=args.max_turns,
        k=args.k,
        n=args.n,
        device=args.device,
    )
    logger.info("probe finished in %.1f s", time.perf_counter() - started)
    evidence = evidence_block(result, raw)
    payload = {"evidence": evidence, **result}
    if not args.per_position:
        payload["per_seed"] = [
            {k: v for k, v in run.items() if k != "positions"} for run in result["per_seed"]
        ]
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / JSON_NAME).write_text(text + "\n", encoding="utf-8")
        (args.out_dir / REPORT_NAME).write_text(render_report(result, evidence), encoding="utf-8")
        logger.info("wrote %s", args.out_dir)
    for name, cell in result["pooled"]["cells"].items():
        print(
            f"{name:30s} H={cell['entropy_nats']['mean']:.3f} "
            f"maxp={cell['max_p']['mean']:.3f} "
            f">thr={cell['share_max_p_above_greedy_threshold']:.3f}"
        )
    if not math.isfinite(sum(c["max_p"]["mean"] for c in result["pooled"]["cells"].values())):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
