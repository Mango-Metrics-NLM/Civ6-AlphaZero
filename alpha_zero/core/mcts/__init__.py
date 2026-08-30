"""Monte Carlo Tree Search variants for Civ6 AlphaZero.

Split out of the former single-module ``alpha_zero.core.civ6_mcts`` so each
search variant is reviewable on its own. ``civ6_mcts`` remains as a re-export
shim, so historical import paths keep resolving to these same objects.

Dependency layering (no cycles)::

    mcts_protocols, gumbel_ops, base    -- leaves
    puct, simple                        -- depend on base + mcts_protocols
    gumbel                              -- mcts_protocols + gumbel_ops
    muzero                              -- mcts_protocols + gumbel_ops

``gumbel`` and ``muzero`` deliberately do not depend on ``base``: both carry
their own node and configuration types.
"""

from __future__ import annotations

from alpha_zero.core.mcts.base import MCTSConfig, MCTSNode, _env_current_player
from alpha_zero.core.mcts.gumbel import GumbelMCTS, GumbelMCTSConfig, GumbelMCTSNode
from alpha_zero.core.mcts.gumbel_ops import compute_completed_q, gumbel_sample_actions
from alpha_zero.core.mcts.mcts_protocols import (
    ActionSpace,
    MCTSEnvironment,
    MCTSNetwork,
    MuZeroNetworkProtocol,
)
from alpha_zero.core.mcts.muzero import MuZeroMCTS, MuZeroMCTSConfig, MuZeroNode
from alpha_zero.core.mcts.puct import Civ6MCTS
from alpha_zero.core.mcts.simple import SimpleMCTS

__all__ = [
    "ActionSpace",
    "Civ6MCTS",
    "GumbelMCTS",
    "GumbelMCTSConfig",
    "GumbelMCTSNode",
    "MCTSConfig",
    "MCTSEnvironment",
    "MCTSNetwork",
    "MCTSNode",
    "MuZeroMCTS",
    "MuZeroMCTSConfig",
    "MuZeroNetworkProtocol",
    "MuZeroNode",
    "SimpleMCTS",
    "_env_current_player",
    "compute_completed_q",
    "gumbel_sample_actions",
]
