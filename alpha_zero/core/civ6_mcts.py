"""Backwards-compatible re-exports for the MCTS family.

The implementation moved to :mod:`alpha_zero.core.mcts`, one module per search
variant, so each is reviewable on its own. This module re-exports the public
names so historical import paths keep working; both paths resolve to the same
objects.

Where each name now lives:

===========================  ===========================================
Name                         Module
===========================  ===========================================
``MCTSNetwork`` and friends   :mod:`alpha_zero.core.mcts.mcts_protocols`
``MCTSConfig``, ``MCTSNode``  :mod:`alpha_zero.core.mcts.base`
``Civ6MCTS``                  :mod:`alpha_zero.core.mcts.puct`
``SimpleMCTS``                :mod:`alpha_zero.core.mcts.simple`
``GumbelMCTS`` and friends    :mod:`alpha_zero.core.mcts.gumbel`
``gumbel_sample_actions``     :mod:`alpha_zero.core.mcts.gumbel_ops`
``MuZeroMCTS`` and friends    :mod:`alpha_zero.core.mcts.muzero`
===========================  ===========================================
"""

from __future__ import annotations

from alpha_zero.core.mcts.base import MCTSConfig, MCTSNode, _env_current_player
from alpha_zero.core.mcts.gumbel import (
    POLICY_TARGET_ONE_HOT,
    POLICY_TARGET_VISIT_COUNTS,
    VALID_POLICY_TARGETS,
    GumbelMCTS,
    GumbelMCTSConfig,
    GumbelMCTSNode,
)
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
    "POLICY_TARGET_ONE_HOT",
    "POLICY_TARGET_VISIT_COUNTS",
    "VALID_POLICY_TARGETS",
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
