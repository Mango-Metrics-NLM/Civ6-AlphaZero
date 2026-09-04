"""PUCT-based MCTS with neural-network guidance (``Civ6MCTS``)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from alpha_zero.core.mcts.backup import backpropagate_path, log_backup_diagnostics
from alpha_zero.core.mcts.base import MCTSConfig, MCTSNode, _env_current_player
from alpha_zero.core.mcts.mcts_protocols import MCTSEnvironment, MCTSNetwork
from alpha_zero.utils.device import place_network

logger = logging.getLogger(__name__)


class Civ6MCTS:
    """Monte Carlo Tree Search for Civ6.

    Uses neural network for policy prior and value estimation.
    Supports action masking for valid moves.
    Optionally integrates with ReflectionMemory for LATS-style learning.
    """

    def __init__(
        self,
        network: MCTSNetwork,
        config: MCTSConfig | None = None,
        device: str = "cpu",
        reflection_memory: Any | None = None,
        llm_prior_fn: Callable[[Any, np.ndarray], np.ndarray] | None = None,
        llm_prior_weight: float = 0.5,
    ) -> None:
        """Initialize MCTS.

        Args:
            network: Neural network implementing MCTSNetwork protocol.
            config: MCTS configuration.
            device: Device for neural network inference.
            reflection_memory: Optional ReflectionMemory for value adjustments
                based on past failures. When provided, actions that led to
                failures in similar states will be penalized.
            llm_prior_fn: Optional callable ``(env, action_mask) -> prior``
                returning a 1-D distribution over actions (same length as
                ``action_mask``). When set, the network policy at expansion
                time is blended as
                ``policy = (1 - w) * network_policy + w * llm_prior``,
                renormalised over legal actions. ``None`` (default) leaves
                MCTS behaviour byte-identical to the pre-extension code path.
            llm_prior_weight: Blend weight ``w`` in ``[0, 1]``. Only used
                when ``llm_prior_fn`` is provided.
        """
        self.network = network
        self.config = config or MCTSConfig()
        # Resolve the request and place the network in one step: storing an
        # unresolved "cuda" here while the network sat on cpu was a real
        # source of mixed-device RuntimeErrors.
        self.device = place_network(network, device)
        # This search owns its randomness. Drawing from the global `np.random`
        # state meant a caller could seed its environments and still get a
        # different search on every run -- the defect that made a "seeded"
        # self-play run only half determined.
        self._rng = np.random.default_rng(self.config.seed)
        self._reflection_memory = reflection_memory
        if not 0.0 <= llm_prior_weight <= 1.0:
            raise ValueError(f"llm_prior_weight must be in [0, 1]; got {llm_prior_weight}")
        self._llm_prior_fn = llm_prior_fn
        self._llm_prior_weight = llm_prior_weight
        self._llm_warned = False
        self._search_count = 0
        logger.debug(
            "Civ6MCTS initialized",
            extra={
                "num_simulations": self.config.num_simulations,
                "c_puct": self.config.c_puct,
                "device": device,
                "has_reflection_memory": reflection_memory is not None,
                "has_llm_prior": llm_prior_fn is not None,
                "llm_prior_weight": llm_prior_weight,
            },
        )

    def search(
        self,
        env: MCTSEnvironment,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Run MCTS search from current state.

        Args:
            env: Game environment (must support clone()).
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            Tuple of (action_probs, root_value).
        """
        self._search_count += 1
        logger.debug(
            "Starting MCTS search",
            extra={
                "search_id": self._search_count,
                "num_simulations": self.config.num_simulations,
                "add_noise": add_noise,
            },
        )

        # A single scalar value cannot express a >2-player game: "flip when the
        # acting player changes" would hand a third player +v after two flips,
        # modelling them as the first player's ally. Rather than raise - N-player
        # training legitimately drives this class through CurrentPlayerValueView
        # - fall back to the historical per-ply alternation there by leaving
        # to_play unset. That keeps >2-player behaviour byte-identical to before
        # this fix while the 2-player product path gets the correct backup.
        # alpha_zero.multiplayer.mcts.NPlayerMCTS keeps a real per-player vector.
        track_to_play = int(getattr(env, "num_players", 2) or 2) <= 2
        if not track_to_play:
            logger.debug(
                "Civ6MCTS: num_players > 2, using legacy per-ply value alternation; "
                "prefer NPlayerMCTS for a correct multiplayer backup"
            )

        # Create root node. Its to_play anchors every sign decision below.
        root = MCTSNode(to_play=_env_current_player(env) if track_to_play else None)

        # Get initial policy and value from network
        obs = env.get_observation()
        spatial = torch.from_numpy(obs["spatial"]).float().to(self.device)
        vector = torch.from_numpy(obs["vector"]).float().to(self.device)
        action_mask = obs["action_mask"]

        policy, value = self._evaluate(spatial, vector, action_mask)

        # Optional LLM-prior blend (no-op when llm_prior_fn is None).
        policy = self._blend_llm_prior(env, policy, action_mask)

        # Expand root
        root.expand(policy, action_mask)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.is_expanded():
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            sim_env = env.clone()

            # Selection: traverse tree to leaf
            search_path = [node]
            while node.is_expanded():
                action, node = node.select_child(self.config.c_puct)
                search_path.append(node)

                # Take action in environment
                _, _, terminated, truncated, _ = sim_env.step(action)
                # Record who moves *after* the step. Reading current_player
                # before stepping would capture the parent's mover and invert
                # every sign decision downstream.
                node.to_play = _env_current_player(sim_env) if track_to_play else None
                if terminated or truncated:
                    break

            # Evaluation
            if terminated or truncated:
                # Game over: use actual result
                info = sim_env.get_info()
                winner = info.get("winner")
                if winner is None:
                    value = 0.0  # Draw
                elif winner == sim_env.current_player:
                    value = 1.0
                else:
                    value = -1.0
            else:
                # Use neural network
                obs = sim_env.get_observation()
                spatial = torch.from_numpy(obs["spatial"]).float().to(self.device)
                vector = torch.from_numpy(obs["vector"]).float().to(self.device)
                action_mask = obs["action_mask"]

                policy, value = self._evaluate(spatial, vector, action_mask)

                # Optional LLM-prior blend (no-op when llm_prior_fn is None).
                policy = self._blend_llm_prior(sim_env, policy, action_mask)

                # Expand leaf node
                node.expand(policy, action_mask)

            # Backpropagation. `value` is from the perspective of whoever moves
            # at the leaf, so the sign flips only where the acting player
            # changes - not on every ply. MicroCiv rotates current_player only
            # in _end_turn, so a player makes many moves per turn. This was the
            # first variant corrected (2026-08-25); it now shares that rule with
            # every other variant instead of keeping its own copy.
            log_backup_diagnostics(search_path, search_name="Civ6MCTS")
            backpropagate_path(search_path, value)

            # Explicit cleanup to prevent memory accumulation
            del sim_env

        # Get action probabilities from visit counts
        action_probs = self._get_action_probs(root, env.action_space.n)

        return action_probs, root.value

    def _evaluate(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: np.ndarray,
        action: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Evaluate state using neural network.

        Optionally applies value adjustments from reflection memory
        to penalize actions that led to failures in similar states.

        Args:
            spatial: Spatial features.
            vector: Vector features.
            action_mask: Valid action mask.
            action: Optional action to get reflection-based adjustment for.

        Returns:
            Tuple of (policy, value).
        """
        self.network.eval()
        with torch.no_grad():
            mask_tensor = torch.from_numpy(action_mask).float().to(self.device)
            policy_probs, value = self.network.predict(
                spatial.unsqueeze(0),
                vector.unsqueeze(0),
                mask_tensor.unsqueeze(0),
            )
            policy = policy_probs.squeeze(0).cpu().numpy()
            # Ensure value is a scalar - squeeze all dimensions then convert
            value_scalar = float(value.squeeze().cpu().item())

        # Apply reflection-based value adjustment if available
        if self._reflection_memory is not None and action is not None:
            try:
                # Check if reflection memory is initialized
                if (
                    hasattr(self._reflection_memory, "initialized")
                    and self._reflection_memory.initialized
                ):
                    state = {
                        "spatial": spatial.cpu().numpy(),
                        "vector": vector.cpu().numpy(),
                    }
                    adjustment = self._reflection_memory.get_value_adjustment_sync(state, action)
                    value_scalar = value_scalar + adjustment
            except Exception as e:
                logger.debug(f"Reflection memory adjustment failed: {e}")

        return policy, value_scalar

    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        """Add Dirichlet noise to root node priors.

        Args:
            node: Root node.
        """
        actions = list(node.children.keys())
        noise = self._rng.dirichlet([self.config.dirichlet_alpha] * len(actions))

        eps = self.config.dirichlet_epsilon
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _blend_llm_prior(
        self,
        env: Any,
        network_policy: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Blend the network policy with an optional LLM-supplied prior.

        Returns the input policy unchanged when ``llm_prior_fn`` is ``None``,
        so behaviour is byte-identical to the pre-extension code path.

        On any failure inside the user-supplied callable (exception or
        shape mismatch), we fall back to the network policy and log a
        warning **once per Civ6MCTS instance** (gated by ``_llm_warned``)
        so a misbehaving LLM cannot spam the MCTS hot loop. MCTS never
        crashes due to advisor failure.
        """
        if self._llm_prior_fn is None:
            return network_policy
        try:
            llm_prior = self._llm_prior_fn(env, action_mask)
        except Exception as exc:  # pragma: no cover - defensive logging
            if not self._llm_warned:
                logger.warning("llm_prior_fn raised %s; using network policy", exc)
                self._llm_warned = True
            return network_policy
        llm_prior = np.asarray(llm_prior, dtype=np.float64)
        if llm_prior.shape != network_policy.shape:
            if not self._llm_warned:
                logger.warning(
                    "llm_prior shape %s != policy shape %s; using network policy",
                    llm_prior.shape,
                    network_policy.shape,
                )
                self._llm_warned = True
            return network_policy
        # Mask + blend + renormalise.
        masked_llm = llm_prior * (action_mask > 0).astype(np.float64)
        total = masked_llm.sum()
        if total <= 0:
            return network_policy
        masked_llm = masked_llm / total
        w = float(self._llm_prior_weight)
        blended = (1.0 - w) * np.asarray(network_policy, dtype=np.float64) + w * masked_llm
        blended = blended * (action_mask > 0).astype(np.float64)
        blended_sum = blended.sum()
        if blended_sum <= 0:
            return network_policy
        renormalised: np.ndarray = (blended / blended_sum).astype(network_policy.dtype)
        return renormalised

    def _get_action_probs(
        self,
        root: MCTSNode,
        action_space_size: int,
    ) -> np.ndarray:
        """Get action probabilities from root visit counts.

        Args:
            root: Root node.
            action_space_size: Total number of actions.

        Returns:
            Action probability distribution.
        """
        action_probs = np.zeros(action_space_size, dtype=np.float32)

        # Collect visit counts
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        # Normalize
        total_visits = action_probs.sum()
        if total_visits > 0:
            action_probs /= total_visits

        return action_probs

    def select_action(
        self,
        action_probs: np.ndarray,
        temperature: float = 1.0,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Select action based on probabilities and temperature.

        Args:
            action_probs: Action probability distribution.
            temperature: Temperature for action selection.
            action_mask: Optional valid action mask (1=valid, 0=invalid).
                        If provided, ensures only valid actions are selected.

        Returns:
            Selected action index.

        Raises:
            ValueError: If no valid actions available when mask is provided.
        """
        # Apply mask if provided to ensure only valid actions are selected
        if action_mask is not None:
            action_probs = action_probs * action_mask
            if action_probs.sum() < 1e-8:
                # Fallback: select first valid action
                valid_actions = np.where(action_mask > 0)[0]
                if len(valid_actions) == 0:
                    raise ValueError("No valid actions available")
                return int(valid_actions[0])
            # Renormalize after masking
            action_probs = action_probs / action_probs.sum()

        if temperature == 0:
            # Greedy selection
            return int(np.argmax(action_probs))

        # Apply temperature
        if temperature != 1.0:
            probs = np.power(action_probs + 1e-8, 1.0 / temperature)
            probs = probs / probs.sum()
        else:
            probs = action_probs

        # Sample action
        return int(self._rng.choice(len(probs), p=probs))
