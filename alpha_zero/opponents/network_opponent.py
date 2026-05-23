"""Network-backed opponent — wraps a trained policy network as an :class:`Opponent`.

Useful for arena/evaluation matchups where you want to pit the current
training agent against a frozen checkpoint, or against a baseline network.
Decoupled from MCTS so it can serve as a fast, deterministic baseline.

The class is intentionally generic — it does not import anything chess-
specific. It works with any environment whose observation dict has
``spatial``, ``vector``, and ``action_mask`` keys and any network exposing
``forward(spatial, vector) -> (policy_logits, value)``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class _NetworkLike(Protocol):
    """Minimal interface required by :class:`NetworkOpponent`."""

    def eval(self) -> Any: ...
    def forward(self, spatial: Any, vector: Any) -> tuple[Any, Any]: ...


class NetworkOpponent:
    """Greedy / softmax-sampled action selection from a frozen network."""

    def __init__(
        self,
        network: _NetworkLike,
        *,
        name: str = "network",
        temperature: float = 0.0,
        device: str = "cpu",
    ) -> None:
        """Initialize.

        Args:
            network: A torch ``nn.Module``-like object with ``forward()``.
            name: Identifier used in tournament/logging output.
            temperature: 0 → argmax over legal actions; >0 → sample from
                temperature-softened policy restricted to the action mask.
            device: ``cpu`` / ``cuda`` / ``cuda:0`` — caller is responsible
                for moving the network there beforehand.
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        self.network = network
        self.name = name
        self.temperature = temperature
        self.device = device
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------ Opponent API

    def get_action(self, env: Any, action_mask: np.ndarray) -> int:
        obs = env.get_observation()
        policy = self._policy_from_network(obs)
        masked = policy * action_mask
        total = masked.sum()
        if total <= 0:
            # Pathological case: network produced ~zero mass on legal actions.
            # Fall back to uniform over legal actions.
            legal = np.flatnonzero(action_mask > 0)
            if legal.size == 0:
                raise RuntimeError("No legal actions available")
            return int(self._rng.choice(legal))
        normalized = masked / total
        if self.temperature == 0.0:
            return int(np.argmax(normalized))
        adjusted = np.power(normalized, 1.0 / self.temperature)
        adjusted_sum = adjusted.sum()
        if adjusted_sum <= 0:
            return int(np.argmax(normalized))
        return int(self._rng.choice(len(adjusted), p=adjusted / adjusted_sum))

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None

    # ------------------------------------------------------------ internals

    def _policy_from_network(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Run a forward pass, returning a 1-D numpy policy distribution.

        Imports torch lazily so the rest of the module is importable in
        environments without torch (e.g., docs / lightweight CI).
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "NetworkOpponent requires torch. Install with: pip install torch"
            ) from exc

        self.network.eval()
        spatial_t = torch.from_numpy(obs["spatial"]).float().unsqueeze(0).to(self.device)
        vector_t = torch.from_numpy(obs["vector"]).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.network.forward(spatial_t, vector_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return np.asarray(probs, dtype=np.float64)
