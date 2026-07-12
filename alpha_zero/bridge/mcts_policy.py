"""MCTS policy implementation for the BridgeAgentLoop.

Implements Civ6MctsPolicy which runs GumbelMCTS over a snapshot-seeded
MicroCivEnv adapter (BridgeMCTSEnv).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, SupportsFloat

import numpy as np

from alpha_zero.core.civ6_mcts import GumbelMCTS, GumbelMCTSConfig
from alpha_zero.core.protocols import EnvironmentProtocol
from alpha_zero.envs.microciv import City, MicroCivEnv, Tile, Unit
from alpha_zero.utils.constants import GAME

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants — no magic literals in class bodies
# ---------------------------------------------------------------------------

#: Default max turns for the simulated MicroCivEnv used inside MCTS.
#: Override via BridgeMCTSEnv(converted_state, max_turns=...).
_DEFAULT_MAX_TURNS: int = 500

#: Maximum number of Civ6 players this module will recognise in a single game.
#: Sourced from the authoritative game constant (also MicroCivEnv's hard limit)
#: rather than duplicating the literal here.
_MAX_CIV_PLAYERS: int = GAME.MAX_PLAYERS

#: Minimum number of players (Civ6 always has at least 2, even in singleplayer).
_MIN_CIV_PLAYERS: int = 2

# Default terrain / unit / city field values used when the bridge state does
# not supply a field.  Centralised here so tests can assert against them and
# code reviews can evaluate them in one place.
_DEFAULT_TERRAIN_TYPE: str = "plains"
_DEFAULT_TILE_YIELDS: dict[str, int] = {"food": 2, "production": 1}

_DEFAULT_UNIT_TYPE: str = "warrior"
_DEFAULT_UNIT_HEALTH: int = 100
_DEFAULT_UNIT_STRENGTH: int = 20
_DEFAULT_UNIT_MOVES: int = 2

_DEFAULT_CITY_NAME: str = "City"
_DEFAULT_CITY_POPULATION: int = 1
_DEFAULT_CITY_DEFENSE: int = 10
_DEFAULT_CITY_PRODUCTION: int = 0

#: diagnostics["policy_kind"] value for the MCTS policy (distinct from greedy=0.0).
_POLICY_KIND_MCTS: float = 1.0


class BridgeMCTSEnv(EnvironmentProtocol):
    """Snapshot-seeded MicroCivEnv adapter for GumbelMCTS.

    Wraps a MicroCivEnv initialized directly from a Bridge converted_state.
    Implements EnvironmentProtocol but forbids reset() since it represents
    a specific point in an ongoing game.

    Args:
        converted_state: Output of ``StateConverter.convert()`` for the
            current game frame.
        max_turns: Maximum turns for the inner simulation.  Defaults to
            :data:`_DEFAULT_MAX_TURNS`.
    """

    def __init__(
        self,
        converted_state: dict[str, Any],
        max_turns: int = _DEFAULT_MAX_TURNS,
    ) -> None:
        self._converted_state = converted_state
        board_size_tuple: tuple[int, int] = converted_state.get("_map_size", (16, 16))
        # The encoder outputs spatial tensors with size (board_size, board_size).
        # We need the max dimension for the MicroCivEnv.
        board_size = max(board_size_tuple[0], board_size_tuple[1])

        # Determine number of players from the state.  Start with the minimum
        # and expand as we find units / cities belonging to higher player IDs.
        players: set[int] = {0, 1}
        for u in converted_state.get("units", []):
            players.add(u.get("player", 0))
        for c in converted_state.get("cities", []):
            players.add(c.get("player", 0))
        num_players = max(_MIN_CIV_PLAYERS, min(_MAX_CIV_PLAYERS, max(players) + 1))

        self.env = MicroCivEnv(board_size=board_size, max_turns=max_turns, num_players=num_players)

        self.env.turn = converted_state.get("turn", 0)
        self.env.current_player = converted_state.get("current_player", 0)

        # Terrain
        self.env.terrain = {}
        for pos, t_dict in converted_state.get("terrain", {}).items():
            self.env.terrain[pos] = Tile(
                terrain_type=t_dict.get("type", _DEFAULT_TERRAIN_TYPE),
                features=list(t_dict.get("features", [])),
                yields=dict(t_dict.get("yields", _DEFAULT_TILE_YIELDS)),
            )

        # Highest valid player index for the env we just created. Civ6 assigns
        # very high ids to barbarians / city-states (e.g. 63); in N-player mode
        # StateConverter passes those owner ids through verbatim, so an unclamped
        # player would index past MicroCivEnv's per-player arrays and raise
        # IndexError mid-MCTS. Clamp to the last real slot (crash-safe; a no-op
        # in the 2-player demo where players are already 0/1).
        max_player_index = num_players - 1

        # Units
        self.env.units = []
        for u_dict in converted_state.get("units", []):
            self.env.units.append(
                Unit(
                    id=u_dict.get("_id", 0),
                    player=self._clamp_player(u_dict.get("player", 0), max_player_index),
                    unit_type=u_dict.get("type", _DEFAULT_UNIT_TYPE),
                    position=u_dict.get("position", (0, 0)),
                    health=u_dict.get("health", _DEFAULT_UNIT_HEALTH),
                    strength=u_dict.get("strength", _DEFAULT_UNIT_STRENGTH),
                    moves_remaining=u_dict.get("moves_remaining", _DEFAULT_UNIT_MOVES),
                )
            )

        # Cities
        self.env.cities = []
        for c_dict in converted_state.get("cities", []):
            self.env.cities.append(
                City(
                    id=c_dict.get("_id", 0),
                    player=self._clamp_player(c_dict.get("player", 0), max_player_index),
                    name=c_dict.get("name", _DEFAULT_CITY_NAME),
                    position=c_dict.get("position", (0, 0)),
                    population=c_dict.get("population", _DEFAULT_CITY_POPULATION),
                    defense=c_dict.get("defense", _DEFAULT_CITY_DEFENSE),
                    production=c_dict.get("production", _DEFAULT_CITY_PRODUCTION),
                    production_queue=None,
                )
            )

        # Reset internal IDs to avoid collisions if we step
        if self.env.units:
            self.env._next_unit_id = max(u.id for u in self.env.units) + 1
        if self.env.cities:
            self.env._next_city_id = max(c.id for c in self.env.cities) + 1

        self.env.done = False

    @staticmethod
    def _clamp_player(player: int, max_player_index: int) -> int:
        """Clamp a player index into ``[0, max_player_index]``.

        Civ6 assigns barbarians / city-states player ids far above the game's
        civ count (e.g. ``63``); passing those through would index past
        MicroCivEnv's per-player arrays and raise ``IndexError`` mid-MCTS.
        Clamping keeps rollouts crash-safe. Emits a debug log when it fires so
        out-of-range ids are visible without spamming normal 2-player play.

        Args:
            player: Raw player index from the converted state.
            max_player_index: Highest valid index (``num_players - 1``).

        Returns:
            ``player`` clamped into ``[0, max_player_index]``.
        """
        if player > max_player_index:
            logger.debug(
                "Clamping out-of-range player %d -> %d for MCTS env",
                player,
                max_player_index,
            )
            return max_player_index
        if player < 0:
            return 0
        return player

    # ------------------------------------------------------------------
    # Alternate constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_bridge_state(
        cls,
        converted_state: dict[str, Any],
        *,
        max_turns: int = _DEFAULT_MAX_TURNS,
    ) -> BridgeMCTSEnv:
        """Named constructor for clarity at call sites.

        Equivalent to ``BridgeMCTSEnv(converted_state, max_turns=max_turns)``
        but is the preferred form when the caller already has
        ``StateConverter.convert()`` output.
        """
        return cls(converted_state, max_turns=max_turns)

    # ------------------------------------------------------------------
    # EnvironmentProtocol implementation
    # ------------------------------------------------------------------

    def clone(self) -> BridgeMCTSEnv:
        """Deep-copy this env for MCTS tree expansion.

        The cloned instance shares *no* mutable state with the original.
        """

        new_env = BridgeMCTSEnv(self._converted_state)
        new_env.env = self.env.clone()
        return new_env

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Not supported — raises immediately.

        BridgeMCTSEnv represents a live game snapshot and may not be reset.
        Use :meth:`from_bridge_state` to create a fresh env from a new snapshot.
        """
        raise NotImplementedError(
            "BridgeMCTSEnv does not support reset(). "
            "It represents a live game snapshot.  "
            "Call BridgeMCTSEnv.from_bridge_state(new_state) instead."
        )

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def get_observation(self) -> dict[str, np.ndarray]:
        return self.env.get_observation()

    def get_info(self) -> dict[str, Any]:
        return self.env.get_info()

    def get_valid_actions(self) -> np.ndarray:
        return self.env.get_valid_actions()

    @property
    def current_player(self) -> int:
        return self.env.current_player

    @property
    def turn(self) -> int:
        return self.env.turn


class Civ6MctsPolicy:
    """Async policy that runs GumbelMCTS over the Civ6 bridge state.

    Args:
        network: Neural network implementing the ``NetworkProtocol``.
        config: :class:`~alpha_zero.core.civ6_mcts.GumbelMCTSConfig`.
        device: Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
        max_turns: Max turns passed to :class:`BridgeMCTSEnv` during search.
    """

    def __init__(
        self,
        network: Any,
        config: GumbelMCTSConfig,
        device: str = "cpu",
        max_turns: int = _DEFAULT_MAX_TURNS,
    ) -> None:
        self.network = network
        self.config = config
        self.device = device
        self.max_turns = max_turns
        self.mcts = GumbelMCTS(network=self.network, config=self.config, device=self.device)

    async def select_action(
        self,
        obs: dict[str, np.ndarray],
        legal_mask: np.ndarray,
        converted_state: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, float]]:
        """Run MCTS in a background thread to pick the best action.

        Args:
            obs: Observation tensors (not used directly; MCTS re-derives from env).
            legal_mask: Boolean mask of legal actions (not used directly).
            converted_state: Output of ``StateConverter.convert()`` — required
                to seed the simulation environment.

        Returns:
            ``(action_index, diagnostics)`` where *diagnostics* contains
            ``value_estimate``, ``policy_kind``, and ``mcts_sims``.

        Raises:
            ValueError: If *converted_state* is ``None``.
        """
        if converted_state is None:
            raise ValueError("Civ6MctsPolicy requires converted_state to seed the simulation env.")

        env = BridgeMCTSEnv.from_bridge_state(converted_state, max_turns=self.max_turns)

        # Run MCTS search in a thread so we don't block the async bridge heartbeat
        action_idx, search_stats = await asyncio.to_thread(self.mcts.search, env)

        diagnostics: dict[str, float] = {
            "value_estimate": float(search_stats.get("root_value", 0.0)),
            "policy_kind": _POLICY_KIND_MCTS,
            "mcts_sims": float(self.config.num_simulations),
        }

        return action_idx, diagnostics
