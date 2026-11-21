"""MCTS policy implementation for the BridgeAgentLoop.

Implements Civ6MctsPolicy which runs GumbelMCTS over a snapshot-seeded
MicroCivEnv adapter (BridgeMCTSEnv).

The search plays in the MicroCiv simulator, not in the real game: its network
must take the simulator's observation, and the action it returns is an index
in the simulator's action space. Nothing translates that index into a live
command yet (GM-B6), so ``civ6_local`` refuses to start this mode on a live
game (GM-B5). How a live state is stood in for inside the simulator is in
:mod:`alpha_zero.bridge.microciv_proxy`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from alpha_zero.bridge.microciv_proxy import (
    SIM_MOUNTAIN,
    raw_unit_types,
    simulator_terrain,
    simulator_unit_class,
)
from alpha_zero.bridge.search_translator import SearchTranslator
from alpha_zero.core.civ6_mcts import GumbelMCTS, GumbelMCTSConfig
from alpha_zero.core.protocols import EnvironmentProtocol
from alpha_zero.envs.microciv import City, MicroCivEnv, Tile, Unit
from alpha_zero.utils.constants import GAME, REWARDS

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

#: What is sent when the search has nothing playable to say.
_END_TURN_COMMAND: dict[str, Any] = {"action_type": "end_turn"}

#: Turns the simulation is always left before its turn limit.
_DEFAULT_MIN_LOOKAHEAD: int = 10

#: What a city that is already building is seeded with. The simulator only
#: needs it to be "not None"; the item itself is not the live one.
_BUSY_PLACEHOLDER: str = "warrior"

#: diagnostics["policy_kind"] value for the MCTS policy (distinct from greedy=0.0).
_POLICY_KIND_MCTS: float = 1.0


def _busy_city_ids(converted_state: Any) -> set[tuple[Any, Any]]:
    """``(owner id, city id)`` of the cities the raw state shows as building."""
    try:
        cities = list(converted_state["_raw_state"].cities)
    except Exception:  # no raw state to read: no city is known to be busy
        return set()
    busy: set[tuple[Any, Any]] = set()
    for city in cities:
        current = getattr(city, "current_production", None)
        if isinstance(current, str) and current:
            busy.add((getattr(city, "owner_id", None), getattr(city, "id", None)))
    return busy


class BridgeMCTSEnv(EnvironmentProtocol):
    """Snapshot-seeded MicroCivEnv adapter for GumbelMCTS.

    Wraps a MicroCivEnv initialized directly from a Bridge converted_state.
    Implements EnvironmentProtocol but forbids reset() since it represents
    a specific point in an ongoing game.

    Units and tiles go in through :mod:`alpha_zero.bridge.microciv_proxy`: a
    unit becomes one of MicroCiv's unit types or is left out, a tile becomes
    one of its terrain types. ``proxy_stats`` counts what that did.

    Attributes:
        proxy_stats: ``units`` put in the simulator, ``units_not_represented``
            left out because MicroCiv has nothing that stands for them, and
            ``impassable_tiles`` seeded as the simulator's mountain.

    Args:
        converted_state: Output of ``StateConverter.convert()`` for the
            current game frame.
        max_turns: Maximum turns for the inner simulation.  Defaults to
            :data:`_DEFAULT_MAX_TURNS`.
        min_lookahead: Turns the simulation is always left before its turn
            limit. The simulator's clock is the live turn clamped to
            ``max_turns - min_lookahead``: a live game past the horizon the
            network was trained for is then a playable root, not a game the
            simulator ends on the first step.
    """

    def __init__(
        self,
        converted_state: dict[str, Any],
        max_turns: int = _DEFAULT_MAX_TURNS,
        min_lookahead: int = _DEFAULT_MIN_LOOKAHEAD,
    ) -> None:
        self._converted_state = converted_state
        self._root_mask: np.ndarray | None = None
        board_size = self._simulator_board_size(converted_state)

        # Determine number of players from the state.  Start with the minimum
        # and expand as we find units / cities belonging to higher player IDs.
        players: set[int] = {0, 1}
        for u in converted_state.get("units", []):
            players.add(u.get("player", 0))
        for c in converted_state.get("cities", []):
            players.add(c.get("player", 0))
        num_players = max(_MIN_CIV_PLAYERS, min(_MAX_CIV_PLAYERS, max(players) + 1))

        self.env = MicroCivEnv(board_size=board_size, max_turns=max_turns, num_players=num_players)

        live_turn = converted_state.get("turn", 0)
        live_turn = live_turn if isinstance(live_turn, int) and live_turn > 0 else 0
        latest = max(0, max_turns - max(0, min(min_lookahead, max_turns)))
        sim_turn = min(live_turn, latest)
        self.env.turn = sim_turn
        self.env.current_player = converted_state.get("current_player", 0)

        # Terrain. The simulator's rules test for its own terrain words, so a
        # tile goes in as one of them (a mountain, water or ice as the type it
        # refuses to enter); features and yields are carried as they came.
        self.env.terrain = {}
        impassable_tiles = 0
        for pos, t_dict in converted_state.get("terrain", {}).items():
            terrain_type = simulator_terrain(t_dict)
            if terrain_type == SIM_MOUNTAIN:
                impassable_tiles += 1
            self.env.terrain[pos] = Tile(
                terrain_type=terrain_type,
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

        # Units. Each becomes one of the simulator's unit types, or is left
        # out when it has none (a builder, a ship): a unit of a type the
        # simulator's rules do not know could neither found a city nor be
        # told from a warrior.
        raw_types = raw_unit_types(converted_state)
        units_not_represented = 0
        self.env.units = []
        for u_dict in converted_state.get("units", []):
            unit_class = simulator_unit_class(
                u_dict.get("type", _DEFAULT_UNIT_TYPE),
                raw_types.get((u_dict.get("_owner_id"), u_dict.get("_id"))),
            )
            if unit_class is None:
                units_not_represented += 1
                continue
            self.env.units.append(
                Unit(
                    id=u_dict.get("_id", 0),
                    player=self._clamp_player(u_dict.get("player", 0), max_player_index),
                    unit_type=unit_class,
                    position=u_dict.get("position", (0, 0)),
                    health=u_dict.get("health", _DEFAULT_UNIT_HEALTH),
                    strength=u_dict.get("strength", _DEFAULT_UNIT_STRENGTH),
                    moves_remaining=u_dict.get("moves_remaining", _DEFAULT_UNIT_MOVES),
                )
            )

        # Cities. One that is already building is seeded busy, so the
        # simulator offers no production choice the live game would not take.
        busy = _busy_city_ids(converted_state)
        busy_cities = 0
        self.env.cities = []
        for c_dict in converted_state.get("cities", []):
            is_busy = (c_dict.get("_owner_id"), c_dict.get("_id")) in busy
            busy_cities += is_busy
            self.env.cities.append(
                City(
                    id=c_dict.get("_id", 0),
                    player=self._clamp_player(c_dict.get("player", 0), max_player_index),
                    name=c_dict.get("name", _DEFAULT_CITY_NAME),
                    position=c_dict.get("position", (0, 0)),
                    population=c_dict.get("population", _DEFAULT_CITY_POPULATION),
                    defense=c_dict.get("defense", _DEFAULT_CITY_DEFENSE),
                    production=c_dict.get("production", _DEFAULT_CITY_PRODUCTION),
                    production_queue=_BUSY_PLACEHOLDER if is_busy else None,
                )
            )

        # Reset internal IDs to avoid collisions if we step
        if self.env.units:
            self.env._next_unit_id = max(u.id for u in self.env.units) + 1
        if self.env.cities:
            self.env._next_city_id = max(c.id for c in self.env.cities) + 1

        self.env.done = False

        # With nobody else in sight (the usual state of an opening) the
        # simulator sees one surviving player and would call the game won on
        # the first step; ``step`` does not let it end for that reason.
        me = self.env.current_player
        self._no_visible_enemy = not any(u.player != me for u in self.env.units) and not any(
            c.player != me for c in self.env.cities
        )

        self.proxy_stats: dict[str, int] = {
            "units": len(self.env.units),
            "units_not_represented": units_not_represented,
            "impassable_tiles": impassable_tiles,
            "busy_cities": busy_cities,
            "no_visible_enemy": int(self._no_visible_enemy),
            "sim_turn": sim_turn,
            "turn_clamped": int(sim_turn != live_turn),
        }
        if units_not_represented:
            logger.debug(
                "Search simulator: %d of %d units have no MicroCiv stand-in and are left out",
                units_not_represented,
                units_not_represented + len(self.env.units),
            )

    @staticmethod
    def _simulator_board_size(converted_state: dict[str, Any]) -> int:
        """The board the simulator is built on.

        Every position in a converted state is scaled onto the converter's
        board, which the converter names in ``_board_size``. ``_map_size`` is
        the game's own map: building the simulator at that size on a scaled
        map leaves it mostly cells that hold nothing and can be walked into.
        A hand-written converted state without ``_board_size`` keeps the old
        reading, the larger side of ``_map_size``.
        """
        board_size = converted_state.get("_board_size")
        if isinstance(board_size, int) and not isinstance(board_size, bool) and board_size > 0:
            return board_size
        map_size: tuple[int, int] = converted_state.get("_map_size", (16, 16))
        return max(map_size[0], map_size[1])

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
        min_lookahead: int = _DEFAULT_MIN_LOOKAHEAD,
    ) -> BridgeMCTSEnv:
        """Named constructor for clarity at call sites.

        Equivalent to ``BridgeMCTSEnv(converted_state, max_turns=max_turns)``
        but is the preferred form when the caller already has
        ``StateConverter.convert()`` output.
        """
        return cls(converted_state, max_turns=max_turns, min_lookahead=min_lookahead)

    # ------------------------------------------------------------------
    # What the translator needs (GM-B6a)
    # ------------------------------------------------------------------

    def own_unit_ids(self) -> list[int]:
        """Live ids of the acting player's simulator units, in simulator order.

        The simulator's ``unit_idx`` counts these; units the proxy left out
        are not among them, so the index is not a place in the live list.
        """
        me = self.env.current_player
        return [int(u.id) for u in self.env.units if u.player == me]

    def own_city_ids(self) -> list[int]:
        """Live ids of the acting player's simulator cities, in simulator order."""
        me = self.env.current_player
        return [int(c.id) for c in self.env.cities if c.player == me]

    def restrict_root(self, mask: np.ndarray) -> None:
        """Limit the legal actions of this position to ``mask`` until the first step.

        The search reads the root's legal actions from the observation, so
        restricting them here makes every root candidate one the mask allows
        without touching the search.
        """
        self._root_mask = np.asarray(mask) > 0

    def _masked(self, valid: np.ndarray) -> np.ndarray:
        if self._root_mask is None:
            return valid
        masked: np.ndarray = valid * self._root_mask.astype(valid.dtype)
        return masked

    # ------------------------------------------------------------------
    # EnvironmentProtocol implementation
    # ------------------------------------------------------------------

    def clone(self) -> BridgeMCTSEnv:
        """Deep-copy this env for MCTS tree expansion.

        The cloned instance shares *no* mutable state with the original. It
        copies the simulator as it stands and does not seed again from the
        converted state: this runs once per simulation.
        """
        new_env = object.__new__(BridgeMCTSEnv)
        new_env._converted_state = self._converted_state
        new_env.env = self.env.clone()
        new_env.proxy_stats = dict(self.proxy_stats)
        new_env._root_mask = self._root_mask
        new_env._no_visible_enemy = self._no_visible_enemy
        return new_env

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Not supported — raises immediately.

        BridgeMCTSEnv represents a live game snapshot and may not be reset.
        Use :meth:`from_bridge_state` to create a fresh env from a new snapshot.
        Signature matches :class:`~alpha_zero.core.protocols.EnvironmentProtocol`.
        """
        raise NotImplementedError(
            "BridgeMCTSEnv does not support reset(). "
            "It represents a live game snapshot.  "
            "Call BridgeMCTSEnv.from_bridge_state(new_state) instead."
        )

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._root_mask = None  # the restriction is on the root position only
        was_done = self.env.done
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        if (
            self._no_visible_enemy
            and terminated
            and not was_done
            and not truncated
            and self.env.turn < self.env.max_turns
        ):
            # Ended only because nobody else is in sight: not a win, and not a
            # loss for whoever is to move next. The simulator applied one or
            # the other by comparing its winner with the mover after the step.
            if self.env.winner == self.env.current_player:
                reward -= float(REWARDS.WIN_REWARD)
            elif self.env.winner is not None:
                reward += float(REWARDS.WIN_REWARD)
            self.env.done = False
            self.env.winner = None
            terminated = False
            obs, info = self.env.get_observation(), self.env.get_info()
        return obs, reward, terminated, truncated, info

    def get_observation(self) -> dict[str, np.ndarray]:
        obs = self.env.get_observation()
        if self._root_mask is not None and "action_mask" in obs:
            obs = dict(obs)
            obs["action_mask"] = self._masked(obs["action_mask"])
        return obs

    def get_info(self) -> dict[str, Any]:
        return self.env.get_info()

    def get_valid_actions(self) -> np.ndarray:
        return self._masked(self.env.get_valid_actions())

    @property
    def action_space(self) -> Any:
        """Delegate to the wrapped env (satisfies ``MCTSEnvironment``)."""
        return self.env.action_space

    @property
    def current_player(self) -> int:
        return self.env.current_player

    @property
    def turn(self) -> int:
        return self.env.turn


class Civ6MctsPolicy:
    """Async policy that runs GumbelMCTS over the Civ6 bridge state.

    The search runs in the MicroCiv simulator. ``network`` must take the
    simulator's observation (13 channels), and the action returned is an index
    in the simulator's action space, not in the live action space the bridge's
    ``ActionConverter`` decodes. Until a translator exists (GM-B6) this policy
    is not something ``BridgeAgentLoop`` can play a live game with.

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
            ``(action_index, diagnostics)`` where *action_index* is in the
            simulator's action space and *diagnostics* contains
            ``value_estimate``, ``policy_kind``, and ``mcts_sims``.

        Raises:
            ValueError: If *converted_state* is ``None``.
        """
        if converted_state is None:
            raise ValueError("Civ6MctsPolicy requires converted_state to seed the simulation env.")

        env = BridgeMCTSEnv.from_bridge_state(converted_state, max_turns=self.max_turns)

        # Run MCTS search in a thread so we don't block the async bridge heartbeat.
        # GumbelMCTS.search returns (policy_probs, root_value) — a probability
        # vector over the action space (one-hot for the selected action) and the
        # root value estimate, NOT (action, stats).
        policy_probs, root_value = await asyncio.to_thread(self.mcts.search, env)
        action_idx = int(np.argmax(policy_probs))

        logger.debug(
            "MCTS search complete: action=%d root_value=%.3f", action_idx, float(root_value)
        )

        diagnostics: dict[str, float] = {
            "value_estimate": float(root_value),
            "policy_kind": _POLICY_KIND_MCTS,
            "mcts_sims": float(self.config.num_simulations),
        }

        return action_idx, diagnostics

    async def select_command(
        self,
        converted_state: dict[str, Any],
        *,
        allowed: Callable[[dict[str, Any]], bool] | None = None,
        decision_seed: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Search, and return the choice as a live command with a record of the decision.

        The root is restricted to the simulator actions that translate into a
        command the game offered (and ``allowed`` does not rule out), so the
        search only weighs what can be played.

        Args:
            converted_state: Output of ``StateConverter.convert()``.
            allowed: Optional veto on commands (the turn's ledger).
            decision_seed: Seed for this decision's search; None keeps the
                policy's configured behaviour.

        Returns:
            ``(command, record)``. With nothing translatable the command is
            ``end_turn`` and ``record["fallback"]`` says why.
        """
        env = BridgeMCTSEnv.from_bridge_state(converted_state, max_turns=self.max_turns)
        translator = SearchTranslator(
            converted_state, env.own_unit_ids(), env.own_city_ids(), allowed=allowed
        )
        decode = env.env.action_space_handler.decode_action
        root_mask, commands, reasons = translator.root_mask(env.env.get_valid_actions(), decode)
        env.restrict_root(root_mask)

        record: dict[str, Any] = {
            "sim_turn": int(env.proxy_stats["sim_turn"]),
            "turn_clamped": int(env.proxy_stats["turn_clamped"]),
            "root_actions": sorted(commands),
            "untranslatable": dict(sorted(reasons.items())),
            "chosen_action": None,
            "chosen_sim": None,
            "command": dict(_END_TURN_COMMAND),
            "policy_mass": 0.0,
            "root_value": 0.0,
            "search": {"root_evals": 0, "sim_evals": 0},
            "proxy": {key: int(value) for key, value in env.proxy_stats.items()},
            "seed": decision_seed if decision_seed is not None else self.config.seed,
            "fallback": None,
        }
        if not commands:
            record["fallback"] = "nothing the search could choose translates to an offered command"
            return dict(_END_TURN_COMMAND), record

        mcts = self.mcts
        if decision_seed is not None:
            mcts = GumbelMCTS(
                network=self.network,
                config=replace(self.config, seed=int(decision_seed)),
                device=self.device,
            )
        policy_probs, root_value = await asyncio.to_thread(mcts.search, env)
        stats = mcts.last_search_stats
        record["root_value"] = float(root_value)
        record["search"] = {"root_evals": int(stats.root_evals), "sim_evals": int(stats.sim_evals)}

        probs = np.asarray(policy_probs, dtype=np.float64)
        best = max(
            commands, key=lambda index: (probs[index] if index < len(probs) else 0.0, -index)
        )
        mass = float(probs[best]) if best < len(probs) else 0.0
        if not mass > 0.0:
            record["fallback"] = "the search put no weight on a translatable action"
            return dict(_END_TURN_COMMAND), record

        action = decode(best)
        record["chosen_action"] = int(best)
        record["chosen_sim"] = {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in action.items()
        }
        record["command"] = dict(commands[best])
        record["policy_mass"] = mass
        return dict(commands[best]), record
