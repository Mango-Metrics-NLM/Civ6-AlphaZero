"""Roots, arms and measurement for the three-arm asymmetry probe (zero games).

Arms are built by patching the ``BridgeMCTSEnv`` class itself (never a subclass: ``clone``
builds ``object.__new__(BridgeMCTSEnv)`` and ``select_command`` builds the base class):

* ``a`` merged rule: no patch.
* ``b`` raw simulator rule: ``step`` is the simulator's own.
* ``c`` symmetric rule: ``__init__`` also sets ``_own_left_out`` to at least 1, so the merged
  ``step`` takes back the root player's loss of every simulator piece before the turn limit.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv, Civ6MctsPolicy
from alpha_zero.bridge.protocol import GameStateData, LegalAction
from alpha_zero.core.civ6_mcts import GumbelMCTSConfig
from alpha_zero.core.civ6_network import MicroCivNetwork
from tests.fixtures.bridge_asymmetry_stats import hex_distance, hex_neighbours
from tests.fixtures.bridge_search_states import city, make_state, unit
from tests.fixtures.bridge_search_translation import convert, is_offered

ARMS = ("a", "b", "c")
POSITIONS = [(5, 5), (8, 3), (3, 9)]
FAMILIES = ("R1", "R2", "R3", "R4", "R5")
DECISION_SEEDS = list(range(1, 11))
INIT_SEEDS = [0, 1, 2]
OWN_CITY = (13, 13)
#: The contract asks for 8; R3 (all six neighbour moves) translates to 10 (8 simulator
#: directions + end turn + skip), so the bound is 10 and each root's count is recorded.
MAX_ROOT_ACTIONS = 10
ENEMY_CITY = (14, 14)  # R5: at least 4 from every position


@dataclass(frozen=True)
class Root:
    family: str
    pos: tuple[int, int]
    own_city: bool

    @property
    def key(self) -> str:
        return f"{self.family}@{self.pos[0]},{self.pos[1]}{'+city' if self.own_city else ''}"


def all_roots() -> list[Root]:
    roots = [Root(f, p, c) for f in FAMILIES[:4] for p in POSITIONS for c in (False, True)]
    return roots + [Root("R5", p, False) for p in POSITIONS]


def build_state(root: Root) -> GameStateData:
    """The root's real ``GameStateData`` (two players: us and one barbarian side, id 9)."""
    x, y = root.pos
    legal = [LegalAction("end_turn"), LegalAction("skip", unit_id=2)]
    three = [(x - 1, y), (x, y + 1), (x, y - 1)]
    enemies = [unit(3, "UNIT_ARCHER", 9, x + 1, y, health=1)]
    if root.family in ("R1", "R5"):
        legal.append(LegalAction("attack", unit_id=2, targets=[(x + 1, y)]))
    elif root.family == "R2":
        legal.append(LegalAction("attack", unit_id=2, targets=[(x + 1, y)]))
        legal.append(LegalAction("move", unit_id=2, targets=three))
    elif root.family == "R3":
        enemies = [unit(3, "UNIT_ARCHER", 9, x + 2, y)]
        legal.append(LegalAction("move", unit_id=2, targets=hex_neighbours(x, y)))
    elif root.family == "R4":
        enemies = [unit(3, "UNIT_WARRIOR", 9, x + 1, y)]
        legal.append(LegalAction("attack", unit_id=2, targets=[(x + 1, y)]))
        legal.append(LegalAction("move", unit_id=2, targets=three))
    cities = []
    if root.own_city:
        cities.append(city(10, 0, *OWN_CITY))
    if root.family == "R5":
        cities.append(city(11, 9, *ENEMY_CITY))
    state = make_state(
        size=16,
        units=[unit(2, "UNIT_WARRIOR", 0, x, y, health=100), *enemies],
        cities=cities,
        legal_actions=legal,
    )
    state.observes_others = True
    state.other_players = [{"id": 9, "kind": "barbarian"}]
    return state


def enemy_tiles(state: GameStateData) -> list[tuple[int, int]]:
    """Visible hostile units and cities (owner 9) in the state."""
    tiles = [(u.x, u.y) for u in state.units if u.owner_id == 9]
    return tiles + [(c.x, c.y) for c in state.cities if c.owner_id == 9]


# --- arms -------------------------------------------------------------------------------


def raw_step(self: BridgeMCTSEnv, action: int) -> Any:
    """``BridgeMCTSEnv.step`` without the GM-B8b rules: the simulator's own verdict."""
    self._root_mask = None
    return self.env.step(action)


@contextmanager
def arm_patched(arm: str, tally: dict[str, int] | None = None) -> Iterator[None]:
    """Apply an arm (and the optional in-tree counting wrapper) for the duration."""
    assert arm in ARMS
    original_init = BridgeMCTSEnv.__init__
    with pytest.MonkeyPatch.context() as mp:
        if arm == "b":
            mp.setattr(BridgeMCTSEnv, "step", raw_step)
        if arm == "c":

            def init_c(self: BridgeMCTSEnv, *args: Any, **kwargs: Any) -> None:
                original_init(self, *args, **kwargs)
                self._own_left_out = max(1, self._own_left_out)

            mp.setattr(BridgeMCTSEnv, "__init__", init_c)
        if tally is not None:
            in_force = BridgeMCTSEnv.step if arm != "b" else raw_step

            def counting(self: BridgeMCTSEnv, action: int) -> Any:
                me = self._root_player
                enemy_before = any(u.player != me for u in self.env.units) or any(
                    c.player != me for c in self.env.cities
                )
                mine_before = self._has_pieces(me)
                result = in_force(self, action)
                enemy_after = any(u.player != me for u in self.env.units) or any(
                    c.player != me for c in self.env.cities
                )
                if enemy_before and not enemy_after:
                    tally["enemy_eliminated"] += 1
                if mine_before and not self._has_pieces(me):
                    tally["own_eliminated"] += 1
                return result

            mp.setattr(BridgeMCTSEnv, "step", counting)
        yield


# --- measurement ------------------------------------------------------------------------


def command_kind(command: dict[str, Any]) -> str:
    return {"unit_attack": "attack", "unit_move": "move", "unit_skip": "skip"}.get(
        str(command["action_type"]), str(command["action_type"])
    )


def delta_d(root: Root, state: GameStateData, command: dict[str, Any]) -> int | None:
    """Hex distance to the nearest visible enemy after a move minus before it."""
    if command_kind(command) != "move":
        return None
    enemies = enemy_tiles(state)
    target = (command["target"]["x"], command["target"]["y"])
    assert hex_distance(root.pos, target) == 1, (root.key, target)
    return min(hex_distance(target, e) for e in enemies) - min(
        hex_distance(root.pos, e) for e in enemies
    )


def measure_chunk(
    arm: str, init: int, roots: list[Root] | None = None, seeds: list[int] | None = None
) -> list[dict[str, Any]]:
    """Search every root at every decision seed for one arm and one network init."""
    roots = all_roots() if roots is None else roots
    seeds = DECISION_SEEDS if seeds is None else seeds
    tally = {"enemy_eliminated": 0, "own_eliminated": 0}
    rows: list[dict[str, Any]] = []
    with arm_patched(arm, tally):
        torch.manual_seed(init)
        network = MicroCivNetwork(board_size=16, action_space_size=376)
        config = GumbelMCTSConfig(num_simulations=24, num_sampled_actions=8)
        policy = Civ6MctsPolicy(network, config, device="cpu", max_turns=30)
        for root in roots:
            state = build_state(root)
            converted = convert(state, 16)
            for seed in seeds:
                before = dict(tally)
                command, record = asyncio.run(policy.select_command(converted, decision_seed=seed))
                assert record["fallback"] is None, (root.key, seed, record["fallback"])
                assert is_offered(command, state.legal_actions), (root.key, seed, command)
                assert len(record["root_actions"]) <= MAX_ROOT_ACTIONS, (
                    root.key,
                    record["root_actions"],
                )
                dd = delta_d(root, state, command)
                kind = command_kind(command)
                contact = kind == "attack" or (dd is not None and dd < 0)
                rows.append(
                    {
                        "arm": arm,
                        "init": init,
                        "seed": seed,
                        "root": root.key,
                        "family": root.family,
                        "own_city": root.own_city,
                        "kind": kind,
                        "command": command,
                        "delta_d": dd,
                        "contact": bool(contact),
                        "retreat": dd is not None and dd > 0,
                        "root_value": float(record["root_value"]),
                        "n_actions": len(record["root_actions"]),
                        "elimination_suppressed": int(record["elimination_suppressed"]),
                        "enemy_eliminated": tally["enemy_eliminated"] - before["enemy_eliminated"],
                        "own_eliminated": tally["own_eliminated"] - before["own_eliminated"],
                        "record": record,
                    }
                )
    return rows


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start
