"""Guards for the sharing that makes ``MicroCivEnv.clone()`` cheap.

``clone()`` is called once per MCTS simulation, so its cost is a large share of
search wall time. It used to ``deepcopy`` the terrain dict, which dominated:
measured 1,123 us per clone at board 8 and 3,207 us at board 16, against 357 us
and 391 us after the terrain reference is shared.

Sharing is only safe while terrain is genuinely immutable after generation.
These tests make that an enforced invariant rather than a comment: if terrain
ever becomes mutable, ``test_terrain_is_never_mutated_during_play`` fails and
``clone()`` must go back to copying it.

Everything that *is* mutable -- units, cities, visited tiles, the RNG -- must
stay isolated between parent and clone, which the rest of these tests check.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alpha_zero.envs.microciv import MicroCivEnv

BOARD_SIZE = 6
MAX_TURNS = 12
SEED = 4242


def _terrain_fingerprint(env: MicroCivEnv) -> tuple:
    """A hashable snapshot of EVERY tile field, so any mutation shows up.

    Deliberately reflective rather than a hand-picked list of attributes. An
    earlier version snapshotted only ``terrain_type`` and ``yields`` and was
    therefore blind to ``Tile.features`` -- the field most likely to become
    mutable (adding a forest or an improvement). A guard that names its fields
    protects only the fields someone remembered.
    """

    def snapshot(tile: object) -> tuple:
        fields = dataclasses.fields(tile) if dataclasses.is_dataclass(tile) else ()
        out: list[tuple[str, str]] = []
        for field in fields:
            value = getattr(tile, field.name)
            # repr, not the value: dicts and lists are unhashable, and repr is
            # sensitive to any change in content or order.
            out.append((field.name, repr(value)))
        return tuple(out)

    return tuple(sorted((pos, snapshot(tile)) for pos, tile in env.terrain.items()))


@pytest.fixture
def env() -> MicroCivEnv:
    environment = MicroCivEnv(board_size=BOARD_SIZE, max_turns=MAX_TURNS, seed=SEED)
    environment.reset(seed=SEED)
    return environment


def test_terrain_is_never_mutated_during_play(env: MicroCivEnv) -> None:
    """The invariant `clone()` relies on: terrain is read-only after generation."""
    before = _terrain_fingerprint(env)
    rng = np.random.default_rng(SEED)

    steps = 0
    while not env.done and steps < 150:
        mask = env.get_valid_actions()
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            break
        env.step(int(rng.choice(valid)))
        steps += 1

    assert steps > 0, "no actions were taken; the comparison would be vacuous"
    assert _terrain_fingerprint(env) == before, (
        "terrain changed during play, so MicroCivEnv.clone() sharing the terrain "
        "reference is no longer safe -- restore the deepcopy there"
    )


def test_clone_shares_terrain(env: MicroCivEnv) -> None:
    """Sharing is the optimisation; assert it explicitly so it cannot regress silently."""
    twin = env.clone()
    assert twin.terrain is env.terrain


def test_clone_isolates_units(env: MicroCivEnv) -> None:
    twin = env.clone()
    assert twin.units is not env.units
    if twin.units:
        twin.units[0].health = -999
        assert env.units[0].health != -999


def test_clone_isolates_cities(env: MicroCivEnv) -> None:
    twin = env.clone()
    assert twin.cities is not env.cities


def test_stepping_a_clone_does_not_change_the_parent(env: MicroCivEnv) -> None:
    """The property that matters for search: a simulation cannot touch the real game."""
    before_turn = env.turn
    before_player = env.current_player
    before_units = [(unit.id, unit.position, unit.health) for unit in env.units]

    twin = env.clone()
    rng = np.random.default_rng(SEED + 1)
    for _ in range(25):
        if twin.done:
            break
        mask = twin.get_valid_actions()
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            break
        twin.step(int(rng.choice(valid)))

    assert env.turn == before_turn
    assert env.current_player == before_player
    assert [(u.id, u.position, u.health) for u in env.units] == before_units


def test_clone_preserves_board_configuration(env: MicroCivEnv) -> None:
    twin = env.clone()
    assert twin.board_size == env.board_size
    assert twin.max_turns == env.max_turns
    assert twin.num_players == env.num_players
    assert twin.turn == env.turn
    assert twin.current_player == env.current_player


def test_the_terrain_guard_detects_a_features_mutation(env: MicroCivEnv) -> None:
    """Negative control for the guard itself.

    ``clone()`` shares the terrain dict, which is only sound while tiles are
    immutable. If this test cannot see a mutation, the invariant test above is
    decoration.
    """
    before = _terrain_fingerprint(env)
    tile = next(iter(env.terrain.values()))
    if not dataclasses.is_dataclass(tile):  # pragma: no cover - shape guard
        pytest.skip("Tile is not a dataclass; the reflective fingerprint does not apply")

    mutated_any = False
    for field in dataclasses.fields(tile):
        value = getattr(tile, field.name)
        if isinstance(value, list):
            value.append("__probe__")
            mutated_any = True
            break
        if isinstance(value, dict):
            value["__probe__"] = 1
            mutated_any = True
            break
    if not mutated_any:  # pragma: no cover - depends on the Tile shape
        pytest.skip("no mutable Tile field to probe")

    assert _terrain_fingerprint(env) != before, (
        "the terrain fingerprint did not notice a mutated tile field, so "
        "test_terrain_is_never_mutated_during_play cannot protect the sharing "
        "that MicroCivEnv.clone() relies on"
    )
