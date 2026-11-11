"""The move action mask must agree with the executor (LF-B1).

``MicroCivActionSpace.get_valid_mask`` marked a move legal when the target was
on the board and not a mountain; ``MicroCivEnv._is_valid_move`` also rejects any
tile occupied by a unit (own or enemy), and a rejected move zeroes the unit's
``moves_remaining``. Masked policy loss and MCTS treat the mask as ground truth,
so every disagreement was a wasted action and a lost unit-turn.

The oracle here is the environment's own signal: ``info["action_success"]``.
Seeded random play that samples only mask-legal actions must never see a failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.envs.microciv import MicroCivEnv

BOARD_SIZE = 8


def _play(
    game_seed: int, board_size: int = BOARD_SIZE, max_turns: int | None = None
) -> tuple[int, int, int]:
    """Play one seeded random game on mask-legal actions only.

    Returns (steps, rejections, invalid_move_rejections).
    """
    kwargs = {} if max_turns is None else {"max_turns": max_turns}
    env = MicroCivEnv(board_size=board_size, seed=game_seed, **kwargs)
    env.reset(seed=game_seed)
    rng = np.random.default_rng(game_seed)
    steps = rejections = invalid_moves = 0
    done = False
    while not done:
        legal = np.flatnonzero(env.get_valid_actions() > 0)
        action = int(rng.choice(legal))
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        if not info["action_success"]:
            rejections += 1
            if "Invalid move" in info["action_error"]:
                invalid_moves += 1
        done = terminated or truncated
    return steps, rejections, invalid_moves


SHIPPED_MAX_TURNS = 30  # the training/evaluation default (board 8 / 30 turns)


def _total_rejections(num_games: int, max_turns: int | None) -> tuple[int, int, int]:
    steps = rejected = invalid = 0
    for seed in range(num_games):
        s, r, i = _play(seed, max_turns=max_turns)
        steps += s
        rejected += r
        invalid += i
    return steps, rejected, invalid


def _assert_clean(num_games: int, max_turns: int | None) -> None:
    steps, rejected, invalid = _total_rejections(num_games, max_turns)
    assert steps > 0
    assert invalid == 0, f"{invalid} 'Invalid move' rejections in {steps} steps"
    assert rejected == 0, f"{rejected} rejected actions of any kind in {steps} steps"


def test_no_invalid_move_in_20_seeded_random_games() -> None:
    """Long games (env default max_turns=200): dense in late-game collisions."""
    _assert_clean(20, None)


def test_no_invalid_move_in_200_seeded_random_games() -> None:
    """200 games at the shipped max_turns=30; fast enough for CI's coverage job.

    The old mask gave ~94 rejections per 20 such games, so 200 keep ample power.
    """
    _assert_clean(200, SHIPPED_MAX_TURNS)


def _occupied_target_env(occupant_is_enemy: bool) -> tuple[MicroCivEnv, int]:
    """Env where the current player's mover has an occupied tile to its east."""
    env = MicroCivEnv(board_size=BOARD_SIZE, seed=1)
    env.reset(seed=1)
    player = env.current_player
    mover = next(u for u in env.units if u.player == player)
    if occupant_is_enemy:
        other = next(u for u in env.units if u.player != player)
    else:
        # Starts hold one unit per player: add a second own unit as the blocker.
        other = type(mover)(
            id=9999,
            player=player,
            unit_type="warrior",
            position=(0, 0),
            strength=mover.strength,
        )
        env.units.append(other)
    src = dst = None
    for (x, y), tile in env.terrain.items():
        east = (x + 1, y)
        if east in env.terrain and "mountain" not in (
            tile.terrain_type,
            env.terrain[east].terrain_type,
        ):
            src, dst = (x, y), east
            break
    assert src is not None and dst is not None
    for u in env.units:
        if u is not mover and u is not other:
            u.position = (-5, -5)  # off board: cannot interfere
    mover.position, other.position = src, dst
    mover.moves_remaining = 1
    own_idx = env._get_player_units(player).index(mover)
    handler = env.action_space_handler
    action = handler.move_offset + own_idx * 8 + handler.directions.index((1, 0))
    return env, action


@pytest.mark.parametrize("occupant_is_enemy", [False, True])
def test_move_onto_occupied_tile_is_masked(occupant_is_enemy: bool) -> None:
    """Own and enemy units both block a move; the mask must say so."""
    env, action = _occupied_target_env(occupant_is_enemy)
    assert env.get_valid_actions()[action] == 0.0
    _, _, _, _, info = env.clone().step(action)
    assert not info["action_success"]  # the executor agrees it is illegal


def test_mask_matches_executor_for_every_move_action() -> None:
    """Exhaustive agreement on move actions at states along a seeded game."""
    env = MicroCivEnv(board_size=BOARD_SIZE, seed=3)
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    handler = env.action_space_handler
    for _ in range(60):
        mask = env.get_valid_actions()
        for action in range(handler.move_offset, handler.attack_offset):
            # Ask the executor directly (no observation encoding) on a throwaway copy.
            _, success, _ = env.clone()._execute_action(handler.decode_action(action))
            assert bool(mask[action]) == success, (
                f"action {action}: mask={mask[action]} executor={success}"
            )
        _, _, term, trunc, _ = env.step(int(rng.choice(np.flatnonzero(mask > 0))))
        if term or trunc:
            break


def _bridge_state() -> dict:
    """Converted-state dict as ``StateConverter.convert`` emits it (cheap to build)."""
    terrain = {
        (x, y): {"type": "plains", "features": [], "yields": {}} for x in range(4) for y in range(4)
    }
    terrain[(2, 0)]["type"] = "mountain"
    units = [
        {"_id": 1, "player": 0, "type": "warrior", "position": (0, 0), "moves_remaining": 2},
        {"_id": 2, "player": 0, "type": "warrior", "position": (1, 0), "moves_remaining": 2},
        {"_id": 3, "player": 1, "type": "warrior", "position": (1, 1), "moves_remaining": 2},
    ]
    return {
        "turn": 1,
        "current_player": 0,
        "_map_size": (4, 4),
        "terrain": terrain,
        "units": units,
        "cities": [],
    }


def _walk_and_compare(env: object, steps: int, seed: int) -> None:
    """Step ``env`` along a seeded trajectory, comparing every mask source each state."""
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        mask = env.get_valid_actions()  # type: ignore[attr-defined]
        assert np.array_equal(env.get_observation()["action_mask"], mask)  # type: ignore[attr-defined]
        _, _, term, trunc, _ = env.step(int(rng.choice(np.flatnonzero(mask > 0))))  # type: ignore[attr-defined]
        if term or trunc:
            break


def test_all_mask_builders_agree() -> None:
    """Every env that exposes a move mask agrees with ``get_valid_actions`` along whole games.

    All paths share ``MicroCivActionSpace.get_valid_mask``; this fails if one ever grows
    its own builder (for example a fogged mask) that drifts from the others.
    """
    from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv
    from alpha_zero.envs.microciv_enhanced import EnhancedMicroCivEnv
    from alpha_zero.envs.vectorized_env import VectorizedEnvConfig, VectorizedMicroCivEnv

    base = MicroCivEnv(board_size=BOARD_SIZE, seed=5, max_turns=SHIPPED_MAX_TURNS)
    base.reset(seed=5)
    _walk_and_compare(base, 150, 5)

    enhanced = EnhancedMicroCivEnv(board_size=BOARD_SIZE, seed=6, max_turns=SHIPPED_MAX_TURNS)
    enhanced.reset(seed=6)
    _walk_and_compare(enhanced, 150, 6)

    _walk_and_compare(BridgeMCTSEnv.from_bridge_state(_bridge_state(), max_turns=40), 100, 7)

    vec = VectorizedMicroCivEnv(
        VectorizedEnvConfig(
            num_envs=3, board_size=BOARD_SIZE, max_turns=SHIPPED_MAX_TURNS, seed_offset=7
        )
    )
    vec.reset_all()
    rng = np.random.default_rng(8)
    for _ in range(60):
        _, _, masks = vec.get_observations()
        for i, sub in enumerate(vec._envs):
            assert np.array_equal(masks[i], sub.get_valid_actions())
        actions = np.array([int(rng.choice(np.flatnonzero(m > 0))) for m in masks], dtype=np.int64)
        vec.step_all(actions)
        vec.reset_done()
