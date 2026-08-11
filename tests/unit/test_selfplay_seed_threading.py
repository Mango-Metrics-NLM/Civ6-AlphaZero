"""Reproducibility of self-play game generation.

Seeding the global RNGs is not enough to make self-play reproducible:
``MicroCivEnv`` holds a private ``numpy`` generator seeded at construction, and
``run_selfplay`` historically built its environments without passing a seed. The
game stream therefore varied run to run even under a fixed global seed — fine
for training, fatal for an A/B comparison meant to isolate the encoder as the
only difference between two arms.

These tests pin the optional ``seed`` argument: identical seeds must reproduce
identical games, different seeds must not, and omitting it must preserve the
historical behaviour so existing callers are unaffected.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch

from alpha_zero.cloud.gcp.selfplay_worker import run_selfplay
from alpha_zero.cloud.gcp.training_worker import create_model
from alpha_zero.envs.env_factory import create_game_env

_SELFPLAY_KWARGS: dict[str, Any] = {
    "num_games": 2,
    "mcts_sims": 2,
    "mcts_type": "gumbel",
    "board_size": 8,
    "max_turns": 6,
    "device": "cpu",
}


@pytest.fixture(scope="module")
def model() -> torch.nn.Module:
    """A small deterministic network shared by the seeding tests."""
    torch.manual_seed(0)
    return create_model(board_size=8, encoder_type="patch_embed", num_res_blocks=2, num_heads=4)


def _actions(trajectories: list[dict]) -> list[list[int]]:
    return [list(t.get("actions", [])) for t in trajectories]


def test_run_selfplay_accepts_a_seed() -> None:
    """The parameter exists and defaults to None, keeping old calls valid."""
    signature = inspect.signature(run_selfplay)

    assert "seed" in signature.parameters
    assert signature.parameters["seed"].default is None


def test_environment_factory_accepts_a_seed() -> None:
    """The seed has somewhere to go — the factory already supported it."""
    assert "seed" in inspect.signature(create_game_env).parameters


def test_environments_with_the_same_seed_match() -> None:
    """Two identically-seeded environments start from the same state."""
    import numpy as np

    first, _ = create_game_env(board_size=8, max_turns=6, seed=99).reset()
    second, _ = create_game_env(board_size=8, max_turns=6, seed=99).reset()

    assert np.array_equal(first["spatial"], second["spatial"])


def test_environments_with_different_seeds_differ() -> None:
    """Confirms seeding actually varies the board, so the test above has teeth."""
    import numpy as np

    first, _ = create_game_env(board_size=8, max_turns=6, seed=1).reset()
    second, _ = create_game_env(board_size=8, max_turns=6, seed=2).reset()

    assert not np.array_equal(first["spatial"], second["spatial"])


def _seed_everything(seed: int) -> None:
    """Seed every RNG self-play draws from.

    Both halves are required. The environment seed fixes board generation, and
    the global seeds fix MCTS action selection — which samples from Python's
    ``random`` and numpy's global state as well as torch. Seeding only torch
    leaves the games varying, which is precisely the trap this suite exists to
    document.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.mark.slow
def test_same_seed_reproduces_the_same_games(model: torch.nn.Module) -> None:
    """A seeded self-play run is reproducible end to end."""
    _seed_everything(7)
    first = run_selfplay(model=model, seed=4242, **_SELFPLAY_KWARGS)
    _seed_everything(7)
    second = run_selfplay(model=model, seed=4242, **_SELFPLAY_KWARGS)

    assert _actions(first) == _actions(second)


@pytest.mark.slow
def test_different_env_seeds_produce_different_games(model: torch.nn.Module) -> None:
    """The env seed genuinely varies the games, so the pinning test has teeth."""
    _seed_everything(7)
    first = run_selfplay(model=model, seed=1000, **_SELFPLAY_KWARGS)
    _seed_everything(7)
    second = run_selfplay(model=model, seed=2000, **_SELFPLAY_KWARGS)

    assert _actions(first) != _actions(second)


@pytest.mark.slow
def test_global_seeding_alone_is_insufficient(model: torch.nn.Module) -> None:
    """Documents *why* the env seed had to be threaded through.

    With the global RNGs fixed but no environment seed, the games still differ,
    because the environment owns a private generator seeded at construction.
    """
    _seed_everything(7)
    first = run_selfplay(model=model, **_SELFPLAY_KWARGS)
    _seed_everything(7)
    second = run_selfplay(model=model, **_SELFPLAY_KWARGS)

    assert _actions(first) != _actions(second), (
        "environments became deterministic without an explicit seed; "
        "if this is now intended, the seed-threading rationale needs revisiting"
    )


@pytest.mark.slow
def test_unseeded_calls_still_work(model: torch.nn.Module) -> None:
    """Omitting the seed preserves the historical code path."""
    trajectories = run_selfplay(model=model, **_SELFPLAY_KWARGS)

    assert len(trajectories) == _SELFPLAY_KWARGS["num_games"]


class TestArenaSeeding:
    """The arena needs the same treatment self-play got.

    ``run_head_to_head`` records the seed in ``arena_result.json``, so an
    unseeded arena would publish a reproducibility claim the artifact cannot
    honour — the environment holds its own generator.
    """

    def test_run_arena_accepts_a_seed(self) -> None:
        from alpha_zero.cloud.gcp.arena import run_arena

        parameters = inspect.signature(run_arena).parameters

        assert "seed" in parameters
        assert parameters["seed"].default is None

    def test_seed_reaches_environment_construction(self) -> None:
        """Each game gets a distinct, derived seed."""
        from unittest.mock import MagicMock, patch

        from alpha_zero.cloud.gcp.arena import run_arena

        with (
            patch("alpha_zero.envs.env_factory.create_game_env") as factory,
            patch("alpha_zero.core.mcts_factory.create_mcts") as mcts_factory,
        ):
            env = MagicMock()
            env.current_player = 0
            env.winner = None
            env.reset.return_value = (None, None)
            env.step.return_value = (None, 0.0, True, False, {})
            factory.return_value = env

            mcts = MagicMock()
            mcts.search.return_value = ([0.5, 0.5], 0.0)
            mcts.select_action.return_value = 0
            mcts_factory.return_value = mcts

            run_arena(
                model_a=MagicMock(spec=torch.nn.Module),
                model_b=MagicMock(spec=torch.nn.Module),
                num_games=3,
                board_size=6,
                max_turns=4,
                seed=500,
            )

        seeds = [call.kwargs["seed"] for call in factory.call_args_list]
        assert seeds == [500, 501, 502], f"expected derived per-game seeds, got {seeds}"

    def test_unseeded_arena_leaves_environments_unseeded(self) -> None:
        """The default preserves the historical behaviour."""
        from unittest.mock import MagicMock, patch

        from alpha_zero.cloud.gcp.arena import run_arena

        with (
            patch("alpha_zero.envs.env_factory.create_game_env") as factory,
            patch("alpha_zero.core.mcts_factory.create_mcts") as mcts_factory,
        ):
            env = MagicMock()
            env.current_player = 0
            env.winner = None
            env.reset.return_value = (None, None)
            env.step.return_value = (None, 0.0, True, False, {})
            factory.return_value = env

            mcts = MagicMock()
            mcts.search.return_value = ([0.5, 0.5], 0.0)
            mcts.select_action.return_value = 0
            mcts_factory.return_value = mcts

            run_arena(
                model_a=MagicMock(spec=torch.nn.Module),
                model_b=MagicMock(spec=torch.nn.Module),
                num_games=2,
                board_size=6,
                max_turns=4,
            )

        assert all(call.kwargs["seed"] is None for call in factory.call_args_list)
