"""A seeded arena is a reproducible arena -- searches included, seats paired.

Before this, ``run_arena(seed=N)`` seeded only the environments: both searches
drew from unseeded generators, so two "seeded" arenas could disagree on every
move, and the recorded ``seed`` in an A/B artifact promised a reproducibility
the artifact did not have. Game ``i`` was also seeded ``seed + i``, so each map
was played from one seat only and a map or first-mover advantage landed on
whichever side drew it.

Now: the map for pair ``p`` is ``seed + p`` and is played from both seats; each
side's search for game ``i`` draws from
``derive_seed(seed, SeedStream.ARENA_SEARCH, i, side)``; a baseline player
supplied through a factory gets ``derive_seed(seed, ARENA_OPPONENT, i, side)``;
``seed=None`` leaves everything unseeded exactly as before.

Negative controls performed when this file was written (each reverted):

* dropping the ``seed`` kwarg from ``create_mcts`` in ``_search_player`` ->
  ``test_same_seed_replays_the_same_moves`` fails (traces differ between two
  runs at one seed);
* restoring ``seed + game_idx`` for the map -> ``test_maps_are_paired`` fails.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from alpha_zero.cloud.gcp.arena import ArenaResult, run_arena
from alpha_zero.utils.seeding import SeedStream, derive_seed

# Tiny on purpose: the property under test is reproducibility, not strength.
_BOARD = 6
_TURNS = 4
_SIMS = 2
_GAMES = 4
_SEED = 1234


def _dummy_model() -> torch.nn.Module:
    # mcts_type="simple" never touches the network; any Module satisfies the type.
    return torch.nn.Linear(1, 1)


def _tiny_network() -> torch.nn.Module:
    """A real (untrained) network so the traced games use the stochastic Gumbel search.

    SimpleMCTS would not do here: at temperature 0 its near-tied visit counts
    resolve to the lowest index (END_TURN) in most positions, so its trace is
    nearly seed-independent and a replay test on it proves little. Gumbel draws
    noise on every search, so an unseeded search genuinely diverges.
    """
    from alpha_zero.core.civ6_network import MicroCivNetwork
    from alpha_zero.envs.microciv import MicroCivEnv

    torch.manual_seed(0)
    return MicroCivNetwork(
        board_size=_BOARD,
        action_space_size=int(MicroCivEnv(board_size=_BOARD).action_space.n),
        num_res_blocks=1,
        num_filters=8,
    ).eval()


def _traced_arena(seed: int | None, **overrides: Any) -> tuple[ArenaResult, list[list[int]]]:
    """Run a real tiny arena and capture the action sequence of every game."""
    from alpha_zero.envs import env_factory

    traces: list[list[int]] = []
    real_create = env_factory.create_game_env

    def tracing_create(**kwargs: Any) -> Any:
        env = real_create(**kwargs)
        trace: list[int] = []
        traces.append(trace)
        original_step = env.step

        def step(action: int) -> Any:
            trace.append(int(action))
            return original_step(action)

        env.step = step
        return env

    kwargs: dict[str, Any] = {
        "num_games": _GAMES,
        "mcts_type": "gumbel",
        "mcts_sims": _SIMS,
        "gumbel_num_actions": 4,
        "board_size": _BOARD,
        "max_turns": _TURNS,
        "seed": seed,
    }
    kwargs.update(overrides)
    model = _dummy_model() if kwargs["mcts_type"] == "simple" else _tiny_network()
    with patch("alpha_zero.envs.env_factory.create_game_env", side_effect=tracing_create):
        result = run_arena(model, model, **kwargs)
    return result, traces


class TestReproducibility:
    def test_same_seed_replays_the_same_moves(self) -> None:
        first, first_traces = _traced_arena(_SEED)
        second, second_traces = _traced_arena(_SEED)
        assert first.to_dict() == second.to_dict()
        assert first_traces == second_traces
        # The trace must be non-trivial, or the equality above proves nothing.
        assert sum(len(t) for t in first_traces) > _GAMES

    def test_different_seeds_play_different_games(self) -> None:
        """Two random players, both seeded through the arena's factories.

        Not SimpleMCTS: at temperature 0 its near-tied visit counts resolve to
        the lowest index (END_TURN) in most positions, so its trace barely
        depends on the seed and the two runs would coincide for the wrong
        reason. Random players make the seed the only source of difference.
        """
        from alpha_zero.evaluation.baseline_agents import RandomAgent

        def factory(seed: int | None) -> RandomAgent:
            return RandomAgent(seed=seed)

        _, first_traces = _traced_arena(_SEED, player_a_factory=factory, player_b_factory=factory)
        _, second_traces = _traced_arena(
            _SEED + 1, player_a_factory=factory, player_b_factory=factory
        )
        assert first_traces != second_traces

    def test_same_seed_replays_the_same_random_players(self) -> None:
        from alpha_zero.evaluation.baseline_agents import RandomAgent

        def factory(seed: int | None) -> RandomAgent:
            return RandomAgent(seed=seed)

        first, first_traces = _traced_arena(
            _SEED, player_a_factory=factory, player_b_factory=factory
        )
        second, second_traces = _traced_arena(
            _SEED, player_a_factory=factory, player_b_factory=factory
        )
        assert first.to_dict() == second.to_dict()
        assert first_traces == second_traces

    def test_unseeded_arena_passes_no_seed_to_the_search(self) -> None:
        """``seed=None`` must build exactly the search it always did."""
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
            mcts.search.return_value = (np.array([0.5, 0.5]), 0.0)
            mcts.select_action.return_value = 0
            mcts_factory.return_value = mcts

            run_arena(_dummy_model(), _dummy_model(), num_games=2, board_size=6, max_turns=4)

        assert all("seed" not in call.kwargs for call in mcts_factory.call_args_list)
        assert all(call.kwargs["seed"] is None for call in factory.call_args_list)


class TestSeedDerivation:
    def test_search_seeds_are_derived_per_game_and_side(self) -> None:
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
            mcts.search.return_value = (np.array([0.5, 0.5]), 0.0)
            mcts.select_action.return_value = 0
            mcts_factory.return_value = mcts

            run_arena(
                _dummy_model(), _dummy_model(), num_games=3, board_size=6, max_turns=4, seed=_SEED
            )

        seeds = [call.kwargs["seed"] for call in mcts_factory.call_args_list]
        expected = [
            derive_seed(_SEED, SeedStream.ARENA_SEARCH, game, side)
            for game in range(3)
            for side in ("a", "b")
        ]
        assert seeds == expected
        assert len(set(seeds)) == len(seeds), "every (game, side) must draw its own stream"

    def test_maps_are_paired(self) -> None:
        """Games 2p and 2p+1 share map ``seed + p``; seats alternate."""
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
            mcts.search.return_value = (np.array([0.5, 0.5]), 0.0)
            mcts.select_action.return_value = 0
            mcts_factory.return_value = mcts

            run_arena(
                _dummy_model(), _dummy_model(), num_games=6, board_size=6, max_turns=4, seed=100
            )

        assert [c.kwargs["seed"] for c in factory.call_args_list] == [100, 100, 101, 101, 102, 102]

    def test_explicit_env_seeds_override_the_arithmetic(self) -> None:
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
            mcts.search.return_value = (np.array([0.5, 0.5]), 0.0)
            mcts.select_action.return_value = 0
            mcts_factory.return_value = mcts

            run_arena(
                _dummy_model(),
                _dummy_model(),
                num_games=4,
                board_size=6,
                max_turns=4,
                seed=5,
                env_seeds=[77, 88],
            )

        assert [c.kwargs["seed"] for c in factory.call_args_list] == [77, 77, 88, 88]

    def test_too_few_env_seeds_is_refused(self) -> None:
        with pytest.raises(ValueError, match="env_seeds"):
            run_arena(_dummy_model(), _dummy_model(), num_games=4, env_seeds=[1])


class TestPlayerFactories:
    def test_baseline_opponent_receives_a_derived_seed_and_plays(self) -> None:
        from alpha_zero.evaluation.baseline_agents import RandomAgent

        received: list[int | None] = []

        def factory(seed: int | None) -> RandomAgent:
            received.append(seed)
            return RandomAgent(seed=seed)

        result = run_arena(
            _dummy_model(),
            None,
            num_games=_GAMES,
            mcts_type="simple",
            mcts_sims=_SIMS,
            board_size=_BOARD,
            max_turns=_TURNS,
            seed=_SEED,
            player_b_factory=factory,
        )
        assert result.total_games == _GAMES
        assert result.wins + result.draws + result.losses == _GAMES
        assert received == [
            derive_seed(_SEED, SeedStream.ARENA_OPPONENT, game, "b") for game in range(_GAMES)
        ]

    def test_missing_model_and_factory_is_refused(self) -> None:
        with pytest.raises(ValueError, match="model_b or player_b_factory"):
            run_arena(_dummy_model(), None, num_games=2)
        with pytest.raises(ValueError, match="model_a or player_a_factory"):
            run_arena(None, _dummy_model(), num_games=2)

    def test_simple_variant_runs_through_the_arena(self) -> None:
        """``SimpleMCTS.search`` lacked ``add_noise``; the arena passes it."""
        result, traces = _traced_arena(_SEED, num_games=2, mcts_type="simple")
        assert result.total_games == 2
        assert all(traces)
