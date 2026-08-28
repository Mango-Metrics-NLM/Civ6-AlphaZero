"""Unit tests for the shared arena module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from alpha_zero.cloud.gcp.arena import ArenaResult, run_arena

# =============================================================================
# TestArenaResult - Pure dataclass tests
# =============================================================================


class TestArenaResult:
    """Tests for ArenaResult dataclass."""

    def test_win_rate_calculation(self) -> None:
        """Calculates win rate correctly."""
        result = ArenaResult(wins=3, losses=1, draws=1, total_games=5)
        assert result.win_rate == 0.6

    def test_draw_rate_calculation(self) -> None:
        """Calculates draw rate correctly."""
        result = ArenaResult(wins=2, losses=2, draws=1, total_games=5)
        assert result.draw_rate == 0.2

    def test_loss_rate_calculation(self) -> None:
        """Calculates loss rate correctly."""
        result = ArenaResult(wins=1, losses=3, draws=1, total_games=5)
        assert result.loss_rate == 0.6

    def test_rates_sum_to_one(self) -> None:
        """Win, draw, and loss rates sum to 1.0 for various combos."""
        test_cases = [
            (3, 1, 1, 5),
            (10, 5, 5, 20),
            (7, 0, 3, 10),
            (0, 0, 10, 10),
            (8, 2, 0, 10),
        ]

        for wins, losses, draws, total in test_cases:
            result = ArenaResult(wins=wins, losses=losses, draws=draws, total_games=total)
            total_rate = result.win_rate + result.draw_rate + result.loss_rate
            assert abs(total_rate - 1.0) < 1e-9, f"Rates don't sum to 1.0 for {result}"

    def test_zero_games_no_division_error(self) -> None:
        """With total_games=0, all rates return 0.0 without division error."""
        result = ArenaResult(wins=0, losses=0, draws=0, total_games=0)
        assert result.win_rate == 0.0
        assert result.draw_rate == 0.0
        assert result.loss_rate == 0.0

    def test_to_dict_keys(self) -> None:
        """to_dict returns all 7 expected keys."""
        result = ArenaResult(wins=5, losses=3, draws=2, total_games=10)
        result_dict = result.to_dict()

        expected_keys = {
            "win_rate",
            "draw_rate",
            "loss_rate",
            "wins",
            "draws",
            "losses",
            "total_games",
            "score_rate",
        }
        assert set(result_dict.keys()) == expected_keys

    def test_to_dict_values_match_properties(self) -> None:
        """to_dict values match properties and attributes."""
        result = ArenaResult(wins=7, losses=2, draws=1, total_games=10)
        result_dict = result.to_dict()

        assert result_dict["win_rate"] == result.win_rate
        assert result_dict["draw_rate"] == result.draw_rate
        assert result_dict["loss_rate"] == result.loss_rate
        assert result_dict["wins"] == result.wins
        assert result_dict["draws"] == result.draws
        assert result_dict["losses"] == result.losses
        assert result_dict["total_games"] == result.total_games
        assert result_dict["score_rate"] == result.score_rate


# =============================================================================
# TestRunArena - Tests using mocks
# =============================================================================


class TestRunArena:
    """Tests for run_arena function using mocks."""

    @patch("alpha_zero.core.mcts_factory.create_mcts")
    @patch("alpha_zero.envs.env_factory.create_game_env")
    def test_returns_arena_result_type(
        self, mock_create_env: MagicMock, mock_create_mcts: MagicMock
    ) -> None:
        """Returns an ArenaResult instance."""
        # Setup mock environment
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"spatial": None, "vector": None}, {})
        mock_env.current_player = 0
        mock_env.step.return_value = (
            {"spatial": None, "vector": None},
            0.0,
            True,  # terminated
            False,  # truncated
            {},
        )
        mock_env.winner = 0
        mock_create_env.return_value = mock_env

        # Setup mock MCTS
        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.9, 0.1], 0.5)
        mock_mcts.select_action.return_value = 0
        mock_create_mcts.return_value = mock_mcts

        # Create dummy models
        model_a = torch.nn.Linear(10, 10)
        model_b = torch.nn.Linear(10, 10)

        result = run_arena(model_a, model_b, num_games=2)

        assert isinstance(result, ArenaResult)

    @patch("alpha_zero.core.mcts_factory.create_mcts")
    @patch("alpha_zero.envs.env_factory.create_game_env")
    def test_total_games_matches_input(
        self, mock_create_env: MagicMock, mock_create_mcts: MagicMock
    ) -> None:
        """Result total_games matches num_games input."""
        # Setup mock environment
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"spatial": None, "vector": None}, {})
        mock_env.current_player = 0
        mock_env.step.return_value = (
            {"spatial": None, "vector": None},
            0.0,
            True,
            False,
            {},
        )
        mock_env.winner = 0
        mock_create_env.return_value = mock_env

        # Setup mock MCTS
        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.9, 0.1], 0.5)
        mock_mcts.select_action.return_value = 0
        mock_create_mcts.return_value = mock_mcts

        model_a = torch.nn.Linear(10, 10)
        model_b = torch.nn.Linear(10, 10)

        result = run_arena(model_a, model_b, num_games=4)

        assert result.total_games == 4

    @patch("alpha_zero.core.mcts_factory.create_mcts")
    @patch("alpha_zero.envs.env_factory.create_game_env")
    def test_wins_draws_losses_sum_to_total(
        self, mock_create_env: MagicMock, mock_create_mcts: MagicMock
    ) -> None:
        """Wins + draws + losses equals total_games."""
        # Setup mock environment with mixed results
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"spatial": None, "vector": None}, {})
        mock_env.current_player = 0

        # Game outcomes: win, draw, loss, win
        winners = [0, None, 1, 0]
        game_idx = [0]

        def step_side_effect(*args, **kwargs):
            return (
                {"spatial": None, "vector": None},
                0.0,
                True,
                False,
                {},
            )

        mock_env.step.side_effect = step_side_effect

        def winner_side_effect():
            idx = min(game_idx[0], len(winners) - 1)
            return winners[idx]

        type(mock_env).winner = property(lambda self: winner_side_effect())

        def reset_side_effect(*args, **kwargs):
            game_idx[0] += 1
            return ({"spatial": None, "vector": None}, {})

        mock_env.reset.side_effect = reset_side_effect
        mock_create_env.return_value = mock_env

        # Setup mock MCTS
        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.9, 0.1], 0.5)
        mock_mcts.select_action.return_value = 0
        mock_create_mcts.return_value = mock_mcts

        model_a = torch.nn.Linear(10, 10)
        model_b = torch.nn.Linear(10, 10)

        result = run_arena(model_a, model_b, num_games=4)

        assert result.wins + result.draws + result.losses == result.total_games

    @patch("alpha_zero.core.mcts_factory.create_mcts")
    @patch("alpha_zero.envs.env_factory.create_game_env")
    def test_alternates_starting_player(
        self, mock_create_env: MagicMock, mock_create_mcts: MagicMock
    ) -> None:
        """Alternates which player model_a controls each game."""
        # Track env resets
        reset_count = [0]

        mock_env = MagicMock()

        def reset_side_effect(*args, **kwargs):
            reset_count[0] += 1
            return ({"spatial": None, "vector": None}, {})

        mock_env.reset.side_effect = reset_side_effect
        mock_env.current_player = 0
        mock_env.step.return_value = (
            {"spatial": None, "vector": None},
            0.0,
            True,
            False,
            {},
        )
        mock_env.winner = 0
        mock_create_env.return_value = mock_env

        # Setup mock MCTS
        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.9, 0.1], 0.5)
        mock_mcts.select_action.return_value = 0
        mock_create_mcts.return_value = mock_mcts

        model_a = torch.nn.Linear(10, 10)
        model_b = torch.nn.Linear(10, 10)

        num_games = 6
        run_arena(model_a, model_b, num_games=num_games)

        # Verify env.reset was called once per game
        # (num_games calls to reset)
        assert reset_count[0] == num_games

        # Verify alternating pattern by checking factory calls
        # Environment should be created fresh each game
        assert mock_create_env.call_count == num_games

    @patch("alpha_zero.core.mcts_factory.create_mcts")
    @patch("alpha_zero.envs.env_factory.create_game_env")
    def test_default_device_is_cpu(
        self, mock_create_env: MagicMock, mock_create_mcts: MagicMock
    ) -> None:
        """When device=None, defaults to CPU device."""
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"spatial": None, "vector": None}, {})
        mock_env.current_player = 0
        mock_env.step.return_value = (
            {"spatial": None, "vector": None},
            0.0,
            True,
            False,
            {},
        )
        mock_env.winner = 0
        mock_create_env.return_value = mock_env

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.9, 0.1], 0.5)
        mock_mcts.select_action.return_value = 0
        mock_create_mcts.return_value = mock_mcts

        model_a = torch.nn.Linear(10, 10)
        model_b = torch.nn.Linear(10, 10)

        run_arena(model_a, model_b, num_games=1, device=None)

        # Check that create_mcts was called with device="cpu"
        calls = mock_create_mcts.call_args_list
        for call_args in calls:
            assert call_args[1]["device"] == "cpu"

    @patch("alpha_zero.core.mcts_factory.create_mcts")
    @patch("alpha_zero.envs.env_factory.create_game_env")
    def test_mcts_search_called_with_no_noise(
        self, mock_create_env: MagicMock, mock_create_mcts: MagicMock
    ) -> None:
        """MCTS search is called with add_noise=False for evaluation."""
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"spatial": None, "vector": None}, {})
        mock_env.current_player = 0
        mock_env.step.return_value = (
            {"spatial": None, "vector": None},
            0.0,
            True,
            False,
            {},
        )
        mock_env.winner = 0
        mock_create_env.return_value = mock_env

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.9, 0.1], 0.5)
        mock_mcts.select_action.return_value = 0
        mock_create_mcts.return_value = mock_mcts

        model_a = torch.nn.Linear(10, 10)
        model_b = torch.nn.Linear(10, 10)

        run_arena(model_a, model_b, num_games=2)

        # Verify all search calls use add_noise=False
        for call_args in mock_mcts.search.call_args_list:
            assert call_args[1]["add_noise"] is False

    @patch("alpha_zero.core.mcts_factory.create_mcts")
    @patch("alpha_zero.envs.env_factory.create_game_env")
    def test_mcts_select_action_temperature_zero(
        self, mock_create_env: MagicMock, mock_create_mcts: MagicMock
    ) -> None:
        """MCTS select_action is called with temperature=0.0 for greedy eval."""
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"spatial": None, "vector": None}, {})
        mock_env.current_player = 0
        mock_env.step.return_value = (
            {"spatial": None, "vector": None},
            0.0,
            True,
            False,
            {},
        )
        mock_env.winner = 0
        mock_create_env.return_value = mock_env

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.9, 0.1], 0.5)
        mock_mcts.select_action.return_value = 0
        mock_create_mcts.return_value = mock_mcts

        model_a = torch.nn.Linear(10, 10)
        model_b = torch.nn.Linear(10, 10)

        run_arena(model_a, model_b, num_games=2)

        # Verify all select_action calls use temperature=0.0
        for call_args in mock_mcts.select_action.call_args_list:
            # call_args is a call object; call_args.args and call_args.kwargs
            args = call_args[0]  # positional args tuple
            kwargs = call_args[1]  # kwargs dict
            # Temperature can be positional (2nd arg) or keyword
            if len(args) > 1:
                assert args[1] == 0.0
            else:
                assert kwargs.get("temperature") == 0.0

    @patch("alpha_zero.core.mcts_factory.create_mcts")
    @patch("alpha_zero.envs.env_factory.create_game_env")
    def test_handles_draw_outcome(
        self, mock_create_env: MagicMock, mock_create_mcts: MagicMock
    ) -> None:
        """Correctly counts draws when env.winner is None."""
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"spatial": None, "vector": None}, {})
        mock_env.current_player = 0
        mock_env.step.return_value = (
            {"spatial": None, "vector": None},
            0.0,
            True,
            False,
            {},
        )
        mock_env.winner = None  # Draw
        mock_create_env.return_value = mock_env

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.9, 0.1], 0.5)
        mock_mcts.select_action.return_value = 0
        mock_create_mcts.return_value = mock_mcts

        model_a = torch.nn.Linear(10, 10)
        model_b = torch.nn.Linear(10, 10)

        result = run_arena(model_a, model_b, num_games=3)

        assert result.draws == 3
        assert result.wins == 0
        assert result.losses == 0

    @patch("alpha_zero.core.mcts_factory.create_mcts")
    @patch("alpha_zero.envs.env_factory.create_game_env")
    def test_handles_all_wins(
        self, mock_create_env: MagicMock, mock_create_mcts: MagicMock
    ) -> None:
        """Correctly counts wins when model_a wins all games."""
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"spatial": None, "vector": None}, {})

        def reset_side_effect(*args, **kwargs):
            return ({"spatial": None, "vector": None}, {})

        mock_env.reset.side_effect = reset_side_effect
        mock_env.current_player = 0
        mock_env.step.return_value = (
            {"spatial": None, "vector": None},
            0.0,
            True,
            False,
            {},
        )

        # Make model_a always win by returning winner matching a_is_player_0 pattern
        def winner_side_effect():
            # Game 0, 2, 4: a_is_player_0 = True, winner should be 0
            # Game 1, 3, 5: a_is_player_0 = False, winner should be 1
            current_game = mock_create_env.call_count - 1
            return 0 if current_game % 2 == 0 else 1

        type(mock_env).winner = property(lambda self: winner_side_effect())
        mock_create_env.return_value = mock_env

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.9, 0.1], 0.5)
        mock_mcts.select_action.return_value = 0
        mock_create_mcts.return_value = mock_mcts

        model_a = torch.nn.Linear(10, 10)
        model_b = torch.nn.Linear(10, 10)

        result = run_arena(model_a, model_b, num_games=4)

        assert result.wins == 4
        assert result.draws == 0
        assert result.losses == 0

    @patch("alpha_zero.core.mcts_factory.create_mcts")
    @patch("alpha_zero.envs.env_factory.create_game_env")
    def test_uses_provided_mcts_parameters(
        self, mock_create_env: MagicMock, mock_create_mcts: MagicMock
    ) -> None:
        """Passes all MCTS parameters correctly to create_mcts."""
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"spatial": None, "vector": None}, {})
        mock_env.current_player = 0
        mock_env.step.return_value = (
            {"spatial": None, "vector": None},
            0.0,
            True,
            False,
            {},
        )
        mock_env.winner = 0
        mock_create_env.return_value = mock_env

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.9, 0.1], 0.5)
        mock_mcts.select_action.return_value = 0
        mock_create_mcts.return_value = mock_mcts

        model_a = torch.nn.Linear(10, 10)
        model_b = torch.nn.Linear(10, 10)

        run_arena(
            model_a,
            model_b,
            num_games=1,
            mcts_type="gumbel",
            mcts_sims=50,
            gumbel_num_actions=32,
        )

        # Check first call (for model_a)
        call_kwargs = mock_create_mcts.call_args_list[0][1]
        assert call_kwargs["variant"] == "gumbel"
        assert call_kwargs["num_simulations"] == 50
        assert call_kwargs["num_sampled_actions"] == 32

    @patch("alpha_zero.core.mcts_factory.create_mcts")
    @patch("alpha_zero.envs.env_factory.create_game_env")
    def test_uses_provided_env_parameters(
        self, mock_create_env: MagicMock, mock_create_mcts: MagicMock
    ) -> None:
        """Passes environment parameters correctly to create_game_env."""
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"spatial": None, "vector": None}, {})
        mock_env.current_player = 0
        mock_env.step.return_value = (
            {"spatial": None, "vector": None},
            0.0,
            True,
            False,
            {},
        )
        mock_env.winner = 0
        mock_create_env.return_value = mock_env

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.9, 0.1], 0.5)
        mock_mcts.select_action.return_value = 0
        mock_create_mcts.return_value = mock_mcts

        model_a = torch.nn.Linear(10, 10)
        model_b = torch.nn.Linear(10, 10)

        run_arena(
            model_a,
            model_b,
            num_games=1,
            board_size=16,
            max_turns=50,
        )

        # Check environment creation via factory
        call_kwargs = mock_create_env.call_args[1]
        assert call_kwargs["board_size"] == 16
        assert call_kwargs["max_turns"] == 50


class TestArenaMCTSVariantKwargs:
    """Arena must build an MCTS for every variant it documents as supported.

    ``create_mcts`` rejects unknown keyword arguments, and arena passed the
    Gumbel-family ``num_sampled_actions`` unconditionally -- so ``civ6``, which
    ``run_arena``'s own docstring lists as an option, raised TypeError on the
    first game of every run. This is the deployed evaluation path.

    The existing variant tests parametrize over ``parse_args`` only, which is
    why they read as end-to-end coverage while never constructing a search.
    """

    @pytest.mark.parametrize("variant", ["civ6", "gumbel", "batch_gumbel"])
    def test_every_documented_variant_constructs(self, variant: str) -> None:
        """Building a search for each variant must not raise on kwargs."""
        import torch

        from alpha_zero.cloud.gcp.arena import _GUMBEL_VARIANTS
        from alpha_zero.core.mcts_factory import create_mcts

        class _Net(torch.nn.Module):
            def predict(self, *args, **kwargs):  # pragma: no cover - not called
                raise AssertionError("network should not be evaluated here")

        # Exactly the kwargs arena builds for this variant.
        kwargs: dict = {"num_simulations": 2}
        if variant in _GUMBEL_VARIANTS:
            kwargs["num_sampled_actions"] = 4

        create_mcts(variant=variant, network=_Net(), device="cpu", **kwargs)

    def test_gumbel_variants_set_matches_the_factory(self) -> None:
        """The set arena filters on must agree with what the factory accepts."""
        import torch

        from alpha_zero.cloud.gcp.arena import _GUMBEL_VARIANTS
        from alpha_zero.core.mcts_factory import create_mcts

        class _Net(torch.nn.Module):
            def predict(self, *args, **kwargs):  # pragma: no cover - not called
                raise AssertionError("network should not be evaluated here")

        for variant in ("civ6",):
            with pytest.raises(TypeError):
                create_mcts(
                    variant=variant,
                    network=_Net(),
                    device="cpu",
                    num_simulations=2,
                    num_sampled_actions=4,
                )
            assert variant not in _GUMBEL_VARIANTS
