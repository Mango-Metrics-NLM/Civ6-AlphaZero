"""Tests for gameplay_preset integration in arena.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from alpha_zero.cloud.gcp.arena import ArenaResult, run_arena


class TestArenaGameplayPreset:
    """Arena correctly propagates gameplay_preset to env creation."""

    def test_run_arena_default_preset_backward_compat(self) -> None:
        """run_arena works without gameplay_preset arg (backward compat)."""
        model_a = MagicMock(spec=torch.nn.Module)
        model_b = MagicMock(spec=torch.nn.Module)

        # Patch source modules (arena imports from these inside run_arena)
        with (
            patch("alpha_zero.envs.env_factory.create_game_env") as mock_factory,
            patch("alpha_zero.core.mcts_factory.create_mcts") as mock_mcts,
        ):
            # Set up mock env
            mock_env = MagicMock()
            mock_env.current_player = 0
            mock_env.winner = None
            mock_env.reset.return_value = (None, None)
            mock_env.step.return_value = (None, 0.0, True, False, {})
            mock_factory.return_value = mock_env

            # Set up mock MCTS
            mock_mcts_instance = MagicMock()
            mock_mcts_instance.search.return_value = ([0.5, 0.5], 0.0)
            mock_mcts_instance.select_action.return_value = 0
            mock_mcts.return_value = mock_mcts_instance

            result = run_arena(
                model_a=model_a,
                model_b=model_b,
                num_games=1,
                board_size=6,
                max_turns=10,
            )

            # Should use default preset
            mock_factory.assert_called_with(
                board_size=6,
                max_turns=10,
                gameplay_preset="default",
            )
            assert isinstance(result, ArenaResult)

    def test_run_arena_enhanced_preset_calls_factory(self) -> None:
        """run_arena with gameplay_preset='enhanced' passes it to factory."""
        model_a = MagicMock(spec=torch.nn.Module)
        model_b = MagicMock(spec=torch.nn.Module)

        with (
            patch("alpha_zero.envs.env_factory.create_game_env") as mock_factory,
            patch("alpha_zero.core.mcts_factory.create_mcts") as mock_mcts,
        ):
            mock_env = MagicMock()
            mock_env.current_player = 0
            mock_env.winner = 0
            mock_env.reset.return_value = (None, None)
            mock_env.step.return_value = (None, 1.0, True, False, {})
            mock_factory.return_value = mock_env

            mock_mcts_instance = MagicMock()
            mock_mcts_instance.search.return_value = ([0.5, 0.5], 0.0)
            mock_mcts_instance.select_action.return_value = 0
            mock_mcts.return_value = mock_mcts_instance

            result = run_arena(
                model_a=model_a,
                model_b=model_b,
                num_games=2,
                board_size=6,
                max_turns=10,
                gameplay_preset="enhanced",
            )

            # Should call factory twice (once per game) with enhanced preset
            assert mock_factory.call_count == 2
            for call_args in mock_factory.call_args_list:
                assert call_args == (
                    (),
                    {"board_size": 6, "max_turns": 10, "gameplay_preset": "enhanced"},
                )
            assert isinstance(result, ArenaResult)
            assert result.total_games == 2


class TestTournamentGameplayPreset:
    """Tournament correctly threads gameplay_preset."""

    def test_run_tournament_gameplay_preset_propagates(self) -> None:
        """run_tournament passes gameplay_preset to run_arena."""
        from alpha_zero.cloud.gcp.tournament_worker import run_tournament

        # run_arena is imported inside run_tournament from arena module
        with patch("alpha_zero.cloud.gcp.arena.run_arena") as mock_arena:
            mock_arena.return_value = ArenaResult(wins=1, losses=0, draws=0, total_games=1)

            run_tournament(
                config=MagicMock(),
                model_versions=[0, 1],
                tournament_games=1,
                board_size=6,
                max_turns=10,
                dry_run=True,
                gameplay_preset="enhanced",
            )

            # run_arena should have been called with gameplay_preset
            assert mock_arena.call_count == 1
            _, kwargs = mock_arena.call_args
            assert kwargs["gameplay_preset"] == "enhanced"

    def test_run_tournament_default_preset(self) -> None:
        """run_tournament defaults gameplay_preset to 'default'."""
        from alpha_zero.cloud.gcp.tournament_worker import run_tournament

        with patch("alpha_zero.cloud.gcp.arena.run_arena") as mock_arena:
            mock_arena.return_value = ArenaResult(wins=0, losses=0, draws=1, total_games=1)

            run_tournament(
                config=MagicMock(),
                model_versions=[0, 1],
                tournament_games=1,
                board_size=6,
                max_turns=10,
                dry_run=True,
            )

            _, kwargs = mock_arena.call_args
            assert kwargs["gameplay_preset"] == "default"
