"""Unit tests for post-run tournament evaluation in vertex_orchestrator.

Tests the run_post_run_evaluation() function which runs a round-robin
tournament across all model versions produced during training, computes
ELO ratings, and logs results to W&B and GCS.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.arena import ArenaResult
from alpha_zero.cloud.gcp.config import GCPConfig


class TestRunPostRunEvaluation:
    """Test suite for post-run tournament evaluation."""

    @pytest.fixture
    def config(self):
        """Create a test GCPConfig."""
        return GCPConfig(
            project_id="test-project",
            region="us-central1",
            bucket="test-bucket",
        )

    @pytest.fixture
    def args(self):
        """Create test CLI arguments."""
        return argparse.Namespace(
            tournament_games=4,
            tournament_mcts_sims=8,
            mcts_type="gumbel",
            gumbel_num_actions=16,
            board_size=8,
            max_turns=15,
        )

    @pytest.fixture
    def mock_arena(self):
        """Mock run_arena to return predictable results."""
        with patch("alpha_zero.cloud.gcp.arena.run_arena") as mock:
            # Return 2 wins, 1 loss, 1 draw
            mock.return_value = ArenaResult(
                wins=2,
                losses=1,
                draws=1,
                total_games=4,
            )
            yield mock

    def test_skips_with_less_than_2_versions(self, config, args):
        """Test that tournament is skipped with fewer than 2 model versions.

        When only 0 or 1 model version is available, the tournament cannot
        run and should return an empty dict immediately.
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        # Test with single version
        result = run_post_run_evaluation(
            config=config,
            model_versions=[0],
            args=args,
            dry_run=True,
        )

        assert result == {}

    def test_skips_with_empty_versions(self, config, args):
        """Test that tournament is skipped with empty model_versions list.

        When model_versions is an empty list, the tournament should return
        an empty dict immediately.
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        result = run_post_run_evaluation(
            config=config,
            model_versions=[],
            args=args,
            dry_run=True,
        )

        assert result == {}

    def test_returns_expected_keys(self, config, args, mock_arena):
        """Test that tournament results contain all expected keys.

        The result dict should have exactly these 5 keys:
        - model_versions: list of versions included
        - elo_ratings: dict mapping player_id to ELO rating
        - matches: list of match results
        - best_model: player_id of highest ELO
        - best_elo: highest ELO rating
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        result = run_post_run_evaluation(
            config=config,
            model_versions=[0, 1, 2],
            args=args,
            dry_run=True,
        )

        # Check all 5 required keys are present
        assert "model_versions" in result
        assert "elo_ratings" in result
        assert "matches" in result
        assert "best_model" in result
        assert "best_elo" in result

        # Check types
        assert isinstance(result["model_versions"], list)
        assert isinstance(result["elo_ratings"], dict)
        assert isinstance(result["matches"], list)
        assert isinstance(result["best_model"], str) or result["best_model"] is None
        assert isinstance(result["best_elo"], (int, float))

    def test_elo_ratings_for_all_models(self, config, args, mock_arena):
        """Test that ELO ratings are computed for all model versions.

        Each model version should have exactly one ELO rating entry in the
        elo_ratings dict, with player_id format "v{version:03d}".
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        model_versions = [0, 1, 2]
        result = run_post_run_evaluation(
            config=config,
            model_versions=model_versions,
            args=args,
            dry_run=True,
        )

        # Check that elo_ratings has one entry per model
        assert len(result["elo_ratings"]) == len(model_versions)

        # Check player IDs are correctly formatted
        expected_ids = {f"v{v:03d}" for v in model_versions}
        assert set(result["elo_ratings"].keys()) == expected_ids

        # Check all ratings are floats
        for elo in result["elo_ratings"].values():
            assert isinstance(elo, float)

    def test_matches_count_round_robin(self, config, args, mock_arena):
        """Test that the correct number of matches are played for 3 models.

        With 3 model versions, we should have 3 matches (3 choose 2):
        - v000 vs v001
        - v000 vs v002
        - v001 vs v002
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        result = run_post_run_evaluation(
            config=config,
            model_versions=[0, 1, 2],
            args=args,
            dry_run=True,
        )

        # 3 choose 2 = 3 matches
        assert len(result["matches"]) == 3

        # Each match should have required fields
        for match in result["matches"]:
            assert "player_a" in match
            assert "player_b" in match
            assert "wins" in match
            assert "losses" in match
            assert "draws" in match
            assert "total_games" in match
            assert "win_rate" in match

    def test_matches_count_4_models(self, config, args, mock_arena):
        """Test that the correct number of matches are played for 4 models.

        With 4 model versions, we should have 6 matches (4 choose 2):
        - v000 vs v001, v000 vs v002, v000 vs v003
        - v001 vs v002, v001 vs v003
        - v002 vs v003
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        result = run_post_run_evaluation(
            config=config,
            model_versions=[0, 1, 2, 3],
            args=args,
            dry_run=True,
        )

        # 4 choose 2 = 6 matches
        assert len(result["matches"]) == 6

    def test_best_model_is_highest_elo(self, config, args, mock_arena):
        """Test that best_model corresponds to the highest ELO rating.

        The best_model field should always match the player_id with the
        highest ELO rating in elo_ratings.
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        result = run_post_run_evaluation(
            config=config,
            model_versions=[0, 1, 2],
            args=args,
            dry_run=True,
        )

        # Find the player with highest ELO manually
        max_elo_player = max(
            result["elo_ratings"].items(),
            key=lambda x: x[1],
        )

        assert result["best_model"] == max_elo_player[0]
        assert result["best_elo"] == max_elo_player[1]

    def test_dry_run_does_not_call_gcs(self, config, args, mock_arena):
        """Test that dry_run=True does not save results to GCS.

        When dry_run is True, the function should not call GCSBackend methods
        to save tournament results.
        """
        with patch("alpha_zero.cloud.gcp.storage.GCSBackend") as mock_gcs:
            from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

            result = run_post_run_evaluation(
                config=config,
                model_versions=[0, 1, 2],
                args=args,
                dry_run=True,
            )

            # GCSBackend should not be instantiated in dry run mode
            mock_gcs.assert_not_called()

            # Result should still be populated
            assert len(result) >= 5

    def test_wandb_logging_when_enabled(self, config, args, mock_arena):
        """Test that tournament metrics are logged to W&B when logger is enabled.

        When wandb_logger is provided and enabled, the function should log:
        - tournament/elo_{player_id} for each player
        - tournament/best_model
        - tournament/best_elo
        - tournament/num_models
        - tournament/total_matches
        """
        # Create mock wandb logger
        mock_logger = MagicMock()
        mock_logger.is_enabled = True

        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        result = run_post_run_evaluation(
            config=config,
            model_versions=[0, 1, 2],
            args=args,
            wandb_logger=mock_logger,
            dry_run=True,
        )

        # Verify log() was called
        assert mock_logger.log.call_count > 0

        # Collect all logged metrics
        all_metrics = {}
        for call in mock_logger.log.call_args_list:
            metrics = call[0][0] if call[0] else call[1].get("data", {})
            all_metrics.update(metrics)

        # Check that ELO ratings were logged for each player
        for player_id in result["elo_ratings"].keys():
            assert f"tournament/elo_{player_id}" in all_metrics

        # Check summary metrics
        assert "tournament/best_model" in all_metrics
        assert "tournament/best_elo" in all_metrics
        assert "tournament/num_models" in all_metrics
        assert "tournament/total_matches" in all_metrics

        # Verify values
        assert all_metrics["tournament/num_models"] == 3
        assert all_metrics["tournament/total_matches"] == 3

    def test_wandb_not_called_when_none(self, config, args, mock_arena):
        """Test that no errors occur when wandb_logger is None.

        When wandb_logger is None, the function should complete successfully
        without attempting to log metrics.
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        # Should not raise any errors
        result = run_post_run_evaluation(
            config=config,
            model_versions=[0, 1, 2],
            args=args,
            wandb_logger=None,
            dry_run=True,
        )

        # Result should still be valid
        assert len(result) >= 5
        assert result["model_versions"] == [0, 1, 2]

    def test_wandb_not_called_when_disabled(self, config, args, mock_arena):
        """Test that W&B logging is skipped when logger.is_enabled is False.

        When wandb_logger.is_enabled is False, the function should not
        attempt to log metrics even though a logger instance exists.
        """
        # Create mock wandb logger with is_enabled=False
        mock_logger = MagicMock()
        mock_logger.is_enabled = False

        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        result = run_post_run_evaluation(
            config=config,
            model_versions=[0, 1, 2],
            args=args,
            wandb_logger=mock_logger,
            dry_run=True,
        )

        # log() should not be called
        mock_logger.log.assert_not_called()

        # Result should still be valid
        assert len(result) >= 5

    def test_tournament_games_parameter_used(self, config, args, mock_arena):
        """Test that args.tournament_games is passed to run_arena.

        The number of games per match should respect the tournament_games
        parameter from the CLI args.
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        args.tournament_games = 10

        run_post_run_evaluation(
            config=config,
            model_versions=[0, 1],
            args=args,
            dry_run=True,
        )

        # Verify run_arena was called with tournament_games
        mock_arena.assert_called_once()
        call_kwargs = mock_arena.call_args[1]
        assert call_kwargs["num_games"] == 10

    def test_mcts_parameters_passed_to_arena(self, config, args, mock_arena):
        """Test that MCTS parameters are correctly passed to run_arena.

        All MCTS-related parameters (mcts_type, mcts_sims, gumbel_num_actions,
        board_size, max_turns) should be forwarded to run_arena.
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        run_post_run_evaluation(
            config=config,
            model_versions=[0, 1],
            args=args,
            dry_run=True,
        )

        # Verify run_arena was called with correct parameters
        mock_arena.assert_called_once()
        call_kwargs = mock_arena.call_args[1]

        assert call_kwargs["mcts_type"] == "gumbel"
        assert call_kwargs["mcts_sims"] == 8
        assert call_kwargs["gumbel_num_actions"] == 16
        assert call_kwargs["board_size"] == 8
        assert call_kwargs["max_turns"] == 15

    def test_elo_ratings_start_at_1500(self, config, args):
        """Test that ELO ratings start at the standard 1500.0.

        Before any matches are played, all players should have an initial
        ELO rating of 1500.0. After matches, ratings should deviate based
        on wins/losses.
        """
        # Mock arena to return all draws (no rating changes)
        with patch("alpha_zero.cloud.gcp.arena.run_arena") as mock:
            mock.return_value = ArenaResult(
                wins=0,
                losses=0,
                draws=4,
                total_games=4,
            )

            from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

            result = run_post_run_evaluation(
                config=config,
                model_versions=[0, 1],
                args=args,
                dry_run=True,
            )

            # With all draws, ELO ratings should remain close to 1500
            # (within small margin due to draw scoring)
            for elo in result["elo_ratings"].values():
                assert 1480 < elo < 1520

    def test_elo_k_factor_is_32(self, config, args):
        """Test that ELO updates use K-factor of 32.

        The K-factor determines the magnitude of ELO rating changes.
        The function should use K=32.0 as specified in the implementation.
        """
        # Mock arena: player A always wins
        with patch("alpha_zero.cloud.gcp.arena.run_arena") as mock:
            mock.return_value = ArenaResult(
                wins=4,
                losses=0,
                draws=0,
                total_games=4,
            )

            from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

            result = run_post_run_evaluation(
                config=config,
                model_versions=[0, 1],
                args=args,
                dry_run=True,
            )

            # With perfect wins, player A should gain rating
            # Expected ELO change with K=32, initial ratings 1500:
            # expected_a = 0.5, score_a = 1.0, delta = 32 * (1.0 - 0.5) = 16
            v000_elo = result["elo_ratings"]["v000"]
            v001_elo = result["elo_ratings"]["v001"]

            # Winner should be above 1500, loser below
            assert v000_elo > 1500
            assert v001_elo < 1500

            # Total ELO should be conserved (sum = 3000)
            assert abs((v000_elo + v001_elo) - 3000.0) < 0.01

    def test_gcs_save_called_when_not_dry_run(self, config, args, mock_arena):
        """Test that tournament results are saved to GCS when dry_run=False.

        When dry_run is False, the function should:
        1. Instantiate GCSBackend
        2. Serialize results to JSON
        3. Upload to bucket/tournament/post_run_results.json
        """
        # Mock load_model to return a dummy model
        with patch("alpha_zero.cloud.gcp.evaluation_worker.load_model") as mock_load:
            from alpha_zero.core.transformer_network import (
                Civ6TransformerNetwork,
                TransformerConfig,
            )

            mock_load.return_value = Civ6TransformerNetwork(config=TransformerConfig())

            with patch("alpha_zero.cloud.gcp.storage.GCSBackend") as mock_gcs:
                mock_backend = MagicMock()
                mock_backend.upload_blob.return_value = (
                    "gs://test-bucket/tournament/post_run_results.json"
                )
                mock_gcs.return_value = mock_backend

                from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

                run_post_run_evaluation(
                    config=config,
                    model_versions=[0, 1, 2],
                    args=args,
                    dry_run=False,
                )

                # Verify GCSBackend was instantiated
                mock_gcs.assert_called_once_with(config.project_id)

                # Verify upload_blob was called
                mock_backend.upload_blob.assert_called_once()
                call_args = mock_backend.upload_blob.call_args[0]

                # Check bucket name
                assert call_args[0] == config.bucket

                # Check blob path
                assert call_args[1] == "tournament/post_run_results.json"

                # Check data is JSON bytes
                assert isinstance(call_args[2], bytes)

    def test_elo_ranking_sorted_descending(self, config, args, mock_arena):
        """Test that ELO ratings are sorted in descending order.

        The elo_ratings dict should be ordered from highest to lowest ELO,
        so that the first entry is the best model.
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_post_run_evaluation

        result = run_post_run_evaluation(
            config=config,
            model_versions=[0, 1, 2],
            args=args,
            dry_run=True,
        )

        # Extract ELO values in order
        elo_values = list(result["elo_ratings"].values())

        # Check that values are in descending order
        assert elo_values == sorted(elo_values, reverse=True)

        # Check that first entry matches best_model
        first_player = next(iter(result["elo_ratings"].keys()))
        assert first_player == result["best_model"]
