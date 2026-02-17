"""Tests for multiplayer scoring and ranking."""

from __future__ import annotations

import pytest

from alpha_zero.multiplayer.scoring import (
    ScoringMethod,
    compute_player_values,
    compute_rankings,
    extract_game_result,
)


class TestComputeRankings:
    """Tests for compute_rankings."""

    def test_winner_gets_rank_0(self):
        rankings = compute_rankings(winner=2, num_players=4)
        assert rankings[2] == 0

    def test_non_winners_share_rank(self):
        rankings = compute_rankings(winner=0, num_players=3)
        assert rankings[0] == 0
        assert rankings[1] == 1
        assert rankings[2] == 1

    def test_draw_all_rank_0(self):
        rankings = compute_rankings(winner=None, num_players=4)
        assert all(r == 0 for r in rankings.values())

    def test_score_based_ranking(self):
        scores = {0: 100, 1: 50, 2: 75}
        rankings = compute_rankings(player_scores=scores, num_players=3)
        assert rankings[0] == 0  # Highest score
        assert rankings[2] == 1  # Second
        assert rankings[1] == 2  # Lowest

    def test_score_ties_share_rank(self):
        scores = {0: 100, 1: 100, 2: 50}
        rankings = compute_rankings(player_scores=scores, num_players=3)
        assert rankings[0] == 0
        assert rankings[1] == 0  # Tied with player 0
        assert rankings[2] == 2

    def test_eliminated_players_ranked_last(self):
        rankings = compute_rankings(
            winner=0, eliminated={2}, num_players=3
        )
        assert rankings[0] == 0
        assert rankings[1] == 1
        assert rankings[2] >= 2

    def test_invalid_num_players(self):
        with pytest.raises(ValueError, match="num_players"):
            compute_rankings(winner=0, num_players=1)

    def test_invalid_winner(self):
        with pytest.raises(ValueError, match="winner"):
            compute_rankings(winner=5, num_players=3)

    def test_two_player_winner(self):
        """2-player game produces correct rankings."""
        rankings = compute_rankings(winner=1, num_players=2)
        assert rankings[1] == 0
        assert rankings[0] == 1


class TestComputePlayerValues:
    """Tests for compute_player_values."""

    def test_linear_ranking_3_players(self):
        rankings = {0: 0, 1: 1, 2: 2}
        values = compute_player_values(rankings, num_players=3)
        assert values[0] == pytest.approx(1.0)
        assert values[1] == pytest.approx(0.0)
        assert values[2] == pytest.approx(-1.0)

    def test_linear_ranking_2_players(self):
        """2-player linear ranking matches traditional +-1."""
        rankings = {0: 0, 1: 1}
        values = compute_player_values(rankings, num_players=2)
        assert values[0] == pytest.approx(1.0)
        assert values[1] == pytest.approx(-1.0)

    def test_linear_ranking_4_players(self):
        rankings = {0: 0, 1: 1, 2: 2, 3: 3}
        values = compute_player_values(rankings, num_players=4)
        assert values[0] == pytest.approx(1.0)
        assert values[1] == pytest.approx(1.0 / 3)
        assert values[2] == pytest.approx(-1.0 / 3)
        assert values[3] == pytest.approx(-1.0)

    def test_winner_take_all(self):
        rankings = {0: 0, 1: 1, 2: 2}
        values = compute_player_values(
            rankings, num_players=3, method=ScoringMethod.WINNER_TAKE_ALL
        )
        assert values[0] == pytest.approx(1.0)
        assert values[1] == pytest.approx(-1.0)
        assert values[2] == pytest.approx(-1.0)

    def test_draw_value(self):
        rankings = {0: 0, 1: 0, 2: 0}
        values = compute_player_values(
            rankings, num_players=3, draw_value=0.5
        )
        assert all(v == pytest.approx(0.5) for v in values.values())

    def test_proportional_scoring(self):
        rankings = {0: 0, 1: 1, 2: 2}
        scores = {0: 100, 1: 50, 2: 0}
        values = compute_player_values(
            rankings,
            num_players=3,
            method=ScoringMethod.PROPORTIONAL,
            player_scores=scores,
        )
        assert values[0] == pytest.approx(1.0)
        assert values[1] == pytest.approx(0.0)
        assert values[2] == pytest.approx(-1.0)

    def test_proportional_equal_scores(self):
        rankings = {0: 0, 1: 0}
        scores = {0: 50, 1: 50}
        values = compute_player_values(
            rankings,
            num_players=2,
            method=ScoringMethod.PROPORTIONAL,
            player_scores=scores,
        )
        assert values[0] == pytest.approx(0.0)
        assert values[1] == pytest.approx(0.0)

    def test_elimination_penalty(self):
        rankings = {0: 0, 1: 1, 2: 2}
        values_no_penalty = compute_player_values(rankings, num_players=3)
        values_penalty = compute_player_values(
            rankings, num_players=3, elimination_penalty=0.2
        )
        # Last place should be penalized more
        assert values_penalty[2] <= values_no_penalty[2]

    def test_values_bounded(self):
        """All values are in [-1, 1] range."""
        for n in range(2, 9):
            rankings = {i: i for i in range(n)}
            values = compute_player_values(rankings, num_players=n)
            for v in values.values():
                assert -1.0 <= v <= 1.0


class TestExtractGameResult:
    """Tests for extract_game_result."""

    def test_with_winner(self):
        info = {"winner": 0}
        values = extract_game_result(info, num_players=3)
        assert values[0] == pytest.approx(1.0)
        assert len(values) == 3

    def test_with_draw(self):
        info = {"winner": None}
        values = extract_game_result(info, num_players=2)
        assert values[0] == pytest.approx(0.0)
        assert values[1] == pytest.approx(0.0)

    def test_with_scores(self):
        info = {"winner": None, "player_scores": {0: 100, 1: 50}}
        values = extract_game_result(info, num_players=2)
        assert values[0] > values[1]

    def test_with_eliminated(self):
        info = {"winner": 0, "eliminated": [2]}
        values = extract_game_result(info, num_players=3)
        assert values[0] > values[1]
        assert values[0] > values[2]
