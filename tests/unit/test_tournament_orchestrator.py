"""Unit tests for tournament orchestrator.

Tests cover:
- Tournament format enums
- Bracket creation for different formats
- Round-robin scheduling
- Single-elimination brackets
- Swiss format pairing
- Match execution and result recording
- Tournament running and completion
- Standings calculation
"""

import uuid
from unittest.mock import Mock

import pytest

from alpha_zero.tournament.config import MatchResult, TournamentConfig
from alpha_zero.tournament.orchestrator import (
    Bracket,
    Matchup,
    PlayerStanding,
    TournamentFormat,
    TournamentOrchestrator,
    TournamentResult,
)


class TestTournamentFormat:
    """Tests for TournamentFormat enum."""

    def test_format_values(self):
        """Test format string values."""
        assert TournamentFormat.ROUND_ROBIN.value == "round_robin"
        assert TournamentFormat.SINGLE_ELIMINATION.value == "single_elimination"
        assert TournamentFormat.DOUBLE_ELIMINATION.value == "double_elimination"
        assert TournamentFormat.SWISS.value == "swiss"


class TestMatchup:
    """Tests for Matchup dataclass."""

    def test_initialization(self):
        """Test matchup initialization."""
        matchup = Matchup(player_a="p1", player_b="p2", round_num=0)
        assert matchup.player_a == "p1"
        assert matchup.player_b == "p2"
        assert matchup.round_num == 0
        assert matchup.result is None

    def test_is_complete(self):
        """Test completion check."""
        matchup = Matchup(player_a="p1", player_b="p2", round_num=0)
        assert not matchup.is_complete

        matchup.result = MatchResult(
            match_id="m1",
            player1_id="p1",
            player2_id="p2",
            winner_id="p1",
        )
        assert matchup.is_complete

    def test_winner_id(self):
        """Test winner retrieval."""
        matchup = Matchup(player_a="p1", player_b="p2", round_num=0)
        assert matchup.winner_id is None

        matchup.result = MatchResult(
            match_id="m1",
            player1_id="p1",
            player2_id="p2",
            winner_id="p1",
        )
        assert matchup.winner_id == "p1"

    def test_to_dict(self):
        """Test serialization."""
        matchup = Matchup(player_a="p1", player_b="p2", round_num=0)
        data = matchup.to_dict()

        assert data["player_a"] == "p1"
        assert data["player_b"] == "p2"
        assert data["round_num"] == 0
        assert data["result"] is None


class TestBracket:
    """Tests for Bracket dataclass."""

    def test_initialization(self):
        """Test bracket initialization."""
        bracket = Bracket(
            format=TournamentFormat.ROUND_ROBIN,
            rounds=[],
            players=["p1", "p2", "p3"],
            seeding={"p1": 1, "p2": 2, "p3": 3},
        )

        assert bracket.format == TournamentFormat.ROUND_ROBIN
        assert len(bracket.players) == 3
        assert bracket.current_round == 0

    def test_is_complete(self):
        """Test completion check."""
        bracket = Bracket(
            format=TournamentFormat.ROUND_ROBIN,
            rounds=[[], []],
            players=["p1", "p2"],
            seeding={"p1": 1, "p2": 2},
        )

        assert not bracket.is_complete
        bracket.current_round = 2
        assert bracket.is_complete

    def test_total_rounds(self):
        """Test round counting."""
        bracket = Bracket(
            format=TournamentFormat.ROUND_ROBIN,
            rounds=[[], [], []],
            players=["p1", "p2", "p3"],
            seeding={},
        )

        assert bracket.total_rounds == 3

    def test_total_matches(self):
        """Test match counting."""
        rounds = [
            [Matchup("p1", "p2", 0), Matchup("p3", "p4", 0)],
            [Matchup("p1", "p3", 1)],
        ]
        bracket = Bracket(
            format=TournamentFormat.ROUND_ROBIN,
            rounds=rounds,
            players=["p1", "p2", "p3", "p4"],
            seeding={},
        )

        assert bracket.total_matches == 3

    def test_matches_complete(self):
        """Test completed matches count."""
        bracket = Bracket(
            format=TournamentFormat.ROUND_ROBIN,
            rounds=[[]],
            players=["p1", "p2"],
            seeding={},
        )

        assert bracket.matches_complete == 0

        bracket.results[0] = [
            MatchResult("m1", "p1", "p2", winner_id="p1"),
            MatchResult("m2", "p3", "p4", winner_id="p3"),
        ]

        assert bracket.matches_complete == 2

    def test_get_player_record(self):
        """Test player record retrieval."""
        bracket = Bracket(
            format=TournamentFormat.ROUND_ROBIN,
            rounds=[[]],
            players=["p1", "p2", "p3"],
            seeding={},
        )

        bracket.results[0] = [
            MatchResult("m1", "p1", "p2", winner_id="p1"),
            MatchResult("m2", "p1", "p3", winner_id="p3"),
        ]

        wins, losses, draws = bracket.get_player_record("p1")
        assert wins == 1
        assert losses == 1
        assert draws == 0


class TestPlayerStanding:
    """Tests for PlayerStanding dataclass."""

    def test_initialization(self):
        """Test standing initialization."""
        standing = PlayerStanding(
            player_id="p1",
            rank=1,
            wins=5,
            losses=2,
            draws=1,
            elo_rating=1600,
            points=5.5,
        )

        assert standing.player_id == "p1"
        assert standing.rank == 1
        assert standing.total_games == 8

    def test_win_rate(self):
        """Test win rate calculation."""
        standing = PlayerStanding(
            player_id="p1",
            rank=1,
            wins=5,
            losses=3,
            draws=2,
            elo_rating=1600,
            points=6.0,
        )

        assert abs(standing.win_rate - 0.5) < 0.01  # 5 / 10 = 0.5

    def test_to_dict(self):
        """Test serialization."""
        standing = PlayerStanding(
            player_id="p1",
            rank=1,
            wins=5,
            losses=2,
            draws=1,
            elo_rating=1600,
            points=5.5,
        )

        data = standing.to_dict()
        assert data["player_id"] == "p1"
        assert data["wins"] == 5
        assert data["total_games"] == 8


class TestTournamentResult:
    """Tests for TournamentResult dataclass."""

    def test_initialization(self):
        """Test result initialization."""
        bracket = Bracket(
            format=TournamentFormat.ROUND_ROBIN,
            rounds=[[]],
            players=["p1", "p2"],
            seeding={},
        )

        result = TournamentResult(
            bracket=bracket,
            winner="p1",
            final_standings=[],
            total_games=10,
            duration_seconds=120.5,
        )

        assert result.winner == "p1"
        assert result.total_games == 10
        assert result.duration_seconds == 120.5

    def test_to_dict(self):
        """Test serialization."""
        bracket = Bracket(
            format=TournamentFormat.ROUND_ROBIN,
            rounds=[[]],
            players=["p1", "p2"],
            seeding={},
        )

        result = TournamentResult(
            bracket=bracket,
            winner="p1",
            final_standings=[],
            total_games=10,
            duration_seconds=120.5,
        )

        data = result.to_dict()
        assert data["winner"] == "p1"
        assert data["total_games"] == 10


class TestTournamentOrchestrator:
    """Tests for TournamentOrchestrator class."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = TournamentOrchestrator()
        assert orchestrator.config is not None
        assert orchestrator.games_per_match == 10

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config."""
        config = TournamentConfig(games_per_match=5)
        orchestrator = TournamentOrchestrator(config=config, games_per_match=5)

        assert orchestrator.config == config
        assert orchestrator.games_per_match == 5

    def test_create_round_robin_bracket(self):
        """Test creating round-robin bracket."""
        orchestrator = TournamentOrchestrator()
        players = ["p1", "p2", "p3", "p4"]

        bracket = orchestrator.create_bracket(
            players, format=TournamentFormat.ROUND_ROBIN
        )

        assert bracket.format == TournamentFormat.ROUND_ROBIN
        assert len(bracket.players) == 4
        # 4 players -> 3 rounds (each plays others once)
        assert bracket.total_rounds == 3
        # Total matches = n * (n-1) / 2 = 4 * 3 / 2 = 6
        assert bracket.total_matches == 6

    def test_create_single_elimination_bracket(self):
        """Test creating single-elimination bracket."""
        orchestrator = TournamentOrchestrator()
        players = ["p1", "p2", "p3", "p4"]

        bracket = orchestrator.create_bracket(
            players, format=TournamentFormat.SINGLE_ELIMINATION
        )

        assert bracket.format == TournamentFormat.SINGLE_ELIMINATION
        # Power of 2 bracket: 4 players -> 2 rounds (semis + final)
        assert bracket.total_rounds == 2

    def test_create_bracket_with_seeding(self):
        """Test bracket creation with custom seeding."""
        orchestrator = TournamentOrchestrator()
        players = ["p1", "p2", "p3", "p4"]
        seeding = {"p1": 1, "p2": 4, "p3": 2, "p4": 3}

        bracket = orchestrator.create_bracket(
            players, format=TournamentFormat.SINGLE_ELIMINATION, seeding=seeding
        )

        assert bracket.seeding == seeding

    def test_create_bracket_too_few_players(self):
        """Test error with too few players."""
        orchestrator = TournamentOrchestrator()

        with pytest.raises(ValueError, match="at least 2 players"):
            orchestrator.create_bracket(["p1"])

    def test_run_match_with_mock_runner(self, tmp_path):
        """Test running match with mock arena runner."""
        mock_result = MatchResult(
            match_id="m1",
            player1_id="p1",
            player2_id="p2",
            player1_score=6.0,
            player2_score=4.0,
            winner_id="p1",
        )

        arena_runner = Mock(return_value=mock_result)
        # Use isolated temporary database
        db_path = tmp_path / "test_tournament.db"
        config = TournamentConfig(database_path=str(db_path))
        orchestrator = TournamentOrchestrator(
            config=config, arena_runner=arena_runner
        )

        result = orchestrator.run_match("p1", "p2", num_games=10)

        arena_runner.assert_called_once_with("p1", "p2", 10)
        assert result.winner_id == "p1"

    def test_run_match_without_runner(self):
        """Test running match without arena runner (mock result)."""
        orchestrator = TournamentOrchestrator()

        result = orchestrator.run_match("p1", "p2", num_games=10)

        assert result.player1_id == "p1"
        assert result.player2_id == "p2"
        assert result.match_id is not None

    def test_get_standings(self):
        """Test calculating standings."""
        orchestrator = TournamentOrchestrator()
        bracket = Bracket(
            format=TournamentFormat.ROUND_ROBIN,
            rounds=[[]],
            players=["p1", "p2", "p3"],
            seeding={},
        )

        # Add some results
        bracket.results[0] = [
            MatchResult("m1", "p1", "p2", winner_id="p1"),
            MatchResult("m2", "p2", "p3", winner_id="p2"),
            MatchResult("m3", "p1", "p3", winner_id="p1"),
        ]

        standings = orchestrator.get_standings(bracket)

        assert len(standings) == 3
        # p1 should be rank 1 (2 wins)
        assert standings[0].player_id == "p1"
        assert standings[0].wins == 2
        assert standings[0].rank == 1

    def test_round_robin_scheduling(self):
        """Test round-robin produces complete schedule."""
        orchestrator = TournamentOrchestrator()
        players = ["p1", "p2", "p3", "p4"]

        bracket = orchestrator.create_bracket(
            players, format=TournamentFormat.ROUND_ROBIN
        )

        # Collect all matchups
        all_pairs = set()
        for round_matchups in bracket.rounds:
            for matchup in round_matchups:
                pair = tuple(sorted([matchup.player_a, matchup.player_b]))
                all_pairs.add(pair)

        # Should have all unique pairs
        expected_pairs = {
            ("p1", "p2"),
            ("p1", "p3"),
            ("p1", "p4"),
            ("p2", "p3"),
            ("p2", "p4"),
            ("p3", "p4"),
        }

        assert all_pairs == expected_pairs

    def test_round_robin_odd_players(self):
        """Test round-robin with odd number of players."""
        orchestrator = TournamentOrchestrator()
        players = ["p1", "p2", "p3"]

        bracket = orchestrator.create_bracket(
            players, format=TournamentFormat.ROUND_ROBIN
        )

        # Should still work, some players get byes
        assert bracket.total_rounds == 3

    def test_single_elimination_with_byes(self):
        """Test single-elimination with non-power-of-2 players."""
        orchestrator = TournamentOrchestrator()
        players = ["p1", "p2", "p3"]  # Not power of 2

        bracket = orchestrator.create_bracket(
            players, format=TournamentFormat.SINGLE_ELIMINATION
        )

        # Should pad to 4 with byes
        assert bracket.total_rounds == 2

    def test_run_tournament_minimal(self):
        """Test running minimal tournament."""

        def mock_runner(p1, p2, num_games):
            # p1 always wins
            return MatchResult(
                match_id=str(uuid.uuid4())[:8],
                player1_id=p1,
                player2_id=p2,
                player1_score=float(num_games),
                player2_score=0.0,
                winner_id=p1,
            )

        orchestrator = TournamentOrchestrator(arena_runner=mock_runner, games_per_match=1)
        bracket = orchestrator.create_bracket(["p1", "p2"], format=TournamentFormat.ROUND_ROBIN)

        result = orchestrator.run_tournament(bracket)

        assert result.winner == "p1"
        assert result.total_games >= 1
        assert len(result.final_standings) == 2

    def test_advance_bracket_single_elimination(self):
        """Test advancing single-elimination bracket."""
        orchestrator = TournamentOrchestrator()
        players = ["p1", "p2", "p3", "p4"]
        bracket = orchestrator.create_bracket(
            players, format=TournamentFormat.SINGLE_ELIMINATION
        )

        # Simulate first round results
        round_results = [
            MatchResult("m1", "p1", "p2", winner_id="p1"),
            MatchResult("m2", "p3", "p4", winner_id="p3"),
        ]

        bracket = orchestrator.advance_bracket(bracket, round_results)

        assert bracket.current_round == 1
        # Next round should have p1 vs p3
        assert len(bracket.rounds[1]) == 1
        matchup = bracket.rounds[1][0]
        assert {matchup.player_a, matchup.player_b} == {"p1", "p3"}

    def test_advance_bracket_round_robin(self):
        """Test advancing round-robin (just increments round)."""
        orchestrator = TournamentOrchestrator()
        bracket = orchestrator.create_bracket(
            ["p1", "p2", "p3"], format=TournamentFormat.ROUND_ROBIN
        )

        original_rounds = len(bracket.rounds)
        bracket = orchestrator.advance_bracket(bracket, [])

        assert bracket.current_round == 1
        assert len(bracket.rounds) == original_rounds  # No new rounds created

    def test_get_or_create_player_stats(self, tmp_path):
        """Test player stats creation."""
        # Use isolated temporary database
        db_path = tmp_path / "test_tournament.db"
        config = TournamentConfig(database_path=str(db_path))
        orchestrator = TournamentOrchestrator(config=config)

        stats = orchestrator._get_or_create_player_stats("p1")
        assert stats.player_id == "p1"
        assert stats.rating == 1500  # Default

        # Should return same instance
        stats2 = orchestrator._get_or_create_player_stats("p1")
        assert stats is stats2

    def test_swiss_format_dynamic_rounds(self):
        """Test Swiss format generates rounds dynamically."""
        orchestrator = TournamentOrchestrator()
        bracket = orchestrator.create_bracket(
            ["p1", "p2", "p3", "p4"], format=TournamentFormat.SWISS
        )

        # Swiss starts with no rounds
        assert len(bracket.rounds) == 0

        # Generate first round
        orchestrator._generate_swiss_round(bracket)
        assert len(bracket.rounds) == 1
        assert len(bracket.rounds[0]) >= 1  # At least one match


class TestTournamentEdgeCases:
    """Edge case tests for tournament orchestration."""

    def test_tournament_with_single_player(self):
        """Tournament with only 1 player should raise error."""
        orchestrator = TournamentOrchestrator()

        with pytest.raises(ValueError, match="at least 2 players"):
            orchestrator.create_bracket(["p1"], format=TournamentFormat.ROUND_ROBIN)

    def test_tournament_with_two_players(self):
        """Minimal tournament with exactly 2 players."""
        def mock_runner(p1, p2, num_games):
            # p1 always wins
            return MatchResult(
                match_id=str(uuid.uuid4())[:8],
                player1_id=p1,
                player2_id=p2,
                player1_score=float(num_games),
                player2_score=0.0,
                winner_id=p1,
            )

        orchestrator = TournamentOrchestrator(arena_runner=mock_runner, games_per_match=1)
        bracket = orchestrator.create_bracket(["p1", "p2"], format=TournamentFormat.ROUND_ROBIN)

        result = orchestrator.run_tournament(bracket)

        assert result.winner == "p1"
        assert len(result.final_standings) == 2
        assert result.total_games == 1

    def test_elo_extreme_rating_difference(self):
        """Tournament with extreme ELO differences should handle properly."""
        orchestrator = TournamentOrchestrator()

        # Create players with extreme ELO difference
        high_elo_stats = orchestrator._get_or_create_player_stats("high")
        low_elo_stats = orchestrator._get_or_create_player_stats("low")

        high_elo_stats.rating = 2800  # Very high
        low_elo_stats.rating = 800   # Very low

        def mock_runner(p1, p2, num_games):
            # High ELO player wins
            winner = "high" if "high" in [p1, p2] else p1
            return MatchResult(
                match_id=str(uuid.uuid4())[:8],
                player1_id=p1,
                player2_id=p2,
                player1_score=float(num_games) if winner == p1 else 0.0,
                player2_score=float(num_games) if winner == p2 else 0.0,
                winner_id=winner,
            )

        orchestrator.arena_runner = mock_runner
        bracket = orchestrator.create_bracket(
            ["high", "low"], format=TournamentFormat.ROUND_ROBIN
        )

        # Should not crash with extreme ratings
        result = orchestrator.run_tournament(bracket)
        assert result.winner in ["high", "low"]

    def test_all_draws_tournament(self):
        """Tournament where every match is a draw."""
        def mock_runner(p1, p2, num_games):
            # All draws
            return MatchResult(
                match_id=str(uuid.uuid4())[:8],
                player1_id=p1,
                player2_id=p2,
                player1_score=float(num_games) / 2,
                player2_score=float(num_games) / 2,
                winner_id=None,  # Draw
            )

        orchestrator = TournamentOrchestrator(arena_runner=mock_runner, games_per_match=2)
        bracket = orchestrator.create_bracket(
            ["p1", "p2", "p3"], format=TournamentFormat.ROUND_ROBIN
        )

        result = orchestrator.run_tournament(bracket)

        # Winner should be determined by tiebreaker (could be any player)
        assert result.winner in ["p1", "p2", "p3"]

        # All players should have same number of wins (0) and draws
        standings = result.final_standings
        assert all(s.wins == 0 for s in standings)
        assert all(s.draws == 2 for s in standings)  # Each plays 2 matches
