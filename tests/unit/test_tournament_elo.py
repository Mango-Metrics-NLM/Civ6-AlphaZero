"""Unit tests for ELO rating system.

Tests cover:
- Expected score calculations
- Rating change computations
- K-factor adjustment
- Rating bounds enforcement
- Match processing
"""

from alpha_zero.tournament.config import ELOConfig, MatchOutcome
from alpha_zero.tournament.elo import (
    ELORatingSystem,
    RatingLeaderboard,
    RatingUpdate,
    calculate_expected_score,
    calculate_rating_change,
)


class TestExpectedScore:
    """Tests for expected score calculation."""

    def test_equal_ratings(self):
        """Equal ratings should give 0.5 expected score."""
        expected = calculate_expected_score(1500, 1500)
        assert abs(expected - 0.5) < 0.001

    def test_higher_rating_favored(self):
        """Higher rated player should have higher expected score."""
        expected = calculate_expected_score(1600, 1400)
        assert expected > 0.5

    def test_lower_rating_underdog(self):
        """Lower rated player should have lower expected score."""
        expected = calculate_expected_score(1400, 1600)
        assert expected < 0.5

    def test_symmetric(self):
        """Expected scores should sum to 1."""
        e1 = calculate_expected_score(1600, 1400)
        e2 = calculate_expected_score(1400, 1600)
        assert abs(e1 + e2 - 1.0) < 0.001

    def test_400_point_difference(self):
        """400 point difference should give ~90% expected score."""
        expected = calculate_expected_score(1900, 1500)
        assert 0.9 < expected < 0.92


class TestRatingChange:
    """Tests for rating change calculation."""

    def test_win_increases_rating(self):
        """Winning should increase rating."""
        change = calculate_rating_change(
            rating=1500, expected_score=0.5, actual_score=1.0, k_factor=32
        )
        assert change > 0

    def test_loss_decreases_rating(self):
        """Losing should decrease rating."""
        change = calculate_rating_change(
            rating=1500, expected_score=0.5, actual_score=0.0, k_factor=32
        )
        assert change < 0

    def test_draw_at_even_ratings(self):
        """Draw at even ratings should give no change."""
        change = calculate_rating_change(
            rating=1500, expected_score=0.5, actual_score=0.5, k_factor=32
        )
        assert abs(change) < 0.01

    def test_upset_gives_bigger_change(self):
        """Upset (underdog winning) should give bigger rating change."""
        # Underdog wins (expected 0.24, got 1.0)
        upset_change = calculate_rating_change(
            rating=1400, expected_score=0.24, actual_score=1.0, k_factor=32
        )
        # Favorite wins (expected 0.76, got 1.0)
        expected_change = calculate_rating_change(
            rating=1600, expected_score=0.76, actual_score=1.0, k_factor=32
        )
        assert upset_change > expected_change

    def test_k_factor_scales_change(self):
        """Higher K-factor should give larger changes."""
        change_low_k = calculate_rating_change(
            rating=1500, expected_score=0.5, actual_score=1.0, k_factor=16
        )
        change_high_k = calculate_rating_change(
            rating=1500, expected_score=0.5, actual_score=1.0, k_factor=32
        )
        assert change_high_k == 2 * change_low_k


class TestELOConfig:
    """Tests for ELO configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ELOConfig()
        assert config.initial_rating == 1500.0
        assert config.k_factor_base == 32.0
        assert config.rating_floor == 100.0

    def test_k_factor_for_new_player(self):
        """New players should have higher K-factor."""
        config = ELOConfig(k_factor_base=32.0, provisional_threshold=30)
        k = config.get_k_factor(games_played=5, is_provisional=True)
        assert k >= config.k_factor_base

    def test_k_factor_for_experienced_player(self):
        """Experienced players should have lower K-factor."""
        config = ELOConfig(k_factor_base=32.0, k_factor_min=10.0)
        k = config.get_k_factor(games_played=100, is_provisional=False)
        assert k <= config.k_factor_base


class TestELORatingSystem:
    """Tests for ELORatingSystem class."""

    def test_initialization(self):
        """Test system initialization."""
        system = ELORatingSystem()
        assert system.config is not None

    def test_calculate_expected_scores(self):
        """Test expected score calculation for both players."""
        system = ELORatingSystem()
        e1, e2 = system.calculate_expected_scores(1600, 1400)

        assert e1 > 0.5
        assert e2 < 0.5
        assert abs(e1 + e2 - 1.0) < 0.001

    def test_calculate_rating_changes(self):
        """Test rating change calculation."""
        system = ELORatingSystem()
        change1, change2 = system.calculate_rating_changes(
            rating1=1500,
            rating2=1500,
            score1=1.0,
            score2=0.0,
        )

        assert change1 > 0
        assert change2 < 0
        # Changes should be symmetric at equal ratings
        assert abs(change1 + change2) < 0.01

    def test_apply_rating_bounds(self):
        """Test rating bounds enforcement."""
        config = ELOConfig(rating_floor=100, rating_ceiling=3000)
        system = ELORatingSystem(config)

        assert system.apply_rating_bounds(50) == 100
        assert system.apply_rating_bounds(3500) == 3000
        assert system.apply_rating_bounds(1500) == 1500

    def test_process_match(self):
        """Test processing a complete match."""
        system = ELORatingSystem()

        update1, update2 = system.process_match(
            player1_id="alice",
            player2_id="bob",
            player1_rating=1500,
            player2_rating=1500,
            score1=1.0,
            score2=0.0,
        )

        assert update1.rating_after > update1.rating_before
        assert update2.rating_after < update2.rating_before
        assert update1.player_id == "alice"
        assert update2.player_id == "bob"

    def test_get_win_probability(self):
        """Test win probability calculation."""
        system = ELORatingSystem()

        prob = system.get_win_probability(1600, 1400)
        assert 0.7 < prob < 0.8

    def test_get_rating_needed_for_probability(self):
        """Test calculating required rating for probability."""
        system = ELORatingSystem()

        # Should need higher rating for higher win probability
        rating_50 = system.get_rating_needed_for_probability(1500, 0.5)
        rating_75 = system.get_rating_needed_for_probability(1500, 0.75)

        assert abs(rating_50 - 1500) < 1
        assert rating_75 > rating_50

    def test_simulate_rating_change(self):
        """Test simulating rating change without recording."""
        system = ELORatingSystem()

        change = system.simulate_rating_change(
            your_rating=1500,
            opponent_rating=1500,
            outcome=MatchOutcome.WIN,
        )

        assert change > 0
        # Should not be recorded
        assert len(system.get_update_history()) == 0

    def test_update_history(self):
        """Test update history tracking."""
        system = ELORatingSystem()

        system.process_match("p1", "p2", 1500, 1500, 1.0, 0.0)
        system.process_match("p1", "p3", 1516, 1500, 0.0, 1.0)

        history = system.get_update_history()
        assert len(history) == 4  # 2 updates per match

        p1_history = system.get_update_history("p1")
        assert len(p1_history) == 2


class TestRatingUpdate:
    """Tests for RatingUpdate dataclass."""

    def test_initialization(self):
        """Test RatingUpdate initialization."""
        update = RatingUpdate(
            player_id="test",
            rating_before=1500,
            rating_after=1516,
            rating_change=16,
            expected_score=0.5,
            actual_score=1.0,
            k_factor=32,
            opponent_rating=1500,
        )

        assert update.player_id == "test"
        assert update.rating_change == 16

    def test_to_dict(self):
        """Test serialization."""
        update = RatingUpdate(
            player_id="test",
            rating_before=1500,
            rating_after=1516,
            rating_change=16,
            expected_score=0.5,
            actual_score=1.0,
            k_factor=32,
            opponent_rating=1500,
        )

        data = update.to_dict()
        assert data["player_id"] == "test"
        assert data["rating_change"] == 16


class TestRatingLeaderboard:
    """Tests for RatingLeaderboard class."""

    def test_initialization(self):
        """Test empty leaderboard."""
        leaderboard = RatingLeaderboard()
        assert len(leaderboard) == 0

    def test_update_and_get_rating(self):
        """Test adding and retrieving ratings."""
        leaderboard = RatingLeaderboard()
        leaderboard.update("alice", 1600)

        assert leaderboard.get_rating("alice") == 1600
        assert leaderboard.get_rating("bob") is None

    def test_get_rank(self):
        """Test rank calculation."""
        leaderboard = RatingLeaderboard()
        leaderboard.update("alice", 1600)
        leaderboard.update("bob", 1500)
        leaderboard.update("carol", 1700)

        assert leaderboard.get_rank("carol") == 1
        assert leaderboard.get_rank("alice") == 2
        assert leaderboard.get_rank("bob") == 3

    def test_get_top(self):
        """Test getting top players."""
        leaderboard = RatingLeaderboard()
        for i, name in enumerate(["a", "b", "c", "d", "e"]):
            leaderboard.update(name, 1500 + i * 100)

        top3 = leaderboard.get_top(3)
        assert len(top3) == 3
        assert top3[0][0] == "e"  # Highest rating
        assert top3[0][2] == 1  # Rank 1

    def test_get_players_in_range(self):
        """Test getting players in rating range."""
        leaderboard = RatingLeaderboard()
        leaderboard.update("alice", 1400)
        leaderboard.update("bob", 1500)
        leaderboard.update("carol", 1600)

        in_range = leaderboard.get_players_in_range(1450, 1550)
        assert len(in_range) == 1
        assert in_range[0][0] == "bob"

    def test_remove(self):
        """Test removing player from leaderboard."""
        leaderboard = RatingLeaderboard()
        leaderboard.update("alice", 1600)
        leaderboard.remove("alice")

        assert leaderboard.get_rating("alice") is None
        assert len(leaderboard) == 0
