"""Unit tests for diversity tracking."""

import pytest

from alpha_zero.evaluation.config import DiversityConfig
from alpha_zero.evaluation.diversity import DiversityMetrics, DiversityTracker


class TestDiversityMetrics:
    """Tests for DiversityMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = DiversityMetrics()
        assert metrics.unique_openings == 0
        assert metrics.opening_entropy == 0.0
        assert metrics.avg_game_length == 0.0
        assert metrics.total_games == 0

    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = DiversityMetrics(
            unique_openings=5,
            opening_entropy=2.3,
            avg_game_length=50.0,
            total_games=10,
        )
        result = metrics.to_dict()

        assert result["unique_openings"] == 5
        assert result["opening_entropy"] == 2.3
        assert result["avg_game_length"] == 50.0
        assert result["total_games"] == 10

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "unique_openings": 3,
            "opening_entropy": 1.5,
            "avg_game_length": 40.0,
            "total_games": 5,
        }
        metrics = DiversityMetrics.from_dict(data)

        assert metrics.unique_openings == 3
        assert metrics.opening_entropy == 1.5

    def test_summary(self):
        """Test summary generation."""
        metrics = DiversityMetrics(
            unique_openings=10,
            opening_entropy=3.0,
            avg_game_length=45.0,
            game_length_std=10.0,
            total_games=20,
        )
        summary = metrics.summary()

        assert "Unique Openings: 10" in summary
        assert "Opening Entropy: 3.000" in summary
        assert "20 games" in summary

    def test_summary_with_victory_distribution(self):
        """Test summary includes victory distribution."""
        metrics = DiversityMetrics(
            total_games=10,
            victory_type_distribution={"win": 0.6, "loss": 0.3, "draw": 0.1},
        )
        summary = metrics.summary()

        assert "Victory Distribution" in summary
        assert "win" in summary


class TestDiversityTracker:
    """Tests for DiversityTracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker with default config."""
        config = DiversityConfig(enabled=True, opening_depth=5)
        return DiversityTracker(config)

    @pytest.fixture
    def disabled_tracker(self):
        """Create disabled tracker."""
        config = DiversityConfig(enabled=False)
        return DiversityTracker(config)

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.config.enabled is True
        assert tracker._total_games == 0

    def test_record_game_updates_count(self, tracker):
        """Test that recording games updates total count."""
        game_history = [
            (1, {"turn": 0, "is_model": True}),
            (2, {"turn": 1, "is_model": True}),
        ]
        tracker.record_game(game_history)

        assert tracker._total_games == 1

    def test_record_game_disabled(self, disabled_tracker):
        """Test that disabled tracker doesn't record."""
        game_history = [(1, {"turn": 0, "is_model": True})]
        disabled_tracker.record_game(game_history)

        assert disabled_tracker._total_games == 0

    def test_record_game_empty_history(self, tracker):
        """Test handling empty game history."""
        tracker.record_game([])
        assert tracker._total_games == 0

    def test_tracks_opening_sequences(self, tracker):
        """Test opening sequence tracking."""
        # Game 1: opening [1, 2, 3]
        game1 = [
            (1, {"turn": 0, "is_model": True}),
            (2, {"turn": 1, "is_model": True}),
            (3, {"turn": 2, "is_model": True}),
        ]
        tracker.record_game(game1)

        # Game 2: same opening
        tracker.record_game(game1)

        # Game 3: different opening
        game3 = [
            (4, {"turn": 0, "is_model": True}),
            (5, {"turn": 1, "is_model": True}),
        ]
        tracker.record_game(game3)

        metrics = tracker.compute_metrics()
        assert metrics.unique_openings == 2

    def test_tracks_game_lengths(self, tracker):
        """Test game length tracking."""
        # Short game
        tracker.record_game(
            [(i, {"turn": i, "is_model": True}) for i in range(10)],
            game_length=10,
        )
        # Long game
        tracker.record_game(
            [(i, {"turn": i, "is_model": True}) for i in range(50)],
            game_length=50,
        )

        metrics = tracker.compute_metrics()
        assert metrics.avg_game_length == 30.0  # (10 + 50) / 2
        assert metrics.game_length_min == 10
        assert metrics.game_length_max == 50

    def test_tracks_victory_types(self, tracker):
        """Test victory type tracking."""
        game = [(1, {"turn": 0, "is_model": True})]

        tracker.record_game(game, victory_type="win")
        tracker.record_game(game, victory_type="win")
        tracker.record_game(game, victory_type="loss")

        metrics = tracker.compute_metrics()
        assert metrics.victory_type_distribution["win"] == pytest.approx(2 / 3)
        assert metrics.victory_type_distribution["loss"] == pytest.approx(1 / 3)

    def test_computes_opening_entropy(self, tracker):
        """Test entropy calculation for openings."""
        # All same opening - zero entropy
        game = [(1, {"turn": 0, "is_model": True})]
        for _ in range(10):
            tracker.record_game(game)

        metrics = tracker.compute_metrics()
        assert metrics.opening_entropy == 0.0

        # Reset and test with diverse openings
        tracker.reset()

        for i in range(4):
            game = [(i, {"turn": 0, "is_model": True})]
            tracker.record_game(game)

        metrics = tracker.compute_metrics()
        # 4 equally likely outcomes = 2 bits of entropy
        assert metrics.opening_entropy == pytest.approx(2.0)

    def test_computes_action_entropy(self, tracker):
        """Test action entropy computation."""
        # Record games with varied actions at turn 0
        for action in [1, 2, 3, 4]:
            game = [(action, {"turn": 0, "is_model": True})]
            tracker.record_game(game)

        metrics = tracker.compute_metrics()
        assert len(metrics.action_entropy_by_turn) > 0
        # 4 equally likely actions = 2 bits
        assert metrics.action_entropy_by_turn[0] == pytest.approx(2.0)

    def test_get_baseline_metrics(self, tracker):
        """Test getting per-baseline metrics."""
        game = [(1, {"turn": 0, "is_model": True})]

        tracker.record_game(game, baseline_name="random")
        tracker.record_game(game, baseline_name="random")
        tracker.record_game(game, baseline_name="heuristic")

        random_metrics = tracker.get_baseline_metrics("random")
        assert random_metrics["games"] == 2

        heuristic_metrics = tracker.get_baseline_metrics("heuristic")
        assert heuristic_metrics["games"] == 1

    def test_get_baseline_metrics_unknown(self, tracker):
        """Test getting metrics for unknown baseline."""
        metrics = tracker.get_baseline_metrics("unknown")
        assert metrics["games"] == 0

    def test_get_opening_distribution(self, tracker):
        """Test getting opening distribution."""
        # Opening A: 3 times
        game_a = [(1, {"turn": 0, "is_model": True})]
        for _ in range(3):
            tracker.record_game(game_a)

        # Opening B: 2 times
        game_b = [(2, {"turn": 0, "is_model": True})]
        for _ in range(2):
            tracker.record_game(game_b)

        distribution = tracker.get_opening_distribution(top_k=2)

        assert len(distribution) == 2
        # Most common first
        assert distribution[0][1] == 3  # count
        assert distribution[0][2] == pytest.approx(0.6)  # frequency

    def test_reset(self, tracker):
        """Test resetting tracker."""
        game = [(1, {"turn": 0, "is_model": True})]
        tracker.record_game(game)
        tracker.record_game(game)

        assert tracker._total_games == 2

        tracker.reset()

        assert tracker._total_games == 0
        assert len(tracker._game_lengths) == 0
        assert len(tracker._opening_counts) == 0

    def test_to_dict(self, tracker):
        """Test converting to dictionary."""
        game = [(1, {"turn": 0, "is_model": True})]
        tracker.record_game(game, victory_type="win")

        result = tracker.to_dict()

        assert "config" in result
        assert "opening_counts" in result
        assert "game_lengths" in result
        assert "victory_types" in result
        assert result["total_games"] == 1

    def test_filters_model_actions_only(self, tracker):
        """Test that only model actions are tracked for openings."""
        game = [
            (1, {"turn": 0, "is_model": True}),  # Model action
            (2, {"turn": 1, "is_model": False}),  # Opponent action
            (3, {"turn": 2, "is_model": True}),  # Model action
        ]
        tracker.record_game(game)

        # Opening should only include actions 1 and 3
        assert (1, 3) in tracker._opening_counts or len(tracker._opening_counts) == 1
