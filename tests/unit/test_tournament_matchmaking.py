"""Unit tests for matchmaking system.

Tests cover:
- Queue management
- Match finding
- Rating range expansion
- Opponent selection
"""

import pytest

from alpha_zero.tournament.config import (
    MatchmakingConfig,
    PlayerStats,
    TournamentConfig,
)
from alpha_zero.tournament.matchmaking import (
    MatchMaker,
    OpponentSelector,
    PendingMatch,
    QueueEntry,
)


class TestQueueEntry:
    """Tests for QueueEntry class."""

    def test_initialization(self):
        """Test queue entry initialization."""
        player = PlayerStats(player_id="test", rating=1500)
        entry = QueueEntry(player=player)

        assert entry.player.player_id == "test"
        assert entry.rating_range == 200.0
        assert entry.priority == 0

    def test_wait_time(self):
        """Test wait time calculation."""
        player = PlayerStats(player_id="test")
        entry = QueueEntry(player=player)

        # Wait time should be small right after creation
        assert entry.wait_time >= 0
        assert entry.wait_time < 1.0

    def test_compatibility_rating_range(self):
        """Test compatibility based on rating range."""
        config = MatchmakingConfig(rating_range_initial=200)

        p1 = PlayerStats(player_id="p1", rating=1500)
        p2 = PlayerStats(player_id="p2", rating=1600)
        p3 = PlayerStats(player_id="p3", rating=1800)

        e1 = QueueEntry(player=p1, rating_range=200)
        e2 = QueueEntry(player=p2, rating_range=200)
        e3 = QueueEntry(player=p3, rating_range=200)

        assert e1.is_compatible_with(e2, config)  # 100 point diff
        assert not e1.is_compatible_with(e3, config)  # 300 point diff

    def test_compatibility_recent_opponents(self):
        """Test avoiding recent opponents."""
        config = MatchmakingConfig()

        p1 = PlayerStats(player_id="p1", rating=1500)
        p2 = PlayerStats(player_id="p2", rating=1500)

        e1 = QueueEntry(player=p1, recent_opponents=["p2"])
        e2 = QueueEntry(player=p2)

        assert not e1.is_compatible_with(e2, config)

    def test_expand_range(self):
        """Test rating range expansion."""
        config = MatchmakingConfig(
            rating_range_initial=200,
            rating_range_max=500,
            rating_range_expand_rate=10,
        )

        player = PlayerStats(player_id="test")
        entry = QueueEntry(player=player, rating_range=200)

        # Manually set joined_at to simulate waiting
        import time

        entry.joined_at = time.time() - 10  # 10 seconds ago

        entry.expand_range(config)
        assert entry.rating_range > 200

    def test_to_dict(self):
        """Test serialization."""
        player = PlayerStats(player_id="test", rating=1500)
        entry = QueueEntry(player=player)

        data = entry.to_dict()
        assert data["player_id"] == "test"
        assert data["rating"] == 1500


class TestPendingMatch:
    """Tests for PendingMatch class."""

    def test_initialization(self):
        """Test pending match initialization."""
        match = PendingMatch()

        assert match.match_id is not None
        assert match.status == "pending"

    def test_is_ready(self):
        """Test readiness check."""
        match = PendingMatch()
        assert not match.is_ready

        match.player1 = PlayerStats(player_id="p1")
        assert not match.is_ready

        match.player2 = PlayerStats(player_id="p2")
        assert match.is_ready

    def test_is_expired(self):
        """Test expiration check."""
        import time

        match = PendingMatch(expires_at=time.time() - 1)
        assert match.is_expired

        match2 = PendingMatch(expires_at=time.time() + 60)
        assert not match2.is_expired


class TestMatchMaker:
    """Tests for MatchMaker class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TournamentConfig()

    @pytest.fixture
    def matchmaker(self, config):
        """Create matchmaker instance."""
        return MatchMaker(config)

    def test_add_to_queue(self, matchmaker):
        """Test adding player to queue."""
        player = PlayerStats(player_id="test", rating=1500)
        entry = matchmaker.add_to_queue(player)

        assert entry is not None
        assert matchmaker.is_in_queue("test")
        assert matchmaker.queue_size == 1

    def test_remove_from_queue(self, matchmaker):
        """Test removing player from queue."""
        player = PlayerStats(player_id="test")
        matchmaker.add_to_queue(player)

        result = matchmaker.remove_from_queue("test")
        assert result is True
        assert not matchmaker.is_in_queue("test")

    def test_remove_nonexistent(self, matchmaker):
        """Test removing player not in queue."""
        result = matchmaker.remove_from_queue("nonexistent")
        assert result is False

    def test_get_queue_position(self, matchmaker):
        """Test queue position."""
        for i in range(3):
            player = PlayerStats(player_id=f"p{i}", rating=1500)
            matchmaker.add_to_queue(player)

        pos = matchmaker.get_queue_position("p1")
        assert pos is not None
        assert 1 <= pos <= 3

    def test_find_match_no_opponents(self, matchmaker):
        """Test finding match with no opponents."""
        player = PlayerStats(player_id="test")
        matchmaker.add_to_queue(player)

        match = matchmaker.find_match("test")
        assert match is None

    def test_find_match_success(self, matchmaker):
        """Test successful match finding."""
        p1 = PlayerStats(player_id="p1", rating=1500)
        p2 = PlayerStats(player_id="p2", rating=1550)

        matchmaker.add_to_queue(p1)
        matchmaker.add_to_queue(p2)

        match = matchmaker.find_match("p1")

        assert match is not None
        assert match.is_ready
        assert not matchmaker.is_in_queue("p1")
        assert not matchmaker.is_in_queue("p2")

    def test_find_match_prefers_closer_rating(self, matchmaker):
        """Test that closer ratings are preferred."""
        p1 = PlayerStats(player_id="p1", rating=1500)
        p2 = PlayerStats(player_id="p2", rating=1510)  # Close
        p3 = PlayerStats(player_id="p3", rating=1600)  # Far

        matchmaker.add_to_queue(p1)
        matchmaker.add_to_queue(p2)
        matchmaker.add_to_queue(p3)

        match = matchmaker.find_match("p1")

        assert match is not None
        # Should match with p2 (closer rating)
        other_player = (
            match.player2.player_id if match.player1.player_id == "p1" else match.player1.player_id
        )
        assert other_player == "p2"

    def test_process_queue(self, matchmaker):
        """Test processing entire queue."""
        for i in range(4):
            player = PlayerStats(player_id=f"p{i}", rating=1500 + i * 10)
            matchmaker.add_to_queue(player)

        matches = matchmaker.process_queue()

        assert len(matches) == 2  # 4 players = 2 matches
        assert matchmaker.queue_size == 0

    def test_confirm_match(self, matchmaker):
        """Test confirming a match."""
        p1 = PlayerStats(player_id="p1")
        p2 = PlayerStats(player_id="p2")

        matchmaker.add_to_queue(p1)
        matchmaker.add_to_queue(p2)

        match = matchmaker.find_match("p1")
        result = matchmaker.confirm_match(match.match_id)

        assert result is True
        assert match.status == "confirmed"

    def test_cancel_match_requeue(self, matchmaker):
        """Test cancelling match with requeue."""
        p1 = PlayerStats(player_id="p1")
        p2 = PlayerStats(player_id="p2")

        matchmaker.add_to_queue(p1)
        matchmaker.add_to_queue(p2)

        match = matchmaker.find_match("p1")
        matchmaker.cancel_match(match.match_id, requeue=True)

        assert matchmaker.is_in_queue("p1")
        assert matchmaker.is_in_queue("p2")

    def test_cancel_match_no_requeue(self, matchmaker):
        """Test cancelling match without requeue."""
        p1 = PlayerStats(player_id="p1")
        p2 = PlayerStats(player_id="p2")

        matchmaker.add_to_queue(p1)
        matchmaker.add_to_queue(p2)

        match = matchmaker.find_match("p1")
        matchmaker.cancel_match(match.match_id, requeue=False)

        assert not matchmaker.is_in_queue("p1")
        assert not matchmaker.is_in_queue("p2")

    def test_get_queue_stats_empty(self, matchmaker):
        """Test queue stats when empty."""
        stats = matchmaker.get_queue_stats()

        assert stats["queue_size"] == 0
        assert stats["pending_matches"] == 0

    def test_get_queue_stats(self, matchmaker):
        """Test queue stats with players."""
        for i in range(3):
            player = PlayerStats(player_id=f"p{i}", rating=1400 + i * 100)
            matchmaker.add_to_queue(player)

        stats = matchmaker.get_queue_stats()

        assert stats["queue_size"] == 3
        assert stats["min_rating"] == 1400
        assert stats["max_rating"] == 1600
        assert stats["avg_rating"] == 1500

    def test_callback_on_match_found(self, matchmaker):
        """Test match found callback."""
        found_matches = []

        def on_match(match):
            found_matches.append(match)

        matchmaker.set_on_match_found(on_match)

        p1 = PlayerStats(player_id="p1")
        p2 = PlayerStats(player_id="p2")
        matchmaker.add_to_queue(p1)
        matchmaker.add_to_queue(p2)
        matchmaker.find_match("p1")

        assert len(found_matches) == 1


class TestOpponentSelector:
    """Tests for OpponentSelector class."""

    @pytest.fixture
    def player_pool(self):
        """Create test player pool."""
        return [
            PlayerStats(player_id=f"p{i}", rating=1400 + i * 50, tier="Silver") for i in range(10)
        ]

    @pytest.fixture
    def selector(self, player_pool):
        """Create opponent selector."""
        selector = OpponentSelector()
        selector.set_player_pool(player_pool)
        return selector

    def test_select_by_rating_proximity(self, selector):
        """Test selecting by rating proximity."""
        opponents = selector.select_by_rating_proximity(target_rating=1500, num_opponents=3)

        assert len(opponents) == 3
        # Should be sorted by closeness to 1500
        for opp in opponents:
            assert abs(opp.rating - 1500) <= 100

    def test_select_by_rating_proximity_with_exclusions(self, selector):
        """Test excluding specific players."""
        opponents = selector.select_by_rating_proximity(
            target_rating=1500,
            num_opponents=3,
            exclude_ids={"p2", "p3"},
        )

        for opp in opponents:
            assert opp.player_id not in {"p2", "p3"}

    def test_select_by_tier(self, selector, player_pool):
        """Test selecting by tier."""
        # All players in pool are Silver tier
        opponents = selector.select_by_tier("Silver", num_opponents=5)
        assert len(opponents) == 5

        # Non-existent tier
        opponents = selector.select_by_tier("Diamond", num_opponents=5)
        assert len(opponents) == 0

    def test_select_diverse(self, selector):
        """Test selecting diverse opponents."""
        opponents = selector.select_diverse(num_opponents=5)

        assert len(opponents) == 5
        # Should span different rating ranges
        ratings = [opp.rating for opp in opponents]
        assert max(ratings) - min(ratings) > 100

    def test_select_challenging(self, selector):
        """Test selecting challenging opponents."""
        opponents = selector.select_challenging(
            player_rating=1450,
            challenge_factor=1.1,
            num_opponents=3,
        )

        for opp in opponents:
            # Should be at least equal rating
            assert opp.rating >= 1450

    def test_empty_pool(self):
        """Test with empty player pool."""
        selector = OpponentSelector()

        opponents = selector.select_by_rating_proximity(1500, num_opponents=3)
        assert len(opponents) == 0
