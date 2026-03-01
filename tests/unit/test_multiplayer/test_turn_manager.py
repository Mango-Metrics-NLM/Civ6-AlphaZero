"""Tests for multiplayer turn manager."""

from __future__ import annotations

import pytest

from alpha_zero.multiplayer.config import TurnManagerConfig
from alpha_zero.multiplayer.turn_manager import TurnManager


class TestTurnManagerSequential:
    """Tests for sequential turn mode."""

    @pytest.fixture
    def manager(self):
        config = TurnManagerConfig(mode="sequential")
        return TurnManager(config, num_players=4)

    def test_initial_state(self, manager):
        assert manager.num_players == 4
        assert manager.turn_number == 0
        assert manager.mode == "sequential"

    def test_begin_turn_sequential(self, manager):
        """Sequential mode: only one player active per turn."""
        state = manager.begin_turn()
        assert len(state.active_players) == 1
        assert state.active_players[0] == 0  # First player

    def test_turn_rotation(self, manager):
        """Players rotate through sequential turns."""
        players_seen = []
        for _ in range(8):
            state = manager.begin_turn()
            players_seen.append(state.active_players[0])
            manager.end_turn()

        # Should see 0,1,2,3,0,1,2,3
        assert players_seen == [0, 1, 2, 3, 0, 1, 2, 3]

    def test_turn_number_increments(self, manager):
        for i in range(3):
            manager.begin_turn()
            assert manager.turn_number == i + 1
            manager.end_turn()

    def test_submit_action(self, manager):
        manager.begin_turn()
        result = manager.submit_action(0, action=42)
        assert result is True

    def test_submit_action_inactive_player(self, manager):
        """Inactive player can't submit."""
        manager.begin_turn()  # Player 0 active
        result = manager.submit_action(1, action=42)
        assert result is False

    def test_all_actions_submitted(self, manager):
        manager.begin_turn()
        assert not manager.all_actions_submitted()
        manager.submit_action(0, action=0)
        assert manager.all_actions_submitted()


class TestTurnManagerSimultaneous:
    """Tests for simultaneous turn mode."""

    @pytest.fixture
    def manager(self):
        config = TurnManagerConfig(mode="simultaneous")
        return TurnManager(config, num_players=3)

    def test_all_players_active(self, manager):
        """Simultaneous mode: all players active."""
        state = manager.begin_turn()
        assert len(state.active_players) == 3
        assert set(state.active_players) == {0, 1, 2}

    def test_all_must_submit(self, manager):
        manager.begin_turn()
        assert not manager.all_actions_submitted()
        manager.submit_action(0, 10)
        assert not manager.all_actions_submitted()
        manager.submit_action(1, 20)
        assert not manager.all_actions_submitted()
        manager.submit_action(2, 30)
        assert manager.all_actions_submitted()


class TestTurnManagerDynamic:
    """Tests for dynamic (hybrid) turn mode."""

    @pytest.fixture
    def manager(self):
        config = TurnManagerConfig(mode="dynamic")
        return TurnManager(config, num_players=3)

    def test_starts_simultaneous(self, manager):
        """Dynamic mode starts as simultaneous (peace)."""
        state = manager.begin_turn()
        assert len(state.active_players) == 3

    def test_war_switches_to_sequential(self, manager):
        """War causes transition to sequential."""
        manager.declare_war(0, 1)
        assert manager.mode == "sequential"

        state = manager.begin_turn()
        assert len(state.active_players) == 1

    def test_peace_switches_back(self, manager):
        """Peace restores simultaneous."""
        manager.declare_war(0, 1)
        assert manager.mode == "sequential"

        manager.make_peace(0, 1)
        assert manager.mode == "simultaneous"

        state = manager.begin_turn()
        assert len(state.active_players) == 3

    def test_multiple_wars(self, manager):
        """Multiple wars keep sequential mode."""
        manager.declare_war(0, 1)
        manager.declare_war(1, 2)
        assert manager.mode == "sequential"

        # Resolving one war doesn't change mode
        manager.make_peace(0, 1)
        assert manager.mode == "sequential"

        # Resolving all wars restores simultaneous
        manager.make_peace(1, 2)
        assert manager.mode == "simultaneous"

    def test_update_from_info(self, manager):
        """War state can be updated from info dict."""
        info = {"at_war": {(0, 1): True}}
        manager.update_from_info(info)
        assert manager.mode == "sequential"


class TestTurnManagerElimination:
    """Tests for player elimination."""

    @pytest.fixture
    def manager(self):
        config = TurnManagerConfig(mode="sequential")
        return TurnManager(config, num_players=4)

    def test_eliminate_player(self, manager):
        manager.eliminate_player(2)
        assert manager.active_player_count == 3

    def test_eliminated_skipped_in_rotation(self, manager):
        """Eliminated players are skipped in sequential rotation."""
        manager.eliminate_player(1)
        players_seen = []
        for _ in range(6):
            state = manager.begin_turn()
            players_seen.append(state.active_players[0])
            manager.end_turn()

        assert 1 not in players_seen
        assert players_seen == [0, 2, 3, 0, 2, 3]

    def test_double_elimination_idempotent(self, manager):
        manager.eliminate_player(0)
        manager.eliminate_player(0)
        assert manager.active_player_count == 3

    def test_eliminate_all_but_one(self, manager):
        manager.eliminate_player(0)
        manager.eliminate_player(1)
        manager.eliminate_player(2)
        assert manager.active_player_count == 1
        state = manager.begin_turn()
        assert state.active_players == [3]


class TestTurnManagerReset:
    """Tests for turn manager reset."""

    def test_reset_clears_state(self):
        config = TurnManagerConfig(mode="dynamic")
        manager = TurnManager(config, num_players=3)

        manager.begin_turn()
        manager.end_turn()
        manager.declare_war(0, 1)
        manager.eliminate_player(2)

        manager.reset()

        assert manager.turn_number == 0
        assert manager.active_player_count == 3
        assert manager.mode == "dynamic"

    def test_get_state_summary(self):
        config = TurnManagerConfig(mode="sequential")
        manager = TurnManager(config, num_players=3)

        manager.begin_turn()
        summary = manager.get_state_summary()

        assert summary["turn_number"] == 1
        assert summary["current_mode"] == "sequential"
        assert len(summary["active_players"]) == 3
        assert len(summary["eliminated"]) == 0


class TestTurnManagerTimeout:
    """Tests for turn timeout."""

    def test_no_timeout_by_default(self):
        config = TurnManagerConfig(mode="sequential", turn_timeout_seconds=0.0)
        manager = TurnManager(config, num_players=2)
        manager.begin_turn()
        assert not manager.is_turn_timed_out()

    def test_timeout_not_expired(self):
        config = TurnManagerConfig(mode="sequential", turn_timeout_seconds=100.0)
        manager = TurnManager(config, num_players=2)
        manager.begin_turn()
        assert not manager.is_turn_timed_out()

    def test_no_turn_no_timeout(self):
        config = TurnManagerConfig(mode="sequential", turn_timeout_seconds=1.0)
        manager = TurnManager(config, num_players=2)
        assert not manager.is_turn_timed_out()
