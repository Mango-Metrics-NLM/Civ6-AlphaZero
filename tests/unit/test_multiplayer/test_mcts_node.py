"""Tests for N-player MCTS node."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.multiplayer.mcts_node import NPlayerMCTSNode


class TestNPlayerMCTSNode:
    """Tests for NPlayerMCTSNode."""

    def test_initial_state(self):
        """New node has correct initial values."""
        node = NPlayerMCTSNode(num_players=4)
        assert node.visit_count == 0
        assert node.num_players == 4
        assert not node.is_expanded()
        assert node.get_value(0) == 0.0

    def test_get_value_unvisited(self):
        """Unvisited node returns 0 for all players."""
        node = NPlayerMCTSNode(num_players=3)
        for p in range(3):
            assert node.get_value(p) == 0.0

    def test_get_all_values_unvisited(self):
        node = NPlayerMCTSNode(num_players=3)
        values = node.get_all_values()
        assert len(values) == 3
        assert all(v == 0.0 for v in values.values())

    def test_backpropagate_single(self):
        """Single backpropagation updates values correctly."""
        node = NPlayerMCTSNode(num_players=3)
        values = {0: 1.0, 1: -0.5, 2: -0.5}
        node.backpropagate(values)
        assert node.visit_count == 1
        assert node.get_value(0) == pytest.approx(1.0)
        assert node.get_value(1) == pytest.approx(-0.5)

    def test_backpropagate_multiple(self):
        """Multiple backpropagations compute correct mean."""
        node = NPlayerMCTSNode(num_players=2)
        node.backpropagate({0: 1.0, 1: -1.0})
        node.backpropagate({0: 0.5, 1: -0.5})
        assert node.visit_count == 2
        assert node.get_value(0) == pytest.approx(0.75)
        assert node.get_value(1) == pytest.approx(-0.75)

    def test_backpropagate_through_parent(self):
        """Values propagate up to parent nodes."""
        parent = NPlayerMCTSNode(num_players=2)
        child = NPlayerMCTSNode(num_players=2, parent=parent)
        child.backpropagate({0: 1.0, 1: -1.0})
        assert parent.visit_count == 1
        assert parent.get_value(0) == pytest.approx(1.0)
        assert child.visit_count == 1
        assert child.get_value(0) == pytest.approx(1.0)

    def test_backpropagate_paranoid(self):
        """Paranoid backprop distributes value correctly."""
        node = NPlayerMCTSNode(num_players=4)
        node.backpropagate_paranoid(root_player_value=1.0, root_player=0)
        assert node.get_value(0) == pytest.approx(1.0)
        # Opponents share negated value
        for p in range(1, 4):
            assert node.get_value(p) == pytest.approx(-1.0 / 3)

    def test_expand(self):
        """Expand creates correct children for valid actions."""
        node = NPlayerMCTSNode(num_players=2)
        policy = np.array([0.3, 0.2, 0.5, 0.0])
        action_mask = np.array([1, 0, 1, 0])

        node.expand(policy, action_mask, next_player=1)

        assert node.is_expanded()
        assert len(node.children) == 2
        assert 0 in node.children  # Valid action
        assert 2 in node.children  # Valid action
        assert 1 not in node.children  # Invalid
        assert 3 not in node.children  # Invalid

    def test_expand_normalizes_policy(self):
        """Policy is renormalized over valid actions."""
        node = NPlayerMCTSNode(num_players=2)
        policy = np.array([0.2, 0.0, 0.3, 0.0])
        mask = np.array([1, 0, 1, 0])

        node.expand(policy, mask, next_player=0)

        total_prior = sum(c.prior for c in node.children.values())
        assert total_prior == pytest.approx(1.0)

    def test_expand_sets_next_player(self):
        """Children have correct player_to_move."""
        node = NPlayerMCTSNode(num_players=3, player_to_move=0)
        policy = np.array([0.5, 0.5])
        mask = np.array([1, 1])
        node.expand(policy, mask, next_player=1)

        for child in node.children.values():
            assert child.player_to_move == 1

    def test_expand_no_valid_actions(self):
        """Expand with no valid actions doesn't crash."""
        node = NPlayerMCTSNode(num_players=2)
        policy = np.array([0.5, 0.5])
        mask = np.array([0, 0])
        node.expand(policy, mask, next_player=0)
        assert not node.is_expanded()

    def test_select_child_max_n(self):
        """Max-N selection picks child maximizing selecting player's value."""
        parent = NPlayerMCTSNode(num_players=2, player_to_move=0)
        policy = np.array([0.5, 0.5])
        mask = np.array([1, 1])
        parent.expand(policy, mask, next_player=1)

        # Give child 0 better value for player 0
        parent.children[0].visit_count = 10
        parent.children[0].value_sums = {0: 8.0, 1: -8.0}

        parent.children[1].visit_count = 10
        parent.children[1].value_sums = {0: 2.0, 1: -2.0}

        parent.visit_count = 20

        action, _child = parent.select_child(c_puct=1.5, selecting_player=0)
        assert action == 0  # Better for player 0

    def test_select_child_paranoid(self):
        """Paranoid selection: root player maximizes, opponents minimize."""
        parent = NPlayerMCTSNode(num_players=3, player_to_move=1)
        policy = np.array([0.5, 0.5])
        mask = np.array([1, 1])
        parent.expand(policy, mask, next_player=2)

        # Child 0: good for root player 0
        parent.children[0].visit_count = 10
        parent.children[0].value_sums = {0: 8.0}

        # Child 1: bad for root player 0
        parent.children[1].visit_count = 10
        parent.children[1].value_sums = {0: 2.0}

        parent.visit_count = 20

        # Opponent (player 1) should pick action that hurts root player 0
        action, _child = parent.select_child_paranoid(c_puct=0.0, root_player=0)
        assert action == 1  # Worse for player 0

    def test_get_action_probabilities_temperature_1(self):
        """Temperature=1 returns visit-proportional probabilities."""
        node = NPlayerMCTSNode(num_players=2)
        policy = np.array([0.5, 0.5, 0.0])
        mask = np.array([1, 1, 0])
        node.expand(policy, mask, next_player=1)

        node.children[0].visit_count = 30
        node.children[1].visit_count = 70

        probs = node.get_action_probabilities(action_space_size=3, temperature=1.0)
        assert probs[0] == pytest.approx(0.3)
        assert probs[1] == pytest.approx(0.7)
        assert probs[2] == pytest.approx(0.0)

    def test_get_action_probabilities_temperature_0(self):
        """Temperature=0 returns one-hot for most visited."""
        node = NPlayerMCTSNode(num_players=2)
        policy = np.array([0.5, 0.5])
        mask = np.array([1, 1])
        node.expand(policy, mask, next_player=1)

        node.children[0].visit_count = 30
        node.children[1].visit_count = 70

        probs = node.get_action_probabilities(action_space_size=2, temperature=0.0)
        assert probs[1] == pytest.approx(1.0)
        assert probs[0] == pytest.approx(0.0)

    def test_get_child_visit_counts(self):
        node = NPlayerMCTSNode(num_players=2)
        policy = np.array([0.3, 0.7])
        mask = np.array([1, 1])
        node.expand(policy, mask, next_player=0)
        node.children[0].visit_count = 5
        node.children[1].visit_count = 10

        counts = node.get_child_visit_counts()
        assert counts[0] == 5
        assert counts[1] == 10

    def test_deep_tree_backpropagation(self):
        """Values propagate correctly through a deep tree."""
        root = NPlayerMCTSNode(num_players=3)
        mid = NPlayerMCTSNode(num_players=3, parent=root)
        leaf = NPlayerMCTSNode(num_players=3, parent=mid)

        values = {0: 1.0, 1: 0.0, 2: -1.0}
        leaf.backpropagate(values)

        # All nodes should have same values (Max-N: no flipping)
        assert root.visit_count == 1
        assert root.get_value(0) == pytest.approx(1.0)
        assert mid.visit_count == 1
        assert mid.get_value(0) == pytest.approx(1.0)
        assert leaf.visit_count == 1
        assert leaf.get_value(0) == pytest.approx(1.0)

    def test_num_players_preserved_in_children(self):
        """Children inherit num_players from parent."""
        node = NPlayerMCTSNode(num_players=6)
        policy = np.array([1.0])
        mask = np.array([1])
        node.expand(policy, mask, next_player=3)

        assert node.children[0].num_players == 6
