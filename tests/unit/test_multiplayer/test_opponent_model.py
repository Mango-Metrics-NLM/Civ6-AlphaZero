"""Tests for opponent modeling."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.multiplayer.config import OpponentModelConfig
from alpha_zero.multiplayer.opponent_model import OpponentModel


class TestOpponentModel:
    """Tests for OpponentModel."""

    @pytest.fixture
    def config(self):
        return OpponentModelConfig(enabled=True, history_window=10, decay_rate=0.95)

    @pytest.fixture
    def model(self, config):
        return OpponentModel(config, num_players=4, action_space_size=10)

    @pytest.fixture
    def valid_actions(self):
        return np.ones(10, dtype=np.float32)

    def test_initial_prediction_uniform(self, model, valid_actions):
        """With no observations, prediction is uniform."""
        probs = model.predict_action_distribution(0, {}, valid_actions)
        assert probs.shape == (10,)
        assert probs.sum() == pytest.approx(1.0)
        # Should be approximately uniform
        expected = 1.0 / 10
        for p in probs:
            assert p == pytest.approx(expected, abs=0.01)

    def test_update_and_predict(self, model, valid_actions):
        """After observing actions, prediction biases toward them."""
        # Observe player 1 taking action 3 many times
        for _ in range(20):
            model.update(player_id=1, game_state={}, action=3)

        probs = model.predict_action_distribution(1, {}, valid_actions)
        # Action 3 should have highest probability
        assert probs[3] == max(probs)
        assert probs[3] > 1.0 / 10  # Higher than uniform

    def test_disabled_returns_uniform(self, valid_actions):
        """Disabled model always returns uniform distribution."""
        config = OpponentModelConfig(enabled=False)
        model = OpponentModel(config, num_players=2, action_space_size=10)
        model.update(0, {}, 3)  # Should be no-op
        probs = model.predict_action_distribution(0, {}, valid_actions)
        expected = 1.0 / 10
        for p in probs:
            assert p == pytest.approx(expected, abs=0.01)

    def test_action_mask_respected(self, model):
        """Prediction respects valid action mask."""
        mask = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        for _ in range(10):
            model.update(player_id=0, game_state={}, action=5)  # Invalid in mask

        probs = model.predict_action_distribution(0, {}, mask)
        assert probs[5] == pytest.approx(0.0)  # Masked out
        assert probs.sum() == pytest.approx(1.0)

    def test_decay_reduces_old_observations(self, model, valid_actions):
        """Decay reduces impact of old observations over time."""
        # Observe action 0 many times early
        for _ in range(20):
            model.update(player_id=0, game_state={}, action=0)

        # Then observe action 1 many times recently
        for _ in range(20):
            model.update(player_id=0, game_state={}, action=1)

        probs = model.predict_action_distribution(0, {}, valid_actions)
        # Recent action 1 should be more prominent than old action 0
        assert probs[1] > probs[0]

    def test_different_players_independent(self, model, valid_actions):
        """Each player has independent model."""
        model.update(player_id=0, game_state={}, action=0)
        model.update(player_id=1, game_state={}, action=5)

        probs_0 = model.predict_action_distribution(0, {}, valid_actions)
        probs_1 = model.predict_action_distribution(1, {}, valid_actions)

        assert probs_0[0] > probs_0[5]
        assert probs_1[5] > probs_1[0]

    def test_most_likely_action(self, model, valid_actions):
        """most_likely_action returns the most probable action."""
        for _ in range(20):
            model.update(player_id=0, game_state={}, action=7)

        best = model.get_most_likely_action(0, {}, valid_actions)
        assert best == 7

    def test_entropy_decreases_with_observations(self, model):
        """Entropy decreases as model becomes more confident."""
        initial_entropy = model.get_action_entropy(0)

        # Observe same action repeatedly
        for _ in range(50):
            model.update(player_id=0, game_state={}, action=3)

        final_entropy = model.get_action_entropy(0)
        assert final_entropy < initial_entropy

    def test_player_summary(self, model, valid_actions):
        """Summary contains expected fields."""
        for i in range(5):
            model.update(player_id=1, game_state={}, action=i % 3)

        summary = model.get_player_summary(1)
        assert summary["player_id"] == 1
        assert summary["total_observations"] == 5
        assert summary["recent_action_count"] == 5
        assert "action_entropy" in summary
        assert "confidence" in summary

    def test_reset_clears_state(self, model, valid_actions):
        """Reset clears all tracked data."""
        for _ in range(10):
            model.update(player_id=0, game_state={}, action=3)

        model.reset()

        probs = model.predict_action_distribution(0, {}, valid_actions)
        # Should be back to uniform
        expected = 1.0 / 10
        for p in probs:
            assert p == pytest.approx(expected, abs=0.01)

    def test_invalid_player_id(self, model, valid_actions):
        """Invalid player ID is handled gracefully."""
        model.update(player_id=-1, game_state={}, action=0)  # No crash
        model.update(player_id=99, game_state={}, action=0)  # No crash

    def test_invalid_action(self, model, valid_actions):
        """Invalid action index is handled gracefully."""
        model.update(player_id=0, game_state={}, action=-1)  # No crash
        model.update(player_id=0, game_state={}, action=999)  # No crash

    def test_confidence_grows_with_observations(self, model):
        """Confidence increases with more observations."""
        summary_before = model.get_player_summary(0)
        assert summary_before["confidence"] == 0.0

        for _ in range(5):
            model.update(player_id=0, game_state={}, action=0)

        summary_after = model.get_player_summary(0)
        assert summary_after["confidence"] > 0.0
        assert summary_after["confidence"] <= 1.0

    def test_history_window_limit(self, model):
        """Recent action buffer respects window size."""
        for i in range(20):
            model.update(player_id=0, game_state={}, action=i % 10)

        # Window is 10, so only last 10 actions should be in recent buffer
        summary = model.get_player_summary(0)
        assert summary["recent_action_count"] == model.config.history_window
