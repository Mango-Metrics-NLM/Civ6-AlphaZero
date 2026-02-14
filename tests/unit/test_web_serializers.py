"""Tests for web serialization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.core.civ6_actions import MicroCivActionSpace
from alpha_zero.web.serializers import (
    decode_action_readable,
    serialize_board_for_template,
    serialize_confidence,
    serialize_move_history,
    serialize_turn_result,
)


@pytest.fixture
def action_space():
    """Create action space for testing."""
    return MicroCivActionSpace(board_size=8)


class TestDecodeActionReadable:
    """Tests for human-readable action decoding."""

    def test_end_turn(self, action_space):
        label = decode_action_readable(0, action_space)
        assert label == "End Turn"

    def test_move_action(self, action_space):
        # Action 1 = move unit 0 direction 0
        label = decode_action_readable(1, action_space)
        assert "Move" in label
        assert "Unit 0" in label

    def test_attack_action(self, action_space):
        idx = action_space.attack_offset
        label = decode_action_readable(idx, action_space)
        assert "Attack" in label

    def test_skip_action(self, action_space):
        idx = action_space.skip_offset
        label = decode_action_readable(idx, action_space)
        assert "Skip" in label

    def test_found_city_action(self, action_space):
        idx = action_space.found_city_offset
        label = decode_action_readable(idx, action_space)
        assert "Found City" in label

    def test_produce_action(self, action_space):
        idx = action_space.produce_offset
        label = decode_action_readable(idx, action_space)
        assert "Produce" in label

    def test_out_of_range_action_still_decodes(self, action_space):
        # Action space decode doesn't bounds-check, so large indices still decode
        label = decode_action_readable(99999, action_space)
        assert isinstance(label, str)
        assert len(label) > 0


class TestSerializeTurnResult:
    """Tests for turn result serialization."""

    def _make_turn_result(self, **kwargs):
        from alpha_zero.human_interface.game_controller import TurnResult

        defaults = {
            "action": 0,
            "player": 0,
            "is_human": True,
            "reward": 0.0,
            "game_over": False,
        }
        defaults.update(kwargs)
        return TurnResult(**defaults)

    def test_basic_serialization(self, action_space):
        result = self._make_turn_result(action=0, player=0, is_human=True)
        data = serialize_turn_result(result, action_space)
        assert data["action"] == 0
        assert data["action_label"] == "End Turn"
        assert data["player"] == 0
        assert data["is_human"] is True
        assert "timestamp" in data

    def test_ai_evaluation_included(self, action_space):
        result = self._make_turn_result(
            is_human=False,
            ai_evaluation=0.5,
        )
        data = serialize_turn_result(result, action_space)
        assert "ai_evaluation" in data
        assert data["ai_evaluation"] == 0.5

    def test_ai_evaluation_absent(self, action_space):
        result = self._make_turn_result(is_human=True)
        data = serialize_turn_result(result, action_space)
        assert "ai_evaluation" not in data

    def test_move_probabilities_top_moves(self, action_space):
        probs = np.zeros(action_space.total_size)
        probs[0] = 0.5
        probs[1] = 0.3
        probs[2] = 0.2
        result = self._make_turn_result(
            is_human=False,
            move_probabilities=probs,
        )
        data = serialize_turn_result(result, action_space)
        assert "top_moves" in data
        assert len(data["top_moves"]) > 0
        assert data["top_moves"][0]["probability"] == 0.5


class TestSerializeMoveHistory:
    """Tests for move history serialization."""

    def test_empty_history(self, action_space):
        result = serialize_move_history([], action_space)
        assert result == []

    def test_history_with_entries(self, action_space):
        from alpha_zero.human_interface.game_controller import TurnResult

        history = [
            TurnResult(action=0, player=0, is_human=True),
            TurnResult(action=1, player=1, is_human=False),
        ]
        result = serialize_move_history(history, action_space)
        assert len(result) == 2
        assert result[0]["is_human"] is True
        assert result[1]["is_human"] is False


class TestSerializeConfidence:
    """Tests for confidence data serialization."""

    def _make_turn_result(self, **kwargs):
        from alpha_zero.human_interface.game_controller import TurnResult

        defaults = {
            "action": 0,
            "player": 0,
            "is_human": False,
            "reward": 0.0,
            "thinking_time": 1.5,
        }
        defaults.update(kwargs)
        return TurnResult(**defaults)

    def test_with_evaluation(self, action_space):
        result = self._make_turn_result(ai_evaluation=0.5)
        data = serialize_confidence(result, action_space)
        assert data["has_evaluation"] is True
        assert data["win_probability"] == 0.75  # (0.5 + 1) / 2
        assert data["win_pct"] == 75.0

    def test_without_evaluation(self, action_space):
        result = self._make_turn_result()
        data = serialize_confidence(result, action_space)
        assert data["has_evaluation"] is False
        assert data["win_probability"] is None

    def test_negative_evaluation(self, action_space):
        result = self._make_turn_result(ai_evaluation=-1.0)
        data = serialize_confidence(result, action_space)
        assert data["win_probability"] == 0.0

    def test_positive_evaluation(self, action_space):
        result = self._make_turn_result(ai_evaluation=1.0)
        data = serialize_confidence(result, action_space)
        assert data["win_probability"] == 1.0

    def test_with_move_probabilities(self, action_space):
        probs = np.zeros(action_space.total_size)
        probs[0] = 0.8
        probs[5] = 0.2
        result = self._make_turn_result(
            ai_evaluation=0.3,
            move_probabilities=probs,
        )
        data = serialize_confidence(result, action_space)
        assert len(data["top_moves"]) == 2
        assert data["top_moves"][0]["pct"] == 80.0


class TestSerializeBoardForTemplate:
    """Tests for template-ready board serialization."""

    def test_board_structure(self):
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.web.adapter import MicroCivGameAdapter

        env = MicroCivEnv(board_size=8, max_turns=50)
        adapter = MicroCivGameAdapter(env)
        adapter.reset(seed=42)
        action_space = MicroCivActionSpace(board_size=8)

        valid_actions = adapter.get_valid_actions()

        data = serialize_board_for_template(
            adapter, valid_actions, action_space, human_player=0,
        )
        assert "grid" in data
        assert len(data["grid"]) == 8
        assert len(data["grid"][0]) == 8
        assert data["board_size"] == 8
        assert "human_player" in data

    def test_cells_have_metadata(self):
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.web.adapter import MicroCivGameAdapter

        env = MicroCivEnv(board_size=8, max_turns=50)
        adapter = MicroCivGameAdapter(env)
        adapter.reset(seed=42)
        action_space = MicroCivActionSpace(board_size=8)

        data = serialize_board_for_template(
            adapter, [], action_space, human_player=0,
        )
        cell = data["grid"][0][0]
        assert "terrain" in cell
        assert "units" in cell
        assert "has_own_unit" in cell
        assert "has_enemy_unit" in cell
