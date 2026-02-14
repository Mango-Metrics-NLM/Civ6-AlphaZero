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

    def test_city_ownership_flags(self):
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

        # Check that city ownership flags are booleans
        for row in data["grid"]:
            for cell in row:
                assert isinstance(cell["has_own_city"], bool)
                assert isinstance(cell["has_enemy_city"], bool)

    def test_valid_action_indices_returned(self):
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.web.adapter import MicroCivGameAdapter

        env = MicroCivEnv(board_size=8, max_turns=50)
        adapter = MicroCivGameAdapter(env)
        adapter.reset(seed=42)
        action_space = MicroCivActionSpace(board_size=8)

        valid_actions = adapter.get_valid_actions()[:5]

        data = serialize_board_for_template(
            adapter, valid_actions, action_space, human_player=0,
        )
        assert data["valid_action_indices"] == valid_actions

    def test_numpy_types_sanitized(self):
        """Board data with numpy types should be fully sanitized."""
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.web.adapter import MicroCivGameAdapter

        env = MicroCivEnv(board_size=8, max_turns=50)
        adapter = MicroCivGameAdapter(env)
        adapter.reset(seed=42)
        action_space = MicroCivActionSpace(board_size=8)

        data = serialize_board_for_template(
            adapter, [], action_space, human_player=0,
        )

        # board_size should be a native int, not numpy
        assert type(data["board_size"]) is int
        assert type(data["current_player"]) is int


class TestSanitizeNumpy:
    """Tests for the _sanitize_numpy helper."""

    def test_numpy_int(self):
        from alpha_zero.web.serializers import _sanitize_numpy
        result = _sanitize_numpy(np.int64(42))
        assert result == 42
        assert type(result) is int

    def test_numpy_float(self):
        from alpha_zero.web.serializers import _sanitize_numpy
        result = _sanitize_numpy(np.float32(3.14))
        assert isinstance(result, float)

    def test_numpy_bool(self):
        from alpha_zero.web.serializers import _sanitize_numpy
        result = _sanitize_numpy(np.bool_(True))
        assert result is True
        assert type(result) is bool

    def test_numpy_array(self):
        from alpha_zero.web.serializers import _sanitize_numpy
        result = _sanitize_numpy(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_nested_dict(self):
        from alpha_zero.web.serializers import _sanitize_numpy
        data = {"a": np.int64(1), "b": {"c": np.float32(2.0)}}
        result = _sanitize_numpy(data)
        assert result == {"a": 1, "b": {"c": 2.0}}
        assert type(result["a"]) is int

    def test_nested_list(self):
        from alpha_zero.web.serializers import _sanitize_numpy
        data = [np.int64(1), [np.float32(2.0)]]
        result = _sanitize_numpy(data)
        assert result == [1, [2.0]]

    def test_passthrough_native_types(self):
        from alpha_zero.web.serializers import _sanitize_numpy
        assert _sanitize_numpy("hello") == "hello"
        assert _sanitize_numpy(42) == 42
        assert _sanitize_numpy(None) is None


class TestSerializeTurnResultEdgeCases:
    """Edge case tests for serialize_turn_result."""

    @pytest.fixture
    def action_space(self):
        return MicroCivActionSpace(board_size=8)

    def _make_turn_result(self, **kwargs):
        from alpha_zero.human_interface.game_controller import TurnResult
        defaults = {
            "action": 0,
            "player": 0,
            "is_human": True,
            "reward": 0.0,
            "game_over": False,
            "winner": None,
            "state_before": None,
            "state_after": None,
            "thinking_time": 0.0,
            "timestamp": "2026-01-01T00:00:00",
            "ai_evaluation": None,
            "move_probabilities": None,
        }
        defaults.update(kwargs)
        return TurnResult(**defaults)

    def test_none_winner_serialized_as_none(self, action_space):
        result = self._make_turn_result(winner=None)
        data = serialize_turn_result(result, action_space)
        assert data["winner"] is None

    def test_integer_winner_serialized(self, action_space):
        result = self._make_turn_result(winner=1, game_over=True)
        data = serialize_turn_result(result, action_space)
        assert data["winner"] == 1
        assert type(data["winner"]) is int

    def test_ai_evaluation_included_when_present(self, action_space):
        result = self._make_turn_result(ai_evaluation=0.75)
        data = serialize_turn_result(result, action_space)
        assert "ai_evaluation" in data
        assert data["ai_evaluation"] == 0.75

    def test_ai_evaluation_excluded_when_none(self, action_space):
        result = self._make_turn_result(ai_evaluation=None)
        data = serialize_turn_result(result, action_space)
        assert "ai_evaluation" not in data

    def test_top_moves_included_when_probabilities_present(self, action_space):
        probs = np.zeros(action_space.total_size)
        probs[0] = 0.9
        probs[1] = 0.1
        result = self._make_turn_result(move_probabilities=probs)
        data = serialize_turn_result(result, action_space)
        assert "top_moves" in data
        assert len(data["top_moves"]) == 2

    def test_top_moves_excluded_when_no_probabilities(self, action_space):
        result = self._make_turn_result(move_probabilities=None)
        data = serialize_turn_result(result, action_space)
        assert "top_moves" not in data

    def test_numpy_reward_converted(self, action_space):
        result = self._make_turn_result(reward=np.float32(0.5))
        data = serialize_turn_result(result, action_space)
        assert type(data["reward"]) is float


class TestDecodeActionEdgeCases:
    """Additional edge cases for decode_action_readable."""

    def test_index_error_fallback(self):
        """Mock action_space to raise IndexError for fallback path."""
        from unittest.mock import MagicMock
        mock_space = MagicMock()
        mock_space.decode_action.side_effect = IndexError("out of range")
        label = decode_action_readable(42, mock_space)
        assert label == "Action 42"

    def test_value_error_fallback(self):
        """Mock action_space to raise ValueError for fallback path."""
        from unittest.mock import MagicMock
        mock_space = MagicMock()
        mock_space.decode_action.side_effect = ValueError("invalid")
        label = decode_action_readable(99, mock_space)
        assert label == "Action 99"

    def test_unknown_action_type(self):
        """Unknown action type produces descriptive string."""
        from unittest.mock import MagicMock
        mock_space = MagicMock()
        mock_space.decode_action.return_value = {"type": "teleport"}
        label = decode_action_readable(5, mock_space)
        assert "teleport" in label
        assert "5" in label
