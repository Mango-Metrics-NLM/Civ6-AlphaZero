"""Unit tests for action space."""

import numpy as np
import pytest

from alpha_zero.core.civ6_actions import (
    ActionCategory,
    ActionSpace,
    ActionSpaceConfig,
    HierarchicalAction,
    MicroCivActionSpace,
    UnitActionType,
)


class TestMicroCivActionSpace:
    """Test suite for MicroCivActionSpace."""

    @pytest.fixture
    def action_space(self):
        return MicroCivActionSpace(board_size=16, max_units=20, max_cities=5)

    @pytest.fixture
    def sample_game_state(self):
        return {
            "current_player": 0,
            "terrain": {(x, y): {"type": "plains"} for x in range(16) for y in range(16)},
            "units": [
                {
                    "id": 0,
                    "player": 0,
                    "type": "warrior",
                    "position": (5, 5),
                    "moves_remaining": 2,
                },
                {
                    "id": 1,
                    "player": 1,
                    "type": "warrior",
                    "position": (6, 5),
                    "moves_remaining": 2,
                },
            ],
            "cities": [
                {
                    "id": 0,
                    "player": 0,
                    "name": "City 0",
                    "position": (2, 2),
                    "producing": None,
                },
            ],
        }

    def test_action_space_size(self, action_space):
        """Action space has expected size."""
        # 1 end_turn + 20*8 move + 20*8 attack + 20 skip + 20 found_city + 5*3 produce
        expected = 1 + 160 + 160 + 20 + 20 + 15
        assert action_space.total_size == expected

    def test_end_turn_always_valid(self, action_space, sample_game_state):
        """END_TURN action is always valid."""
        mask = action_space.get_valid_mask(sample_game_state)
        assert mask[action_space.end_turn_offset] == 1.0

    def test_unit_movement_valid(self, action_space, sample_game_state):
        """Unit at (5,5) can move in all 8 directions."""
        mask = action_space.get_valid_mask(sample_game_state)

        # Unit 0 should have 8 valid move directions (minus one occupied by enemy)
        move_start = action_space.move_offset
        valid_moves = mask[move_start : move_start + 8]
        # Most directions should be valid
        assert valid_moves.sum() >= 7

    def test_attack_valid_when_adjacent_enemy(self, action_space, sample_game_state):
        """Attack is valid when enemy unit is adjacent."""
        mask = action_space.get_valid_mask(sample_game_state)

        # Unit 0 at (5,5) should be able to attack unit 1 at (6,5)
        attack_start = action_space.attack_offset
        # Direction (1,0) is index 4 in our direction list
        directions = action_space.directions
        dir_idx = directions.index((1, 0))
        assert mask[attack_start + 0 * 8 + dir_idx] == 1.0

    def test_production_valid_for_idle_city(self, action_space, sample_game_state):
        """City without production can start producing."""
        mask = action_space.get_valid_mask(sample_game_state)

        produce_start = action_space.produce_offset
        # City 0 should be able to produce all 3 unit types
        for unit_type in range(3):
            assert mask[produce_start + unit_type] == 1.0

    def test_encode_decode_roundtrip(self, action_space):
        """Encoding and decoding actions is consistent."""
        test_actions = [
            {"type": "end_turn"},
            {"type": "move", "unit_idx": 5, "direction": (1, 0)},
            {"type": "attack", "unit_idx": 3, "direction": (-1, -1)},
            {"type": "skip", "unit_idx": 7},
            {"type": "produce", "city_idx": 2, "unit_type": "settler"},
        ]

        for action in test_actions:
            encoded = action_space.encode_action(action)
            decoded = action_space.decode_action(encoded)
            assert decoded["type"] == action["type"]

    def test_no_moves_when_exhausted(self, action_space):
        """Unit with 0 moves remaining cannot move."""
        state = {
            "current_player": 0,
            "terrain": {(x, y): {"type": "plains"} for x in range(16) for y in range(16)},
            "units": [
                {
                    "id": 0,
                    "player": 0,
                    "type": "warrior",
                    "position": (5, 5),
                    "moves_remaining": 0,  # No moves left
                },
            ],
            "cities": [],
        }
        mask = action_space.get_valid_mask(state)

        # No movement or skip actions should be valid for unit 0
        move_start = action_space.move_offset
        assert mask[move_start : move_start + 8].sum() == 0

    def test_found_city_end_to_end(self):
        """Test complete flow of finding a settler and founding a city."""
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=8, max_turns=50)
        env.reset()

        # Find a settler
        settler = next((u for u in env.units if u.unit_type == "settler"), None)
        if settler is None:
            pytest.skip("No settler found at start of episode")

        # Check action mask
        mask = env.get_valid_actions()
        handler = env.action_space_handler

        # Calculate action index
        # Find unit_idx in player units
        own_units = env._get_player_units(env.current_player)
        unit_idx = next((i for i, u in enumerate(own_units) if u.id == settler.id), -1)
        assert unit_idx != -1, "Settler not found in player units"

        action_idx = handler.found_city_offset + unit_idx

        # Verify action is masked as valid
        assert mask[action_idx] == 1.0, "FOUND_CITY action should be valid"

        # Execute action
        _obs, _reward, _terminated, _, _info = env.step(action_idx)

        # Verify city was founded
        assert any(c.position == settler.position for c in env.cities), (
            "City should be founded at settler position"
        )


class TestHierarchicalActionSpace:
    """Test suite for full hierarchical action space."""

    @pytest.fixture
    def action_space(self):
        config = ActionSpaceConfig(board_size=16)
        return ActionSpace(config)

    def test_category_count(self, action_space):
        """Action space has all 9 categories."""
        assert action_space.num_categories == 9

    def test_end_turn_encoding(self, action_space):
        """END_TURN encodes to 0."""
        action = HierarchicalAction(category=ActionCategory.END_TURN)
        encoded = action_space.encode(action)
        assert encoded == 0

    def test_encode_decode_unit_action(self, action_space):
        """Unit action encode/decode roundtrip."""
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=5,
            action_type=UnitActionType.MOVE,
            target=(10, 12),
        )
        encoded = action_space.encode(action)
        decoded = action_space.decode(encoded)

        assert decoded.category == ActionCategory.UNIT_ACTION
        assert decoded.selector == 5
        assert decoded.action_type == UnitActionType.MOVE
        assert decoded.target == (10, 12)

    def test_encode_decode_tech_research(self, action_space):
        """Tech research encode/decode roundtrip."""
        action = HierarchicalAction(
            category=ActionCategory.RESEARCH_TECH,
            selector=42,
        )
        encoded = action_space.encode(action)
        decoded = action_space.decode(encoded)

        assert decoded.category == ActionCategory.RESEARCH_TECH
        assert decoded.selector == 42

    def test_category_mask(self, action_space):
        """Category mask respects game state."""
        state = {
            "current_player": 0,
            "units": [{"player": 0, "moves_remaining": 1}],
            "cities": [{"player": 0}],
            "current_tech": None,
            "current_civic": None,
            "num_players": 2,
        }
        mask = action_space.get_category_mask(state)

        assert mask[ActionCategory.END_TURN] == 1.0
        assert mask[ActionCategory.UNIT_ACTION] == 1.0
        assert mask[ActionCategory.CITY_ACTION] == 1.0
        assert mask[ActionCategory.RESEARCH_TECH] == 1.0
        assert mask[ActionCategory.DIPLOMACY] == 1.0

    def test_sample_valid_action(self, action_space):
        """Sampling respects action mask."""
        mask = np.zeros(action_space.total_size, dtype=np.float32)
        mask[0] = 1.0  # Only END_TURN valid
        mask[100] = 1.0  # And one other action

        rng = np.random.default_rng(42)
        for _ in range(10):
            action = action_space.sample_valid_action(mask, rng)
            assert action in [0, 100]
