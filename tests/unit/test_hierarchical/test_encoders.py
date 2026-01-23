"""Tests for hierarchical state and goal encoders."""
from __future__ import annotations

import pytest
import numpy as np

from alpha_zero.hierarchical.config import HierarchyLevel
from alpha_zero.hierarchical.encoders.state_encoder import (
    EncoderConfig,
    StrategicStateEncoder,
    TacticalStateEncoder,
    OperationalStateEncoder,
    HierarchicalStateEncoder,
)
from alpha_zero.hierarchical.encoders.goal_encoder import (
    GoalEncoderConfig,
    GoalEncoder,
    GoalEmbedding,
    HierarchicalGoalEncoder,
)
from alpha_zero.hierarchical.goals.goal_types import (
    Goal,
    GoalTarget,
    StrategicGoalType,
    TacticalGoalType,
    OperationalGoalType,
)


class TestEncoderConfig:
    """Tests for EncoderConfig."""

    def test_default_config(self) -> None:
        """Test default encoder configuration."""
        config = EncoderConfig()

        assert config.embed_dim == 128
        assert config.board_size == (16, 16)
        assert config.use_attention is True

    def test_level_specific_config(self) -> None:
        """Test level-specific configuration."""
        strategic = EncoderConfig.for_level(HierarchyLevel.STRATEGIC)
        tactical = EncoderConfig.for_level(HierarchyLevel.TACTICAL)
        operational = EncoderConfig.for_level(HierarchyLevel.OPERATIONAL)

        # Strategic should have largest embed_dim
        assert strategic.embed_dim >= tactical.embed_dim
        assert tactical.embed_dim >= operational.embed_dim


class TestStrategicStateEncoder:
    """Tests for StrategicStateEncoder."""

    def test_encoder_creation(self) -> None:
        """Test creating strategic encoder."""
        encoder = StrategicStateEncoder()

        assert encoder.level == HierarchyLevel.STRATEGIC
        assert encoder.output_dim > 0

    def test_encode_empty_state(self) -> None:
        """Test encoding empty game state."""
        encoder = StrategicStateEncoder()
        game_state = {}

        encoded = encoder.encode(game_state)

        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (encoder.output_dim,)
        assert encoded.dtype == np.float32

    def test_encode_full_state(self) -> None:
        """Test encoding full game state."""
        encoder = StrategicStateEncoder()
        game_state = {
            "victory_progress": {
                "domination": 0.3,
                "science": 0.5,
                "culture": 0.2,
                "religious": 0.0,
                "diplomatic": 0.1,
                "score": 0.4,
            },
            "diplomacy": {
                "relationships": {
                    "civ_0": {"score": 50},
                    "civ_1": {"score": -30},
                },
            },
            "resources": {"gold": 500, "science": 100},
            "yields": {"gold_per_turn": 50, "science_per_turn": 40},
            "resource_trends": {"gold": 1, "science": 1},
            "military": {"strength": 500, "at_war": 0},
            "territory": {"tiles_owned": 100},
            "cities": [{"player": 0} for _ in range(5)],
            "technology": {"era": 2, "techs_researched": 20},
            "civics": {"civics_completed": 15},
        }

        encoded = encoder.encode(game_state)

        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (encoder.output_dim,)
        # Should have non-zero values
        assert np.any(encoded != 0)


class TestTacticalStateEncoder:
    """Tests for TacticalStateEncoder."""

    def test_encoder_creation(self) -> None:
        """Test creating tactical encoder."""
        encoder = TacticalStateEncoder()

        assert encoder.level == HierarchyLevel.TACTICAL
        assert encoder.output_dim > 0

    def test_encode_with_focus(self) -> None:
        """Test encoding with focus location."""
        encoder = TacticalStateEncoder()
        game_state = {
            "terrain": {},
            "units": [{"player": 0, "x": 5, "y": 5, "class": "melee"}],
            "cities": [{"player": 0, "x": 5, "y": 5, "population": 5}],
            "improvements": {},
            "current_player": 0,
            "resources": {"gold": 100},
            "yields": {"gold_per_turn": 10},
        }

        encoded = encoder.encode(game_state, focus_location=(5, 5))

        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (encoder.output_dim,)

    def test_encode_cities(self) -> None:
        """Test city encoding."""
        encoder = TacticalStateEncoder()
        game_state = {
            "cities": [
                {
                    "player": 0,
                    "x": 5,
                    "y": 5,
                    "population": 10,
                    "production": 20,
                    "food": 15,
                    "gold": 10,
                    "science": 8,
                    "culture": 5,
                    "housing": 12,
                    "amenities": 3,
                    "districts": 4,
                    "buildings": 8,
                }
            ],
            "current_player": 0,
            "terrain": {},
            "units": [],
            "improvements": {},
            "resources": {},
            "yields": {},
        }

        encoded = encoder.encode(game_state)

        assert isinstance(encoded, np.ndarray)


class TestOperationalStateEncoder:
    """Tests for OperationalStateEncoder."""

    def test_encoder_creation(self) -> None:
        """Test creating operational encoder."""
        encoder = OperationalStateEncoder()

        assert encoder.level == HierarchyLevel.OPERATIONAL
        assert encoder.output_dim > 0

    def test_encode_with_active_unit(self) -> None:
        """Test encoding with active unit."""
        encoder = OperationalStateEncoder()
        active_unit = {
            "player": 0,
            "x": 8,
            "y": 8,
            "strength": 50,
            "health": 80,
            "movement": 2,
            "can_attack": True,
        }
        game_state = {
            "terrain": {},
            "units": [active_unit],
            "cities": [],
            "improvements": {},
            "current_player": 0,
        }

        encoded = encoder.encode(game_state, active_unit=active_unit)

        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (encoder.output_dim,)

    def test_encode_threats(self) -> None:
        """Test threat encoding."""
        encoder = OperationalStateEncoder()
        game_state = {
            "terrain": {},
            "units": [
                {"player": 0, "x": 8, "y": 8},  # Own unit
                {"player": 1, "x": 9, "y": 8, "strength": 50},  # Enemy nearby
            ],
            "cities": [],
            "improvements": {},
            "current_player": 0,
        }

        encoded = encoder.encode(game_state)

        assert isinstance(encoded, np.ndarray)


class TestHierarchicalStateEncoder:
    """Tests for HierarchicalStateEncoder."""

    def test_encoder_creation(self) -> None:
        """Test creating hierarchical encoder."""
        encoder = HierarchicalStateEncoder()

        assert encoder.strategic is not None
        assert encoder.tactical is not None
        assert encoder.operational is not None

    def test_encode_single_level(self) -> None:
        """Test encoding for single level."""
        encoder = HierarchicalStateEncoder()
        game_state = {"turn": 0}

        strategic = encoder.encode(game_state, HierarchyLevel.STRATEGIC)
        tactical = encoder.encode(game_state, HierarchyLevel.TACTICAL)
        operational = encoder.encode(game_state, HierarchyLevel.OPERATIONAL)

        assert isinstance(strategic, np.ndarray)
        assert isinstance(tactical, np.ndarray)
        assert isinstance(operational, np.ndarray)

    def test_encode_all_levels(self) -> None:
        """Test encoding all levels simultaneously."""
        encoder = HierarchicalStateEncoder()
        game_state = {"turn": 0}

        all_encoded = encoder.encode_all(game_state)

        assert HierarchyLevel.STRATEGIC in all_encoded
        assert HierarchyLevel.TACTICAL in all_encoded
        assert HierarchyLevel.OPERATIONAL in all_encoded

    def test_get_output_dims(self) -> None:
        """Test getting output dimensions."""
        encoder = HierarchicalStateEncoder()

        dims = encoder.get_output_dims()

        assert HierarchyLevel.STRATEGIC in dims
        assert all(d > 0 for d in dims.values())


class TestGoalEncoder:
    """Tests for GoalEncoder."""

    def test_encoder_creation(self) -> None:
        """Test creating goal encoder."""
        encoder = GoalEncoder()

        assert encoder.output_dim > 0

    def test_encode_strategic_goal(self) -> None:
        """Test encoding strategic goal."""
        encoder = GoalEncoder()
        goal = Goal.strategic(StrategicGoalType.SCIENCE_VICTORY)

        embedding = encoder.encode(goal)

        assert isinstance(embedding, GoalEmbedding)
        assert embedding.goal_id == goal.id
        assert embedding.level == HierarchyLevel.STRATEGIC
        assert embedding.embedding.shape == (encoder.output_dim,)

    def test_encode_tactical_goal(self) -> None:
        """Test encoding tactical goal."""
        encoder = GoalEncoder()
        goal = Goal.tactical(TacticalGoalType.CAPTURE_CITY)
        goal.target = GoalTarget(location=(10, 10))

        embedding = encoder.encode(goal)

        assert isinstance(embedding, GoalEmbedding)
        assert embedding.level == HierarchyLevel.TACTICAL

    def test_encode_operational_goal(self) -> None:
        """Test encoding operational goal."""
        encoder = GoalEncoder()
        goal = Goal.operational(OperationalGoalType.MOVE_UNIT)

        embedding = encoder.encode(goal)

        assert isinstance(embedding, GoalEmbedding)
        assert embedding.level == HierarchyLevel.OPERATIONAL

    def test_encode_with_context(self) -> None:
        """Test encoding with parent goal context."""
        encoder = GoalEncoder()

        parent = Goal.strategic(StrategicGoalType.DOMINATION_VICTORY)
        child = Goal.tactical(TacticalGoalType.BUILD_ARMY)
        child.parent_id = parent.id

        embedding = encoder.encode(child, parent_goal=parent)

        assert isinstance(embedding, GoalEmbedding)

    def test_encode_batch(self) -> None:
        """Test encoding multiple goals."""
        encoder = GoalEncoder()
        goals = [
            Goal.tactical(TacticalGoalType.RESEARCH_TECH),
            Goal.tactical(TacticalGoalType.BUILD_CAMPUS),
            Goal.tactical(TacticalGoalType.DEVELOP_CITY),
        ]

        batch = encoder.encode_batch(goals)

        assert isinstance(batch, np.ndarray)
        assert batch.shape[0] == encoder.config.max_goals
        assert batch.shape[1] == encoder.output_dim

    def test_encode_with_progress(self) -> None:
        """Test encoding includes progress information."""
        encoder = GoalEncoder(GoalEncoderConfig(include_progress=True))

        goal = Goal.strategic(StrategicGoalType.CULTURE_VICTORY)
        goal.progress = 0.5
        goal.created_turn = 0
        goal.deadline_turn = 100

        game_state = {"turn": 50}
        embedding = encoder.encode(goal, game_state=game_state)

        assert embedding.embedding.shape == (encoder.output_dim,)


class TestHierarchicalGoalEncoder:
    """Tests for HierarchicalGoalEncoder."""

    def test_encoder_creation(self) -> None:
        """Test creating hierarchical goal encoder."""
        encoder = HierarchicalGoalEncoder()

        assert encoder.strategic_encoder is not None
        assert encoder.tactical_encoder is not None
        assert encoder.operational_encoder is not None

    def test_encode_goal(self) -> None:
        """Test encoding single goal."""
        encoder = HierarchicalGoalEncoder()

        goal = Goal.tactical(TacticalGoalType.SETTLE_CITY)
        embedding = encoder.encode(goal)

        assert isinstance(embedding, GoalEmbedding)

    def test_encode_goals_for_level(self) -> None:
        """Test encoding goals for specific level."""
        encoder = HierarchicalGoalEncoder()
        goals = [
            Goal.tactical(TacticalGoalType.BUILD_ARMY),
            Goal.tactical(TacticalGoalType.DEFEND_CITY),
            Goal.strategic(StrategicGoalType.DOMINATION_VICTORY),  # Different level
        ]

        tactical_batch = encoder.encode_goals_for_level(
            goals, HierarchyLevel.TACTICAL
        )

        assert isinstance(tactical_batch, np.ndarray)

    def test_encode_active_goals(self) -> None:
        """Test encoding active goals across levels."""
        encoder = HierarchicalGoalEncoder()
        goals = [
            Goal.strategic(StrategicGoalType.SCIENCE_VICTORY),
            Goal.tactical(TacticalGoalType.RESEARCH_TECH),
            Goal.operational(OperationalGoalType.SELECT_RESEARCH),
        ]

        encoded = encoder.encode_active_goals(goals)

        assert HierarchyLevel.STRATEGIC in encoded
        assert HierarchyLevel.TACTICAL in encoded
        assert HierarchyLevel.OPERATIONAL in encoded

    def test_get_output_dims(self) -> None:
        """Test getting output dimensions."""
        encoder = HierarchicalGoalEncoder()

        dims = encoder.get_output_dims()

        assert len(dims) == 3
        assert all(d > 0 for d in dims.values())
