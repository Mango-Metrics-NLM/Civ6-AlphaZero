"""Unit tests for CivRealm environment wrapper.

Tests cover:
- CivRealmConfig validation
- CivRealmEncoder observation encoding
- CivRealmActionHandler action handling
- CivRealmCurriculum progression
- CivRealmWrapper Gymnasium interface
- Mock environment behavior

Note: These tests use mock environments and do not require a running CivRealm server.
"""

from __future__ import annotations

import numpy as np
import pytest

# Import the civrealm_wrapper module directly to avoid lazy import issues
# This handles the case where CivRealm tries to connect to a server during import
try:
    from alpha_zero.envs.civrealm_wrapper import (
        CIVREALM_AVAILABLE,
        ActionType,
        CivRealmActionHandler,
        CivRealmConfig,
        CivRealmCurriculum,
        CivRealmEncoder,
        CivRealmWrapper,
        MiniGameTask,
        MockCivRealmEnv,
        TerrainType,
        UnitCategory,
        create_civrealm_env,
        make,
    )

    IMPORT_ERROR = None
except Exception as e:
    # If import fails due to CivRealm server issues, skip tests
    IMPORT_ERROR = str(e)
    CivRealmConfig = None
    CivRealmEncoder = None
    CivRealmActionHandler = None
    CivRealmCurriculum = None
    CivRealmWrapper = None
    MiniGameTask = None
    MockCivRealmEnv = None
    TerrainType = None
    UnitCategory = None
    ActionType = None
    create_civrealm_env = None
    make = None
    CIVREALM_AVAILABLE = False


# Skip all tests if import failed
pytestmark = pytest.mark.skipif(
    IMPORT_ERROR is not None, reason=f"CivRealm wrapper import failed: {IMPORT_ERROR}"
)


# ============================================================================
# CivRealmConfig Tests
# ============================================================================


class TestCivRealmConfig:
    """Tests for CivRealmConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CivRealmConfig()

        assert config.task == "development_build_city"
        assert config.difficulty == 1
        assert config.spatial_size == 32
        assert config.max_units == 100
        assert config.max_cities == 50
        assert config.max_actions_per_turn == 100
        assert config.host == "localhost"
        assert config.port == 6300
        assert config.client_port == 6001
        assert config.username == "alphazero_agent"
        assert config.max_turns == 500
        assert config.render_mode is None
        assert config.use_mock is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CivRealmConfig(
            task="battle_ancient",
            difficulty=5,
            spatial_size=64,
            max_units=50,
            max_cities=25,
            max_actions_per_turn=200,
            port=7000,
            use_mock=True,
        )

        assert config.task == "battle_ancient"
        assert config.difficulty == 5
        assert config.spatial_size == 64
        assert config.max_units == 50
        assert config.max_cities == 25
        assert config.max_actions_per_turn == 200
        assert config.port == 7000
        assert config.use_mock is True

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = CivRealmConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_difficulty_low(self):
        """Test validation fails for difficulty < 1."""
        config = CivRealmConfig(difficulty=0)
        with pytest.raises(ValueError, match="Difficulty must be 1-10"):
            config.validate()

    def test_validate_invalid_difficulty_high(self):
        """Test validation fails for difficulty > 10."""
        config = CivRealmConfig(difficulty=11)
        with pytest.raises(ValueError, match="Difficulty must be 1-10"):
            config.validate()

    def test_validate_invalid_spatial_size(self):
        """Test validation fails for small spatial_size."""
        config = CivRealmConfig(spatial_size=4)
        with pytest.raises(ValueError, match="Spatial size must be >= 8"):
            config.validate()

    def test_validate_invalid_max_units(self):
        """Test validation fails for max_units < 1."""
        config = CivRealmConfig(max_units=0)
        with pytest.raises(ValueError, match="max_units must be >= 1"):
            config.validate()

    def test_validate_invalid_max_cities(self):
        """Test validation fails for max_cities < 1."""
        config = CivRealmConfig(max_cities=0)
        with pytest.raises(ValueError, match="max_cities must be >= 1"):
            config.validate()

    def test_validate_invalid_port(self):
        """Test validation fails for invalid port."""
        config = CivRealmConfig(port=70000)
        with pytest.raises(ValueError, match="Port must be 1-65535"):
            config.validate()

    def test_validate_unknown_task_warning(self):
        """Test validation warns for unknown task but doesn't fail."""
        config = CivRealmConfig(task="unknown_task")
        # Should not raise, just warn
        config.validate()


# ============================================================================
# CivRealmEncoder Tests
# ============================================================================


class TestCivRealmEncoder:
    """Tests for CivRealmEncoder class."""

    @pytest.fixture
    def encoder(self):
        """Create encoder with default config."""
        config = CivRealmConfig(spatial_size=16, max_actions_per_turn=50)
        return CivRealmEncoder(config)

    @pytest.fixture
    def sample_observation(self):
        """Create sample CivRealm observation."""
        return {
            "turn": 10,
            "year": -3500,
            "player_id": 0,
            "map": {
                "terrain": {
                    (0, 0): {"terrain": 5, "food": 2, "production": 1, "trade": 0},
                    (1, 0): {
                        "terrain": 6,
                        "food": 3,
                        "production": 1,
                        "trade": 1,
                        "has_road": True,
                    },
                    (0, 1): {
                        "terrain": 13,
                        "food": 1,
                        "production": 2,
                        "trade": 0,
                        "has_mine": True,
                    },
                },
                "visibility": {
                    (0, 0): "visible",
                    (1, 0): "visible",
                    (0, 1): "seen",
                    (1, 1): "unexplored",
                },
            },
            "units": {
                "unit_0": {
                    "owner": 0,
                    "position": {"x": 0, "y": 0},
                    "type_name": "Warrior",
                    "hp": 100,
                    "moves_left": 2,
                },
                "unit_1": {
                    "owner": 1,
                    "position": {"x": 1, "y": 0},
                    "type_name": "Archer",
                    "hp": 75,
                    "moves_left": 1,
                },
                "unit_2": {
                    "owner": 0,
                    "position": {"x": 2, "y": 0},
                    "type_name": "Settler",
                    "hp": 100,
                    "moves_left": 2,
                },
            },
            "cities": {
                "city_0": {
                    "owner": 0,
                    "position": {"x": 0, "y": 1},
                    "population": 5,
                    "food_surplus": 2,
                    "production": 10,
                },
                "city_1": {
                    "owner": 1,
                    "position": {"x": 5, "y": 5},
                    "population": 3,
                    "food_surplus": 1,
                    "production": 5,
                },
            },
            "player": {
                "gold": 500,
                "gold_income": 10,
                "science": 25,
            },
            "diplomacy": {"player_1": "war"},
            "score": {"total": 100, "cities": 50, "units": 30},
            "available_actions": {
                "action_0": {"type": "unit_move"},
                "action_1": {"type": "unit_build"},
                "action_2": {"type": "city_produce"},
            },
        }

    def test_encoder_initialization(self, encoder):
        """Test encoder initializes correctly."""
        assert encoder.spatial_channels == 51
        assert encoder.vector_dim == 256
        assert encoder.config.spatial_size == 16

    def test_encode_observation_structure(self, encoder, sample_observation):
        """Test encoded observation has correct structure."""
        encoded = encoder.encode_observation(sample_observation)

        assert "spatial" in encoded
        assert "vector" in encoded
        assert "action_mask" in encoded

        assert encoded["spatial"].dtype == np.float32
        assert encoded["vector"].dtype == np.float32
        assert encoded["action_mask"].dtype == np.float32

    def test_encode_spatial_shape(self, encoder, sample_observation):
        """Test spatial encoding has correct shape."""
        encoded = encoder.encode_observation(sample_observation)
        spatial = encoded["spatial"]

        expected_shape = (51, 16, 16)
        assert spatial.shape == expected_shape

    def test_encode_vector_shape(self, encoder, sample_observation):
        """Test vector encoding has correct shape."""
        encoded = encoder.encode_observation(sample_observation)
        vector = encoded["vector"]

        expected_shape = (256,)
        assert vector.shape == expected_shape

    def test_encode_action_mask_shape(self, encoder, sample_observation):
        """Test action mask has correct shape."""
        encoded = encoder.encode_observation(sample_observation)
        action_mask = encoded["action_mask"]

        expected_shape = (50,)  # max_actions_per_turn
        assert action_mask.shape == expected_shape

    def test_encode_spatial_terrain(self, encoder, sample_observation):
        """Test terrain encoding in spatial tensor."""
        encoded = encoder.encode_observation(sample_observation)
        spatial = encoded["spatial"]

        # Check plains terrain at (0,0) - terrain type 5
        assert spatial[5, 0, 0] == 1.0

        # Check grassland at (1,0) - terrain type 6
        assert spatial[6, 0, 1] == 1.0

        # Check hills at (0,1) - terrain type 13
        assert spatial[13, 1, 0] == 1.0

    def test_encode_spatial_units(self, encoder, sample_observation):
        """Test unit encoding in spatial tensor."""
        encoded = encoder.encode_observation(sample_observation)
        spatial = encoded["spatial"]

        # Own unit (Warrior) at (0,0) - category LAND_MILITARY = 0
        assert spatial[15, 0, 0] == 1.0  # OWN_UNITS_START + 0

        # Enemy unit (Archer) at (1,0) - category LAND_MILITARY = 0
        assert spatial[24, 0, 1] > 0  # ENEMY_UNITS_START + 0

        # Own unit (Settler) at (2,0) - category SETTLER = 4
        assert spatial[19, 0, 2] == 1.0  # OWN_UNITS_START + 4

    def test_encode_spatial_cities(self, encoder, sample_observation):
        """Test city encoding in spatial tensor."""
        encoded = encoder.encode_observation(sample_observation)
        spatial = encoded["spatial"]

        # Own city at (0,1)
        assert spatial[33, 1, 0] == 1.0  # OWN_CITIES_START (presence)
        assert spatial[34, 1, 0] == pytest.approx(5 / 50.0, rel=1e-3)  # Population normalized

        # Enemy city at (5,5)
        assert spatial[35, 5, 5] == 1.0  # ENEMY_CITIES_START (presence)

    def test_encode_spatial_fog(self, encoder, sample_observation):
        """Test fog of war encoding."""
        encoded = encoder.encode_observation(sample_observation)
        spatial = encoded["spatial"]

        # Visible at (0,0)
        assert spatial[37, 0, 0] == 1.0

        # Seen at (0,1)
        assert spatial[38, 1, 0] == 1.0

    def test_encode_spatial_bias(self, encoder, sample_observation):
        """Test bias channel is all ones."""
        encoded = encoder.encode_observation(sample_observation)
        spatial = encoded["spatial"]

        assert np.all(spatial[50] == 1.0)

    def test_encode_vector_turn_info(self, encoder, sample_observation):
        """Test turn info in vector encoding."""
        encoded = encoder.encode_observation(sample_observation)
        vector = encoded["vector"]

        # Turn 10 / 500.0
        assert vector[0] == pytest.approx(10 / 500.0, rel=1e-3)

    def test_encode_vector_resources(self, encoder, sample_observation):
        """Test resource info in vector encoding."""
        encoded = encoder.encode_observation(sample_observation)
        vector = encoded["vector"]

        # Gold 500 / 10000.0
        assert vector[20] == pytest.approx(500 / 10000.0, rel=1e-3)

        # Gold income 10 / 100.0
        assert vector[21] == pytest.approx(10 / 100.0, rel=1e-3)

    def test_encode_vector_unit_counts(self, encoder, sample_observation):
        """Test unit count in vector encoding."""
        encoded = encoder.encode_observation(sample_observation)
        vector = encoded["vector"]

        # 2 own units / 100.0
        assert vector[50] == pytest.approx(2 / 100.0, rel=1e-3)

        # 1 enemy unit / 100.0
        assert vector[51] == pytest.approx(1 / 100.0, rel=1e-3)

    def test_encode_action_mask_valid_actions(self, encoder, sample_observation):
        """Test action mask marks valid actions."""
        encoded = encoder.encode_observation(sample_observation)
        action_mask = encoded["action_mask"]

        # First 3 actions should be valid
        assert action_mask[0] == 1.0
        assert action_mask[1] == 1.0
        assert action_mask[2] == 1.0

        # End turn always valid (last action)
        assert action_mask[-1] == 1.0

        # At least one valid action
        assert action_mask.sum() >= 1

    def test_observation_space(self, encoder):
        """Test observation space definition."""
        obs_space = encoder.observation_space

        assert "spatial" in obs_space.spaces
        assert "vector" in obs_space.spaces
        assert "action_mask" in obs_space.spaces

        assert obs_space["spatial"].shape == (51, 16, 16)
        assert obs_space["vector"].shape == (256,)
        assert obs_space["action_mask"].shape == (50,)

    def test_spatial_shape_property(self, encoder):
        """Test spatial_shape property."""
        assert encoder.spatial_shape == (51, 16, 16)

    def test_unit_category_mapping(self, encoder):
        """Test unit type to category mapping."""
        # Military
        assert encoder._get_unit_category("Warrior") == UnitCategory.LAND_MILITARY.value
        assert encoder._get_unit_category("ARCHER") == UnitCategory.LAND_MILITARY.value

        # Settlers
        assert encoder._get_unit_category("Settler") == UnitCategory.SETTLER.value
        assert encoder._get_unit_category("Pioneer") == UnitCategory.SETTLER.value

        # Workers
        assert encoder._get_unit_category("Worker") == UnitCategory.WORKER.value
        assert encoder._get_unit_category("Engineer") == UnitCategory.WORKER.value

        # Naval
        assert encoder._get_unit_category("Trireme") == UnitCategory.NAVAL_MILITARY.value
        assert encoder._get_unit_category("Battleship") == UnitCategory.NAVAL_MILITARY.value

        # Air
        assert encoder._get_unit_category("Fighter") == UnitCategory.AIR_MILITARY.value
        assert encoder._get_unit_category("Bomber") == UnitCategory.AIR_MILITARY.value

        # Trade
        assert encoder._get_unit_category("Caravan") == UnitCategory.CARAVAN.value

        # Diplomats/Spies
        assert encoder._get_unit_category("Diplomat") == UnitCategory.DIPLOMAT.value
        assert encoder._get_unit_category("Spy") == UnitCategory.SPY.value


# ============================================================================
# CivRealmActionHandler Tests
# ============================================================================


class TestCivRealmActionHandler:
    """Tests for CivRealmActionHandler class."""

    @pytest.fixture
    def handler(self):
        """Create action handler."""
        config = CivRealmConfig(max_actions_per_turn=50)
        return CivRealmActionHandler(config)

    @pytest.fixture
    def sample_actions(self):
        """Sample available actions."""
        return {
            "move_unit_0_north": {"type": "unit_move", "unit_id": "unit_0", "direction": "north"},
            "move_unit_0_south": {"type": "unit_move", "unit_id": "unit_0", "direction": "south"},
            "build_city": {"type": "unit_build", "unit_id": "unit_1"},
            "produce_warrior": {"type": "city_produce", "city_id": "city_0", "item": "Warrior"},
        }

    def test_handler_initialization(self, handler):
        """Test handler initializes correctly."""
        assert handler.action_size == 50

    def test_build_action_mapping(self, handler, sample_actions):
        """Test building action mapping."""
        handler.build_action_mapping(sample_actions)

        assert len(handler._action_cache) > 0
        assert len(handler._reverse_cache) > 0

        # End turn should always be mapped
        assert "end_turn" in handler._reverse_cache

    def test_decode_action_valid(self, handler, sample_actions):
        """Test decoding valid action."""
        action = handler.decode_action(0, sample_actions)

        assert "action_key" in action or "type" in action

    def test_decode_action_end_turn(self, handler, sample_actions):
        """Test decoding end turn action."""
        action = handler.decode_action(49, sample_actions)  # Last index

        assert action["type"] == "end_turn"

    def test_decode_action_invalid_fallback(self, handler):
        """Test decoding invalid action falls back to end_turn."""
        action = handler.decode_action(999, {})

        assert action["type"] == "end_turn"

    def test_encode_action(self, handler, sample_actions):
        """Test encoding action to index."""
        handler.build_action_mapping(sample_actions)

        # Encode a known action
        action = {"action_key": "move_unit_0_north"}
        idx = handler.encode_action(action, sample_actions)

        assert 0 <= idx < handler.action_size

    def test_encode_action_end_turn(self, handler, sample_actions):
        """Test encoding end_turn action."""
        action = {"type": "end_turn"}
        idx = handler.encode_action(action, sample_actions)

        assert idx == handler.action_size - 1

    def test_get_action_mask(self, handler, sample_actions):
        """Test generating action mask."""
        mask = handler.get_action_mask(sample_actions)

        assert mask.shape == (50,)
        assert mask.dtype == np.float32

        # Should have some valid actions
        assert mask.sum() >= 1

        # End turn always valid
        assert mask[-1] == 1.0

    def test_sample_valid_action(self, handler, sample_actions):
        """Test sampling valid action."""
        mask = handler.get_action_mask(sample_actions)
        rng = np.random.default_rng(42)

        action_idx = handler.sample_valid_action(mask, rng)

        assert 0 <= action_idx < handler.action_size
        assert mask[action_idx] == 1.0

    def test_sample_valid_action_empty_mask(self, handler):
        """Test sampling with empty mask returns end_turn."""
        mask = np.zeros(50, dtype=np.float32)
        rng = np.random.default_rng(42)

        action_idx = handler.sample_valid_action(mask, rng)

        assert action_idx == handler.action_size - 1  # end_turn

    def test_action_space(self, handler):
        """Test action space property."""
        space = handler.action_space

        assert space.n == 50

    def test_many_actions_truncation(self, handler):
        """Test truncation when too many actions."""
        # Create more actions than max_actions_per_turn - 1
        many_actions = {f"action_{i}": {"type": "test"} for i in range(100)}

        mask = handler.get_action_mask(many_actions)

        # Should still have correct shape
        assert mask.shape == (50,)


# ============================================================================
# CivRealmCurriculum Tests
# ============================================================================


class TestCivRealmCurriculum:
    """Tests for CivRealmCurriculum class."""

    @pytest.fixture
    def curriculum(self):
        """Create curriculum with default tasks."""
        return CivRealmCurriculum()

    def test_curriculum_initialization(self, curriculum):
        """Test curriculum initializes with default tasks."""
        assert len(curriculum.tasks) > 0
        assert len(curriculum.task_order) > 0
        assert "development_build_city" in curriculum.tasks

    def test_get_current_task(self, curriculum):
        """Test getting current task."""
        task = curriculum.get_current_task()

        assert isinstance(task, MiniGameTask)
        assert task.name == "development_build_city"  # First task

    def test_get_next_task_initial(self, curriculum):
        """Test getting next task when starting."""
        task_name = curriculum.get_next_task(agent_skill_level=0.0)

        # Should return first unmastered task
        assert task_name == "development_build_city"

    def test_get_next_task_with_skill(self, curriculum):
        """Test getting appropriate task for skill level."""
        # High skill should allow harder tasks (if prerequisites met)
        task_name = curriculum.get_next_task(agent_skill_level=0.9)

        assert task_name in curriculum.tasks

    def test_record_result_success(self, curriculum):
        """Test recording successful game."""
        result = curriculum.record_result("development_build_city", success=True)

        assert result["task"] == "development_build_city"
        assert result["success"] is True
        assert result["games_played"] == 1
        assert curriculum.games_played["development_build_city"] == 1

    def test_record_result_failure(self, curriculum):
        """Test recording failed game."""
        result = curriculum.record_result("development_build_city", success=False)

        assert result["success"] is False

    def test_record_result_unknown_task(self, curriculum):
        """Test recording result for unknown task."""
        result = curriculum.record_result("unknown_task", success=True)

        assert "error" in result

    def test_task_mastery(self, curriculum):
        """Test task mastery detection."""
        task_name = "development_build_city"

        # Not mastered initially
        assert not curriculum._is_task_mastered(task_name)

        # Record many wins to achieve mastery
        for _ in range(50):
            curriculum.record_result(task_name, success=True)

        # Should be mastered now
        assert curriculum._is_task_mastered(task_name)

    def test_get_progress(self, curriculum):
        """Test getting progress summary."""
        # Record some games
        for _ in range(5):
            curriculum.record_result("development_build_city", success=True)

        progress = curriculum.get_progress()

        assert "total_tasks" in progress
        assert "mastered_tasks" in progress
        assert "total_games" in progress
        assert progress["total_games"] == 5

    def test_reset(self, curriculum):
        """Test resetting curriculum."""
        # Record some games
        curriculum.record_result("development_build_city", success=True)

        curriculum.reset()

        assert curriculum.games_played["development_build_city"] == 0
        assert len(curriculum.success_rates["development_build_city"]) == 0

    def test_task_order_by_difficulty(self, curriculum):
        """Test tasks are ordered by difficulty."""
        difficulties = [curriculum.tasks[t].difficulty for t in curriculum.task_order]

        # Should be sorted ascending
        assert difficulties == sorted(difficulties)

    def test_custom_tasks(self):
        """Test curriculum with custom tasks."""
        custom_tasks = [
            MiniGameTask(
                name="custom_easy",
                display_name="Easy Task",
                description="Test easy task",
                difficulty=0.1,
            ),
            MiniGameTask(
                name="custom_hard",
                display_name="Hard Task",
                description="Test hard task",
                difficulty=0.9,
                prerequisites=["custom_easy"],
            ),
        ]

        curriculum = CivRealmCurriculum(custom_tasks=custom_tasks)

        assert len(curriculum.tasks) == 2
        assert "custom_easy" in curriculum.tasks
        assert "custom_hard" in curriculum.tasks

    def test_prerequisites_check(self):
        """Test prerequisite checking for task selection."""
        custom_tasks = [
            MiniGameTask(name="task_a", display_name="A", description="A", difficulty=0.1),
            MiniGameTask(
                name="task_b",
                display_name="B",
                description="B",
                difficulty=0.5,
                prerequisites=["task_a"],
            ),
        ]

        curriculum = CivRealmCurriculum(custom_tasks=custom_tasks)

        # Task B requires task A to be mastered
        task = curriculum.get_next_task(0.5)
        assert task == "task_a"  # Should select A first since B has unmet prerequisites


# ============================================================================
# MockCivRealmEnv Tests
# ============================================================================


class TestMockCivRealmEnv:
    """Tests for MockCivRealmEnv class."""

    @pytest.fixture
    def mock_env(self):
        """Create mock environment."""
        config = CivRealmConfig(spatial_size=16, max_turns=100, max_actions_per_turn=50)
        return MockCivRealmEnv(config)

    def test_mock_initialization(self, mock_env):
        """Test mock environment initializes."""
        assert mock_env._turn == 0
        assert mock_env._done is False

    def test_mock_reset(self, mock_env):
        """Test mock reset returns observation."""
        obs = mock_env.reset()

        assert isinstance(obs, dict)
        assert "turn" in obs
        assert "units" in obs
        assert "cities" in obs
        assert "available_actions" in obs
        assert obs["turn"] == 0

    def test_mock_step(self, mock_env):
        """Test mock step execution."""
        mock_env.reset()

        obs, reward, done, info = mock_env.step({"type": "end_turn"})

        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert obs["turn"] == 1

    def test_mock_termination(self, mock_env):
        """Test mock terminates at max_turns."""
        mock_env.reset()

        for _ in range(100):
            _obs, _reward, done, _info = mock_env.step({"type": "end_turn"})

        assert done is True

    def test_mock_close(self, mock_env):
        """Test mock close doesn't raise."""
        mock_env.close()  # Should not raise

    def test_mock_observation_structure(self, mock_env):
        """Test mock observation has expected structure."""
        obs = mock_env.reset()

        assert "map" in obs
        assert "terrain" in obs["map"]
        assert "visibility" in obs["map"]
        assert "player" in obs
        assert "player_id" in obs


# ============================================================================
# CivRealmWrapper Tests
# ============================================================================


class TestCivRealmWrapper:
    """Tests for CivRealmWrapper class."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper with mock environment."""
        config = CivRealmConfig(
            task="development_build_city",
            spatial_size=16,
            max_actions_per_turn=50,
            max_turns=100,
            use_mock=True,  # Force mock for testing
        )
        return CivRealmWrapper(config)

    def test_wrapper_initialization(self, wrapper):
        """Test wrapper initializes correctly."""
        assert wrapper.config is not None
        assert wrapper.encoder is not None
        assert wrapper.action_handler is not None
        assert wrapper._env is not None

    def test_wrapper_observation_space(self, wrapper):
        """Test wrapper has valid observation space."""
        obs_space = wrapper.observation_space

        assert "spatial" in obs_space.spaces
        assert "vector" in obs_space.spaces
        assert "action_mask" in obs_space.spaces

    def test_wrapper_action_space(self, wrapper):
        """Test wrapper has valid action space."""
        action_space = wrapper.action_space

        assert action_space.n == 50

    def test_wrapper_reset(self, wrapper):
        """Test wrapper reset returns valid observation."""
        obs, _info = wrapper.reset()

        assert "spatial" in obs
        assert "vector" in obs
        assert "action_mask" in obs

        assert obs["spatial"].shape == (51, 16, 16)
        assert obs["vector"].shape == (256,)
        assert obs["action_mask"].shape == (50,)

    def test_wrapper_step(self, wrapper):
        """Test wrapper step execution."""
        obs, _info = wrapper.reset()

        # Take action 0 (or end_turn)
        obs, reward, terminated, truncated, info = wrapper.step(0)

        assert "spatial" in obs
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_wrapper_step_with_numpy_action(self, wrapper):
        """Test wrapper accepts numpy array action."""
        wrapper.reset()

        action = np.array(0)
        obs, _reward, _terminated, _truncated, _info = wrapper.step(action)

        assert "spatial" in obs

    def test_wrapper_get_valid_actions(self, wrapper):
        """Test getting valid action mask."""
        wrapper.reset()

        mask = wrapper.get_valid_actions()

        assert mask.shape == (50,)
        assert mask.sum() >= 1  # At least one valid action

    def test_wrapper_render_ansi(self):
        """Test ANSI rendering."""
        config = CivRealmConfig(
            render_mode="ansi",
            use_mock=True,
        )
        wrapper = CivRealmWrapper(config)
        wrapper.reset()

        output = wrapper.render()

        assert isinstance(output, str)
        assert "CivRealm" in output

    def test_wrapper_close(self, wrapper):
        """Test wrapper close."""
        wrapper.reset()
        wrapper.close()

        assert wrapper._env is None

    def test_wrapper_truncation(self, wrapper):
        """Test wrapper truncates at max_turns."""
        wrapper.reset()

        # Run until truncation
        for _ in range(100):
            _obs, _reward, terminated, truncated, _info = wrapper.step(49)  # end_turn
            if terminated or truncated:
                break

        assert truncated or terminated

    def test_wrapper_with_curriculum(self):
        """Test wrapper with curriculum."""
        config = CivRealmConfig(use_mock=True)
        curriculum = CivRealmCurriculum()
        wrapper = CivRealmWrapper(config, curriculum)

        _obs, info = wrapper.reset()

        assert wrapper.curriculum is not None
        assert "task" in info

    def test_wrapper_clone(self, wrapper):
        """Test wrapper clone creates new instance."""
        wrapper.reset()

        cloned = wrapper.clone()

        assert cloned is not wrapper
        assert cloned.config.task == wrapper.config.task

    def test_wrapper_is_available(self):
        """Test is_available static method."""
        result = CivRealmWrapper.is_available()

        # Should return boolean
        assert isinstance(result, bool)

    def test_wrapper_info_includes_turn(self, wrapper):
        """Test info dict includes turn info."""
        obs, info = wrapper.reset()

        assert "turn" in info
        assert "task" in info

        _obs, _reward, _terminated, _truncated, info = wrapper.step(0)

        assert "turn" in info


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_civrealm_env_basic(self):
        """Test basic environment creation."""
        env = create_civrealm_env(task="battle_ancient", use_mock=True)

        assert isinstance(env, CivRealmWrapper)
        assert env.config.task == "battle_ancient"

    def test_create_civrealm_env_with_kwargs(self):
        """Test environment creation with kwargs."""
        env = create_civrealm_env(
            task="exploration",
            difficulty=5,
            spatial_size=64,
            use_mock=True,
        )

        assert env.config.task == "exploration"
        assert env.config.difficulty == 5
        assert env.config.spatial_size == 64

    def test_create_civrealm_env_with_curriculum(self):
        """Test environment creation with curriculum."""
        env = create_civrealm_env(
            task="development_build_city",
            use_curriculum=True,
            use_mock=True,
        )

        assert env.curriculum is not None

    def test_make_function(self):
        """Test make() factory function."""
        env = make(task="battle_medieval", use_mock=True)

        assert isinstance(env, CivRealmWrapper)
        assert env.config.task == "battle_medieval"


# ============================================================================
# Enum Tests
# ============================================================================


class TestEnums:
    """Tests for enum definitions."""

    def test_terrain_type_values(self):
        """Test TerrainType enum values."""
        assert TerrainType.UNKNOWN.value == 0
        assert TerrainType.OCEAN.value == 1
        assert TerrainType.PLAINS.value == 5
        assert TerrainType.MOUNTAINS.value == 14

    def test_unit_category_values(self):
        """Test UnitCategory enum values."""
        assert UnitCategory.LAND_MILITARY.value == 0
        assert UnitCategory.SETTLER.value == 4
        assert UnitCategory.WORKER.value == 3

    def test_action_type_values(self):
        """Test ActionType enum values."""
        assert ActionType.END_TURN.value == 0
        assert ActionType.UNIT_MOVE.value == 1
        assert ActionType.CITY_PRODUCE.value == 6


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full environment interaction."""

    def test_full_episode_mock(self):
        """Test running a full episode with mock environment."""
        config = CivRealmConfig(
            task="development_build_city",
            max_turns=10,
            use_mock=True,
        )
        env = CivRealmWrapper(config)

        obs, _info = env.reset()
        total_reward = 0.0
        steps = 0

        while steps < 20:  # Safety limit
            mask = env.get_valid_actions()
            valid_actions = np.where(mask > 0)[0]
            action = np.random.choice(valid_actions)

            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        assert steps > 0
        env.close()

    def test_curriculum_progression(self):
        """Test curriculum progression through games."""
        config = CivRealmConfig(use_mock=True, max_turns=5)
        curriculum = CivRealmCurriculum()
        env = CivRealmWrapper(config, curriculum)

        # Play several episodes
        for _ in range(3):
            obs, _info = env.reset()

            # Quick game
            for _ in range(5):
                mask = env.get_valid_actions()
                action = np.argmax(mask)  # Just pick first valid
                _obs, _reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

        # Should have recorded games
        progress = curriculum.get_progress()
        assert progress["total_games"] >= 3

        env.close()

    def test_observation_consistency(self):
        """Test observation shapes are consistent across steps."""
        config = CivRealmConfig(use_mock=True)
        env = CivRealmWrapper(config)

        obs1, _ = env.reset()
        obs2, _, _, _, _ = env.step(0)

        assert obs1["spatial"].shape == obs2["spatial"].shape
        assert obs1["vector"].shape == obs2["vector"].shape
        assert obs1["action_mask"].shape == obs2["action_mask"].shape

        env.close()


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_step_after_done(self):
        """Test stepping after episode is done."""
        config = CivRealmConfig(use_mock=True, max_turns=1)
        env = CivRealmWrapper(config)

        env.reset()
        env.step(0)  # Should end episode

        # Step again - should handle gracefully
        _obs, reward, terminated, truncated, _info = env.step(0)

        assert terminated or truncated
        assert reward == 0.0

        env.close()

    def test_empty_available_actions(self):
        """Test handling empty available actions."""
        config = CivRealmConfig(use_mock=True)
        encoder = CivRealmEncoder(config)

        # Observation with no available actions
        obs = {
            "turn": 0,
            "player_id": 0,
            "map": {"terrain": {}, "visibility": {}},
            "units": {},
            "cities": {},
            "player": {},
            "available_actions": {},
        }

        encoded = encoder.encode_observation(obs)

        # Should still have end_turn valid
        assert encoded["action_mask"][-1] == 1.0

    def test_out_of_bounds_positions(self):
        """Test encoder handles out-of-bounds positions."""
        config = CivRealmConfig(spatial_size=8, use_mock=True)
        encoder = CivRealmEncoder(config)

        # Observation with out-of-bounds positions
        obs = {
            "turn": 0,
            "player_id": 0,
            "map": {"terrain": {(100, 100): {"terrain": 5}}, "visibility": {}},
            "units": {"unit_0": {"owner": 0, "position": {"x": 100, "y": 100}}},
            "cities": {},
            "player": {},
            "available_actions": {},
        }

        # Should not raise
        encoded = encoder.encode_observation(obs)
        assert encoded["spatial"].shape == (51, 8, 8)

    def test_malformed_position_keys(self):
        """Test encoder handles malformed position keys."""
        config = CivRealmConfig(spatial_size=8, use_mock=True)
        encoder = CivRealmEncoder(config)

        obs = {
            "turn": 0,
            "player_id": 0,
            "map": {
                "terrain": {
                    "(0, 0)": {"terrain": 5},  # String key
                    (1, 1): {"terrain": 6},  # Tuple key
                },
                "visibility": {},
            },
            "units": {},
            "cities": {},
            "player": {},
            "available_actions": {},
        }

        # Should not raise
        encoded = encoder.encode_observation(obs)
        assert encoded["spatial"].shape == (51, 8, 8)
