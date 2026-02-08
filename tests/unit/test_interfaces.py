"""Unit tests for game interface components."""

import json
import tempfile
from pathlib import Path

from interfaces.base import (
    ActionType,
    City,
    GameAction,
    GameState,
    InterfaceConfig,
    Position,
    Unit,
)
from interfaces.civ6_bridge import (
    Civ6Bridge,
    Civ6LuaExporter,
    LuaStateParser,
)
from interfaces.log_parser import (
    Civ6LogParser,
    LogEventType,
    MovementLog,
)
from interfaces.mini_games import (
    CityDevelopmentMiniGame,
    CombatMiniGame,
    ExpansionMiniGame,
    MiniGameConfig,
    MiniGameGenerator,
    MiniGameType,
)
from interfaces.unciv_interface import (
    UncivActionSerializer,
    UncivInterface,
    UncivStateParser,
)


class TestPosition:
    """Tests for Position class."""

    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(x=5, y=10)
        assert pos.x == 5
        assert pos.y == 10

    def test_position_to_tuple(self):
        """Test converting position to tuple."""
        pos = Position(x=3, y=7)
        assert pos.to_tuple() == (3, 7)

    def test_position_from_tuple(self):
        """Test creating position from tuple."""
        pos = Position.from_tuple((4, 8))
        assert pos.x == 4
        assert pos.y == 8

    def test_position_distance(self):
        """Test Manhattan distance calculation."""
        pos1 = Position(x=0, y=0)
        pos2 = Position(x=3, y=4)
        assert pos1.distance_to(pos2) == 7

    def test_position_hash(self):
        """Test position can be used in sets and dicts."""
        pos1 = Position(x=1, y=2)
        pos2 = Position(x=1, y=2)
        positions = {pos1}
        assert pos2 in positions


class TestUnit:
    """Tests for Unit class."""

    def test_unit_creation(self):
        """Test creating a unit."""
        unit = Unit(
            id="unit1",
            type="warrior",
            player=0,
            position=Position(x=5, y=5),
            health=100,
            movement=2,
            strength=20,
        )
        assert unit.id == "unit1"
        assert unit.type == "warrior"
        assert unit.player == 0
        assert unit.health == 100

    def test_unit_to_dict(self):
        """Test converting unit to dictionary."""
        unit = Unit(
            id="unit1",
            type="warrior",
            player=0,
            position=Position(x=5, y=5),
        )
        d = unit.to_dict()
        assert d["id"] == "unit1"
        assert d["type"] == "warrior"
        assert d["position"] == (5, 5)

    def test_unit_from_dict(self):
        """Test creating unit from dictionary."""
        d = {
            "id": "unit2",
            "type": "archer",
            "player": 1,
            "position": [3, 7],
            "health": 80,
        }
        unit = Unit.from_dict(d)
        assert unit.id == "unit2"
        assert unit.type == "archer"
        assert unit.position.x == 3
        assert unit.health == 80


class TestCity:
    """Tests for City class."""

    def test_city_creation(self):
        """Test creating a city."""
        city = City(
            id="city1",
            name="Capital",
            player=0,
            position=Position(x=10, y=10),
            population=3,
        )
        assert city.name == "Capital"
        assert city.population == 3

    def test_city_to_dict(self):
        """Test converting city to dictionary."""
        city = City(
            id="city1",
            name="Test City",
            player=0,
            position=Position(x=5, y=5),
        )
        d = city.to_dict()
        assert d["name"] == "Test City"
        assert d["position"] == (5, 5)


class TestGameState:
    """Tests for GameState class."""

    def test_game_state_creation(self):
        """Test creating a game state."""
        state = GameState(
            turn=10,
            current_player=1,
            num_players=2,
        )
        assert state.turn == 10
        assert state.current_player == 1

    def test_game_state_get_unit_by_id(self):
        """Test finding unit by ID."""
        state = GameState()
        unit = Unit(
            id="test_unit",
            type="warrior",
            player=0,
            position=Position(x=0, y=0),
        )
        state.units.append(unit)

        found = state.get_unit_by_id("test_unit")
        assert found is not None
        assert found.id == "test_unit"

        not_found = state.get_unit_by_id("nonexistent")
        assert not_found is None

    def test_game_state_get_own_units(self):
        """Test getting own units."""
        state = GameState(current_player=0)
        state.units = [
            Unit(id="u1", type="warrior", player=0, position=Position(0, 0)),
            Unit(id="u2", type="warrior", player=1, position=Position(1, 1)),
            Unit(id="u3", type="archer", player=0, position=Position(2, 2)),
        ]

        own = state.get_own_units()
        assert len(own) == 2
        assert all(u.player == 0 for u in own)

    def test_game_state_to_encoder_dict(self):
        """Test converting state to encoder format."""
        state = GameState(turn=5, current_player=0)
        state.units.append(Unit(id="u1", type="warrior", player=0, position=Position(0, 0)))
        state.cities.append(City(id="c1", name="City", player=0, position=Position(5, 5)))

        d = state.to_encoder_dict()
        assert d["turn"] == 5
        assert len(d["units"]) == 1
        assert len(d["cities"]) == 1


class TestGameAction:
    """Tests for GameAction class."""

    def test_end_turn_action(self):
        """Test creating end turn action."""
        action = GameAction.end_turn()
        assert action.action_type == ActionType.END_TURN

    def test_move_action(self):
        """Test creating move action."""
        action = GameAction.move("unit1", Position(5, 5))
        assert action.action_type == ActionType.MOVE
        assert action.unit_id == "unit1"
        assert action.target_position.x == 5

    def test_attack_action(self):
        """Test creating attack action."""
        action = GameAction.attack("unit1", Position(3, 3))
        assert action.action_type == ActionType.ATTACK
        assert action.unit_id == "unit1"

    def test_produce_action(self):
        """Test creating produce action."""
        action = GameAction.produce("city1", "warrior")
        assert action.action_type == ActionType.PRODUCE
        assert action.city_id == "city1"
        assert action.production_type == "warrior"

    def test_action_to_json(self):
        """Test serializing action to JSON."""
        action = GameAction.move("unit1", Position(5, 5))
        json_str = action.to_json()
        data = json.loads(json_str)
        assert data["type"] == "move"
        assert data["unit_id"] == "unit1"

    def test_action_from_dict(self):
        """Test creating action from dictionary."""
        d = {
            "type": "attack",
            "unit_id": "u1",
            "target": [3, 4],
        }
        action = GameAction.from_dict(d)
        assert action.action_type == ActionType.ATTACK
        assert action.unit_id == "u1"
        assert action.target_position.x == 3


class TestLuaStateParser:
    """Tests for LuaStateParser."""

    def test_parse_basic_state(self):
        """Test parsing basic game state."""
        parser = LuaStateParser(player_id=0)
        data = {
            "turn": 50,
            "currentPlayer": 0,
            "numPlayers": 2,
            "mapWidth": 80,
            "mapHeight": 50,
            "units": [],
            "cities": [],
            "terrain": {},
            "visibility": {},
            "gold": 100,
            "era": 1,
        }

        state = parser.parse(data)
        assert state.turn == 50
        assert state.current_player == 0
        assert state.gold == 100

    def test_parse_units(self):
        """Test parsing units from Lua format."""
        parser = LuaStateParser()
        data = {
            "turn": 1,
            "units": [
                {
                    "id": "1",
                    "type": "UNIT_WARRIOR",
                    "owner": 0,
                    "x": 10,
                    "y": 20,
                    "damage": 25,
                    "movesRemaining": 2,
                    "combatStrength": 20,
                }
            ],
            "cities": [],
            "terrain": {},
            "visibility": {},
        }

        state = parser.parse(data)
        assert len(state.units) == 1
        assert state.units[0].type == "UNIT_WARRIOR"
        assert state.units[0].health == 75  # 100 - 25 damage


class TestCiv6LuaExporter:
    """Tests for Civ6LuaExporter."""

    def test_generate_mod_files(self):
        """Test generating Lua mod files."""
        exporter = Civ6LuaExporter()
        files = exporter.generate_mod()

        assert "ALPHAZERO_BRIDGE.modinfo" in files
        assert "StateExporter.lua" in files
        assert "ActionExecutor.lua" in files
        assert "Bridge.lua" in files

    def test_write_mod(self):
        """Test writing mod to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Civ6LuaExporter()
            mod_dir = exporter.write_mod(Path(tmpdir) / "mod")

            assert mod_dir.exists()
            assert (mod_dir / "Bridge.lua").exists()
            assert (mod_dir / "StateExporter.lua").exists()


class TestCiv6Bridge:
    """Tests for Civ6Bridge."""

    def test_bridge_initialization(self):
        """Test bridge initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = Civ6Bridge(
                state_dir=Path(tmpdir) / "state",
                action_dir=Path(tmpdir) / "action",
            )
            assert not bridge.is_connected

    def test_bridge_connect_creates_directories(self):
        """Test that connect creates directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            action_dir = Path(tmpdir) / "action"

            bridge = Civ6Bridge(
                state_dir=state_dir,
                action_dir=action_dir,
            )
            bridge.connect()

            assert state_dir.exists()
            assert action_dir.exists()
            bridge.disconnect()

    def test_bridge_read_state(self):
        """Test reading state from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()

            # Write test state
            state_data = {
                "turn": 10,
                "currentPlayer": 0,
                "units": [],
                "cities": [],
                "terrain": {},
                "visibility": {},
            }
            (state_dir / "state.json").write_text(json.dumps(state_data))

            bridge = Civ6Bridge(state_dir=state_dir)
            bridge.connect()

            state = bridge.read_state()
            assert state.turn == 10
            bridge.disconnect()


class TestUncivStateParser:
    """Tests for UncivStateParser."""

    def test_parse_save_data(self):
        """Test parsing Unciv save data."""
        parser = UncivStateParser(player_civ="Babylon")
        data = {
            "gameInfo": {"turns": 25},
            "civilizations": [
                {"civName": "Babylon", "isDefeated": False, "units": [], "cities": []},
                {"civName": "Egypt", "isDefeated": False, "units": [], "cities": []},
            ],
            "tileMap": {
                "mapParameters": {"size": {"width": 40, "height": 25}},
                "tileList": [],
            },
        }

        state = parser.parse(data)
        assert state.turn == 25
        assert state.current_player == 0  # Babylon is first


class TestUncivActionSerializer:
    """Tests for UncivActionSerializer."""

    def test_serialize_end_turn(self):
        """Test serializing end turn action."""
        serializer = UncivActionSerializer()
        save_data = {
            "civilizations": [
                {"civName": "Babylon", "hasPlayedTurn": False, "isDefeated": False},
                {"civName": "Egypt", "hasPlayedTurn": True, "isDefeated": False},
            ],
            "gameInfo": {"turns": 1},
        }

        action = GameAction.end_turn()
        result = serializer.serialize(action, save_data, 0)

        # Since both players have now ended, turn should advance
        assert result["gameInfo"]["turns"] == 2
        # And flags should be reset
        assert result["civilizations"][0]["hasPlayedTurn"] is False

    def test_serialize_move(self):
        """Test serializing move action."""
        serializer = UncivActionSerializer()
        save_data = {
            "civilizations": [
                {
                    "civName": "Babylon",
                    "units": [{"id": "u1", "currentTile": "5,5", "currentMovement": 2}],
                },
            ],
        }

        action = GameAction.move("u1", Position(6, 5))
        result = serializer.serialize(action, save_data, 0)

        assert result["civilizations"][0]["units"][0]["currentTile"] == "6,5"


class TestUncivInterface:
    """Tests for UncivInterface."""

    def test_create_new_game(self):
        """Test creating a new game."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "game.json"
            interface = UncivInterface(save_path=save_path)

            state = interface.create_new_game(
                player_civs=["Babylon", "Egypt"],
                map_size=(20, 15),
            )

            assert state.num_players == 2
            assert len(state.units) == 4  # 2 settlers + 2 warriors

    def test_save_and_load(self):
        """Test saving and loading game."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "game.json"
            interface = UncivInterface(save_path=save_path)

            # Create game
            interface.create_new_game(player_civs=["Babylon"])
            interface.save()

            # Load in new interface
            interface2 = UncivInterface(save_path=save_path)
            interface2.connect()
            state = interface2.read_state()

            assert state.turn == 0


class TestCiv6LogParser:
    """Tests for Civ6LogParser."""

    def test_parse_turn_start(self):
        """Test parsing turn start event."""
        parser = Civ6LogParser()
        line = "[10:30:45] Turn 25 has started for player 0"
        entry = parser.parse_line(line)

        assert entry is not None
        assert entry.event_type == LogEventType.TURN_START
        assert entry.data["turn"] == 25
        assert entry.data["player_id"] == 0

    def test_parse_unit_move(self):
        """Test parsing unit move event."""
        parser = Civ6LogParser()
        line = "[10:30:46] Unit Warrior (5) moved from (10,20) to (11,20)"
        entry = parser.parse_line(line)

        assert entry is not None
        assert entry.event_type == LogEventType.UNIT_MOVE
        assert entry.data["unit_type"] == "Warrior"
        assert entry.data["from_x"] == 10
        assert entry.data["to_x"] == 11

    def test_movement_log_from_entry(self):
        """Test creating MovementLog from entry."""
        parser = Civ6LogParser()
        line = "[10:30:46] Unit Warrior (5) moved from (10,20) to (11,20)"
        entry = parser.parse_line(line)

        movement = MovementLog.from_log_entry(entry)
        assert movement is not None
        assert movement.unit_type == "Warrior"
        assert movement.from_position.x == 10
        assert movement.to_position.x == 11

    def test_parse_file(self):
        """Test parsing entire log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            log_path.write_text(
                "[10:30:45] Turn 1 has started for player 0\n"
                "[10:30:46] Unit Warrior (1) moved from (5,5) to (6,5)\n"
                "[10:30:47] Player 0 ended their turn\n"
            )

            parser = Civ6LogParser(log_dir=Path(tmpdir), log_file="test.log")
            entries = parser.parse_file()

            assert len(entries) == 3


class TestMiniGameConfig:
    """Tests for MiniGameConfig."""

    def test_config_creation(self):
        """Test creating config."""
        config = MiniGameConfig(
            game_type=MiniGameType.COMBAT_1V1,
            board_size=8,
            max_turns=50,
        )
        assert config.game_type == MiniGameType.COMBAT_1V1
        assert config.board_size == 8

    def test_with_difficulty(self):
        """Test creating config with different difficulty."""
        config = MiniGameConfig(game_type=MiniGameType.COMBAT_1V1)
        harder = config.with_difficulty(0.8)

        assert harder.difficulty == 0.8
        assert config.difficulty == 0.5


class TestCombatMiniGame:
    """Tests for CombatMiniGame."""

    def test_generate_initial_state(self):
        """Test generating initial combat state."""
        config = MiniGameConfig(
            game_type=MiniGameType.COMBAT_1V1,
            board_size=8,
        )
        game = CombatMiniGame(config)
        state = game.state

        # Should have units for both players
        p0_units = [u for u in state.units if u.player == 0]
        p1_units = [u for u in state.units if u.player == 1]
        assert len(p0_units) >= 1
        assert len(p1_units) >= 1

    def test_game_step(self):
        """Test stepping the game."""
        config = MiniGameConfig(
            game_type=MiniGameType.COMBAT_1V1,
            board_size=8,
            random_seed=42,
        )
        game = CombatMiniGame(config)

        # End turn
        action = GameAction.end_turn()
        new_state, _reward, _done = game.step(action)

        assert new_state.current_player == 1

    def test_combat_victory(self):
        """Test combat victory detection."""
        config = MiniGameConfig(
            game_type=MiniGameType.COMBAT_1V1,
            board_size=8,
        )
        game = CombatMiniGame(config)

        # Remove all player 1 units
        game.state.units = [u for u in game.state.units if u.player == 0]

        winner = game.check_victory()
        assert winner == 0

    def test_get_valid_actions(self):
        """Test getting valid actions."""
        config = MiniGameConfig(
            game_type=MiniGameType.COMBAT_1V1,
            board_size=8,
        )
        game = CombatMiniGame(config)

        actions = game.get_valid_actions()
        # Should at least have end turn and skip actions
        assert len(actions) > 0
        assert any(a.action_type == ActionType.END_TURN for a in actions)


class TestCityDevelopmentMiniGame:
    """Tests for CityDevelopmentMiniGame."""

    def test_generate_initial_state(self):
        """Test generating initial city state."""
        config = MiniGameConfig(
            game_type=MiniGameType.CITY_PRODUCTION,
            board_size=12,
        )
        game = CityDevelopmentMiniGame(config)
        state = game.state

        # Should have cities for both players
        p0_cities = [c for c in state.cities if c.player == 0]
        p1_cities = [c for c in state.cities if c.player == 1]
        assert len(p0_cities) >= 1
        assert len(p1_cities) >= 1


class TestExpansionMiniGame:
    """Tests for ExpansionMiniGame."""

    def test_generate_initial_state(self):
        """Test generating initial expansion state."""
        config = MiniGameConfig(
            game_type=MiniGameType.EXPANSION_EARLY,
            board_size=16,
        )
        game = ExpansionMiniGame(config)
        state = game.state

        # Should have settlers
        settlers = [u for u in state.units if "settler" in u.type.lower()]
        assert len(settlers) >= 2


class TestMiniGameGenerator:
    """Tests for MiniGameGenerator."""

    def test_create_game(self):
        """Test creating a game."""
        generator = MiniGameGenerator(seed=42)
        game = generator.create_game(MiniGameType.COMBAT_1V1)

        assert game is not None
        assert game.state is not None

    def test_create_batch(self):
        """Test creating a batch of games."""
        generator = MiniGameGenerator(seed=42)
        games = generator.create_batch(
            game_types=[MiniGameType.COMBAT_1V1, MiniGameType.COMBAT_SMALL],
            batch_size=10,
        )

        assert len(games) == 10

    def test_generate_curriculum(self):
        """Test generating curriculum."""
        generator = MiniGameGenerator(seed=42)
        curriculum = generator.generate_curriculum(
            stages=2,
            games_per_stage=5,
        )

        assert len(curriculum) == 2
        assert len(curriculum[0]) == 5
        assert len(curriculum[1]) == 5

    def test_create_random_game(self):
        """Test creating random game."""
        generator = MiniGameGenerator(seed=42)
        game1 = generator.create_random_game()
        game2 = generator.create_random_game()

        # Both should be valid games
        assert game1.state is not None
        assert game2.state is not None


class TestInterfaceConfig:
    """Tests for InterfaceConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = InterfaceConfig()
        assert config.poll_interval_ms == 100
        assert config.timeout_seconds == 30.0

    def test_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            "state_dir": "/path/to/state",
            "poll_interval_ms": 200,
            "board_size": 32,
        }
        config = InterfaceConfig.from_dict(d)

        assert config.state_dir == Path("/path/to/state")
        assert config.poll_interval_ms == 200
        assert config.board_size == 32
