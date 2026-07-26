"""Unit tests for Human Interface module.

Tests cover:
- Configuration loading and validation
- Input parsing and handling
- Game controller turn management
- Visualization rendering
- Replay recording and playback
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from alpha_zero.human_interface.config import (
    BoardTheme,
    HumanInterfaceConfig,
    InputMode,
)
from alpha_zero.human_interface.game_controller import (
    GameController,
    GamePhase,
    TurnResult,
)
from alpha_zero.human_interface.input_handler import (
    ActionParser,
    ActionType,
    HumanInputHandler,
)
from alpha_zero.human_interface.replay import (
    GameRecorder,
    GameRecording,
    GameReplay,
    ReplayManager,
)
from alpha_zero.human_interface.visualizer import StateVisualizer


class TestHumanInterfaceConfig:
    """Tests for HumanInterfaceConfig."""

    def test_default_initialization(self):
        """Test default configuration."""
        config = HumanInterfaceConfig()

        assert config.thinking_time_seconds == 10.0
        assert config.show_ai_evaluation is True
        assert config.board_theme == "classic"
        assert config.input_mode == "coordinate"

    def test_custom_initialization(self):
        """Test custom configuration."""
        config = HumanInterfaceConfig(
            thinking_time_seconds=20.0,
            board_theme="dark",
            confirm_moves=False,
        )

        assert config.thinking_time_seconds == 20.0
        assert config.board_theme == "dark"
        assert config.confirm_moves is False

    def test_validation_thinking_time(self):
        """Test thinking time validation."""
        with pytest.raises(ValueError):
            HumanInterfaceConfig(thinking_time_seconds=-1)

    def test_validation_max_thinking_time(self):
        """Test max thinking time validation."""
        with pytest.raises(ValueError):
            HumanInterfaceConfig(
                thinking_time_seconds=30,
                max_thinking_time=20,  # Less than thinking_time
            )

    def test_validation_board_theme(self):
        """Test board theme validation."""
        with pytest.raises(ValueError):
            HumanInterfaceConfig(board_theme="invalid_theme")

    def test_validation_input_mode(self):
        """Test input mode validation."""
        with pytest.raises(ValueError):
            HumanInterfaceConfig(input_mode="invalid_mode")

    def test_get_enums(self):
        """Test enum getters."""
        config = HumanInterfaceConfig(board_theme="dark", input_mode="index")

        assert config.get_board_theme_enum() == BoardTheme.DARK
        assert config.get_input_mode_enum() == InputMode.INDEX

    def test_to_dict(self):
        """Test serialization."""
        config = HumanInterfaceConfig(thinking_time_seconds=15.0)
        data = config.to_dict()

        assert data["thinking_time_seconds"] == 15.0
        assert "board_theme" in data

    def test_save_and_load(self):
        """Test save and load from YAML."""
        config = HumanInterfaceConfig(
            thinking_time_seconds=25.0,
            confirm_moves=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.save(path)

            loaded = HumanInterfaceConfig.from_yaml(path)

            assert loaded.thinking_time_seconds == 25.0
            assert loaded.confirm_moves is False


class TestActionParser:
    """Tests for ActionParser."""

    @pytest.fixture
    def parser(self):
        """Create parser with 10x10 board."""
        return ActionParser(board_size=(10, 10))

    def test_parse_coordinate_a1(self, parser):
        """Test parsing A1 coordinate."""
        result = parser.parse("A1")

        assert result.action_type == ActionType.MOVE
        assert result.action_index == 0
        assert result.coordinate == "A1"

    def test_parse_coordinate_b3(self, parser):
        """Test parsing B3 coordinate."""
        result = parser.parse("B3")

        assert result.action_type == ActionType.MOVE
        assert result.action_index == 2 * 10 + 1  # row 2, col 1
        assert result.coordinate == "B3"

    def test_parse_coordinate_lowercase(self, parser):
        """Test lowercase coordinate."""
        result = parser.parse("c5")

        assert result.action_type == ActionType.MOVE
        assert result.coordinate == "C5"

    def test_parse_coordinate_out_of_bounds(self, parser):
        """Test out of bounds coordinate."""
        result = parser.parse("Z99")

        assert result.action_type == ActionType.INVALID
        assert "out of bounds" in result.error_message.lower()

    def test_parse_invalid_format(self, parser):
        """Test invalid format."""
        result = parser.parse("xyz")

        assert result.action_type == ActionType.INVALID

    def test_parse_with_valid_actions(self, parser):
        """Test parsing with valid action list."""
        result = parser.parse("A1", valid_actions=[0, 1, 2])
        assert result.action_type == ActionType.MOVE

        result = parser.parse("A2", valid_actions=[0, 1, 2])
        assert result.action_type == ActionType.INVALID
        assert "not a valid move" in result.error_message.lower()

    def test_parse_undo_command(self, parser):
        """Test undo command."""
        result = parser.parse("undo")
        assert result.action_type == ActionType.UNDO

        result = parser.parse("u")
        assert result.action_type == ActionType.UNDO

    def test_parse_resign_command(self, parser):
        """Test resign command."""
        result = parser.parse("resign")
        assert result.action_type == ActionType.RESIGN

        result = parser.parse("q")
        assert result.action_type == ActionType.RESIGN

    def test_parse_help_command(self, parser):
        """Test help command."""
        result = parser.parse("help")
        assert result.action_type == ActionType.HELP

        result = parser.parse("?")
        assert result.action_type == ActionType.HELP

    def test_index_to_coordinate(self, parser):
        """Test index to coordinate conversion."""
        assert parser.index_to_coordinate(0) == "A1"
        assert parser.index_to_coordinate(11) == "B2"
        assert parser.index_to_coordinate(99) == "J10"

    def test_coordinate_to_index(self, parser):
        """Test coordinate to index conversion."""
        assert parser.coordinate_to_index("A1") == 0
        assert parser.coordinate_to_index("B2") == 11
        assert parser.coordinate_to_index("J10") == 99


class TestActionParserIndexMode:
    """Tests for ActionParser in index mode."""

    @pytest.fixture
    def parser(self):
        """Create parser in index mode."""
        return ActionParser(board_size=(10, 10), input_mode=InputMode.INDEX)

    def test_parse_index(self, parser):
        """Test parsing index."""
        result = parser.parse("0")

        assert result.action_type == ActionType.MOVE
        assert result.action_index == 0
        assert result.coordinate == "A1"

    def test_parse_index_with_validation(self, parser):
        """Test parsing with valid actions."""
        result = parser.parse("5", valid_actions=[0, 5, 10])
        assert result.action_type == ActionType.MOVE

        result = parser.parse("7", valid_actions=[0, 5, 10])
        assert result.action_type == ActionType.INVALID


class TestHumanInputHandler:
    """Tests for HumanInputHandler."""

    @pytest.fixture
    def handler(self):
        """Create input handler."""
        config = HumanInterfaceConfig(confirm_moves=False)
        return HumanInputHandler(config, board_size=(10, 10))

    def test_initialization(self, handler):
        """Test handler initialization."""
        assert handler.board_size == (10, 10)

    def test_show_help(self, handler):
        """Test help text generation."""
        help_text = handler.show_help()

        assert "undo" in help_text.lower()
        assert "resign" in help_text.lower()
        assert "help" in help_text.lower()

    def test_show_valid_moves(self, handler):
        """Test valid moves display."""
        valid_actions = [0, 1, 2, 10, 11]
        result = handler.show_valid_moves(valid_actions)

        assert "A1" in result
        assert "B1" in result

    def test_get_action_with_mock_input(self, handler):
        """Test get_action with mocked input."""
        handler.set_input_function(lambda prompt: "A1")

        result = handler.get_action([0, 1, 2])

        assert result.action_type == ActionType.MOVE
        assert result.action_index == 0


class TestTurnResult:
    """Tests for TurnResult."""

    def test_initialization(self):
        """Test turn result initialization."""
        result = TurnResult(
            action=5,
            player=0,
            is_human=True,
            reward=0.0,
        )

        assert result.action == 5
        assert result.player == 0
        assert result.is_human is True

    def test_to_dict(self):
        """Test serialization."""
        result = TurnResult(
            action=5,
            player=0,
            is_human=True,
            reward=1.0,
            game_over=True,
            winner=0,
        )

        data = result.to_dict()

        assert data["action"] == 5
        assert data["winner"] == 0


class TestGameController:
    """Tests for GameController."""

    @pytest.fixture
    def mock_game(self):
        """Create mock game interface."""
        game = MagicMock()
        game.reset.return_value = np.zeros((3, 10, 10))
        game.step.return_value = (
            np.zeros((3, 10, 10)),  # observation
            0.0,  # reward
            False,  # done
            False,  # truncated
            {},  # info
        )
        game.get_valid_actions.return_value = list(range(100))
        game.action_space_size = 100
        return game

    @pytest.fixture
    def controller(self, mock_game):
        """Create game controller."""
        config = HumanInterfaceConfig()
        controller = GameController(config)
        controller.set_game_interface(mock_game)
        return controller

    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller._state.phase == GamePhase.NOT_STARTED

    def test_start_game(self, controller):
        """Test starting a game."""
        state = controller.start_game(human_player=0)

        assert state.phase in (GamePhase.IN_PROGRESS, GamePhase.HUMAN_TURN)
        assert state.human_player == 0
        assert state.turn_number == 0

    def test_is_human_turn(self, controller):
        """Test human turn detection."""
        controller.start_game(human_player=0)

        assert controller.is_human_turn is True

    def test_play_human_move(self, controller, mock_game):
        """Test playing human move."""
        controller.start_game(human_player=0)

        result = controller.play_human_move(5)

        assert result.action == 5
        assert result.is_human is True
        mock_game.step.assert_called_once_with(5)

    def test_invalid_move_raises(self, controller, mock_game):
        """Test that invalid move raises error."""
        mock_game.get_valid_actions.return_value = [0, 1, 2]
        controller.start_game(human_player=0)

        with pytest.raises(ValueError):
            controller.play_human_move(50)

    def test_undo_move(self, controller, mock_game):
        """Test undoing a move."""
        controller.start_game(human_player=0)
        controller.play_human_move(5)

        assert controller.undo_move() is True
        assert controller._state.turn_number == 0

    def test_resign(self, controller):
        """Test resigning."""
        controller.start_game(human_player=0)

        result = controller.resign()

        assert result.game_over is True
        assert result.winner == 1  # Opponent wins
        assert controller.is_game_over is True

    def test_game_summary(self, controller, mock_game):
        """Test game summary."""
        controller.start_game(human_player=0)
        controller.play_human_move(5)

        summary = controller.get_game_summary()

        assert summary["total_turns"] == 1
        assert summary["human_moves"] == 1


class TestTurnOwnership:
    """The controller must agree with the game about whose turn it is.

    GameController used to flip ``current_player`` after every action
    (``1 - current_player``). That is right for a game with one ply per turn,
    but MicroCiv rotates ``current_player`` only inside ``_end_turn`` — a
    player moves every unit first. So after the human moved one unit the web
    layer believed it was the AI's turn and let the AI spend the human's
    remaining units as player 0; the human then issued actions for the AI's
    civ. Neither side ever controlled its own units.
    """

    @staticmethod
    def _controller():
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.human_interface.config import HumanInterfaceConfig
        from alpha_zero.human_interface.game_controller import GameController
        from alpha_zero.web.adapter import MicroCivGameAdapter

        adapter = MicroCivGameAdapter(MicroCivEnv(board_size=8, seed=42))
        controller = GameController(config=HumanInterfaceConfig(), game_interface=adapter)
        controller.start_game(human_player=0)
        return controller, adapter

    def test_controller_tracks_env_current_player(self):
        """Across a whole turn, the two must never disagree."""
        controller, adapter = self._controller()

        for _ in range(8):
            valid = adapter.get_valid_actions()
            if not valid:
                break
            # Prefer a non-end-turn action so we stay inside one player's turn,
            # which is exactly where the old alternation broke down.
            action = next((a for a in valid if a != 0), valid[0])
            result = controller._execute_move(action, controller.current_state.current_player, 0.0)
            if result is None or result.game_over:
                break

            assert controller.current_state.current_player == adapter.current_player, (
                "controller and env disagree about the acting player"
            )

    def test_turn_does_not_change_on_a_non_end_turn_action(self):
        """Moving a unit must not hand the turn over."""
        controller, adapter = self._controller()
        before = adapter.current_player

        valid = adapter.get_valid_actions()
        move = next((a for a in valid if a != 0), None)
        if move is None:
            pytest.skip("no non-end-turn action available from the initial state")
        controller._execute_move(move, before, 0.0)

        assert adapter.current_player == before, "env rotated on a plain move"
        assert controller.current_state.current_player == before

    def test_end_turn_hands_over(self):
        """end_turn (action 0) is what actually rotates the player."""
        controller, adapter = self._controller()
        before = adapter.current_player

        controller._execute_move(0, before, 0.0)

        assert adapter.current_player != before
        assert controller.current_state.current_player == adapter.current_player

    def test_falls_back_to_alternation_without_current_player(self):
        """Interfaces that report no current player keep the old behaviour."""
        from alpha_zero.human_interface.config import HumanInterfaceConfig
        from alpha_zero.human_interface.game_controller import GameController

        interface = MagicMock(spec=["reset", "step", "get_valid_actions", "action_space_size"])
        interface.reset.return_value = np.zeros(4)
        interface.step.return_value = (np.zeros(4), 0.0, False, False, {})
        interface.get_valid_actions.return_value = [0, 1]
        interface.action_space_size = 2

        controller = GameController(config=HumanInterfaceConfig(), game_interface=interface)
        controller.start_game(human_player=0)
        controller._execute_move(1, 0, 0.0)

        assert controller.current_state.current_player == 1


class TestStateVisualizer:
    """Tests for StateVisualizer."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer."""
        config = HumanInterfaceConfig()
        return StateVisualizer(config)

    def test_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer._use_colors is True

    def test_render_board(self, visualizer):
        """Test board rendering."""
        observation = np.zeros((3, 5, 5))
        observation[0, 0, 0] = 1  # Player 1 at A1
        observation[0, 2, 2] = -1  # Player 2 at C3

        output = visualizer.render_board(observation, valid_actions=[1, 2, 3])

        # Should contain coordinate labels
        assert "A" in output or "1" in output

    def test_render_board_without_colors(self, visualizer):
        """Test board rendering without colors."""
        visualizer.disable_colors()

        observation = np.zeros((3, 5, 5))
        output = visualizer.render_board(observation)

        # Should not contain ANSI escape codes
        assert "\033[" not in output

    def test_render_evaluation(self, visualizer):
        """Test evaluation rendering."""
        output = visualizer.render_evaluation(0.3)

        assert "Evaluation" in output

    def test_render_game_info(self, visualizer):
        """Test game info rendering."""
        output = visualizer.render_game_info(
            turn_number=5,
            current_player=0,
            is_human_turn=True,
            valid_move_count=10,
        )

        assert "Turn 5" in output
        assert "Your turn" in output

    def test_render_game_over(self, visualizer):
        """Test game over rendering."""
        output = visualizer.render_game_over(
            winner=0,
            human_player=0,
            summary={"total_turns": 10},
        )

        assert "WIN" in output.upper()


class TestGameRecording:
    """Tests for GameRecording."""

    def test_initialization(self):
        """Test recording initialization."""
        recording = GameRecording(game_id="test123")

        assert recording.game_id == "test123"
        assert len(recording.turns) == 0

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        recording = GameRecording(
            game_id="test123",
            human_player=1,
            turns=[{"action": 5, "player": 0}],
        )

        data = recording.to_dict()
        restored = GameRecording.from_dict(data)

        assert restored.game_id == "test123"
        assert restored.human_player == 1
        assert len(restored.turns) == 1

    def test_save_and_load(self):
        """Test save and load."""
        recording = GameRecording(
            game_id="test123",
            turns=[{"action": 5, "player": 0}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "recording.json"
            recording.save(path)

            loaded = GameRecording.load(path)

            assert loaded.game_id == "test123"
            assert len(loaded.turns) == 1


class TestGameRecorder:
    """Tests for GameRecorder."""

    @pytest.fixture
    def recorder(self):
        """Create recorder."""
        return GameRecorder()

    def test_start_recording(self, recorder):
        """Test starting recording."""
        game_id = recorder.start(human_player=0)

        assert recorder.is_recording is True
        assert game_id is not None

    def test_record_turn(self, recorder):
        """Test recording a turn."""
        recorder.start()

        turn_result = TurnResult(action=5, player=0, is_human=True)
        recorder.record_turn(turn_result)

        recording = recorder.get_current_recording()
        assert len(recording.turns) == 1

    def test_stop_recording(self, recorder):
        """Test stopping recording."""
        recorder.start()
        recorder.record_turn(TurnResult(action=5, player=0, is_human=True))

        recording = recorder.stop()

        assert recorder.is_recording is False
        assert len(recording.turns) == 1


class TestGameReplay:
    """Tests for GameReplay."""

    @pytest.fixture
    def recording(self):
        """Create test recording."""
        return GameRecording(
            game_id="test",
            turns=[
                {"action": 0, "player": 0},
                {"action": 5, "player": 1},
                {"action": 10, "player": 0},
            ],
        )

    @pytest.fixture
    def replay(self, recording):
        """Create replay from recording."""
        return GameReplay(recording)

    def test_initialization(self, replay):
        """Test replay initialization."""
        assert replay.current_turn == 0
        assert replay.total_turns == 3
        assert replay.is_at_start is True

    def test_step_forward(self, replay):
        """Test stepping forward."""
        turn = replay.step_forward()

        assert turn["action"] == 0
        assert replay.current_turn == 1

    def test_step_backward(self, replay):
        """Test stepping backward."""
        replay.step_forward()  # current_turn goes to 1, returns turn 0 (action=0)
        replay.step_forward()  # current_turn goes to 2, returns turn 1 (action=5)
        turn = replay.step_backward()  # current_turn goes to 1, returns turn 1 (action=5)

        # step_backward decrements then returns the turn at that index
        assert turn["action"] == 5
        assert replay.current_turn == 1

    def test_jump_to_turn(self, replay):
        """Test jumping to specific turn."""
        replay.jump_to_turn(2)

        assert replay.current_turn == 2

    def test_is_at_end(self, replay):
        """Test end detection."""
        for _ in range(3):
            replay.step_forward()

        assert replay.is_at_end is True

    def test_iterate_turns(self, replay):
        """Test iterating all turns."""
        turns = list(replay.iterate_turns())

        assert len(turns) == 3
        assert turns[0]["action"] == 0

    def test_get_summary(self, replay):
        """Test getting summary."""
        summary = replay.get_summary()

        assert summary["game_id"] == "test"
        assert summary["total_turns"] == 3


class TestReplayManager:
    """Tests for ReplayManager."""

    @pytest.fixture
    def manager(self):
        """Create replay manager with temp directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HumanInterfaceConfig(
                replay_dir=str(Path(tmpdir) / "replays"),
                save_dir=str(Path(tmpdir) / "saves"),
            )
            yield ReplayManager(config)

    def test_save_and_load(self, manager):
        """Test saving and loading recording."""
        recording = GameRecording(
            game_id="test123",
            turns=[{"action": 5}],
        )

        path = manager.save_recording(recording)
        assert path.exists()

        replay = manager.load_replay("test123")
        assert replay.recording.game_id == "test123"

    def test_list_replays(self, manager):
        """Test listing replays."""
        recording = GameRecording(game_id="test123")
        manager.save_recording(recording)

        replays = manager.list_replays()

        assert len(replays) == 1
        assert replays[0]["game_id"] == "test123"

    def test_delete_replay(self, manager):
        """Test deleting replay."""
        recording = GameRecording(game_id="test123")
        manager.save_recording(recording)

        result = manager.delete_replay("test123")

        assert result is True
        assert len(manager.list_replays()) == 0
