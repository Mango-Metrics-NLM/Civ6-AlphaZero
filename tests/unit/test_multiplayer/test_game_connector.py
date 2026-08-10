"""Tests for multiplayer game connectors."""

from __future__ import annotations

import json

import pytest

from alpha_zero.multiplayer.game_connector import (
    GameConnectorConfig,
    MockGameConnector,
    UncivMultiplayerConnector,
    create_game_connector,
)


class TestGameConnectorConfig:
    """Tests for GameConnectorConfig."""

    def test_default_values(self):
        config = GameConnectorConfig()
        assert config.connector_type == "unciv"
        assert config.player_id == 0
        assert config.num_players == 2

    def test_custom_values(self):
        config = GameConnectorConfig(
            connector_type="mock",
            player_id=2,
            num_players=4,
            poll_interval_seconds=0.5,
        )
        assert config.connector_type == "mock"
        assert config.player_id == 2
        assert config.num_players == 4


class TestMockGameConnector:
    """Tests for MockGameConnector."""

    @pytest.fixture
    def connector(self):
        config = GameConnectorConfig(connector_type="mock", num_players=3)
        return MockGameConnector(config)

    def test_connect(self, connector):
        assert connector.connect() is True
        assert connector.is_connected is True

    def test_disconnect(self, connector):
        connector.connect()
        connector.disconnect()
        assert connector.is_connected is False

    def test_read_state(self, connector):
        connector.connect()
        state = connector.read_game_state()
        assert state is not None
        assert state["current_player"] == 0
        assert state["num_players"] == 3

    def test_read_state_disconnected(self, connector):
        state = connector.read_game_state()
        assert state is None

    def test_submit_action(self, connector):
        connector.connect()
        state = connector.read_game_state()
        result = connector.submit_action(state, action=0)
        assert result is True

    def test_submit_advances_player(self, connector):
        connector.connect()
        state = connector.read_game_state()
        assert state["current_player"] == 0

        connector.submit_action(state, action=0)
        state = connector.read_game_state()
        assert state["current_player"] == 1

        connector.submit_action(state, action=1)
        state = connector.read_game_state()
        assert state["current_player"] == 2

    def test_turn_advances_on_wrap(self, connector):
        connector.connect()
        state = connector.read_game_state()
        assert state["turn_number"] == 0

        # Go through all 3 players
        for _ in range(3):
            state = connector.read_game_state()
            connector.submit_action(state, action=0)

        state = connector.read_game_state()
        assert state["turn_number"] == 1

    def test_wait_for_turn_immediate(self, connector):
        """If it's already our turn, return immediately."""
        connector.connect()
        state = connector.wait_for_turn()
        assert state is not None

    def test_wait_for_turn_not_our_turn(self, connector):
        """If not our turn, returns None (mock doesn't advance)."""
        config = GameConnectorConfig(connector_type="mock", player_id=1, num_players=3)
        conn = MockGameConnector(config)
        conn.connect()
        state = conn.wait_for_turn()
        assert state is None  # Player 0's turn, not player 1

    def test_is_game_over(self, connector):
        connector.connect()
        assert connector.is_game_over() is False

    def test_context_manager(self, connector):
        with connector as c:
            assert c.is_connected is True
        assert connector.is_connected is False

    def test_custom_initial_state(self):
        config = GameConnectorConfig(connector_type="mock")
        custom_state = {"current_player": 2, "game_over": True}
        conn = MockGameConnector(config, initial_state=custom_state)
        conn.connect()
        assert conn.is_game_over() is True
        state = conn.read_game_state()
        assert state["current_player"] == 2


class TestUncivMultiplayerConnector:
    """Tests for UncivMultiplayerConnector."""

    @pytest.fixture
    def save_dir(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def connector(self, save_dir):
        config = GameConnectorConfig(
            connector_type="unciv",
            save_dir=str(save_dir),
            player_id=0,
            num_players=2,
        )
        return UncivMultiplayerConnector(config)

    def test_connect_existing_dir(self, connector):
        assert connector.connect() is True
        assert connector.is_connected is True

    def test_connect_nonexistent_dir(self, tmp_path):
        config = GameConnectorConfig(
            save_dir=str(tmp_path / "nonexistent"),
        )
        conn = UncivMultiplayerConnector(config)
        assert conn.connect() is False

    def test_read_state_no_files(self, connector):
        connector.connect()
        state = connector.read_game_state()
        assert state is None

    def test_read_state_with_save_file(self, connector, save_dir):
        """Can read a valid Unciv-style save file."""
        save_data = {
            "turns": 5,
            "currentPlayer": "Rome",
            "gameOver": False,
            "civilizations": [
                {"civName": "Rome", "isDefeated": False, "gold": 100, "cities": [{}]},
                {"civName": "Egypt", "isDefeated": False, "gold": 50, "cities": [{}]},
            ],
        }
        save_file = save_dir / "test_game.json"
        with open(save_file, "w") as f:
            json.dump(save_data, f)

        connector.connect()
        state = connector.read_game_state()
        assert state is not None
        assert state["turn_number"] == 5
        assert state["current_player"] == 0  # Rome is index 0
        assert state["num_players"] == 2
        assert len(state["players"]) == 2

    def test_read_state_invalid_json(self, connector, save_dir):
        """Handles corrupted save files gracefully."""
        save_file = save_dir / "bad.json"
        save_file.write_text("not valid json {{{")

        connector.connect()
        state = connector.read_game_state()
        assert state is None

    def test_submit_action_raises_not_implemented(self, connector, save_dir):
        """submit_action raises NotImplementedError (action decoder not yet integrated)."""
        save_data = {
            "turns": 1,
            "currentPlayer": "Rome",
            "civilizations": [{"civName": "Rome"}],
        }
        save_file = save_dir / "game.json"
        with open(save_file, "w") as f:
            json.dump(save_data, f)

        connector.connect()
        state = connector.read_game_state()
        with pytest.raises(NotImplementedError, match="action decoder"):
            connector.submit_action(state, action=0)

    def test_disconnect(self, connector):
        connector.connect()
        connector.disconnect()
        assert connector.is_connected is False


class TestCreateGameConnector:
    """Tests for create_game_connector factory."""

    def test_create_mock(self):
        config = GameConnectorConfig(connector_type="mock")
        conn = create_game_connector(config)
        assert isinstance(conn, MockGameConnector)

    def test_create_unciv(self, tmp_path):
        config = GameConnectorConfig(connector_type="unciv", save_dir=str(tmp_path))
        conn = create_game_connector(config)
        assert isinstance(conn, UncivMultiplayerConnector)

    def test_unknown_type(self):
        config = GameConnectorConfig(connector_type="unknown")
        with pytest.raises(ValueError, match="Unknown connector_type"):
            create_game_connector(config)
