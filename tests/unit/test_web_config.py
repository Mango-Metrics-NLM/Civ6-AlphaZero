"""Unit tests for web configuration module."""

from __future__ import annotations

import pytest
import yaml

from alpha_zero.human_interface.config import BoardTheme, HumanInterfaceConfig
from alpha_zero.web.config import VALID_LOG_LEVELS, VALID_OPPONENT_TYPES, WebConfig


class TestWebConfigDefaults:
    """Tests for WebConfig default values."""

    def test_default_initialization(self):
        """Test default values are set correctly."""
        config = WebConfig()

        # Server defaults
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.debug is False
        assert config.log_level == "info"
        assert config.workers == 1

        # Game defaults
        assert config.board_size == 16
        assert config.max_turns == 200
        assert config.default_thinking_time == 5.0
        assert config.max_concurrent_sessions == 10
        assert config.session_timeout_seconds == 3600.0

        # AI defaults
        assert config.checkpoint_path is None
        assert config.use_mcts is True
        assert config.mcts_simulations == 50
        assert config.ai_temperature == 0.0
        assert config.device == "cpu"
        assert config.default_opponent == "random"

        # Display defaults
        assert config.board_theme == "classic"
        assert config.show_ai_evaluation is True
        assert config.show_move_probabilities is True
        assert config.show_mcts_stats is True

        # Replay/save defaults
        assert config.enable_replay is True
        assert config.replay_dir == "replays"
        assert config.save_dir == "saves/web_games"

        # Security defaults
        assert config.cors_origins == ["*"]

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        config = WebConfig(
            host="127.0.0.1",
            port=3000,
            debug=True,
            log_level="debug",
            workers=4,
            board_size=8,
            max_turns=50,
            default_thinking_time=2.0,
            max_concurrent_sessions=5,
            session_timeout_seconds=1800.0,
            use_mcts=False,
            mcts_simulations=25,
            ai_temperature=0.5,
            device="cuda",
            default_opponent="heuristic",
            board_theme="dark",
            show_ai_evaluation=False,
            show_move_probabilities=False,
            show_mcts_stats=False,
            enable_replay=False,
            replay_dir="custom_replays",
            save_dir="custom_saves",
            cors_origins=["http://localhost:3000"],
        )

        assert config.host == "127.0.0.1"
        assert config.port == 3000
        assert config.debug is True
        assert config.log_level == "debug"
        assert config.workers == 4
        assert config.board_size == 8
        assert config.max_turns == 50
        assert config.default_thinking_time == 2.0
        assert config.max_concurrent_sessions == 5
        assert config.session_timeout_seconds == 1800.0
        assert config.use_mcts is False
        assert config.mcts_simulations == 25
        assert config.ai_temperature == 0.5
        assert config.device == "cuda"
        assert config.default_opponent == "heuristic"
        assert config.board_theme == "dark"
        assert config.show_ai_evaluation is False
        assert config.show_move_probabilities is False
        assert config.show_mcts_stats is False
        assert config.enable_replay is False
        assert config.replay_dir == "custom_replays"
        assert config.save_dir == "custom_saves"
        assert config.cors_origins == ["http://localhost:3000"]


class TestWebConfigValidation:
    """Tests for WebConfig validation in __post_init__."""

    # --- Port validation ---

    def test_port_zero_raises(self):
        """Test that port 0 raises ValueError."""
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            WebConfig(port=0)

    def test_port_negative_raises(self):
        """Test that negative port raises ValueError."""
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            WebConfig(port=-1)

    def test_port_above_max_raises(self):
        """Test that port above 65535 raises ValueError."""
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            WebConfig(port=65536)

    def test_port_min_valid(self):
        """Test that port 1 is valid."""
        config = WebConfig(port=1)
        assert config.port == 1

    def test_port_max_valid(self):
        """Test that port 65535 is valid."""
        config = WebConfig(port=65535)
        assert config.port == 65535

    # --- Log level validation ---

    def test_invalid_log_level_raises(self):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log_level"):
            WebConfig(log_level="verbose")

    @pytest.mark.parametrize("level", list(VALID_LOG_LEVELS))
    def test_valid_log_levels(self, level):
        """Test that all valid log levels are accepted."""
        config = WebConfig(log_level=level)
        assert config.log_level == level

    def test_log_level_case_insensitive(self):
        """Test that log level validation is case-insensitive."""
        config = WebConfig(log_level="INFO")
        assert config.log_level == "INFO"

    # --- Workers validation ---

    def test_workers_zero_raises(self):
        """Test that zero workers raises ValueError."""
        with pytest.raises(ValueError, match="workers must be >= 1"):
            WebConfig(workers=0)

    def test_workers_negative_raises(self):
        """Test that negative workers raises ValueError."""
        with pytest.raises(ValueError, match="workers must be >= 1"):
            WebConfig(workers=-1)

    def test_workers_one_valid(self):
        """Test that one worker is valid."""
        config = WebConfig(workers=1)
        assert config.workers == 1

    # --- Board size validation ---

    def test_board_size_too_small_raises(self):
        """Test that board_size < 4 raises ValueError."""
        with pytest.raises(ValueError, match="board_size must be >= 4"):
            WebConfig(board_size=3)

    def test_board_size_zero_raises(self):
        """Test that board_size 0 raises ValueError."""
        with pytest.raises(ValueError, match="board_size must be >= 4"):
            WebConfig(board_size=0)

    def test_board_size_negative_raises(self):
        """Test that negative board_size raises ValueError."""
        with pytest.raises(ValueError, match="board_size must be >= 4"):
            WebConfig(board_size=-1)

    def test_board_size_min_valid(self):
        """Test that board_size 4 is valid."""
        config = WebConfig(board_size=4)
        assert config.board_size == 4

    # --- Max turns validation ---

    def test_max_turns_zero_raises(self):
        """Test that max_turns 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            WebConfig(max_turns=0)

    def test_max_turns_negative_raises(self):
        """Test that negative max_turns raises ValueError."""
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            WebConfig(max_turns=-1)

    def test_max_turns_one_valid(self):
        """Test that max_turns 1 is valid."""
        config = WebConfig(max_turns=1)
        assert config.max_turns == 1

    # --- Thinking time validation ---

    def test_thinking_time_zero_raises(self):
        """Test that zero thinking time raises ValueError."""
        with pytest.raises(ValueError, match="default_thinking_time must be positive"):
            WebConfig(default_thinking_time=0.0)

    def test_thinking_time_negative_raises(self):
        """Test that negative thinking time raises ValueError."""
        with pytest.raises(ValueError, match="default_thinking_time must be positive"):
            WebConfig(default_thinking_time=-1.0)

    def test_thinking_time_small_positive_valid(self):
        """Test that a small positive thinking time is valid."""
        config = WebConfig(default_thinking_time=0.001)
        assert config.default_thinking_time == 0.001

    # --- Max concurrent sessions validation ---

    def test_max_concurrent_sessions_zero_raises(self):
        """Test that zero sessions raises ValueError."""
        with pytest.raises(ValueError, match="max_concurrent_sessions must be >= 1"):
            WebConfig(max_concurrent_sessions=0)

    def test_max_concurrent_sessions_negative_raises(self):
        """Test that negative sessions raises ValueError."""
        with pytest.raises(ValueError, match="max_concurrent_sessions must be >= 1"):
            WebConfig(max_concurrent_sessions=-1)

    # --- Session timeout validation ---

    def test_session_timeout_zero_raises(self):
        """Test that zero timeout raises ValueError."""
        with pytest.raises(ValueError, match="session_timeout_seconds must be positive"):
            WebConfig(session_timeout_seconds=0.0)

    def test_session_timeout_negative_raises(self):
        """Test that negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="session_timeout_seconds must be positive"):
            WebConfig(session_timeout_seconds=-1.0)

    # --- MCTS simulations validation ---

    def test_mcts_simulations_zero_raises(self):
        """Test that zero MCTS simulations raises ValueError."""
        with pytest.raises(ValueError, match="mcts_simulations must be >= 1"):
            WebConfig(mcts_simulations=0)

    def test_mcts_simulations_negative_raises(self):
        """Test that negative MCTS simulations raises ValueError."""
        with pytest.raises(ValueError, match="mcts_simulations must be >= 1"):
            WebConfig(mcts_simulations=-1)

    def test_mcts_simulations_one_valid(self):
        """Test that 1 MCTS simulation is valid."""
        config = WebConfig(mcts_simulations=1)
        assert config.mcts_simulations == 1

    # --- AI temperature validation ---

    def test_ai_temperature_negative_raises(self):
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="ai_temperature must be >= 0"):
            WebConfig(ai_temperature=-0.1)

    def test_ai_temperature_zero_valid(self):
        """Test that zero temperature (greedy) is valid."""
        config = WebConfig(ai_temperature=0.0)
        assert config.ai_temperature == 0.0

    # --- Default opponent validation ---

    def test_invalid_opponent_raises(self):
        """Test that invalid opponent type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid default_opponent"):
            WebConfig(default_opponent="superhuman")

    @pytest.mark.parametrize("opponent", list(VALID_OPPONENT_TYPES))
    def test_valid_opponent_types(self, opponent):
        """Test that all valid opponent types are accepted."""
        config = WebConfig(default_opponent=opponent)
        assert config.default_opponent == opponent

    # --- Board theme validation ---

    def test_invalid_board_theme_raises(self):
        """Test that invalid board theme raises ValueError."""
        with pytest.raises(ValueError, match="Invalid board_theme"):
            WebConfig(board_theme="neon")

    @pytest.mark.parametrize("theme", [t.value for t in BoardTheme])
    def test_valid_board_themes(self, theme):
        """Test that all BoardTheme enum values are accepted."""
        config = WebConfig(board_theme=theme)
        assert config.board_theme == theme

    # --- Checkpoint path warning ---

    def test_checkpoint_path_nonexistent_warns(self, tmp_path):
        """Test that nonexistent checkpoint path logs a warning."""
        nonexistent = str(tmp_path / "nonexistent_model.pt")
        # Should not raise, just log a warning
        config = WebConfig(checkpoint_path=nonexistent)
        assert config.checkpoint_path == nonexistent

    def test_checkpoint_path_none_valid(self):
        """Test that None checkpoint path is valid."""
        config = WebConfig(checkpoint_path=None)
        assert config.checkpoint_path is None

    def test_checkpoint_path_exists_valid(self, tmp_path):
        """Test that existing checkpoint path is valid."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_text("fake model data")
        config = WebConfig(checkpoint_path=str(checkpoint))
        assert config.checkpoint_path == str(checkpoint)


class TestWebConfigFromYaml:
    """Tests for WebConfig.from_yaml()."""

    def test_from_yaml_flat_structure(self, tmp_path):
        """Test loading from YAML with flat key structure."""
        yaml_data = {
            "host": "127.0.0.1",
            "port": 3000,
            "debug": True,
            "log_level": "debug",
            "board_size": 8,
            "max_turns": 50,
            "default_opponent": "heuristic",
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))

        config = WebConfig.from_yaml(yaml_path)

        assert config.host == "127.0.0.1"
        assert config.port == 3000
        assert config.debug is True
        assert config.log_level == "debug"
        assert config.board_size == 8
        assert config.max_turns == 50
        assert config.default_opponent == "heuristic"

    def test_from_yaml_web_section(self, tmp_path):
        """Test loading from YAML with 'web' section wrapper."""
        yaml_data = {
            "web": {
                "host": "0.0.0.0",
                "port": 9090,
                "debug": False,
                "board_size": 12,
            }
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))

        config = WebConfig.from_yaml(yaml_path)

        assert config.port == 9090
        assert config.board_size == 12

    def test_from_yaml_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            WebConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_empty_file(self, tmp_path):
        """Test loading from empty YAML file uses defaults."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")

        config = WebConfig.from_yaml(yaml_path)

        # Should get defaults
        assert config.port == 8080
        assert config.board_size == 16

    def test_from_yaml_ignores_unknown_keys(self, tmp_path):
        """Test that unknown YAML keys are silently ignored."""
        yaml_data = {
            "port": 4000,
            "unknown_setting": "should_be_ignored",
            "another_unknown": 42,
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))

        config = WebConfig.from_yaml(yaml_path)
        assert config.port == 4000

    def test_from_yaml_string_path(self, tmp_path):
        """Test that string path is accepted."""
        yaml_data = {"port": 5000}
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))

        config = WebConfig.from_yaml(str(yaml_path))
        assert config.port == 5000

    def test_from_yaml_all_fields(self, tmp_path):
        """Test loading all fields from YAML."""
        yaml_data = {
            "host": "192.168.1.1",
            "port": 7070,
            "debug": True,
            "log_level": "warning",
            "workers": 8,
            "board_size": 32,
            "max_turns": 500,
            "default_thinking_time": 10.0,
            "max_concurrent_sessions": 50,
            "session_timeout_seconds": 7200.0,
            "use_mcts": False,
            "mcts_simulations": 200,
            "ai_temperature": 1.0,
            "device": "cuda:0",
            "default_opponent": "neural",
            "board_theme": "dark",
            "show_ai_evaluation": False,
            "show_move_probabilities": False,
            "show_mcts_stats": False,
            "enable_replay": False,
            "replay_dir": "data/replays",
            "save_dir": "data/saves",
            "cors_origins": ["http://localhost:3000", "https://example.com"],
        }
        yaml_path = tmp_path / "full_config.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))

        config = WebConfig.from_yaml(yaml_path)

        assert config.host == "192.168.1.1"
        assert config.port == 7070
        assert config.debug is True
        assert config.log_level == "warning"
        assert config.workers == 8
        assert config.board_size == 32
        assert config.max_turns == 500
        assert config.default_thinking_time == 10.0
        assert config.max_concurrent_sessions == 50
        assert config.session_timeout_seconds == 7200.0
        assert config.use_mcts is False
        assert config.mcts_simulations == 200
        assert config.ai_temperature == 1.0
        assert config.device == "cuda:0"
        assert config.default_opponent == "neural"
        assert config.board_theme == "dark"
        assert config.show_ai_evaluation is False
        assert config.show_move_probabilities is False
        assert config.show_mcts_stats is False
        assert config.enable_replay is False
        assert config.replay_dir == "data/replays"
        assert config.save_dir == "data/saves"
        assert config.cors_origins == ["http://localhost:3000", "https://example.com"]

    def test_from_yaml_nested_structure_flattened(self, tmp_path):
        """Test YAML with nested dicts gets flattened into field names."""
        # The _flatten_dict method handles nested YAML sections
        yaml_data = {
            "port": 4000,
            "nested_section": {
                "board_size": 10,
            },
        }
        yaml_path = tmp_path / "nested.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))

        config = WebConfig.from_yaml(yaml_path)
        assert config.port == 4000
        # board_size should be flattened from the nested dict
        assert config.board_size == 10

    def test_from_yaml_invalid_values_raise(self, tmp_path):
        """Test that invalid values in YAML raise during validation."""
        yaml_data = {"port": 0}
        yaml_path = tmp_path / "invalid.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))

        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            WebConfig.from_yaml(yaml_path)


class TestWebConfigFromEnv:
    """Tests for WebConfig.from_env()."""

    def test_from_env_no_vars_uses_defaults(self, monkeypatch):
        """Test that from_env with no environment variables uses defaults."""
        # Clear all CIV6AZ_WEB_ variables
        for key in list(monkeypatch._patches if hasattr(monkeypatch, "_patches") else []):
            if key.startswith("CIV6AZ_WEB_"):
                monkeypatch.delenv(key, raising=False)

        config = WebConfig.from_env()

        assert config.port == 8080
        assert config.host == "0.0.0.0"
        assert config.board_size == 16

    def test_from_env_port_override(self, monkeypatch):
        """Test that CIV6AZ_WEB_PORT overrides port."""
        monkeypatch.setenv("CIV6AZ_WEB_PORT", "3000")

        config = WebConfig.from_env()

        assert config.port == 3000

    def test_from_env_host_override(self, monkeypatch):
        """Test that CIV6AZ_WEB_HOST overrides host."""
        monkeypatch.setenv("CIV6AZ_WEB_HOST", "127.0.0.1")

        config = WebConfig.from_env()

        assert config.host == "127.0.0.1"

    def test_from_env_debug_true(self, monkeypatch):
        """Test that CIV6AZ_WEB_DEBUG=true enables debug mode."""
        monkeypatch.setenv("CIV6AZ_WEB_DEBUG", "true")

        config = WebConfig.from_env()

        assert config.debug is True

    def test_from_env_debug_false(self, monkeypatch):
        """Test that CIV6AZ_WEB_DEBUG=false disables debug mode."""
        monkeypatch.setenv("CIV6AZ_WEB_DEBUG", "false")

        config = WebConfig.from_env()

        assert config.debug is False

    def test_from_env_log_level(self, monkeypatch):
        """Test that CIV6AZ_WEB_LOG_LEVEL overrides log level."""
        monkeypatch.setenv("CIV6AZ_WEB_LOG_LEVEL", "warning")

        config = WebConfig.from_env()

        assert config.log_level == "warning"

    def test_from_env_workers(self, monkeypatch):
        """Test that CIV6AZ_WEB_WORKERS overrides workers count."""
        monkeypatch.setenv("CIV6AZ_WEB_WORKERS", "4")

        config = WebConfig.from_env()

        assert config.workers == 4

    def test_from_env_board_size(self, monkeypatch):
        """Test that CIV6AZ_WEB_BOARD_SIZE overrides board size."""
        monkeypatch.setenv("CIV6AZ_WEB_BOARD_SIZE", "8")

        config = WebConfig.from_env()

        assert config.board_size == 8

    def test_from_env_max_turns(self, monkeypatch):
        """Test that CIV6AZ_WEB_MAX_TURNS overrides max turns."""
        monkeypatch.setenv("CIV6AZ_WEB_MAX_TURNS", "100")

        config = WebConfig.from_env()

        assert config.max_turns == 100

    def test_from_env_thinking_time(self, monkeypatch):
        """Test that CIV6AZ_WEB_THINKING_TIME overrides thinking time."""
        monkeypatch.setenv("CIV6AZ_WEB_THINKING_TIME", "2.5")

        config = WebConfig.from_env()

        assert config.default_thinking_time == 2.5

    def test_from_env_max_sessions(self, monkeypatch):
        """Test that CIV6AZ_WEB_MAX_SESSIONS overrides max sessions."""
        monkeypatch.setenv("CIV6AZ_WEB_MAX_SESSIONS", "20")

        config = WebConfig.from_env()

        assert config.max_concurrent_sessions == 20

    def test_from_env_session_timeout(self, monkeypatch):
        """Test that CIV6AZ_WEB_SESSION_TIMEOUT overrides session timeout."""
        monkeypatch.setenv("CIV6AZ_WEB_SESSION_TIMEOUT", "1800.0")

        config = WebConfig.from_env()

        assert config.session_timeout_seconds == 1800.0

    def test_from_env_checkpoint(self, monkeypatch):
        """Test that CIV6AZ_WEB_CHECKPOINT overrides checkpoint path."""
        monkeypatch.setenv("CIV6AZ_WEB_CHECKPOINT", "/models/best.pt")

        config = WebConfig.from_env()

        assert config.checkpoint_path == "/models/best.pt"

    def test_from_env_use_mcts(self, monkeypatch):
        """Test that CIV6AZ_WEB_USE_MCTS overrides use_mcts."""
        monkeypatch.setenv("CIV6AZ_WEB_USE_MCTS", "false")

        config = WebConfig.from_env()

        assert config.use_mcts is False

    def test_from_env_mcts_simulations(self, monkeypatch):
        """Test that CIV6AZ_WEB_MCTS_SIMULATIONS overrides simulations."""
        monkeypatch.setenv("CIV6AZ_WEB_MCTS_SIMULATIONS", "200")

        config = WebConfig.from_env()

        assert config.mcts_simulations == 200

    def test_from_env_ai_temperature(self, monkeypatch):
        """Test that CIV6AZ_WEB_AI_TEMPERATURE overrides temperature."""
        monkeypatch.setenv("CIV6AZ_WEB_AI_TEMPERATURE", "0.8")

        config = WebConfig.from_env()

        assert config.ai_temperature == 0.8

    def test_from_env_device(self, monkeypatch):
        """Test that CIV6AZ_WEB_DEVICE overrides device."""
        monkeypatch.setenv("CIV6AZ_WEB_DEVICE", "cuda:1")

        config = WebConfig.from_env()

        assert config.device == "cuda:1"

    def test_from_env_default_opponent(self, monkeypatch):
        """Test that CIV6AZ_WEB_DEFAULT_OPPONENT overrides opponent."""
        monkeypatch.setenv("CIV6AZ_WEB_DEFAULT_OPPONENT", "neural")

        config = WebConfig.from_env()

        assert config.default_opponent == "neural"

    def test_from_env_board_theme(self, monkeypatch):
        """Test that CIV6AZ_WEB_BOARD_THEME overrides theme."""
        monkeypatch.setenv("CIV6AZ_WEB_BOARD_THEME", "dark")

        config = WebConfig.from_env()

        assert config.board_theme == "dark"

    def test_from_env_show_evaluation(self, monkeypatch):
        """Test that CIV6AZ_WEB_SHOW_EVALUATION overrides show_ai_evaluation."""
        monkeypatch.setenv("CIV6AZ_WEB_SHOW_EVALUATION", "false")

        config = WebConfig.from_env()

        assert config.show_ai_evaluation is False

    def test_from_env_show_probabilities(self, monkeypatch):
        """Test that CIV6AZ_WEB_SHOW_PROBABILITIES overrides show_move_probabilities."""
        monkeypatch.setenv("CIV6AZ_WEB_SHOW_PROBABILITIES", "false")

        config = WebConfig.from_env()

        assert config.show_move_probabilities is False

    def test_from_env_show_mcts_stats(self, monkeypatch):
        """Test that CIV6AZ_WEB_SHOW_MCTS_STATS overrides show_mcts_stats."""
        monkeypatch.setenv("CIV6AZ_WEB_SHOW_MCTS_STATS", "false")

        config = WebConfig.from_env()

        assert config.show_mcts_stats is False

    def test_from_env_enable_replay(self, monkeypatch):
        """Test that CIV6AZ_WEB_ENABLE_REPLAY overrides enable_replay."""
        monkeypatch.setenv("CIV6AZ_WEB_ENABLE_REPLAY", "false")

        config = WebConfig.from_env()

        assert config.enable_replay is False

    def test_from_env_replay_dir(self, monkeypatch):
        """Test that CIV6AZ_WEB_REPLAY_DIR overrides replay_dir."""
        monkeypatch.setenv("CIV6AZ_WEB_REPLAY_DIR", "/data/replays")

        config = WebConfig.from_env()

        assert config.replay_dir == "/data/replays"

    def test_from_env_save_dir(self, monkeypatch):
        """Test that CIV6AZ_WEB_SAVE_DIR overrides save_dir."""
        monkeypatch.setenv("CIV6AZ_WEB_SAVE_DIR", "/data/saves")

        config = WebConfig.from_env()

        assert config.save_dir == "/data/saves"

    def test_from_env_with_base_config(self, monkeypatch):
        """Test from_env with a base config applies overrides on top."""
        base = WebConfig(port=9090, board_size=8, default_opponent="heuristic")
        monkeypatch.setenv("CIV6AZ_WEB_PORT", "4000")

        config = WebConfig.from_env(base_config=base)

        assert config.port == 4000
        assert config.board_size == 8
        assert config.default_opponent == "heuristic"

    def test_from_env_invalid_int_ignored(self, monkeypatch):
        """Test that invalid integer env var is silently ignored."""
        monkeypatch.setenv("CIV6AZ_WEB_PORT", "not_a_number")

        # Should not raise, invalid value is logged and skipped
        config = WebConfig.from_env()
        assert config.port == 8080  # Default

    def test_from_env_invalid_float_ignored(self, monkeypatch):
        """Test that invalid float env var is silently ignored."""
        monkeypatch.setenv("CIV6AZ_WEB_THINKING_TIME", "invalid")

        config = WebConfig.from_env()
        assert config.default_thinking_time == 5.0  # Default

    def test_from_env_multiple_overrides(self, monkeypatch):
        """Test that multiple env vars are applied simultaneously."""
        monkeypatch.setenv("CIV6AZ_WEB_PORT", "5000")
        monkeypatch.setenv("CIV6AZ_WEB_HOST", "10.0.0.1")
        monkeypatch.setenv("CIV6AZ_WEB_DEBUG", "true")
        monkeypatch.setenv("CIV6AZ_WEB_BOARD_SIZE", "12")
        monkeypatch.setenv("CIV6AZ_WEB_DEFAULT_OPPONENT", "neural")

        config = WebConfig.from_env()

        assert config.port == 5000
        assert config.host == "10.0.0.1"
        assert config.debug is True
        assert config.board_size == 12
        assert config.default_opponent == "neural"


class TestWebConfigFromPreset:
    """Tests for WebConfig.from_preset()."""

    def test_dev_preset(self):
        """Test dev preset has expected values."""
        config = WebConfig.from_preset("dev")

        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.debug is True
        assert config.log_level == "debug"
        assert config.board_size == 8
        assert config.max_turns == 50
        assert config.default_thinking_time == 1.0
        assert config.max_concurrent_sessions == 3
        assert config.session_timeout_seconds == 600.0
        assert config.use_mcts is False
        assert config.mcts_simulations == 10
        assert config.default_opponent == "random"

    def test_production_preset(self):
        """Test production preset has expected values."""
        config = WebConfig.from_preset("production")

        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.debug is False
        assert config.log_level == "info"
        assert config.board_size == 16
        assert config.max_turns == 200
        assert config.default_thinking_time == 5.0
        assert config.max_concurrent_sessions == 20
        assert config.session_timeout_seconds == 7200.0
        assert config.use_mcts is True
        assert config.mcts_simulations == 100
        assert config.default_opponent == "heuristic"

    def test_unknown_preset_raises(self):
        """Test that unknown preset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            WebConfig.from_preset("nonexistent")

    def test_unknown_preset_lists_valid(self):
        """Test that error message lists valid preset names."""
        with pytest.raises(ValueError, match="dev"):
            WebConfig.from_preset("invalid")

    def test_preset_returns_webconfig_instance(self):
        """Test that preset returns a WebConfig instance."""
        config = WebConfig.from_preset("dev")
        assert isinstance(config, WebConfig)

    def test_dev_preset_is_valid(self):
        """Test that dev preset passes all validation."""
        # No exception means validation passed
        config = WebConfig.from_preset("dev")
        assert config is not None

    def test_production_preset_is_valid(self):
        """Test that production preset passes all validation."""
        config = WebConfig.from_preset("production")
        assert config is not None


class TestWebConfigToDict:
    """Tests for WebConfig.to_dict()."""

    def test_to_dict_contains_all_fields(self):
        """Test that to_dict includes all configuration fields."""
        config = WebConfig()
        d = config.to_dict()

        expected_keys = {
            "host",
            "port",
            "debug",
            "log_level",
            "workers",
            "board_size",
            "max_turns",
            "default_thinking_time",
            "max_concurrent_sessions",
            "session_timeout_seconds",
            "checkpoint_path",
            "use_mcts",
            "mcts_simulations",
            "ai_temperature",
            "device",
            "default_opponent",
            "board_theme",
            "show_ai_evaluation",
            "show_move_probabilities",
            "show_mcts_stats",
            "enable_replay",
            "replay_dir",
            "save_dir",
            "cors_origins",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match_attributes(self):
        """Test that to_dict values match the object attributes."""
        config = WebConfig(
            host="10.0.0.1",
            port=9090,
            debug=True,
            board_size=12,
            default_opponent="neural",
            board_theme="dark",
        )
        d = config.to_dict()

        assert d["host"] == "10.0.0.1"
        assert d["port"] == 9090
        assert d["debug"] is True
        assert d["board_size"] == 12
        assert d["default_opponent"] == "neural"
        assert d["board_theme"] == "dark"

    def test_to_dict_round_trip(self):
        """Test that to_dict can recreate an equivalent config."""
        original = WebConfig(
            host="127.0.0.1",
            port=3000,
            debug=True,
            log_level="debug",
            workers=2,
            board_size=8,
            max_turns=100,
            default_thinking_time=2.0,
            max_concurrent_sessions=5,
            session_timeout_seconds=1800.0,
            use_mcts=False,
            mcts_simulations=25,
            ai_temperature=0.5,
            device="cuda",
            default_opponent="heuristic",
            board_theme="dark",
            show_ai_evaluation=False,
            show_move_probabilities=False,
            show_mcts_stats=False,
            enable_replay=False,
            replay_dir="custom_replays",
            save_dir="custom_saves",
            cors_origins=["http://localhost"],
        )
        d = original.to_dict()
        restored = WebConfig(**d)

        assert restored.to_dict() == original.to_dict()

    def test_to_dict_returns_new_dict(self):
        """Test that to_dict returns a new dict, not internal state."""
        config = WebConfig()
        d1 = config.to_dict()
        d2 = config.to_dict()

        assert d1 == d2
        assert d1 is not d2

    def test_to_dict_cors_origins_is_list(self):
        """Test that cors_origins in to_dict is a list."""
        config = WebConfig(cors_origins=["http://example.com", "http://test.com"])
        d = config.to_dict()

        assert isinstance(d["cors_origins"], list)
        assert len(d["cors_origins"]) == 2

    def test_to_dict_default_checkpoint_is_none(self):
        """Test that default checkpoint_path is None in to_dict."""
        config = WebConfig()
        d = config.to_dict()

        assert d["checkpoint_path"] is None


class TestWebConfigToHumanInterfaceConfig:
    """Tests for WebConfig.to_human_interface_config()."""

    def test_returns_human_interface_config_instance(self):
        """Test that method returns a HumanInterfaceConfig."""
        config = WebConfig()
        hi_config = config.to_human_interface_config()

        assert isinstance(hi_config, HumanInterfaceConfig)

    def test_thinking_time_mapped(self):
        """Test that thinking time is mapped correctly."""
        config = WebConfig(default_thinking_time=3.0)
        hi_config = config.to_human_interface_config()

        assert hi_config.thinking_time_seconds == 3.0

    def test_max_thinking_time_is_at_least_60(self):
        """Test that max_thinking_time is at least 60.0."""
        config = WebConfig(default_thinking_time=5.0)
        hi_config = config.to_human_interface_config()

        # max(5.0 * 2, 60.0) = 60.0
        assert hi_config.max_thinking_time == 60.0

    def test_max_thinking_time_doubles_when_large(self):
        """Test that max_thinking_time is 2x thinking time when that exceeds 60."""
        config = WebConfig(default_thinking_time=50.0)
        hi_config = config.to_human_interface_config()

        # max(50.0 * 2, 60.0) = 100.0
        assert hi_config.max_thinking_time == 100.0

    def test_max_thinking_time_boundary(self):
        """Test max_thinking_time at the boundary (30s * 2 = 60)."""
        config = WebConfig(default_thinking_time=30.0)
        hi_config = config.to_human_interface_config()

        # max(30.0 * 2, 60.0) = 60.0
        assert hi_config.max_thinking_time == 60.0

    def test_show_flags_mapped(self):
        """Test that display flags are correctly mapped."""
        config = WebConfig(
            show_ai_evaluation=False,
            show_move_probabilities=False,
            show_mcts_stats=False,
        )
        hi_config = config.to_human_interface_config()

        assert hi_config.show_ai_evaluation is False
        assert hi_config.show_move_probabilities is False
        assert hi_config.show_mcts_stats is False

    def test_show_flags_mapped_true(self):
        """Test that enabled display flags are mapped."""
        config = WebConfig(
            show_ai_evaluation=True,
            show_move_probabilities=True,
            show_mcts_stats=True,
        )
        hi_config = config.to_human_interface_config()

        assert hi_config.show_ai_evaluation is True
        assert hi_config.show_move_probabilities is True
        assert hi_config.show_mcts_stats is True

    def test_board_theme_mapped(self):
        """Test that board theme is mapped correctly."""
        config = WebConfig(board_theme="dark")
        hi_config = config.to_human_interface_config()

        assert hi_config.board_theme == "dark"

    def test_replay_settings_mapped(self):
        """Test that replay settings are mapped correctly."""
        config = WebConfig(
            enable_replay=True,
            replay_dir="my_replays",
        )
        hi_config = config.to_human_interface_config()

        assert hi_config.enable_replay is True
        assert hi_config.replay_dir == "my_replays"

    def test_save_dir_mapped(self):
        """Test that save_dir is mapped correctly."""
        config = WebConfig(save_dir="my_saves")
        hi_config = config.to_human_interface_config()

        assert hi_config.save_dir == "my_saves"

    def test_enable_undo_is_true(self):
        """Test that enable_undo is hardcoded to True."""
        config = WebConfig()
        hi_config = config.to_human_interface_config()

        assert hi_config.enable_undo is True

    def test_max_undo_depth_is_10(self):
        """Test that max_undo_depth is hardcoded to 10."""
        config = WebConfig()
        hi_config = config.to_human_interface_config()

        assert hi_config.max_undo_depth == 10

    @pytest.mark.parametrize("theme", [t.value for t in BoardTheme])
    def test_all_board_themes_map_correctly(self, theme):
        """Test that all valid board themes map to HumanInterfaceConfig."""
        config = WebConfig(board_theme=theme)
        hi_config = config.to_human_interface_config()

        assert hi_config.board_theme == theme


class TestWebConfigFieldNames:
    """Tests for WebConfig._field_names() helper."""

    def test_field_names_returns_set(self):
        """Test that _field_names returns a set."""
        names = WebConfig._field_names()
        assert isinstance(names, set)

    def test_field_names_contains_expected(self):
        """Test that _field_names includes key fields."""
        names = WebConfig._field_names()
        assert "host" in names
        assert "port" in names
        assert "debug" in names
        assert "board_size" in names
        assert "cors_origins" in names

    def test_field_names_does_not_contain_methods(self):
        """Test that _field_names does not include method names."""
        names = WebConfig._field_names()
        assert "to_dict" not in names
        assert "from_yaml" not in names
        assert "_validate" not in names


class TestWebConfigFlattenDict:
    """Tests for WebConfig._flatten_dict() helper."""

    def test_flat_dict_unchanged(self):
        """Test that a flat dict passes through unchanged."""
        data = {"port": 8080, "host": "localhost"}
        result = WebConfig._flatten_dict(data)
        assert result == {"port": 8080, "host": "localhost"}

    def test_nested_dict_flattened(self):
        """Test that nested dict values are flattened."""
        data = {
            "server": {
                "port": 3000,
            },
            "host": "localhost",
        }
        result = WebConfig._flatten_dict(data)
        # port is a known field name, so it should be kept as "port"
        assert "port" in result
        assert result["port"] == 3000
        assert result["host"] == "localhost"

    def test_empty_dict(self):
        """Test that empty dict returns empty dict."""
        result = WebConfig._flatten_dict({})
        assert result == {}


class TestWebConfigEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_cors_origins_empty_list(self):
        """Test that empty cors_origins list is valid."""
        config = WebConfig(cors_origins=[])
        assert config.cors_origins == []

    def test_cors_origins_multiple_entries(self):
        """Test that multiple CORS origins work."""
        origins = ["http://localhost:3000", "https://example.com", "https://app.example.com"]
        config = WebConfig(cors_origins=origins)
        assert config.cors_origins == origins

    def test_large_board_size(self):
        """Test that large board sizes are valid."""
        config = WebConfig(board_size=128)
        assert config.board_size == 128

    def test_large_max_turns(self):
        """Test that large max_turns values are valid."""
        config = WebConfig(max_turns=10000)
        assert config.max_turns == 10000

    def test_high_temperature(self):
        """Test that high temperature values are valid."""
        config = WebConfig(ai_temperature=100.0)
        assert config.ai_temperature == 100.0

    def test_many_mcts_simulations(self):
        """Test that many MCTS simulations are valid."""
        config = WebConfig(mcts_simulations=10000)
        assert config.mcts_simulations == 10000

    def test_many_workers(self):
        """Test that many workers are valid."""
        config = WebConfig(workers=64)
        assert config.workers == 64

    def test_device_strings(self):
        """Test various device string values."""
        for device in ("cpu", "cuda", "cuda:0", "cuda:1", "mps"):
            config = WebConfig(device=device)
            assert config.device == device

    def test_from_preset_then_to_dict_round_trip(self):
        """Test that preset -> to_dict -> WebConfig round trip works."""
        original = WebConfig.from_preset("dev")
        d = original.to_dict()
        restored = WebConfig(**d)

        assert restored.to_dict() == original.to_dict()

    def test_from_yaml_then_to_dict_round_trip(self, tmp_path):
        """Test YAML load -> to_dict -> WebConfig round trip."""
        yaml_data = {
            "port": 5000,
            "board_size": 10,
            "default_opponent": "heuristic",
            "board_theme": "minimal",
        }
        yaml_path = tmp_path / "rt.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))

        loaded = WebConfig.from_yaml(yaml_path)
        d = loaded.to_dict()
        restored = WebConfig(**d)

        assert restored.port == 5000
        assert restored.board_size == 10
        assert restored.default_opponent == "heuristic"
        assert restored.board_theme == "minimal"

    def test_default_config_cors_origins_is_independent(self):
        """Test that default cors_origins list is independent between instances."""
        config1 = WebConfig()
        config2 = WebConfig()

        config1.cors_origins.append("http://test.com")
        assert "http://test.com" not in config2.cors_origins
