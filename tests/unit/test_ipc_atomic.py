"""Unit tests for atomic IPC bridge operations and dynamic path generation."""

import json
import tempfile
from pathlib import Path

import pytest

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.file_bridge import FileBridge


@pytest.mark.unit
class TestIPCAtomic:
    """Test suite for atomic IPC bridge behavior."""

    def test_bridge_config_comm_dir_resolution(self):
        """Test that BridgeConfig resolves non-empty valid path."""
        config = BridgeConfig()
        assert config.comm_dir is not None
        assert isinstance(config.comm_dir, Path)
        assert "AlphaZeroAI" in str(config.comm_dir) or ".civ6_alphazero" in str(config.comm_dir)

    def test_generate_lua_path_config(self):
        """Test generation of GeneratedPaths.lua module."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tmp_path = Path(tmpdir)
            mod_scripts = tmp_path / "Scripts"
            mod_scripts.mkdir(parents=True, exist_ok=True)

            config = BridgeConfig(comm_dir=tmp_path / "AlphaZeroAI")
            config.generate_lua_path_config(mod_scripts_dir=mod_scripts)

            generated_file = mod_scripts / "GeneratedPaths.lua"
            assert generated_file.exists()
            content = generated_file.read_text(encoding="utf-8")
            assert "GeneratedPaths" in content
            assert "COMM_DIR" in content

    @pytest.mark.asyncio
    async def test_poll_state_atomic_retry(self):
        """Test poll_state recovers cleanly from temporary JSON decode errors."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            comm_dir = Path(tmpdir)
            config = BridgeConfig(comm_dir=comm_dir)
            bridge = FileBridge(config)
            config.ensure_comm_dir()

            state_file = config.state_file
            # Write invalid partial JSON
            state_file.write_text('{"turn_id": 1, "state": ', encoding="utf-8")

            # Polling should return None without crashing
            state = await bridge.poll_state()
            assert state is None

            # Now write complete valid JSON
            valid_payload = {
                "msg_id": 1,
                "timestamp": 100.0,
                "state": {
                    "turn_id": 1,
                    "active_player_id": 0,
                    "is_game_over": False,
                    "winner_id": None,
                    "players": [{"player_id": 0, "name": "Player 1", "score": 10}],
                    "units": [],
                    "cities": [],
                    "tiles": [],
                },
            }
            state_file.write_text(json.dumps(valid_payload), encoding="utf-8")

            state_res = await bridge.poll_state()
            assert state_res is not None
            assert state_res.turn_id == 1
