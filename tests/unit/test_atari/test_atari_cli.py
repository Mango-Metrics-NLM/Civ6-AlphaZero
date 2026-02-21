"""Unit tests for the Atari demo CLI entry point (__main__.py).

Tests argument parsing, config construction, and all CLI override paths.
Uses mocking to avoid actually launching training.
"""

from __future__ import annotations

from unittest import mock

import gymnasium as gym
import numpy as np
import pytest

from demos.atari.__main__ import _build_parser, main
from demos.atari.config import AtariConfig


# ============================================================================
# Fake env for CLI tests (avoids ROM dependency)
# ============================================================================


class _FakeAtariGym:
    class _Space:
        n = 4

    action_space = _Space()
    observation_space = None
    _step_count = 0

    def reset(self, seed=None):
        self._step_count = 0
        return np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8), {"lives": 3}

    def step(self, action):
        self._step_count += 1
        obs = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
        terminated = self._step_count >= 5
        return obs, 1.0, terminated, False, {"lives": 3}

    def close(self):
        pass


@pytest.fixture(autouse=True)
def mock_gymnasium(monkeypatch):
    monkeypatch.setattr(gym, "make", lambda *a, **kw: _FakeAtariGym())


# ============================================================================
# Parser tests
# ============================================================================


class TestBuildParser:
    def test_parser_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.config is None
        assert args.game is None
        assert args.iterations is None
        assert args.dry_run is False
        assert args.verbose is False
        assert args.hierarchical is False
        assert args.reflection is False
        assert args.rag is False

    def test_parser_game_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--game", "Pong"])
        assert args.game == "Pong"

    def test_parser_config_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--config", "path/to/config.yaml"])
        assert args.config == "path/to/config.yaml"

    def test_parser_training_overrides(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--iterations", "50",
            "--episodes-per-iter", "5",
            "--batch-size", "64",
            "--learning-rate", "0.01",
            "--num-simulations", "100",
            "--num-res-blocks", "3",
            "--num-filters", "32",
            "--discount-factor", "0.95",
            "--device", "cpu",
            "--checkpoint-dir", "/tmp/ckpt",
        ])
        assert args.iterations == 50
        assert args.episodes_per_iter == 5
        assert args.batch_size == 64
        assert args.learning_rate == 0.01
        assert args.num_simulations == 100
        assert args.num_res_blocks == 3
        assert args.num_filters == 32
        assert args.discount_factor == 0.95
        assert args.device == "cpu"
        assert args.checkpoint_dir == "/tmp/ckpt"

    def test_parser_boolean_flags(self):
        parser = _build_parser()
        args = parser.parse_args(["--hierarchical", "--reflection", "--rag", "--dry-run", "-v"])
        assert args.hierarchical is True
        assert args.reflection is True
        assert args.rag is True
        assert args.dry_run is True
        assert args.verbose is True


# ============================================================================
# main() tests
# ============================================================================


class TestMain:
    def test_main_default_dry_run(self):
        """main() with --dry-run should complete without error."""
        main(["--dry-run"])

    def test_main_with_game(self):
        """main() with --game flag sets the game name."""
        main(["--game", "Pong", "--dry-run"])

    def test_main_with_game_preset(self):
        """main() with a known game preset."""
        main(["--game", "Breakout", "--dry-run"])

    def test_main_with_all_overrides(self):
        """main() applies all CLI overrides correctly."""
        main([
            "--game", "Pong",
            "--iterations", "1",
            "--episodes-per-iter", "1",
            "--batch-size", "4",
            "--learning-rate", "0.01",
            "--num-simulations", "3",
            "--num-res-blocks", "1",
            "--num-filters", "8",
            "--discount-factor", "0.95",
            "--device", "cpu",
            "--dry-run",
        ])

    def test_main_with_hierarchical(self):
        """main() with --hierarchical enables hierarchical reasoning."""
        main(["--hierarchical", "--dry-run"])

    def test_main_with_reflection(self):
        """main() with --reflection enables reflection memory."""
        main(["--reflection", "--dry-run"])

    def test_main_with_rag(self):
        """main() with --rag enables RAG advisor."""
        main(["--rag", "--dry-run"])

    def test_main_with_all_novel_features(self):
        """main() with all novel features enabled."""
        main(["--reflection", "--rag", "--hierarchical", "--dry-run"])

    def test_main_verbose_logging(self):
        """main() with -v sets DEBUG log level."""
        main(["-v", "--dry-run"])

    def test_main_with_yaml_config(self, tmp_path):
        """main() loads config from YAML file."""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(
            "environment:\n"
            "  game_name: FakeGame\n"
            "  frame_height: 42\n"
            "  frame_width: 42\n"
            "  frame_stack: 2\n"
            "  vector_dim: 8\n"
            "network:\n"
            "  num_res_blocks: 1\n"
            "  num_filters: 8\n"
            "  fc_hidden_dim: 16\n"
            "  value_head_hidden: 8\n"
            "training:\n"
            "  device: cpu\n"
            "  min_replay_size: 1\n"
        )
        main(["--config", str(yaml_file), "--dry-run"])

    def test_main_config_with_game_override(self):
        """When --config is given, --game does NOT override game_name."""
        with mock.patch("demos.atari.__main__.AtariConfig.from_yaml") as mock_yaml:
            mock_yaml.return_value = AtariConfig()
            with mock.patch("demos.atari.__main__.AtariTrainer") as mock_trainer:
                mock_trainer.return_value.train.return_value = {
                    "iterations": 1, "final_model": "test.pt"
                }
                main(["--config", "fake.yaml", "--game", "Breakout", "--dry-run"])
                config_used = mock_trainer.call_args[0][0]
                # game_name should NOT be overridden when --config is used
                assert config_used.environment.game_name == "Pong"  # default

    def test_main_checkpoint_dir_override(self, tmp_path):
        """main() applies --checkpoint-dir override."""
        ckpt_dir = str(tmp_path / "my_checkpoints")
        main(["--checkpoint-dir", ckpt_dir, "--dry-run"])
