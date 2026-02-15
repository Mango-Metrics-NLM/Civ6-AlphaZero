"""Smoke tests for the Strategic Planning SDK.

These tests verify that the SDK can be imported and basic operations work.
They are designed to run fast (< 5 seconds total).
"""
from __future__ import annotations

import numpy as np


class TestImports:
    """Verify all SDK packages can be imported."""

    def test_core_types_import(self):
        from src.core.types import DeviceType
        assert DeviceType.cpu == "cpu"

    def test_core_config_import(self):
        from src.core.config import SDKConfig, load_config
        config = load_config()
        assert isinstance(config, SDKConfig)

    def test_core_exceptions_import(self):
        from src.core.exceptions import ConfigError, SDKError
        assert issubclass(ConfigError, SDKError)

    def test_core_logging_import(self):
        from src.core.logging import get_logger
        log = get_logger("smoke_test")
        assert log is not None

    def test_core_protocols_import(self):
        from src.core.protocols import StrategicEnvironment
        assert StrategicEnvironment is not None

    def test_mcts_import(self):
        from src.core.mcts import MCTSNode
        assert MCTSNode is not None

    def test_training_import(self):
        from src.core.training import ReplayBuffer
        assert ReplayBuffer is not None

    def test_environments_import(self):
        from src.environments.mini_games import MiniResourceEnv
        assert MiniResourceEnv is not None

    def test_adapters_import(self):
        from src.adapters import MicroCivAdapter
        assert MicroCivAdapter is not None

    def test_sdk_top_level_import(self):
        from src import SDKConfig
        assert SDKConfig is not None


class TestSmokeOperations:
    """Quick operational smoke tests."""

    def test_create_and_reset_environment(self):
        from src.environments.mini_games import MiniResourceEnv
        env = MiniResourceEnv(board_size=4, max_steps=10, seed=42)
        obs = env.reset(seed=42)
        assert "spatial" in obs
        assert "vector" in obs

    def test_step_environment(self):
        from src.environments.mini_games import MiniResourceEnv
        env = MiniResourceEnv(board_size=4, max_steps=10)
        env.reset(seed=0)
        _obs, reward, _terminated, _truncated, _info = env.step(0)
        assert -1.0 <= reward <= 1.0

    def test_mcts_node_lifecycle(self):
        from src.core.mcts import MCTSNode
        root = MCTSNode()
        policy = np.array([0.2, 0.3, 0.5], dtype=np.float32)
        mask = np.array([True, True, True])
        root.expand(policy, mask)
        assert root.is_expanded
        assert len(root.children) == 3

    def test_replay_buffer_lifecycle(self):
        from src.core.training import ReplayBuffer
        from src.core.types import TrainingExample

        buf = ReplayBuffer(capacity=10)
        example = TrainingExample(
            spatial=np.zeros((1, 2, 2), dtype=np.float32),
            vector=np.zeros(2, dtype=np.float32),
            action_mask=np.ones(3, dtype=np.bool_),
            policy=np.array([0.3, 0.3, 0.4], dtype=np.float32),
            value=0.5,
        )
        buf.add(example)
        assert buf.size == 1
        sampled = buf.sample(1)
        assert len(sampled) == 1

    def test_config_round_trip(self, tmp_path):
        import yaml

        from src.core.config import SDKConfig

        config = SDKConfig(seed=123, project_name="smoke-test")
        data = config.model_dump(mode="json")

        yaml_path = tmp_path / "smoke_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        loaded = SDKConfig.from_yaml(yaml_path)
        assert loaded.seed == 123
        assert loaded.project_name == "smoke-test"

    def test_protocol_validation_passes(self):
        from src.core.protocols import assert_valid_environment
        from src.environments.mini_games import MiniResourceEnv

        env = MiniResourceEnv(board_size=4, max_steps=10, seed=0)
        assert_valid_environment(env)  # Should not raise
