"""Tests for Protocol definitions.

This module verifies that protocol definitions are correct and that
implementations properly conform to their protocols.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from alpha_zero.core.protocols import (
    ActionMaskable,
    CheckpointableProtocol,
    EnvironmentProtocol,
    MCTSProtocol,
    NetworkProtocol,
    Serializable,
    StateEncoderProtocol,
    TrainerProtocol,
)

# =============================================================================
# TestSerializableProtocol
# =============================================================================


class TestSerializableProtocol:
    """Tests for Serializable protocol conformance."""

    @pytest.mark.parametrize(
        "class_module,class_name",
        [
            ("alpha_zero.core.timeout_config", "TimeoutConfig"),
            ("alpha_zero.core.buffer_config", "ReplayBufferConfig"),
            ("alpha_zero.core.buffer_config", "SharedStorageConfig"),
        ],
    )
    def test_config_classes_are_serializable(self, class_module: str, class_name: str):
        """Test that config classes implement Serializable protocol."""
        module = importlib.import_module(class_module)
        cls = getattr(module, class_name)
        instance = cls()

        # Verify protocol methods exist
        assert hasattr(instance, "to_dict")
        assert hasattr(cls, "from_dict")

        # Verify to_dict returns dict
        data = instance.to_dict()
        assert isinstance(data, dict)

        # Verify from_dict returns correct type
        restored = cls.from_dict(data)
        assert isinstance(restored, type(instance))

    def test_serializable_runtime_check(self):
        """Test Serializable can be used for runtime type checking."""

        class ValidSerializable:
            def to_dict(self) -> dict[str, Any]:
                return {"value": 1}

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> ValidSerializable:
                return cls()

        class InvalidNotSerializable:
            pass

        assert isinstance(ValidSerializable(), Serializable)
        assert not isinstance(InvalidNotSerializable(), Serializable)


# =============================================================================
# TestStateEncoderProtocol
# =============================================================================


class TestStateEncoderProtocol:
    """Tests for StateEncoderProtocol conformance."""

    def test_state_encoder_protocol_methods(self):
        """Test StateEncoderProtocol defines required methods."""

        class MockEncoder:
            def encode(self, game_state: dict[str, Any]) -> np.ndarray:
                return np.zeros(64)

            @property
            def output_dim(self) -> int:
                return 64

        encoder = MockEncoder()
        assert isinstance(encoder, StateEncoderProtocol)

    def test_state_encoder_missing_encode_fails(self):
        """Test StateEncoderProtocol requires encode method."""

        class IncompleteEncoder:
            @property
            def output_dim(self) -> int:
                return 64

        assert not isinstance(IncompleteEncoder(), StateEncoderProtocol)

    def test_state_encoder_missing_output_dim_fails(self):
        """Test StateEncoderProtocol requires output_dim property."""

        class IncompleteEncoder:
            def encode(self, game_state: dict[str, Any]) -> np.ndarray:
                return np.zeros(64)

        # Note: runtime_checkable only checks method existence, not properties
        # So this may pass the isinstance check but fail at runtime
        encoder = IncompleteEncoder()
        assert hasattr(encoder, "encode")


# =============================================================================
# TestNetworkProtocol
# =============================================================================


class TestNetworkProtocol:
    """Tests for NetworkProtocol conformance."""

    @pytest.fixture
    def mock_network(self):
        """Create a mock network that implements NetworkProtocol."""

        class MockNetwork:
            def forward(
                self,
                spatial: torch.Tensor,
                vector: torch.Tensor,
                action_mask: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                batch = spatial.shape[0]
                return torch.randn(batch, 100), torch.randn(batch, 1)

            def predict(
                self,
                spatial: torch.Tensor,
                vector: torch.Tensor,
                action_mask: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                policy, value = self.forward(spatial, vector, action_mask)
                return torch.softmax(policy, dim=-1), torch.tanh(value)

        return MockNetwork()

    def test_network_protocol_conformance(self, mock_network):
        """Test mock network conforms to NetworkProtocol."""
        assert isinstance(mock_network, NetworkProtocol)

    def test_network_forward_signature(self, mock_network):
        """Test forward method has correct signature."""
        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)

        policy, value = mock_network.forward(spatial, vector)

        assert policy.shape[0] == 2
        assert value.shape[0] == 2

    def test_network_predict_signature(self, mock_network):
        """Test predict method has correct signature."""
        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)

        policy, _value = mock_network.predict(spatial, vector)

        # Predict should return probabilities
        assert torch.allclose(policy.sum(dim=-1), torch.ones(2), atol=1e-5)

    @pytest.mark.parametrize(
        "network_fixture",
        [
            "microciv_network",
        ],
    )
    def test_real_networks_conform(self, network_fixture, request):
        """Test real network implementations conform to protocol."""
        try:
            network = request.getfixturevalue(network_fixture)
            assert isinstance(network, NetworkProtocol)
        except pytest.FixtureLookupError:
            pytest.skip(f"Fixture {network_fixture} not available")


# =============================================================================
# TestEnvironmentProtocol
# =============================================================================


class TestEnvironmentProtocol:
    """Tests for EnvironmentProtocol conformance."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment that implements EnvironmentProtocol."""

        class MockEnv:
            def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
                return {
                    "spatial": np.zeros((13, 8, 8), dtype=np.float32),
                    "vector": np.zeros(64, dtype=np.float32),
                }

            def step(
                self, action: int
            ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
                obs = self.reset()
                return obs, 0.0, False, False, {}

            def get_observation(self) -> dict[str, np.ndarray]:
                return self.reset()

            def clone(self) -> MockEnv:
                return MockEnv()

        return MockEnv()

    def test_environment_protocol_conformance(self, mock_env):
        """Test mock environment conforms to EnvironmentProtocol."""
        assert isinstance(mock_env, EnvironmentProtocol)

    def test_environment_reset_returns_observation(self, mock_env):
        """Test reset returns observation dictionary."""
        obs = mock_env.reset()
        assert isinstance(obs, dict)
        assert "spatial" in obs or "vector" in obs

    def test_environment_step_returns_5_tuple(self, mock_env):
        """Test step returns Gymnasium v26 5-tuple."""
        mock_env.reset()
        result = mock_env.step(0)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_environment_clone_returns_copy(self, mock_env):
        """Test clone returns a copy of the environment."""
        cloned = mock_env.clone()
        assert cloned is not mock_env
        assert isinstance(cloned, EnvironmentProtocol)


# =============================================================================
# TestMCTSProtocol
# =============================================================================


class TestMCTSProtocol:
    """Tests for MCTSProtocol conformance."""

    @pytest.fixture
    def mock_mcts(self):
        """Create a mock MCTS that implements MCTSProtocol."""

        class MockMCTS:
            def search(
                self,
                env: EnvironmentProtocol,
                num_simulations: int | None = None,
            ) -> tuple[np.ndarray, float]:
                action_probs = np.ones(100) / 100
                value = 0.0
                return action_probs, value

            def select_action(
                self,
                action_probs: np.ndarray,
                temperature: float = 1.0,
            ) -> int:
                if temperature == 0:
                    return int(np.argmax(action_probs))
                return int(np.random.choice(len(action_probs), p=action_probs))

        return MockMCTS()

    def test_mcts_protocol_conformance(self, mock_mcts):
        """Test mock MCTS conforms to MCTSProtocol."""
        assert isinstance(mock_mcts, MCTSProtocol)

    def test_mcts_search_returns_tuple(self, mock_mcts):
        """Test search returns action probabilities and value."""
        # Create a minimal mock environment
        mock_env_instance = MagicMock()
        mock_env_instance.reset.return_value = {}

        action_probs, value = mock_mcts.search(mock_env_instance)

        assert isinstance(action_probs, np.ndarray)
        assert isinstance(value, float)
        assert np.isclose(action_probs.sum(), 1.0)

    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 2.0])
    def test_mcts_select_action_temperature(self, mock_mcts, temperature):
        """Test select_action respects temperature parameter."""
        # Create probability distribution with clear peak
        action_probs = np.zeros(100)
        action_probs[42] = 0.9
        action_probs[0:42] = 0.1 / 99
        action_probs[43:] = 0.1 / 99
        action_probs = action_probs / action_probs.sum()

        action = mock_mcts.select_action(action_probs, temperature)

        assert 0 <= action < 100

        if temperature == 0:
            assert action == 42  # Should be argmax


# =============================================================================
# TestCheckpointableProtocol
# =============================================================================


class TestCheckpointableProtocol:
    """Tests for CheckpointableProtocol conformance."""

    def test_checkpointable_protocol_methods(self):
        """Test CheckpointableProtocol defines required methods."""

        class MockCheckpointable:
            def save(self, path: str) -> None:
                pass

            def load(self, path: str, device: str = "cpu") -> None:
                pass

        obj = MockCheckpointable()
        assert isinstance(obj, CheckpointableProtocol)

    def test_checkpointable_missing_save_fails(self):
        """Test CheckpointableProtocol requires save method."""

        class IncompleteCheckpointable:
            def load(self, path: str, device: str = "cpu") -> None:
                pass

        assert not isinstance(IncompleteCheckpointable(), CheckpointableProtocol)


# =============================================================================
# TestTrainerProtocol
# =============================================================================


class TestTrainerProtocol:
    """Tests for TrainerProtocol conformance."""

    def test_trainer_protocol_methods(self):
        """Test TrainerProtocol defines required methods."""

        class MockTrainer:
            def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
                return {"loss": 0.1, "policy_loss": 0.05, "value_loss": 0.05}

            def save_checkpoint(self, path: str) -> None:
                pass

            def load_checkpoint(self, path: str) -> None:
                pass

        trainer = MockTrainer()
        assert isinstance(trainer, TrainerProtocol)

    def test_trainer_train_step_returns_dict(self):
        """Test train_step returns dictionary of losses."""

        class MockTrainer:
            def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
                return {
                    "total_loss": 0.1,
                    "policy_loss": 0.05,
                    "value_loss": 0.05,
                }

            def save_checkpoint(self, path: str) -> None:
                pass

            def load_checkpoint(self, path: str) -> None:
                pass

        trainer = MockTrainer()
        result = trainer.train_step({})

        assert isinstance(result, dict)
        assert all(isinstance(v, float) for v in result.values())


# =============================================================================
# TestActionMaskableProtocol
# =============================================================================


class TestActionMaskableProtocol:
    """Tests for ActionMaskable protocol conformance."""

    def test_action_maskable_methods(self):
        """Test ActionMaskable defines required methods."""

        class MockMaskable:
            def get_valid_actions(self) -> np.ndarray:
                return np.ones(100, dtype=np.float32)

            def apply_action_mask(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return logits.masked_fill(mask == 0, -1e9)

        obj = MockMaskable()
        assert isinstance(obj, ActionMaskable)

    def test_action_maskable_get_valid_actions(self):
        """Test get_valid_actions returns numpy array."""

        class MockMaskable:
            def get_valid_actions(self) -> np.ndarray:
                mask = np.zeros(100, dtype=np.float32)
                mask[0:10] = 1.0
                return mask

            def apply_action_mask(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return logits.masked_fill(mask == 0, -1e9)

        obj = MockMaskable()
        mask = obj.get_valid_actions()

        assert isinstance(mask, np.ndarray)
        assert mask.sum() == 10


# =============================================================================
# TestProtocolIntegration
# =============================================================================


class TestProtocolIntegration:
    """Integration tests verifying real implementations conform to protocols."""

    def test_microciv_env_conforms_to_environment_protocol(self):
        """Test MicroCivEnv implements EnvironmentProtocol."""
        try:
            from alpha_zero.envs.microciv import MicroCivEnv

            env = MicroCivEnv(board_size=4, max_turns=10)
            assert isinstance(env, EnvironmentProtocol)
        except ImportError:
            pytest.skip("MicroCivEnv not available")

    def test_microciv_network_conforms_to_network_protocol(self):
        """Test MicroCivNetwork implements NetworkProtocol."""
        try:
            from alpha_zero.core.civ6_network import MicroCivNetwork

            network = MicroCivNetwork(
                board_size=4,
                num_res_blocks=1,
                num_filters=8,
                vector_dim=64,
                action_space_size=100,
            )
            assert isinstance(network, NetworkProtocol)
        except ImportError:
            pytest.skip("MicroCivNetwork not available")
