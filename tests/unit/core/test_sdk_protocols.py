"""Comprehensive tests for the SDK's src.core.protocols module.

Covers:
- All 10 runtime_checkable protocols (isinstance positive, negative, partial)
- MockStrategicEnvironment with full behavioural contract
- assert_valid_environment (passing + all failure modes)
- check_protocol_compliance (fully compliant, partial, multiple protocols)
- __all__ exports (count, importability, no duplicates)
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.core.exceptions import ProtocolViolationError
from src.core.protocols import (
    CheckpointableProtocol,
    MCTSProtocol,
    NetworkProtocol,
    RewardShaper,
    SelectionPolicy,
    Serializable,
    StateEncoder,
    StorageBackend,
    StrategicEnvironment,
    TrainerProtocol,
    assert_valid_environment,
    check_protocol_compliance,
)
from src.core import protocols as protocols_module
from src.core.types import ObservationSpec


# =========================================================================
# Helpers -- minimal mock implementations
# =========================================================================


class _Empty:
    """An empty class that satisfies no protocol."""

    pass


# ---------------------------------------------------------------------------
# MockStrategicEnvironment
# ---------------------------------------------------------------------------


class MockStrategicEnvironment:
    """Minimal mock that satisfies all StrategicEnvironment contracts.

    Board is 8x8, 4 channels, vector dim 16, action space size 64.
    """

    _SPATIAL_SHAPE = (4, 8, 8)
    _VECTOR_DIM = 16
    _ACTION_SIZE = 64

    def __init__(self) -> None:
        self._terminal = False
        self._step_count = 0
        self._seed: int | None = None

    @property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(spatial_shape=self._SPATIAL_SHAPE, vector_dim=self._VECTOR_DIM)

    @property
    def action_space_size(self) -> int:
        return self._ACTION_SIZE

    def reset(self, *, seed: int | None = None) -> dict[str, NDArray[np.float32]]:
        self._terminal = False
        self._step_count = 0
        self._seed = seed
        return self._make_obs()

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        terminated = self._step_count >= 100
        self._terminal = terminated
        return self._make_obs(), 0.0, terminated, False, {}

    def get_valid_actions(self) -> NDArray[np.bool_]:
        return np.ones(self._ACTION_SIZE, dtype=np.bool_)

    def get_observation(self) -> dict[str, NDArray[np.float32]]:
        return self._make_obs()

    def is_terminal(self) -> bool:
        return self._terminal

    def clone(self) -> MockStrategicEnvironment:
        return copy.deepcopy(self)

    def get_symmetries(
        self,
        observation: NDArray[np.float32],
        policy: NDArray[np.float32],
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        return [(observation, policy)]

    # -- internal helper ---------------------------------------------------

    def _make_obs(self) -> dict[str, NDArray[np.float32]]:
        return {
            "spatial": np.zeros(self._SPATIAL_SHAPE, dtype=np.float32),
            "vector": np.zeros(self._VECTOR_DIM, dtype=np.float32),
        }


# ---------------------------------------------------------------------------
# Other minimal mock implementations
# ---------------------------------------------------------------------------


class MockSerializable:
    def to_dict(self) -> dict[str, Any]:
        return {"key": "value"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockSerializable:
        return cls()


class MockStateEncoder:
    @property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(spatial_shape=(4, 8, 8), vector_dim=16)

    def encode(self, raw_state: dict[str, Any]) -> dict[str, NDArray[np.float32]]:
        return {
            "spatial": np.zeros((4, 8, 8), dtype=np.float32),
            "vector": np.zeros(16, dtype=np.float32),
        }

    def decode_action(self, action_index: int) -> dict[str, Any]:
        return {"type": "noop", "index": action_index}


class MockNetworkProtocol:
    def forward(
        self,
        spatial: Any,
        vector: Any,
        action_mask: Any | None = None,
    ) -> tuple[Any, Any]:
        return np.zeros(10), np.array([0.0])

    def predict(
        self,
        spatial: Any,
        vector: Any,
        action_mask: Any | None = None,
    ) -> tuple[Any, Any]:
        return np.ones(10) / 10, np.array([0.0])


class MockMCTSProtocol:
    def search(
        self,
        env: StrategicEnvironment,
        num_simulations: int | None = None,
    ) -> tuple[NDArray[np.float32], float]:
        n = env.action_space_size
        probs = np.ones(n, dtype=np.float32) / n
        return probs, 0.0

    def select_action(
        self,
        action_probs: NDArray[np.float32],
        temperature: float = 1.0,
        action_mask: NDArray[np.bool_] | None = None,
    ) -> int:
        return int(np.argmax(action_probs))


class MockSelectionPolicy:
    def select_child(
        self,
        parent_visit_count: int,
        children_priors: NDArray[np.float32],
        children_values: NDArray[np.float32],
        children_visit_counts: NDArray[np.int64],
        valid_mask: NDArray[np.bool_],
    ) -> int:
        valid_indices = np.where(valid_mask)[0]
        return int(valid_indices[0])


class MockRewardShaper:
    def shape_reward(
        self,
        raw_reward: float,
        observation: dict[str, NDArray[np.float32]],
        action: int,
        info: dict[str, Any],
    ) -> float:
        return raw_reward


class MockCheckpointable:
    def save(self, path: str) -> None:
        pass

    def load(self, path: str, device: str = "cpu") -> None:
        pass


class MockTrainer:
    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        return {"total_loss": 0.1}

    def save_checkpoint(self, path: str) -> None:
        pass

    def load_checkpoint(self, path: str) -> None:
        pass


class MockStorageBackend:
    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def save(self, key: str, data: bytes) -> None:
        self._data[key] = data

    def load(self, key: str) -> bytes:
        return self._data[key]

    def exists(self, key: str) -> bool:
        return key in self._data

    def list_keys(self, prefix: str = "") -> list[str]:
        return [k for k in self._data if k.startswith(prefix)]

    def delete(self, key: str) -> None:
        self._data.pop(key, None)


# ---------------------------------------------------------------------------
# Partial implementations (missing one method each)
# ---------------------------------------------------------------------------


class _PartialSerializable:
    """Has to_dict but no from_dict."""

    def to_dict(self) -> dict[str, Any]:
        return {}


class _PartialStrategicEnvironment:
    """Has most StrategicEnvironment methods but missing clone()."""

    @property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(spatial_shape=(4, 8, 8), vector_dim=16)

    @property
    def action_space_size(self) -> int:
        return 64

    def reset(self, *, seed: int | None = None) -> dict[str, NDArray[np.float32]]:
        return {}

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        return {}, 0.0, False, False, {}

    def get_valid_actions(self) -> NDArray[np.bool_]:
        return np.ones(64, dtype=np.bool_)

    def get_observation(self) -> dict[str, NDArray[np.float32]]:
        return {}

    def is_terminal(self) -> bool:
        return False

    # clone() is deliberately missing

    def get_symmetries(
        self,
        observation: NDArray[np.float32],
        policy: NDArray[np.float32],
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        return [(observation, policy)]


class _PartialNetwork:
    """Has forward but no predict."""

    def forward(self, spatial: Any, vector: Any, action_mask: Any | None = None) -> tuple[Any, Any]:
        return None, None


class _PartialStorageBackend:
    """Has save and load but missing exists, list_keys, delete."""

    def save(self, key: str, data: bytes) -> None:
        pass

    def load(self, key: str) -> bytes:
        return b""


class _PartialTrainer:
    """Has train_step but no save_checkpoint or load_checkpoint."""

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        return {}


# =========================================================================
# Section 1: Serializable protocol
# =========================================================================


class TestSerializableProtocol:
    """Tests for the Serializable protocol isinstance checks."""

    def test_full_implementation_is_instance(self) -> None:
        assert isinstance(MockSerializable(), Serializable)

    def test_empty_class_is_not_instance(self) -> None:
        assert not isinstance(_Empty(), Serializable)

    def test_partial_missing_from_dict_is_not_instance(self) -> None:
        assert not isinstance(_PartialSerializable(), Serializable)


# =========================================================================
# Section 2: StrategicEnvironment protocol
# =========================================================================


class TestStrategicEnvironmentProtocol:
    """Tests for the StrategicEnvironment protocol isinstance checks."""

    def test_full_implementation_is_instance(self) -> None:
        assert isinstance(MockStrategicEnvironment(), StrategicEnvironment)

    def test_empty_class_is_not_instance(self) -> None:
        assert not isinstance(_Empty(), StrategicEnvironment)

    def test_partial_missing_clone_is_not_instance(self) -> None:
        assert not isinstance(_PartialStrategicEnvironment(), StrategicEnvironment)

    def test_mock_observation_spec(self) -> None:
        env = MockStrategicEnvironment()
        spec = env.observation_spec
        assert isinstance(spec, ObservationSpec)
        assert spec.spatial_shape == (4, 8, 8)
        assert spec.vector_dim == 16

    def test_mock_action_space_size(self) -> None:
        env = MockStrategicEnvironment()
        assert env.action_space_size == 64

    def test_mock_reset_returns_valid_obs(self) -> None:
        env = MockStrategicEnvironment()
        obs = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert "spatial" in obs
        assert "vector" in obs
        assert obs["spatial"].shape == (4, 8, 8)
        assert obs["vector"].shape == (16,)

    def test_mock_step_returns_5_tuple(self) -> None:
        env = MockStrategicEnvironment()
        env.reset()
        result = env.step(0)
        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_mock_get_valid_actions_shape(self) -> None:
        env = MockStrategicEnvironment()
        env.reset()
        mask = env.get_valid_actions()
        assert mask.shape == (64,)
        assert mask.dtype == np.bool_
        assert np.all(mask)

    def test_mock_is_terminal_false_after_reset(self) -> None:
        env = MockStrategicEnvironment()
        env.reset()
        assert not env.is_terminal()

    def test_mock_clone_returns_different_object(self) -> None:
        env = MockStrategicEnvironment()
        env.reset(seed=7)
        cloned = env.clone()
        assert cloned is not env

    def test_mock_clone_independent(self) -> None:
        env = MockStrategicEnvironment()
        env.reset(seed=7)
        cloned = env.clone()
        # Step the clone, original should not change
        cloned.step(0)
        assert env._step_count == 0
        assert cloned._step_count == 1

    def test_mock_get_symmetries_returns_identity(self) -> None:
        env = MockStrategicEnvironment()
        env.reset()
        obs = env.get_observation()
        policy = np.ones(64, dtype=np.float32) / 64
        syms = env.get_symmetries(obs["spatial"], policy)
        assert isinstance(syms, list)
        assert len(syms) == 1
        np.testing.assert_array_equal(syms[0][0], obs["spatial"])
        np.testing.assert_array_equal(syms[0][1], policy)


# =========================================================================
# Section 3: StateEncoder protocol
# =========================================================================


class TestStateEncoderProtocol:
    """Tests for the StateEncoder protocol isinstance checks."""

    def test_full_implementation_is_instance(self) -> None:
        assert isinstance(MockStateEncoder(), StateEncoder)

    def test_empty_class_is_not_instance(self) -> None:
        assert not isinstance(_Empty(), StateEncoder)

    def test_partial_missing_decode_action(self) -> None:
        class Partial:
            @property
            def observation_spec(self) -> ObservationSpec:
                return ObservationSpec(spatial_shape=(4, 8, 8), vector_dim=16)

            def encode(self, raw_state: dict[str, Any]) -> dict[str, NDArray[np.float32]]:
                return {}

        assert not isinstance(Partial(), StateEncoder)


# =========================================================================
# Section 4: NetworkProtocol
# =========================================================================


class TestNetworkProtocol:
    """Tests for the NetworkProtocol isinstance checks."""

    def test_full_implementation_is_instance(self) -> None:
        assert isinstance(MockNetworkProtocol(), NetworkProtocol)

    def test_empty_class_is_not_instance(self) -> None:
        assert not isinstance(_Empty(), NetworkProtocol)

    def test_partial_missing_predict_is_not_instance(self) -> None:
        assert not isinstance(_PartialNetwork(), NetworkProtocol)


# =========================================================================
# Section 5: MCTSProtocol
# =========================================================================


class TestMCTSProtocol:
    """Tests for the MCTSProtocol isinstance checks."""

    def test_full_implementation_is_instance(self) -> None:
        assert isinstance(MockMCTSProtocol(), MCTSProtocol)

    def test_empty_class_is_not_instance(self) -> None:
        assert not isinstance(_Empty(), MCTSProtocol)

    def test_partial_missing_select_action(self) -> None:
        class Partial:
            def search(
                self, env: Any, num_simulations: int | None = None
            ) -> tuple[NDArray[np.float32], float]:
                return np.zeros(1, dtype=np.float32), 0.0

        assert not isinstance(Partial(), MCTSProtocol)


# =========================================================================
# Section 6: SelectionPolicy
# =========================================================================


class TestSelectionPolicyProtocol:
    """Tests for the SelectionPolicy protocol isinstance checks."""

    def test_full_implementation_is_instance(self) -> None:
        assert isinstance(MockSelectionPolicy(), SelectionPolicy)

    def test_empty_class_is_not_instance(self) -> None:
        assert not isinstance(_Empty(), SelectionPolicy)

    def test_partial_wrong_method_name(self) -> None:
        class Wrong:
            def choose_child(self, *args: Any, **kwargs: Any) -> int:
                return 0

        assert not isinstance(Wrong(), SelectionPolicy)


# =========================================================================
# Section 7: RewardShaper
# =========================================================================


class TestRewardShaperProtocol:
    """Tests for the RewardShaper protocol isinstance checks."""

    def test_full_implementation_is_instance(self) -> None:
        assert isinstance(MockRewardShaper(), RewardShaper)

    def test_empty_class_is_not_instance(self) -> None:
        assert not isinstance(_Empty(), RewardShaper)

    def test_partial_wrong_method_name(self) -> None:
        class Wrong:
            def transform_reward(self, *args: Any, **kwargs: Any) -> float:
                return 0.0

        assert not isinstance(Wrong(), RewardShaper)


# =========================================================================
# Section 8: CheckpointableProtocol
# =========================================================================


class TestCheckpointableProtocol:
    """Tests for the CheckpointableProtocol isinstance checks."""

    def test_full_implementation_is_instance(self) -> None:
        assert isinstance(MockCheckpointable(), CheckpointableProtocol)

    def test_empty_class_is_not_instance(self) -> None:
        assert not isinstance(_Empty(), CheckpointableProtocol)

    def test_partial_missing_load(self) -> None:
        class Partial:
            def save(self, path: str) -> None:
                pass

        assert not isinstance(Partial(), CheckpointableProtocol)


# =========================================================================
# Section 9: TrainerProtocol
# =========================================================================


class TestTrainerProtocol:
    """Tests for the TrainerProtocol isinstance checks."""

    def test_full_implementation_is_instance(self) -> None:
        assert isinstance(MockTrainer(), TrainerProtocol)

    def test_empty_class_is_not_instance(self) -> None:
        assert not isinstance(_Empty(), TrainerProtocol)

    def test_partial_missing_checkpoints_is_not_instance(self) -> None:
        assert not isinstance(_PartialTrainer(), TrainerProtocol)


# =========================================================================
# Section 10: StorageBackend
# =========================================================================


class TestStorageBackendProtocol:
    """Tests for the StorageBackend protocol isinstance checks."""

    def test_full_implementation_is_instance(self) -> None:
        assert isinstance(MockStorageBackend(), StorageBackend)

    def test_empty_class_is_not_instance(self) -> None:
        assert not isinstance(_Empty(), StorageBackend)

    def test_partial_missing_exists_list_delete(self) -> None:
        assert not isinstance(_PartialStorageBackend(), StorageBackend)


# =========================================================================
# Section 11: assert_valid_environment -- passing cases
# =========================================================================


class TestAssertValidEnvironmentPasses:
    """assert_valid_environment should pass on a fully compliant mock."""

    def test_passes_on_valid_mock(self) -> None:
        env = MockStrategicEnvironment()
        # Should NOT raise
        assert_valid_environment(env)

    def test_passes_with_different_seed(self) -> None:
        env = MockStrategicEnvironment()
        env.reset(seed=999)
        assert_valid_environment(env)


# =========================================================================
# Section 12: assert_valid_environment -- failure modes
# =========================================================================


class TestAssertValidEnvironmentFailures:
    """assert_valid_environment should raise ProtocolViolationError for violations."""

    def test_wrong_observation_spec_type(self) -> None:
        class BadSpecEnv(MockStrategicEnvironment):
            @property
            def observation_spec(self) -> Any:
                return {"spatial_shape": (4, 8, 8), "vector_dim": 16}

        with pytest.raises(ProtocolViolationError, match="observation_spec"):
            assert_valid_environment(BadSpecEnv())

    def test_action_space_size_zero(self) -> None:
        class ZeroActionEnv(MockStrategicEnvironment):
            @property
            def action_space_size(self) -> int:
                return 0

        with pytest.raises(ProtocolViolationError, match="action_space_size"):
            assert_valid_environment(ZeroActionEnv())

    def test_action_space_size_negative(self) -> None:
        class NegativeActionEnv(MockStrategicEnvironment):
            @property
            def action_space_size(self) -> int:
                return -5

        with pytest.raises(ProtocolViolationError, match="action_space_size"):
            assert_valid_environment(NegativeActionEnv())

    def test_reset_returns_non_dict(self) -> None:
        class BadResetEnv(MockStrategicEnvironment):
            def reset(self, *, seed: int | None = None) -> Any:
                self._terminal = False
                self._step_count = 0
                return "not a dict"

        with pytest.raises(ProtocolViolationError, match="reset"):
            assert_valid_environment(BadResetEnv())

    def test_reset_wrong_spatial_shape(self) -> None:
        class WrongSpatialEnv(MockStrategicEnvironment):
            def reset(self, *, seed: int | None = None) -> dict[str, NDArray[np.float32]]:
                self._terminal = False
                self._step_count = 0
                return {
                    "spatial": np.zeros((3, 8, 8), dtype=np.float32),  # wrong channels
                    "vector": np.zeros(self._VECTOR_DIM, dtype=np.float32),
                }

        with pytest.raises(ProtocolViolationError, match="spatial shape"):
            assert_valid_environment(WrongSpatialEnv())

    def test_is_terminal_true_after_reset(self) -> None:
        class TerminalAfterResetEnv(MockStrategicEnvironment):
            def reset(self, *, seed: int | None = None) -> dict[str, NDArray[np.float32]]:
                self._terminal = True  # Bug: terminal immediately
                self._step_count = 0
                return self._make_obs()

        with pytest.raises(ProtocolViolationError, match="is_terminal.*True.*after reset"):
            assert_valid_environment(TerminalAfterResetEnv())

    def test_get_valid_actions_wrong_shape(self) -> None:
        class WrongMaskShapeEnv(MockStrategicEnvironment):
            def get_valid_actions(self) -> NDArray[np.bool_]:
                return np.ones(32, dtype=np.bool_)  # 32 != 64

        with pytest.raises(ProtocolViolationError, match="get_valid_actions.*shape"):
            assert_valid_environment(WrongMaskShapeEnv())

    def test_step_returns_wrong_tuple_length(self) -> None:
        class BadStepEnv(MockStrategicEnvironment):
            def step(self, action: int) -> Any:
                self._step_count += 1
                return self._make_obs(), 0.0, False  # Only 3 elements

        with pytest.raises(ProtocolViolationError, match="step.*5-tuple|step.*length"):
            assert_valid_environment(BadStepEnv())

    def test_step_reward_outside_range(self) -> None:
        class BadRewardEnv(MockStrategicEnvironment):
            def step(
                self, action: int
            ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
                self._step_count += 1
                return self._make_obs(), 5.0, False, False, {}  # reward > 1.0

        with pytest.raises(ProtocolViolationError, match="reward.*outside"):
            assert_valid_environment(BadRewardEnv())

    def test_clone_returns_same_object(self) -> None:
        class BadCloneEnv(MockStrategicEnvironment):
            def clone(self) -> BadCloneEnv:
                return self  # Returns self instead of a copy

        with pytest.raises(ProtocolViolationError, match="clone.*same object"):
            assert_valid_environment(BadCloneEnv())

    def test_get_symmetries_returns_empty_list(self) -> None:
        class EmptySymEnv(MockStrategicEnvironment):
            def get_symmetries(
                self,
                observation: NDArray[np.float32],
                policy: NDArray[np.float32],
            ) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
                return []

        with pytest.raises(ProtocolViolationError, match="get_symmetries.*empty list"):
            assert_valid_environment(EmptySymEnv())

    def test_structurally_missing_method(self) -> None:
        """An object missing required methods fails the structural check."""

        class Incomplete:
            @property
            def observation_spec(self) -> ObservationSpec:
                return ObservationSpec(spatial_shape=(4, 8, 8), vector_dim=16)

            @property
            def action_space_size(self) -> int:
                return 64

        with pytest.raises(ProtocolViolationError, match="missing StrategicEnvironment methods"):
            assert_valid_environment(Incomplete())  # type: ignore[arg-type]


# =========================================================================
# Section 13: check_protocol_compliance
# =========================================================================


class TestCheckProtocolCompliance:
    """Tests for the check_protocol_compliance utility function."""

    # -- Fully compliant objects return empty lists -------------------------

    def test_fully_compliant_strategic_environment(self) -> None:
        missing = check_protocol_compliance(MockStrategicEnvironment(), StrategicEnvironment)
        assert missing == []

    def test_fully_compliant_serializable(self) -> None:
        missing = check_protocol_compliance(MockSerializable(), Serializable)
        assert missing == []

    def test_fully_compliant_network(self) -> None:
        missing = check_protocol_compliance(MockNetworkProtocol(), NetworkProtocol)
        assert missing == []

    def test_fully_compliant_storage_backend(self) -> None:
        missing = check_protocol_compliance(MockStorageBackend(), StorageBackend)
        assert missing == []

    def test_fully_compliant_trainer(self) -> None:
        missing = check_protocol_compliance(MockTrainer(), TrainerProtocol)
        assert missing == []

    def test_fully_compliant_mcts(self) -> None:
        missing = check_protocol_compliance(MockMCTSProtocol(), MCTSProtocol)
        assert missing == []

    def test_fully_compliant_selection_policy(self) -> None:
        missing = check_protocol_compliance(MockSelectionPolicy(), SelectionPolicy)
        assert missing == []

    def test_fully_compliant_reward_shaper(self) -> None:
        missing = check_protocol_compliance(MockRewardShaper(), RewardShaper)
        assert missing == []

    def test_fully_compliant_checkpointable(self) -> None:
        missing = check_protocol_compliance(MockCheckpointable(), CheckpointableProtocol)
        assert missing == []

    def test_fully_compliant_state_encoder(self) -> None:
        missing = check_protocol_compliance(MockStateEncoder(), StateEncoder)
        assert missing == []

    # -- Partial implementations return missing method names ----------------

    def test_empty_class_against_strategic_environment(self) -> None:
        missing = check_protocol_compliance(_Empty(), StrategicEnvironment)
        assert len(missing) > 0
        # Must include at least these
        for name in ["reset", "step", "clone", "is_terminal", "get_valid_actions"]:
            assert name in missing, f"Expected '{name}' in missing list"

    def test_partial_strategic_environment_missing_clone(self) -> None:
        missing = check_protocol_compliance(
            _PartialStrategicEnvironment(), StrategicEnvironment
        )
        assert "clone" in missing

    def test_partial_network_missing_predict(self) -> None:
        missing = check_protocol_compliance(_PartialNetwork(), NetworkProtocol)
        assert "predict" in missing

    def test_partial_storage_backend(self) -> None:
        missing = check_protocol_compliance(_PartialStorageBackend(), StorageBackend)
        assert "exists" in missing
        assert "list_keys" in missing
        assert "delete" in missing

    def test_partial_trainer_missing_checkpoints(self) -> None:
        missing = check_protocol_compliance(_PartialTrainer(), TrainerProtocol)
        assert "save_checkpoint" in missing
        assert "load_checkpoint" in missing

    def test_result_is_sorted(self) -> None:
        missing = check_protocol_compliance(_Empty(), StrategicEnvironment)
        assert missing == sorted(missing)

    def test_empty_class_against_serializable(self) -> None:
        missing = check_protocol_compliance(_Empty(), Serializable)
        assert "to_dict" in missing
        assert "from_dict" in missing

    def test_empty_class_against_checkpointable(self) -> None:
        missing = check_protocol_compliance(_Empty(), CheckpointableProtocol)
        assert "save" in missing
        assert "load" in missing


# =========================================================================
# Section 14: __all__ exports
# =========================================================================


class TestAllExports:
    """Verify __all__ is complete and every name is importable."""

    def test_all_has_12_entries(self) -> None:
        """10 protocols + 2 utility functions = 12."""
        assert len(protocols_module.__all__) == 12

    def test_all_names_are_importable(self) -> None:
        for name in protocols_module.__all__:
            assert hasattr(protocols_module, name), (
                f"{name} is listed in __all__ but not accessible on the module"
            )

    def test_all_contains_no_duplicates(self) -> None:
        assert len(protocols_module.__all__) == len(set(protocols_module.__all__))

    def test_all_contains_all_protocols(self) -> None:
        expected_protocols = [
            "CheckpointableProtocol",
            "MCTSProtocol",
            "NetworkProtocol",
            "RewardShaper",
            "SelectionPolicy",
            "Serializable",
            "StateEncoder",
            "StorageBackend",
            "StrategicEnvironment",
            "TrainerProtocol",
        ]
        for name in expected_protocols:
            assert name in protocols_module.__all__, f"Missing protocol: {name}"

    def test_all_contains_utility_functions(self) -> None:
        assert "assert_valid_environment" in protocols_module.__all__
        assert "check_protocol_compliance" in protocols_module.__all__

    def test_all_is_sorted(self) -> None:
        assert protocols_module.__all__ == sorted(protocols_module.__all__)
