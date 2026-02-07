"""Additional coverage tests for src.core.protocols.

Targets uncovered lines in:
- check_protocol_compliance: MRO traversal, callable vs property checks
- assert_valid_environment: exception branches, missing keys, type mismatches
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
from src.core.types import ObservationSpec

# =========================================================================
# Helpers -- MinimalValidEnv and broken variants
# =========================================================================

_BOARD = 4
_CHANNELS = 2
_VECTOR_DIM = 3
_ACTION_SIZE = 5
_SPATIAL_SHAPE = (_CHANNELS, _BOARD, _BOARD)


class MinimalValidEnv:
    """Minimal environment that satisfies all StrategicEnvironment contracts."""

    def __init__(
        self,
        board_size: int = _BOARD,
        action_size: int = _ACTION_SIZE,
        vector_dim: int = _VECTOR_DIM,
    ) -> None:
        self._obs_spec = ObservationSpec(
            spatial_shape=(_CHANNELS, board_size, board_size),
            vector_dim=vector_dim,
        )
        self._action_size = action_size
        self._terminated = False

    @property
    def observation_spec(self) -> ObservationSpec:
        return self._obs_spec

    @property
    def action_space_size(self) -> int:
        return self._action_size

    def reset(self, *, seed: int | None = None) -> dict[str, NDArray[np.float32]]:
        self._terminated = False
        return self._make_obs()

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        return self._make_obs(), 0.0, False, False, {}

    def get_valid_actions(self) -> NDArray[np.bool_]:
        return np.ones(self._action_size, dtype=np.bool_)

    def get_observation(self) -> dict[str, NDArray[np.float32]]:
        return self._make_obs()

    def is_terminal(self) -> bool:
        return self._terminated

    def clone(self) -> MinimalValidEnv:
        return copy.deepcopy(self)

    def get_symmetries(
        self,
        observation: NDArray[np.float32],
        policy: NDArray[np.float32],
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        return [(observation, policy)]

    def _make_obs(self) -> dict[str, NDArray[np.float32]]:
        return {
            "spatial": np.zeros(self._obs_spec.spatial_shape, dtype=np.float32),
            "vector": np.zeros(self._obs_spec.vector_dim, dtype=np.float32),
        }


# =========================================================================
# 1. TestCheckProtocolCompliance -- target lines 1100-1121
# =========================================================================


class TestCheckProtocolComplianceCoverage:
    """Tests targeting MRO traversal and callable-check logic in
    check_protocol_compliance (lines 1100-1121)."""

    def test_non_callable_attribute_where_callable_expected(self) -> None:
        """An object that has a name matching a protocol method but the
        attribute is not callable should be flagged as missing."""

        class BadReset:
            """Has 'reset' as a plain string attribute, not a method."""

            observation_spec = ObservationSpec(spatial_shape=(2, 4, 4), vector_dim=3)
            action_space_size = 5
            reset = "not_callable"
            step = "not_callable"
            get_valid_actions = "not_callable"
            get_observation = "not_callable"
            is_terminal = "not_callable"
            clone = "not_callable"
            get_symmetries = "not_callable"

        missing = check_protocol_compliance(BadReset(), StrategicEnvironment)
        # All method-like attributes are strings (non-callable), so they
        # should be reported as missing.
        assert "reset" in missing
        assert "step" in missing
        assert "clone" in missing

    def test_property_attribute_not_flagged_as_non_callable(self) -> None:
        """Properties are not callable on instances.  check_protocol_compliance
        should NOT flag them as missing when the protocol defines them."""

        class PropEnv(MinimalValidEnv):
            """Overrides observation_spec and action_space_size as properties
            (same as MinimalValidEnv, but explicitly verifying the property
            path in check_protocol_compliance)."""

        missing = check_protocol_compliance(PropEnv(), StrategicEnvironment)
        assert "observation_spec" not in missing
        assert "action_space_size" not in missing

    def test_fully_compliant_returns_empty(self) -> None:
        assert check_protocol_compliance(MinimalValidEnv(), StrategicEnvironment) == []

    def test_partially_compliant_returns_missing(self) -> None:
        class Partial:
            def search(self, env: Any, num_simulations: int | None = None) -> Any:
                pass

        missing = check_protocol_compliance(Partial(), MCTSProtocol)
        assert "select_action" in missing
        assert "search" not in missing

    def test_completely_non_compliant(self) -> None:
        class Empty:
            pass

        missing = check_protocol_compliance(Empty(), StorageBackend)
        assert len(missing) == 5
        for name in ["save", "load", "exists", "list_keys", "delete"]:
            assert name in missing

    @pytest.mark.parametrize(
        ("protocol", "expected_methods"),
        [
            (Serializable, ["from_dict", "to_dict"]),
            (StateEncoder, ["decode_action", "encode", "observation_spec"]),
            (NetworkProtocol, ["forward", "predict"]),
            (MCTSProtocol, ["search", "select_action"]),
            (SelectionPolicy, ["select_child"]),
            (RewardShaper, ["shape_reward"]),
            (CheckpointableProtocol, ["load", "save"]),
            (TrainerProtocol, ["load_checkpoint", "save_checkpoint", "train_step"]),
            (StorageBackend, ["delete", "exists", "list_keys", "load", "save"]),
        ],
        ids=[
            "Serializable",
            "StateEncoder",
            "NetworkProtocol",
            "MCTSProtocol",
            "SelectionPolicy",
            "RewardShaper",
            "CheckpointableProtocol",
            "TrainerProtocol",
            "StorageBackend",
        ],
    )
    def test_empty_class_reports_all_methods(
        self, protocol: type, expected_methods: list[str]
    ) -> None:
        """An empty class should be missing every method of each protocol."""

        class Empty:
            pass

        missing = check_protocol_compliance(Empty(), protocol)
        assert sorted(missing) == sorted(expected_methods)

    def test_mro_traversal_with_inherited_protocol(self) -> None:
        """Protocol members defined in parent protocol classes should also
        be detected through MRO traversal."""
        # StorageBackend defines 5 methods; an empty class misses all.
        missing = check_protocol_compliance(object(), StorageBackend)
        assert len(missing) == 5


# =========================================================================
# 2. TestAssertValidEnvironment -- target uncovered branches
# =========================================================================


class TestAssertValidEnvironmentCoverage:
    """Tests targeting all uncovered branches in assert_valid_environment."""

    # -- observation_spec exception branch (lines 866-867) -----------------

    def test_observation_spec_raises_exception(self) -> None:
        """Property must pass hasattr (which only catches AttributeError)
        during structural check, then raise during the actual validation.
        We use a counter so it succeeds on the first access (in
        check_protocol_compliance via hasattr/getattr) but raises on the
        second access (in assert_valid_environment's property check)."""

        class BadSpecEnv(MinimalValidEnv):
            _spec_access_count = 0

            @property
            def observation_spec(self) -> ObservationSpec:
                BadSpecEnv._spec_access_count += 1
                if BadSpecEnv._spec_access_count > 2:
                    raise RuntimeError("broken obs spec")
                return ObservationSpec(
                    spatial_shape=(_CHANNELS, _BOARD, _BOARD),
                    vector_dim=_VECTOR_DIM,
                )

        BadSpecEnv._spec_access_count = 0
        with pytest.raises(ProtocolViolationError, match="observation_spec raised"):
            assert_valid_environment(BadSpecEnv())

    # -- action_space_size exception branch (lines 875-876) ----------------

    def test_action_space_size_raises_exception(self) -> None:
        """Same approach as observation_spec: succeed during structural
        check, then raise during property validation."""

        class BadActionEnv(MinimalValidEnv):
            _size_access_count = 0

            @property
            def action_space_size(self) -> int:
                BadActionEnv._size_access_count += 1
                if BadActionEnv._size_access_count > 2:
                    raise RuntimeError("broken action size")
                return _ACTION_SIZE

        BadActionEnv._size_access_count = 0
        with pytest.raises(ProtocolViolationError, match="action_space_size raised"):
            assert_valid_environment(BadActionEnv())

    # -- action_space_size returns non-int (line 871) ----------------------

    def test_action_space_size_non_int(self) -> None:
        class FloatActionEnv(MinimalValidEnv):
            @property
            def action_space_size(self) -> Any:
                return 5.5

        with pytest.raises(ProtocolViolationError, match="action_space_size"):
            assert_valid_environment(FloatActionEnv())

    # -- reset() raises exception (lines 893-894) --------------------------

    def test_reset_raises_exception(self) -> None:
        class ExplodingResetEnv(MinimalValidEnv):
            def reset(self, *, seed: int | None = None) -> Any:
                raise ValueError("reset is broken")

        with pytest.raises(ProtocolViolationError, match=r"reset.*raised.*ValueError"):
            assert_valid_environment(ExplodingResetEnv())

    # -- reset() missing 'spatial' key (line 905) --------------------------

    def test_reset_missing_spatial_key(self) -> None:
        class NoSpatialEnv(MinimalValidEnv):
            def reset(self, *, seed: int | None = None) -> dict[str, Any]:
                self._terminated = False
                return {
                    "vector": np.zeros(self._obs_spec.vector_dim, dtype=np.float32),
                }

        with pytest.raises(ProtocolViolationError, match="missing 'spatial'"):
            assert_valid_environment(NoSpatialEnv())

    # -- reset() missing 'vector' key (line 915) ---------------------------

    def test_reset_missing_vector_key(self) -> None:
        class NoVectorEnv(MinimalValidEnv):
            def reset(self, *, seed: int | None = None) -> dict[str, Any]:
                self._terminated = False
                return {
                    "spatial": np.zeros(
                        self._obs_spec.spatial_shape, dtype=np.float32
                    ),
                }

        with pytest.raises(ProtocolViolationError, match="missing 'vector'"):
            assert_valid_environment(NoVectorEnv())

    # -- reset() vector shape mismatch (line 920) --------------------------

    def test_reset_vector_shape_mismatch(self) -> None:
        class WrongVectorEnv(MinimalValidEnv):
            def reset(self, *, seed: int | None = None) -> dict[str, Any]:
                self._terminated = False
                return {
                    "spatial": np.zeros(
                        self._obs_spec.spatial_shape, dtype=np.float32
                    ),
                    "vector": np.zeros(99, dtype=np.float32),  # wrong dim
                }

        with pytest.raises(ProtocolViolationError, match="vector shape"):
            assert_valid_environment(WrongVectorEnv())

    # -- is_terminal() raises exception (lines 931-932) --------------------

    def test_is_terminal_raises_exception(self) -> None:
        class BrokenTerminalEnv(MinimalValidEnv):
            def is_terminal(self) -> bool:
                raise RuntimeError("terminal check broken")

        with pytest.raises(ProtocolViolationError, match=r"is_terminal.*raised"):
            assert_valid_environment(BrokenTerminalEnv())

    # -- get_valid_actions() all False after reset (lines 944-947) ---------

    def test_get_valid_actions_all_false(self) -> None:
        class NoValidActionsEnv(MinimalValidEnv):
            def get_valid_actions(self) -> NDArray[np.bool_]:
                return np.zeros(self._action_size, dtype=np.bool_)

        with pytest.raises(ProtocolViolationError, match="no True entries"):
            assert_valid_environment(NoValidActionsEnv())

    # -- get_valid_actions() raises exception (lines 949-951) --------------

    def test_get_valid_actions_raises_exception(self) -> None:
        class ExplodingActionsEnv(MinimalValidEnv):
            def get_valid_actions(self) -> NDArray[np.bool_]:
                raise RuntimeError("actions broken")

        with pytest.raises(ProtocolViolationError, match=r"get_valid_actions.*raised"):
            assert_valid_environment(ExplodingActionsEnv())

    # -- get_observation() returns non-dict (lines 959-961) ----------------

    def test_get_observation_returns_non_dict(self) -> None:
        class BadGetObsEnv(MinimalValidEnv):
            def get_observation(self) -> Any:
                return "not a dict"

        with pytest.raises(ProtocolViolationError, match=r"get_observation.*returned"):
            assert_valid_environment(BadGetObsEnv())

    # -- get_observation() raises exception (lines 964-966) ----------------

    def test_get_observation_raises_exception(self) -> None:
        class ExplodingGetObsEnv(MinimalValidEnv):
            def get_observation(self) -> dict[str, Any]:
                raise RuntimeError("get_observation broken")

        with pytest.raises(ProtocolViolationError, match=r"get_observation.*raised"):
            assert_valid_environment(ExplodingGetObsEnv())

    # -- step() obs not dict (line 984) ------------------------------------

    def test_step_obs_not_dict(self) -> None:
        class BadStepObsEnv(MinimalValidEnv):
            def step(self, action: int) -> Any:
                return "not_dict", 0.0, False, False, {}

        with pytest.raises(ProtocolViolationError, match=r"step.*obs.*expected dict"):
            assert_valid_environment(BadStepObsEnv())

    # -- step() reward not float (line 988) --------------------------------

    def test_step_reward_not_float(self) -> None:
        class BadRewardTypeEnv(MinimalValidEnv):
            def step(self, action: int) -> Any:
                return self._make_obs(), "zero", False, False, {}

        with pytest.raises(ProtocolViolationError, match=r"step.*reward.*expected float"):
            assert_valid_environment(BadRewardTypeEnv())

    # -- step() terminated not bool (line 996) -----------------------------

    def test_step_terminated_not_bool(self) -> None:
        class BadTermEnv(MinimalValidEnv):
            def step(self, action: int) -> Any:
                return self._make_obs(), 0.0, 0, False, {}

        with pytest.raises(ProtocolViolationError, match=r"terminated.*expected bool"):
            assert_valid_environment(BadTermEnv())

    # -- step() truncated not bool (line 1001) -----------------------------

    def test_step_truncated_not_bool(self) -> None:
        class BadTruncEnv(MinimalValidEnv):
            def step(self, action: int) -> Any:
                return self._make_obs(), 0.0, False, 0, {}

        with pytest.raises(ProtocolViolationError, match=r"truncated.*expected bool"):
            assert_valid_environment(BadTruncEnv())

    # -- step() info not dict (line 1006) ----------------------------------

    def test_step_info_not_dict(self) -> None:
        class BadInfoEnv(MinimalValidEnv):
            def step(self, action: int) -> Any:
                return self._make_obs(), 0.0, False, False, "info"

        with pytest.raises(ProtocolViolationError, match=r"step.*info.*expected dict"):
            assert_valid_environment(BadInfoEnv())

    # -- step() raises exception (lines 1011-1012) -------------------------

    def test_step_raises_exception(self) -> None:
        class ExplodingStepEnv(MinimalValidEnv):
            def step(self, action: int) -> Any:
                raise RuntimeError("step broken")

        with pytest.raises(ProtocolViolationError, match=r"step.*raised"):
            assert_valid_environment(ExplodingStepEnv())

    # -- clone() raises exception (lines 1023-1024) ------------------------

    def test_clone_raises_exception(self) -> None:
        class ExplodingCloneEnv(MinimalValidEnv):
            def clone(self) -> Any:
                raise RuntimeError("clone broken")

        with pytest.raises(ProtocolViolationError, match=r"clone.*raised"):
            assert_valid_environment(ExplodingCloneEnv())

    # -- get_symmetries() returns non-list (line 1036) ---------------------

    def test_get_symmetries_returns_non_list(self) -> None:
        class BadSymTypeEnv(MinimalValidEnv):
            def get_symmetries(self, observation: Any, policy: Any) -> Any:
                return "not_a_list"

        with pytest.raises(ProtocolViolationError, match=r"get_symmetries.*returned"):
            assert_valid_environment(BadSymTypeEnv())

    # -- get_symmetries() raises exception (lines 1045-1046) ---------------

    def test_get_symmetries_raises_exception(self) -> None:
        class ExplodingSymEnv(MinimalValidEnv):
            def get_symmetries(self, observation: Any, policy: Any) -> Any:
                raise RuntimeError("symmetries broken")

        with pytest.raises(ProtocolViolationError, match=r"get_symmetries.*raised"):
            assert_valid_environment(ExplodingSymEnv())

    # -- Valid environment passes (sanity) ---------------------------------

    def test_valid_env_passes(self) -> None:
        assert_valid_environment(MinimalValidEnv())

    # -- step() with reward exactly at boundary ----------------------------

    def test_step_reward_negative_one_is_valid(self) -> None:
        class NegOneRewardEnv(MinimalValidEnv):
            def step(self, action: int) -> Any:
                return self._make_obs(), -1.0, False, False, {}

        # Should not raise
        assert_valid_environment(NegOneRewardEnv())

    def test_step_reward_positive_one_is_valid(self) -> None:
        class PosOneRewardEnv(MinimalValidEnv):
            def step(self, action: int) -> Any:
                return self._make_obs(), 1.0, False, False, {}

        assert_valid_environment(PosOneRewardEnv())

    def test_step_reward_below_negative_one(self) -> None:
        class TooNegativeRewardEnv(MinimalValidEnv):
            def step(self, action: int) -> Any:
                return self._make_obs(), -1.5, False, False, {}

        with pytest.raises(ProtocolViolationError, match=r"reward.*outside"):
            assert_valid_environment(TooNegativeRewardEnv())

    # -- Multiple violations accumulated ----------------------------------

    def test_multiple_violations_accumulated(self) -> None:
        """When multiple checks fail after reset, all violations are reported."""

        class MultiFailEnv(MinimalValidEnv):
            def is_terminal(self) -> bool:
                return True  # violation: terminal after reset

            def get_valid_actions(self) -> NDArray[np.bool_]:
                return np.zeros(self._action_size, dtype=np.bool_)  # violation

            def get_observation(self) -> Any:
                return "not_dict"  # violation

        with pytest.raises(ProtocolViolationError) as exc_info:
            assert_valid_environment(MultiFailEnv())
        msg = str(exc_info.value)
        assert "is_terminal" in msg
        assert "get_valid_actions" in msg or "no True" in msg


# =========================================================================
# 3. TestIsinstanceChecks -- runtime_checkable isinstance checks
# =========================================================================


class TestIsinstanceChecks:
    """Test runtime_checkable isinstance checks for all 10 protocols."""

    def test_valid_env_is_strategic_environment(self) -> None:
        assert isinstance(MinimalValidEnv(), StrategicEnvironment)

    def test_missing_method_not_strategic_environment(self) -> None:
        class MissingClone:
            observation_spec = ObservationSpec(spatial_shape=(2, 4, 4), vector_dim=3)
            action_space_size = 5

            def reset(self, *, seed: int | None = None) -> Any:
                pass

            def step(self, action: int) -> Any:
                pass

            def get_valid_actions(self) -> Any:
                pass

            def get_observation(self) -> Any:
                pass

            def is_terminal(self) -> Any:
                pass

            def get_symmetries(self, obs: Any, pol: Any) -> Any:
                pass

        assert not isinstance(MissingClone(), StrategicEnvironment)

    @pytest.mark.parametrize(
        ("protocol", "impl"),
        [
            (
                Serializable,
                type(
                    "S",
                    (),
                    {
                        "to_dict": lambda self: {},
                        "from_dict": classmethod(lambda cls, d: cls()),
                    },
                ),
            ),
            (
                StateEncoder,
                type(
                    "SE",
                    (),
                    {
                        "observation_spec": property(
                            lambda self: ObservationSpec(
                                spatial_shape=(1,), vector_dim=1
                            )
                        ),
                        "encode": lambda self, raw: {},
                        "decode_action": lambda self, idx: {},
                    },
                ),
            ),
            (
                NetworkProtocol,
                type(
                    "NP",
                    (),
                    {
                        "forward": lambda self, s, v, m=None: (None, None),
                        "predict": lambda self, s, v, m=None: (None, None),
                    },
                ),
            ),
            (
                MCTSProtocol,
                type(
                    "MP",
                    (),
                    {
                        "search": lambda self, e, n=None: (None, 0.0),
                        "select_action": lambda self, p, t=1.0, m=None: 0,
                    },
                ),
            ),
            (
                SelectionPolicy,
                type(
                    "SP",
                    (),
                    {
                        "select_child": lambda self, pv, cp, cv, cc, vm: 0,
                    },
                ),
            ),
            (
                RewardShaper,
                type(
                    "RS",
                    (),
                    {
                        "shape_reward": lambda self, r, o, a, i: r,
                    },
                ),
            ),
            (
                CheckpointableProtocol,
                type(
                    "CP",
                    (),
                    {
                        "save": lambda self, p: None,
                        "load": lambda self, p, d="cpu": None,
                    },
                ),
            ),
            (
                TrainerProtocol,
                type(
                    "TP",
                    (),
                    {
                        "train_step": lambda self, b: {},
                        "save_checkpoint": lambda self, p: None,
                        "load_checkpoint": lambda self, p: None,
                    },
                ),
            ),
            (
                StorageBackend,
                type(
                    "SB",
                    (),
                    {
                        "save": lambda self, k, d: None,
                        "load": lambda self, k: b"",
                        "exists": lambda self, k: False,
                        "list_keys": lambda self, p="": [],
                        "delete": lambda self, k: None,
                    },
                ),
            ),
        ],
        ids=[
            "Serializable",
            "StateEncoder",
            "NetworkProtocol",
            "MCTSProtocol",
            "SelectionPolicy",
            "RewardShaper",
            "CheckpointableProtocol",
            "TrainerProtocol",
            "StorageBackend",
        ],
    )
    def test_isinstance_positive_for_all_protocols(
        self, protocol: type, impl: type
    ) -> None:
        assert isinstance(impl(), protocol)

    @pytest.mark.parametrize(
        "protocol",
        [
            Serializable,
            StrategicEnvironment,
            StateEncoder,
            NetworkProtocol,
            MCTSProtocol,
            SelectionPolicy,
            RewardShaper,
            CheckpointableProtocol,
            TrainerProtocol,
            StorageBackend,
        ],
        ids=[
            "Serializable",
            "StrategicEnvironment",
            "StateEncoder",
            "NetworkProtocol",
            "MCTSProtocol",
            "SelectionPolicy",
            "RewardShaper",
            "CheckpointableProtocol",
            "TrainerProtocol",
            "StorageBackend",
        ],
    )
    def test_isinstance_negative_empty_class(self, protocol: type) -> None:
        class Empty:
            pass

        assert not isinstance(Empty(), protocol)

    def test_isinstance_strategic_env_positive(self) -> None:
        assert isinstance(MinimalValidEnv(), StrategicEnvironment)
