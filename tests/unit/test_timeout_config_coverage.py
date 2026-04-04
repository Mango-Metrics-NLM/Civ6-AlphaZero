"""Comprehensive tests for alpha_zero.utils.timeout_config module."""

from __future__ import annotations

from unittest.mock import patch

import alpha_zero.utils.timeout_config as tc_module
from alpha_zero.utils.timeout_config import TimeoutConfig, get_timeout_config

# ---------------------------------------------------------------------------
# TimeoutConfig defaults
# ---------------------------------------------------------------------------


class TestTimeoutConfigDefaults:
    def test_default_values(self):
        cfg = TimeoutConfig()
        assert cfg.trajectory_submission == 5.0
        assert cfg.process_join == 1.0
        assert cfg.actor_pool_stop == 10.0
        assert cfg.state_reading == 60.0
        assert cfg.action_execution == 30.0
        assert cfg.game_action == 5.0
        assert cfg.game_state == 2.0
        assert cfg.agent_planning == 300.0
        assert cfg.agent_implementation == 600.0
        assert cfg.agent_testing == 600.0
        assert cfg.agent_verification == 300.0
        assert cfg.agent_documentation == 300.0
        assert cfg.exploration_timeout_minutes == 10.0


# ---------------------------------------------------------------------------
# TimeoutConfig.from_env
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_from_env_defaults(self):
        cfg = TimeoutConfig.from_env()
        assert cfg.trajectory_submission == 5.0

    @patch.dict(
        "os.environ",
        {"CIV6AZ_TIMEOUT_TRAJECTORY_SUBMISSION": "20.0"},
    )
    def test_from_env_overrides_single_field(self):
        cfg = TimeoutConfig.from_env()
        assert cfg.trajectory_submission == 20.0
        # Other fields keep defaults
        assert cfg.process_join == 1.0

    @patch.dict(
        "os.environ",
        {
            "CIV6AZ_TIMEOUT_PROCESS_JOIN": "3.5",
            "CIV6AZ_TIMEOUT_ACTOR_POOL_STOP": "25.0",
            "CIV6AZ_TIMEOUT_STATE_READING": "120.0",
            "CIV6AZ_TIMEOUT_ACTION_EXECUTION": "60.0",
            "CIV6AZ_TIMEOUT_GAME_ACTION": "10.0",
            "CIV6AZ_TIMEOUT_GAME_STATE": "4.0",
            "CIV6AZ_TIMEOUT_AGENT_PLANNING": "500.0",
            "CIV6AZ_TIMEOUT_AGENT_IMPLEMENTATION": "900.0",
            "CIV6AZ_TIMEOUT_AGENT_TESTING": "800.0",
            "CIV6AZ_TIMEOUT_AGENT_VERIFICATION": "400.0",
            "CIV6AZ_TIMEOUT_AGENT_DOCUMENTATION": "350.0",
            "CIV6AZ_TIMEOUT_EXPLORATION_TIMEOUT_MINUTES": "15.0",
        },
    )
    def test_from_env_all_fields(self):
        cfg = TimeoutConfig.from_env()
        assert cfg.process_join == 3.5
        assert cfg.actor_pool_stop == 25.0
        assert cfg.state_reading == 120.0
        assert cfg.action_execution == 60.0
        assert cfg.game_action == 10.0
        assert cfg.game_state == 4.0
        assert cfg.agent_planning == 500.0
        assert cfg.agent_implementation == 900.0
        assert cfg.agent_testing == 800.0
        assert cfg.agent_verification == 400.0
        assert cfg.agent_documentation == 350.0
        assert cfg.exploration_timeout_minutes == 15.0

    @patch.dict(
        "os.environ",
        {"CIV6AZ_TIMEOUT_TRAJECTORY_SUBMISSION": "not_a_number"},
    )
    def test_from_env_invalid_value_keeps_default(self):
        cfg = TimeoutConfig.from_env()
        assert cfg.trajectory_submission == 5.0

    def test_from_env_custom_prefix(self):
        with patch.dict("os.environ", {"MY_PREFIX_PROCESS_JOIN": "99.0"}):
            cfg = TimeoutConfig.from_env(prefix="MY_PREFIX_")
            assert cfg.process_join == 99.0


# ---------------------------------------------------------------------------
# TimeoutConfig.from_dict
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_from_dict_partial(self):
        cfg = TimeoutConfig.from_dict({"trajectory_submission": 10})
        assert cfg.trajectory_submission == 10.0
        assert cfg.process_join == 1.0  # default

    def test_from_dict_all_fields(self):
        data = {
            "trajectory_submission": 1.0,
            "process_join": 2.0,
            "actor_pool_stop": 3.0,
            "state_reading": 4.0,
            "action_execution": 5.0,
            "game_action": 6.0,
            "game_state": 7.0,
            "agent_planning": 8.0,
            "agent_implementation": 9.0,
            "agent_testing": 10.0,
            "agent_verification": 11.0,
            "agent_documentation": 12.0,
            "exploration_timeout_minutes": 13.0,
        }
        cfg = TimeoutConfig.from_dict(data)
        assert cfg.trajectory_submission == 1.0
        assert cfg.exploration_timeout_minutes == 13.0

    def test_from_dict_ignores_unknown_keys(self):
        cfg = TimeoutConfig.from_dict({"unknown_field": 999, "process_join": 7.0})
        assert cfg.process_join == 7.0

    def test_from_dict_empty(self):
        cfg = TimeoutConfig.from_dict({})
        assert cfg.trajectory_submission == 5.0

    def test_from_dict_converts_int_to_float(self):
        cfg = TimeoutConfig.from_dict({"game_action": 10})
        assert cfg.game_action == 10.0
        assert isinstance(cfg.game_action, float)


# ---------------------------------------------------------------------------
# TimeoutConfig.to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    def test_to_dict_roundtrip(self):
        original = TimeoutConfig(
            trajectory_submission=1.5,
            process_join=2.5,
            actor_pool_stop=3.5,
        )
        d = original.to_dict()
        restored = TimeoutConfig.from_dict(d)
        assert restored.to_dict() == d

    def test_to_dict_contains_all_fields(self):
        cfg = TimeoutConfig()
        d = cfg.to_dict()
        expected_keys = {
            "trajectory_submission",
            "process_join",
            "actor_pool_stop",
            "state_reading",
            "action_execution",
            "game_action",
            "game_state",
            "agent_planning",
            "agent_implementation",
            "agent_testing",
            "agent_verification",
            "agent_documentation",
            "exploration_timeout_minutes",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_are_floats(self):
        d = TimeoutConfig().to_dict()
        for v in d.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# get_timeout_config singleton
# ---------------------------------------------------------------------------


class TestGetTimeoutConfig:
    def test_returns_instance(self):
        # Reset the module-level singleton
        tc_module._default_config = None
        cfg = get_timeout_config()
        assert isinstance(cfg, TimeoutConfig)

    def test_returns_same_instance(self):
        tc_module._default_config = None
        cfg1 = get_timeout_config()
        cfg2 = get_timeout_config()
        assert cfg1 is cfg2

    @patch.dict("os.environ", {"CIV6AZ_TIMEOUT_GAME_STATE": "42.0"})
    def test_singleton_reads_env_on_first_call(self):
        tc_module._default_config = None
        cfg = get_timeout_config()
        assert cfg.game_state == 42.0

    def test_reset_forces_reinit(self):
        tc_module._default_config = None
        cfg1 = get_timeout_config()
        tc_module._default_config = None
        cfg2 = get_timeout_config()
        # They are different objects after reset
        assert cfg1 is not cfg2
