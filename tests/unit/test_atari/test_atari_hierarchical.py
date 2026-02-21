"""Unit tests for the Atari hierarchical reasoning adapter."""

from __future__ import annotations

import numpy as np
import pytest

from demos.atari.config import AtariHierarchicalConfig
from demos.atari.hierarchical import (
    AtariGoal,
    AtariHierarchicalAdapter,
    AtariOperationalGoal,
    AtariStrategicGoal,
    AtariTacticalGoal,
)


@pytest.fixture
def enabled_config():
    return AtariHierarchicalConfig(
        enabled=True,
        strategic_update_freq=10,
        tactical_update_freq=3,
        max_active_goals_per_level=3,
    )


@pytest.fixture
def disabled_config():
    return AtariHierarchicalConfig(enabled=False)


@pytest.fixture
def dummy_obs():
    return {
        "spatial": np.random.rand(4, 84, 84).astype(np.float32),
        "vector": np.zeros(32, dtype=np.float32),
        "action_mask": np.ones(6, dtype=np.float32),
    }


# ============================================================================
# Goal enums
# ============================================================================


class TestGoalEnums:
    def test_strategic_goals_exist(self):
        assert len(AtariStrategicGoal) >= 5
        assert AtariStrategicGoal.MAXIMIZE_SCORE is not None
        assert AtariStrategicGoal.SURVIVE is not None
        assert AtariStrategicGoal.AGGRESSIVE_PLAY is not None

    def test_tactical_goals_exist(self):
        assert len(AtariTacticalGoal) >= 6
        assert AtariTacticalGoal.CLEAR_AREA is not None
        assert AtariTacticalGoal.AVOID_HAZARD is not None

    def test_operational_goals_exist(self):
        assert len(AtariOperationalGoal) >= 4
        assert AtariOperationalGoal.MOVE_LEFT is not None
        assert AtariOperationalGoal.FIRE is not None


class TestAtariGoal:
    def test_creation(self):
        goal = AtariGoal("strategic", AtariStrategicGoal.MAXIMIZE_SCORE)
        assert goal.level == "strategic"
        assert goal.progress == 0.0
        assert goal.active is True
        assert goal.created_step == 0

    def test_progress_update(self):
        goal = AtariGoal("tactical", AtariTacticalGoal.CLEAR_AREA)
        goal.progress = 0.5
        assert goal.progress == 0.5


# ============================================================================
# AtariHierarchicalAdapter
# ============================================================================


class TestAtariHierarchicalAdapter:
    def test_disabled_passthrough(self, disabled_config, dummy_obs):
        adapter = AtariHierarchicalAdapter(disabled_config, num_actions=6)
        base_policy = np.ones(6, dtype=np.float32) / 6
        result = adapter.decide(dummy_obs, base_policy=base_policy)
        assert result["policy"] is base_policy  # unchanged
        assert result["strategic_goals"] == []
        assert result["tactical_goals"] == []
        assert result["reward_shaping"] == 0.0

    def test_enabled_produces_goals(self, enabled_config, dummy_obs):
        adapter = AtariHierarchicalAdapter(enabled_config, num_actions=6)
        # Simulate enough steps to trigger strategic + tactical updates
        for _ in range(enabled_config.strategic_update_freq + 1):
            result = adapter.decide(dummy_obs)
            adapter.update(1.0, False)
        assert len(result["strategic_goals"]) > 0

    def test_strategic_update_timing(self, enabled_config, dummy_obs):
        adapter = AtariHierarchicalAdapter(enabled_config, num_actions=6)
        # First call should trigger strategic update (step 1, last_update=0)
        result = adapter.decide(dummy_obs)
        # After 1 step, should not have updated yet (need 10 steps)
        assert adapter._strategic.last_update_step == 0  # not enough steps yet

        # Advance to trigger threshold
        for i in range(enabled_config.strategic_update_freq):
            adapter.decide(dummy_obs)
            adapter.update(0.0, False)

        assert adapter._strategic.last_update_step > 0

    def test_tactical_decomposition(self, enabled_config, dummy_obs):
        adapter = AtariHierarchicalAdapter(enabled_config, num_actions=6)
        # Force strategic goals
        adapter._strategic.active_goals = [
            AtariGoal("strategic", AtariStrategicGoal.AGGRESSIVE_PLAY)
        ]
        # Trigger tactical update
        adapter._tactical.last_update_step = -999
        adapter._step = 100
        adapter.decide(dummy_obs)
        assert len(adapter._tactical.active_goals) > 0

    def test_reset(self, enabled_config, dummy_obs):
        adapter = AtariHierarchicalAdapter(enabled_config, num_actions=6)
        for _ in range(20):
            adapter.decide(dummy_obs)
            adapter.update(1.0, False)
        adapter.reset()
        assert adapter._step == 0
        assert adapter._episode_reward == 0.0
        assert len(adapter._strategic.active_goals) == 0
        assert len(adapter._tactical.active_goals) == 0

    def test_update_on_done_resets(self, enabled_config, dummy_obs):
        adapter = AtariHierarchicalAdapter(enabled_config, num_actions=6)
        adapter.decide(dummy_obs)
        adapter.update(1.0, done=True)
        assert adapter._step == 0

    def test_get_active_goals(self, enabled_config):
        adapter = AtariHierarchicalAdapter(enabled_config, num_actions=6)
        adapter._strategic.active_goals = [
            AtariGoal("strategic", AtariStrategicGoal.MAXIMIZE_SCORE, active=True),
            AtariGoal("strategic", AtariStrategicGoal.SURVIVE, active=False),
        ]
        goals = adapter.get_active_goals()
        assert "MAXIMIZE_SCORE" in goals["strategic"]
        assert "SURVIVE" not in goals["strategic"]

    def test_goal_progress_on_positive_reward(self, enabled_config, dummy_obs):
        adapter = AtariHierarchicalAdapter(enabled_config, num_actions=6)
        adapter._tactical.active_goals = [
            AtariGoal("tactical", AtariTacticalGoal.CLEAR_AREA, progress=0.0, active=True)
        ]
        adapter.update(1.0, False)
        assert adapter._tactical.active_goals[0].progress > 0.0

    def test_goal_completion(self, enabled_config, dummy_obs):
        adapter = AtariHierarchicalAdapter(enabled_config, num_actions=6)
        goal = AtariGoal("tactical", AtariTacticalGoal.CLEAR_AREA, progress=0.99, active=True)
        adapter._tactical.active_goals = [goal]
        adapter.update(1.0, False)
        assert goal.active is False  # should be deactivated

    def test_reward_shaping(self, enabled_config, dummy_obs):
        adapter = AtariHierarchicalAdapter(enabled_config, num_actions=6)
        adapter._tactical.active_goals = [
            AtariGoal("tactical", AtariTacticalGoal.CLEAR_AREA, progress=0.5, active=True)
        ]
        result = adapter.decide(dummy_obs)
        assert result["reward_shaping"] != 0.0

    def test_max_goals_per_level(self, enabled_config, dummy_obs):
        adapter = AtariHierarchicalAdapter(enabled_config, num_actions=6)
        # Force many strategic goals
        adapter._strategic.active_goals = [
            AtariGoal("strategic", AtariStrategicGoal.AGGRESSIVE_PLAY)
        ] * 10
        adapter._tactical.last_update_step = -999
        adapter._step = 100
        adapter.decide(dummy_obs)
        assert len(adapter._tactical.active_goals) <= enabled_config.max_active_goals_per_level
