"""Unit tests for action pruning module.

Tests verify that the action pruning module can be imported and
basic classes exist. Full tests require torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# Skip all tests if torch not available (needed by alpha_zero.core)
torch = pytest.importorskip("torch", reason="torch required for action pruning tests")


# =============================================================================
# Import Tests
# =============================================================================


class TestActionPruningImports:
    """Tests for action pruning module imports."""

    def test_import_pruning_config(self):
        """Test PrunerConfig can be imported."""
        from alpha_zero.core.action_pruning import PrunerConfig

        assert PrunerConfig is not None

    def test_import_pruning_level(self):
        """Test PruningLevel can be imported."""
        from alpha_zero.core.action_pruning import PruningLevel

        assert PruningLevel is not None

    def test_import_action_pruner(self):
        """Test ActionPruner can be imported."""
        from alpha_zero.core.action_pruning import ActionPruner

        assert ActionPruner is not None

    def test_import_macro_action(self):
        """Test MacroAction can be imported."""
        from alpha_zero.core.action_pruning import MacroAction, MacroActionType

        assert MacroAction is not None
        assert MacroActionType is not None

    def test_import_macro_planner(self):
        """Test MacroActionPlanner can be imported."""
        from alpha_zero.core.action_pruning import MacroActionPlanner

        assert MacroActionPlanner is not None


# =============================================================================
# Configuration Tests
# =============================================================================


class TestPrunerConfig:
    """Tests for PrunerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from alpha_zero.core.action_pruning import PrunerConfig

        config = PrunerConfig()
        assert config.min_pruned_actions >= 0
        assert config.enable_feasibility is True

    def test_pruning_levels_exist(self):
        """Test pruning levels are defined."""
        from alpha_zero.core.action_pruning import PruningLevel

        levels = list(PruningLevel)
        assert len(levels) >= 3  # At least NONE, STANDARD, EXTREME


# =============================================================================
# Action Pruner Tests
# =============================================================================


class TestActionPruner:
    """Tests for ActionPruner class."""

    @pytest.fixture
    def sample_state(self) -> dict[str, Any]:
        """Create sample game state."""
        return {
            "current_player": 0,
            "turn": 10,
            "units": [{"id": "unit_0", "player": 0, "position": (5, 5)}],
            "cities": [{"id": "city_0", "player": 0, "position": (4, 4)}],
            "resources": {"gold": 100},
        }

    @pytest.fixture
    def sample_mask(self) -> np.ndarray:
        """Create sample action mask."""
        mask = np.zeros(500, dtype=np.float32)
        mask[::2] = 1.0
        return mask

    def test_pruner_creation(self):
        """Test ActionPruner can be created."""
        from alpha_zero.core.action_pruning import ActionPruner, PrunerConfig

        config = PrunerConfig()
        pruner = ActionPruner(config)
        assert pruner is not None

    def test_prune_returns_result(self, sample_state, sample_mask):
        """Test prune method returns a result."""
        from alpha_zero.core.action_pruning import ActionPruner, PrunedActions, PrunerConfig

        config = PrunerConfig()
        pruner = ActionPruner(config)
        result = pruner.prune(sample_state, sample_mask)
        assert isinstance(result, PrunedActions)
        assert result.mask.shape == sample_mask.shape

    def test_prune_reduces_actions(self, sample_state, sample_mask):
        """Test pruning can reduce action count."""
        from alpha_zero.core.action_pruning import ActionPruner, PrunerConfig, PruningLevel

        config = PrunerConfig(pruning_level=PruningLevel.AGGRESSIVE)
        pruner = ActionPruner(config)
        result = pruner.prune(sample_state, sample_mask)
        assert result.stats.final_actions <= sample_mask.sum()


# =============================================================================
# Macro Action Tests
# =============================================================================


class TestMacroAction:
    """Tests for MacroAction class."""

    def test_macro_creation(self):
        """Test MacroAction can be created."""
        from alpha_zero.core.action_pruning import MacroAction, MacroActionType

        macro = MacroAction(
            macro_type=MacroActionType.EXPAND_TO_RESOURCE,
            goal="test_goal",
            steps=[1, 2, 3],
            priority=0.5,
        )
        assert macro.goal == "test_goal"
        assert macro.macro_type == MacroActionType.EXPAND_TO_RESOURCE
        assert len(macro.steps) == 3

    def test_macro_types_exist(self):
        """Test macro action types are defined."""
        from alpha_zero.core.action_pruning import MacroActionType

        types = list(MacroActionType)
        assert MacroActionType.EXPAND_TO_RESOURCE in types
        assert MacroActionType.CONQUER_CITY in types
        assert MacroActionType.DEFEND_POSITION in types


# =============================================================================
# Macro Planner Tests
# =============================================================================


class TestMacroActionPlanner:
    """Tests for MacroActionPlanner class."""

    @pytest.fixture
    def sample_state(self) -> dict[str, Any]:
        """Create sample game state."""
        return {
            "current_player": 0,
            "units": [{"type": "settler", "player": 0}],
            "cities": [{"player": 0}],
        }

    def test_planner_creation(self):
        """Test MacroActionPlanner can be created."""
        from alpha_zero.core.action_pruning import MacroActionConfig, MacroActionPlanner

        config = MacroActionConfig()
        planner = MacroActionPlanner(config)
        assert planner is not None

    def test_get_suggested_macros(self, sample_state):
        """Test getting available macro actions."""
        from alpha_zero.core.action_pruning import MacroActionConfig, MacroActionPlanner

        config = MacroActionConfig()
        planner = MacroActionPlanner(config)
        macros = planner.get_suggested_macros(sample_state)
        assert isinstance(macros, list)
