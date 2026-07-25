"""Tests for determinized MCTS and world sampling.

Tests cover:
- DeterminizationConfig defaults and validation
- WorldSampler producing valid states
- Sampled worlds consistent with observations
- DeterminizedMCTS combining results
- ObservationModel and ObservationModelConfig
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from alpha_zero.multiplayer.determinization import (
    DeterminizationConfig,
    DeterminizedMCTS,
    WorldSampler,
)
from alpha_zero.multiplayer.fog_of_war import (
    FogOfWarConfig,
    FogOfWarManager,
    VisibilityState,
)
from alpha_zero.multiplayer.observation_model import (
    ObservationModel,
    ObservationModelConfig,
)

# =====================================================================
# DeterminizationConfig tests
# =====================================================================


class TestDeterminizationConfig:
    """Tests for DeterminizationConfig."""

    def test_defaults(self) -> None:
        config = DeterminizationConfig()
        assert config.num_worlds == 8
        assert config.combine_method == "average"

    def test_frozen(self) -> None:
        config = DeterminizationConfig()
        with pytest.raises(AttributeError):
            config.num_worlds = 4  # type: ignore[misc]

    def test_invalid_num_worlds(self) -> None:
        with pytest.raises(ValueError, match="num_worlds"):
            DeterminizationConfig(num_worlds=0)

    def test_invalid_combine_method(self) -> None:
        with pytest.raises(ValueError, match="combine_method"):
            DeterminizationConfig(combine_method="median")

    def test_max_combine_method(self) -> None:
        config = DeterminizationConfig(combine_method="max")
        assert config.combine_method == "max"


# =====================================================================
# WorldSampler tests
# =====================================================================


class TestWorldSampler:
    """Tests for WorldSampler."""

    @pytest.fixture()
    def fog_manager(self) -> FogOfWarManager:
        config = FogOfWarConfig(enabled=True, default_sight_range=2)
        mgr = FogOfWarManager(config, board_size=8, num_players=2)
        return mgr

    @pytest.fixture()
    def sampler(self) -> WorldSampler:
        return WorldSampler(board_size=8, num_players=2, rng=np.random.default_rng(42))

    def _make_terrain(self, board_size: int = 8) -> dict[tuple[int, int], dict]:
        """Create simple plains terrain."""
        return {
            (x, y): {"type": "plains", "yields": {"food": 2, "production": 1}}
            for x in range(board_size)
            for y in range(board_size)
        }

    def test_sample_world_returns_dict(
        self, sampler: WorldSampler, fog_manager: FogOfWarManager
    ) -> None:
        terrain = self._make_terrain()
        units = [{"player": 0, "position": (1, 1), "type": "warrior"}]
        fog_manager.compute_visibility(units, [])

        observed = fog_manager.mask_observation(
            {
                "terrain": terrain,
                "units": units,
                "cities": [],
                "current_player": 0,
                "num_players": 2,
            },
            player=0,
        )

        world = sampler.sample_world(observed, player=0, fog_manager=fog_manager)
        assert isinstance(world, dict)
        assert "units" in world
        assert "cities" in world

    def test_sampled_world_retains_own_units(
        self, sampler: WorldSampler, fog_manager: FogOfWarManager
    ) -> None:
        """Player's own units should always be present in sampled world."""
        terrain = self._make_terrain()
        own_units = [
            {"player": 0, "position": (1, 1), "type": "warrior"},
            {"player": 0, "position": (2, 2), "type": "archer"},
        ]
        fog_manager.compute_visibility(own_units, [])

        observed = fog_manager.mask_observation(
            {
                "terrain": terrain,
                "units": own_units,
                "cities": [],
                "current_player": 0,
                "num_players": 2,
            },
            player=0,
        )

        world = sampler.sample_world(observed, player=0, fog_manager=fog_manager)
        own_in_world = [u for u in world["units"] if u["player"] == 0]
        assert len(own_in_world) == 2

    def test_sampled_world_consistent_with_observations(
        self, sampler: WorldSampler, fog_manager: FogOfWarManager
    ) -> None:
        """Sampled enemy units must be on non-visible tiles."""
        terrain = self._make_terrain()
        own_units = [{"player": 0, "position": (1, 1), "type": "warrior"}]
        fog_manager.compute_visibility(own_units, [])

        observed = fog_manager.mask_observation(
            {
                "terrain": terrain,
                "units": own_units,
                "cities": [],
                "current_player": 0,
                "num_players": 2,
            },
            player=0,
        )

        grid = fog_manager.get_visibility(0)

        for _ in range(10):
            world = sampler.sample_world(observed, player=0, fog_manager=fog_manager)
            enemy_units = [u for u in world["units"] if u["player"] != 0]
            for unit in enemy_units:
                x, y = unit["position"]
                # Enemy units placed by sampler should be on non-visible tiles
                assert grid[y, x] != VisibilityState.VISIBLE, (
                    f"Sampled enemy at ({x},{y}) which is visible to player 0"
                )

    def test_sampled_world_avoids_mountains(
        self, sampler: WorldSampler, fog_manager: FogOfWarManager
    ) -> None:
        """Sampled units should not be placed on mountain tiles."""
        terrain = self._make_terrain()
        # Make most tiles mountains (leave only a few passable)
        for x in range(8):
            for y in range(8):
                if (x, y) != (0, 0) and (x, y) != (7, 7):
                    terrain[(x, y)] = {"type": "mountain"}

        own_units = [{"player": 0, "position": (0, 0), "type": "warrior"}]
        fog_manager.compute_visibility(own_units, [])

        observed = fog_manager.mask_observation(
            {
                "terrain": terrain,
                "units": own_units,
                "cities": [],
                "current_player": 0,
                "num_players": 2,
            },
            player=0,
        )

        world = sampler.sample_world(observed, player=0, fog_manager=fog_manager)
        for unit in world["units"]:
            pos = tuple(unit["position"])
            tile = terrain.get(pos, {})
            assert tile.get("type", "plains") != "mountain", f"Unit placed on mountain at {pos}"

    def test_sampled_world_no_candidates(
        self, sampler: WorldSampler, fog_manager: FogOfWarManager
    ) -> None:
        """When no candidate tiles exist, no hidden units are placed."""
        # Make all tiles visible by using large per-unit sight ranges
        config = FogOfWarConfig(
            enabled=True,
            default_sight_range=100,
            unit_sight_ranges={"warrior": 100, "settler": 100, "archer": 100},
        )
        big_fog = FogOfWarManager(config, board_size=8, num_players=2)

        terrain = self._make_terrain()
        own_units = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        big_fog.compute_visibility(own_units, [])

        observed = big_fog.mask_observation(
            {
                "terrain": terrain,
                "units": own_units,
                "cities": [],
                "current_player": 0,
                "num_players": 2,
            },
            player=0,
        )

        world = sampler.sample_world(observed, player=0, fog_manager=big_fog)
        # Only own units should be present (all tiles visible, no place for hidden)
        assert all(u["player"] == 0 for u in world["units"])


# =====================================================================
# DeterminizedMCTS tests
# =====================================================================


class TestDeterminizedMCTS:
    """Tests for DeterminizedMCTS wrapper."""

    def _make_mock_env(self, num_players: int = 2) -> MagicMock:
        env = MagicMock()
        env.num_players = num_players
        env.current_player = 0
        env.action_space.n = 10
        env.clone.return_value = env

        # Mock units/cities for _inject_world
        env.units = []
        env.cities = []
        return env

    def _make_mock_mcts(self, action_size: int = 10, num_players: int = 2) -> MagicMock:
        """Create a mock MCTS that returns deterministic probabilities."""
        mcts = MagicMock()
        probs = np.ones(action_size) / action_size
        values = dict.fromkeys(range(num_players), 0.5)
        mcts.search.return_value = (probs, values)
        return mcts

    def test_search_returns_correct_shapes(self) -> None:
        config = DeterminizationConfig(num_worlds=3)
        sampler = WorldSampler(board_size=8, num_players=2, rng=np.random.default_rng(0))

        fog_config = FogOfWarConfig(enabled=True, default_sight_range=2)
        fog_mgr = FogOfWarManager(fog_config, board_size=8, num_players=2)

        env = self._make_mock_env()
        mcts = self._make_mock_mcts()

        det = DeterminizedMCTS(mcts, config, sampler)

        game_state: dict[str, Any] = {
            "terrain": {(x, y): {"type": "plains"} for x in range(8) for y in range(8)},
            "units": [{"player": 0, "position": (1, 1), "type": "warrior"}],
            "cities": [],
            "current_player": 0,
            "num_players": 2,
        }
        fog_mgr.compute_visibility(game_state["units"], [])

        probs, values = det.search(env, game_state, player=0, fog_manager=fog_mgr)

        assert probs.shape == (10,)
        assert len(values) == 2
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_search_combines_multiple_worlds(self) -> None:
        """Verify multiple worlds are searched (MCTS called num_worlds times)."""
        config = DeterminizationConfig(num_worlds=5)
        sampler = WorldSampler(board_size=8, num_players=2, rng=np.random.default_rng(0))

        fog_config = FogOfWarConfig(enabled=True, default_sight_range=2)
        fog_mgr = FogOfWarManager(fog_config, board_size=8, num_players=2)

        env = self._make_mock_env()
        mcts = self._make_mock_mcts()

        det = DeterminizedMCTS(mcts, config, sampler)

        game_state: dict[str, Any] = {
            "terrain": {(x, y): {"type": "plains"} for x in range(8) for y in range(8)},
            "units": [{"player": 0, "position": (1, 1), "type": "warrior"}],
            "cities": [],
            "current_player": 0,
            "num_players": 2,
        }
        fog_mgr.compute_visibility(game_state["units"], [])

        det.search(env, game_state, player=0, fog_manager=fog_mgr)

        assert mcts.search.call_count == 5

    def test_max_combine_method(self) -> None:
        """Max combine method takes element-wise max."""
        config = DeterminizationConfig(num_worlds=2, combine_method="max")
        sampler = WorldSampler(board_size=8, num_players=2, rng=np.random.default_rng(0))

        fog_config = FogOfWarConfig(enabled=True, default_sight_range=2)
        fog_mgr = FogOfWarManager(fog_config, board_size=8, num_players=2)

        env = self._make_mock_env()
        mcts = MagicMock()

        # Return different distributions per call
        probs1 = np.zeros(10)
        probs1[0] = 0.8
        probs1[1] = 0.2
        probs2 = np.zeros(10)
        probs2[1] = 0.7
        probs2[2] = 0.3

        values = {0: 0.5, 1: -0.5}
        mcts.search.side_effect = [(probs1, values), (probs2, values)]

        det = DeterminizedMCTS(mcts, config, sampler)

        game_state: dict[str, Any] = {
            "terrain": {(x, y): {"type": "plains"} for x in range(8) for y in range(8)},
            "units": [{"player": 0, "position": (1, 1), "type": "warrior"}],
            "cities": [],
            "current_player": 0,
            "num_players": 2,
        }
        fog_mgr.compute_visibility(game_state["units"], [])

        probs, _ = det.search(env, game_state, player=0, fog_manager=fog_mgr)

        # Max of (0.8, 0) = 0.8, max of (0.2, 0.7) = 0.7, max of (0, 0.3) = 0.3
        assert probs[0] > 0
        assert probs[1] > 0
        assert probs[2] > 0

    def test_fallback_on_search_failure(self) -> None:
        """If all MCTS searches fail, return uniform distribution."""
        config = DeterminizationConfig(num_worlds=2)
        sampler = WorldSampler(board_size=8, num_players=2, rng=np.random.default_rng(0))

        fog_config = FogOfWarConfig(enabled=True, default_sight_range=2)
        fog_mgr = FogOfWarManager(fog_config, board_size=8, num_players=2)

        env = self._make_mock_env()
        mcts = MagicMock()
        mcts.search.side_effect = RuntimeError("MCTS failed")

        det = DeterminizedMCTS(mcts, config, sampler)

        game_state: dict[str, Any] = {
            "terrain": {(x, y): {"type": "plains"} for x in range(8) for y in range(8)},
            "units": [{"player": 0, "position": (1, 1), "type": "warrior"}],
            "cities": [],
            "current_player": 0,
            "num_players": 2,
        }
        fog_mgr.compute_visibility(game_state["units"], [])

        probs, values = det.search(env, game_state, player=0, fog_manager=fog_mgr)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert all(v == 0.0 for v in values.values())


# =====================================================================
# ObservationModelConfig tests
# =====================================================================


class TestObservationModelConfig:
    """Tests for ObservationModelConfig."""

    def test_defaults(self) -> None:
        config = ObservationModelConfig()
        assert config.method == "uniform"
        assert len(config.default_unit_types) == 3
        assert len(config.default_unit_weights) == 3

    def test_frozen(self) -> None:
        config = ObservationModelConfig()
        with pytest.raises(AttributeError):
            config.method = "learned"  # type: ignore[misc]

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="unit_types length"):
            ObservationModelConfig(
                default_unit_types=("warrior",),
                default_unit_weights=(0.5, 0.5),
            )

    def test_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown observation model method"):
            ObservationModelConfig(method="neural")


# =====================================================================
# ObservationModel tests
# =====================================================================


class TestObservationModel:
    """Tests for ObservationModel."""

    @pytest.fixture()
    def model(self) -> ObservationModel:
        return ObservationModel(
            board_size=8,
            num_players=2,
            rng=np.random.default_rng(42),
        )

    def test_predict_hidden_units_count(self, model: ObservationModel) -> None:
        candidates = [(x, y) for x in range(3, 8) for y in range(3, 8)]
        units = model.predict_hidden_units(
            visible_state={"num_players": 2},
            player=0,
            num_hidden_units=3,
            candidate_tiles=candidates,
        )
        assert len(units) == 3

    def test_predict_hidden_units_zero(self, model: ObservationModel) -> None:
        units = model.predict_hidden_units(
            visible_state={"num_players": 2},
            player=0,
            num_hidden_units=0,
            candidate_tiles=[(1, 1)],
        )
        assert len(units) == 0

    def test_predict_hidden_units_no_candidates(self, model: ObservationModel) -> None:
        units = model.predict_hidden_units(
            visible_state={"num_players": 2},
            player=0,
            num_hidden_units=5,
            candidate_tiles=[],
        )
        assert len(units) == 0

    def test_predict_hidden_units_positions_unique(self, model: ObservationModel) -> None:
        candidates = [(x, y) for x in range(5) for y in range(5)]
        units = model.predict_hidden_units(
            visible_state={"num_players": 2},
            player=0,
            num_hidden_units=10,
            candidate_tiles=candidates,
        )
        positions = [tuple(u["position"]) for u in units]
        assert len(positions) == len(set(positions)), "Positions should be unique"

    def test_predict_hidden_units_are_opponents(self, model: ObservationModel) -> None:
        candidates = [(x, y) for x in range(3, 8) for y in range(3, 8)]
        units = model.predict_hidden_units(
            visible_state={"num_players": 3},
            player=0,
            num_hidden_units=5,
            candidate_tiles=candidates,
        )
        for unit in units:
            assert unit["player"] != 0, "Hidden units should be opponents"
            assert unit["player"] in (1, 2)

    def test_predict_hidden_units_valid_types(self, model: ObservationModel) -> None:
        candidates = [(x, y) for x in range(3, 8) for y in range(3, 8)]
        units = model.predict_hidden_units(
            visible_state={"num_players": 2},
            player=0,
            num_hidden_units=5,
            candidate_tiles=candidates,
        )
        for unit in units:
            assert unit["type"] in ("warrior", "archer", "settler")

    def test_reset_is_noop(self, model: ObservationModel) -> None:
        # Just make sure reset() doesn't error
        model.reset()

    def test_more_requested_than_candidates(self, model: ObservationModel) -> None:
        """When requesting more hidden units than candidate tiles, cap at available."""
        candidates = [(0, 0), (1, 1)]
        units = model.predict_hidden_units(
            visible_state={"num_players": 2},
            player=0,
            num_hidden_units=100,
            candidate_tiles=candidates,
        )
        assert len(units) == 2
