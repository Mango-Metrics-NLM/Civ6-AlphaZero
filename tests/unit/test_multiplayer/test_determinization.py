"""Tests for determinized MCTS with fog of war."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from alpha_zero.multiplayer.determinization import (
    DeterminizationConfig,
    DeterminizedMCTS,
    WorldSampler,
)
from alpha_zero.multiplayer.fog_of_war import FogOfWarManager, VisibilityState

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_visibility_grid(
    board_size: int,
    visible_tiles: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Create a visibility grid with specified visible tiles.

    All tiles default to UNEXPLORED. Tiles in *visible_tiles* are set to
    VISIBLE.
    """
    grid = np.full(
        (board_size, board_size), VisibilityState.UNEXPLORED, dtype=np.int8
    )
    for x, y in (visible_tiles or []):
        grid[y, x] = VisibilityState.VISIBLE
    return grid


def _make_fog_manager(
    board_size: int,
    visible_tiles: list[tuple[int, int]] | None = None,
) -> MagicMock:
    """Build a mock FogOfWarManager that returns a fixed visibility grid."""
    grid = _make_visibility_grid(board_size, visible_tiles)
    mock = MagicMock(spec=FogOfWarManager)
    mock.get_visibility.return_value = grid
    mock.mask_observation.side_effect = lambda gs, p: copy.deepcopy(gs)
    return mock


def _make_env(
    action_size: int = 10,
    num_players: int = 2,
    *,
    with_units_cities: bool = False,
) -> MagicMock:
    """Build a mock environment for DeterminizedMCTS tests."""
    env = MagicMock()
    env.action_space = SimpleNamespace(n=action_size)
    env.num_players = num_players
    env.clone.return_value = MagicMock()
    env.clone.return_value.action_space = SimpleNamespace(n=action_size)
    env.clone.return_value.num_players = num_players
    if with_units_cities:
        env.units = []
        env.cities = []
        env.clone.return_value.units = []
        env.clone.return_value.cities = []
    return env


def _make_base_mcts(
    action_size: int = 10,
    num_players: int = 2,
    *,
    fail: bool = False,
) -> MagicMock:
    """Build a mock base MCTS that returns uniform probs and zero values."""
    mock = MagicMock()
    if fail:
        mock.search.side_effect = RuntimeError("MCTS exploded")
    else:
        probs = np.ones(action_size) / action_size
        values = dict.fromkeys(range(num_players), 0.0)
        mock.search.return_value = (probs, values)
    return mock


# ---------------------------------------------------------------------------
# DeterminizationConfig tests
# ---------------------------------------------------------------------------


class TestDeterminizationConfig:
    """Tests for DeterminizationConfig dataclass."""

    def test_defaults(self):
        """Default config has 8 worlds and 'average' combining."""
        cfg = DeterminizationConfig()
        assert cfg.num_worlds == 8
        assert cfg.combine_method == "average"

    def test_custom_values(self):
        """Custom values are accepted."""
        cfg = DeterminizationConfig(num_worlds=16, combine_method="max")
        assert cfg.num_worlds == 16
        assert cfg.combine_method == "max"

    def test_num_worlds_one_is_valid(self):
        """Boundary: num_worlds=1 is the minimum valid value."""
        cfg = DeterminizationConfig(num_worlds=1)
        assert cfg.num_worlds == 1

    def test_num_worlds_zero_raises(self):
        """num_worlds=0 must raise ValueError."""
        with pytest.raises(ValueError, match="num_worlds must be >= 1"):
            DeterminizationConfig(num_worlds=0)

    def test_num_worlds_negative_raises(self):
        """Negative num_worlds must raise ValueError."""
        with pytest.raises(ValueError, match="num_worlds must be >= 1"):
            DeterminizationConfig(num_worlds=-5)

    @pytest.mark.parametrize("method", ["min", "sum", "median", "", "Average"])
    def test_invalid_combine_method_raises(self, method: str):
        """Invalid combine_method values must raise ValueError."""
        with pytest.raises(ValueError, match="combine_method must be"):
            DeterminizationConfig(combine_method=method)

    def test_frozen(self):
        """Config is frozen; attribute assignment must raise."""
        cfg = DeterminizationConfig()
        with pytest.raises(AttributeError):
            cfg.num_worlds = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# WorldSampler tests
# ---------------------------------------------------------------------------


class TestWorldSampler:
    """Tests for WorldSampler."""

    @pytest.fixture
    def board_size(self) -> int:
        return 4

    @pytest.fixture
    def sampler(self, board_size: int) -> WorldSampler:
        return WorldSampler(board_size=board_size, num_players=2)

    def test_construction(self, board_size: int):
        """WorldSampler stores board_size and num_players."""
        ws = WorldSampler(board_size=board_size, num_players=3)
        assert ws.board_size == board_size
        assert ws.num_players == 3

    def test_construction_with_rng(self, board_size: int):
        """Custom rng is stored."""
        rng = np.random.default_rng(42)
        ws = WorldSampler(board_size=board_size, num_players=2, rng=rng)
        assert ws._rng is rng

    def test_sample_world_returns_deep_copy(self, sampler: WorldSampler, board_size: int):
        """Returned world is a deep copy; modifying it does not affect input."""
        fog = _make_fog_manager(board_size, visible_tiles=[(0, 0)])
        observed = {"units": [], "terrain": {}}
        world = sampler.sample_world(observed, player=0, fog_manager=fog)
        world["extra_key"] = True
        assert "extra_key" not in observed

    def test_sample_world_no_units_no_crash(self, sampler: WorldSampler, board_size: int):
        """With no units at all, sample_world should still return a state."""
        fog = _make_fog_manager(board_size, visible_tiles=[(0, 0)])
        observed = {"units": [], "terrain": {}}
        world = sampler.sample_world(observed, player=0, fog_manager=fog)
        assert isinstance(world, dict)
        # No own units -> own_unit_count = 0 -> estimated_hidden = 0 -> no units placed
        assert world.get("units") == []

    def test_sample_world_places_hidden_units(self, board_size: int):
        """When player has units, hidden opponent units are placed on non-visible tiles."""
        rng = np.random.default_rng(123)
        sampler = WorldSampler(board_size=board_size, num_players=2, rng=rng)

        # Make only tile (0,0) visible
        fog = _make_fog_manager(board_size, visible_tiles=[(0, 0)])

        observed = {
            "units": [
                {"id": 1, "player": 0, "type": "warrior", "position": (0, 0)},
            ],
            "terrain": {},
        }

        world = sampler.sample_world(observed, player=0, fog_manager=fog)

        # Opponent (player 1) should have units placed
        opponent_units = [u for u in world["units"] if u["player"] == 1]
        assert len(opponent_units) >= 1

        # All placed units should be on non-visible tiles (not (0,0))
        for u in opponent_units:
            assert u["position"] != (0, 0)
            assert u["id"] == -1  # synthetic
            assert u["health"] == 100

    def test_sample_world_respects_mountain_filtering(self, board_size: int):
        """Mountains should not be candidates for unit placement."""
        rng = np.random.default_rng(0)
        sampler = WorldSampler(board_size=board_size, num_players=2, rng=rng)

        # Only (0,0) visible; all non-visible tiles are mountains except (1,1)
        fog = _make_fog_manager(board_size, visible_tiles=[(0, 0)])

        terrain: dict[tuple[int, int], dict[str, str]] = {}
        for y in range(board_size):
            for x in range(board_size):
                if (x, y) != (0, 0) and (x, y) != (1, 1):
                    terrain[(x, y)] = {"type": "mountain"}

        observed = {
            "units": [
                {"id": 1, "player": 0, "type": "warrior", "position": (0, 0)},
            ],
            "terrain": terrain,
        }

        world = sampler.sample_world(observed, player=0, fog_manager=fog)

        # The only non-mountain, non-visible candidate is (1,1)
        opponent_units = [u for u in world["units"] if u["player"] == 1]
        for u in opponent_units:
            assert u["position"] == (1, 1), (
                f"Unit placed on mountain or visible tile: {u['position']}"
            )

    def test_sample_world_all_tiles_visible_returns_unchanged(
        self, board_size: int
    ):
        """When all tiles are visible, no hidden units are placed."""
        sampler = WorldSampler(board_size=board_size, num_players=2)

        all_tiles = [(x, y) for y in range(board_size) for x in range(board_size)]
        fog = _make_fog_manager(board_size, visible_tiles=all_tiles)

        observed = {
            "units": [
                {"id": 1, "player": 0, "type": "warrior", "position": (0, 0)},
            ],
            "terrain": {},
        }

        world = sampler.sample_world(observed, player=0, fog_manager=fog)
        opponent_units = [u for u in world["units"] if u["player"] == 1]
        assert len(opponent_units) == 0

    def test_sample_world_empty_candidates_after_own_position_filter(
        self, board_size: int
    ):
        """If all candidate tiles are occupied by own units, return early."""
        # Board is 1x1 so only tile is (0,0) which is both visible -> no
        # candidates initially. But let's use board_size=2 with 3 visible and
        # 1 candidate occupied by own unit.
        bs = 2
        sampler = WorldSampler(board_size=bs, num_players=2)
        fog = _make_fog_manager(bs, visible_tiles=[(0, 0), (1, 0), (0, 1)])
        # Only non-visible tile is (1,1), but own unit is there
        observed = {
            "units": [
                {"id": 1, "player": 0, "type": "warrior", "position": (1, 1)},
            ],
            "terrain": {},
        }
        world = sampler.sample_world(observed, player=0, fog_manager=fog)
        opponent_units = [u for u in world["units"] if u["player"] == 1]
        assert len(opponent_units) == 0

    def test_sample_world_multiple_opponents(self):
        """With 3 players, units are placed for each opponent."""
        board_size = 6
        rng = np.random.default_rng(42)
        sampler = WorldSampler(board_size=board_size, num_players=3, rng=rng)

        fog = _make_fog_manager(board_size, visible_tiles=[(0, 0)])

        observed = {
            "units": [
                {"id": 1, "player": 0, "type": "warrior", "position": (0, 0)},
                {"id": 2, "player": 0, "type": "archer", "position": (0, 0)},
            ],
            "num_players": 3,
            "terrain": {},
        }

        world = sampler.sample_world(observed, player=0, fog_manager=fog)

        p1_units = [u for u in world["units"] if u["player"] == 1]
        p2_units = [u for u in world["units"] if u["player"] == 2]

        # Each opponent should get up to own_unit_count (2) hidden units
        assert len(p1_units) > 0
        assert len(p2_units) > 0

    def test_rng_determinism(self, board_size: int):
        """Identical RNG seeds produce identical sampled worlds."""
        fog = _make_fog_manager(board_size, visible_tiles=[(0, 0)])
        observed = {
            "units": [
                {"id": 1, "player": 0, "type": "warrior", "position": (0, 0)},
            ],
            "terrain": {},
        }

        ws1 = WorldSampler(board_size=board_size, num_players=2, rng=np.random.default_rng(99))
        ws2 = WorldSampler(board_size=board_size, num_players=2, rng=np.random.default_rng(99))

        world1 = ws1.sample_world(observed, player=0, fog_manager=fog)
        world2 = ws2.sample_world(observed, player=0, fog_manager=fog)

        units1 = [(u["player"], u["position"], u["type"]) for u in world1["units"] if u["id"] == -1]
        units2 = [(u["player"], u["position"], u["type"]) for u in world2["units"] if u["id"] == -1]

        assert units1 == units2

    def test_placed_units_have_correct_schema(self, board_size: int):
        """Each placed synthetic unit has all required fields with valid values."""
        rng = np.random.default_rng(7)
        sampler = WorldSampler(board_size=board_size, num_players=2, rng=rng)
        fog = _make_fog_manager(board_size, visible_tiles=[(0, 0)])
        observed = {
            "units": [
                {"id": 1, "player": 0, "type": "warrior", "position": (0, 0)},
            ],
            "terrain": {},
        }

        world = sampler.sample_world(observed, player=0, fog_manager=fog)
        for u in world["units"]:
            if u["id"] == -1:
                assert u["player"] != 0
                assert u["type"] in ("warrior", "archer", "settler")
                assert u["health"] == 100
                assert u["moves_remaining"] == 2
                expected_strength = {"warrior": 20, "archer": 15, "settler": 0}
                assert u["strength"] == expected_strength[u["type"]]


# ---------------------------------------------------------------------------
# DeterminizedMCTS tests
# ---------------------------------------------------------------------------


class TestDeterminizedMCTSCombineProbs:
    """Tests for DeterminizedMCTS._combine_probs."""

    def test_average_single(self):
        """Average of one array is itself (renormalized)."""
        cfg = DeterminizationConfig(num_worlds=1, combine_method="average")
        det = DeterminizedMCTS(base_mcts=None, config=cfg, world_sampler=None)  # type: ignore[arg-type]

        probs = [np.array([0.2, 0.8])]
        result = det._combine_probs(probs)
        np.testing.assert_allclose(result, [0.2, 0.8], atol=1e-7)

    def test_average_multiple(self):
        """Average combining produces the mean of inputs."""
        cfg = DeterminizationConfig(num_worlds=2, combine_method="average")
        det = DeterminizedMCTS(base_mcts=None, config=cfg, world_sampler=None)  # type: ignore[arg-type]

        probs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        result = det._combine_probs(probs)
        # Mean = [0.5, 0.0, 0.5]; sum = 1.0 -> normalized to same
        np.testing.assert_allclose(result, [0.5, 0.0, 0.5], atol=1e-7)

    def test_max_combining(self):
        """Max combining takes element-wise max and renormalizes."""
        cfg = DeterminizationConfig(num_worlds=2, combine_method="max")
        det = DeterminizedMCTS(base_mcts=None, config=cfg, world_sampler=None)  # type: ignore[arg-type]

        probs = [
            np.array([0.6, 0.4, 0.0]),
            np.array([0.0, 0.3, 0.7]),
        ]
        result = det._combine_probs(probs)
        # Max = [0.6, 0.4, 0.7]; sum = 1.7; normalized
        expected = np.array([0.6, 0.4, 0.7])
        expected = expected / expected.sum()
        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_combine_probs_renormalizes(self):
        """Result sums to 1.0 regardless of input."""
        cfg = DeterminizationConfig(num_worlds=3, combine_method="average")
        det = DeterminizedMCTS(base_mcts=None, config=cfg, world_sampler=None)  # type: ignore[arg-type]

        probs = [
            np.array([0.5, 0.5]),
            np.array([0.3, 0.7]),
            np.array([0.1, 0.9]),
        ]
        result = det._combine_probs(probs)
        assert result.sum() == pytest.approx(1.0)

    def test_combine_probs_all_zeros_no_division(self):
        """When all probs sum to 0, no division occurs and zeros are returned."""
        cfg = DeterminizationConfig(num_worlds=2, combine_method="average")
        det = DeterminizedMCTS(base_mcts=None, config=cfg, world_sampler=None)  # type: ignore[arg-type]

        probs = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        ]
        result = det._combine_probs(probs)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])


class TestDeterminizedMCTSAverageValues:
    """Tests for DeterminizedMCTS._average_values static method."""

    def test_empty_values(self):
        """Empty input returns zeros for all players."""
        result = DeterminizedMCTS._average_values([], num_players=3)
        assert result == {0: 0.0, 1: 0.0, 2: 0.0}

    def test_single_world(self):
        """Single world returns its values unchanged."""
        values = [{0: 0.5, 1: -0.5}]
        result = DeterminizedMCTS._average_values(values, num_players=2)
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(-0.5)

    def test_multiple_worlds(self):
        """Multiple worlds are averaged correctly."""
        values = [
            {0: 1.0, 1: -1.0},
            {0: 0.0, 1: 0.0},
            {0: -0.5, 1: 0.5},
        ]
        result = DeterminizedMCTS._average_values(values, num_players=2)
        assert result[0] == pytest.approx((1.0 + 0.0 - 0.5) / 3)
        assert result[1] == pytest.approx((-1.0 + 0.0 + 0.5) / 3)

    def test_missing_player_key_defaults_zero(self):
        """If a player key is missing from a values dict, it counts as 0."""
        values = [
            {0: 0.8},  # player 1 missing
            {0: 0.2, 1: 0.4},
        ]
        result = DeterminizedMCTS._average_values(values, num_players=2)
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(0.2)


class TestDeterminizedMCTSInjectWorld:
    """Tests for DeterminizedMCTS._inject_world."""

    def test_inject_units_and_cities(self):
        """Units and cities in the world dict are injected into the env."""
        env = MagicMock()
        env.units = []
        env.cities = []

        world: dict[str, Any] = {
            "units": [
                {"id": 1, "player": 0, "type": "warrior", "position": (2, 3),
                 "health": 100, "moves_remaining": 2, "strength": 20},
            ],
            "cities": [
                {"id": 10, "player": 0, "name": "TestCity", "position": (1, 1),
                 "population": 3, "production": 5, "producing": "warrior",
                 "defense": 25},
            ],
        }

        DeterminizedMCTS._inject_world(env, world)

        assert len(env.units) == 1
        assert env.units[0].player == 0
        assert env.units[0].unit_type == "warrior"
        assert env.units[0].position == (2, 3)

        assert len(env.cities) == 1
        assert env.cities[0].player == 0
        assert env.cities[0].name == "TestCity"
        assert env.cities[0].population == 3

    def test_inject_world_no_units_cities_attrs(self):
        """If env has no units/cities attrs, _inject_world is a no-op."""
        env = MagicMock(spec=[])  # empty spec -> no attributes
        world: dict[str, Any] = {"units": [{"id": 1, "player": 0}]}
        # Should not raise
        DeterminizedMCTS._inject_world(env, world)

    def test_inject_world_clears_existing(self):
        """Existing units and cities on env are replaced, not appended to."""
        env = MagicMock()
        env.units = ["old_unit"]
        env.cities = ["old_city"]

        world: dict[str, Any] = {
            "units": [
                {"id": 1, "player": 0, "type": "archer", "position": (0, 0),
                 "health": 80, "moves_remaining": 1, "strength": 15},
            ],
            "cities": [],
        }

        DeterminizedMCTS._inject_world(env, world)
        assert len(env.units) == 1
        assert env.units[0].unit_type == "archer"
        assert env.cities == []


class TestDeterminizedMCTSSearch:
    """Tests for DeterminizedMCTS.search integration."""

    @pytest.fixture
    def board_size(self) -> int:
        return 4

    @pytest.fixture
    def fog_manager(self, board_size: int) -> MagicMock:
        return _make_fog_manager(board_size, visible_tiles=[(0, 0)])

    @pytest.fixture
    def game_state(self) -> dict[str, Any]:
        return {
            "units": [{"id": 1, "player": 0, "type": "warrior", "position": (0, 0)}],
            "cities": [],
            "terrain": {},
        }

    def test_search_successful(
        self, board_size: int, fog_manager: MagicMock, game_state: dict[str, Any]
    ):
        """Basic search returns combined probs and values."""
        action_size = 5
        cfg = DeterminizationConfig(num_worlds=3, combine_method="average")
        base_mcts = _make_base_mcts(action_size=action_size, num_players=2)
        sampler = WorldSampler(board_size=board_size, num_players=2, rng=np.random.default_rng(1))
        env = _make_env(action_size=action_size, num_players=2)

        det = DeterminizedMCTS(base_mcts, cfg, sampler)
        probs, values = det.search(env, game_state, player=0, fog_manager=fog_manager)

        assert probs.shape == (action_size,)
        assert probs.sum() == pytest.approx(1.0)
        assert 0 in values and 1 in values

    def test_search_calls_base_mcts_num_worlds_times(
        self, board_size: int, fog_manager: MagicMock, game_state: dict[str, Any]
    ):
        """Base MCTS search is called once per world."""
        cfg = DeterminizationConfig(num_worlds=5, combine_method="average")
        base_mcts = _make_base_mcts(action_size=10, num_players=2)
        sampler = WorldSampler(board_size=board_size, num_players=2, rng=np.random.default_rng(0))
        env = _make_env(action_size=10, num_players=2)

        det = DeterminizedMCTS(base_mcts, cfg, sampler)
        det.search(env, game_state, player=0, fog_manager=fog_manager)

        assert base_mcts.search.call_count == 5

    def test_search_first_world_adds_noise(
        self, board_size: int, fog_manager: MagicMock, game_state: dict[str, Any]
    ):
        """Only the first world search uses add_noise=True."""
        cfg = DeterminizationConfig(num_worlds=3, combine_method="average")
        base_mcts = _make_base_mcts(action_size=10, num_players=2)
        sampler = WorldSampler(board_size=board_size, num_players=2, rng=np.random.default_rng(0))
        env = _make_env(action_size=10, num_players=2)

        det = DeterminizedMCTS(base_mcts, cfg, sampler)
        det.search(env, game_state, player=0, fog_manager=fog_manager)

        calls = base_mcts.search.call_args_list
        # First call: add_noise=True
        assert calls[0][1].get("add_noise") is True or (calls[0][0] == (env.clone.return_value,) and calls[0][1].get("add_noise") is True)
        # Remaining calls: add_noise=False
        for call in calls[1:]:
            assert call[1].get("add_noise") is False

    def test_search_all_worlds_fail_returns_uniform(
        self, board_size: int, fog_manager: MagicMock, game_state: dict[str, Any]
    ):
        """If every world's MCTS fails, fallback to uniform distribution."""
        action_size = 4
        cfg = DeterminizationConfig(num_worlds=3, combine_method="average")
        base_mcts = _make_base_mcts(action_size=action_size, num_players=2, fail=True)
        sampler = WorldSampler(board_size=board_size, num_players=2, rng=np.random.default_rng(0))
        env = _make_env(action_size=action_size, num_players=2)

        det = DeterminizedMCTS(base_mcts, cfg, sampler)
        probs, values = det.search(env, game_state, player=0, fog_manager=fog_manager)

        # Uniform over action_size
        expected_probs = np.ones(action_size) / action_size
        np.testing.assert_allclose(probs, expected_probs)

        # Values all 0
        assert values == {0: 0.0, 1: 0.0}

    def test_search_partial_failures_uses_successful_worlds(
        self, board_size: int, fog_manager: MagicMock, game_state: dict[str, Any]
    ):
        """If some worlds fail, results from successful ones are still used."""
        action_size = 4
        cfg = DeterminizationConfig(num_worlds=4, combine_method="average")

        # First call succeeds with biased probs; rest fail
        good_probs = np.array([0.7, 0.1, 0.1, 0.1])
        good_values = {0: 0.6, 1: -0.6}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (good_probs.copy(), dict(good_values))
            raise RuntimeError("boom")

        base_mcts = MagicMock()
        base_mcts.search.side_effect = side_effect

        sampler = WorldSampler(board_size=board_size, num_players=2, rng=np.random.default_rng(0))
        env = _make_env(action_size=action_size, num_players=2)

        det = DeterminizedMCTS(base_mcts, cfg, sampler)
        probs, values = det.search(env, game_state, player=0, fog_manager=fog_manager)

        # Only one successful world -> result matches good_probs
        np.testing.assert_allclose(probs, good_probs, atol=1e-7)
        assert values[0] == pytest.approx(0.6)
        assert values[1] == pytest.approx(-0.6)

    def test_search_clones_env_per_world(
        self, board_size: int, fog_manager: MagicMock, game_state: dict[str, Any]
    ):
        """Each world gets its own cloned environment."""
        cfg = DeterminizationConfig(num_worlds=4, combine_method="average")
        base_mcts = _make_base_mcts(action_size=10, num_players=2)
        sampler = WorldSampler(board_size=board_size, num_players=2, rng=np.random.default_rng(0))
        env = _make_env(action_size=10, num_players=2)

        det = DeterminizedMCTS(base_mcts, cfg, sampler)
        det.search(env, game_state, player=0, fog_manager=fog_manager)

        assert env.clone.call_count == 4

    def test_search_masks_observation_for_player(
        self, board_size: int, fog_manager: MagicMock, game_state: dict[str, Any]
    ):
        """Search calls fog_manager.mask_observation for the querying player."""
        cfg = DeterminizationConfig(num_worlds=1, combine_method="average")
        base_mcts = _make_base_mcts(action_size=10, num_players=2)
        sampler = WorldSampler(board_size=board_size, num_players=2, rng=np.random.default_rng(0))
        env = _make_env(action_size=10, num_players=2)

        det = DeterminizedMCTS(base_mcts, cfg, sampler)
        det.search(env, game_state, player=0, fog_manager=fog_manager)

        fog_manager.mask_observation.assert_called_once_with(game_state, 0)

    def test_search_with_varying_mcts_outputs(
        self, board_size: int, fog_manager: MagicMock, game_state: dict[str, Any]
    ):
        """Different per-world MCTS outputs are combined correctly."""
        action_size = 3
        cfg = DeterminizationConfig(num_worlds=2, combine_method="average")

        results = [
            (np.array([0.6, 0.2, 0.2]), {0: 0.5, 1: -0.5}),
            (np.array([0.2, 0.6, 0.2]), {0: -0.3, 1: 0.3}),
        ]

        call_idx = 0

        def mock_search(*args, **kwargs):
            nonlocal call_idx
            r = results[call_idx]
            call_idx += 1
            return (r[0].copy(), dict(r[1]))

        base_mcts = MagicMock()
        base_mcts.search.side_effect = mock_search

        sampler = WorldSampler(board_size=board_size, num_players=2, rng=np.random.default_rng(0))
        env = _make_env(action_size=action_size, num_players=2)

        det = DeterminizedMCTS(base_mcts, cfg, sampler)
        probs, values = det.search(env, game_state, player=0, fog_manager=fog_manager)

        # Average probs: [0.4, 0.4, 0.2]; sum=1.0 -> normalized same
        np.testing.assert_allclose(probs, [0.4, 0.4, 0.2], atol=1e-7)
        # Average values
        assert values[0] == pytest.approx(0.1)
        assert values[1] == pytest.approx(-0.1)

    def test_search_max_combining(
        self, board_size: int, fog_manager: MagicMock, game_state: dict[str, Any]
    ):
        """Search with max combining picks element-wise maximum."""
        action_size = 3
        cfg = DeterminizationConfig(num_worlds=2, combine_method="max")

        results = [
            (np.array([0.8, 0.1, 0.1]), {0: 0.5, 1: -0.5}),
            (np.array([0.1, 0.8, 0.1]), {0: 0.3, 1: -0.3}),
        ]

        call_idx = 0

        def mock_search(*args, **kwargs):
            nonlocal call_idx
            r = results[call_idx]
            call_idx += 1
            return (r[0].copy(), dict(r[1]))

        base_mcts = MagicMock()
        base_mcts.search.side_effect = mock_search

        sampler = WorldSampler(board_size=board_size, num_players=2, rng=np.random.default_rng(0))
        env = _make_env(action_size=action_size, num_players=2)

        det = DeterminizedMCTS(base_mcts, cfg, sampler)
        probs, _values = det.search(env, game_state, player=0, fog_manager=fog_manager)

        # Max = [0.8, 0.8, 0.1]; sum=1.7; normalized
        expected = np.array([0.8, 0.8, 0.1])
        expected = expected / expected.sum()
        np.testing.assert_allclose(probs, expected, atol=1e-7)
