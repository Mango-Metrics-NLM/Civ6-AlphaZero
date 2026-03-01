"""Tests for observation model predicting hidden game state."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.multiplayer.observation_model import (
    ObservationModel,
    ObservationModelConfig,
    _UNIT_MOVES,
    _UNIT_STRENGTH,
)


# ---------------------------------------------------------------------------
# ObservationModelConfig tests
# ---------------------------------------------------------------------------


class TestObservationModelConfig:
    """Tests for ObservationModelConfig frozen dataclass."""

    def test_default_values(self):
        """Default config uses uniform method with warrior/archer/settler."""
        config = ObservationModelConfig()
        assert config.method == "uniform"
        assert config.default_unit_types == ("warrior", "archer", "settler")
        assert config.default_unit_weights == (0.5, 0.35, 0.15)

    def test_frozen(self):
        """Config is immutable (frozen dataclass)."""
        config = ObservationModelConfig()
        with pytest.raises(AttributeError):
            config.method = "learned"  # type: ignore[misc]

    @pytest.mark.parametrize("method", ["uniform", "learned", "heuristic"])
    def test_valid_methods(self, method: str):
        """All three valid methods are accepted."""
        config = ObservationModelConfig(method=method)
        assert config.method == method

    def test_invalid_method_raises(self):
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown observation model method"):
            ObservationModelConfig(method="deep_learning")

    def test_mismatched_types_weights_raises(self):
        """Mismatched unit_types and unit_weights lengths raise ValueError."""
        with pytest.raises(ValueError, match="unit_types length.*must match.*unit_weights length"):
            ObservationModelConfig(
                default_unit_types=("warrior", "archer"),
                default_unit_weights=(0.5, 0.3, 0.2),
            )

    def test_mismatched_weights_shorter(self):
        """Fewer weights than types also raises ValueError."""
        with pytest.raises(ValueError, match="unit_types length"):
            ObservationModelConfig(
                default_unit_types=("warrior", "archer", "settler"),
                default_unit_weights=(0.6, 0.4),
            )

    def test_custom_unit_types(self):
        """Custom unit types and weights are accepted if lengths match."""
        config = ObservationModelConfig(
            default_unit_types=("pikeman", "cavalry"),
            default_unit_weights=(0.7, 0.3),
        )
        assert config.default_unit_types == ("pikeman", "cavalry")
        assert config.default_unit_weights == (0.7, 0.3)

    def test_single_unit_type(self):
        """Single unit type with matching single weight is valid."""
        config = ObservationModelConfig(
            default_unit_types=("warrior",),
            default_unit_weights=(1.0,),
        )
        assert len(config.default_unit_types) == 1

    def test_empty_types_and_weights(self):
        """Empty types and weights are accepted (lengths match at 0)."""
        config = ObservationModelConfig(
            default_unit_types=(),
            default_unit_weights=(),
        )
        assert len(config.default_unit_types) == 0


# ---------------------------------------------------------------------------
# ObservationModel.__init__ tests
# ---------------------------------------------------------------------------


class TestObservationModelInit:
    """Tests for ObservationModel construction."""

    def test_default_config(self):
        """Constructing with no config uses defaults."""
        model = ObservationModel(board_size=16, num_players=2)
        assert model.board_size == 16
        assert model.num_players == 2
        assert model.config.method == "uniform"

    def test_custom_config(self):
        """Custom config is stored on the model."""
        config = ObservationModelConfig(method="heuristic")
        model = ObservationModel(board_size=8, num_players=4, config=config)
        assert model.config.method == "heuristic"
        assert model.num_players == 4

    def test_custom_rng(self):
        """Custom rng is used for reproducibility."""
        rng = np.random.default_rng(42)
        model = ObservationModel(board_size=16, num_players=2, rng=rng)
        assert model._rng is rng

    def test_default_rng_created(self):
        """Without explicit rng, a default generator is created."""
        model = ObservationModel(board_size=16, num_players=2)
        assert isinstance(model._rng, np.random.Generator)


# ---------------------------------------------------------------------------
# ObservationModel.predict_hidden_units tests
# ---------------------------------------------------------------------------


class TestPredictHiddenUnits:
    """Tests for predict_hidden_units."""

    @pytest.fixture
    def model(self) -> ObservationModel:
        """Seeded model for reproducible tests."""
        return ObservationModel(
            board_size=16,
            num_players=2,
            rng=np.random.default_rng(12345),
        )

    @pytest.fixture
    def visible_state(self) -> dict:
        """Minimal visible state dict."""
        return {"num_players": 2}

    @pytest.fixture
    def candidate_tiles(self) -> list[tuple[int, int]]:
        """A set of candidate positions."""
        return [(1, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]

    def test_zero_hidden_units_returns_empty(self, model, visible_state, candidate_tiles):
        """num_hidden_units=0 returns empty list immediately."""
        result = model.predict_hidden_units(visible_state, player=0, num_hidden_units=0, candidate_tiles=candidate_tiles)
        assert result == []

    def test_negative_hidden_units_returns_empty(self, model, visible_state, candidate_tiles):
        """Negative num_hidden_units returns empty list."""
        result = model.predict_hidden_units(visible_state, player=0, num_hidden_units=-5, candidate_tiles=candidate_tiles)
        assert result == []

    def test_empty_candidates_returns_empty(self, model, visible_state):
        """Empty candidate_tiles returns empty list."""
        result = model.predict_hidden_units(visible_state, player=0, num_hidden_units=3, candidate_tiles=[])
        assert result == []

    def test_correct_number_of_units(self, model, visible_state, candidate_tiles):
        """Returns exactly num_hidden_units when enough candidates."""
        result = model.predict_hidden_units(
            visible_state, player=0, num_hidden_units=3, candidate_tiles=candidate_tiles
        )
        assert len(result) == 3

    def test_capped_by_candidate_count(self, model, visible_state):
        """When num_hidden_units > candidates, returns len(candidates)."""
        tiles = [(0, 0), (1, 1)]
        result = model.predict_hidden_units(
            visible_state, player=0, num_hidden_units=10, candidate_tiles=tiles
        )
        assert len(result) == 2

    def test_single_candidate_single_unit(self, model, visible_state):
        """Single candidate and single requested unit works."""
        tiles = [(5, 5)]
        result = model.predict_hidden_units(
            visible_state, player=0, num_hidden_units=1, candidate_tiles=tiles
        )
        assert len(result) == 1
        assert result[0]["position"] == (5, 5)

    def test_unit_dict_structure(self, model, visible_state, candidate_tiles):
        """Each unit dict has all required keys with correct types."""
        result = model.predict_hidden_units(
            visible_state, player=0, num_hidden_units=1, candidate_tiles=candidate_tiles
        )
        unit = result[0]
        assert "id" in unit
        assert "player" in unit
        assert "type" in unit
        assert "position" in unit
        assert "health" in unit
        assert "moves_remaining" in unit
        assert "strength" in unit

        assert unit["id"] == -1
        assert unit["health"] == 100
        assert isinstance(unit["position"], tuple)
        assert unit["type"] in ("warrior", "archer", "settler")

    def test_positions_from_candidates(self, model, visible_state, candidate_tiles):
        """All positions come from candidate_tiles."""
        result = model.predict_hidden_units(
            visible_state, player=0, num_hidden_units=4, candidate_tiles=candidate_tiles
        )
        for unit in result:
            assert unit["position"] in candidate_tiles

    def test_positions_without_replacement(self, model, visible_state, candidate_tiles):
        """Positions are sampled without replacement (no duplicates)."""
        result = model.predict_hidden_units(
            visible_state, player=0, num_hidden_units=len(candidate_tiles), candidate_tiles=candidate_tiles
        )
        positions = [unit["position"] for unit in result]
        assert len(positions) == len(set(positions))

    def test_unit_types_from_config(self, model, visible_state, candidate_tiles):
        """All unit types come from the configured types."""
        result = model.predict_hidden_units(
            visible_state, player=0, num_hidden_units=5, candidate_tiles=candidate_tiles
        )
        for unit in result:
            assert unit["type"] in model.config.default_unit_types

    def test_strength_and_moves_lookup(self, model, visible_state, candidate_tiles):
        """Strength and moves_remaining come from lookup tables."""
        result = model.predict_hidden_units(
            visible_state, player=0, num_hidden_units=6, candidate_tiles=candidate_tiles
        )
        for unit in result:
            utype = unit["type"]
            assert unit["strength"] == _UNIT_STRENGTH.get(utype, 20)
            assert unit["moves_remaining"] == _UNIT_MOVES.get(utype, 2)

    def test_unknown_unit_type_defaults(self):
        """Unknown unit types fall back to default strength=20, moves=2."""
        config = ObservationModelConfig(
            default_unit_types=("catapult",),
            default_unit_weights=(1.0,),
        )
        model = ObservationModel(board_size=8, num_players=2, config=config, rng=np.random.default_rng(0))
        result = model.predict_hidden_units(
            {"num_players": 2}, player=0, num_hidden_units=1, candidate_tiles=[(3, 3)]
        )
        assert result[0]["type"] == "catapult"
        assert result[0]["strength"] == 20  # default from .get fallback
        assert result[0]["moves_remaining"] == 2  # default from .get fallback

    def test_all_units_assigned_to_opponent(self, model, visible_state, candidate_tiles):
        """All hidden units are assigned to opponents, not the observing player."""
        result = model.predict_hidden_units(
            visible_state, player=0, num_hidden_units=5, candidate_tiles=candidate_tiles
        )
        for unit in result:
            assert unit["player"] != 0

    def test_reproducible_with_seed(self, visible_state, candidate_tiles):
        """Same seed produces identical results."""
        model_a = ObservationModel(board_size=16, num_players=2, rng=np.random.default_rng(999))
        model_b = ObservationModel(board_size=16, num_players=2, rng=np.random.default_rng(999))

        result_a = model_a.predict_hidden_units(visible_state, player=0, num_hidden_units=3, candidate_tiles=candidate_tiles)
        result_b = model_b.predict_hidden_units(visible_state, player=0, num_hidden_units=3, candidate_tiles=candidate_tiles)

        assert result_a == result_b

    def test_weight_distribution_statistical(self):
        """Over many samples, unit type distribution matches configured weights."""
        config = ObservationModelConfig(
            default_unit_types=("warrior", "settler"),
            default_unit_weights=(0.8, 0.2),
        )
        model = ObservationModel(board_size=8, num_players=2, config=config, rng=np.random.default_rng(42))
        tiles = [(i, j) for i in range(8) for j in range(8)]

        counts: dict[str, int] = {"warrior": 0, "settler": 0}
        num_trials = 500
        for _ in range(num_trials):
            result = model.predict_hidden_units(
                {"num_players": 2}, player=0, num_hidden_units=1, candidate_tiles=tiles
            )
            counts[result[0]["type"]] += 1

        warrior_frac = counts["warrior"] / num_trials
        # With 500 samples, warrior should be approximately 0.8 (within tolerance)
        assert 0.65 < warrior_frac < 0.95


# ---------------------------------------------------------------------------
# ObservationModel._pick_opponent tests
# ---------------------------------------------------------------------------


class TestPickOpponent:
    """Tests for _pick_opponent."""

    def test_returns_valid_opponent_2p(self):
        """In a 2-player game, the opponent of player 0 is player 1."""
        model = ObservationModel(board_size=8, num_players=2, rng=np.random.default_rng(0))
        state = {"num_players": 2}
        opp = model._pick_opponent(player=0, visible_state=state)
        assert opp == 1

    def test_returns_valid_opponent_not_self(self):
        """Opponent is never the observing player."""
        model = ObservationModel(board_size=8, num_players=4, rng=np.random.default_rng(0))
        state = {"num_players": 4}
        for _ in range(50):
            opp = model._pick_opponent(player=2, visible_state=state)
            assert opp != 2
            assert 0 <= opp < 4

    def test_uniform_distribution_among_opponents(self):
        """Opponents are sampled uniformly (statistical test)."""
        model = ObservationModel(board_size=8, num_players=4, rng=np.random.default_rng(42))
        state = {"num_players": 4}

        counts = {0: 0, 1: 0, 3: 0}
        num_trials = 600
        for _ in range(num_trials):
            opp = model._pick_opponent(player=2, visible_state=state)
            counts[opp] += 1

        # With 600 trials and 3 opponents, each should get ~200
        for opp_id, count in counts.items():
            assert count > 100, f"Opponent {opp_id} undersampled: {count}"

    def test_uses_visible_state_num_players(self):
        """Picks from visible_state's num_players, not the model's."""
        # Model says 4 players, but visible state says 3
        model = ObservationModel(board_size=8, num_players=4, rng=np.random.default_rng(0))
        state = {"num_players": 3}
        for _ in range(30):
            opp = model._pick_opponent(player=0, visible_state=state)
            assert opp in (1, 2)  # only players 0,1,2 exist per visible_state

    def test_falls_back_to_model_num_players(self):
        """When visible_state has no num_players key, model's value is used."""
        model = ObservationModel(board_size=8, num_players=3, rng=np.random.default_rng(0))
        state = {}  # no num_players key
        for _ in range(30):
            opp = model._pick_opponent(player=1, visible_state=state)
            assert opp in (0, 2)

    def test_no_opponents_fallback(self):
        """When num_players=1, the fallback formula is used (line 174)."""
        model = ObservationModel(board_size=8, num_players=1, rng=np.random.default_rng(0))
        state = {"num_players": 1}
        # Only player=0, opponents=[], falls back to (0+1) % max(1,2) = 1 % 2 = 1
        opp = model._pick_opponent(player=0, visible_state=state)
        assert opp == 1

    def test_no_opponents_fallback_player_1(self):
        """Fallback with player=1 and num_players=1 still returns an int."""
        model = ObservationModel(board_size=8, num_players=1, rng=np.random.default_rng(0))
        state = {"num_players": 1}
        # (1+1) % max(1,2) = 2 % 2 = 0
        opp = model._pick_opponent(player=1, visible_state=state)
        assert opp == 0

    def test_no_opponents_zero_players(self):
        """Edge case: num_players=0 uses max(0,2)=2 in fallback."""
        model = ObservationModel(board_size=8, num_players=0, rng=np.random.default_rng(0))
        state = {"num_players": 0}
        # opponents=[], fallback: (0+1) % max(0,2) = 1 % 2 = 1
        opp = model._pick_opponent(player=0, visible_state=state)
        assert opp == 1


# ---------------------------------------------------------------------------
# ObservationModel.reset tests
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for reset (no-op for uniform model)."""

    def test_reset_is_noop(self):
        """Reset returns None and does not change state."""
        model = ObservationModel(board_size=8, num_players=2)
        result = model.reset()
        assert result is None

    def test_reset_preserves_config(self):
        """Config and board_size are unchanged after reset."""
        config = ObservationModelConfig(method="heuristic")
        model = ObservationModel(board_size=16, num_players=3, config=config)
        model.reset()
        assert model.config.method == "heuristic"
        assert model.board_size == 16
        assert model.num_players == 3


# ---------------------------------------------------------------------------
# Integration-style tests combining multiple paths
# ---------------------------------------------------------------------------


class TestObservationModelIntegration:
    """Integration tests exercising multiple code paths together."""

    def test_full_workflow(self):
        """Reset, predict, verify the full cycle."""
        model = ObservationModel(board_size=8, num_players=3, rng=np.random.default_rng(7))
        model.reset()

        tiles = [(r, c) for r in range(4) for c in range(4)]
        state = {"num_players": 3}

        units = model.predict_hidden_units(state, player=1, num_hidden_units=5, candidate_tiles=tiles)
        assert len(units) == 5
        for u in units:
            assert u["player"] in (0, 2)  # not player 1
            assert u["position"] in tiles
            assert u["health"] == 100
            assert u["id"] == -1

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_various_player_counts(self, num_players: int):
        """Prediction works for various player counts."""
        model = ObservationModel(
            board_size=8, num_players=num_players, rng=np.random.default_rng(0)
        )
        tiles = [(0, 0), (1, 1), (2, 2), (3, 3)]
        state = {"num_players": num_players}

        units = model.predict_hidden_units(state, player=0, num_hidden_units=2, candidate_tiles=tiles)
        assert len(units) == 2
        for u in units:
            assert u["player"] != 0
            assert 0 <= u["player"] < num_players

    def test_predict_all_candidates_used(self):
        """When requesting exactly len(candidates) units, all tiles are used."""
        tiles = [(0, 0), (1, 1), (2, 2)]
        model = ObservationModel(board_size=8, num_players=2, rng=np.random.default_rng(0))
        state = {"num_players": 2}

        units = model.predict_hidden_units(state, player=0, num_hidden_units=3, candidate_tiles=tiles)
        positions = {u["position"] for u in units}
        assert positions == set(tiles)
