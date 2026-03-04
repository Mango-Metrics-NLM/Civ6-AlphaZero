"""Unit tests for N-player dynamic channel encoding in MicroCivEncoder."""

from __future__ import annotations

import pytest
import torch

from alpha_zero.envs.civ6_encoder import MicroCivEncoder
from tests.test_constants import TestDefaults, microciv_channels

# ---------------------------------------------------------------------------
# Channel count computation
# ---------------------------------------------------------------------------


class TestChannelCountFormula:
    """Verify the channel-count formula matches expected values."""

    @pytest.mark.parametrize(
        "num_players,expected_channels",
        [
            (2, 13),
            (3, 17),
            (4, 21),
            (6, 29),
            (8, 37),
        ],
    )
    def test_channel_count(self, num_players: int, expected_channels: int) -> None:
        encoder = MicroCivEncoder(
            board_size=TestDefaults.SMALL_BOARD_SIZE, num_players=num_players
        )
        assert encoder.total_channels == expected_channels

    @pytest.mark.parametrize(
        "num_players,expected_channels",
        [
            (2, 13),
            (3, 17),
            (4, 21),
            (6, 29),
            (8, 37),
        ],
    )
    def test_microciv_channels_helper(
        self, num_players: int, expected_channels: int
    ) -> None:
        assert microciv_channels(num_players) == expected_channels

    def test_test_defaults_constants(self) -> None:
        assert TestDefaults.MICROCIV_CHANNELS == 13
        assert TestDefaults.MICROCIV_4P_CHANNELS == 21
        assert TestDefaults.MICROCIV_8P_CHANNELS == 37


# ---------------------------------------------------------------------------
# Spatial shape
# ---------------------------------------------------------------------------


class TestSpatialShape:
    """spatial_shape property must match total_channels."""

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_spatial_shape_matches_channels(self, num_players: int) -> None:
        bs = TestDefaults.SMALL_BOARD_SIZE
        encoder = MicroCivEncoder(board_size=bs, num_players=num_players)
        expected = (microciv_channels(num_players), bs, bs)
        assert encoder.spatial_shape == expected

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_encode_output_shape(self, num_players: int) -> None:
        bs = TestDefaults.SMALL_BOARD_SIZE
        encoder = MicroCivEncoder(board_size=bs, num_players=num_players)
        state: dict = {"current_player": 0}
        spatial, vector = encoder.encode(state)
        assert spatial.shape == (microciv_channels(num_players), bs, bs)
        assert vector.shape == (64,)


# ---------------------------------------------------------------------------
# Player-relative spatial encoding
# ---------------------------------------------------------------------------


class TestPlayerRelativeEncoding:
    """Own units should always appear in the first unit-channel block."""

    @pytest.fixture
    def board_size(self) -> int:
        return TestDefaults.SMALL_BOARD_SIZE

    def _make_state(
        self,
        current_player: int,
        units: list[dict],
        cities: list[dict] | None = None,
        num_players: int = 2,
    ) -> dict:
        return {
            "current_player": current_player,
            "units": units,
            "cities": cities or [],
        }

    def test_own_units_always_channels_3_to_5_two_player(self, board_size: int) -> None:
        """With 2 players own units occupy channels 3-5 regardless of player id."""
        for current in (0, 1):
            encoder = MicroCivEncoder(board_size=board_size, num_players=2)
            state = self._make_state(
                current_player=current,
                units=[
                    {"player": current, "type": "warrior", "position": (1, 1), "health": 100},
                    {"player": current, "type": "settler", "position": (2, 2), "health": 80},
                ],
            )
            spatial, _ = encoder.encode(state)
            assert spatial[3, 1, 1].item() == pytest.approx(1.0)
            assert spatial[4, 2, 2].item() == pytest.approx(0.8)

    def test_enemy_in_opponent_channels_two_player(self, board_size: int) -> None:
        """Enemy units land in channels 6-8 for 2-player."""
        encoder = MicroCivEncoder(board_size=board_size, num_players=2)
        state = self._make_state(
            current_player=0,
            units=[
                {"player": 1, "type": "archer", "position": (3, 3), "health": 50},
            ],
        )
        spatial, _ = encoder.encode(state)
        assert spatial[8, 3, 3].item() == pytest.approx(0.5)

    def test_relative_indexing_wraps(self, board_size: int) -> None:
        """Player-relative index uses modular arithmetic."""
        encoder = MicroCivEncoder(board_size=board_size, num_players=4)
        # current_player=3, unit player=1 => rel = (1-3)%4 = 2
        state = self._make_state(
            current_player=3,
            units=[
                {"player": 1, "type": "warrior", "position": (0, 0), "health": 100},
            ],
            num_players=4,
        )
        spatial, _ = encoder.encode(state)
        # rel_player=2 -> channel offset: 3 + 2*3 + 0 = 9
        assert spatial[9, 0, 0].item() == pytest.approx(1.0)

    def test_four_player_distinct_channels(self, board_size: int) -> None:
        """Each of 4 players occupies distinct unit channel blocks."""
        encoder = MicroCivEncoder(board_size=board_size, num_players=4)
        units = [
            {"player": p, "type": "warrior", "position": (p, 0), "health": 100}
            for p in range(4)
        ]
        state = self._make_state(current_player=0, units=units, num_players=4)
        spatial, _ = encoder.encode(state)

        for rel_p in range(4):
            abs_p = rel_p  # current_player=0 so rel==abs
            channel = 3 + rel_p * 3  # warrior = type_idx 0
            assert spatial[channel, 0, abs_p].item() == pytest.approx(1.0), (
                f"rel_player={rel_p}, channel={channel}"
            )

    def test_four_player_city_channels(self, board_size: int) -> None:
        """Each of 4 players has a dedicated city channel."""
        encoder = MicroCivEncoder(board_size=board_size, num_players=4)
        cities = [{"player": p, "position": (p, 1)} for p in range(4)]
        state = self._make_state(current_player=0, units=[], cities=cities, num_players=4)
        spatial, _ = encoder.encode(state)

        city_base = 3 + 4 * 3  # = 15
        for rel_p in range(4):
            channel = city_base + rel_p
            assert spatial[channel, 1, rel_p].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Default 2-player encoder must behave identically to the original."""

    @pytest.fixture
    def sample_state(self) -> dict:
        return {
            "terrain": {
                (0, 0): {"type": "plains"},
                (1, 0): {"type": "hills"},
                (2, 0): {"type": "mountain"},
            },
            "units": [
                {"id": 0, "player": 0, "type": "warrior", "position": (5, 5), "health": 100},
                {"id": 1, "player": 1, "type": "archer", "position": (3, 3), "health": 75},
            ],
            "cities": [
                {"id": 0, "player": 0, "position": (2, 2)},
                {"id": 1, "player": 1, "position": (6, 6)},
            ],
            "turn": 50,
            "current_player": 0,
        }

    def test_default_total_channels(self) -> None:
        encoder = MicroCivEncoder(board_size=TestDefaults.SMALL_BOARD_SIZE)
        assert encoder.total_channels == MicroCivEncoder.TOTAL_CHANNELS == 13

    def test_class_level_constant_unchanged(self) -> None:
        assert MicroCivEncoder.TOTAL_CHANNELS == 13

    def test_spatial_output_identical(self, sample_state: dict) -> None:
        """Default encoder spatial tensor has 13 channels."""
        bs = TestDefaults.BOARD_SIZE
        encoder = MicroCivEncoder(board_size=bs)
        spatial, _ = encoder.encode(sample_state)
        assert spatial.shape == (13, bs, bs)

    def test_terrain_encoding(self, sample_state: dict) -> None:
        encoder = MicroCivEncoder(board_size=TestDefaults.BOARD_SIZE)
        spatial, _ = encoder.encode(sample_state)
        assert spatial[0, 0, 0] == 1.0  # plains
        assert spatial[1, 0, 1] == 1.0  # hills
        assert spatial[2, 0, 2] == 1.0  # mountain

    def test_own_unit_channel(self, sample_state: dict) -> None:
        encoder = MicroCivEncoder(board_size=TestDefaults.BOARD_SIZE)
        spatial, _ = encoder.encode(sample_state)
        # Own warrior at (5,5) -> channel 3
        assert spatial[3, 5, 5].item() == pytest.approx(1.0)

    def test_enemy_unit_channel(self, sample_state: dict) -> None:
        encoder = MicroCivEncoder(board_size=TestDefaults.BOARD_SIZE)
        spatial, _ = encoder.encode(sample_state)
        # Enemy archer at (3,3) -> channel 8 (6 + type_idx 2)
        assert spatial[8, 3, 3].item() == pytest.approx(0.75)

    def test_own_city_channel(self, sample_state: dict) -> None:
        encoder = MicroCivEncoder(board_size=TestDefaults.BOARD_SIZE)
        spatial, _ = encoder.encode(sample_state)
        assert spatial[9, 2, 2].item() == pytest.approx(1.0)

    def test_enemy_city_channel(self, sample_state: dict) -> None:
        encoder = MicroCivEncoder(board_size=TestDefaults.BOARD_SIZE)
        spatial, _ = encoder.encode(sample_state)
        assert spatial[10, 6, 6].item() == pytest.approx(1.0)

    def test_visibility_channel(self, sample_state: dict) -> None:
        encoder = MicroCivEncoder(board_size=TestDefaults.BOARD_SIZE)
        spatial, _ = encoder.encode(sample_state)
        # No visibility key => all visible => channel 11 all 1.0
        assert torch.all(spatial[11] == 1.0)

    def test_bias_channel(self, sample_state: dict) -> None:
        encoder = MicroCivEncoder(board_size=TestDefaults.BOARD_SIZE)
        spatial, _ = encoder.encode(sample_state)
        assert torch.all(spatial[12] == 1.0)

    def test_vector_indices_match_original(self, sample_state: dict) -> None:
        """Key vector indices must be identical to the original layout."""
        encoder = MicroCivEncoder(board_size=TestDefaults.BOARD_SIZE)
        _, vector = encoder.encode(sample_state)

        # [0] turn / 200
        assert vector[0].item() == pytest.approx(50.0 / 200.0)
        # [1] current player
        assert vector[1].item() == pytest.approx(0.0)
        # [2] own unit count / 20 (1 warrior)
        assert vector[2].item() == pytest.approx(1.0 / 20.0)
        # [3] enemy unit count / 20 (1 archer)
        assert vector[3].item() == pytest.approx(1.0 / 20.0)
        # [4] own city count / 5 (1 city)
        assert vector[4].item() == pytest.approx(1.0 / 5.0)
        # [5] enemy city count / 5 (1 city)
        assert vector[5].item() == pytest.approx(1.0 / 5.0)
        # [6] own avg health (100/100)
        assert vector[6].item() == pytest.approx(1.0)
        # [7] enemy avg health (75/100)
        assert vector[7].item() == pytest.approx(0.75)
        # [11] visibility (all visible, no visibility key)
        assert vector[11].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Edge-case and validation tests."""

    def test_invalid_num_players(self) -> None:
        with pytest.raises(ValueError, match="num_players must be >= 2"):
            MicroCivEncoder(board_size=8, num_players=1)

    def test_output_dtypes(self) -> None:
        encoder = MicroCivEncoder(board_size=TestDefaults.SMALL_BOARD_SIZE, num_players=4)
        state: dict = {"current_player": 0}
        spatial, vector = encoder.encode(state)
        assert spatial.dtype == torch.float32
        assert vector.dtype == torch.float32

    def test_empty_state_encodes(self) -> None:
        """Encoding an empty game state should not raise."""
        for n in (2, 3, 6):
            encoder = MicroCivEncoder(board_size=TestDefaults.SMALL_BOARD_SIZE, num_players=n)
            spatial, _vector = encoder.encode({"current_player": 0})
            assert spatial.shape[0] == microciv_channels(n)

    def test_unit_out_of_bounds_ignored(self) -> None:
        encoder = MicroCivEncoder(board_size=4, num_players=3)
        state = {
            "current_player": 0,
            "units": [
                {"player": 0, "type": "warrior", "position": (10, 10), "health": 100},
            ],
        }
        spatial, _ = encoder.encode(state)
        # Should not crash and out-of-bounds unit is ignored
        assert spatial[3].sum().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4-player game state (integration-style)
# ---------------------------------------------------------------------------


class TestFourPlayerEncoding:
    """End-to-end encoding of a 4-player game state."""

    @pytest.fixture
    def four_player_state(self) -> dict:
        return {
            "current_player": 2,
            "terrain": {
                (0, 0): {"type": "plains"},
                (1, 1): {"type": "hills"},
            },
            "units": [
                {"player": 0, "type": "warrior", "position": (0, 0), "health": 100},
                {"player": 1, "type": "settler", "position": (1, 0), "health": 80},
                {"player": 2, "type": "archer", "position": (2, 0), "health": 60},
                {"player": 3, "type": "warrior", "position": (3, 0), "health": 40},
            ],
            "cities": [
                {"player": 0, "position": (0, 1)},
                {"player": 2, "position": (2, 1)},
                {"player": 3, "position": (3, 1)},
            ],
        }

    def test_shape(self, four_player_state: dict) -> None:
        bs = TestDefaults.SMALL_BOARD_SIZE
        encoder = MicroCivEncoder(board_size=bs, num_players=4)
        spatial, vector = encoder.encode(four_player_state)
        assert spatial.shape == (21, bs, bs)
        assert vector.shape == (64,)

    def test_own_unit_position(self, four_player_state: dict) -> None:
        """Current player (2) has archer at (2,0) -> channel 3+0*3+2 = 5."""
        bs = TestDefaults.SMALL_BOARD_SIZE
        encoder = MicroCivEncoder(board_size=bs, num_players=4)
        spatial, _ = encoder.encode(four_player_state)
        # rel_player=0 (self), archer type_idx=2
        assert spatial[5, 0, 2].item() == pytest.approx(0.6)

    def test_opponent_unit_channels(self, four_player_state: dict) -> None:
        """Player 3 is rel_player=1 from current=2, warrior type_idx=0."""
        bs = TestDefaults.SMALL_BOARD_SIZE
        encoder = MicroCivEncoder(board_size=bs, num_players=4)
        spatial, _ = encoder.encode(four_player_state)
        # rel_player = (3 - 2) % 4 = 1, warrior=0 -> channel 3 + 1*3 + 0 = 6
        assert spatial[6, 0, 3].item() == pytest.approx(0.4)

    def test_own_city_channel(self, four_player_state: dict) -> None:
        """Current player (2) has city at (2,1) -> city_base+0."""
        bs = TestDefaults.SMALL_BOARD_SIZE
        encoder = MicroCivEncoder(board_size=bs, num_players=4)
        spatial, _ = encoder.encode(four_player_state)
        city_base = 3 + 4 * 3  # = 15
        assert spatial[city_base, 1, 2].item() == pytest.approx(1.0)

    def test_opponent_city_channels(self, four_player_state: dict) -> None:
        """Player 0 is rel_player=2 from current=2 (in 4-player).
        City at (0,1) -> city_base+2 = 17."""
        bs = TestDefaults.SMALL_BOARD_SIZE
        encoder = MicroCivEncoder(board_size=bs, num_players=4)
        spatial, _ = encoder.encode(four_player_state)
        city_base = 3 + 4 * 3  # = 15
        # rel_player = (0 - 2) % 4 = 2
        assert spatial[city_base + 2, 1, 0].item() == pytest.approx(1.0)

    def test_visibility_and_bias_channels(self, four_player_state: dict) -> None:
        """Vis and bias are the last two channels (19, 20) for 4 players."""
        bs = TestDefaults.SMALL_BOARD_SIZE
        encoder = MicroCivEncoder(board_size=bs, num_players=4)
        spatial, _ = encoder.encode(four_player_state)
        # No visibility key => all visible
        assert torch.all(spatial[19] == 1.0)
        # Bias channel
        assert torch.all(spatial[20] == 1.0)

    def test_vector_per_opponent_breakdown(self, four_player_state: dict) -> None:
        """With 4 players, per-opponent stats appear at indices 12+."""
        bs = TestDefaults.SMALL_BOARD_SIZE
        encoder = MicroCivEncoder(board_size=bs, num_players=4)
        _, vector = encoder.encode(four_player_state)

        # Own units (player 2): 1 archer
        assert vector[2].item() == pytest.approx(1.0 / 20.0)
        # Total enemy units: 3
        assert vector[3].item() == pytest.approx(3.0 / 20.0)

        # Per-opponent: rel_p=1 is player 3 (1 unit, 1 city)
        assert vector[12].item() == pytest.approx(1.0 / 20.0)
        assert vector[13].item() == pytest.approx(1.0 / 5.0)

        # Per-opponent: rel_p=2 is player 0 (1 unit, 1 city)
        assert vector[14].item() == pytest.approx(1.0 / 20.0)
        assert vector[15].item() == pytest.approx(1.0 / 5.0)

        # Per-opponent: rel_p=3 is player 1 (1 unit, 0 cities)
        assert vector[16].item() == pytest.approx(1.0 / 20.0)
        assert vector[17].item() == pytest.approx(0.0)
