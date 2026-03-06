"""Unit tests for state encoder."""

import pytest
import torch

from alpha_zero.envs.civ6_encoder import Civ6StateEncoder, EncoderConfig, MicroCivEncoder


class TestMicroCivEncoder:
    """Test suite for MicroCivEncoder."""

    @pytest.fixture
    def encoder(self):
        return MicroCivEncoder(board_size=16)

    @pytest.fixture
    def sample_state(self):
        return {
            "terrain": {
                (0, 0): {"type": "plains"},
                (1, 0): {"type": "hills"},
                (2, 0): {"type": "mountain"},
            },
            "units": [
                {
                    "id": 0,
                    "player": 0,
                    "type": "warrior",
                    "position": (5, 5),
                    "health": 100,
                },
                {
                    "id": 1,
                    "player": 1,
                    "type": "archer",
                    "position": (10, 10),
                    "health": 75,
                },
            ],
            "cities": [
                {"id": 0, "player": 0, "position": (2, 2)},
                {"id": 1, "player": 1, "position": (13, 13)},
            ],
            "turn": 50,
            "current_player": 0,
        }

    def test_spatial_shape(self, encoder, sample_state):
        """Spatial tensor has correct shape (28 channels for N-player support)."""
        spatial, _ = encoder.encode(sample_state)
        assert spatial.shape == (28, 16, 16)

    def test_vector_shape(self, encoder, sample_state):
        """Vector tensor has correct shape."""
        _, vector = encoder.encode(sample_state)
        assert vector.shape == (64,)

    def test_terrain_encoding(self, encoder, sample_state):
        """Terrain is correctly one-hot encoded."""
        spatial, _ = encoder.encode(sample_state)

        # Plains at (0,0) should be channel 0
        assert spatial[0, 0, 0] == 1.0
        assert spatial[1, 0, 0] == 0.0
        assert spatial[2, 0, 0] == 0.0

        # Hills at (1,0) should be channel 1
        assert spatial[0, 0, 1] == 0.0
        assert spatial[1, 0, 1] == 1.0
        assert spatial[2, 0, 1] == 0.0

        # Mountain at (2,0) should be channel 2
        assert spatial[0, 0, 2] == 0.0
        assert spatial[1, 0, 2] == 0.0
        assert spatial[2, 0, 2] == 1.0

    def test_own_unit_encoding(self, encoder, sample_state):
        """Own units are encoded in own unit channels."""
        spatial, _ = encoder.encode(sample_state)

        # Warrior (type 0) at (5,5) with health 100
        assert spatial[3, 5, 5] == 1.0  # channel 3 = own warrior

    def test_enemy_unit_encoding(self, encoder, sample_state):
        """Enemy units are encoded in enemy unit channels."""
        spatial, _ = encoder.encode(sample_state)

        # Enemy archer (type 2) at (10,10) with health 75
        assert spatial[8, 10, 10] == 0.75  # channel 8 = enemy archer

    def test_city_encoding(self, encoder, sample_state):
        """Cities are correctly encoded (updated for 28-channel layout)."""
        spatial, _ = encoder.encode(sample_state)

        # Own city at (2,2) - now at channel 21 (was 9)
        assert spatial[21, 2, 2] == 1.0

        # Enemy city at (13,13) - now at channel 22 (was 10)
        assert spatial[22, 13, 13] == 1.0

    def test_constant_channel(self, encoder, sample_state):
        """Constant/fog channel is all ones (now at channel 27)."""
        spatial, _ = encoder.encode(sample_state)
        assert torch.all(spatial[27] == 1.0)

    def test_encoding_deterministic(self, encoder, sample_state):
        """Encoding is deterministic."""
        s1, v1 = encoder.encode(sample_state)
        s2, v2 = encoder.encode(sample_state)

        assert torch.allclose(s1, s2)
        assert torch.allclose(v1, v2)

    def test_vector_turn_encoding(self, encoder, sample_state):
        """Turn is encoded in vector."""
        _, vector = encoder.encode(sample_state)
        # Turn 50, max 200, so turn/200 = 0.25
        assert abs(vector[0].item() - 0.25) < 0.01

    def test_output_dtypes(self, encoder, sample_state):
        """Output tensors are float32."""
        spatial, vector = encoder.encode(sample_state)
        assert spatial.dtype == torch.float32
        assert vector.dtype == torch.float32

    def test_4player_distinguishes_opponents(self):
        """4-player game encodes each opponent in separate channels."""
        encoder = MicroCivEncoder(board_size=8, num_players=4)

        state = {
            "current_player": 0,
            "terrain": {},
            "units": [
                # Player 0 (own) units → channels 3-5
                {"player": 0, "type": "warrior", "position": (1, 1), "health": 100},
                # Player 1 (opponent 1) units → channels 6-8
                {"player": 1, "type": "settler", "position": (2, 2), "health": 100},
                # Player 2 (opponent 2) units → channels 9-11
                {"player": 2, "type": "archer", "position": (3, 3), "health": 100},
                # Player 3 (opponent 3) units → channels 12-14
                {"player": 3, "type": "warrior", "position": (4, 4), "health": 100},
            ],
            "cities": [],
            "turn": 0,
        }

        spatial, _ = encoder.encode(state)

        # Player 0 (own) warrior at (1,1) → channel 3
        assert spatial[3, 1, 1] == 1.0, "Own warrior should be at channel 3"

        # Player 1 (opponent 1) settler at (2,2) → channel 6+1=7
        assert spatial[7, 2, 2] == 1.0, "Opponent 1 settler should be at channel 7"

        # Player 2 (opponent 2) archer at (3,3) → channel 9+2=11
        assert spatial[11, 3, 3] == 1.0, "Opponent 2 archer should be at channel 11"

        # Player 3 (opponent 3) warrior at (4,4) → channel 12+0=12
        assert spatial[12, 4, 4] == 1.0, "Opponent 3 warrior should be at channel 12"

        # Verify channels 15-20 are unused (only 4 players)
        assert torch.all(spatial[15:21] == 0.0), "Channels 15-20 should be zero (unused)"

    def test_6player_all_channels_used(self):
        """6-player game uses all opponent channels."""
        encoder = MicroCivEncoder(board_size=8, num_players=6)

        state = {
            "current_player": 0,
            "terrain": {},
            "units": [
                {"player": 0, "type": "warrior", "position": (0, 0), "health": 100},
                {"player": 1, "type": "warrior", "position": (1, 1), "health": 100},
                {"player": 2, "type": "warrior", "position": (2, 2), "health": 100},
                {"player": 3, "type": "warrior", "position": (3, 3), "health": 100},
                {"player": 4, "type": "warrior", "position": (4, 4), "health": 100},
                {"player": 5, "type": "warrior", "position": (5, 5), "health": 100},
            ],
            "cities": [],
            "turn": 0,
        }

        spatial, _ = encoder.encode(state)

        # All 6 players should have warriors in their respective channels
        assert spatial[3, 0, 0] == 1.0  # Player 0 (own) → channel 3
        assert spatial[6, 1, 1] == 1.0  # Player 1 → channel 6
        assert spatial[9, 2, 2] == 1.0  # Player 2 → channel 9
        assert spatial[12, 3, 3] == 1.0  # Player 3 → channel 12
        assert spatial[15, 4, 4] == 1.0  # Player 4 → channel 15
        assert spatial[18, 5, 5] == 1.0  # Player 5 → channel 18

    def test_2player_backward_compatibility(self):
        """2-player game uses same channels as legacy 13-channel encoder."""
        encoder = MicroCivEncoder(board_size=8, num_players=2)

        state = {
            "current_player": 0,
            "terrain": {(0, 0): {"type": "plains"}},
            "units": [
                {"player": 0, "type": "warrior", "position": (1, 1), "health": 100},
                {"player": 1, "type": "archer", "position": (2, 2), "health": 75},
            ],
            "cities": [
                {"player": 0, "position": (3, 3)},
                {"player": 1, "position": (4, 4)},
            ],
            "turn": 0,
        }

        spatial, _ = encoder.encode(state)

        # Verify 2-player uses same channels 0-12 as before
        assert spatial[0, 0, 0] == 1.0  # Terrain (plains)
        assert spatial[3, 1, 1] == 1.0  # Own warrior → channel 3
        assert spatial[8, 2, 2] == 0.75  # Enemy archer → channel 8
        assert spatial[21, 3, 3] == 1.0  # Own city → channel 21 (updated)
        assert spatial[22, 4, 4] == 1.0  # Enemy city → channel 22 (updated)

        # Channels 9-20 should be zero (unused opponents in 2-player)
        assert torch.all(spatial[9:21] == 0.0), "Channels 9-20 should be zero in 2-player"

    def test_relative_perspective_rotation(self):
        """Relative player indexing rotates perspective based on current_player."""
        encoder = MicroCivEncoder(board_size=8, num_players=4)

        # Same game state, but viewed from Player 2's perspective
        state = {
            "current_player": 2,  # Changed from 0 to 2
            "terrain": {},
            "units": [
                {"player": 0, "type": "warrior", "position": (0, 0), "health": 100},
                {"player": 1, "type": "warrior", "position": (1, 1), "health": 100},
                {"player": 2, "type": "warrior", "position": (2, 2), "health": 100},
                {"player": 3, "type": "warrior", "position": (3, 3), "health": 100},
            ],
            "cities": [],
            "turn": 0,
        }

        spatial, _ = encoder.encode(state)

        # From Player 2's perspective:
        # Player 2 (self) → relative = (2-2)%4 = 0 → channels 3-5
        # Player 3 (next in turn) → relative = (3-2)%4 = 1 → channels 6-8
        # Player 0 (wraparound) → relative = (0-2)%4 = 2 → channels 9-11
        # Player 1 → relative = (1-2)%4 = 3 → channels 12-14

        assert spatial[3, 2, 2] == 1.0, "Player 2 (self) should be at channel 3"
        assert spatial[6, 3, 3] == 1.0, "Player 3 (next) should be at channel 6"
        assert spatial[9, 0, 0] == 1.0, "Player 0 (wraparound) should be at channel 9"
        assert spatial[12, 1, 1] == 1.0, "Player 1 should be at channel 12"


class TestCiv6StateEncoder:
    """Test suite for full Civ6StateEncoder."""

    @pytest.fixture
    def encoder(self):
        config = EncoderConfig(board_size=16)
        return Civ6StateEncoder(config)

    @pytest.fixture
    def sample_state(self):
        return {
            "terrain": {
                (0, 0): {"type": "plains", "features": ["hills"], "yields": {"food": 2}},
                (1, 1): {"type": "grassland", "features": ["forest"]},
            },
            "units": [
                {
                    "player": 0,
                    "type": "warrior",
                    "position": (3, 3),
                    "health": 100,
                    "strength": 20,
                },
            ],
            "cities": [
                {
                    "player": 0,
                    "position": (5, 5),
                    "population": 10,
                    "production": 20,
                    "defense": 30,
                },
            ],
            "visibility": {(3, 3): "visible", (0, 0): "seen"},
            "turn": 100,
            "era": 2,
            "current_player": 0,
            "gold": 500,
            "gold_per_turn": 25,
            "science_per_turn": 30,
            "culture_per_turn": 15,
            "resources": {"iron": 5, "horses": 3},
            "technologies": [0, 1, 5, 10],
            "civics": [0, 2],
            "military_strength": 100,
        }

    def test_full_spatial_shape(self, encoder, sample_state):
        """Full encoder produces 84 channels."""
        spatial, _ = encoder.encode(sample_state)
        assert spatial.shape == (84, 16, 16)

    def test_full_vector_shape(self, encoder, sample_state):
        """Full encoder produces 512-dim vector."""
        _, vector = encoder.encode(sample_state)
        assert vector.shape == (512,)

    def test_terrain_channels(self, encoder, sample_state):
        """Terrain uses channels 0-6."""
        spatial, _ = encoder.encode(sample_state)

        # Plains at (0,0) = channel 0
        assert spatial[0, 0, 0] == 1.0

        # Grassland at (1,1) = channel 1
        assert spatial[1, 1, 1] == 1.0

    def test_feature_channels(self, encoder, sample_state):
        """Features use channels 7-14."""
        spatial, _ = encoder.encode(sample_state)

        # Hills at (0,0) = channel 11
        assert spatial[11, 0, 0] == 1.0

        # Forest at (1,1) = channel 7
        assert spatial[7, 1, 1] == 1.0

    def test_fog_of_war_channels(self, encoder, sample_state):
        """Fog of war uses channels 63-65."""
        spatial, _ = encoder.encode(sample_state)

        # Visible at (3,3) = channel 63
        assert spatial[63, 3, 3] == 1.0

        # Seen at (0,0) = channel 64
        assert spatial[64, 0, 0] == 1.0

    def test_city_encoding_channels(self, encoder, sample_state):
        """Own cities use channels 51-54."""
        spatial, _ = encoder.encode(sample_state)

        # City presence at (5,5) = channel 51
        assert spatial[51, 5, 5] == 1.0

        # Population 10/30 ≈ 0.33 at channel 52
        assert abs(spatial[52, 5, 5].item() - 10 / 30) < 0.01

    def test_vector_era_encoding(self, encoder, sample_state):
        """Era is one-hot encoded in vector."""
        _, vector = encoder.encode(sample_state)

        # Era 2 should have vector[12] = 1.0
        assert vector[12].item() == 1.0

    def test_vector_economy_encoding(self, encoder, sample_state):
        """Economic values are encoded in vector."""
        _, vector = encoder.encode(sample_state)

        # Gold 500/1000 = 0.5
        assert abs(vector[300].item() - 0.5) < 0.01

        # GPT 25/100 = 0.25
        assert abs(vector[301].item() - 0.25) < 0.01
