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
        """Spatial tensor has correct shape."""
        spatial, _ = encoder.encode(sample_state)
        assert spatial.shape == (13, 16, 16)

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
        """Cities are correctly encoded."""
        spatial, _ = encoder.encode(sample_state)

        # Own city at (2,2)
        assert spatial[9, 2, 2] == 1.0

        # Enemy city at (13,13)
        assert spatial[10, 13, 13] == 1.0

    def test_constant_channel(self, encoder, sample_state):
        """Constant channel is all ones."""
        spatial, _ = encoder.encode(sample_state)
        assert torch.all(spatial[12] == 1.0)

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
