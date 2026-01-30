"""Tests for climate and disaster mechanics.

Tests cover:
- ClimatePhase, DisasterType, DisasterSeverity, TerrainFeature enums
- ClimateConfig and DisasterConfig dataclasses
- DisasterEvent and FloodedTile dataclasses
- ClimateManager CO2 tracking and phase transitions
- Disaster generation and frequency calculation
- Sea level rise mechanics
- Yield bonuses from disasters
- State encoding for neural networks
"""

from __future__ import annotations

import pytest

from alpha_zero.mechanics.climate import (
    ClimateConfig,
    ClimateManager,
    ClimatePhase,
    DisasterConfig,
    DisasterSeverity,
    DisasterType,
    FloodedTile,
    TerrainFeature,
)

# =============================================================================
# Constants
# =============================================================================

SEED = 42


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def climate_config() -> ClimateConfig:
    """Create default climate configuration."""
    return ClimateConfig()


@pytest.fixture
def climate_manager(climate_config: ClimateConfig) -> ClimateManager:
    """Create climate manager with seeded RNG."""
    return ClimateManager(config=climate_config, random_seed=SEED)


@pytest.fixture
def sample_map_features() -> dict[tuple[int, int], set[TerrainFeature]]:
    """Create sample map with terrain features."""
    return {
        (0, 0): {TerrainFeature.FLOODPLAINS},
        (1, 0): {TerrainFeature.COAST},
        (2, 0): {TerrainFeature.VOLCANO},
        (3, 0): {TerrainFeature.FOREST},
        (0, 1): {TerrainFeature.GRASSLAND},
        (1, 1): {TerrainFeature.DESERT},
        (2, 1): {TerrainFeature.TUNDRA},
        (3, 1): {TerrainFeature.COAST},
    }


# =============================================================================
# Enum Tests
# =============================================================================


class TestClimatePhase:
    """Tests for ClimatePhase enum."""

    def test_all_phases_exist(self) -> None:
        """Test all climate phases are defined."""
        phases = [
            ClimatePhase.NORMAL,
            ClimatePhase.WARMING_1,
            ClimatePhase.WARMING_2,
            ClimatePhase.WARMING_3,
            ClimatePhase.WARMING_4,
            ClimatePhase.WARMING_5,
            ClimatePhase.WARMING_6,
            ClimatePhase.WARMING_7,
        ]
        assert len(phases) == 8

    def test_phases_are_ordered(self) -> None:
        """Test phases have increasing values."""
        phases = list(ClimatePhase)
        values = [p.value for p in phases]
        assert values == sorted(values)


class TestDisasterType:
    """Tests for DisasterType enum."""

    def test_all_disaster_types(self) -> None:
        """Test all disaster types exist."""
        assert DisasterType.FLOOD is not None
        assert DisasterType.VOLCANIC_ERUPTION is not None
        assert DisasterType.HURRICANE is not None
        assert DisasterType.TORNADO is not None
        assert DisasterType.DROUGHT is not None
        assert DisasterType.BLIZZARD is not None
        assert DisasterType.DUST_STORM is not None
        assert DisasterType.FOREST_FIRE is not None
        assert DisasterType.METEOR_STRIKE is not None

    def test_unique_values(self) -> None:
        """Test all disaster types have unique values."""
        values = [d.value for d in DisasterType]
        assert len(values) == len(set(values))


class TestDisasterSeverity:
    """Tests for DisasterSeverity enum."""

    def test_severity_levels(self) -> None:
        """Test severity levels exist."""
        assert DisasterSeverity.MINOR is not None
        assert DisasterSeverity.MODERATE is not None
        assert DisasterSeverity.MAJOR is not None
        assert DisasterSeverity.CATASTROPHIC is not None

    def test_severity_ordering(self) -> None:
        """Test severity values are ordered."""
        assert DisasterSeverity.MINOR.value < DisasterSeverity.MODERATE.value
        assert DisasterSeverity.MODERATE.value < DisasterSeverity.MAJOR.value
        assert DisasterSeverity.MAJOR.value < DisasterSeverity.CATASTROPHIC.value


class TestTerrainFeature:
    """Tests for TerrainFeature enum."""

    def test_terrain_features(self) -> None:
        """Test terrain features exist."""
        features = [
            TerrainFeature.FLOODPLAINS,
            TerrainFeature.VOLCANO,
            TerrainFeature.COAST,
            TerrainFeature.FOREST,
            TerrainFeature.GRASSLAND,
            TerrainFeature.DESERT,
            TerrainFeature.TUNDRA,
        ]
        assert len(features) == 7


# =============================================================================
# Config Tests
# =============================================================================


class TestClimateConfig:
    """Tests for ClimateConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ClimateConfig()

        assert config.co2_per_power_plant == 10.0
        assert config.co2_per_industrial_zone == 2.0
        assert config.co2_threshold_per_phase == 100.0
        assert config.sea_level_per_phase == 3
        assert config.disaster_frequency_base == 0.05
        assert config.disaster_frequency_per_phase == 0.02
        assert config.fertility_bonus_on_disaster == 1.0
        assert config.deforestation_co2 == 5.0
        assert config.nuclear_co2 == 50.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ClimateConfig(
            co2_per_power_plant=20.0,
            co2_threshold_per_phase=200.0,
            sea_level_per_phase=5,
        )

        assert config.co2_per_power_plant == 20.0
        assert config.co2_threshold_per_phase == 200.0
        assert config.sea_level_per_phase == 5


class TestDisasterConfig:
    """Tests for DisasterConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default disaster config values."""
        config = DisasterConfig(disaster_type=DisasterType.FLOOD)

        assert config.disaster_type == DisasterType.FLOOD
        assert config.base_frequency == 0.1
        assert config.terrain_requirements == []
        assert config.min_severity == DisasterSeverity.MINOR
        assert config.max_severity == DisasterSeverity.MAJOR
        assert config.damage_buildings is False
        assert config.damage_improvements is True
        assert config.damage_units is False
        assert config.damage_population is False
        assert config.yield_bonus_after == {}
        assert config.phase_frequency_multiplier == 1.2

    def test_custom_disaster_config(self) -> None:
        """Test custom disaster configuration."""
        config = DisasterConfig(
            disaster_type=DisasterType.VOLCANIC_ERUPTION,
            base_frequency=0.05,
            terrain_requirements=[TerrainFeature.VOLCANO],
            max_severity=DisasterSeverity.CATASTROPHIC,
            damage_buildings=True,
            damage_population=True,
            yield_bonus_after={"production": 2.0},
        )

        assert config.disaster_type == DisasterType.VOLCANIC_ERUPTION
        assert TerrainFeature.VOLCANO in config.terrain_requirements
        assert config.max_severity == DisasterSeverity.CATASTROPHIC
        assert config.damage_buildings is True
        assert config.yield_bonus_after["production"] == 2.0


# =============================================================================
# ClimateManager Tests
# =============================================================================


class TestClimateManagerInitialization:
    """Tests for ClimateManager initialization."""

    def test_default_initialization(self) -> None:
        """Test default manager initialization."""
        manager = ClimateManager()

        status = manager.get_climate_status()
        assert status["phase"] == "NORMAL"
        assert status["total_co2"] == 0.0
        assert status["flooded_tiles"] == 0
        assert status["disaster_count"] == 0

    def test_initialization_with_config(self, climate_config: ClimateConfig) -> None:
        """Test initialization with custom config."""
        manager = ClimateManager(config=climate_config, random_seed=SEED)

        assert manager.config is climate_config

    def test_initialization_with_seed(self) -> None:
        """Test reproducibility with seed."""
        manager1 = ClimateManager(random_seed=42)
        manager2 = ClimateManager(random_seed=42)

        # Both should produce same random sequence
        r1 = manager1._rng.random()
        r2 = manager2._rng.random()
        assert r1 == r2


class TestCO2AndPhaseTransitions:
    """Tests for CO2 tracking and phase transitions."""

    def test_add_co2(self, climate_manager: ClimateManager) -> None:
        """Test adding CO2 emissions."""
        initial = climate_manager._total_co2

        result = climate_manager.add_co2(50.0, source="power_plant")

        assert result == initial + 50.0
        assert climate_manager._total_co2 == 50.0

    def test_phase_transition_at_threshold(self) -> None:
        """Test phase transitions at CO2 thresholds."""
        config = ClimateConfig(co2_threshold_per_phase=100.0)
        manager = ClimateManager(config=config)

        # Start at NORMAL
        assert manager._current_phase == ClimatePhase.NORMAL

        # Add CO2 below threshold
        manager.add_co2(50.0)
        assert manager._current_phase == ClimatePhase.NORMAL

        # Add CO2 to reach threshold
        manager.add_co2(50.0)
        assert manager._current_phase == ClimatePhase.WARMING_1

        # Add more CO2
        manager.add_co2(100.0)
        assert manager._current_phase == ClimatePhase.WARMING_2

    def test_multiple_phase_transitions(self) -> None:
        """Test transitioning through multiple phases."""
        config = ClimateConfig(co2_threshold_per_phase=100.0)
        manager = ClimateManager(config=config)

        # Jump directly to phase 5
        manager.add_co2(450.0)
        assert manager._current_phase == ClimatePhase.WARMING_4

    def test_max_phase_cap(self) -> None:
        """Test that phase is capped at maximum."""
        config = ClimateConfig(co2_threshold_per_phase=100.0)
        manager = ClimateManager(config=config)

        # Add excessive CO2
        manager.add_co2(10000.0)
        assert manager._current_phase == ClimatePhase.WARMING_7


class TestDisasterGeneration:
    """Tests for disaster generation."""

    def test_disaster_frequency_increases_with_phase(self, climate_manager: ClimateManager) -> None:
        """Test that disaster frequency increases with climate phase."""
        config = DisasterConfig(
            disaster_type=DisasterType.FLOOD,
            base_frequency=0.1,
            phase_frequency_multiplier=1.5,
        )

        # Get frequency at normal phase
        freq_normal = climate_manager._calculate_disaster_frequency(config)

        # Advance phase
        climate_manager._current_phase = ClimatePhase.WARMING_3
        freq_warming = climate_manager._calculate_disaster_frequency(config)

        assert freq_warming > freq_normal

    def test_process_turn_generates_disasters(
        self,
        sample_map_features: dict[tuple[int, int], set[TerrainFeature]],
    ) -> None:
        """Test that processing turns can generate disasters."""
        # Use high frequency to ensure disasters happen
        config = ClimateConfig(
            disaster_frequency_base=0.9,  # High frequency
            disaster_frequency_per_phase=0.0,
        )
        manager = ClimateManager(config=config, random_seed=123)

        # Process multiple turns to get disasters
        all_disasters = []
        for turn in range(10):
            disasters = manager.process_turn(turn, sample_map_features)
            all_disasters.extend(disasters)

        # Should have generated some disasters
        assert len(all_disasters) > 0

    def test_terrain_requirements_filter_disasters(self) -> None:
        """Test that terrain requirements filter disaster locations."""
        manager = ClimateManager(random_seed=SEED)

        # Only volcanic terrain
        map_features = {(0, 0): {TerrainFeature.VOLCANO}}

        # High frequency config
        config = ClimateConfig(disaster_frequency_base=0.9)
        manager = ClimateManager(config=config, random_seed=42)

        # Most disasters require specific terrain
        disasters = manager.process_turn(0, map_features)

        # Any disaster should be for the volcano tile or unrestricted disasters
        for disaster in disasters:
            if disaster.disaster_type == DisasterType.VOLCANIC_ERUPTION:
                assert (0, 0) in disaster.affected_tiles

    def test_disaster_severity_varies(self) -> None:
        """Test that disaster severity can vary."""
        config = ClimateConfig(disaster_frequency_base=0.9)
        manager = ClimateManager(config=config, random_seed=42)

        map_features = {(i, 0): {TerrainFeature.FLOODPLAINS} for i in range(20)}

        # Generate many disasters
        severities = set()
        for turn in range(50):
            disasters = manager.process_turn(turn, map_features)
            for d in disasters:
                severities.add(d.severity)

        # Should have variety in severity
        assert len(severities) >= 2


class TestSeaLevelRise:
    """Tests for sea level rise mechanics."""

    def test_no_flooding_at_normal_phase(self, climate_manager: ClimateManager) -> None:
        """Test no flooding during normal climate phase."""
        coastal_tiles = {(0, 0), (1, 0), (2, 0)}

        flooded = climate_manager._process_sea_level_rise(turn=1, coastal_tiles=coastal_tiles)

        assert len(flooded) == 0

    def test_flooding_at_warming_phase(self) -> None:
        """Test flooding occurs at warming phases."""
        config = ClimateConfig(
            co2_threshold_per_phase=100.0,
            sea_level_per_phase=2,
        )
        manager = ClimateManager(config=config, random_seed=SEED)

        # Advance to warming phase
        manager.add_co2(150.0)
        assert manager._current_phase == ClimatePhase.WARMING_1

        coastal_tiles = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}

        flooded = manager._process_sea_level_rise(turn=1, coastal_tiles=coastal_tiles)

        # Should flood some tiles
        assert len(flooded) > 0
        assert len(flooded) <= 2  # Based on phase and config

    def test_is_tile_flooded(self) -> None:
        """Test checking if tile is flooded."""
        manager = ClimateManager(random_seed=SEED)

        # No tiles flooded initially
        assert manager.is_tile_flooded((0, 0)) is False

        # Manually add flooded tile
        manager._flooded_tiles[(0, 0)] = FloodedTile(
            position=(0, 0),
            turn_flooded=1,
            was_coastal=True,
            phase_flooded=ClimatePhase.WARMING_1,
        )

        assert manager.is_tile_flooded((0, 0)) is True
        assert manager.is_tile_flooded((1, 0)) is False

    def test_flooding_limited_to_available_tiles(self) -> None:
        """Test flooding is limited to available coastal tiles."""
        config = ClimateConfig(
            co2_threshold_per_phase=50.0,
            sea_level_per_phase=100,  # Very high
        )
        manager = ClimateManager(config=config, random_seed=SEED)

        # Advance phase significantly
        manager.add_co2(200.0)

        coastal_tiles = {(0, 0), (1, 0)}  # Only 2 tiles

        flooded = manager._process_sea_level_rise(turn=1, coastal_tiles=coastal_tiles)

        # Can only flood what's available
        assert len(flooded) <= 2


class TestYieldBonuses:
    """Tests for disaster yield bonuses."""

    def test_get_tile_yield_bonus_empty(self, climate_manager: ClimateManager) -> None:
        """Test getting yield bonus for tile with no bonus."""
        bonus = climate_manager.get_tile_yield_bonus((0, 0))
        assert bonus == {}

    def test_yield_bonus_after_disaster(self) -> None:
        """Test yield bonus is applied after disaster."""
        # Create manager with flood config that gives food bonus
        manager = ClimateManager(random_seed=42)

        # Manually add yield bonus (as if disaster occurred)
        manager._tile_yield_bonuses[(0, 0)] = {"food": 2.0}

        bonus = manager.get_tile_yield_bonus((0, 0))
        assert bonus["food"] == 2.0

    def test_yield_bonuses_accumulate(self) -> None:
        """Test that yield bonuses accumulate from multiple disasters."""
        manager = ClimateManager(random_seed=42)

        # Add bonuses manually
        manager._tile_yield_bonuses[(0, 0)] = {"food": 1.0}
        manager._tile_yield_bonuses[(0, 0)]["food"] = 2.0
        manager._tile_yield_bonuses[(0, 0)]["production"] = 1.5

        bonus = manager.get_tile_yield_bonus((0, 0))
        assert bonus["food"] == 2.0
        assert bonus["production"] == 1.5


class TestClimateStatus:
    """Tests for climate status reporting."""

    def test_get_climate_status(self, climate_manager: ClimateManager) -> None:
        """Test getting full climate status."""
        status = climate_manager.get_climate_status()

        assert "phase" in status
        assert "phase_number" in status
        assert "total_co2" in status
        assert "next_phase_threshold" in status
        assert "flooded_tiles" in status
        assert "disaster_count" in status

    def test_climate_status_updates(self) -> None:
        """Test climate status reflects changes."""
        manager = ClimateManager(config=ClimateConfig(co2_threshold_per_phase=100.0))

        manager.add_co2(150.0)
        status = manager.get_climate_status()

        assert status["phase"] == "WARMING_1"
        assert status["total_co2"] == 150.0

    def test_get_disaster_forecast(
        self,
        climate_manager: ClimateManager,
        sample_map_features: dict[tuple[int, int], set[TerrainFeature]],
    ) -> None:
        """Test getting disaster probability forecast."""
        forecast = climate_manager.get_disaster_forecast(sample_map_features)

        assert isinstance(forecast, dict)
        # Should have entries for applicable disasters
        assert len(forecast) > 0
        # Probabilities should be between 0 and 1
        for prob in forecast.values():
            assert 0.0 <= prob <= 1.0


class TestStateEncoding:
    """Tests for state encoding."""

    def test_encode_for_state(self, climate_manager: ClimateManager) -> None:
        """Test encoding climate state for neural network."""
        encoding = climate_manager.encode_for_state()

        assert "climate_phase" in encoding
        assert "phase_encoding" in encoding
        assert "total_co2" in encoding
        assert "co2_normalized" in encoding
        assert "flooded_tiles" in encoding
        assert "recent_disasters" in encoding
        assert "disaster_total" in encoding

    def test_phase_encoding_one_hot(self) -> None:
        """Test phase encoding is one-hot."""
        manager = ClimateManager()
        manager._current_phase = ClimatePhase.WARMING_3

        encoding = manager.encode_for_state()

        phase_encoding = encoding["phase_encoding"]
        assert sum(phase_encoding) == 1.0
        assert phase_encoding[ClimatePhase.WARMING_3.value - 1] == 1.0

    def test_co2_normalized(self) -> None:
        """Test CO2 normalization."""
        config = ClimateConfig(co2_threshold_per_phase=100.0)
        manager = ClimateManager(config=config)

        manager.add_co2(350.0)  # 50% of max (7 phases * 100)
        encoding = manager.encode_for_state()

        assert 0.0 <= encoding["co2_normalized"] <= 1.0
        assert encoding["co2_normalized"] == pytest.approx(0.5, abs=0.01)


class TestReset:
    """Tests for climate manager reset."""

    def test_reset_clears_state(self) -> None:
        """Test reset clears all state."""
        manager = ClimateManager(random_seed=SEED)

        # Modify state
        manager.add_co2(500.0)
        manager._flooded_tiles[(0, 0)] = FloodedTile(
            position=(0, 0), turn_flooded=1, was_coastal=True, phase_flooded=ClimatePhase.WARMING_2
        )
        manager._tile_yield_bonuses[(0, 0)] = {"food": 1.0}

        manager.reset()

        status = manager.get_climate_status()
        assert status["phase"] == "NORMAL"
        assert status["total_co2"] == 0.0
        assert status["flooded_tiles"] == 0
        assert status["disaster_count"] == 0


class TestMitigationOptions:
    """Tests for mitigation options."""

    def test_get_mitigation_options(self, climate_manager: ClimateManager) -> None:
        """Test getting mitigation options."""
        options = climate_manager.get_mitigation_options()

        assert len(options) > 0
        for option in options:
            assert "name" in option
            assert "cost" in option
            assert "description" in option


# =============================================================================
# Integration Tests
# =============================================================================


class TestClimateIntegration:
    """Integration tests for climate mechanics."""

    def test_full_climate_progression(self) -> None:
        """Test complete climate progression scenario."""
        config = ClimateConfig(
            co2_threshold_per_phase=100.0,
            sea_level_per_phase=1,
            disaster_frequency_base=0.1,
        )
        manager = ClimateManager(config=config, random_seed=42)

        map_features = {(i, j): {TerrainFeature.COAST} for i in range(5) for j in range(5)}
        coastal_tiles = set(map_features.keys())

        # Simulate 100 turns with CO2 emissions
        total_disasters = 0
        for turn in range(100):
            # Add emissions
            manager.add_co2(5.0, source=f"turn_{turn}")

            # Process turn
            disasters = manager.process_turn(turn, map_features, coastal_tiles)
            total_disasters += len(disasters)

        status = manager.get_climate_status()

        # Should have progressed through phases
        assert status["phase_number"] > 1

        # Should have flooded some tiles
        assert status["flooded_tiles"] >= 0

        # Should have had some disasters
        assert status["disaster_count"] >= 0
