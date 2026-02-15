"""Unit tests for the scenario engine.

Tests scenario management and analysis including:
- Scenario creation and validation
- ScenarioEngine lifecycle (add, remove, get, list)
- Probability normalization and sampling
- Expected value computation and robustness scoring
- Serialization round-trips
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.strategic.scenarios import Scenario, ScenarioEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def optimistic_scenario() -> Scenario:
    """Create an optimistic scenario."""
    return Scenario(
        name="optimistic",
        parameters={"growth_rate": 0.05, "disruption_prob": 0.1},
        probability=0.3,
        description="High growth, low disruption.",
    )


@pytest.fixture
def pessimistic_scenario() -> Scenario:
    """Create a pessimistic scenario."""
    return Scenario(
        name="pessimistic",
        parameters={"growth_rate": 0.01, "disruption_prob": 0.5},
        probability=0.5,
        description="Low growth, high disruption.",
    )


@pytest.fixture
def neutral_scenario() -> Scenario:
    """Create a neutral scenario."""
    return Scenario(
        name="neutral",
        parameters={"growth_rate": 0.03, "disruption_prob": 0.25},
        probability=0.2,
        description="Moderate assumptions.",
    )


@pytest.fixture
def engine_with_scenarios(
    optimistic_scenario: Scenario,
    pessimistic_scenario: Scenario,
    neutral_scenario: Scenario,
) -> ScenarioEngine:
    """Create a ScenarioEngine populated with three scenarios."""
    engine = ScenarioEngine()
    engine.add_scenario(optimistic_scenario)
    engine.add_scenario(pessimistic_scenario)
    engine.add_scenario(neutral_scenario)
    return engine


# ---------------------------------------------------------------------------
# TestScenario
# ---------------------------------------------------------------------------


class TestScenario:
    """Tests for the Scenario dataclass."""

    def test_creation(self, optimistic_scenario: Scenario) -> None:
        """Test creating a Scenario with all fields."""
        s = optimistic_scenario
        assert s.name == "optimistic"
        assert s.parameters == {"growth_rate": 0.05, "disruption_prob": 0.1}
        assert s.probability == pytest.approx(0.3)
        assert s.description == "High growth, low disruption."

    def test_invalid_probability(self) -> None:
        """Test that invalid probability values raise ValueError."""
        with pytest.raises(ValueError, match="probability"):
            Scenario(name="bad_low", probability=-0.1)

        with pytest.raises(ValueError, match="probability"):
            Scenario(name="bad_high", probability=1.5)

    def test_to_from_dict(self, optimistic_scenario: Scenario) -> None:
        """Test round-trip serialization through dictionary."""
        d = optimistic_scenario.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == "optimistic"
        assert d["probability"] == pytest.approx(0.3)
        assert d["parameters"]["growth_rate"] == pytest.approx(0.05)

        restored = Scenario.from_dict(d)
        assert restored.name == optimistic_scenario.name
        assert restored.probability == pytest.approx(optimistic_scenario.probability)
        assert restored.parameters == optimistic_scenario.parameters
        assert restored.description == optimistic_scenario.description


# ---------------------------------------------------------------------------
# TestScenarioEngine
# ---------------------------------------------------------------------------


class TestScenarioEngine:
    """Tests for ScenarioEngine."""

    def test_empty_engine(self) -> None:
        """Test creating an empty engine with no scenarios."""
        engine = ScenarioEngine()
        assert engine.num_scenarios == 0
        assert engine.scenarios == []

    def test_add_scenario(self, optimistic_scenario: Scenario) -> None:
        """Test adding a scenario increases count and is retrievable."""
        engine = ScenarioEngine()
        engine.add_scenario(optimistic_scenario)

        assert engine.num_scenarios == 1
        assert engine.scenarios[0].name == "optimistic"

        # Duplicate name should raise
        with pytest.raises(ValueError, match="already exists"):
            engine.add_scenario(optimistic_scenario)

    def test_remove_scenario(
        self, engine_with_scenarios: ScenarioEngine
    ) -> None:
        """Test removing a scenario by name."""
        engine = engine_with_scenarios
        assert engine.num_scenarios == 3

        engine.remove_scenario("pessimistic")
        assert engine.num_scenarios == 2
        assert "pessimistic" not in [s.name for s in engine.scenarios]

        # Removing non-existent should raise KeyError
        with pytest.raises(KeyError, match="pessimistic"):
            engine.remove_scenario("pessimistic")

    def test_get_scenario(
        self, engine_with_scenarios: ScenarioEngine
    ) -> None:
        """Test retrieving a scenario by name."""
        engine = engine_with_scenarios

        s = engine.get_scenario("neutral")
        assert s.name == "neutral"
        assert s.probability == pytest.approx(0.2)

        with pytest.raises(KeyError, match="nonexistent"):
            engine.get_scenario("nonexistent")

    def test_normalize_probabilities(
        self, engine_with_scenarios: ScenarioEngine
    ) -> None:
        """Test probability normalization sums to 1.0."""
        engine = engine_with_scenarios
        # Before normalization: 0.3 + 0.5 + 0.2 = 1.0 already
        engine.normalize_probabilities()

        total = sum(s.probability for s in engine.scenarios)
        assert total == pytest.approx(1.0, abs=1e-8)

        # Check ratios are preserved
        s_opt = engine.get_scenario("optimistic")
        s_pes = engine.get_scenario("pessimistic")
        assert s_opt.probability == pytest.approx(0.3, abs=1e-6)
        assert s_pes.probability == pytest.approx(0.5, abs=1e-6)

    def test_normalize_all_zero_probabilities(self) -> None:
        """Test normalization when all probabilities are zero."""
        engine = ScenarioEngine()
        engine.add_scenario(Scenario(name="a", probability=0.0))
        engine.add_scenario(Scenario(name="b", probability=0.0))

        engine.normalize_probabilities()

        # Should assign equal probability
        for s in engine.scenarios:
            assert s.probability == pytest.approx(0.5, abs=1e-8)

    def test_sample_scenario(
        self, engine_with_scenarios: ScenarioEngine
    ) -> None:
        """Test sampling returns a valid scenario with reproducible results."""
        engine = engine_with_scenarios
        engine.normalize_probabilities()

        rng = np.random.default_rng(42)
        sampled = engine.sample_scenario(rng=rng)

        assert isinstance(sampled, Scenario)
        assert sampled.name in {"optimistic", "pessimistic", "neutral"}

        # Sampling from empty engine should raise
        empty_engine = ScenarioEngine()
        with pytest.raises(ValueError, match="no scenarios"):
            empty_engine.sample_scenario()

    def test_expected_value(
        self, engine_with_scenarios: ScenarioEngine
    ) -> None:
        """Test probability-weighted expected value computation."""
        engine = engine_with_scenarios
        engine.normalize_probabilities()

        values = {
            "optimistic": np.array([0.9, 0.8]),
            "pessimistic": np.array([0.3, 0.2]),
            "neutral": np.array([0.6, 0.5]),
        }

        ev = engine.expected_value(values)

        assert ev.shape == (2,)
        # Manual: 0.3*[0.9,0.8] + 0.5*[0.3,0.2] + 0.2*[0.6,0.5]
        expected = 0.3 * np.array([0.9, 0.8]) + 0.5 * np.array([0.3, 0.2]) + 0.2 * np.array([0.6, 0.5])
        np.testing.assert_allclose(ev, expected, atol=1e-6)

    def test_expected_value_missing_scenario(
        self, engine_with_scenarios: ScenarioEngine
    ) -> None:
        """Test expected value raises when no scenarios match."""
        engine = engine_with_scenarios
        with pytest.raises(ValueError, match="No matching"):
            engine.expected_value({"unknown": np.array([1.0])})

    def test_robustness_score(
        self, engine_with_scenarios: ScenarioEngine
    ) -> None:
        """Test robustness score is in [0, 1]."""
        engine = engine_with_scenarios

        # Identical outcomes -> perfectly robust
        identical = {
            "optimistic": np.array([0.5, 0.5]),
            "pessimistic": np.array([0.5, 0.5]),
            "neutral": np.array([0.5, 0.5]),
        }
        score = engine.robustness_score(identical)
        assert score == pytest.approx(1.0)

        # Varied outcomes -> less robust
        varied = {
            "optimistic": np.array([0.9, 0.8]),
            "pessimistic": np.array([0.1, 0.1]),
            "neutral": np.array([0.5, 0.5]),
        }
        score = engine.robustness_score(varied)
        assert 0.0 <= score <= 1.0
        # worst sum = 0.2, best sum = 1.7 -> 0.2/1.7 ~ 0.12
        assert score < 0.5

    def test_to_from_dict(
        self, engine_with_scenarios: ScenarioEngine
    ) -> None:
        """Test round-trip serialization through dictionary."""
        d = engine_with_scenarios.to_dict()

        assert isinstance(d, dict)
        assert "scenarios" in d
        assert len(d["scenarios"]) == 3

        restored = ScenarioEngine.from_dict(d)
        assert restored.num_scenarios == 3

        names = [s.name for s in restored.scenarios]
        assert "optimistic" in names
        assert "pessimistic" in names
        assert "neutral" in names

        # Verify probability is preserved
        s = restored.get_scenario("optimistic")
        assert s.probability == pytest.approx(0.3)
