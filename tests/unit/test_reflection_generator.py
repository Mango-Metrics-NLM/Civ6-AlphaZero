"""Tests for the reflection generator module."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from alpha_zero.core.reflection_generator import (
    DefaultGameStateDescriber,
    GeneratedCritique,
    ReflectionGenerator,
    ReflectionGeneratorConfig,
    create_reflection_generator,
)
from alpha_zero.distributed.shared_storage import GameTrajectory, TrainingExample


class TestReflectionGeneratorConfig:
    """Tests for ReflectionGeneratorConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = ReflectionGeneratorConfig()

        assert config.llm_provider == "auto"
        assert config.temperature == 0.3
        assert config.max_tokens == 500
        assert config.max_trajectory_states == 10
        assert config.cache_critiques is True
        assert config.use_fallback_on_failure is True

    def test_custom_values(self):
        """Should accept custom values."""
        config = ReflectionGeneratorConfig(
            temperature=0.5,
            max_tokens=1000,
            cache_critiques=False,
        )

        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.cache_critiques is False

    def test_from_env(self):
        """Should load from environment variables."""
        with patch.dict(os.environ, {
            "CIV6AZ_REFLECTION_LLM_TEMP": "0.5",
            "CIV6AZ_REFLECTION_LLM_MAX_TOKENS": "1000",
            "CIV6AZ_REFLECTION_CACHE": "false",
        }):
            config = ReflectionGeneratorConfig.from_env()

            assert config.temperature == 0.5
            assert config.max_tokens == 1000
            assert config.cache_critiques is False


class TestGeneratedCritique:
    """Tests for GeneratedCritique dataclass."""

    def test_default_values(self):
        """Should have empty defaults."""
        critique = GeneratedCritique()

        assert critique.summary == ""
        assert critique.key_mistake == ""
        assert critique.mistake_action is None
        assert critique.confidence == 0.5

    def test_full_critique_property(self):
        """Should combine all fields into full critique."""
        critique = GeneratedCritique(
            summary="Lost due to weak defense",
            key_mistake="Did not build enough units",
            suggested_alternative="Build warriors early",
            strategic_lesson="Defense is important",
        )

        full = critique.full_critique

        assert "Summary: Lost due to weak defense" in full
        assert "Key Mistake: Did not build enough units" in full
        assert "Alternative: Build warriors early" in full
        assert "Lesson: Defense is important" in full

    def test_full_critique_fallback(self):
        """Should use raw response if no structured content."""
        critique = GeneratedCritique(raw_response="Some raw response")

        assert critique.full_critique == "Some raw response"


class TestDefaultGameStateDescriber:
    """Tests for DefaultGameStateDescriber."""

    @pytest.fixture
    def describer(self):
        """Create describer for testing."""
        return DefaultGameStateDescriber()

    @pytest.fixture
    def sample_obs(self):
        """Create sample observation."""
        return {
            "vector": np.array([0.5, 0.3, 0.4, 0.3, 0.6, 0.4] + [0.0] * 26, dtype=np.float32),
            "spatial": np.random.randn(12, 8, 8).astype(np.float32),
        }

    def test_describe_state(self, describer, sample_obs):
        """Should describe state as text."""
        description = describer.describe_state(sample_obs)

        assert isinstance(description, str)
        assert len(description) > 0
        assert "Turn" in description or "Units" in description

    def test_describe_state_empty_vector(self, describer):
        """Should handle empty observation."""
        obs = {}  # No vector key

        description = describer.describe_state(obs)

        assert "Unknown" in description

    def test_describe_action_without_action_space(self, describer):
        """Should return basic action description without action space."""
        description = describer.describe_action(42)

        assert "42" in description

    def test_describe_trajectory(self, describer):
        """Should describe trajectory as text."""
        examples = [
            TrainingExample(
                spatial=np.random.randn(12, 8, 8).astype(np.float32),
                vector=np.random.randn(32).astype(np.float32),
                action_mask=np.ones(100, dtype=np.float32),
                policy=np.zeros(100, dtype=np.float32),
                value=0.0,
            )
            for _ in range(20)
        ]
        # Set one-hot policy
        for i, ex in enumerate(examples):
            ex.policy[i % 100] = 1.0

        trajectory = GameTrajectory(
            examples=examples,
            game_id=1,
            game_length=20,
            winner=1,
        )

        description = describer.describe_trajectory(trajectory)

        assert "Game 1" in description
        assert "20 turns" in description
        assert "Key moments" in description

    def test_describe_empty_trajectory(self, describer):
        """Should handle empty trajectory."""
        trajectory = GameTrajectory(examples=[])

        description = describer.describe_trajectory(trajectory)

        assert "Empty" in description


class TestReflectionGenerator:
    """Tests for ReflectionGenerator."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ReflectionGeneratorConfig(
            cache_critiques=True,
            use_fallback_on_failure=True,
        )

    @pytest.fixture
    def sample_trajectory(self):
        """Create sample trajectory."""
        examples = [
            TrainingExample(
                spatial=np.random.randn(12, 8, 8).astype(np.float32),
                vector=np.random.randn(32).astype(np.float32),
                action_mask=np.ones(100, dtype=np.float32),
                policy=np.zeros(100, dtype=np.float32),
                value=0.0,
            )
            for _ in range(10)
        ]
        for i, ex in enumerate(examples):
            ex.policy[i % 100] = 1.0

        return GameTrajectory(
            examples=examples,
            game_id=1,
            game_length=10,
            winner=1,
        )

    @pytest.fixture
    def mock_llm_response(self):
        """Create mock LLM response."""
        response = MagicMock()
        response.content = """
SUMMARY: Lost due to insufficient defense

KEY_MISTAKE: Did not build enough military units early in the game

ALTERNATIVE: Prioritize warrior production in the first 10 turns

LESSON: Early military investment prevents mid-game collapse
"""
        return response

    @pytest.fixture
    def mock_llm_client(self, mock_llm_response):
        """Create mock LLM client."""
        client = MagicMock()
        client.generate = AsyncMock(return_value=mock_llm_response)
        return client

    @pytest.mark.asyncio
    async def test_generate_critique_with_llm(
        self, config, sample_trajectory, mock_llm_client
    ):
        """Should generate critique using LLM."""
        generator = ReflectionGenerator(config, llm_client=mock_llm_client)

        critique = await generator.generate_critique(sample_trajectory)

        assert isinstance(critique, GeneratedCritique)
        assert "defense" in critique.summary.lower() or "lost" in critique.summary.lower()
        mock_llm_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_critique_parses_response(
        self, config, sample_trajectory, mock_llm_client
    ):
        """Should parse LLM response into structured critique."""
        generator = ReflectionGenerator(config, llm_client=mock_llm_client)

        critique = await generator.generate_critique(sample_trajectory)

        assert critique.summary != ""
        assert critique.key_mistake != ""
        assert critique.suggested_alternative != ""
        assert critique.strategic_lesson != ""
        assert critique.confidence > 0

    def test_generate_critique_sync(
        self, config, sample_trajectory, mock_llm_client
    ):
        """Should work with sync wrapper."""
        generator = ReflectionGenerator(config, llm_client=mock_llm_client)

        critique = generator.generate_critique_sync(sample_trajectory)

        assert isinstance(critique, GeneratedCritique)
        mock_llm_client.generate.assert_called()

    @pytest.mark.asyncio
    async def test_caching(self, config, sample_trajectory, mock_llm_client):
        """Should cache critiques."""
        generator = ReflectionGenerator(config, llm_client=mock_llm_client)

        # First call
        critique1 = await generator.generate_critique(sample_trajectory)

        # Second call should use cache
        critique2 = await generator.generate_critique(sample_trajectory)

        assert critique1.summary == critique2.summary
        # LLM should only be called once
        assert mock_llm_client.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self, config, sample_trajectory):
        """Should use fallback when LLM fails."""
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(side_effect=Exception("LLM error"))

        generator = ReflectionGenerator(config, llm_client=mock_client)

        critique = await generator.generate_critique(sample_trajectory)

        assert isinstance(critique, GeneratedCritique)
        assert critique.confidence == 0.3  # Fallback confidence
        assert "Fallback" in critique.raw_response

    @pytest.mark.asyncio
    async def test_fallback_disabled(self, sample_trajectory):
        """Should raise error when fallback is disabled."""
        config = ReflectionGeneratorConfig(use_fallback_on_failure=False)
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(side_effect=Exception("LLM error"))

        generator = ReflectionGenerator(config, llm_client=mock_client)

        with pytest.raises(Exception):
            await generator.generate_critique(sample_trajectory)

    def test_clear_cache(self, config, mock_llm_client):
        """Should clear cache."""
        generator = ReflectionGenerator(config, llm_client=mock_llm_client)
        generator._cache["test_key"] = GeneratedCritique(summary="cached")

        generator.clear_cache()

        assert len(generator._cache) == 0


class TestResponseParsing:
    """Tests for LLM response parsing."""

    @pytest.fixture
    def generator(self):
        """Create generator for testing."""
        return ReflectionGenerator()

    def test_parse_complete_response(self, generator):
        """Should parse complete structured response."""
        response = """
SUMMARY: The player lost due to weak early game

KEY_MISTAKE: Failed to expand quickly enough

ALTERNATIVE: Build a settler within the first 20 turns

LESSON: Early expansion is crucial for late-game success
"""
        critique = generator._parse_response(response)

        assert "weak early game" in critique.summary
        assert "expand" in critique.key_mistake.lower()
        assert "settler" in critique.suggested_alternative.lower()
        assert "expansion" in critique.strategic_lesson.lower()
        assert critique.confidence == 1.0

    def test_parse_partial_response(self, generator):
        """Should handle partial response."""
        response = """
SUMMARY: Lost the game

KEY_MISTAKE: Unknown tactical error
"""
        critique = generator._parse_response(response)

        assert critique.summary != ""
        assert critique.key_mistake != ""
        assert critique.suggested_alternative == ""
        assert critique.confidence == 0.5  # 2/4 fields

    def test_parse_unstructured_response(self, generator):
        """Should handle unstructured response."""
        response = "The player made several mistakes that led to defeat."

        critique = generator._parse_response(response)

        # Should use first line as summary
        assert "mistakes" in critique.summary.lower()
        assert critique.raw_response == response

    def test_parse_extracts_action_number(self, generator):
        """Should extract action number from response."""
        response = """
SUMMARY: Lost due to action 42 being a bad choice

KEY_MISTAKE: Taking action 42 at the wrong time
"""
        critique = generator._parse_response(response)

        assert critique.mistake_action == 42


class TestCreateReflectionGenerator:
    """Tests for factory function."""

    def test_create_with_default_config(self):
        """Should create generator with default config."""
        generator = create_reflection_generator()

        assert generator is not None
        assert generator.config.temperature == 0.3

    def test_create_with_custom_config(self):
        """Should create generator with custom config."""
        config = ReflectionGeneratorConfig(temperature=0.7)

        generator = create_reflection_generator(config)

        assert generator.config.temperature == 0.7

    def test_create_with_llm_client(self):
        """Should accept LLM client."""
        mock_client = MagicMock()

        generator = create_reflection_generator(llm_client=mock_client)

        assert generator._llm is mock_client


class TestFallbackCritique:
    """Tests for fallback critique generation."""

    @pytest.fixture
    def generator(self):
        """Create generator for testing."""
        return ReflectionGenerator()

    def test_fallback_short_game(self, generator):
        """Should identify early game issues for short games."""
        examples = [
            TrainingExample(
                spatial=np.zeros((12, 8, 8), dtype=np.float32),
                vector=np.zeros(32, dtype=np.float32),
                action_mask=np.ones(100, dtype=np.float32),
                policy=np.zeros(100, dtype=np.float32),
                value=0.0,
            )
            for _ in range(20)
        ]
        trajectory = GameTrajectory(
            examples=examples,
            game_length=20,
            winner=1,
        )

        critique = generator._generate_fallback_critique(trajectory)

        assert "early" in critique.key_mistake.lower()

    def test_fallback_long_game(self, generator):
        """Should identify late game issues for long games."""
        examples = [
            TrainingExample(
                spatial=np.zeros((12, 8, 8), dtype=np.float32),
                vector=np.zeros(32, dtype=np.float32),
                action_mask=np.ones(100, dtype=np.float32),
                policy=np.zeros(100, dtype=np.float32),
                value=0.0,
            )
            for _ in range(100)
        ]
        trajectory = GameTrajectory(
            examples=examples,
            game_length=100,
            winner=1,
        )

        critique = generator._generate_fallback_critique(trajectory)

        assert "late" in critique.key_mistake.lower()

    def test_fallback_empty_trajectory(self, generator):
        """Should handle empty trajectory."""
        trajectory = GameTrajectory(examples=[], game_length=0)

        critique = generator._generate_fallback_critique(trajectory)

        assert critique.summary != ""
        assert critique.confidence == 0.3
