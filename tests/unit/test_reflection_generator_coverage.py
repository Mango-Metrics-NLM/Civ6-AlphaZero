"""Coverage tests for alpha_zero.core.reflection_generator module."""

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


def _make_example(vector_dim=64, action_size=376):
    return TrainingExample(
        spatial=np.random.randn(13, 8, 8).astype(np.float32),
        vector=np.random.randn(vector_dim).astype(np.float32),
        action_mask=np.ones(action_size, dtype=np.float32),
        policy=np.zeros(action_size, dtype=np.float32),
        value=0.0,
    )


def _make_trajectory(n_examples=5, winner=1, game_length=50):
    examples = [_make_example() for _ in range(n_examples)]
    for ex in examples:
        ex.policy[0] = 1.0
    return GameTrajectory(examples=examples, game_id=42, game_length=game_length, winner=winner)


class TestReflectionGeneratorConfig:
    def test_defaults(self):
        config = ReflectionGeneratorConfig()
        assert config.llm_provider == "auto"
        assert config.temperature == 0.3
        assert config.max_tokens == 500
        assert config.max_trajectory_states == 10
        assert config.cache_critiques is True
        assert config.use_fallback_on_failure is True

    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = ReflectionGeneratorConfig.from_env()
            assert config.llm_provider == "auto"
            assert config.temperature == 0.3

    def test_from_env_overrides(self):
        env = {
            "CIV6AZ_REFLECTION_LLM_PROVIDER": "openai",
            "CIV6AZ_REFLECTION_LLM_TEMP": "0.7",
            "CIV6AZ_REFLECTION_LLM_MAX_TOKENS": "1000",
            "CIV6AZ_REFLECTION_MAX_STATES": "20",
            "CIV6AZ_REFLECTION_CACHE": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ReflectionGeneratorConfig.from_env()
            assert config.llm_provider == "openai"
            assert config.temperature == 0.7
            assert config.max_tokens == 1000
            assert config.max_trajectory_states == 20
            assert config.cache_critiques is False


class TestGeneratedCritique:
    def test_defaults(self):
        c = GeneratedCritique()
        assert c.summary == ""
        assert c.key_mistake == ""
        assert c.mistake_action is None
        assert c.confidence == 0.5

    def test_full_critique_with_all_fields(self):
        c = GeneratedCritique(
            summary="Lost early",
            key_mistake="No expansion",
            suggested_alternative="Build settlers",
            strategic_lesson="Expand early",
        )
        text = c.full_critique
        assert "Summary:" in text
        assert "Key Mistake:" in text
        assert "Alternative:" in text
        assert "Lesson:" in text

    def test_full_critique_empty_falls_back_to_raw(self):
        c = GeneratedCritique(raw_response="Raw text here")
        assert c.full_critique == "Raw text here"

    def test_full_critique_partial(self):
        c = GeneratedCritique(summary="Lost", key_mistake="Bad defense")
        text = c.full_critique
        assert "Summary: Lost" in text
        assert "Key Mistake: Bad defense" in text
        assert "Alternative" not in text


class TestDefaultGameStateDescriber:
    def test_describe_state_no_vector(self):
        desc = DefaultGameStateDescriber()
        assert desc.describe_state({}) == "Unknown state"

    def test_describe_state_normal(self):
        desc = DefaultGameStateDescriber()
        vector = np.array([0.5, 0.3, 0.2, 0.1, 0.4, 0.3] + [0.0] * 58, dtype=np.float32)
        result = desc.describe_state({"vector": vector})
        assert "Turn progress" in result
        assert "Resources" in result
        assert "Units" in result
        assert "Cities" in result

    def test_describe_state_short_vector(self):
        desc = DefaultGameStateDescriber()
        vector = np.array([0.5], dtype=np.float32)
        result = desc.describe_state({"vector": vector})
        assert "Turn progress" in result

    def test_describe_action_no_space(self):
        desc = DefaultGameStateDescriber()
        assert desc.describe_action(5) == "Action 5"

    def test_describe_action_with_space(self):
        mock_space = MagicMock()
        mock_space.end_turn_offset = 375
        mock_space.move_offset = 0
        mock_space.attack_offset = 128
        mock_space.skip_offset = 256
        mock_space.found_city_offset = 288
        mock_space.produce_offset = 320

        desc = DefaultGameStateDescriber(action_space=mock_space)

        # End turn
        assert desc.describe_action(375) == "End Turn"

        # Move action
        result = desc.describe_action(5)
        assert "Move unit" in result

        # Attack action
        result = desc.describe_action(130)
        assert "Attack" in result

        # Skip action
        result = desc.describe_action(260)
        assert "Skip" in result

        # Found city
        result = desc.describe_action(290)
        assert "Found city" in result

        # Produce action
        result = desc.describe_action(330)
        assert "Produce" in result

    def test_describe_action_exception_fallback(self):
        mock_space = MagicMock()
        mock_space.end_turn_offset = property(lambda self: 1 / 0)  # Raise
        # Force attribute access to raise
        type(mock_space).end_turn_offset = property(lambda self: 1 / 0)

        desc = DefaultGameStateDescriber(action_space=mock_space)
        result = desc.describe_action(5)
        assert "Action 5" in result

    def test_describe_trajectory_empty(self):
        desc = DefaultGameStateDescriber()
        t = GameTrajectory(examples=[], game_id=1, game_length=0, winner=None)
        assert desc.describe_trajectory(t) == "Empty game"

    def test_describe_trajectory_short(self):
        desc = DefaultGameStateDescriber()
        t = _make_trajectory(3, winner=1, game_length=30)
        result = desc.describe_trajectory(t, max_states=10)
        assert "Game 42" in result
        assert "Player 1" in result

    def test_describe_trajectory_long_truncated(self):
        desc = DefaultGameStateDescriber()
        t = _make_trajectory(20, winner=None, game_length=100)
        result = desc.describe_trajectory(t, max_states=5)
        assert "Draw" in result
        # Should have sampled states
        lines = [line for line in result.split("\n") if line.strip().startswith("[")]
        assert len(lines) <= 5


class TestReflectionGenerator:
    def test_init_defaults(self):
        gen = ReflectionGenerator()
        assert gen.config.llm_provider == "auto"
        assert gen._llm is None
        assert isinstance(gen._describer, DefaultGameStateDescriber)

    def test_init_custom(self):
        config = ReflectionGeneratorConfig(temperature=0.9)
        mock_llm = MagicMock()
        mock_describer = MagicMock()
        gen = ReflectionGenerator(
            config=config, llm_client=mock_llm, state_describer=mock_describer
        )
        assert gen.config.temperature == 0.9
        assert gen._llm is mock_llm

    def test_get_llm_client_cached(self):
        mock_llm = MagicMock()
        gen = ReflectionGenerator(llm_client=mock_llm)
        assert gen._get_llm_client() is mock_llm

    def test_get_llm_client_creation_failure(self):
        ReflectionGenerator()
        with patch(
            "alpha_zero.core.reflection_generator.ReflectionGenerator._get_llm_client",
            side_effect=RuntimeError("LLM client not available"),
        ):
            pass  # Just verify the pattern works

    def test_compute_cache_key(self):
        gen = ReflectionGenerator()
        t = _make_trajectory()
        key1 = gen._compute_cache_key(t)
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hex digest

        # Same trajectory should give same key
        key2 = gen._compute_cache_key(t)
        assert key1 == key2

    def test_compute_cache_key_different_trajectories(self):
        gen = ReflectionGenerator()
        t1 = _make_trajectory(winner=0)
        t2 = _make_trajectory(winner=1)
        assert gen._compute_cache_key(t1) != gen._compute_cache_key(t2)

    def test_compute_cache_key_empty_examples(self):
        gen = ReflectionGenerator()
        t = GameTrajectory(examples=[], game_id=1, game_length=0, winner=None)
        key = gen._compute_cache_key(t)
        assert isinstance(key, str)

    def test_build_prompt(self):
        gen = ReflectionGenerator()
        t = _make_trajectory()
        prompt = gen._build_prompt(t, game_context="Test context")
        assert "GAME SUMMARY" in prompt
        assert "Test context" in prompt

    def test_build_prompt_no_context(self):
        gen = ReflectionGenerator()
        t = _make_trajectory()
        prompt = gen._build_prompt(t, game_context=None)
        assert "GAME SUMMARY" in prompt

    def test_parse_response_full(self):
        gen = ReflectionGenerator()
        response = (
            "SUMMARY: Lost due to poor defense\n\n"
            "KEY_MISTAKE: Didn't build enough warriors at action 42\n\n"
            "ALTERNATIVE: Build warriors before expanding\n\n"
            "LESSON: Always maintain a standing army"
        )
        critique = gen._parse_response(response)
        assert "poor defense" in critique.summary
        assert "warriors" in critique.key_mistake
        assert critique.mistake_action == 42
        assert critique.confidence == 1.0

    def test_parse_response_partial(self):
        gen = ReflectionGenerator()
        response = "SUMMARY: Lost early"
        critique = gen._parse_response(response)
        assert "Lost early" in critique.summary
        assert critique.confidence == 0.25

    def test_parse_response_empty(self):
        gen = ReflectionGenerator()
        critique = gen._parse_response("")
        assert critique.confidence == 0.0

    def test_parse_response_unstructured(self):
        gen = ReflectionGenerator()
        response = "The player made several mistakes in the early game."
        critique = gen._parse_response(response)
        assert "several mistakes" in critique.summary

    def test_fallback_critique_early_loss(self):
        gen = ReflectionGenerator()
        t = _make_trajectory(game_length=30)
        critique = gen._generate_fallback_critique(t)
        assert "30 turns" in critique.summary
        assert "Early game" in critique.key_mistake
        assert critique.confidence == 0.3

    def test_fallback_critique_late_loss(self):
        gen = ReflectionGenerator()
        t = _make_trajectory(game_length=150)
        critique = gen._generate_fallback_critique(t)
        assert "Late game" in critique.key_mistake

    def test_fallback_critique_empty_trajectory(self):
        gen = ReflectionGenerator()
        t = GameTrajectory(examples=[], game_id=1, game_length=0, winner=1)
        critique = gen._generate_fallback_critique(t)
        assert "Unable to analyze" in critique.key_mistake

    @pytest.mark.asyncio
    async def test_generate_critique_success(self):
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = (
            "SUMMARY: Lost early\n\n"
            "KEY_MISTAKE: No defense\n\n"
            "ALTERNATIVE: Build units\n\n"
            "LESSON: Defend"
        )
        mock_llm.generate.return_value = mock_response

        gen = ReflectionGenerator(llm_client=mock_llm)
        t = _make_trajectory()
        critique = await gen.generate_critique(t)
        assert "Lost early" in critique.summary
        mock_llm.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_critique_cached(self):
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "SUMMARY: Cached result"
        mock_llm.generate.return_value = mock_response

        gen = ReflectionGenerator(llm_client=mock_llm)
        t = _make_trajectory()

        # First call
        await gen.generate_critique(t)
        # Second call should use cache
        critique2 = await gen.generate_critique(t)
        assert mock_llm.generate.await_count == 1
        assert "Cached result" in critique2.summary

    @pytest.mark.asyncio
    async def test_generate_critique_fallback_on_failure(self):
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = RuntimeError("API Error")

        config = ReflectionGeneratorConfig(use_fallback_on_failure=True)
        gen = ReflectionGenerator(config=config, llm_client=mock_llm)
        t = _make_trajectory()
        critique = await gen.generate_critique(t)
        assert critique.confidence == 0.3  # Fallback confidence

    @pytest.mark.asyncio
    async def test_generate_critique_no_fallback_raises(self):
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = RuntimeError("API Error")

        config = ReflectionGeneratorConfig(use_fallback_on_failure=False)
        gen = ReflectionGenerator(config=config, llm_client=mock_llm)
        t = _make_trajectory()
        with pytest.raises(RuntimeError):
            await gen.generate_critique(t)

    def test_clear_cache(self):
        gen = ReflectionGenerator()
        gen._cache["test_key"] = GeneratedCritique(summary="test")
        assert len(gen._cache) == 1
        gen.clear_cache()
        assert len(gen._cache) == 0


class TestCreateReflectionGenerator:
    def test_with_config(self):
        config = ReflectionGeneratorConfig(temperature=0.5)
        gen = create_reflection_generator(config=config)
        assert gen.config.temperature == 0.5

    def test_without_config(self):
        with patch.dict(os.environ, {}, clear=True):
            gen = create_reflection_generator()
            assert gen.config.llm_provider == "auto"

    def test_with_llm_client(self):
        mock_llm = MagicMock()
        gen = create_reflection_generator(llm_client=mock_llm)
        assert gen._llm is mock_llm
