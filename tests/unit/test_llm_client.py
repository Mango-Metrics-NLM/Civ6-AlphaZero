"""Tests for LLM client module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from alpha_zero.utils.llm_client import (
    LLMConfig,
    LLMConfigError,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
)


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == LLMProvider.AUTO
        assert config.openai_model == "gpt-4"
        assert config.anthropic_model == "claude-sonnet-4-20250514"
        assert config.max_retries == 3
        assert config.timeout == 60.0

    def test_from_env_with_no_vars(self):
        """Test from_env with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig.from_env()
            assert config.provider == LLMProvider.AUTO
            assert config.openai_api_key is None
            assert config.anthropic_api_key is None

    def test_from_env_with_provider(self):
        """Test from_env with provider set."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            config = LLMConfig.from_env()
            assert config.provider == LLMProvider.OPENAI

        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}, clear=True):
            config = LLMConfig.from_env()
            assert config.provider == LLMProvider.ANTHROPIC

    def test_from_env_with_api_keys(self):
        """Test from_env with API keys set."""
        env_vars = {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = LLMConfig.from_env()
            assert config.openai_api_key == "test-openai-key"
            assert config.anthropic_api_key == "test-anthropic-key"

    def test_from_env_with_custom_models(self):
        """Test from_env with custom model names."""
        env_vars = {
            "OPENAI_MODEL": "gpt-4-turbo",
            "ANTHROPIC_MODEL": "claude-3-opus",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = LLMConfig.from_env()
            assert config.openai_model == "gpt-4-turbo"
            assert config.anthropic_model == "claude-3-opus"

    def test_from_env_with_numeric_settings(self):
        """Test from_env with numeric settings."""
        env_vars = {
            "OPENAI_MAX_TOKENS": "8192",
            "OPENAI_TEMPERATURE": "0.5",
            "LLM_MAX_RETRIES": "5",
            "LLM_TIMEOUT": "120.0",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = LLMConfig.from_env()
            assert config.openai_max_tokens == 8192
            assert config.openai_temperature == 0.5
            assert config.max_retries == 5
            assert config.timeout == 120.0


class TestLLMMessage:
    """Tests for LLMMessage dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None

    def test_message_with_name(self):
        """Test creating a message with name."""
        msg = LLMMessage(role="assistant", content="Hi there", name="Claude")
        assert msg.name == "Claude"

    def test_to_openai_format(self):
        """Test conversion to OpenAI format."""
        msg = LLMMessage(role="user", content="Test")
        result = msg.to_openai_format()
        assert result == {"role": "user", "content": "Test"}

    def test_to_openai_format_with_name(self):
        """Test conversion to OpenAI format with name."""
        msg = LLMMessage(role="system", content="Instructions", name="system")
        result = msg.to_openai_format()
        assert result == {"role": "system", "content": "Instructions", "name": "system"}

    def test_to_anthropic_format(self):
        """Test conversion to Anthropic format."""
        msg = LLMMessage(role="user", content="Test")
        result = msg.to_anthropic_format()
        assert result == {"role": "user", "content": "Test"}


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_response(self):
        """Test creating a response."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4",
            provider=LLMProvider.OPENAI,
        )
        assert response.content == "Hello, world!"
        assert response.model == "gpt-4"
        assert response.provider == LLMProvider.OPENAI

    def test_response_with_usage(self):
        """Test response with usage data."""
        response = LLMResponse(
            content="Test",
            model="claude-3",
            provider=LLMProvider.ANTHROPIC,
            usage={"input_tokens": 10, "output_tokens": 20},
        )
        assert response.input_tokens == 10
        assert response.output_tokens == 20
        assert response.total_tokens == 30

    def test_response_without_usage(self):
        """Test response without usage data returns 0."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            provider=LLMProvider.OPENAI,
        )
        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.total_tokens == 0


class TestLLMClientComplete:
    """Tests for the synchronous complete() alias on LLMClient.

    The StrategyAdvisor calls ``llm_client.complete(prompt)``; this alias makes
    that call path work against a real client by delegating to generate_sync.
    We bypass __init__ (which requires API keys) and stub generate_sync.
    """

    def _make_client(self):
        from alpha_zero.utils.llm_client import LLMClient

        return LLMClient.__new__(LLMClient)

    def test_complete_returns_content_string(self):
        """complete() returns the .content of the generate_sync response."""
        client = self._make_client()
        captured = {}

        def fake_generate_sync(prompt, system_prompt=None, **kwargs):
            captured["prompt"] = prompt
            captured["system_prompt"] = system_prompt
            return LLMResponse(
                content="completed text",
                model="m",
                provider=LLMProvider.OPENAI,
            )

        client.generate_sync = fake_generate_sync  # type: ignore[method-assign]

        result = client.complete("a prompt", system_prompt="be terse")

        assert result == "completed text"
        assert captured == {"prompt": "a prompt", "system_prompt": "be terse"}

    def test_complete_passes_kwargs_through(self):
        """Extra kwargs are forwarded to generate_sync."""
        client = self._make_client()
        captured = {}

        def fake_generate_sync(prompt, system_prompt=None, **kwargs):
            captured.update(kwargs)
            return LLMResponse(content="ok", model="m", provider=LLMProvider.OPENAI)

        client.generate_sync = fake_generate_sync  # type: ignore[method-assign]

        assert client.complete("p", temperature=0.1) == "ok"
        assert captured["temperature"] == 0.1


class TestLLMProvider:
    """Tests for LLMProvider enum."""

    def test_provider_values(self):
        """Test provider enum values."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.AUTO.value == "auto"

    def test_provider_from_string(self):
        """Test creating provider from string."""
        assert LLMProvider("openai") == LLMProvider.OPENAI
        assert LLMProvider("anthropic") == LLMProvider.ANTHROPIC
        assert LLMProvider("auto") == LLMProvider.AUTO


class TestLLMErrors:
    """Tests for LLM error classes."""

    def test_llm_error(self):
        """Test base LLM error."""
        error = LLMError("Test error")
        assert str(error) == "Test error"

    def test_llm_config_error(self):
        """Test config error."""
        error = LLMConfigError("Missing API key")
        assert str(error) == "Missing API key"
        assert isinstance(error, LLMError)


class TestLLMClientInitialization:
    """Tests for LLM client initialization."""

    def test_no_clients_raises_error(self):
        """Test that no available clients raises error."""
        config = LLMConfig()  # No API keys

        # Import here to avoid import errors if packages not installed
        try:
            from alpha_zero.utils.llm_client import LLMClient

            with pytest.raises(LLMConfigError):
                LLMClient(config)
        except ImportError:
            pytest.skip("LLM packages not installed")

    def test_client_with_api_key(self):
        """Test client initialization with API key."""
        # This test only runs if the openai or anthropic package is installed
        try:
            from alpha_zero.utils.llm_client import ANTHROPIC_AVAILABLE, OPENAI_AVAILABLE, LLMClient

            if not OPENAI_AVAILABLE and not ANTHROPIC_AVAILABLE:
                pytest.skip("No LLM packages installed")

            config = LLMConfig()
            if OPENAI_AVAILABLE:
                config.openai_api_key = "test-key"
            if ANTHROPIC_AVAILABLE:
                config.anthropic_api_key = "test-key"

            client = LLMClient(config)
            assert len(client.available_providers) > 0
        except ImportError:
            pytest.skip("LLM packages not installed")
