"""Tests for LM Studio (and other OpenAI-compatible local servers) integration.

These tests verify the ``base_url`` extension to ``LLMConfig``/``OpenAIClient``
in ``alpha_zero/utils/llm_client.py``. The contract under test:

1. ``LLMConfig.from_env`` reads ``LM_STUDIO_BASE_URL`` and ``OPENAI_BASE_URL``.
2. ``LM_STUDIO_BASE_URL`` takes precedence when both are set.
3. When ``base_url`` is set and no ``OPENAI_API_KEY`` exists, a placeholder
   ``"lm-studio"`` is injected (LM Studio ignores the value).
4. ``OpenAIClient`` passes ``base_url`` through to the ``AsyncOpenAI`` and
   ``OpenAI`` SDK constructors.
5. **Backwards-compat invariant**: when ``base_url`` is ``None``, the SDK is
   constructed *without* the ``base_url`` kwarg — i.e., behavior is byte-
   identical to the pre-extension code path.
6. ``LLMClient`` initializes an OpenAI provider when only a ``base_url`` is set
   (no real API key required for local servers).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.utils.llm_client import (
    LLMClient,
    LLMConfig,
    LLMConfigError,
    LLMProvider,
    OpenAIClient,
)


class TestLLMConfigBaseURL:
    """Verify base_url field on LLMConfig and its env-var sourcing."""

    def test_default_base_url_is_none(self) -> None:
        config = LLMConfig()
        assert config.openai_base_url is None

    def test_explicit_base_url(self) -> None:
        config = LLMConfig(openai_base_url="http://localhost:1234/v1")
        assert config.openai_base_url == "http://localhost:1234/v1"

    def test_from_env_reads_lm_studio_base_url(self) -> None:
        env = {"LM_STUDIO_BASE_URL": "http://localhost:1234/v1"}
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()
        assert config.openai_base_url == "http://localhost:1234/v1"

    def test_from_env_reads_openai_base_url(self) -> None:
        env = {"OPENAI_BASE_URL": "http://my-vllm-server:8000/v1"}
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()
        assert config.openai_base_url == "http://my-vllm-server:8000/v1"

    def test_lm_studio_takes_precedence_over_openai_base_url(self) -> None:
        env = {
            "LM_STUDIO_BASE_URL": "http://localhost:1234/v1",
            "OPENAI_BASE_URL": "http://other:9999/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()
        assert config.openai_base_url == "http://localhost:1234/v1"

    def test_placeholder_api_key_when_base_url_set_without_key(self) -> None:
        env = {"LM_STUDIO_BASE_URL": "http://localhost:1234/v1"}
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()
        assert config.openai_api_key == "lm-studio"

    def test_real_api_key_preserved_when_base_url_set(self) -> None:
        env = {
            "LM_STUDIO_BASE_URL": "http://localhost:1234/v1",
            "OPENAI_API_KEY": "sk-real-key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()
        assert config.openai_api_key == "sk-real-key"

    def test_no_placeholder_when_base_url_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig.from_env()
        assert config.openai_api_key is None
        assert config.openai_base_url is None


class TestOpenAIClientBaseURL:
    """Verify OpenAIClient plumbs base_url through to the SDK."""

    def test_constructor_passes_base_url_to_sdk(self) -> None:
        config = LLMConfig(
            openai_api_key="lm-studio",
            openai_base_url="http://localhost:1234/v1",
        )
        with (
            patch("alpha_zero.utils.llm_client.AsyncOpenAI") as mock_async,
            patch("alpha_zero.utils.llm_client.OpenAI") as mock_sync,
        ):
            OpenAIClient(config)

        assert mock_async.call_count == 1
        async_kwargs = mock_async.call_args.kwargs
        assert async_kwargs["base_url"] == "http://localhost:1234/v1"
        assert async_kwargs["api_key"] == "lm-studio"

        assert mock_sync.call_count == 1
        sync_kwargs = mock_sync.call_args.kwargs
        assert sync_kwargs["base_url"] == "http://localhost:1234/v1"

    def test_constructor_omits_base_url_when_none_backwards_compat(self) -> None:
        """Critical backwards-compat invariant.

        With ``base_url=None``, the SDK must be constructed *without* a
        ``base_url`` kwarg — same as before the extension. This guards against
        any subtle behavior change in the OpenAI SDK when it receives an
        explicit ``base_url=None`` vs. omission.
        """
        config = LLMConfig(openai_api_key="sk-real-key")
        with (
            patch("alpha_zero.utils.llm_client.AsyncOpenAI") as mock_async,
            patch("alpha_zero.utils.llm_client.OpenAI") as mock_sync,
        ):
            OpenAIClient(config)

        assert "base_url" not in mock_async.call_args.kwargs
        assert "base_url" not in mock_sync.call_args.kwargs

    def test_placeholder_key_when_base_url_and_no_key(self) -> None:
        config = LLMConfig(openai_base_url="http://localhost:1234/v1")
        with (
            patch("alpha_zero.utils.llm_client.AsyncOpenAI") as mock_async,
            patch("alpha_zero.utils.llm_client.OpenAI"),
        ):
            OpenAIClient(config)
        assert mock_async.call_args.kwargs["api_key"] == "lm-studio"

    def test_no_key_no_base_url_raises(self) -> None:
        config = LLMConfig()
        with pytest.raises(LLMConfigError, match="OpenAI API key not configured"):
            OpenAIClient(config)


class TestLLMClientInitializesWithBaseURLOnly:
    """LLMClient should treat a base_url as sufficient OpenAI credentials."""

    def test_initialization_with_base_url_no_key(self) -> None:
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            openai_base_url="http://localhost:1234/v1",
        )
        with (
            patch("alpha_zero.utils.llm_client.AsyncOpenAI", MagicMock()),
            patch("alpha_zero.utils.llm_client.OpenAI", MagicMock()),
        ):
            client = LLMClient(config)

        assert LLMProvider.OPENAI in client.available_providers

    def test_no_credentials_raises(self) -> None:
        config = LLMConfig()
        with pytest.raises(LLMConfigError, match="No LLM clients could be initialized"):
            LLMClient(config)
