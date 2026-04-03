"""Tests for alpha_zero.utils.llm_client to increase coverage.

Covers:
- LLMConfig creation and from_env
- LLMMessage formatting (OpenAI and Anthropic)
- LLMResponse properties (token counts)
- Error classes
- OpenAIClient initialization and generation
- AnthropicClient initialization and generation
- LLMClient unified client with provider selection and fallback
- Retry logic with exponential backoff
- Synchronous wrappers
- create_llm_client factory
- Convenience functions (generate, generate_sync)
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alpha_zero.utils.llm_client import (
    AnthropicClient,
    BaseLLMClient,
    LLMClient,
    LLMConfig,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    OpenAIClient,
    create_llm_client,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def openai_config() -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        openai_api_key="test-openai-key",
        openai_model="gpt-4",
        openai_max_tokens=100,
        openai_temperature=0.5,
        max_retries=2,
        retry_delay=0.01,
        retry_exponential_base=2.0,
        timeout=5.0,
        log_requests=True,
        log_responses=True,
    )


@pytest.fixture()
def anthropic_config() -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        anthropic_api_key="test-anthropic-key",
        anthropic_model="claude-sonnet-4-20250514",
        anthropic_max_tokens=100,
        anthropic_temperature=0.5,
        max_retries=2,
        retry_delay=0.01,
        retry_exponential_base=2.0,
        timeout=5.0,
        log_requests=True,
        log_responses=True,
    )


@pytest.fixture()
def both_config() -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.AUTO,
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        max_retries=2,
        retry_delay=0.01,
        retry_exponential_base=2.0,
    )


# ---------------------------------------------------------------------------
# LLMProvider enum
# ---------------------------------------------------------------------------


class TestLLMProvider:
    def test_values(self) -> None:
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.AUTO.value == "auto"


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------


class TestLLMConfig:
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == LLMProvider.AUTO
        assert cfg.openai_api_key is None
        assert cfg.anthropic_api_key is None
        assert cfg.max_retries == 3
        assert cfg.timeout == 60.0

    def test_from_env_defaults(self) -> None:
        env = {
            "LLM_PROVIDER": "auto",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = LLMConfig.from_env()
        assert cfg.provider == LLMProvider.AUTO

    def test_from_env_openai(self) -> None:
        env = {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_ORG_ID": "org-test",
            "OPENAI_MODEL": "gpt-4-turbo",
            "OPENAI_MAX_TOKENS": "2048",
            "OPENAI_TEMPERATURE": "0.3",
            "LLM_MAX_RETRIES": "5",
            "LLM_RETRY_DELAY": "2.0",
            "LLM_TIMEOUT": "120.0",
            "LLM_LOG_REQUESTS": "false",
            "LLM_LOG_RESPONSES": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = LLMConfig.from_env()
        assert cfg.provider == LLMProvider.OPENAI
        assert cfg.openai_api_key == "sk-test"
        assert cfg.openai_org_id == "org-test"
        assert cfg.openai_model == "gpt-4-turbo"
        assert cfg.openai_max_tokens == 2048
        assert cfg.openai_temperature == 0.3
        assert cfg.max_retries == 5
        assert cfg.retry_delay == 2.0
        assert cfg.timeout == 120.0
        assert cfg.log_requests is False
        assert cfg.log_responses is True

    def test_from_env_anthropic(self) -> None:
        env = {
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "ANTHROPIC_MODEL": "claude-opus-4-20250514",
            "ANTHROPIC_MAX_TOKENS": "8192",
            "ANTHROPIC_TEMPERATURE": "0.9",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = LLMConfig.from_env()
        assert cfg.provider == LLMProvider.ANTHROPIC
        assert cfg.anthropic_api_key == "sk-ant-test"
        assert cfg.anthropic_model == "claude-opus-4-20250514"
        assert cfg.anthropic_max_tokens == 8192
        assert cfg.anthropic_temperature == 0.9

    def test_from_env_unknown_provider_defaults_to_auto(self) -> None:
        env = {"LLM_PROVIDER": "unknown_provider"}
        with patch.dict(os.environ, env, clear=False):
            cfg = LLMConfig.from_env()
        assert cfg.provider == LLMProvider.AUTO


# ---------------------------------------------------------------------------
# LLMMessage
# ---------------------------------------------------------------------------


class TestLLMMessage:
    def test_to_openai_format_basic(self) -> None:
        msg = LLMMessage(role="user", content="Hello")
        fmt = msg.to_openai_format()
        assert fmt == {"role": "user", "content": "Hello"}

    def test_to_openai_format_with_name(self) -> None:
        msg = LLMMessage(role="user", content="Hi", name="alice")
        fmt = msg.to_openai_format()
        assert fmt["name"] == "alice"

    def test_to_anthropic_format(self) -> None:
        msg = LLMMessage(role="assistant", content="Sure")
        fmt = msg.to_anthropic_format()
        assert fmt == {"role": "assistant", "content": "Sure"}


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------


class TestLLMResponse:
    def test_token_properties(self) -> None:
        resp = LLMResponse(
            content="Hello",
            model="gpt-4",
            provider=LLMProvider.OPENAI,
            usage={"input_tokens": 10, "output_tokens": 20},
        )
        assert resp.input_tokens == 10
        assert resp.output_tokens == 20
        assert resp.total_tokens == 30

    def test_token_properties_empty_usage(self) -> None:
        resp = LLMResponse(
            content="Hi",
            model="gpt-4",
            provider=LLMProvider.OPENAI,
        )
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.total_tokens == 0

    def test_finish_reason(self) -> None:
        resp = LLMResponse(
            content="done",
            model="gpt-4",
            provider=LLMProvider.OPENAI,
            finish_reason="stop",
        )
        assert resp.finish_reason == "stop"


# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------


class TestErrors:
    def test_llm_error(self) -> None:
        err = LLMError("base error")
        assert str(err) == "base error"

    def test_llm_config_error(self) -> None:
        err = LLMConfigError("config error")
        assert isinstance(err, LLMError)

    def test_llm_connection_error(self) -> None:
        err = LLMConnectionError("conn error")
        assert isinstance(err, LLMError)

    def test_llm_rate_limit_error(self) -> None:
        err = LLMRateLimitError("rate limit")
        assert isinstance(err, LLMError)

    def test_llm_provider_error(self) -> None:
        err = LLMProviderError("provider err", LLMProvider.OPENAI)
        assert err.provider == LLMProvider.OPENAI
        assert isinstance(err, LLMError)


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------


class TestOpenAIClient:
    def test_init_raises_without_api_key(self) -> None:
        cfg = LLMConfig(provider=LLMProvider.OPENAI, openai_api_key=None)
        with patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            with pytest.raises(LLMConfigError, match="API key"):
                OpenAIClient(cfg)

    def test_init_raises_without_package(self) -> None:
        cfg = LLMConfig(provider=LLMProvider.OPENAI, openai_api_key="key")
        with patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", False):
            with pytest.raises(LLMConfigError, match="not installed"):
                OpenAIClient(cfg)

    @pytest.mark.asyncio()
    async def test_generate(self, openai_config: LLMConfig) -> None:
        """Test OpenAI generate with mocked async client."""
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI") as MockAsync, patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ) as MockSync, patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            # Set up mock response
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 5
            mock_usage.completion_tokens = 10

            mock_choice = MagicMock()
            mock_choice.message.content = "Test response"
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage

            mock_async_instance = MagicMock()
            mock_async_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            MockAsync.return_value = mock_async_instance

            client = OpenAIClient(openai_config)
            resp = await client.generate(
                "Hello", system_prompt="You are helpful"
            )

            assert resp.content == "Test response"
            assert resp.provider == LLMProvider.OPENAI
            assert resp.usage["input_tokens"] == 5
            assert resp.usage["output_tokens"] == 10
            assert resp.finish_reason == "stop"

    @pytest.mark.asyncio()
    async def test_generate_no_usage(self, openai_config: LLMConfig) -> None:
        """Test when response.usage is None."""
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI") as MockAsync, patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            mock_choice = MagicMock()
            mock_choice.message.content = "Hello"
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = None

            mock_async_instance = MagicMock()
            mock_async_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            MockAsync.return_value = mock_async_instance

            client = OpenAIClient(openai_config)
            resp = await client.generate("Hi")

            assert resp.usage["input_tokens"] == 0
            assert resp.usage["output_tokens"] == 0

    @pytest.mark.asyncio()
    async def test_generate_none_content(self, openai_config: LLMConfig) -> None:
        """Test when message content is None."""
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI") as MockAsync, patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            mock_choice = MagicMock()
            mock_choice.message.content = None
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = None

            mock_async_instance = MagicMock()
            mock_async_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            MockAsync.return_value = mock_async_instance

            client = OpenAIClient(openai_config)
            resp = await client.generate("Hi")
            assert resp.content == ""

    def test_build_messages(self, openai_config: LLMConfig) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            client = OpenAIClient(openai_config)
            msgs = client._build_messages(
                "Hello",
                "Be helpful",
                [LLMMessage(role="assistant", content="Sure")],
            )
            assert len(msgs) == 3
            assert msgs[0].role == "system"
            assert msgs[1].role == "assistant"
            assert msgs[2].role == "user"

    def test_build_messages_no_system_no_history(
        self, openai_config: LLMConfig
    ) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            client = OpenAIClient(openai_config)
            msgs = client._build_messages("Hello", None, None)
            assert len(msgs) == 1
            assert msgs[0].content == "Hello"


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------


class TestAnthropicClient:
    def test_init_raises_without_api_key(self) -> None:
        cfg = LLMConfig(provider=LLMProvider.ANTHROPIC, anthropic_api_key=None)
        with patch("alpha_zero.utils.llm_client.ANTHROPIC_AVAILABLE", True):
            with pytest.raises(LLMConfigError, match="API key"):
                AnthropicClient(cfg)

    def test_init_raises_without_package(self) -> None:
        cfg = LLMConfig(
            provider=LLMProvider.ANTHROPIC, anthropic_api_key="key"
        )
        with patch("alpha_zero.utils.llm_client.ANTHROPIC_AVAILABLE", False):
            with pytest.raises(LLMConfigError, match="not installed"):
                AnthropicClient(cfg)

    @pytest.mark.asyncio()
    async def test_generate(self, anthropic_config: LLMConfig) -> None:
        with patch(
            "alpha_zero.utils.llm_client.AsyncAnthropic"
        ) as MockAsync, patch(
            "alpha_zero.utils.llm_client.Anthropic"
        ), patch(
            "alpha_zero.utils.llm_client.ANTHROPIC_AVAILABLE", True
        ):
            mock_block = MagicMock()
            mock_block.text = "Anthropic response"

            mock_usage = MagicMock()
            mock_usage.input_tokens = 8
            mock_usage.output_tokens = 15

            mock_response = MagicMock()
            mock_response.content = [mock_block]
            mock_response.usage = mock_usage
            mock_response.stop_reason = "end_turn"

            mock_async_instance = MagicMock()
            mock_async_instance.messages.create = AsyncMock(
                return_value=mock_response
            )
            MockAsync.return_value = mock_async_instance

            client = AnthropicClient(anthropic_config)
            resp = await client.generate(
                "Hello", system_prompt="Be concise"
            )

            assert resp.content == "Anthropic response"
            assert resp.provider == LLMProvider.ANTHROPIC
            assert resp.usage["input_tokens"] == 8
            assert resp.usage["output_tokens"] == 15
            assert resp.finish_reason == "end_turn"

    @pytest.mark.asyncio()
    async def test_generate_no_system_prompt(
        self, anthropic_config: LLMConfig
    ) -> None:
        with patch(
            "alpha_zero.utils.llm_client.AsyncAnthropic"
        ) as MockAsync, patch(
            "alpha_zero.utils.llm_client.Anthropic"
        ), patch(
            "alpha_zero.utils.llm_client.ANTHROPIC_AVAILABLE", True
        ):
            mock_block = MagicMock()
            mock_block.text = "Response"

            mock_response = MagicMock()
            mock_response.content = [mock_block]
            mock_response.usage = MagicMock(input_tokens=1, output_tokens=2)
            mock_response.stop_reason = "end_turn"

            mock_async_instance = MagicMock()
            mock_async_instance.messages.create = AsyncMock(
                return_value=mock_response
            )
            MockAsync.return_value = mock_async_instance

            client = AnthropicClient(anthropic_config)
            resp = await client.generate("Hello")

            # Verify system was not passed
            call_kwargs = mock_async_instance.messages.create.call_args
            assert "system" not in call_kwargs.kwargs

    @pytest.mark.asyncio()
    async def test_generate_empty_content(
        self, anthropic_config: LLMConfig
    ) -> None:
        with patch(
            "alpha_zero.utils.llm_client.AsyncAnthropic"
        ) as MockAsync, patch(
            "alpha_zero.utils.llm_client.Anthropic"
        ), patch(
            "alpha_zero.utils.llm_client.ANTHROPIC_AVAILABLE", True
        ):
            mock_response = MagicMock()
            mock_response.content = []
            mock_response.usage = None
            mock_response.stop_reason = "end_turn"

            mock_async_instance = MagicMock()
            mock_async_instance.messages.create = AsyncMock(
                return_value=mock_response
            )
            MockAsync.return_value = mock_async_instance

            client = AnthropicClient(anthropic_config)
            resp = await client.generate("Hello")
            assert resp.content == ""
            assert resp.usage["input_tokens"] == 0

    def test_build_messages(self, anthropic_config: LLMConfig) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncAnthropic"), patch(
            "alpha_zero.utils.llm_client.Anthropic"
        ), patch("alpha_zero.utils.llm_client.ANTHROPIC_AVAILABLE", True):
            client = AnthropicClient(anthropic_config)
            msgs = client._build_messages(
                "Hello",
                [LLMMessage(role="assistant", content="Prior")],
            )
            assert len(msgs) == 2
            assert msgs[0].role == "assistant"
            assert msgs[1].role == "user"

    def test_build_messages_no_history(
        self, anthropic_config: LLMConfig
    ) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncAnthropic"), patch(
            "alpha_zero.utils.llm_client.Anthropic"
        ), patch("alpha_zero.utils.llm_client.ANTHROPIC_AVAILABLE", True):
            client = AnthropicClient(anthropic_config)
            msgs = client._build_messages("Hello", None)
            assert len(msgs) == 1


# ---------------------------------------------------------------------------
# BaseLLMClient sync wrapper
# ---------------------------------------------------------------------------


class TestBaseLLMClientSync:
    def test_generate_sync(self) -> None:
        """Test that generate_sync calls asyncio.run on generate."""

        class DummyClient(BaseLLMClient):
            async def generate(self, prompt, system_prompt=None, messages=None, **kwargs):
                return LLMResponse(
                    content="sync result",
                    model="test",
                    provider=LLMProvider.OPENAI,
                )

            async def generate_stream(self, prompt, system_prompt=None, messages=None, **kwargs):
                yield "chunk"

        client = DummyClient()
        resp = client.generate_sync("Hello")
        assert resp.content == "sync result"


# ---------------------------------------------------------------------------
# LLMClient (unified)
# ---------------------------------------------------------------------------


class TestLLMClient:
    def test_no_clients_raises(self) -> None:
        cfg = LLMConfig(openai_api_key=None, anthropic_api_key=None)
        with pytest.raises(LLMConfigError, match="No LLM clients"):
            LLMClient(cfg)

    def test_available_providers_openai_only(
        self, openai_config: LLMConfig
    ) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            client = LLMClient(openai_config)
            assert LLMProvider.OPENAI in client.available_providers

    def test_available_providers_anthropic_only(
        self, anthropic_config: LLMConfig
    ) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncAnthropic"), patch(
            "alpha_zero.utils.llm_client.Anthropic"
        ), patch("alpha_zero.utils.llm_client.ANTHROPIC_AVAILABLE", True):
            client = LLMClient(anthropic_config)
            assert LLMProvider.ANTHROPIC in client.available_providers

    def test_provider_order_openai_preferred(
        self, openai_config: LLMConfig
    ) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            client = LLMClient(openai_config)
            order = client._get_provider_order()
            assert order[0] == LLMProvider.OPENAI

    def test_provider_order_anthropic_preferred(
        self, anthropic_config: LLMConfig
    ) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncAnthropic"), patch(
            "alpha_zero.utils.llm_client.Anthropic"
        ), patch("alpha_zero.utils.llm_client.ANTHROPIC_AVAILABLE", True):
            client = LLMClient(anthropic_config)
            order = client._get_provider_order()
            assert order[0] == LLMProvider.ANTHROPIC

    def test_provider_order_auto_prefers_anthropic(
        self, both_config: LLMConfig
    ) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True), patch(
            "alpha_zero.utils.llm_client.AsyncAnthropic"
        ), patch(
            "alpha_zero.utils.llm_client.Anthropic"
        ), patch(
            "alpha_zero.utils.llm_client.ANTHROPIC_AVAILABLE", True
        ):
            client = LLMClient(both_config)
            order = client._get_provider_order()
            assert order[0] == LLMProvider.ANTHROPIC

    @pytest.mark.asyncio()
    async def test_generate_success(self, openai_config: LLMConfig) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            client = LLMClient(openai_config)

            mock_resp = LLMResponse(
                content="Result",
                model="gpt-4",
                provider=LLMProvider.OPENAI,
            )
            mock_inner = AsyncMock(return_value=mock_resp)
            client._clients[LLMProvider.OPENAI].generate = mock_inner

            resp = await client.generate("Hello")
            assert resp.content == "Result"

    @pytest.mark.asyncio()
    async def test_generate_retries_on_failure(
        self, openai_config: LLMConfig
    ) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            client = LLMClient(openai_config)

            mock_resp = LLMResponse(
                content="OK",
                model="gpt-4",
                provider=LLMProvider.OPENAI,
            )
            # Fail first, succeed second
            mock_inner = AsyncMock(
                side_effect=[RuntimeError("transient"), mock_resp]
            )
            client._clients[LLMProvider.OPENAI].generate = mock_inner

            with patch("asyncio.sleep", new_callable=AsyncMock):
                resp = await client.generate("Hello")

            assert resp.content == "OK"
            assert mock_inner.call_count == 2

    @pytest.mark.asyncio()
    async def test_generate_all_retries_exhausted_raises(
        self, openai_config: LLMConfig
    ) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            client = LLMClient(openai_config)

            mock_inner = AsyncMock(
                side_effect=RuntimeError("persistent error")
            )
            client._clients[LLMProvider.OPENAI].generate = mock_inner

            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(LLMError, match="All providers failed"):
                    await client.generate("Hello")

    @pytest.mark.asyncio()
    async def test_generate_fallback_to_second_provider(
        self, both_config: LLMConfig
    ) -> None:
        """When first provider fails all retries, fall back to second."""
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True), patch(
            "alpha_zero.utils.llm_client.AsyncAnthropic"
        ), patch(
            "alpha_zero.utils.llm_client.Anthropic"
        ), patch(
            "alpha_zero.utils.llm_client.ANTHROPIC_AVAILABLE", True
        ):
            client = LLMClient(both_config)

            # Anthropic fails (first in auto order)
            anthropic_mock = AsyncMock(
                side_effect=RuntimeError("anthropic down")
            )
            client._clients[LLMProvider.ANTHROPIC].generate = anthropic_mock

            # OpenAI succeeds
            ok_resp = LLMResponse(
                content="Fallback OK",
                model="gpt-4",
                provider=LLMProvider.OPENAI,
            )
            openai_mock = AsyncMock(return_value=ok_resp)
            client._clients[LLMProvider.OPENAI].generate = openai_mock

            with patch("asyncio.sleep", new_callable=AsyncMock):
                resp = await client.generate("Hello")

            assert resp.content == "Fallback OK"
            assert resp.provider == LLMProvider.OPENAI

    def test_generate_sync(self, openai_config: LLMConfig) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            client = LLMClient(openai_config)

            mock_resp = LLMResponse(
                content="Sync result",
                model="gpt-4",
                provider=LLMProvider.OPENAI,
            )
            mock_inner = AsyncMock(return_value=mock_resp)
            client._clients[LLMProvider.OPENAI].generate = mock_inner

            resp = client.generate_sync("Hello")
            assert resp.content == "Sync result"

    def test_from_env(self) -> None:
        env = {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-env-test",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "alpha_zero.utils.llm_client.AsyncOpenAI"
        ), patch("alpha_zero.utils.llm_client.OpenAI"), patch(
            "alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True
        ):
            client = LLMClient.from_env()
            assert LLMProvider.OPENAI in client.available_providers


# ---------------------------------------------------------------------------
# LLMClient streaming
# ---------------------------------------------------------------------------


class TestLLMClientStreaming:
    @pytest.mark.asyncio()
    async def test_generate_stream_success(
        self, openai_config: LLMConfig
    ) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            client = LLMClient(openai_config)

            async def mock_stream(*args, **kwargs):
                for chunk in ["Hello", " ", "world"]:
                    yield chunk

            client._clients[LLMProvider.OPENAI].generate_stream = mock_stream

            chunks = []
            async for chunk in client.generate_stream("Hi"):
                chunks.append(chunk)
            assert chunks == ["Hello", " ", "world"]

    @pytest.mark.asyncio()
    async def test_generate_stream_fallback(
        self, both_config: LLMConfig
    ) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True), patch(
            "alpha_zero.utils.llm_client.AsyncAnthropic"
        ), patch(
            "alpha_zero.utils.llm_client.Anthropic"
        ), patch(
            "alpha_zero.utils.llm_client.ANTHROPIC_AVAILABLE", True
        ):
            client = LLMClient(both_config)

            async def failing_stream(*args, **kwargs):
                raise RuntimeError("stream failed")
                yield  # noqa: unreachable - makes it an async generator

            async def ok_stream(*args, **kwargs):
                yield "ok"

            client._clients[
                LLMProvider.ANTHROPIC
            ].generate_stream = failing_stream
            client._clients[LLMProvider.OPENAI].generate_stream = ok_stream

            chunks = []
            async for chunk in client.generate_stream("Hi"):
                chunks.append(chunk)
            assert chunks == ["ok"]

    @pytest.mark.asyncio()
    async def test_generate_stream_all_fail(
        self, openai_config: LLMConfig
    ) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            client = LLMClient(openai_config)

            async def failing_stream(*args, **kwargs):
                raise RuntimeError("fail")
                yield  # noqa: unreachable

            client._clients[
                LLMProvider.OPENAI
            ].generate_stream = failing_stream

            with pytest.raises(LLMError, match="All providers failed"):
                async for _ in client.generate_stream("Hi"):
                    pass


# ---------------------------------------------------------------------------
# create_llm_client factory
# ---------------------------------------------------------------------------


class TestCreateLLMClient:
    def test_with_config(self, openai_config: LLMConfig) -> None:
        with patch("alpha_zero.utils.llm_client.AsyncOpenAI"), patch(
            "alpha_zero.utils.llm_client.OpenAI"
        ), patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            client = create_llm_client(openai_config)
            assert isinstance(client, LLMClient)

    def test_without_config_uses_env(self) -> None:
        env = {
            "OPENAI_API_KEY": "sk-factory-test",
            "LLM_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "alpha_zero.utils.llm_client.AsyncOpenAI"
        ), patch("alpha_zero.utils.llm_client.OpenAI"), patch(
            "alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True
        ):
            client = create_llm_client()
            assert isinstance(client, LLMClient)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_generate_sync_convenience(self) -> None:
        """Test module-level generate_sync function."""
        from alpha_zero.utils import llm_client as mod

        mock_resp = LLMResponse(
            content="convenience",
            model="gpt-4",
            provider=LLMProvider.OPENAI,
        )

        with patch.object(
            mod, "create_llm_client"
        ) as mock_create:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=mock_resp)
            mock_create.return_value = mock_client

            result = mod.generate_sync("Hello")
            assert result == "convenience"
