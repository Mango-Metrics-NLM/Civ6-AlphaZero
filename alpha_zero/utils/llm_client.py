"""LLM Client Module for OpenAI and Anthropic API Integration.

This module provides a unified, configurable interface for interacting with
LLM providers (OpenAI and Anthropic) with automatic fallback, retry logic,
and comprehensive logging.

Usage:
    from alpha_zero.utils.llm_client import LLMClient, create_llm_client

    # Create client from environment variables
    client = create_llm_client()

    # Generate a response
    response = await client.generate("Explain MCTS in one paragraph")

    # Or use synchronous version
    response = client.generate_sync("Explain MCTS")
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

# Optional imports - gracefully handle missing dependencies
try:
    from openai import AsyncOpenAI, OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Type stubs for when openai is not installed
    AsyncOpenAI = None  # type: ignore[misc, assignment]
    OpenAI = None  # type: ignore[misc, assignment]

try:
    from anthropic import Anthropic, AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    # Type stubs for when anthropic is not installed
    AsyncAnthropic = None  # type: ignore[misc, assignment]
    Anthropic = None  # type: ignore[misc, assignment]

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AUTO = "auto"


@dataclass
class LLMConfig:
    """Configuration for LLM clients."""

    # Provider selection
    provider: LLMProvider = LLMProvider.AUTO

    # OpenAI settings
    openai_api_key: str | None = None
    openai_org_id: str | None = None
    openai_model: str = "gpt-4"
    openai_max_tokens: int = 4096
    openai_temperature: float = 0.7

    # Anthropic settings
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_max_tokens: int = 4096
    anthropic_temperature: float = 0.7

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_exponential_base: float = 2.0

    # Timeout settings
    timeout: float = 60.0

    # Logging
    log_requests: bool = True
    log_responses: bool = False  # Set to True for debugging, but be careful with sensitive data

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Create configuration from environment variables.

        Returns:
            LLMConfig instance populated from environment.
        """
        provider_str = os.getenv("LLM_PROVIDER", "auto").lower()
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "anthropic": LLMProvider.ANTHROPIC,
            "auto": LLMProvider.AUTO,
        }
        provider = provider_map.get(provider_str, LLMProvider.AUTO)

        return cls(
            provider=provider,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_org_id=os.getenv("OPENAI_ORG_ID"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4"),
            openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            anthropic_max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
            anthropic_temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("LLM_RETRY_DELAY", "1.0")),
            timeout=float(os.getenv("LLM_TIMEOUT", "60.0")),
            log_requests=os.getenv("LLM_LOG_REQUESTS", "true").lower() == "true",
            log_responses=os.getenv("LLM_LOG_RESPONSES", "false").lower() == "true",
        )


@dataclass
class LLMMessage:
    """Represents a message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str
    name: str | None = None

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI message format."""
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        return msg

    def to_anthropic_format(self) -> dict[str, str]:
        """Convert to Anthropic message format."""
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""

    content: str
    model: str
    provider: LLMProvider
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw_response: Any = None

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get("input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get("output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.input_tokens + self.output_tokens


class LLMError(Exception):
    """Base exception for LLM errors."""


class LLMConfigError(LLMError):
    """Configuration error."""


class LLMConnectionError(LLMError):
    """Connection error."""


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""


class LLMProviderError(LLMError):
    """Provider-specific error."""

    def __init__(self, message: str, provider: LLMProvider):
        super().__init__(message)
        self.provider = provider


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        messages: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM."""

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        messages: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM."""

    def generate_sync(
        self,
        prompt: str,
        system_prompt: str | None = None,
        messages: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous wrapper for generate."""
        return asyncio.run(self.generate(prompt, system_prompt, messages, **kwargs))


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self, config: LLMConfig):
        if not OPENAI_AVAILABLE:
            raise LLMConfigError("OpenAI package not installed. Run: pip install openai")

        if not config.openai_api_key:
            raise LLMConfigError("OpenAI API key not configured")

        self.config = config
        self._async_client = AsyncOpenAI(
            api_key=config.openai_api_key,
            organization=config.openai_org_id,
            timeout=config.timeout,
        )
        self._sync_client = OpenAI(
            api_key=config.openai_api_key,
            organization=config.openai_org_id,
            timeout=config.timeout,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        messages: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using OpenAI."""
        request_messages = self._build_messages(prompt, system_prompt, messages)

        model = kwargs.get("model", self.config.openai_model)
        max_tokens = kwargs.get("max_tokens", self.config.openai_max_tokens)
        temperature = kwargs.get("temperature", self.config.openai_temperature)

        if self.config.log_requests:
            logger.debug(f"OpenAI request: model={model}, messages={len(request_messages)}")

        response = await self._async_client.chat.completions.create(
            model=model,
            # OpenAI SDK types require specific message param types; our dict format is compatible
            messages=[m.to_openai_format() for m in request_messages],  # type: ignore[misc, arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = response.choices[0].message.content or ""
        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        if self.config.log_responses:
            logger.debug(f"OpenAI response: {len(content)} chars, {usage}")

        return LLMResponse(
            content=content,
            model=model,
            provider=LLMProvider.OPENAI,
            usage=usage,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response,
        )

    async def generate_stream(  # type: ignore[override]
        self,
        prompt: str,
        system_prompt: str | None = None,
        messages: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using OpenAI."""
        request_messages = self._build_messages(prompt, system_prompt, messages)

        model = kwargs.get("model", self.config.openai_model)
        max_tokens = kwargs.get("max_tokens", self.config.openai_max_tokens)
        temperature = kwargs.get("temperature", self.config.openai_temperature)

        stream = await self._async_client.chat.completions.create(
            model=model,
            # OpenAI SDK types require specific message param types; our dict format is compatible
            messages=[m.to_openai_format() for m in request_messages],  # type: ignore[misc, arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        # Type assertion for stream - openai SDK returns AsyncStream when stream=True
        async for chunk in stream:  # type: ignore[union-attr]
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None,
        messages: list[LLMMessage] | None,
    ) -> list[LLMMessage]:
        """Build message list for the request."""
        result = []
        if system_prompt:
            result.append(LLMMessage(role="system", content=system_prompt))
        if messages:
            result.extend(messages)
        result.append(LLMMessage(role="user", content=prompt))
        return result


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""

    def __init__(self, config: LLMConfig):
        if not ANTHROPIC_AVAILABLE:
            raise LLMConfigError("Anthropic package not installed. Run: pip install anthropic")

        if not config.anthropic_api_key:
            raise LLMConfigError("Anthropic API key not configured")

        self.config = config
        self._async_client = AsyncAnthropic(
            api_key=config.anthropic_api_key,
            timeout=config.timeout,
        )
        self._sync_client = Anthropic(
            api_key=config.anthropic_api_key,
            timeout=config.timeout,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        messages: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using Anthropic."""
        request_messages = self._build_messages(prompt, messages)

        model = kwargs.get("model", self.config.anthropic_model)
        max_tokens = kwargs.get("max_tokens", self.config.anthropic_max_tokens)
        temperature = kwargs.get("temperature", self.config.anthropic_temperature)

        if self.config.log_requests:
            logger.debug(f"Anthropic request: model={model}, messages={len(request_messages)}")

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [m.to_anthropic_format() for m in request_messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            create_kwargs["system"] = system_prompt

        response = await self._async_client.messages.create(**create_kwargs)

        content = ""
        if response.content:
            content = "".join(block.text for block in response.content if hasattr(block, "text"))

        usage = {
            "input_tokens": response.usage.input_tokens if response.usage else 0,
            "output_tokens": response.usage.output_tokens if response.usage else 0,
        }

        if self.config.log_responses:
            logger.debug(f"Anthropic response: {len(content)} chars, {usage}")

        return LLMResponse(
            content=content,
            model=model,
            provider=LLMProvider.ANTHROPIC,
            usage=usage,
            finish_reason=response.stop_reason,
            raw_response=response,
        )

    async def generate_stream(  # type: ignore[override]
        self,
        prompt: str,
        system_prompt: str | None = None,
        messages: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Anthropic."""
        request_messages = self._build_messages(prompt, messages)

        model = kwargs.get("model", self.config.anthropic_model)
        max_tokens = kwargs.get("max_tokens", self.config.anthropic_max_tokens)
        temperature = kwargs.get("temperature", self.config.anthropic_temperature)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [m.to_anthropic_format() for m in request_messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            create_kwargs["system"] = system_prompt

        async with self._async_client.messages.stream(**create_kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    def _build_messages(
        self,
        prompt: str,
        messages: list[LLMMessage] | None,
    ) -> list[LLMMessage]:
        """Build message list for the request."""
        result = []
        if messages:
            result.extend(messages)
        result.append(LLMMessage(role="user", content=prompt))
        return result


class LLMClient:
    """Unified LLM client with automatic provider selection and fallback.

    This client provides a single interface for both OpenAI and Anthropic APIs
    with automatic retry logic and fallback between providers.

    Usage:
        client = LLMClient.from_env()
        response = await client.generate("Hello, world!")
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._clients: dict[LLMProvider, BaseLLMClient] = {}
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize available LLM clients."""
        # Try to initialize Anthropic client
        if self.config.anthropic_api_key and ANTHROPIC_AVAILABLE:
            try:
                self._clients[LLMProvider.ANTHROPIC] = AnthropicClient(self.config)
                logger.info("Anthropic client initialized")
            except LLMConfigError as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")

        # Try to initialize OpenAI client
        if self.config.openai_api_key and OPENAI_AVAILABLE:
            try:
                self._clients[LLMProvider.OPENAI] = OpenAIClient(self.config)
                logger.info("OpenAI client initialized")
            except LLMConfigError as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

        if not self._clients:
            raise LLMConfigError(
                "No LLM clients could be initialized. "
                "Please configure OPENAI_API_KEY or ANTHROPIC_API_KEY",
            )

    @classmethod
    def from_env(cls) -> LLMClient:
        """Create client from environment variables."""
        config = LLMConfig.from_env()
        return cls(config)

    @property
    def available_providers(self) -> list[LLMProvider]:
        """Get list of available providers."""
        return list(self._clients.keys())

    def _get_provider_order(self) -> list[LLMProvider]:
        """Get provider order based on configuration."""
        if self.config.provider == LLMProvider.OPENAI:
            order = [LLMProvider.OPENAI, LLMProvider.ANTHROPIC]
        elif self.config.provider == LLMProvider.ANTHROPIC:
            order = [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]
        else:  # AUTO - prefer Anthropic
            order = [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]

        return [p for p in order if p in self._clients]

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        messages: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response with automatic retry and fallback.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt.
            messages: Optional list of previous messages for context.
            **kwargs: Additional arguments passed to the provider.

        Returns:
            LLMResponse with the generated content.

        Raises:
            LLMError: If all providers fail after retries.
        """
        providers = self._get_provider_order()
        last_error: Exception | None = None

        for provider in providers:
            client = self._clients[provider]

            for attempt in range(self.config.max_retries):
                try:
                    return await client.generate(
                        prompt,
                        system_prompt,
                        messages,
                        **kwargs,
                    )
                except Exception as e:
                    last_error = e
                    delay = self.config.retry_delay * (self.config.retry_exponential_base**attempt)
                    logger.warning(
                        f"Provider {provider.value} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s",
                    )
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(delay)

            logger.warning(f"All retries exhausted for {provider.value}")

        raise LLMError(f"All providers failed. Last error: {last_error}")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        messages: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt.
            messages: Optional list of previous messages for context.
            **kwargs: Additional arguments passed to the provider.

        Yields:
            String chunks as they are received.

        Raises:
            LLMError: If the primary provider fails.
        """
        providers = self._get_provider_order()

        for provider in providers:
            client = self._clients[provider]
            try:
                stream = client.generate_stream(
                    prompt,
                    system_prompt,
                    messages,
                    **kwargs,
                )
                async for chunk in stream:  # type: ignore[attr-defined]
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Stream from {provider.value} failed: {e}")
                continue

        raise LLMError("All providers failed for streaming")

    def generate_sync(
        self,
        prompt: str,
        system_prompt: str | None = None,
        messages: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous wrapper for generate."""
        return asyncio.run(self.generate(prompt, system_prompt, messages, **kwargs))


def create_llm_client(config: LLMConfig | None = None) -> LLMClient:
    """Create an LLM client.

    Args:
        config: Optional configuration. If None, loads from environment.

    Returns:
        Configured LLMClient instance.
    """
    if config is None:
        config = LLMConfig.from_env()
    return LLMClient(config)


# Convenience functions for quick usage
async def generate(prompt: str, **kwargs: Any) -> str:
    """Quick generate function using environment configuration.

    Args:
        prompt: The prompt to send.
        **kwargs: Additional arguments.

    Returns:
        Generated text content.
    """
    client = create_llm_client()
    response = await client.generate(prompt, **kwargs)
    return response.content


def generate_sync(prompt: str, **kwargs: Any) -> str:
    """Synchronous quick generate function.

    Args:
        prompt: The prompt to send.
        **kwargs: Additional arguments.

    Returns:
        Generated text content.
    """
    return asyncio.run(generate(prompt, **kwargs))
