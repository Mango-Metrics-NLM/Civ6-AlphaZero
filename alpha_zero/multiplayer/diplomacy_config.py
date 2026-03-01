"""Configuration for LLM-based diplomatic reasoning.

Provides a frozen dataclass for all diplomacy agent settings, including
LLM provider selection, prompt parameters, proposal categories, and
evaluation thresholds. All values are overridable via environment
variables with the ``CIV6AZ_DIPLOMACY_`` prefix.

Usage:
    from alpha_zero.multiplayer.diplomacy_config import DiplomacyConfig

    config = DiplomacyConfig.default()
    config = DiplomacyConfig.enabled_default()
    config = DiplomacyConfig.from_env()
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_PROPOSAL_CATEGORIES: tuple[str, ...] = (
    "alliance",
    "trade",
    "peace",
    "war_declaration",
    "open_borders",
    "joint_war",
    "denounce",
)

_DEFAULT_EVALUATION_PROMPT = (
    "You are a diplomatic advisor for a Civilization 6 AI player. "
    "Evaluate the following diplomatic proposal and respond with a JSON object "
    'containing "score" (float 0-1), "reasoning" (string), and optionally '
    '"counter_terms" (object with modified terms).\n\n'
    "Game State:\n{game_state}\n\n"
    "Proposal:\n{proposal}\n\n"
    "Respond ONLY with valid JSON."
)

_DEFAULT_GENERATION_PROMPT = (
    "You are a diplomatic advisor for a Civilization 6 AI player. "
    "Based on the game state, generate a diplomatic proposal from player "
    "{player_id} to player {target_id}. Respond with a JSON object containing "
    '"action_type" (one of {categories}), "terms" (object), and '
    '"rationale" (string).\n\n'
    "Game State:\n{game_state}\n\n"
    "Respond ONLY with valid JSON."
)


@dataclass(frozen=True)
class DiplomacyConfig:
    """Configuration for the diplomacy agent.

    Attributes:
        enabled: Whether the diplomacy agent is active.
        llm_provider: LLM provider to use (auto, openai, anthropic).
        model_name: Model name override; None uses llm_client default.
        temperature: Sampling temperature for LLM calls.
        max_tokens: Maximum tokens in LLM responses.
        proposal_categories: Allowed diplomatic proposal types.
        evaluation_threshold: Minimum score (0-1) to accept a proposal.
        cache_ttl_seconds: Time-to-live for cached LLM responses.
        evaluation_prompt_template: Template for evaluation prompts.
        generation_prompt_template: Template for proposal generation prompts.
    """

    enabled: bool = False
    llm_provider: str = "auto"
    model_name: str | None = None
    temperature: float = 0.7
    max_tokens: int = 512
    proposal_categories: tuple[str, ...] = _DEFAULT_PROPOSAL_CATEGORIES
    evaluation_threshold: float = 0.5
    cache_ttl_seconds: int = 300
    evaluation_prompt_template: str = _DEFAULT_EVALUATION_PROMPT
    generation_prompt_template: str = _DEFAULT_GENERATION_PROMPT

    @classmethod
    def default(cls) -> DiplomacyConfig:
        """Create a default (disabled) configuration."""
        return cls()

    @classmethod
    def enabled_default(cls) -> DiplomacyConfig:
        """Create a default configuration with diplomacy enabled."""
        return cls(enabled=True)

    @classmethod
    def from_env(cls) -> DiplomacyConfig:
        """Create configuration from ``CIV6AZ_DIPLOMACY_`` environment variables.

        Supported variables:
            CIV6AZ_DIPLOMACY_ENABLED
            CIV6AZ_DIPLOMACY_LLM_PROVIDER
            CIV6AZ_DIPLOMACY_MODEL_NAME
            CIV6AZ_DIPLOMACY_TEMPERATURE
            CIV6AZ_DIPLOMACY_MAX_TOKENS
            CIV6AZ_DIPLOMACY_EVALUATION_THRESHOLD
            CIV6AZ_DIPLOMACY_CACHE_TTL_SECONDS

        Returns:
            DiplomacyConfig populated from environment.
        """
        def _get(key: str, default: str | None = None) -> str | None:
            return os.getenv(f"CIV6AZ_DIPLOMACY_{key}", default)

        enabled_str = _get("ENABLED", "false")
        enabled = enabled_str is not None and enabled_str.lower() in ("true", "1", "yes")

        model_name = _get("MODEL_NAME")

        return cls(
            enabled=enabled,
            llm_provider=_get("LLM_PROVIDER", "auto") or "auto",
            model_name=model_name if model_name else None,
            temperature=float(_get("TEMPERATURE", "0.7") or "0.7"),
            max_tokens=int(_get("MAX_TOKENS", "512") or "512"),
            evaluation_threshold=float(_get("EVALUATION_THRESHOLD", "0.5") or "0.5"),
            cache_ttl_seconds=int(_get("CACHE_TTL_SECONDS", "300") or "300"),
        )
