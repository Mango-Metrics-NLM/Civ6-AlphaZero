"""Tests for the LLM-based diplomacy agent.

Covers:
- DiplomacyConfig creation, defaults, and env override
- CoalitionManager alliance / enmity lifecycle
- DiplomacyHistory event recording and trust scores
- DiplomacyAgent with mocked LLM client
- Proposal parsing (valid, malformed, edge-case JSON)
- Graceful degradation when LLM is unavailable
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alpha_zero.multiplayer.diplomacy_agent import (
    CoalitionManager,
    DiplomacyAgent,
    DiplomacyHistory,
    DiplomaticProposal,
    ProposalEvaluation,
)
from alpha_zero.multiplayer.diplomacy_config import DiplomacyConfig

# =========================================================================
# DiplomacyConfig tests
# =========================================================================


class TestDiplomacyConfig:
    """Tests for DiplomacyConfig."""

    def test_default_is_disabled(self) -> None:
        config = DiplomacyConfig.default()
        assert config.enabled is False
        assert config.llm_provider == "auto"
        assert config.model_name is None
        assert config.temperature == 0.7
        assert config.max_tokens == 512

    def test_enabled_default(self) -> None:
        config = DiplomacyConfig.enabled_default()
        assert config.enabled is True
        assert config.llm_provider == "auto"

    def test_proposal_categories(self) -> None:
        config = DiplomacyConfig.default()
        assert "alliance" in config.proposal_categories
        assert "trade" in config.proposal_categories
        assert "peace" in config.proposal_categories
        assert "war_declaration" in config.proposal_categories
        assert "denounce" in config.proposal_categories
        assert len(config.proposal_categories) == 7

    def test_frozen(self) -> None:
        config = DiplomacyConfig.default()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore[misc]

    def test_evaluation_threshold(self) -> None:
        config = DiplomacyConfig(evaluation_threshold=0.8)
        assert config.evaluation_threshold == 0.8

    def test_cache_ttl(self) -> None:
        config = DiplomacyConfig(cache_ttl_seconds=600)
        assert config.cache_ttl_seconds == 600

    def test_from_env_defaults(self) -> None:
        """from_env with no env vars should produce disabled config."""
        with patch.dict(os.environ, {}, clear=True):
            config = DiplomacyConfig.from_env()
        assert config.enabled is False
        assert config.llm_provider == "auto"
        assert config.temperature == 0.7

    def test_from_env_enabled(self) -> None:
        env = {
            "CIV6AZ_DIPLOMACY_ENABLED": "true",
            "CIV6AZ_DIPLOMACY_LLM_PROVIDER": "openai",
            "CIV6AZ_DIPLOMACY_TEMPERATURE": "0.9",
            "CIV6AZ_DIPLOMACY_MAX_TOKENS": "1024",
            "CIV6AZ_DIPLOMACY_EVALUATION_THRESHOLD": "0.6",
            "CIV6AZ_DIPLOMACY_CACHE_TTL_SECONDS": "120",
        }
        with patch.dict(os.environ, env, clear=True):
            config = DiplomacyConfig.from_env()
        assert config.enabled is True
        assert config.llm_provider == "openai"
        assert config.temperature == 0.9
        assert config.max_tokens == 1024
        assert config.evaluation_threshold == 0.6
        assert config.cache_ttl_seconds == 120

    def test_from_env_model_name(self) -> None:
        env = {"CIV6AZ_DIPLOMACY_MODEL_NAME": "gpt-4o"}
        with patch.dict(os.environ, env, clear=True):
            config = DiplomacyConfig.from_env()
        assert config.model_name == "gpt-4o"

    def test_custom_prompt_templates(self) -> None:
        config = DiplomacyConfig(
            evaluation_prompt_template="Custom eval: {game_state} {proposal}",
            generation_prompt_template="Custom gen: {game_state} {player_id} {target_id} {categories}",
        )
        assert "Custom eval" in config.evaluation_prompt_template
        assert "Custom gen" in config.generation_prompt_template


# =========================================================================
# CoalitionManager tests
# =========================================================================


class TestCoalitionManager:
    """Tests for CoalitionManager."""

    def test_initial_state(self) -> None:
        cm = CoalitionManager(num_players=4)
        assert cm.num_players == 4
        assert cm.get_allies(0) == set()
        assert cm.get_enemies(0) == set()

    def test_form_alliance(self) -> None:
        cm = CoalitionManager(num_players=3)
        cm.form_alliance(0, 1)
        assert cm.is_allied(0, 1)
        assert cm.is_allied(1, 0)  # symmetric
        assert not cm.is_allied(0, 2)

    def test_break_alliance(self) -> None:
        cm = CoalitionManager(num_players=3)
        cm.form_alliance(0, 1)
        cm.break_alliance(0, 1)
        assert not cm.is_allied(0, 1)
        assert not cm.is_allied(1, 0)

    def test_break_nonexistent_alliance(self) -> None:
        cm = CoalitionManager(num_players=3)
        cm.break_alliance(0, 1)  # should not raise
        assert not cm.is_allied(0, 1)

    def test_form_alliance_removes_enmity(self) -> None:
        cm = CoalitionManager(num_players=3)
        cm.declare_enemy(0, 1)
        assert cm.is_enemy(0, 1)
        cm.form_alliance(0, 1)
        assert cm.is_allied(0, 1)
        assert not cm.is_enemy(0, 1)

    def test_declare_enemy(self) -> None:
        cm = CoalitionManager(num_players=3)
        cm.declare_enemy(0, 2)
        assert cm.is_enemy(0, 2)
        assert cm.is_enemy(2, 0)
        assert not cm.is_allied(0, 2)

    def test_declare_enemy_removes_alliance(self) -> None:
        cm = CoalitionManager(num_players=3)
        cm.form_alliance(0, 1)
        cm.declare_enemy(0, 1)
        assert not cm.is_allied(0, 1)
        assert cm.is_enemy(0, 1)

    def test_remove_enemy(self) -> None:
        cm = CoalitionManager(num_players=3)
        cm.declare_enemy(0, 1)
        cm.remove_enemy(0, 1)
        assert not cm.is_enemy(0, 1)

    def test_get_allies(self) -> None:
        cm = CoalitionManager(num_players=4)
        cm.form_alliance(0, 1)
        cm.form_alliance(0, 3)
        allies = cm.get_allies(0)
        assert allies == {1, 3}

    def test_get_enemies(self) -> None:
        cm = CoalitionManager(num_players=4)
        cm.declare_enemy(1, 2)
        cm.declare_enemy(1, 3)
        enemies = cm.get_enemies(1)
        assert enemies == {2, 3}

    def test_invalid_player_raises(self) -> None:
        cm = CoalitionManager(num_players=3)
        with pytest.raises(ValueError, match="out of range"):
            cm.form_alliance(0, 5)
        with pytest.raises(ValueError, match="out of range"):
            cm.get_allies(-1)

    def test_self_relationship_raises(self) -> None:
        cm = CoalitionManager(num_players=3)
        with pytest.raises(ValueError, match="Cannot form relationship with self"):
            cm.form_alliance(1, 1)

    def test_get_returns_copy(self) -> None:
        """get_allies / get_enemies should return a copy, not internal state."""
        cm = CoalitionManager(num_players=3)
        cm.form_alliance(0, 1)
        allies = cm.get_allies(0)
        allies.add(2)
        assert 2 not in cm.get_allies(0)


# =========================================================================
# DiplomacyHistory tests
# =========================================================================


class TestDiplomacyHistory:
    """Tests for DiplomacyHistory."""

    @staticmethod
    def _make_proposal(
        proposer: int = 0,
        target: int = 1,
        action_type: str = "alliance",
    ) -> DiplomaticProposal:
        return DiplomaticProposal(
            proposer=proposer,
            target=target,
            action_type=action_type,
            terms={"duration": 10},
            rationale="Test rationale",
        )

    def test_empty_history(self) -> None:
        history = DiplomacyHistory()
        assert history.event_count == 0
        assert history.get_history() == []

    def test_record_event(self) -> None:
        history = DiplomacyHistory()
        proposal = self._make_proposal()
        history.record_event(proposal, accepted=True, turn=5)
        assert history.event_count == 1

    def test_get_history_all(self) -> None:
        history = DiplomacyHistory()
        history.record_event(self._make_proposal(0, 1), accepted=True, turn=1)
        history.record_event(self._make_proposal(2, 3), accepted=False, turn=2)
        events = history.get_history()
        assert len(events) == 2
        assert events[0]["proposer"] == 0
        assert events[0]["accepted"] is True
        assert events[1]["proposer"] == 2
        assert events[1]["accepted"] is False

    def test_get_history_filtered(self) -> None:
        history = DiplomacyHistory()
        history.record_event(self._make_proposal(0, 1), accepted=True, turn=1)
        history.record_event(self._make_proposal(2, 3), accepted=False, turn=2)
        history.record_event(self._make_proposal(1, 0), accepted=True, turn=3)

        # Player 0 is involved in events 0 and 2
        events = history.get_history(player=0)
        assert len(events) == 2
        assert all(e["proposer"] == 0 or e["target"] == 0 for e in events)

    def test_trust_score_no_interactions(self) -> None:
        history = DiplomacyHistory()
        score = history.get_trust_score(0, 1)
        assert score == 0.5  # neutral default

    def test_trust_score_all_accepted(self) -> None:
        history = DiplomacyHistory()
        history.record_event(self._make_proposal(0, 1), accepted=True, turn=1)
        history.record_event(self._make_proposal(1, 0), accepted=True, turn=2)
        score = history.get_trust_score(0, 1)
        assert score == 1.0

    def test_trust_score_all_rejected(self) -> None:
        history = DiplomacyHistory()
        history.record_event(self._make_proposal(0, 1), accepted=False, turn=1)
        history.record_event(self._make_proposal(0, 1), accepted=False, turn=2)
        score = history.get_trust_score(0, 1)
        assert score == 0.0

    def test_trust_score_mixed(self) -> None:
        history = DiplomacyHistory()
        history.record_event(self._make_proposal(0, 1), accepted=True, turn=1)
        history.record_event(self._make_proposal(0, 1), accepted=False, turn=2)
        history.record_event(self._make_proposal(1, 0), accepted=True, turn=3)
        score = history.get_trust_score(0, 1)
        assert score == pytest.approx(2.0 / 3.0)

    def test_trust_score_ignores_other_pairs(self) -> None:
        history = DiplomacyHistory()
        history.record_event(self._make_proposal(0, 1), accepted=True, turn=1)
        history.record_event(self._make_proposal(2, 3), accepted=False, turn=2)
        # Score between 0 and 1 only considers their events
        score = history.get_trust_score(0, 1)
        assert score == 1.0


# =========================================================================
# DiplomaticProposal tests
# =========================================================================


class TestDiplomaticProposal:
    """Tests for DiplomaticProposal dataclass."""

    def test_to_dict(self) -> None:
        proposal = DiplomaticProposal(
            proposer=0,
            target=1,
            action_type="alliance",
            terms={"duration": 10},
            rationale="Strong together",
        )
        d = proposal.to_dict()
        assert d["proposer"] == 0
        assert d["target"] == 1
        assert d["action_type"] == "alliance"
        assert d["terms"] == {"duration": 10}
        assert d["rationale"] == "Strong together"

    def test_defaults(self) -> None:
        proposal = DiplomaticProposal(proposer=0, target=1, action_type="trade")
        assert proposal.terms == {}
        assert proposal.rationale == ""


# =========================================================================
# DiplomacyAgent tests (mocked LLM)
# =========================================================================


def _make_game_state() -> dict[str, Any]:
    """Create a sample game state for testing."""
    return {
        "turn": 50,
        "num_players": 4,
        "players": {
            "0": {"cities": 3, "military": 10, "gold": 500},
            "1": {"cities": 2, "military": 8, "gold": 300},
        },
        "relations": {"0_1": "neutral", "0_2": "friendly"},
    }


class TestDiplomacyAgentDisabled:
    """Tests for DiplomacyAgent when disabled."""

    def test_disabled_agent_evaluate(self) -> None:
        agent = DiplomacyAgent(DiplomacyConfig.default())  # disabled
        proposal = DiplomaticProposal(
            proposer=0,
            target=1,
            action_type="alliance",
        )
        result = asyncio.run(agent.evaluate_proposal(_make_game_state(), proposal))
        assert isinstance(result, ProposalEvaluation)
        assert result.score == 0.5
        assert "disabled" in result.reasoning.lower() or "unavailable" in result.reasoning.lower()

    def test_disabled_agent_generate(self) -> None:
        agent = DiplomacyAgent(DiplomacyConfig.default())
        result = asyncio.run(agent.generate_proposal(_make_game_state(), player_id=0, target_id=1))
        assert result is None


class TestDiplomacyAgentWithMockedLLM:
    """Tests for DiplomacyAgent with a mocked LLM client."""

    @staticmethod
    def _make_agent_with_mock(
        eval_response: str | None = None,
        gen_response: str | None = None,
    ) -> tuple[DiplomacyAgent, AsyncMock]:
        """Create an agent with a mocked LLM client."""
        config = DiplomacyConfig.enabled_default()
        agent = DiplomacyAgent.__new__(DiplomacyAgent)
        agent.config = config

        mock_client = AsyncMock()

        # Configure evaluate mock
        if eval_response is not None:
            mock_resp = MagicMock()
            mock_resp.content = eval_response
            mock_client.generate = AsyncMock(return_value=mock_resp)

        # For generate: reuse the same mock but allow overriding
        if gen_response is not None:
            mock_resp = MagicMock()
            mock_resp.content = gen_response
            mock_client.generate = AsyncMock(return_value=mock_resp)

        agent.llm_client = mock_client
        return agent, mock_client

    def test_evaluate_valid_json(self) -> None:
        response_json = json.dumps(
            {
                "score": 0.8,
                "reasoning": "This alliance strengthens both players",
                "counter_terms": {"duration": 20},
            }
        )
        agent, mock = self._make_agent_with_mock(eval_response=response_json)
        proposal = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
        result = asyncio.run(agent.evaluate_proposal(_make_game_state(), proposal))

        assert result.score == 0.8
        assert "strengthens" in result.reasoning
        assert result.counter_terms == {"duration": 20}
        mock.generate.assert_awaited_once()

    def test_evaluate_score_clamped(self) -> None:
        response_json = json.dumps({"score": 1.5, "reasoning": "Great"})
        agent, _ = self._make_agent_with_mock(eval_response=response_json)
        proposal = DiplomaticProposal(proposer=0, target=1, action_type="trade")
        result = asyncio.run(agent.evaluate_proposal(_make_game_state(), proposal))
        assert result.score == 1.0  # clamped

    def test_evaluate_score_clamped_negative(self) -> None:
        response_json = json.dumps({"score": -0.5, "reasoning": "Bad"})
        agent, _ = self._make_agent_with_mock(eval_response=response_json)
        proposal = DiplomaticProposal(proposer=0, target=1, action_type="trade")
        result = asyncio.run(agent.evaluate_proposal(_make_game_state(), proposal))
        assert result.score == 0.0  # clamped

    def test_evaluate_malformed_json(self) -> None:
        agent, _ = self._make_agent_with_mock(eval_response="not json at all")
        proposal = DiplomaticProposal(proposer=0, target=1, action_type="peace")
        result = asyncio.run(agent.evaluate_proposal(_make_game_state(), proposal))
        assert result.score == 0.5  # fallback
        assert "Failed to parse" in result.reasoning

    def test_evaluate_json_in_code_fence(self) -> None:
        inner = json.dumps({"score": 0.7, "reasoning": "Wrapped in markdown"})
        response = f"```json\n{inner}\n```"
        agent, _ = self._make_agent_with_mock(eval_response=response)
        proposal = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
        result = asyncio.run(agent.evaluate_proposal(_make_game_state(), proposal))
        assert result.score == 0.7

    def test_evaluate_llm_exception(self) -> None:
        config = DiplomacyConfig.enabled_default()
        agent = DiplomacyAgent.__new__(DiplomacyAgent)
        agent.config = config
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        agent.llm_client = mock_client

        proposal = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
        result = asyncio.run(agent.evaluate_proposal(_make_game_state(), proposal))
        assert result.score == 0.5
        assert "failed" in result.reasoning.lower()

    def test_generate_valid_json(self) -> None:
        response_json = json.dumps(
            {
                "action_type": "trade",
                "terms": {"gold_per_turn": 5},
                "rationale": "Mutual benefit",
            }
        )
        agent, mock = self._make_agent_with_mock(gen_response=response_json)
        result = asyncio.run(agent.generate_proposal(_make_game_state(), player_id=0, target_id=1))

        assert result is not None
        assert result.proposer == 0
        assert result.target == 1
        assert result.action_type == "trade"
        assert result.terms == {"gold_per_turn": 5}
        assert result.rationale == "Mutual benefit"

    def test_generate_unknown_action_type_falls_back(self) -> None:
        response_json = json.dumps(
            {
                "action_type": "nuclear_strike",
                "terms": {},
                "rationale": "Boom",
            }
        )
        agent, _ = self._make_agent_with_mock(gen_response=response_json)
        result = asyncio.run(agent.generate_proposal(_make_game_state(), player_id=0, target_id=1))
        assert result is not None
        # Falls back to first category
        assert result.action_type == agent.config.proposal_categories[0]

    def test_generate_malformed_json(self) -> None:
        agent, _ = self._make_agent_with_mock(gen_response="I can't generate JSON")
        result = asyncio.run(agent.generate_proposal(_make_game_state(), player_id=0, target_id=1))
        assert result is None

    def test_generate_llm_exception(self) -> None:
        config = DiplomacyConfig.enabled_default()
        agent = DiplomacyAgent.__new__(DiplomacyAgent)
        agent.config = config
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=ConnectionError("Network error"))
        agent.llm_client = mock_client

        result = asyncio.run(agent.generate_proposal(_make_game_state(), player_id=0, target_id=1))
        assert result is None

    def test_generate_non_dict_terms(self) -> None:
        response_json = json.dumps(
            {
                "action_type": "peace",
                "terms": "not a dict",
                "rationale": "Peace!",
            }
        )
        agent, _ = self._make_agent_with_mock(gen_response=response_json)
        result = asyncio.run(agent.generate_proposal(_make_game_state(), player_id=0, target_id=1))
        assert result is not None
        assert result.terms == {}  # coerced to empty dict

    def test_evaluate_non_dict_counter_terms(self) -> None:
        response_json = json.dumps(
            {
                "score": 0.6,
                "reasoning": "ok",
                "counter_terms": "not a dict",
            }
        )
        agent, _ = self._make_agent_with_mock(eval_response=response_json)
        proposal = DiplomaticProposal(proposer=0, target=1, action_type="trade")
        result = asyncio.run(agent.evaluate_proposal(_make_game_state(), proposal))
        assert result.counter_terms is None  # coerced to None


class TestDiplomacyAgentGracefulDegradation:
    """Tests that the agent degrades gracefully when LLM is unavailable."""

    def test_no_llm_client_evaluate(self) -> None:
        """When create_llm_client fails, evaluate returns neutral."""
        config = DiplomacyConfig.enabled_default()
        with patch(
            "alpha_zero.multiplayer.diplomacy_agent.create_llm_client",
            side_effect=Exception("No API keys"),
        ):
            agent = DiplomacyAgent(config)

        assert agent.llm_client is None
        proposal = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
        result = asyncio.run(agent.evaluate_proposal(_make_game_state(), proposal))
        assert result.score == 0.5

    def test_no_llm_client_generate(self) -> None:
        """When create_llm_client fails, generate returns None."""
        config = DiplomacyConfig.enabled_default()
        with patch(
            "alpha_zero.multiplayer.diplomacy_agent.create_llm_client",
            side_effect=Exception("No API keys"),
        ):
            agent = DiplomacyAgent(config)

        result = asyncio.run(agent.generate_proposal(_make_game_state(), player_id=0, target_id=1))
        assert result is None


class TestDiplomacyAgentJSONExtraction:
    """Tests for the _extract_json static helper."""

    def test_plain_json(self) -> None:
        text = '{"score": 0.5, "reasoning": "ok"}'
        result = DiplomacyAgent._extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.5

    def test_json_in_code_fence(self) -> None:
        text = '```json\n{"score": 0.9}\n```'
        result = DiplomacyAgent._extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.9

    def test_json_with_surrounding_text(self) -> None:
        text = 'Here is my analysis: {"score": 0.3, "reasoning": "weak"} and more'
        result = DiplomacyAgent._extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.3

    def test_nested_json(self) -> None:
        text = '{"score": 0.6, "counter_terms": {"gold": 100}}'
        result = DiplomacyAgent._extract_json(text)
        parsed = json.loads(result)
        assert parsed["counter_terms"]["gold"] == 100

    def test_no_json(self) -> None:
        text = "No JSON here at all"
        result = DiplomacyAgent._extract_json(text)
        assert result == text  # returns text unchanged

    def test_code_fence_no_language(self) -> None:
        text = '```\n{"key": "value"}\n```'
        result = DiplomacyAgent._extract_json(text)
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_braces_in_string_literal_do_not_close_object(self) -> None:
        """Regression: '}' inside a JSON string must not end the object."""
        text = '{"reasoning": "Treaty {fair} accepted", "score": 0.55}'
        result = DiplomacyAgent._extract_json(text)
        parsed = json.loads(result)
        assert parsed["reasoning"] == "Treaty {fair} accepted"
        assert parsed["score"] == 0.55

    def test_escaped_quote_in_string_literal(self) -> None:
        """Escaped \\" must not flip string-mode tracking."""
        text = r'{"q": "say \"}\" loud", "ok": true}'
        result = DiplomacyAgent._extract_json(text)
        parsed = json.loads(result)
        assert parsed["q"] == 'say "}" loud'
        assert parsed["ok"] is True


class TestDiplomacyAgentFormatting:
    """Tests for the internal _format_game_state_prompt helper."""

    def test_format_basic(self) -> None:
        agent = DiplomacyAgent(DiplomacyConfig.default())
        state = {"turn": 10, "num_players": 3}
        result = agent._format_game_state_prompt(state, player_id=0)
        assert "Turn: 10" in result
        assert "Players: 3" in result
        assert "Player perspective: 0" in result

    def test_format_with_players(self) -> None:
        agent = DiplomacyAgent(DiplomacyConfig.default())
        state = {
            "turn": 20,
            "num_players": 2,
            "players": {
                "0": {"cities": 3},
                "1": {"cities": 2},
            },
        }
        result = agent._format_game_state_prompt(state, player_id=0)
        assert "You:" in result  # player 0 is "You"
        assert "Player 1:" in result

    def test_format_with_relations(self) -> None:
        agent = DiplomacyAgent(DiplomacyConfig.default())
        state = {
            "turn": 30,
            "relations": {"0_1": "allied"},
        }
        result = agent._format_game_state_prompt(state, player_id=0)
        assert "Relations:" in result
