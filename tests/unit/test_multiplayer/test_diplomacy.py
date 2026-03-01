"""Comprehensive tests for diplomacy_config and diplomacy_agent modules.

Covers:
- DiplomacyConfig: defaults, factory methods, env-var overrides
- DiplomaticProposal / ProposalEvaluation: data classes and serialization
- CoalitionManager: alliance/enmity CRUD, validation, symmetry
- DiplomacyHistory: event recording, filtering, trust-score computation
- DiplomacyAgent: init paths, LLM-powered and fallback evaluate/generate,
  JSON extraction, parsing helpers
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alpha_zero.multiplayer.diplomacy_agent import (
    CoalitionManager,
    DiplomacyAgent,
    DiplomacyHistory,
    DiplomaticProposal,
    ProposalEvaluation,
    _HistoryEntry,
)
from alpha_zero.multiplayer.diplomacy_config import (
    _DEFAULT_EVALUATION_PROMPT,
    _DEFAULT_GENERATION_PROMPT,
    _DEFAULT_PROPOSAL_CATEGORIES,
    DiplomacyConfig,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def default_config() -> DiplomacyConfig:
    """Default (disabled) diplomacy config."""
    return DiplomacyConfig.default()


@pytest.fixture
def enabled_config() -> DiplomacyConfig:
    """Enabled diplomacy config."""
    return DiplomacyConfig.enabled_default()


@pytest.fixture
def coalition_3p() -> CoalitionManager:
    """3-player coalition manager."""
    return CoalitionManager(num_players=3)


@pytest.fixture
def coalition_4p() -> CoalitionManager:
    """4-player coalition manager."""
    return CoalitionManager(num_players=4)


@pytest.fixture
def history() -> DiplomacyHistory:
    """Empty diplomacy history."""
    return DiplomacyHistory()


@pytest.fixture
def sample_proposal() -> DiplomaticProposal:
    """A sample proposal from player 0 to player 1."""
    return DiplomaticProposal(
        proposer=0,
        target=1,
        action_type="alliance",
        terms={"duration": 10, "shared_vision": True},
        rationale="We face a common enemy.",
    )


@pytest.fixture
def sample_game_state() -> dict:
    """A sample game state dictionary."""
    return {
        "turn": 42,
        "num_players": 4,
        "players": {
            "0": {"score": 100, "cities": 3},
            "1": {"score": 80, "cities": 2},
            "2": {"score": 120, "cities": 4},
        },
        "relations": {"0-1": "neutral", "0-2": "hostile"},
    }


@pytest.fixture
def mock_llm_response():
    """Factory for mock LLM responses."""

    def _make(content: str):
        resp = MagicMock()
        resp.content = content
        return resp

    return _make


@pytest.fixture
def disabled_agent(default_config: DiplomacyConfig) -> DiplomacyAgent:
    """A DiplomacyAgent with diplomacy disabled."""
    return DiplomacyAgent(config=default_config)


# ======================================================================
# PART 1: DiplomacyConfig tests
# ======================================================================


class TestDiplomacyConfigDefaults:
    """Test DiplomacyConfig default values and factory methods."""

    def test_default_is_disabled(self):
        config = DiplomacyConfig.default()
        assert config.enabled is False

    def test_default_field_values(self):
        config = DiplomacyConfig.default()
        assert config.llm_provider == "auto"
        assert config.model_name is None
        assert config.temperature == pytest.approx(0.7)
        assert config.max_tokens == 512
        assert config.evaluation_threshold == pytest.approx(0.5)
        assert config.cache_ttl_seconds == 300

    def test_default_proposal_categories(self):
        config = DiplomacyConfig.default()
        assert config.proposal_categories == _DEFAULT_PROPOSAL_CATEGORIES
        assert "alliance" in config.proposal_categories
        assert "trade" in config.proposal_categories
        assert "peace" in config.proposal_categories
        assert "war_declaration" in config.proposal_categories
        assert "open_borders" in config.proposal_categories
        assert "joint_war" in config.proposal_categories
        assert "denounce" in config.proposal_categories

    def test_default_prompt_templates(self):
        config = DiplomacyConfig.default()
        assert config.evaluation_prompt_template == _DEFAULT_EVALUATION_PROMPT
        assert config.generation_prompt_template == _DEFAULT_GENERATION_PROMPT

    def test_enabled_default(self):
        config = DiplomacyConfig.enabled_default()
        assert config.enabled is True
        # Other fields should be same as default
        assert config.temperature == pytest.approx(0.7)
        assert config.max_tokens == 512

    def test_frozen_dataclass(self):
        config = DiplomacyConfig.default()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore[misc]

    def test_custom_values(self):
        config = DiplomacyConfig(
            enabled=True,
            llm_provider="openai",
            model_name="gpt-4",
            temperature=0.3,
            max_tokens=1024,
            evaluation_threshold=0.8,
            cache_ttl_seconds=600,
        )
        assert config.enabled is True
        assert config.llm_provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.temperature == pytest.approx(0.3)
        assert config.max_tokens == 1024
        assert config.evaluation_threshold == pytest.approx(0.8)
        assert config.cache_ttl_seconds == 600


class TestDiplomacyConfigFromEnv:
    """Test DiplomacyConfig.from_env() with environment variable overrides."""

    def test_from_env_defaults(self):
        """With no env vars set, from_env returns defaults."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = DiplomacyConfig.from_env()
        assert config.enabled is False
        assert config.llm_provider == "auto"
        assert config.model_name is None
        assert config.temperature == pytest.approx(0.7)
        assert config.max_tokens == 512
        assert config.evaluation_threshold == pytest.approx(0.5)
        assert config.cache_ttl_seconds == 300

    def test_from_env_enabled_true(self):
        with mock.patch.dict(os.environ, {"CIV6AZ_DIPLOMACY_ENABLED": "true"}):
            config = DiplomacyConfig.from_env()
        assert config.enabled is True

    def test_from_env_enabled_yes(self):
        with mock.patch.dict(os.environ, {"CIV6AZ_DIPLOMACY_ENABLED": "yes"}):
            config = DiplomacyConfig.from_env()
        assert config.enabled is True

    def test_from_env_enabled_one(self):
        with mock.patch.dict(os.environ, {"CIV6AZ_DIPLOMACY_ENABLED": "1"}):
            config = DiplomacyConfig.from_env()
        assert config.enabled is True

    def test_from_env_enabled_false(self):
        with mock.patch.dict(os.environ, {"CIV6AZ_DIPLOMACY_ENABLED": "false"}):
            config = DiplomacyConfig.from_env()
        assert config.enabled is False

    def test_from_env_enabled_case_insensitive(self):
        with mock.patch.dict(os.environ, {"CIV6AZ_DIPLOMACY_ENABLED": "TRUE"}):
            config = DiplomacyConfig.from_env()
        assert config.enabled is True

    def test_from_env_llm_provider(self):
        with mock.patch.dict(os.environ, {"CIV6AZ_DIPLOMACY_LLM_PROVIDER": "anthropic"}):
            config = DiplomacyConfig.from_env()
        assert config.llm_provider == "anthropic"

    def test_from_env_model_name(self):
        with mock.patch.dict(os.environ, {"CIV6AZ_DIPLOMACY_MODEL_NAME": "gpt-4o"}):
            config = DiplomacyConfig.from_env()
        assert config.model_name == "gpt-4o"

    def test_from_env_model_name_empty_is_none(self):
        with mock.patch.dict(os.environ, {"CIV6AZ_DIPLOMACY_MODEL_NAME": ""}):
            config = DiplomacyConfig.from_env()
        assert config.model_name is None

    def test_from_env_temperature(self):
        with mock.patch.dict(os.environ, {"CIV6AZ_DIPLOMACY_TEMPERATURE": "0.2"}):
            config = DiplomacyConfig.from_env()
        assert config.temperature == pytest.approx(0.2)

    def test_from_env_max_tokens(self):
        with mock.patch.dict(os.environ, {"CIV6AZ_DIPLOMACY_MAX_TOKENS": "2048"}):
            config = DiplomacyConfig.from_env()
        assert config.max_tokens == 2048

    def test_from_env_evaluation_threshold(self):
        with mock.patch.dict(os.environ, {"CIV6AZ_DIPLOMACY_EVALUATION_THRESHOLD": "0.75"}):
            config = DiplomacyConfig.from_env()
        assert config.evaluation_threshold == pytest.approx(0.75)

    def test_from_env_cache_ttl(self):
        with mock.patch.dict(os.environ, {"CIV6AZ_DIPLOMACY_CACHE_TTL_SECONDS": "600"}):
            config = DiplomacyConfig.from_env()
        assert config.cache_ttl_seconds == 600

    def test_from_env_all_overrides(self):
        env = {
            "CIV6AZ_DIPLOMACY_ENABLED": "1",
            "CIV6AZ_DIPLOMACY_LLM_PROVIDER": "openai",
            "CIV6AZ_DIPLOMACY_MODEL_NAME": "gpt-4",
            "CIV6AZ_DIPLOMACY_TEMPERATURE": "0.1",
            "CIV6AZ_DIPLOMACY_MAX_TOKENS": "256",
            "CIV6AZ_DIPLOMACY_EVALUATION_THRESHOLD": "0.9",
            "CIV6AZ_DIPLOMACY_CACHE_TTL_SECONDS": "60",
        }
        with mock.patch.dict(os.environ, env):
            config = DiplomacyConfig.from_env()
        assert config.enabled is True
        assert config.llm_provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.temperature == pytest.approx(0.1)
        assert config.max_tokens == 256
        assert config.evaluation_threshold == pytest.approx(0.9)
        assert config.cache_ttl_seconds == 60


# ======================================================================
# PART 2: DiplomaticProposal tests
# ======================================================================


class TestDiplomaticProposal:
    """Tests for DiplomaticProposal dataclass."""

    def test_basic_creation(self, sample_proposal: DiplomaticProposal):
        assert sample_proposal.proposer == 0
        assert sample_proposal.target == 1
        assert sample_proposal.action_type == "alliance"
        assert sample_proposal.rationale == "We face a common enemy."
        assert sample_proposal.terms == {"duration": 10, "shared_vision": True}

    def test_defaults(self):
        proposal = DiplomaticProposal(proposer=0, target=1, action_type="peace")
        assert proposal.terms == {}
        assert proposal.rationale == ""

    def test_to_dict(self, sample_proposal: DiplomaticProposal):
        d = sample_proposal.to_dict()
        assert d["proposer"] == 0
        assert d["target"] == 1
        assert d["action_type"] == "alliance"
        assert d["terms"]["duration"] == 10
        assert d["terms"]["shared_vision"] is True
        assert d["rationale"] == "We face a common enemy."

    def test_to_dict_roundtrip(self, sample_proposal: DiplomaticProposal):
        d = sample_proposal.to_dict()
        reconstructed = DiplomaticProposal(**d)
        assert reconstructed == sample_proposal

    def test_to_dict_empty_terms(self):
        proposal = DiplomaticProposal(proposer=2, target=3, action_type="denounce")
        d = proposal.to_dict()
        assert d["terms"] == {}
        assert d["rationale"] == ""


# ======================================================================
# PART 3: ProposalEvaluation tests
# ======================================================================


class TestProposalEvaluation:
    """Tests for ProposalEvaluation dataclass."""

    def test_basic_creation(self):
        ev = ProposalEvaluation(score=0.8, reasoning="Good deal")
        assert ev.score == pytest.approx(0.8)
        assert ev.reasoning == "Good deal"
        assert ev.counter_terms is None

    def test_with_counter_terms(self):
        ev = ProposalEvaluation(
            score=0.3,
            reasoning="Needs modification",
            counter_terms={"duration": 5},
        )
        assert ev.counter_terms == {"duration": 5}

    def test_default_counter_terms_none(self):
        ev = ProposalEvaluation(score=0.5, reasoning="neutral")
        assert ev.counter_terms is None


# ======================================================================
# PART 4: CoalitionManager tests
# ======================================================================


class TestCoalitionManagerInit:
    """Test CoalitionManager initialization."""

    def test_init_3_players(self, coalition_3p: CoalitionManager):
        assert coalition_3p.num_players == 3
        for p in range(3):
            assert coalition_3p.get_allies(p) == set()
            assert coalition_3p.get_enemies(p) == set()

    def test_init_single_player(self):
        cm = CoalitionManager(num_players=1)
        assert cm.num_players == 1
        assert cm.get_allies(0) == set()
        assert cm.get_enemies(0) == set()


class TestCoalitionManagerAlliance:
    """Test alliance operations."""

    def test_form_alliance_symmetric(self, coalition_3p: CoalitionManager):
        coalition_3p.form_alliance(0, 1)
        assert coalition_3p.is_allied(0, 1)
        assert coalition_3p.is_allied(1, 0)

    def test_form_alliance_returns_in_get_allies(self, coalition_3p: CoalitionManager):
        coalition_3p.form_alliance(0, 2)
        assert 2 in coalition_3p.get_allies(0)
        assert 0 in coalition_3p.get_allies(2)

    def test_form_alliance_removes_enmity(self, coalition_3p: CoalitionManager):
        coalition_3p.declare_enemy(0, 1)
        assert coalition_3p.is_enemy(0, 1)
        coalition_3p.form_alliance(0, 1)
        assert coalition_3p.is_allied(0, 1)
        assert not coalition_3p.is_enemy(0, 1)

    def test_break_alliance(self, coalition_3p: CoalitionManager):
        coalition_3p.form_alliance(0, 1)
        coalition_3p.break_alliance(0, 1)
        assert not coalition_3p.is_allied(0, 1)
        assert not coalition_3p.is_allied(1, 0)

    def test_break_nonexistent_alliance_noop(self, coalition_3p: CoalitionManager):
        """Breaking a non-existent alliance should not raise."""
        coalition_3p.break_alliance(0, 1)
        assert not coalition_3p.is_allied(0, 1)

    def test_form_alliance_idempotent(self, coalition_3p: CoalitionManager):
        coalition_3p.form_alliance(0, 1)
        coalition_3p.form_alliance(0, 1)
        assert coalition_3p.is_allied(0, 1)
        allies = coalition_3p.get_allies(0)
        assert allies == {1}

    def test_multiple_alliances(self, coalition_4p: CoalitionManager):
        coalition_4p.form_alliance(0, 1)
        coalition_4p.form_alliance(0, 2)
        coalition_4p.form_alliance(0, 3)
        assert coalition_4p.get_allies(0) == {1, 2, 3}
        assert coalition_4p.get_allies(1) == {0}

    def test_get_allies_returns_copy(self, coalition_3p: CoalitionManager):
        coalition_3p.form_alliance(0, 1)
        allies = coalition_3p.get_allies(0)
        allies.add(99)
        assert 99 not in coalition_3p.get_allies(0)


class TestCoalitionManagerEnmity:
    """Test enmity operations."""

    def test_declare_enemy_symmetric(self, coalition_3p: CoalitionManager):
        coalition_3p.declare_enemy(0, 1)
        assert coalition_3p.is_enemy(0, 1)
        assert coalition_3p.is_enemy(1, 0)

    def test_declare_enemy_removes_alliance(self, coalition_3p: CoalitionManager):
        coalition_3p.form_alliance(0, 1)
        coalition_3p.declare_enemy(0, 1)
        assert not coalition_3p.is_allied(0, 1)
        assert coalition_3p.is_enemy(0, 1)

    def test_remove_enemy(self, coalition_3p: CoalitionManager):
        coalition_3p.declare_enemy(0, 2)
        coalition_3p.remove_enemy(0, 2)
        assert not coalition_3p.is_enemy(0, 2)
        assert not coalition_3p.is_enemy(2, 0)

    def test_remove_nonexistent_enemy_noop(self, coalition_3p: CoalitionManager):
        coalition_3p.remove_enemy(0, 1)
        assert not coalition_3p.is_enemy(0, 1)

    def test_get_enemies_returns_copy(self, coalition_3p: CoalitionManager):
        coalition_3p.declare_enemy(0, 1)
        enemies = coalition_3p.get_enemies(0)
        enemies.add(99)
        assert 99 not in coalition_3p.get_enemies(0)

    def test_get_enemies(self, coalition_4p: CoalitionManager):
        coalition_4p.declare_enemy(0, 1)
        coalition_4p.declare_enemy(0, 2)
        assert coalition_4p.get_enemies(0) == {1, 2}
        assert coalition_4p.get_enemies(3) == set()


class TestCoalitionManagerValidation:
    """Test validation in CoalitionManager."""

    def test_validate_player_negative(self, coalition_3p: CoalitionManager):
        with pytest.raises(ValueError, match="out of range"):
            coalition_3p.get_allies(-1)

    def test_validate_player_out_of_range(self, coalition_3p: CoalitionManager):
        with pytest.raises(ValueError, match="out of range"):
            coalition_3p.get_allies(3)

    def test_validate_player_exactly_num_players(self, coalition_3p: CoalitionManager):
        with pytest.raises(ValueError, match="out of range"):
            coalition_3p.get_enemies(3)

    def test_self_alliance_rejected(self, coalition_3p: CoalitionManager):
        with pytest.raises(ValueError, match="Cannot form relationship with self"):
            coalition_3p.form_alliance(1, 1)

    def test_self_enmity_rejected(self, coalition_3p: CoalitionManager):
        with pytest.raises(ValueError, match="Cannot form relationship with self"):
            coalition_3p.declare_enemy(0, 0)

    def test_self_break_alliance_rejected(self, coalition_3p: CoalitionManager):
        with pytest.raises(ValueError, match="Cannot form relationship with self"):
            coalition_3p.break_alliance(2, 2)

    def test_self_remove_enemy_rejected(self, coalition_3p: CoalitionManager):
        with pytest.raises(ValueError, match="Cannot form relationship with self"):
            coalition_3p.remove_enemy(0, 0)

    def test_is_allied_self_rejected(self, coalition_3p: CoalitionManager):
        with pytest.raises(ValueError, match="Cannot form relationship with self"):
            coalition_3p.is_allied(0, 0)

    def test_is_enemy_self_rejected(self, coalition_3p: CoalitionManager):
        with pytest.raises(ValueError, match="Cannot form relationship with self"):
            coalition_3p.is_enemy(1, 1)

    def test_form_alliance_invalid_first(self, coalition_3p: CoalitionManager):
        with pytest.raises(ValueError, match="out of range"):
            coalition_3p.form_alliance(-1, 0)

    def test_form_alliance_invalid_second(self, coalition_3p: CoalitionManager):
        with pytest.raises(ValueError, match="out of range"):
            coalition_3p.form_alliance(0, 5)


# ======================================================================
# PART 5: DiplomacyHistory tests
# ======================================================================


class TestDiplomacyHistoryBasic:
    """Test basic history operations."""

    def test_empty_history(self, history: DiplomacyHistory):
        assert history.event_count == 0
        assert history.get_history() == []

    def test_record_single_event(
        self, history: DiplomacyHistory, sample_proposal: DiplomaticProposal
    ):
        history.record_event(sample_proposal, accepted=True, turn=1)
        assert history.event_count == 1

    def test_record_multiple_events(self, history: DiplomacyHistory):
        p1 = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
        p2 = DiplomaticProposal(proposer=1, target=2, action_type="trade")
        p3 = DiplomaticProposal(proposer=2, target=0, action_type="peace")
        history.record_event(p1, accepted=True, turn=1)
        history.record_event(p2, accepted=False, turn=2)
        history.record_event(p3, accepted=True, turn=3)
        assert history.event_count == 3


class TestDiplomacyHistoryGetHistory:
    """Test history retrieval with and without player filter."""

    def test_get_all_history(self, history: DiplomacyHistory):
        p1 = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
        p2 = DiplomaticProposal(proposer=1, target=2, action_type="trade")
        history.record_event(p1, accepted=True, turn=1)
        history.record_event(p2, accepted=False, turn=2)
        result = history.get_history()
        assert len(result) == 2

    def test_get_history_filtered_proposer(self, history: DiplomacyHistory):
        p1 = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
        p2 = DiplomaticProposal(proposer=1, target=2, action_type="trade")
        history.record_event(p1, accepted=True, turn=1)
        history.record_event(p2, accepted=False, turn=2)
        result = history.get_history(player=0)
        assert len(result) == 1
        assert result[0]["proposer"] == 0

    def test_get_history_filtered_target(self, history: DiplomacyHistory):
        p1 = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
        p2 = DiplomaticProposal(proposer=1, target=2, action_type="trade")
        history.record_event(p1, accepted=True, turn=1)
        history.record_event(p2, accepted=False, turn=2)
        # Player 1 appears as target in p1 and proposer in p2
        result = history.get_history(player=1)
        assert len(result) == 2

    def test_get_history_no_match(self, history: DiplomacyHistory):
        p1 = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
        history.record_event(p1, accepted=True, turn=1)
        result = history.get_history(player=5)
        assert len(result) == 0

    def test_get_history_serialized_format(
        self, history: DiplomacyHistory, sample_proposal: DiplomaticProposal
    ):
        history.record_event(sample_proposal, accepted=True, turn=10)
        result = history.get_history()
        assert len(result) == 1
        entry = result[0]
        assert entry["proposer"] == 0
        assert entry["target"] == 1
        assert entry["action_type"] == "alliance"
        assert entry["terms"]["duration"] == 10
        assert entry["rationale"] == "We face a common enemy."
        assert entry["accepted"] is True
        assert entry["turn"] == 10

    def test_get_history_default_turn(self, history: DiplomacyHistory):
        p = DiplomaticProposal(proposer=0, target=1, action_type="peace")
        history.record_event(p, accepted=False)
        result = history.get_history()
        assert result[0]["turn"] == 0


class TestDiplomacyHistoryTrustScore:
    """Test trust score computation."""

    def test_trust_no_interactions_neutral(self, history: DiplomacyHistory):
        score = history.get_trust_score(0, 1)
        assert score == pytest.approx(0.5)

    def test_trust_all_accepted(self, history: DiplomacyHistory):
        for _ in range(5):
            p = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
            history.record_event(p, accepted=True)
        score = history.get_trust_score(0, 1)
        assert score == pytest.approx(1.0)

    def test_trust_all_rejected(self, history: DiplomacyHistory):
        for _ in range(5):
            p = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
            history.record_event(p, accepted=False)
        score = history.get_trust_score(0, 1)
        assert score == pytest.approx(0.0)

    def test_trust_mixed(self, history: DiplomacyHistory):
        for i in range(4):
            p = DiplomaticProposal(proposer=0, target=1, action_type="trade")
            history.record_event(p, accepted=(i < 3))
        # 3 accepted out of 4 total
        score = history.get_trust_score(0, 1)
        assert score == pytest.approx(0.75)

    def test_trust_symmetric(self, history: DiplomacyHistory):
        p = DiplomaticProposal(proposer=0, target=1, action_type="peace")
        history.record_event(p, accepted=True)
        assert history.get_trust_score(0, 1) == history.get_trust_score(1, 0)

    def test_trust_ignores_other_pairs(self, history: DiplomacyHistory):
        p01 = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
        p02 = DiplomaticProposal(proposer=0, target=2, action_type="alliance")
        history.record_event(p01, accepted=True)
        history.record_event(p02, accepted=False)
        # Trust between 0 and 1 should only consider their interaction
        assert history.get_trust_score(0, 1) == pytest.approx(1.0)
        assert history.get_trust_score(0, 2) == pytest.approx(0.0)

    def test_trust_reverse_proposer_target(self, history: DiplomacyHistory):
        """Trust computation considers both directions of the pair."""
        p1 = DiplomaticProposal(proposer=0, target=1, action_type="trade")
        p2 = DiplomaticProposal(proposer=1, target=0, action_type="trade")
        history.record_event(p1, accepted=True)
        history.record_event(p2, accepted=False)
        # 1 accepted / 2 total = 0.5
        assert history.get_trust_score(0, 1) == pytest.approx(0.5)


# ======================================================================
# PART 6: _HistoryEntry tests
# ======================================================================


class TestHistoryEntry:
    """Test the internal _HistoryEntry dataclass."""

    def test_creation(self, sample_proposal: DiplomaticProposal):
        entry = _HistoryEntry(proposal=sample_proposal, accepted=True, turn=5)
        assert entry.proposal == sample_proposal
        assert entry.accepted is True
        assert entry.turn == 5

    def test_default_turn(self, sample_proposal: DiplomaticProposal):
        entry = _HistoryEntry(proposal=sample_proposal, accepted=False)
        assert entry.turn == 0


# ======================================================================
# PART 7: DiplomacyAgent tests -- init and fallback paths
# ======================================================================


class TestDiplomacyAgentInit:
    """Test DiplomacyAgent initialization."""

    def test_init_default_config(self):
        agent = DiplomacyAgent()
        assert agent.config.enabled is False
        assert agent.llm_client is None

    def test_init_disabled_no_llm(self, default_config: DiplomacyConfig):
        agent = DiplomacyAgent(config=default_config)
        assert agent.llm_client is None

    def test_init_enabled_no_llm_available(self, enabled_config: DiplomacyConfig):
        """When enabled but LLM import fails, client should be None."""
        with patch(
            "alpha_zero.multiplayer.diplomacy_agent._LLM_AVAILABLE", False
        ):
            agent = DiplomacyAgent(config=enabled_config)
            assert agent.llm_client is None

    def test_init_enabled_create_fails(self, enabled_config: DiplomacyConfig):
        """When create_llm_client raises, agent degrades gracefully."""
        mock_create = MagicMock(side_effect=RuntimeError("No API key"))
        with patch(
            "alpha_zero.multiplayer.diplomacy_agent._LLM_AVAILABLE", True
        ), patch(
            "alpha_zero.multiplayer.diplomacy_agent.create_llm_client", mock_create
        ):
            agent = DiplomacyAgent(config=enabled_config)
            assert agent.llm_client is None

    def test_init_enabled_create_succeeds(self, enabled_config: DiplomacyConfig):
        """When create_llm_client succeeds, client is set."""
        mock_client = MagicMock()
        mock_create = MagicMock(return_value=mock_client)
        with patch(
            "alpha_zero.multiplayer.diplomacy_agent._LLM_AVAILABLE", True
        ), patch(
            "alpha_zero.multiplayer.diplomacy_agent.create_llm_client", mock_create
        ):
            agent = DiplomacyAgent(config=enabled_config)
            assert agent.llm_client is mock_client


# ======================================================================
# PART 8: DiplomacyAgent -- evaluate_proposal / generate_proposal
# ======================================================================


class TestDiplomacyAgentEvaluateFallback:
    """Test evaluate_proposal when LLM is unavailable."""

    def test_disabled_returns_neutral(
        self,
        disabled_agent: DiplomacyAgent,
        sample_proposal: DiplomaticProposal,
        sample_game_state: dict,
    ):
        result = disabled_agent.evaluate_proposal_sync(sample_game_state, sample_proposal)
        assert result.score == pytest.approx(0.5)
        assert "disabled" in result.reasoning.lower() or "unavailable" in result.reasoning.lower()

    def test_no_client_returns_neutral(
        self,
        sample_proposal: DiplomaticProposal,
        sample_game_state: dict,
    ):
        config = DiplomacyConfig(enabled=True)
        agent = DiplomacyAgent.__new__(DiplomacyAgent)
        agent.config = config
        agent.llm_client = None
        result = asyncio.run(agent.evaluate_proposal(sample_game_state, sample_proposal))
        assert result.score == pytest.approx(0.5)


class TestDiplomacyAgentGenerateFallback:
    """Test generate_proposal when LLM is unavailable."""

    def test_disabled_returns_none(
        self, disabled_agent: DiplomacyAgent, sample_game_state: dict
    ):
        result = disabled_agent.generate_proposal_sync(sample_game_state, 0, 1)
        assert result is None

    def test_no_client_returns_none(self, sample_game_state: dict):
        config = DiplomacyConfig(enabled=True)
        agent = DiplomacyAgent.__new__(DiplomacyAgent)
        agent.config = config
        agent.llm_client = None
        result = asyncio.run(agent.generate_proposal(sample_game_state, 0, 1))
        assert result is None


class TestDiplomacyAgentEvaluateWithLLM:
    """Test evaluate_proposal with a mock LLM."""

    def _make_agent_with_mock_llm(
        self,
    ) -> tuple[DiplomacyAgent, AsyncMock]:
        config = DiplomacyConfig.enabled_default()
        agent = DiplomacyAgent.__new__(DiplomacyAgent)
        agent.config = config
        agent.llm_client = AsyncMock()
        return agent, agent.llm_client

    def test_evaluate_success(self, sample_proposal, sample_game_state, mock_llm_response):
        agent, llm = self._make_agent_with_mock_llm()
        response_json = json.dumps({"score": 0.8, "reasoning": "Mutually beneficial"})
        llm.generate = AsyncMock(return_value=mock_llm_response(response_json))

        result = asyncio.run(agent.evaluate_proposal(sample_game_state, sample_proposal))
        assert result.score == pytest.approx(0.8)
        assert result.reasoning == "Mutually beneficial"
        assert result.counter_terms is None

    def test_evaluate_with_counter_terms(
        self, sample_proposal, sample_game_state, mock_llm_response
    ):
        agent, llm = self._make_agent_with_mock_llm()
        response_json = json.dumps(
            {
                "score": 0.4,
                "reasoning": "Needs changes",
                "counter_terms": {"duration": 5},
            }
        )
        llm.generate = AsyncMock(return_value=mock_llm_response(response_json))

        result = asyncio.run(agent.evaluate_proposal(sample_game_state, sample_proposal))
        assert result.score == pytest.approx(0.4)
        assert result.counter_terms == {"duration": 5}

    def test_evaluate_clamps_score(self, sample_proposal, sample_game_state, mock_llm_response):
        agent, llm = self._make_agent_with_mock_llm()
        # Score > 1 should be clamped to 1.0
        response_json = json.dumps({"score": 1.5, "reasoning": "over"})
        llm.generate = AsyncMock(return_value=mock_llm_response(response_json))

        result = asyncio.run(agent.evaluate_proposal(sample_game_state, sample_proposal))
        assert result.score == pytest.approx(1.0)

    def test_evaluate_clamps_negative_score(
        self, sample_proposal, sample_game_state, mock_llm_response
    ):
        agent, llm = self._make_agent_with_mock_llm()
        response_json = json.dumps({"score": -0.5, "reasoning": "under"})
        llm.generate = AsyncMock(return_value=mock_llm_response(response_json))

        result = asyncio.run(agent.evaluate_proposal(sample_game_state, sample_proposal))
        assert result.score == pytest.approx(0.0)

    def test_evaluate_invalid_json_fallback(
        self, sample_proposal, sample_game_state, mock_llm_response
    ):
        agent, llm = self._make_agent_with_mock_llm()
        llm.generate = AsyncMock(return_value=mock_llm_response("not valid json at all"))

        result = asyncio.run(agent.evaluate_proposal(sample_game_state, sample_proposal))
        assert result.score == pytest.approx(0.5)
        assert "Failed to parse" in result.reasoning

    def test_evaluate_llm_exception(self, sample_proposal, sample_game_state):
        agent, llm = self._make_agent_with_mock_llm()
        llm.generate = AsyncMock(side_effect=RuntimeError("API error"))

        result = asyncio.run(agent.evaluate_proposal(sample_game_state, sample_proposal))
        assert result.score == pytest.approx(0.5)
        assert "failed" in result.reasoning.lower() or "Evaluation failed" in result.reasoning

    def test_evaluate_counter_terms_non_dict_ignored(
        self, sample_proposal, sample_game_state, mock_llm_response
    ):
        agent, llm = self._make_agent_with_mock_llm()
        response_json = json.dumps(
            {"score": 0.6, "reasoning": "ok", "counter_terms": "invalid"}
        )
        llm.generate = AsyncMock(return_value=mock_llm_response(response_json))

        result = asyncio.run(agent.evaluate_proposal(sample_game_state, sample_proposal))
        assert result.counter_terms is None

    def test_evaluate_missing_score_defaults(
        self, sample_proposal, sample_game_state, mock_llm_response
    ):
        agent, llm = self._make_agent_with_mock_llm()
        response_json = json.dumps({"reasoning": "no score here"})
        llm.generate = AsyncMock(return_value=mock_llm_response(response_json))

        result = asyncio.run(agent.evaluate_proposal(sample_game_state, sample_proposal))
        assert result.score == pytest.approx(0.5)


class TestDiplomacyAgentGenerateWithLLM:
    """Test generate_proposal with a mock LLM."""

    def _make_agent_with_mock_llm(
        self,
    ) -> tuple[DiplomacyAgent, AsyncMock]:
        config = DiplomacyConfig.enabled_default()
        agent = DiplomacyAgent.__new__(DiplomacyAgent)
        agent.config = config
        agent.llm_client = AsyncMock()
        return agent, agent.llm_client

    def test_generate_success(self, sample_game_state, mock_llm_response):
        agent, llm = self._make_agent_with_mock_llm()
        response_json = json.dumps(
            {
                "action_type": "alliance",
                "terms": {"duration": 10},
                "rationale": "Strength in unity",
            }
        )
        llm.generate = AsyncMock(return_value=mock_llm_response(response_json))

        result = asyncio.run(agent.generate_proposal(sample_game_state, 0, 1))
        assert result is not None
        assert result.proposer == 0
        assert result.target == 1
        assert result.action_type == "alliance"
        assert result.terms == {"duration": 10}
        assert result.rationale == "Strength in unity"

    def test_generate_unknown_action_type_fallback(
        self, sample_game_state, mock_llm_response
    ):
        agent, llm = self._make_agent_with_mock_llm()
        response_json = json.dumps(
            {
                "action_type": "bribe",  # not in proposal_categories
                "terms": {},
                "rationale": "test",
            }
        )
        llm.generate = AsyncMock(return_value=mock_llm_response(response_json))

        result = asyncio.run(agent.generate_proposal(sample_game_state, 0, 1))
        assert result is not None
        # Falls back to first category
        assert result.action_type == agent.config.proposal_categories[0]

    def test_generate_invalid_json_returns_none(
        self, sample_game_state, mock_llm_response
    ):
        agent, llm = self._make_agent_with_mock_llm()
        llm.generate = AsyncMock(return_value=mock_llm_response("garbage data"))

        result = asyncio.run(agent.generate_proposal(sample_game_state, 0, 1))
        assert result is None

    def test_generate_llm_exception_returns_none(self, sample_game_state):
        agent, llm = self._make_agent_with_mock_llm()
        llm.generate = AsyncMock(side_effect=RuntimeError("timeout"))

        result = asyncio.run(agent.generate_proposal(sample_game_state, 0, 1))
        assert result is None

    def test_generate_invalid_terms_becomes_empty_dict(
        self, sample_game_state, mock_llm_response
    ):
        agent, llm = self._make_agent_with_mock_llm()
        response_json = json.dumps(
            {
                "action_type": "trade",
                "terms": "not_a_dict",
                "rationale": "test",
            }
        )
        llm.generate = AsyncMock(return_value=mock_llm_response(response_json))

        result = asyncio.run(agent.generate_proposal(sample_game_state, 0, 1))
        assert result is not None
        assert result.terms == {}

    def test_generate_missing_rationale_defaults(
        self, sample_game_state, mock_llm_response
    ):
        agent, llm = self._make_agent_with_mock_llm()
        response_json = json.dumps({"action_type": "peace", "terms": {}})
        llm.generate = AsyncMock(return_value=mock_llm_response(response_json))

        result = asyncio.run(agent.generate_proposal(sample_game_state, 0, 1))
        assert result is not None
        assert result.rationale == ""


# ======================================================================
# PART 9: DiplomacyAgent -- sync wrappers
# ======================================================================


class TestDiplomacyAgentSyncWrappers:
    """Test evaluate_proposal_sync and generate_proposal_sync."""

    def test_evaluate_sync_returns_same_as_async(
        self,
        disabled_agent: DiplomacyAgent,
        sample_proposal: DiplomaticProposal,
        sample_game_state: dict,
    ):
        sync_result = disabled_agent.evaluate_proposal_sync(
            sample_game_state, sample_proposal
        )
        async_result = asyncio.run(
            disabled_agent.evaluate_proposal(sample_game_state, sample_proposal)
        )
        assert sync_result.score == pytest.approx(async_result.score)
        assert sync_result.reasoning == async_result.reasoning

    def test_generate_sync_returns_same_as_async(
        self, disabled_agent: DiplomacyAgent, sample_game_state: dict
    ):
        sync_result = disabled_agent.generate_proposal_sync(sample_game_state, 0, 1)
        async_result = asyncio.run(
            disabled_agent.generate_proposal(sample_game_state, 0, 1)
        )
        assert sync_result == async_result  # Both should be None


# ======================================================================
# PART 10: DiplomacyAgent -- _extract_json
# ======================================================================


class TestExtractJson:
    """Test DiplomacyAgent._extract_json static method."""

    def test_plain_json(self):
        text = '{"score": 0.8}'
        assert DiplomacyAgent._extract_json(text) == '{"score": 0.8}'

    def test_json_with_markdown_fences(self):
        text = '```json\n{"score": 0.8}\n```'
        result = DiplomacyAgent._extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.8

    def test_json_with_plain_fences(self):
        text = '```\n{"score": 0.5}\n```'
        result = DiplomacyAgent._extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.5

    def test_json_with_surrounding_text(self):
        text = 'Here is my evaluation: {"score": 0.7, "reasoning": "ok"} end'
        result = DiplomacyAgent._extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.7
        assert parsed["reasoning"] == "ok"

    def test_nested_json(self):
        text = '{"outer": {"inner": 1}}'
        result = DiplomacyAgent._extract_json(text)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == 1

    def test_no_json_returns_stripped(self):
        text = "no json here"
        result = DiplomacyAgent._extract_json(text)
        assert result == "no json here"

    def test_empty_string(self):
        result = DiplomacyAgent._extract_json("")
        assert result == ""

    def test_whitespace_around_json(self):
        text = '   \n  {"a": 1}  \n  '
        result = DiplomacyAgent._extract_json(text)
        parsed = json.loads(result)
        assert parsed["a"] == 1

    def test_unclosed_brace_returns_from_start(self):
        text = '{"unclosed": true'
        result = DiplomacyAgent._extract_json(text)
        # Fallback: returns from first brace onward
        assert result.startswith("{")


# ======================================================================
# PART 11: DiplomacyAgent -- _format_game_state_prompt
# ======================================================================


class TestFormatGameStatePrompt:
    """Test DiplomacyAgent._format_game_state_prompt."""

    def _make_agent(self) -> DiplomacyAgent:
        agent = DiplomacyAgent.__new__(DiplomacyAgent)
        agent.config = DiplomacyConfig.default()
        agent.llm_client = None
        return agent

    def test_basic_format(self, sample_game_state: dict):
        agent = self._make_agent()
        result = agent._format_game_state_prompt(sample_game_state, player_id=0)
        assert "Player perspective: 0" in result
        assert "Turn: 42" in result
        assert "Players: 4" in result

    def test_player_labeled_you(self, sample_game_state: dict):
        agent = self._make_agent()
        result = agent._format_game_state_prompt(sample_game_state, player_id=0)
        assert "You:" in result

    def test_other_player_labeled(self, sample_game_state: dict):
        agent = self._make_agent()
        result = agent._format_game_state_prompt(sample_game_state, player_id=0)
        assert "Player 1:" in result or "Player 2:" in result

    def test_relations_included(self, sample_game_state: dict):
        agent = self._make_agent()
        result = agent._format_game_state_prompt(sample_game_state, player_id=0)
        assert "Relations:" in result

    def test_empty_game_state(self):
        agent = self._make_agent()
        result = agent._format_game_state_prompt({}, player_id=0)
        assert "Player perspective: 0" in result
        # No turn, no players, no relations
        assert "Turn:" not in result
        assert "Players:" not in result
        assert "Relations:" not in result

    def test_missing_relations(self):
        agent = self._make_agent()
        gs = {"turn": 1, "num_players": 2}
        result = agent._format_game_state_prompt(gs, player_id=0)
        assert "Relations:" not in result

    def test_empty_relations(self):
        agent = self._make_agent()
        gs = {"turn": 1, "relations": {}}
        result = agent._format_game_state_prompt(gs, player_id=0)
        # Empty relations dict is falsy, should not appear
        assert "Relations:" not in result

    def test_players_not_dict(self):
        agent = self._make_agent()
        gs = {"players": ["a", "b"]}
        result = agent._format_game_state_prompt(gs, player_id=0)
        # Non-dict players should be skipped
        assert "Player perspective: 0" in result


# ======================================================================
# PART 12: DiplomacyAgent -- _parse_evaluation_response
# ======================================================================


class TestParseEvaluationResponse:
    """Test DiplomacyAgent._parse_evaluation_response."""

    def _make_agent(self) -> DiplomacyAgent:
        agent = DiplomacyAgent.__new__(DiplomacyAgent)
        agent.config = DiplomacyConfig.default()
        agent.llm_client = None
        return agent

    def test_valid_response(self):
        agent = self._make_agent()
        resp = json.dumps({"score": 0.9, "reasoning": "Excellent"})
        result = agent._parse_evaluation_response(resp)
        assert result.score == pytest.approx(0.9)
        assert result.reasoning == "Excellent"

    def test_score_clamped_high(self):
        agent = self._make_agent()
        resp = json.dumps({"score": 2.0, "reasoning": "over"})
        result = agent._parse_evaluation_response(resp)
        assert result.score == pytest.approx(1.0)

    def test_score_clamped_low(self):
        agent = self._make_agent()
        resp = json.dumps({"score": -1.0, "reasoning": "under"})
        result = agent._parse_evaluation_response(resp)
        assert result.score == pytest.approx(0.0)

    def test_invalid_json(self):
        agent = self._make_agent()
        result = agent._parse_evaluation_response("not json")
        assert result.score == pytest.approx(0.5)
        assert "Failed to parse" in result.reasoning

    def test_counter_terms_non_dict_set_to_none(self):
        agent = self._make_agent()
        resp = json.dumps({"score": 0.5, "reasoning": "ok", "counter_terms": [1, 2]})
        result = agent._parse_evaluation_response(resp)
        assert result.counter_terms is None

    def test_counter_terms_valid_dict(self):
        agent = self._make_agent()
        resp = json.dumps(
            {"score": 0.5, "reasoning": "ok", "counter_terms": {"gold": 100}}
        )
        result = agent._parse_evaluation_response(resp)
        assert result.counter_terms == {"gold": 100}


# ======================================================================
# PART 13: DiplomacyAgent -- _parse_proposal_response
# ======================================================================


class TestParseProposalResponse:
    """Test DiplomacyAgent._parse_proposal_response."""

    def _make_agent(self) -> DiplomacyAgent:
        agent = DiplomacyAgent.__new__(DiplomacyAgent)
        agent.config = DiplomacyConfig.default()
        agent.llm_client = None
        return agent

    def test_valid_response(self):
        agent = self._make_agent()
        resp = json.dumps(
            {"action_type": "trade", "terms": {"gold": 50}, "rationale": "Fair trade"}
        )
        result = agent._parse_proposal_response(resp, proposer=0, target=1)
        assert result is not None
        assert result.proposer == 0
        assert result.target == 1
        assert result.action_type == "trade"
        assert result.terms == {"gold": 50}
        assert result.rationale == "Fair trade"

    def test_unknown_action_type_fallback(self):
        agent = self._make_agent()
        resp = json.dumps({"action_type": "bribe", "terms": {}, "rationale": "x"})
        result = agent._parse_proposal_response(resp, proposer=0, target=1)
        assert result is not None
        assert result.action_type == agent.config.proposal_categories[0]

    def test_invalid_terms_becomes_dict(self):
        agent = self._make_agent()
        resp = json.dumps({"action_type": "peace", "terms": 42, "rationale": "x"})
        result = agent._parse_proposal_response(resp, proposer=0, target=1)
        assert result is not None
        assert result.terms == {}

    def test_invalid_json_returns_none(self):
        agent = self._make_agent()
        result = agent._parse_proposal_response("garbage", proposer=0, target=1)
        assert result is None

    def test_missing_action_type_uses_empty_then_fallback(self):
        agent = self._make_agent()
        resp = json.dumps({"terms": {}, "rationale": "x"})
        result = agent._parse_proposal_response(resp, proposer=0, target=1)
        assert result is not None
        # Empty string not in categories -> falls back to first
        assert result.action_type == agent.config.proposal_categories[0]

    def test_missing_rationale_defaults_empty(self):
        agent = self._make_agent()
        resp = json.dumps({"action_type": "peace", "terms": {}})
        result = agent._parse_proposal_response(resp, proposer=0, target=1)
        assert result is not None
        assert result.rationale == ""


# ======================================================================
# PART 14: Integration-style tests combining components
# ======================================================================


class TestDiplomacyIntegration:
    """Test interactions between diplomacy components."""

    def test_proposal_to_history_roundtrip(self):
        """Record a proposal, retrieve it, check trust score."""
        proposal = DiplomaticProposal(
            proposer=0, target=1, action_type="trade", terms={"gold": 100}
        )
        history = DiplomacyHistory()
        history.record_event(proposal, accepted=True, turn=5)
        events = history.get_history(player=0)
        assert len(events) == 1
        assert events[0]["action_type"] == "trade"
        assert history.get_trust_score(0, 1) == pytest.approx(1.0)

    def test_coalition_then_history(self):
        """Form alliance in coalition, record acceptance in history."""
        cm = CoalitionManager(num_players=3)
        history = DiplomacyHistory()

        proposal = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
        accepted = True

        if accepted:
            cm.form_alliance(proposal.proposer, proposal.target)
        history.record_event(proposal, accepted=accepted, turn=1)

        assert cm.is_allied(0, 1)
        assert history.get_trust_score(0, 1) == pytest.approx(1.0)

    def test_multiple_proposals_trust_evolves(self):
        """Trust score changes as more events are recorded."""
        history = DiplomacyHistory()
        # First event: accepted
        p1 = DiplomaticProposal(proposer=0, target=1, action_type="alliance")
        history.record_event(p1, accepted=True, turn=1)
        assert history.get_trust_score(0, 1) == pytest.approx(1.0)

        # Second event: rejected
        p2 = DiplomaticProposal(proposer=0, target=1, action_type="trade")
        history.record_event(p2, accepted=False, turn=2)
        assert history.get_trust_score(0, 1) == pytest.approx(0.5)

        # Third event: rejected
        p3 = DiplomaticProposal(proposer=1, target=0, action_type="peace")
        history.record_event(p3, accepted=False, turn=3)
        assert history.get_trust_score(0, 1) == pytest.approx(1.0 / 3.0)

    def test_coalition_enemy_then_alliance_transition(self):
        """Transitioning from enemy to ally removes enmity."""
        cm = CoalitionManager(num_players=4)
        cm.declare_enemy(0, 1)
        cm.declare_enemy(0, 2)
        assert cm.get_enemies(0) == {1, 2}

        cm.form_alliance(0, 1)
        assert cm.is_allied(0, 1)
        assert not cm.is_enemy(0, 1)
        assert cm.is_enemy(0, 2)
