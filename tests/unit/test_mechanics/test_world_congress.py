"""Unit tests for alpha_zero/mechanics/world_congress.py."""

from __future__ import annotations

import pytest

from alpha_zero.mechanics.world_congress import (
    DEFAULT_RESOLUTIONS,
    ActiveResolution,
    CongressConfig,
    CongressPhase,
    Emergency,
    EmergencyType,
    ResolutionEffect,
    VoteType,
    WorldCongressManager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> CongressConfig:
    return CongressConfig(
        favor_per_city=0.5,
        favor_per_ally=2.0,
        session_interval_turns=30,
        proposal_turns=5,
        discussion_turns=3,
        voting_turns=5,
        base_votes_per_player=1,
        favor_per_vote=20,
        max_proposals_per_player=1,
    )


@pytest.fixture()
def manager(config: CongressConfig) -> WorldCongressManager:
    return WorldCongressManager(config, num_players=4)


@pytest.fixture()
def manager_in_proposal(manager: WorldCongressManager) -> WorldCongressManager:
    manager.start_session(turn=30)
    return manager


@pytest.fixture()
def manager_in_voting(manager_in_proposal: WorldCongressManager) -> WorldCongressManager:
    m = manager_in_proposal
    m.propose_resolution(player_id=0, resolution_id="trade_policy")
    # Advance through proposal and discussion phases
    for _ in range(m.config.proposal_turns):
        m.process_turn(turn=m._turn + 1)
    for _ in range(m.config.discussion_turns):
        m.process_turn(turn=m._turn + 1)
    return m


# ---------------------------------------------------------------------------
# CongressConfig
# ---------------------------------------------------------------------------


class TestCongressConfig:
    def test_defaults(self) -> None:
        cfg = CongressConfig()
        assert cfg.favor_per_city == pytest.approx(0.5)
        assert cfg.session_interval_turns == 30
        assert cfg.base_votes_per_player == 1

    def test_custom_values(self) -> None:
        cfg = CongressConfig(favor_per_city=1.0, session_interval_turns=20)
        assert cfg.favor_per_city == pytest.approx(1.0)
        assert cfg.session_interval_turns == 20


# ---------------------------------------------------------------------------
# ResolutionEffect / Resolution
# ---------------------------------------------------------------------------


class TestResolutionDataclasses:
    def test_resolution_effect_defaults(self) -> None:
        eff = ResolutionEffect(effect_type="gold_bonus")
        assert eff.value == pytest.approx(0.0)
        assert eff.duration == -1

    def test_resolution_has_options(self) -> None:
        res = DEFAULT_RESOLUTIONS[0]
        assert len(res.options) == 2
        assert res.resolution_id

    def test_default_resolutions_populated(self) -> None:
        ids = {r.resolution_id for r in DEFAULT_RESOLUTIONS}
        assert "trade_policy" in ids
        assert "science_funding" in ids

    def test_active_resolution_defaults(self) -> None:
        res = DEFAULT_RESOLUTIONS[0]
        ar = ActiveResolution(resolution=res)
        assert ar.votes_for == 0
        assert ar.votes_against == 0
        assert ar.winning_option is None


# ---------------------------------------------------------------------------
# WorldCongressManager – favor
# ---------------------------------------------------------------------------


class TestFavor:
    def test_add_favor(self, manager: WorldCongressManager) -> None:
        total = manager.add_favor(0, 10.0, source="city")
        assert total == pytest.approx(10.0)
        assert manager.get_favor(0) == pytest.approx(10.0)

    def test_add_favor_accumulates(self, manager: WorldCongressManager) -> None:
        manager.add_favor(0, 5.0)
        manager.add_favor(0, 7.0)
        assert manager.get_favor(0) == pytest.approx(12.0)

    def test_spend_favor_success(self, manager: WorldCongressManager) -> None:
        manager.add_favor(0, 50.0)
        result = manager.spend_favor(0, 30.0)
        assert result is True
        assert manager.get_favor(0) == pytest.approx(20.0)

    def test_spend_favor_insufficient(self, manager: WorldCongressManager) -> None:
        manager.add_favor(0, 10.0)
        result = manager.spend_favor(0, 20.0)
        assert result is False
        assert manager.get_favor(0) == pytest.approx(10.0)

    def test_get_favor_new_player_returns_zero(self, manager: WorldCongressManager) -> None:
        assert manager.get_favor(99) == pytest.approx(0.0)

    def test_favor_independent_per_player(self, manager: WorldCongressManager) -> None:
        manager.add_favor(0, 10.0)
        manager.add_favor(1, 25.0)
        assert manager.get_favor(0) == pytest.approx(10.0)
        assert manager.get_favor(1) == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# WorldCongressManager – start_session / propose_resolution
# ---------------------------------------------------------------------------


class TestSessionAndProposals:
    def test_start_session_sets_proposal_phase(self, manager: WorldCongressManager) -> None:
        result = manager.start_session(turn=30)
        assert result is True
        assert manager._phase == CongressPhase.PROPOSAL

    def test_start_session_fails_if_already_active(
        self, manager_in_proposal: WorldCongressManager
    ) -> None:
        result = manager_in_proposal.start_session(turn=35)
        assert result is False

    def test_propose_resolution_success(self, manager_in_proposal: WorldCongressManager) -> None:
        result = manager_in_proposal.propose_resolution(0, "trade_policy")
        assert result is True
        assert "trade_policy" in manager_in_proposal._active_resolutions

    def test_propose_unknown_resolution_fails(
        self, manager_in_proposal: WorldCongressManager
    ) -> None:
        result = manager_in_proposal.propose_resolution(0, "nonexistent_id")
        assert result is False

    def test_propose_duplicate_resolution_fails(
        self, manager_in_proposal: WorldCongressManager
    ) -> None:
        manager_in_proposal.propose_resolution(0, "trade_policy")
        result = manager_in_proposal.propose_resolution(1, "trade_policy")
        assert result is False

    def test_propose_outside_proposal_phase_fails(self, manager: WorldCongressManager) -> None:
        result = manager.propose_resolution(0, "trade_policy")
        assert result is False

    def test_propose_limit_enforced(self, manager_in_proposal: WorldCongressManager) -> None:
        manager_in_proposal.propose_resolution(0, "trade_policy")
        # Player 0 already at limit (1)
        result = manager_in_proposal.propose_resolution(0, "science_funding")
        assert result is False


# ---------------------------------------------------------------------------
# WorldCongressManager – cast_vote
# ---------------------------------------------------------------------------


class TestVoting:
    def test_cast_vote_for(self, manager_in_voting: WorldCongressManager) -> None:
        result = manager_in_voting.cast_vote(0, "trade_policy", VoteType.FOR)
        assert result is True
        ar = manager_in_voting._active_resolutions["trade_policy"]
        assert ar.votes_for == 1

    def test_cast_vote_against(self, manager_in_voting: WorldCongressManager) -> None:
        manager_in_voting.cast_vote(0, "trade_policy", VoteType.AGAINST)
        ar = manager_in_voting._active_resolutions["trade_policy"]
        assert ar.votes_against == 1

    def test_cast_vote_abstain_no_count(self, manager_in_voting: WorldCongressManager) -> None:
        manager_in_voting.cast_vote(0, "trade_policy", VoteType.ABSTAIN)
        ar = manager_in_voting._active_resolutions["trade_policy"]
        assert ar.votes_for == 0
        assert ar.votes_against == 0

    def test_double_vote_prevented(self, manager_in_voting: WorldCongressManager) -> None:
        manager_in_voting.cast_vote(0, "trade_policy", VoteType.FOR)
        result = manager_in_voting.cast_vote(0, "trade_policy", VoteType.FOR)
        assert result is False

    def test_extra_votes_spend_favor(self, manager_in_voting: WorldCongressManager) -> None:
        manager_in_voting.add_favor(1, 50.0)
        result = manager_in_voting.cast_vote(1, "trade_policy", VoteType.FOR, extra_votes=2)
        assert result is True
        # 2 extra votes * 20 favor = 40 favor spent
        assert manager_in_voting.get_favor(1) == pytest.approx(10.0)

    def test_extra_votes_insufficient_favor_fails(
        self, manager_in_voting: WorldCongressManager
    ) -> None:
        result = manager_in_voting.cast_vote(2, "trade_policy", VoteType.FOR, extra_votes=5)
        assert result is False

    def test_vote_outside_voting_phase_fails(
        self, manager_in_proposal: WorldCongressManager
    ) -> None:
        manager_in_proposal.propose_resolution(0, "trade_policy")
        result = manager_in_proposal.cast_vote(0, "trade_policy", VoteType.FOR)
        assert result is False

    def test_vote_unknown_resolution_fails(self, manager_in_voting: WorldCongressManager) -> None:
        result = manager_in_voting.cast_vote(0, "ghost_resolution", VoteType.FOR)
        assert result is False


# ---------------------------------------------------------------------------
# WorldCongressManager – phase advancement and vote resolution
# ---------------------------------------------------------------------------


class TestPhaseAdvancement:
    def test_advance_through_all_phases(self, manager_in_proposal: WorldCongressManager) -> None:
        m = manager_in_proposal
        # PROPOSAL → DISCUSSION
        for _ in range(m.config.proposal_turns):
            m.process_turn(turn=m._turn + 1)
        assert m._phase == CongressPhase.DISCUSSION

        # DISCUSSION → VOTING
        for _ in range(m.config.discussion_turns):
            m.process_turn(turn=m._turn + 1)
        assert m._phase == CongressPhase.VOTING

        # VOTING → RESOLUTION
        for _ in range(m.config.voting_turns):
            m.process_turn(turn=m._turn + 1)
        assert m._phase == CongressPhase.RESOLUTION

        # RESOLUTION → INACTIVE
        m.process_turn(turn=m._turn + 1)
        assert m._phase == CongressPhase.INACTIVE

    def test_resolve_votes_winner(self, manager_in_voting: WorldCongressManager) -> None:
        m = manager_in_voting
        m.cast_vote(0, "trade_policy", VoteType.FOR)
        m.cast_vote(1, "trade_policy", VoteType.AGAINST)
        m.cast_vote(2, "trade_policy", VoteType.FOR)
        # FOR=2, AGAINST=1 → option A wins
        for _ in range(m.config.voting_turns):
            m.process_turn(turn=m._turn + 1)
        passed = m._passed_resolutions
        assert len(passed) >= 1
        winning = passed[-1]
        assert winning.winning_option == winning.resolution.options[0]

    def test_resolve_votes_tie_no_winner(self, manager_in_voting: WorldCongressManager) -> None:
        m = manager_in_voting
        m.cast_vote(0, "trade_policy", VoteType.FOR)
        m.cast_vote(1, "trade_policy", VoteType.AGAINST)
        for _ in range(m.config.voting_turns):
            m.process_turn(turn=m._turn + 1)
        passed = m._passed_resolutions
        if passed:
            assert passed[-1].winning_option is None

    def test_active_effects_applied_after_resolution(
        self, manager_in_voting: WorldCongressManager
    ) -> None:
        m = manager_in_voting
        m.cast_vote(0, "trade_policy", VoteType.FOR, extra_votes=0)
        m.cast_vote(1, "trade_policy", VoteType.AGAINST, extra_votes=0)
        m.cast_vote(2, "trade_policy", VoteType.FOR)
        for _ in range(m.config.voting_turns):
            m.process_turn(turn=m._turn + 1)
        # After resolution phase (1 more turn) effects are active
        m.process_turn(turn=m._turn + 1)
        effects = m.get_active_effects()
        assert isinstance(effects, dict)


# ---------------------------------------------------------------------------
# WorldCongressManager – emergencies
# ---------------------------------------------------------------------------


class TestEmergencies:
    def test_call_emergency(self, manager: WorldCongressManager) -> None:
        emergency = manager.call_emergency(
            EmergencyType.MILITARY_EMERGENCY,
            target_player=1,
            target_city="Berlin",
        )
        assert isinstance(emergency, Emergency)
        assert emergency.is_active is True
        assert emergency.emergency_type == EmergencyType.MILITARY_EMERGENCY

    def test_contribute_to_emergency(self, manager: WorldCongressManager) -> None:
        em = manager.call_emergency(EmergencyType.MILITARY_EMERGENCY, target_player=1)
        result = manager.contribute_to_emergency(0, em.emergency_id, 50.0)
        assert result is True
        assert em.participants[0] == pytest.approx(50.0)

    def test_contribute_accumulates(self, manager: WorldCongressManager) -> None:
        em = manager.call_emergency(EmergencyType.MILITARY_EMERGENCY, target_player=1)
        manager.contribute_to_emergency(0, em.emergency_id, 30.0)
        manager.contribute_to_emergency(0, em.emergency_id, 20.0)
        assert em.participants[0] == pytest.approx(50.0)

    def test_contribute_unknown_emergency_fails(self, manager: WorldCongressManager) -> None:
        result = manager.contribute_to_emergency(0, "ghost_em", 10.0)
        assert result is False

    def test_contribute_inactive_emergency_fails(self, manager: WorldCongressManager) -> None:
        em = manager.call_emergency(EmergencyType.MILITARY_EMERGENCY, target_player=1)
        em.is_active = False
        result = manager.contribute_to_emergency(0, em.emergency_id, 10.0)
        assert result is False

    def test_emergency_expires_after_turns(self, manager: WorldCongressManager) -> None:
        em = manager.call_emergency(EmergencyType.NUCLEAR_EMERGENCY, target_player=2)
        em.turns_remaining = 1
        manager.process_turn(turn=1)
        assert em.is_active is False

    def test_reward_pool_stored(self, manager: WorldCongressManager) -> None:
        em = manager.call_emergency(
            EmergencyType.SCORED_COMPETITION,
            target_player=1,
            reward_pool={"gold": 100.0, "science": 50.0},
        )
        assert em.reward_pool["gold"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# WorldCongressManager – get_active_effects
# ---------------------------------------------------------------------------


class TestActiveEffects:
    def test_active_effects_empty_initially(self, manager: WorldCongressManager) -> None:
        effects = manager.get_active_effects()
        assert effects == {}

    def test_active_effects_expire(self, manager: WorldCongressManager) -> None:
        eff = ResolutionEffect("test_bonus", value=5.0, duration=2)
        manager._active_effects.append((eff, 2))
        manager.process_turn(turn=1)
        manager.process_turn(turn=2)
        # After 2 decrements, remaining=0 → expired
        assert len(manager._active_effects) == 0


# ---------------------------------------------------------------------------
# WorldCongressManager – get_congress_status / encode_for_state
# ---------------------------------------------------------------------------


class TestStatusAndEncoding:
    def test_get_congress_status_inactive(self, manager: WorldCongressManager) -> None:
        status = manager.get_congress_status()
        assert status["phase"] == "INACTIVE"
        assert status["active_resolutions"] == 0

    def test_get_congress_status_in_session(
        self, manager_in_proposal: WorldCongressManager
    ) -> None:
        status = manager_in_proposal.get_congress_status()
        assert status["phase"] == "PROPOSAL"

    def test_encode_for_state_returns_dict(self, manager: WorldCongressManager) -> None:
        enc = manager.encode_for_state(0)
        assert "phase" in enc
        assert "player_favor" in enc
        assert "is_in_session" in enc

    def test_encode_favor_reflects_balance(self, manager: WorldCongressManager) -> None:
        manager.add_favor(0, 35.0)
        enc = manager.encode_for_state(0)
        assert enc["player_favor"] == pytest.approx(35.0)

    def test_encode_phase_encoding_length(self, manager: WorldCongressManager) -> None:
        enc = manager.encode_for_state(0)
        assert len(enc["phase_encoding"]) == len(CongressPhase)


# ---------------------------------------------------------------------------
# WorldCongressManager – reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_favor(self, manager: WorldCongressManager) -> None:
        manager.add_favor(0, 100.0)
        manager.reset()
        assert manager.get_favor(0) == pytest.approx(0.0)

    def test_reset_clears_phase(self, manager_in_proposal: WorldCongressManager) -> None:
        manager_in_proposal.reset()
        assert manager_in_proposal._phase == CongressPhase.INACTIVE

    def test_reset_clears_emergencies(self, manager: WorldCongressManager) -> None:
        manager.call_emergency(EmergencyType.CLIMATE_ACCORDS, target_player=0)
        manager.reset()
        assert len(manager._emergencies) == 0

    def test_reset_resets_turn_counter(self, manager: WorldCongressManager) -> None:
        manager.process_turn(50)
        manager.reset()
        assert manager._turn == 0

    def test_auto_session_start_after_interval(self, manager: WorldCongressManager) -> None:
        # After session_interval_turns turns with no active session, a new session starts
        manager.process_turn(turn=manager.config.session_interval_turns)
        assert manager._phase == CongressPhase.PROPOSAL
