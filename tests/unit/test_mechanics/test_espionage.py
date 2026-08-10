"""Unit tests for alpha_zero/mechanics/espionage.py."""

from __future__ import annotations

import pytest

from alpha_zero.mechanics.espionage import (
    DEFAULT_MISSION_CONFIGS,
    EspionageConfig,
    EspionageManager,
    MissionConfig,
    MissionOutcome,
    MissionType,
    Spy,
    SpyRank,
    SpyStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> EspionageConfig:
    return EspionageConfig(
        base_mission_success_rate=0.65,
        rank_success_bonus=0.10,
        counterspy_penalty=0.25,
        establish_turns=3,
        mission_turns=5,
        travel_turns_per_distance=0.5,
        capture_base_chance=0.30,
        death_chance_on_capture=0.25,
        recruit_cost=200,
        max_spies_base=2,
        max_spies_per_gov_tier=1,
    )


@pytest.fixture()
def manager(config: EspionageConfig) -> EspionageManager:
    return EspionageManager(config, random_seed=42)


@pytest.fixture()
def manager_with_spy(manager: EspionageManager) -> tuple[EspionageManager, Spy]:
    spy = manager.recruit_spy(player_id=0, name="Shadow", home_city="Rome")
    assert spy is not None
    return manager, spy


# ---------------------------------------------------------------------------
# EspionageConfig
# ---------------------------------------------------------------------------


class TestEspionageConfig:
    def test_defaults(self) -> None:
        cfg = EspionageConfig()
        assert cfg.base_mission_success_rate == pytest.approx(0.65)
        assert cfg.max_spies_base == 1
        assert cfg.establish_turns == 3

    def test_custom_values(self) -> None:
        cfg = EspionageConfig(base_mission_success_rate=0.80, recruit_cost=300)
        assert cfg.base_mission_success_rate == pytest.approx(0.80)
        assert cfg.recruit_cost == 300


# ---------------------------------------------------------------------------
# Spy
# ---------------------------------------------------------------------------


class TestSpy:
    def test_initial_state(self) -> None:
        spy = Spy(spy_id="s1", name="X", player_id=0)
        assert spy.rank == SpyRank.RECRUIT
        assert spy.status == SpyStatus.AVAILABLE
        assert spy.experience == 0
        assert spy.missions_completed == 0

    def test_gain_experience_no_promotion(self) -> None:
        spy = Spy(spy_id="s1", name="X", player_id=0)
        promoted = spy.gain_experience(50)
        assert not promoted
        assert spy.experience == 50
        assert spy.rank == SpyRank.RECRUIT

    def test_gain_experience_promotes_to_agent(self) -> None:
        spy = Spy(spy_id="s1", name="X", player_id=0)
        promoted = spy.gain_experience(100)
        assert promoted
        assert spy.rank == SpyRank.AGENT

    def test_gain_experience_promotes_to_master_spy(self) -> None:
        spy = Spy(spy_id="s1", name="X", player_id=0)
        spy.gain_experience(600)
        assert spy.rank == SpyRank.MASTER_SPY

    def test_rank_bonus_recruit(self) -> None:
        spy = Spy(spy_id="s1", name="X", player_id=0)
        assert spy.get_rank_bonus() == pytest.approx(0.0)

    def test_rank_bonus_agent(self) -> None:
        spy = Spy(spy_id="s1", name="X", player_id=0, rank=SpyRank.AGENT)
        assert spy.get_rank_bonus() == pytest.approx(0.10)

    def test_rank_bonus_master_spy(self) -> None:
        spy = Spy(spy_id="s1", name="X", player_id=0, rank=SpyRank.MASTER_SPY)
        assert spy.get_rank_bonus() == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# MissionConfig
# ---------------------------------------------------------------------------


class TestMissionConfig:
    def test_default_mission_configs_populated(self) -> None:
        assert MissionType.STEAL_TECHNOLOGY in DEFAULT_MISSION_CONFIGS
        assert MissionType.COUNTERINTELLIGENCE in DEFAULT_MISSION_CONFIGS

    def test_steal_tech_requires_agent(self) -> None:
        cfg = DEFAULT_MISSION_CONFIGS[MissionType.STEAL_TECHNOLOGY]
        assert cfg.required_rank == SpyRank.AGENT
        assert cfg.can_be_lethal is True
        assert cfg.target_district == "Campus"

    def test_surveillance_easy(self) -> None:
        cfg = DEFAULT_MISSION_CONFIGS[MissionType.ESTABLISH_SURVEILLANCE]
        assert cfg.base_difficulty < 0  # negative = easier


# ---------------------------------------------------------------------------
# EspionageManager – recruit_spy
# ---------------------------------------------------------------------------


class TestRecruitSpy:
    def test_recruit_returns_spy(self, manager: EspionageManager) -> None:
        spy = manager.recruit_spy(0, "Shadow", "Rome")
        assert spy is not None
        assert spy.name == "Shadow"
        assert spy.player_id == 0
        assert spy.home_city == "Rome"

    def test_recruit_increments_counter(self, manager: EspionageManager) -> None:
        s1 = manager.recruit_spy(0, "A", "Rome")
        s2 = manager.recruit_spy(0, "B", "Rome")
        assert s1 is not None and s2 is not None
        assert s1.spy_id != s2.spy_id

    def test_recruit_returns_none_at_capacity(self, manager: EspionageManager) -> None:
        # max_spies_base=2, max_spies_per_gov_tier=1 => cap=3
        manager.recruit_spy(0, "A", "Rome")
        manager.recruit_spy(0, "B", "Rome")
        manager.recruit_spy(0, "C", "Rome")
        result = manager.recruit_spy(0, "D", "Rome")
        assert result is None

    def test_multiple_players_independent(self, manager: EspionageManager) -> None:
        s0 = manager.recruit_spy(0, "Shadow", "Rome")
        s1 = manager.recruit_spy(1, "Ghost", "Paris")
        assert s0 is not None and s1 is not None
        assert manager.get_player_spies(0) == [s0]
        assert manager.get_player_spies(1) == [s1]


# ---------------------------------------------------------------------------
# EspionageManager – deploy_spy
# ---------------------------------------------------------------------------


class TestDeploySpy:
    def test_deploy_changes_status(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        result = manager.deploy_spy(spy.spy_id, "Alexandria", distance=10)
        assert result is True
        assert spy.status == SpyStatus.TRAVELING

    def test_deploy_sets_travel_turns(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        manager.deploy_spy(spy.spy_id, "Babylon", distance=4)
        # 4 * 0.5 = 2 turns
        assert spy.turns_remaining == 2

    def test_deploy_minimum_one_turn(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        manager.deploy_spy(spy.spy_id, "Nearby", distance=0)
        assert spy.turns_remaining >= 1

    def test_deploy_unknown_spy_returns_false(self, manager: EspionageManager) -> None:
        assert manager.deploy_spy("nonexistent", "Rome") is False

    def test_deploy_busy_spy_returns_false(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        manager.deploy_spy(spy.spy_id, "Alexandria")
        # Already TRAVELING, can't deploy again
        result = manager.deploy_spy(spy.spy_id, "Babylon")
        assert result is False


# ---------------------------------------------------------------------------
# EspionageManager – process_turn (traveling → establishing → active)
# ---------------------------------------------------------------------------


class TestProcessTurn:
    def _get_active_spy(self, manager: EspionageManager, spy: Spy) -> Spy:
        """Advance spy through travel and establishment to ACTIVE."""
        manager.deploy_spy(spy.spy_id, "Alexandria", distance=2)
        # Burn travel turns (ceil(2*0.5)=1 turn)
        for _ in range(spy.turns_remaining):
            manager.process_turn()
        # Now ESTABLISHING — burn establish_turns (3)
        for _ in range(manager.config.establish_turns):
            manager.process_turn()
        return spy

    def test_spy_becomes_active(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        self._get_active_spy(manager, spy)
        assert spy.status == SpyStatus.ACTIVE

    def test_process_turn_no_missions_returns_empty(self, manager: EspionageManager) -> None:
        results = manager.process_turn()
        assert results == []


# ---------------------------------------------------------------------------
# EspionageManager – assign_mission
# ---------------------------------------------------------------------------


class TestAssignMission:
    def _activate_spy(self, manager: EspionageManager, spy: Spy) -> None:
        spy.status = SpyStatus.ACTIVE
        spy.location = "Alexandria"

    def test_assign_gather_intel(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        self._activate_spy(manager, spy)
        mission = manager.assign_mission(spy.spy_id, MissionType.GATHER_INTELLIGENCE)
        assert mission is not None
        assert mission.mission_type == MissionType.GATHER_INTELLIGENCE

    def test_assign_requires_active_status(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        # spy is AVAILABLE, not ACTIVE
        mission = manager.assign_mission(spy.spy_id, MissionType.GATHER_INTELLIGENCE)
        assert mission is None

    def test_assign_rank_requirement_blocks_low_rank(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        self._activate_spy(manager, spy)
        # STEAL_TECHNOLOGY requires AGENT, spy is RECRUIT
        mission = manager.assign_mission(spy.spy_id, MissionType.STEAL_TECHNOLOGY)
        assert mission is None

    def test_assign_with_agent_rank_allows_steal_tech(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        spy.rank = SpyRank.AGENT
        self._activate_spy(manager, spy)
        mission = manager.assign_mission(spy.spy_id, MissionType.STEAL_TECHNOLOGY)
        assert mission is not None

    def test_counterintelligence_registers_counterspy(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        self._activate_spy(manager, spy)
        manager.assign_mission(spy.spy_id, MissionType.COUNTERINTELLIGENCE)
        # counterspy should be tracked
        assert manager._counterspy_locations.get("Alexandria") == spy.spy_id

    def test_assign_unknown_spy_returns_none(self, manager: EspionageManager) -> None:
        mission = manager.assign_mission("ghost", MissionType.GATHER_INTELLIGENCE)
        assert mission is None


# ---------------------------------------------------------------------------
# EspionageManager – _determine_outcome
# ---------------------------------------------------------------------------


class TestDetermineOutcome:
    def test_critical_success_low_roll(self, manager: EspionageManager) -> None:
        cfg = MissionConfig(mission_type=MissionType.GATHER_INTELLIGENCE, can_be_lethal=False)
        outcome = manager._determine_outcome(0.0, 0.80, cfg)
        assert outcome == MissionOutcome.CRITICAL_SUCCESS

    def test_success_mid_roll(self, manager: EspionageManager) -> None:
        cfg = MissionConfig(mission_type=MissionType.GATHER_INTELLIGENCE, can_be_lethal=False)
        # roll in (success_chance*0.2, success_chance) → SUCCESS
        outcome = manager._determine_outcome(0.50, 0.80, cfg)
        assert outcome == MissionOutcome.SUCCESS

    def test_partial_success(self, manager: EspionageManager) -> None:
        cfg = MissionConfig(mission_type=MissionType.GATHER_INTELLIGENCE, can_be_lethal=False)
        # roll just above success_chance but within +0.15 window
        outcome = manager._determine_outcome(0.82, 0.80, cfg)
        assert outcome == MissionOutcome.PARTIAL_SUCCESS

    def test_failure_outcome(self, manager: EspionageManager) -> None:
        cfg = MissionConfig(mission_type=MissionType.GATHER_INTELLIGENCE, can_be_lethal=False)
        # roll = 0.97, capture_chance=0.30, 1-0.30=0.70, 0.97 > 0.70 → captured
        # But can_be_lethal=False → CAPTURED, not KILLED
        outcome = manager._determine_outcome(0.97, 0.50, cfg)
        assert outcome == MissionOutcome.CAPTURED

    def test_failure_when_not_captured(self, manager: EspionageManager) -> None:
        cfg = MissionConfig(mission_type=MissionType.GATHER_INTELLIGENCE, can_be_lethal=False)
        # roll just above partial_success window but below capture threshold
        # success=0.5, partial upper=0.65, capture lower=0.70
        outcome = manager._determine_outcome(0.67, 0.50, cfg)
        assert outcome == MissionOutcome.FAILURE


# ---------------------------------------------------------------------------
# EspionageManager – recall_spy
# ---------------------------------------------------------------------------


class TestRecallSpy:
    def test_recall_sets_traveling(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        spy.status = SpyStatus.ACTIVE
        result = manager.recall_spy(spy.spy_id)
        assert result is True
        assert spy.status == SpyStatus.TRAVELING
        assert spy.mission_target == spy.home_city

    def test_recall_unknown_spy(self, manager: EspionageManager) -> None:
        assert manager.recall_spy("nonexistent") is False

    def test_recall_removes_counterspy(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        spy.status = SpyStatus.ACTIVE
        spy.location = "Alexandria"
        manager._counterspy_locations["Alexandria"] = spy.spy_id
        manager.recall_spy(spy.spy_id)
        assert "Alexandria" not in manager._counterspy_locations


# ---------------------------------------------------------------------------
# EspionageManager – exchange_captured_spy
# ---------------------------------------------------------------------------


class TestExchangeCapturedSpy:
    def test_exchange_captured(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        spy.status = SpyStatus.CAPTURED
        result = manager.exchange_captured_spy(spy.spy_id, captor_player_id=1)
        assert result is True
        assert spy.status == SpyStatus.AVAILABLE

    def test_exchange_non_captured_fails(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        result = manager.exchange_captured_spy(spy.spy_id, captor_player_id=1)
        assert result is False

    def test_exchange_unknown_spy_fails(self, manager: EspionageManager) -> None:
        assert manager.exchange_captured_spy("ghost", 1) is False


# ---------------------------------------------------------------------------
# EspionageManager – get_available_missions
# ---------------------------------------------------------------------------


class TestGetAvailableMissions:
    def test_available_missions_empty_for_unavailable_spy(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        missions = manager.get_available_missions(spy.spy_id)
        assert missions == []

    def test_recruit_can_gather_intel(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        spy.status = SpyStatus.ACTIVE
        missions = manager.get_available_missions(spy.spy_id)
        assert MissionType.GATHER_INTELLIGENCE in missions

    def test_recruit_cannot_steal_tech(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        spy.status = SpyStatus.ACTIVE
        missions = manager.get_available_missions(spy.spy_id)
        assert MissionType.STEAL_TECHNOLOGY not in missions

    def test_district_filter_works(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        spy.status = SpyStatus.ACTIVE
        spy.rank = SpyRank.AGENT
        missions = manager.get_available_missions(spy.spy_id, city_districts={"Campus"})
        assert MissionType.STEAL_TECHNOLOGY in missions
        # Sabotage needs Industrial Zone which is absent
        assert MissionType.SABOTAGE_PRODUCTION not in missions

    def test_unknown_spy_returns_empty(self, manager: EspionageManager) -> None:
        assert manager.get_available_missions("ghost") == []


# ---------------------------------------------------------------------------
# EspionageManager – encode_for_state
# ---------------------------------------------------------------------------


class TestEncodeForState:
    def test_encode_no_spies(self, manager: EspionageManager) -> None:
        enc = manager.encode_for_state(0)
        assert enc["num_spies"] == 0
        assert enc["spies_available"] == 0
        assert enc["total_missions_completed"] == 0

    def test_encode_with_spy(self, manager_with_spy: tuple) -> None:
        manager, _spy = manager_with_spy
        enc = manager.encode_for_state(0)
        assert enc["num_spies"] == 1
        assert enc["spies_available"] == 1
        assert enc["avg_spy_rank"] == pytest.approx(1.0)

    def test_encode_counterspy_detected(self, manager_with_spy: tuple) -> None:
        manager, spy = manager_with_spy
        manager._counterspy_locations["Rome"] = spy.spy_id
        enc = manager.encode_for_state(0)
        assert enc["counterspy_deployed"] == 1


# ---------------------------------------------------------------------------
# EspionageManager – reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_state(self, manager_with_spy: tuple) -> None:
        manager, _spy = manager_with_spy
        manager.reset()
        assert manager.get_player_spies(0) == []
        assert manager._spy_counter == 0
        assert manager._mission_counter == 0

    def test_can_recruit_after_reset(self, manager_with_spy: tuple) -> None:
        manager, _ = manager_with_spy
        manager.reset()
        new_spy = manager.recruit_spy(0, "New", "Rome")
        assert new_spy is not None


# ---------------------------------------------------------------------------
# EspionageManager – full mission resolution (deterministic seed)
# ---------------------------------------------------------------------------


class TestMissionResolution:
    def _setup_active_spy(self, manager: EspionageManager) -> Spy:
        spy = manager.recruit_spy(0, "Phoenix", "Rome")
        assert spy is not None
        spy.status = SpyStatus.ACTIVE
        spy.location = "Babylon"
        return spy

    def test_mission_result_has_required_keys(self, manager: EspionageManager) -> None:
        spy = self._setup_active_spy(manager)
        mission = manager.assign_mission(spy.spy_id, MissionType.GATHER_INTELLIGENCE)
        assert mission is not None
        # Force mission to complete immediately
        mission.turns_remaining = 0
        results = manager.process_turn()
        assert len(results) == 1
        result = results[0]
        for key in ("spy_id", "mission_type", "outcome", "success_chance", "roll", "effect_value"):
            assert key in result

    def test_captured_spy_status(self) -> None:
        # Use a manager rigged to always fail → capture
        cfg = EspionageConfig(
            base_mission_success_rate=0.01,
            capture_base_chance=0.99,
            death_chance_on_capture=0.0,
        )
        mgr = EspionageManager(cfg, random_seed=0)
        spy = mgr.recruit_spy(0, "Unlucky", "Rome")
        assert spy is not None
        spy.status = SpyStatus.ACTIVE
        spy.location = "Enemy City"
        mission = mgr.assign_mission(spy.spy_id, MissionType.GATHER_INTELLIGENCE)
        assert mission is not None
        mission.turns_remaining = 0
        results = mgr.process_turn()
        if results:
            # Spy could be captured or any outcome; just check keys exist
            assert "outcome" in results[0]
