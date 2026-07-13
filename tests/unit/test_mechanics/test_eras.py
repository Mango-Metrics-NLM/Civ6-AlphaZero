"""Unit tests for alpha_zero/mechanics/eras.py."""

from __future__ import annotations

import pytest

from alpha_zero.mechanics.eras import (
    DEFAULT_DEDICATIONS,
    DEFAULT_MOMENT_CONFIGS,
    AgeType,
    DedicationType,
    Era,
    EraConfig,
    EraManager,
    EraProgress,
    HistoricMoment,
    HistoricMomentConfig,
    HistoricMomentType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> EraConfig:
    return EraConfig(
        dark_age_threshold_ratio=0.5,
        golden_age_threshold_ratio=1.0,
        base_era_score_threshold=12,
        era_scaling_factor=1.5,
        dark_age_loyalty_penalty=3,
        golden_age_loyalty_bonus=3,
        max_dedications_golden=1,
        max_dedications_heroic=3,
    )


@pytest.fixture()
def manager(config: EraConfig) -> EraManager:
    return EraManager(config)


# ---------------------------------------------------------------------------
# EraConfig
# ---------------------------------------------------------------------------


class TestEraConfig:
    def test_defaults(self) -> None:
        cfg = EraConfig()
        assert cfg.dark_age_threshold_ratio == pytest.approx(0.5)
        assert cfg.base_era_score_threshold == 12
        assert cfg.max_dedications_golden == 1

    def test_custom_values(self) -> None:
        cfg = EraConfig(base_era_score_threshold=20, max_dedications_heroic=5)
        assert cfg.base_era_score_threshold == 20
        assert cfg.max_dedications_heroic == 5


# ---------------------------------------------------------------------------
# HistoricMomentConfig / defaults
# ---------------------------------------------------------------------------


class TestHistoricMomentConfig:
    def test_default_configs_populated(self) -> None:
        assert HistoricMomentType.WONDER_BUILT in DEFAULT_MOMENT_CONFIGS
        assert HistoricMomentType.CITY_FOUNDED in DEFAULT_MOMENT_CONFIGS

    def test_circumnavigation_once_per_game(self) -> None:
        cfg = DEFAULT_MOMENT_CONFIGS[HistoricMomentType.CIRCUMNAVIGATION]
        assert cfg.once_per_game is True
        assert cfg.base_score == 5

    def test_tech_researched_era_scaling(self) -> None:
        cfg = DEFAULT_MOMENT_CONFIGS[HistoricMomentType.TECH_RESEARCHED]
        assert cfg.era_scaling is True

    def test_first_contact_no_once_per_era(self) -> None:
        cfg = DEFAULT_MOMENT_CONFIGS[HistoricMomentType.FIRST_CONTACT]
        assert cfg.once_per_era is False


# ---------------------------------------------------------------------------
# Dedications
# ---------------------------------------------------------------------------


class TestDedications:
    def test_default_dedications_populated(self) -> None:
        assert DedicationType.MONUMENTALITY in DEFAULT_DEDICATIONS
        assert DedicationType.FREE_INQUIRY in DEFAULT_DEDICATIONS

    def test_free_inquiry_has_effects(self) -> None:
        ded = DEFAULT_DEDICATIONS[DedicationType.FREE_INQUIRY]
        assert "commercial_harbor_science" in ded.effects
        assert "campus_science_bonus" in ded.golden_effects

    def test_reform_the_coinage_gold_effect(self) -> None:
        ded = DEFAULT_DEDICATIONS[DedicationType.REFORM_THE_COINAGE]
        assert ded.effects.get("city_gold_bonus", 0.0) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# EraManager – _get_progress
# ---------------------------------------------------------------------------


class TestGetProgress:
    def test_creates_default_progress(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        assert isinstance(progress, EraProgress)
        assert progress.current_era == Era.ANCIENT
        assert progress.age_type == AgeType.NORMAL

    def test_returns_same_instance(self, manager: EraManager) -> None:
        p1 = manager._get_progress(0)
        p2 = manager._get_progress(0)
        assert p1 is p2

    def test_independent_per_player(self, manager: EraManager) -> None:
        p0 = manager._get_progress(0)
        p1 = manager._get_progress(1)
        assert p0 is not p1


# ---------------------------------------------------------------------------
# EraManager – record_moment
# ---------------------------------------------------------------------------


class TestRecordMoment:
    def test_record_returns_moment(self, manager: EraManager) -> None:
        moment = manager.record_moment(0, HistoricMomentType.CITY_FOUNDED, turn=10)
        assert isinstance(moment, HistoricMoment)
        assert moment.moment_type == HistoricMomentType.CITY_FOUNDED
        assert moment.player_id == 0

    def test_record_increases_era_score(self, manager: EraManager) -> None:
        manager.record_moment(0, HistoricMomentType.CITY_FOUNDED, turn=10)
        progress = manager._get_progress(0)
        assert progress.era_score > 0

    def test_first_in_world_bonus(self, manager: EraManager) -> None:
        cfg = DEFAULT_MOMENT_CONFIGS[HistoricMomentType.DISTRICT_BUILT]
        moment = manager.record_moment(0, HistoricMomentType.DISTRICT_BUILT, turn=5)
        assert moment is not None
        assert moment.is_first is True
        expected = cfg.base_score + cfg.first_bonus
        assert moment.score == expected

    def test_not_first_no_bonus(self, manager: EraManager) -> None:
        cfg = DEFAULT_MOMENT_CONFIGS[HistoricMomentType.DISTRICT_BUILT]
        manager.record_moment(0, HistoricMomentType.DISTRICT_BUILT, turn=5)
        moment2 = manager.record_moment(1, HistoricMomentType.DISTRICT_BUILT, turn=6)
        assert moment2 is not None
        assert moment2.is_first is False
        assert moment2.score == cfg.base_score

    def test_once_per_game_blocks_second_player(self, manager: EraManager) -> None:
        manager.record_moment(0, HistoricMomentType.CIRCUMNAVIGATION, turn=50)
        # Different player cannot get the same once-per-game moment
        moment = manager.record_moment(1, HistoricMomentType.CIRCUMNAVIGATION, turn=55)
        assert moment is None

    def test_once_per_era_blocks_repeat(self, manager: EraManager) -> None:
        # Use a custom once_per_era moment config
        custom_cfg = HistoricMomentConfig(
            moment_type=HistoricMomentType.FIRST_CONTACT,
            base_score=1,
            once_per_era=True,
        )
        custom_moment_configs = {HistoricMomentType.FIRST_CONTACT: custom_cfg}
        mgr = EraManager(moment_configs=custom_moment_configs)
        mgr.record_moment(0, HistoricMomentType.FIRST_CONTACT, turn=1)
        result = mgr.record_moment(0, HistoricMomentType.FIRST_CONTACT, turn=2)
        assert result is None

    def test_unknown_moment_type_returns_none(self, manager: EraManager) -> None:
        # Moment type not in configs → returns None
        # Note: empty dict is falsy, so we must provide a non-empty dict that
        # simply lacks the requested moment type.
        custom_cfg = HistoricMomentConfig(
            moment_type=HistoricMomentType.FIRST_CONTACT,
            base_score=1,
        )
        custom_mgr = EraManager(moment_configs={HistoricMomentType.FIRST_CONTACT: custom_cfg})
        result = custom_mgr.record_moment(0, HistoricMomentType.WONDER_BUILT, turn=1)
        assert result is None

    def test_era_scaling_increases_score(self, manager: EraManager) -> None:
        # Record in Classical (era value > 1) to get scaling
        progress = manager._get_progress(0)
        progress.current_era = Era.CLASSICAL
        moment = manager.record_moment(0, HistoricMomentType.TECH_RESEARCHED, turn=30)
        assert moment is not None
        base = DEFAULT_MOMENT_CONFIGS[HistoricMomentType.TECH_RESEARCHED].base_score
        # first_bonus adds to score; then era scaling multiplies
        assert moment.score >= base

    def test_details_stored_in_moment(self, manager: EraManager) -> None:
        moment = manager.record_moment(
            0, HistoricMomentType.WONDER_BUILT, turn=25, details="Stonehenge"
        )
        assert moment is not None
        assert moment.details == "Stonehenge"


# ---------------------------------------------------------------------------
# EraManager – get_era_threshold
# ---------------------------------------------------------------------------


class TestGetEraThreshold:
    def test_ancient_era_threshold(self, manager: EraManager) -> None:
        thresholds = manager.get_era_threshold(0)
        # Ancient era: base=12, scaling=1.5^0=1.0 → scaled=12
        # dark = 12*0.5=6, golden = 12*1.0=12
        assert thresholds["dark_threshold"] == 6
        assert thresholds["golden_threshold"] == 12

    def test_threshold_scales_with_era(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        progress.current_era = Era.CLASSICAL  # era value=2
        thresholds_classical = manager.get_era_threshold(0)
        # Classical: 12 * 1.5^1 = 18
        assert thresholds_classical["golden_threshold"] == 18

    def test_current_score_in_thresholds(self, manager: EraManager) -> None:
        manager.record_moment(0, HistoricMomentType.CITY_FOUNDED, turn=1)
        thresholds = manager.get_era_threshold(0)
        assert thresholds["current_score"] > 0


# ---------------------------------------------------------------------------
# EraManager – calculate_next_age
# ---------------------------------------------------------------------------


class TestCalculateNextAge:
    def test_normal_age_mid_score(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        manager.get_era_threshold(0)
        # Score between dark (6) and golden (12)
        progress.era_score = 8
        age = manager.calculate_next_age(0)
        assert age == AgeType.NORMAL

    def test_dark_age_low_score(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        progress.era_score = 2  # below dark threshold of 6
        age = manager.calculate_next_age(0)
        assert age == AgeType.DARK

    def test_golden_age_high_score(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        progress.era_score = 15  # above golden threshold of 12
        age = manager.calculate_next_age(0)
        assert age == AgeType.GOLDEN

    def test_heroic_age_after_dark(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        progress.era_score = 15
        progress.was_dark_age = True
        age = manager.calculate_next_age(0)
        assert age == AgeType.HEROIC


# ---------------------------------------------------------------------------
# EraManager – advance_era
# ---------------------------------------------------------------------------


class TestAdvanceEra:
    def test_advance_era_moves_to_next(self, manager: EraManager) -> None:
        result = manager.advance_era(0)
        assert result["old_era"] == "ANCIENT"
        assert result["new_era"] == "CLASSICAL"

    def test_advance_era_resets_score(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        progress.era_score = 20
        manager.advance_era(0)
        assert manager._get_progress(0).era_score == 0

    def test_advance_era_records_was_dark_age(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        # Set current age to DARK so that was_dark_age gets recorded on advance
        progress.age_type = AgeType.DARK
        manager.advance_era(0)
        assert manager._get_progress(0).was_dark_age is True

    def test_advance_era_result_keys(self, manager: EraManager) -> None:
        result = manager.advance_era(0)
        for key in ("old_era", "new_era", "old_age", "new_age", "final_score", "is_heroic"):
            assert key in result

    def test_advance_era_at_future_stays_there(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        progress.current_era = Era.FUTURE
        result = manager.advance_era(0)
        assert result["new_era"] == "FUTURE"

    def test_advance_era_clears_moments(self, manager: EraManager) -> None:
        manager.record_moment(0, HistoricMomentType.CITY_FOUNDED, turn=1)
        manager.advance_era(0)
        assert manager._get_progress(0).moments == []

    def test_advance_era_is_heroic_flag(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        progress.era_score = 20
        progress.was_dark_age = True
        result = manager.advance_era(0)
        assert result["is_heroic"] is True


# ---------------------------------------------------------------------------
# EraManager – select_dedication
# ---------------------------------------------------------------------------


class TestSelectDedication:
    def _set_golden(self, manager: EraManager, player_id: int = 0) -> None:
        progress = manager._get_progress(player_id)
        progress.age_type = AgeType.GOLDEN

    def _set_heroic(self, manager: EraManager, player_id: int = 0) -> None:
        progress = manager._get_progress(player_id)
        progress.age_type = AgeType.HEROIC

    def test_select_dedication_in_golden_age(self, manager: EraManager) -> None:
        self._set_golden(manager)
        result = manager.select_dedication(0, DedicationType.FREE_INQUIRY)
        assert result is True

    def test_select_dedication_exceeds_golden_limit(self, manager: EraManager) -> None:
        self._set_golden(manager)
        manager.select_dedication(0, DedicationType.FREE_INQUIRY)
        # max_dedications_golden=1, cannot add more
        result = manager.select_dedication(0, DedicationType.MONUMENTALITY)
        assert result is False

    def test_select_dedication_heroic_allows_more(self, manager: EraManager) -> None:
        self._set_heroic(manager)
        manager.select_dedication(0, DedicationType.FREE_INQUIRY)
        manager.select_dedication(0, DedicationType.MONUMENTALITY)
        result = manager.select_dedication(0, DedicationType.PEN_BRUSH_VOICE)
        assert result is True

    def test_select_duplicate_dedication_fails(self, manager: EraManager) -> None:
        self._set_golden(manager)
        manager.select_dedication(0, DedicationType.FREE_INQUIRY)
        # Already selected
        manager._get_progress(0).age_type = AgeType.HEROIC  # expand capacity
        result = manager.select_dedication(0, DedicationType.FREE_INQUIRY)
        assert result is False

    def test_select_in_normal_age_fails(self, manager: EraManager) -> None:
        # NORMAL age → max_dedications=0
        result = manager.select_dedication(0, DedicationType.FREE_INQUIRY)
        assert result is False

    def test_select_unknown_dedication_fails(self, manager: EraManager) -> None:
        self._set_golden(manager)
        result = manager.select_dedication(0, DedicationType.HICS_SUNT_DRACONES)
        assert result is False  # not in DEFAULT_DEDICATIONS


# ---------------------------------------------------------------------------
# EraManager – get_dedication_effects
# ---------------------------------------------------------------------------


class TestGetDedicationEffects:
    def test_no_dedications_empty(self, manager: EraManager) -> None:
        effects = manager.get_dedication_effects(0)
        assert effects == {}

    def test_normal_effects_in_normal_age(self, manager: EraManager) -> None:
        # Force a dedication without going through select_dedication restrictions
        progress = manager._get_progress(0)
        progress.age_type = AgeType.NORMAL
        progress.dedications.append(DedicationType.FREE_INQUIRY)
        effects = manager.get_dedication_effects(0)
        assert "commercial_harbor_science" in effects

    def test_golden_effects_in_golden_age(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        progress.age_type = AgeType.GOLDEN
        progress.dedications.append(DedicationType.FREE_INQUIRY)
        effects = manager.get_dedication_effects(0)
        assert "campus_science_bonus" in effects

    def test_golden_effects_in_heroic_age(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        progress.age_type = AgeType.HEROIC
        progress.dedications.append(DedicationType.FREE_INQUIRY)
        effects = manager.get_dedication_effects(0)
        assert "campus_science_bonus" in effects

    def test_multiple_dedications_stack(self, manager: EraManager) -> None:
        progress = manager._get_progress(0)
        progress.age_type = AgeType.NORMAL
        progress.dedications.extend([DedicationType.FREE_INQUIRY, DedicationType.REFORM_THE_COINAGE])
        effects = manager.get_dedication_effects(0)
        assert "commercial_harbor_science" in effects
        assert "city_gold_bonus" in effects


# ---------------------------------------------------------------------------
# EraManager – get_age_effects
# ---------------------------------------------------------------------------


class TestGetAgeEffects:
    def test_normal_age_no_effects(self, manager: EraManager) -> None:
        effects = manager.get_age_effects(0)
        assert effects == {}

    def test_dark_age_loyalty_penalty(self, manager: EraManager) -> None:
        manager._get_progress(0).age_type = AgeType.DARK
        effects = manager.get_age_effects(0)
        assert "loyalty_penalty" in effects
        assert effects["loyalty_penalty"] == pytest.approx(3.0)
        assert "dark_age_policies" in effects

    def test_golden_age_loyalty_bonus(self, manager: EraManager) -> None:
        manager._get_progress(0).age_type = AgeType.GOLDEN
        effects = manager.get_age_effects(0)
        assert "loyalty_bonus" in effects
        assert effects["loyalty_bonus"] == pytest.approx(3.0)

    def test_heroic_age_has_extra_bonus(self, manager: EraManager) -> None:
        manager._get_progress(0).age_type = AgeType.HEROIC
        effects = manager.get_age_effects(0)
        assert "heroic_bonus" in effects
        assert "loyalty_bonus" in effects


# ---------------------------------------------------------------------------
# EraManager – process_turn
# ---------------------------------------------------------------------------


class TestProcessTurn:
    def test_game_turn_increments(self, manager: EraManager) -> None:
        manager.process_turn()
        assert manager._game_turn == 1

    def test_era_turns_increments(self, manager: EraManager) -> None:
        manager._get_progress(0)  # create entry
        manager.process_turn()
        assert manager._get_progress(0).era_turns == 1

    def test_era_turns_multiple_players(self, manager: EraManager) -> None:
        manager._get_progress(0)
        manager._get_progress(1)
        manager.process_turn()
        manager.process_turn()
        assert manager._get_progress(0).era_turns == 2
        assert manager._get_progress(1).era_turns == 2


# ---------------------------------------------------------------------------
# EraManager – get_player_era_info
# ---------------------------------------------------------------------------


class TestGetPlayerEraInfo:
    def test_era_info_fields(self, manager: EraManager) -> None:
        info = manager.get_player_era_info(0)
        for key in (
            "era",
            "era_number",
            "age_type",
            "era_score",
            "dark_threshold",
            "golden_threshold",
            "dedications",
            "moments_count",
            "era_turns",
            "was_dark_age",
        ):
            assert key in info

    def test_era_info_defaults(self, manager: EraManager) -> None:
        info = manager.get_player_era_info(0)
        assert info["era"] == "ANCIENT"
        assert info["age_type"] == "NORMAL"
        assert info["was_dark_age"] is False

    def test_era_info_reflects_changes(self, manager: EraManager) -> None:
        manager.record_moment(0, HistoricMomentType.WONDER_BUILT, turn=10)
        info = manager.get_player_era_info(0)
        assert info["era_score"] > 0
        assert info["moments_count"] == 1


# ---------------------------------------------------------------------------
# EraManager – encode_for_state
# ---------------------------------------------------------------------------


class TestEncodeForState:
    def test_encode_returns_dict(self, manager: EraManager) -> None:
        enc = manager.encode_for_state(0)
        assert isinstance(enc, dict)

    def test_encode_fields_present(self, manager: EraManager) -> None:
        enc = manager.encode_for_state(0)
        for key in (
            "era_number",
            "age_type",
            "is_dark_age",
            "is_golden_age",
            "is_heroic_age",
            "era_score",
            "score_toward_golden",
            "score_toward_dark",
            "dedications_count",
            "moments_this_era",
        ):
            assert key in enc

    def test_encode_ancient_era_number(self, manager: EraManager) -> None:
        enc = manager.encode_for_state(0)
        assert enc["era_number"] == Era.ANCIENT.value

    def test_encode_dark_age_flag(self, manager: EraManager) -> None:
        manager._get_progress(0).age_type = AgeType.DARK
        enc = manager.encode_for_state(0)
        assert enc["is_dark_age"] == pytest.approx(1.0)
        assert enc["is_golden_age"] == pytest.approx(0.0)

    def test_encode_golden_age_flag(self, manager: EraManager) -> None:
        manager._get_progress(0).age_type = AgeType.GOLDEN
        enc = manager.encode_for_state(0)
        assert enc["is_golden_age"] == pytest.approx(1.0)
        assert enc["is_dark_age"] == pytest.approx(0.0)

    def test_encode_score_toward_golden_clamped(self, manager: EraManager) -> None:
        manager._get_progress(0).era_score = 1000
        enc = manager.encode_for_state(0)
        assert enc["score_toward_golden"] <= 1.0


# ---------------------------------------------------------------------------
# EraManager – reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_progress(self, manager: EraManager) -> None:
        manager.record_moment(0, HistoricMomentType.CITY_FOUNDED, turn=1)
        manager.reset()
        assert manager._player_progress == {}

    def test_reset_clears_global_firsts(self, manager: EraManager) -> None:
        manager.record_moment(0, HistoricMomentType.CIRCUMNAVIGATION, turn=50)
        manager.reset()
        assert manager._global_firsts == {}

    def test_reset_resets_turn(self, manager: EraManager) -> None:
        manager.process_turn()
        manager.process_turn()
        manager.reset()
        assert manager._game_turn == 0

    def test_can_trigger_once_per_game_after_reset(self, manager: EraManager) -> None:
        manager.record_moment(0, HistoricMomentType.CIRCUMNAVIGATION, turn=50)
        manager.reset()
        moment = manager.record_moment(0, HistoricMomentType.CIRCUMNAVIGATION, turn=1)
        assert moment is not None
