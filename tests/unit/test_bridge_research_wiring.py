"""GM-B3: research wiring, protocol, state converter and action converter.

Covers ``LuaCommand.item_id`` as a string, ``StateConverter`` (encoder slots 70+/180+)
and ``ActionConverter`` (research bits in the mask, legality, the game type name in the
command). The Lua-harness tests are in ``test_bridge_research_wiring_e2e.py`` and the review
delta in ``test_bridge_research_wiring_review.py``. ``alpha_zero.bridge.tech_registry`` is
imported inside fixtures and tests, so while it does not exist only the tests that need it fail.
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from alpha_zero.bridge.action_converter import ActionConverter, InvalidActionError
from alpha_zero.bridge.config import GameConstants
from alpha_zero.bridge.protocol import (
    ActionType,
    LuaCommand,
    ProtocolValidationError,
)
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.core.civ6_actions import ActionCategory
from alpha_zero.rules import parse_ruleset
from tests.fixtures.research_wiring import (
    CIVIC_RANGE,
    CIVIC_TYPES,
    END_ONLY,
    GAME_STATE,
    RULESET,
    TECH_IDS,
    TECH_RANGE,
    action_converter,
    big_registry,
    civic_action,
    encode,
    flat,
    payload,
    research_bits,
    tech_action,
)


@pytest.fixture
def tr() -> ModuleType:
    return importlib.import_module("alpha_zero.bridge.tech_registry")


@pytest.fixture
def reg(tr: ModuleType) -> Any:
    techs = "\n".join(f"  - id: {i}\n    era: 1\n    cost: 10" for i in TECH_IDS)
    return tr.from_ruleset(parse_ruleset(RULESET.format(techs=techs)), CIVIC_TYPES)


# ---------------------------------------------------------------------------
# protocol: LuaCommand.item_id may be a string
# ---------------------------------------------------------------------------
class TestLuaCommandItemId:
    def test_string_item_id_serialises(self) -> None:
        cmd = LuaCommand(action_type=ActionType.RESEARCH_TECH, item_id="TECH_WRITING")
        assert cmd.to_dict()["item_id"] == "TECH_WRITING"

    def test_string_item_id_parses(self) -> None:
        cmd = LuaCommand.from_dict({"action_type": "research_tech", "item_id": "TECH_WRITING"})
        assert cmd.item_id == "TECH_WRITING"
        assert cmd.action_type == ActionType.RESEARCH_TECH

    def test_string_item_id_round_trips(self) -> None:
        cmd = LuaCommand(action_type=ActionType.RESEARCH_CIVIC, item_id="CIVIC_FOREIGN_TRADE")
        assert LuaCommand.from_dict(cmd.to_dict()) == cmd

    def test_int_item_id_stays_an_int(self) -> None:
        cmd = LuaCommand.from_dict({"action_type": "research_tech", "item_id": 5})
        assert cmd.item_id == 5
        assert type(cmd.item_id) is int
        assert LuaCommand(action_type=ActionType.RESEARCH_TECH, item_id=5).to_dict()["item_id"] == 5

    def test_numeric_string_is_converted_to_int_as_before(self) -> None:
        # Pinned from today's behaviour (_validate_int): "42" becomes 42.
        cmd = LuaCommand.from_dict({"action_type": "research_tech", "item_id": "42"})
        assert cmd.item_id == 42
        assert type(cmd.item_id) is int

    def test_no_item_id_is_still_none(self) -> None:
        cmd = LuaCommand.from_dict({"action_type": "end_turn"})
        assert cmd.item_id is None
        assert "item_id" not in cmd.to_dict()

    def test_unparseable_non_string_item_id_is_still_refused(self) -> None:
        with pytest.raises(ProtocolValidationError):
            LuaCommand.from_dict({"action_type": "research_tech", "item_id": [1]})


# ---------------------------------------------------------------------------
# StateConverter
# ---------------------------------------------------------------------------
class TestStateConverterWithRegistry:
    def test_attribute_is_the_given_registry(self, reg: Any) -> None:
        assert StateConverter(research_registry=reg).research_registry is reg

    def test_technologies_and_civics_become_slots_in_researched_order(self, reg: Any) -> None:
        conv = StateConverter(research_registry=reg)
        out = conv.convert(
            payload(
                techs=("TECH_MINING", "TECH_POTTERY"),
                civics=("CIVIC_CODE_OF_LAWS", "CIVIC_FOREIGN_TRADE"),
            )
        )
        assert out["technologies"] == [3, 2]
        assert out["civics"] == [1, 0]
        assert all(type(s) is int for s in out["technologies"] + out["civics"])

    def test_encoder_vector_has_exactly_the_slot_positions_set(self, reg: Any) -> None:
        conv = StateConverter(research_registry=reg)
        state = payload(techs=("TECH_MINING", "TECH_POTTERY"), civics=("CIVIC_CODE_OF_LAWS",))
        vector = encode(conv, state)
        assert vector[70 + 3] == 1.0
        assert vector[70 + 2] == 1.0
        assert vector[180 + 1] == 1.0
        assert float(vector[70:170].sum()) == 2.0
        assert float(vector[180:280].sum()) == 1.0

    def test_only_the_research_positions_differ_from_the_no_registry_vector(self, reg: Any) -> None:
        state = payload(techs=("TECH_MINING", "TECH_POTTERY"), civics=("CIVIC_CODE_OF_LAWS",))
        with_reg = encode(StateConverter(research_registry=reg), state)
        without = encode(StateConverter(), state)
        assert float(without[70:170].sum()) == 0.0, "control is vacuous"
        changed = {int(i) for i in np.nonzero(with_reg != without)[0]}
        assert changed == {73, 72, 181}

    def test_unmapped_techs_and_civics_are_left_out_and_counted(self, reg: Any) -> None:
        conv = StateConverter(research_registry=reg)
        out = conv.convert(
            payload(
                techs=("TECH_MINING", "TECH_NOPE", "TECH_NADA"),
                civics=("CIVIC_NOPE", "CIVIC_FOREIGN_TRADE"),
            )
        )
        assert out["technologies"] == [3]
        assert out["civics"] == [0]
        stats = conv.research_stats
        assert stats["unmapped_techs"] == 2
        assert stats["unmapped_civics"] == 1
        assert stats["slots_out_of_range"] == 0
        assert stats["conversions_without_registry"] == 0

    def test_unmapped_count_is_one_per_occurrence_per_conversion(self, reg: Any) -> None:
        conv = StateConverter(research_registry=reg)
        conv.convert(payload(techs=("TECH_NOPE",), turn=3))
        conv.convert(payload(techs=("TECH_NOPE",), turn=4))
        assert conv.research_stats["unmapped_techs"] == 2

    def test_slots_of_100_or_more_are_left_out_and_counted(self, tr: ModuleType) -> None:
        conv = StateConverter(research_registry=big_registry(tr, 101, 101))
        out = conv.convert(
            payload(techs=("TECH_T99", "TECH_T100"), civics=("CIVIC_C99", "CIVIC_C100"))
        )
        assert out["technologies"] == [99]
        assert out["civics"] == [99]
        stats = conv.research_stats
        assert stats["slots_out_of_range"] == 2
        assert stats["unmapped_techs"] == 0
        assert stats["unmapped_civics"] == 0

    def test_slot_99_reaches_the_encoder(self, tr: ModuleType) -> None:
        conv = StateConverter(research_registry=big_registry(tr, 101, 101))
        vector = encode(conv, payload(techs=("TECH_T99",), civics=("CIVIC_C99",)))
        assert vector[70 + 99] == 1.0
        assert vector[180 + 99] == 1.0

    def test_science_progress_still_counts_researched_techs(self, reg: Any) -> None:
        total = GameConstants.ESTIMATED_TOTAL_TECHS
        state = payload(techs=("TECH_MINING", "TECH_NOPE"))
        with_reg = StateConverter(research_registry=reg).convert(state)
        without = StateConverter().convert(state)
        assert with_reg["victory_progress"]["science"] == pytest.approx(2 / total)
        assert without["victory_progress"]["science"] == pytest.approx(2 / total)

    def test_no_warning_is_logged_with_a_registry(
        self, reg: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        conv = StateConverter(research_registry=reg)
        with caplog.at_level(logging.WARNING):
            conv.convert(payload(techs=("TECH_MINING",)))
        assert [r for r in caplog.records if "CIV6AZ_BRIDGE_RULESET" in r.getMessage()] == []
        assert conv.research_stats["conversions_without_registry"] == 0


class TestStateConverterWithoutRegistry:
    def test_names_stay_names_and_the_encoder_slots_stay_zero(self) -> None:
        conv = StateConverter()
        assert conv.research_registry is None
        state = payload(techs=("TECH_POTTERY",), civics=("CIVIC_CODE_OF_LAWS",))
        out = conv.convert(state)
        assert out["technologies"] == ["pottery"]
        assert out["civics"] == ["code_of_laws"]
        vector = encode(conv, state)
        assert float(vector[70:170].sum()) == 0.0
        assert float(vector[180:280].sum()) == 0.0

    def test_counts_each_conversion_that_has_research(self) -> None:
        conv = StateConverter()
        conv.convert(payload(techs=("TECH_POTTERY",)))
        assert conv.research_stats["conversions_without_registry"] == 1
        conv.convert(payload(civics=("CIVIC_CODE_OF_LAWS",)))
        assert conv.research_stats["conversions_without_registry"] == 2
        conv.convert(payload(techs=("TECH_POTTERY",), civics=("CIVIC_CODE_OF_LAWS",)))
        assert conv.research_stats["conversions_without_registry"] == 3

    def test_a_conversion_with_nothing_researched_is_not_counted(self) -> None:
        conv = StateConverter()
        conv.convert(payload())
        assert conv.research_stats["conversions_without_registry"] == 0
        conv.convert(payload(techs=("TECH_POTTERY",)))
        assert conv.research_stats["conversions_without_registry"] == 1

    def test_one_warning_per_converter_naming_the_variable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def warnings() -> list[str]:
            return [
                r.getMessage()
                for r in caplog.records
                if r.levelno == logging.WARNING and "CIV6AZ_BRIDGE_RULESET" in r.getMessage()
            ]

        conv = StateConverter()
        with caplog.at_level(logging.WARNING):
            conv.convert(payload())  # nothing researched: not the first such conversion
            assert warnings() == []
            conv.convert(payload(techs=("TECH_POTTERY",)))
            assert len(warnings()) == 1
            conv.convert(payload(techs=("TECH_POTTERY",)))
            conv.convert(payload(civics=("CIVIC_CODE_OF_LAWS",)))
            assert len(warnings()) == 1
            other = StateConverter()
            other.convert(payload(techs=("TECH_POTTERY",)))
            assert len(warnings()) == 2

    def test_science_progress_is_unchanged(self) -> None:
        out = StateConverter().convert(payload(techs=("TECH_POTTERY", "TECH_MINING")))
        assert out["victory_progress"]["science"] == pytest.approx(
            2 / GameConstants.ESTIMATED_TOTAL_TECHS
        )


class TestStateConverterStatsAndEnv:
    def test_stats_has_exactly_the_four_keys_at_zero(self) -> None:
        stats = StateConverter().research_stats
        assert stats == {
            "conversions_without_registry": 0,
            "unmapped_techs": 0,
            "unmapped_civics": 0,
            "slots_out_of_range": 0,
        }

    def test_stats_is_a_fresh_dict_each_time(self) -> None:
        conv = StateConverter()
        first = conv.research_stats
        first["unmapped_techs"] = 99
        first["extra"] = 1
        assert conv.research_stats["unmapped_techs"] == 0
        assert "extra" not in conv.research_stats
        assert conv.research_stats is not conv.research_stats

    def test_environment_selects_the_registry(
        self, tr: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(tr.RULESET_ENV, "science-v1-draft")
        conv = StateConverter()
        assert conv.research_registry == tr.load_registry("science-v1-draft")
        out = conv.convert(payload(techs=("TECH_MINING", "TECH_POTTERY")))
        # science-v1-draft: agriculture 0, mining 1, ...; it has no pottery.
        assert out["technologies"] == [1]
        assert conv.research_stats["unmapped_techs"] == 1

    def test_bad_environment_raises_at_construction(
        self, tr: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(tr.RULESET_ENV, "no-such-ruleset")
        with pytest.raises(tr.ResearchRegistryError):
            StateConverter()

    def test_an_explicit_registry_does_not_consult_the_environment(
        self, tr: ModuleType, reg: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(tr.RULESET_ENV, "no-such-ruleset")
        assert StateConverter(research_registry=reg).research_registry is reg


# ---------------------------------------------------------------------------
# ActionConverter: the mask
# ---------------------------------------------------------------------------


class TestMaskWithRegistry:
    def test_attribute_is_the_given_registry(self, reg: Any) -> None:
        assert action_converter(reg).research_registry is reg

    def test_names_set_their_slots_in_the_right_range(self, reg: Any) -> None:
        conv = action_converter(reg)
        legal = [
            *END_ONLY,
            tech_action(["TECH_WRITING", "TECH_MINING"], [2, 1]),
            civic_action(["CIVIC_FOREIGN_TRADE", "CIVIC_CRAFTSMANSHIP"], [2, 1]),
        ]
        mask = conv.generate_action_mask(legal, GAME_STATE)
        assert research_bits(conv, mask, TECH_RANGE) == {0, 3}
        assert research_bits(conv, mask, CIVIC_RANGE) == {0, 3}

    def test_tech_and_civic_do_not_leak_into_each_other(self, reg: Any) -> None:
        conv = action_converter(reg)
        only_tech = conv.generate_action_mask(
            [*END_ONLY, tech_action(["TECH_POTTERY"], [0])], GAME_STATE
        )
        assert research_bits(conv, only_tech, TECH_RANGE) == {2}
        assert research_bits(conv, only_tech, CIVIC_RANGE) == set()
        only_civic = conv.generate_action_mask(
            [*END_ONLY, civic_action(["CIVIC_CODE_OF_LAWS"], [0])], GAME_STATE
        )
        assert research_bits(conv, only_civic, CIVIC_RANGE) == {1}
        assert research_bits(conv, only_civic, TECH_RANGE) == set()

    def test_only_research_bits_are_added(self, reg: Any) -> None:
        conv = action_converter(reg)
        base = conv.generate_action_mask(END_ONLY, GAME_STATE)
        full = conv.generate_action_mask(
            [*END_ONLY, tech_action(["TECH_WRITING"], [2]), civic_action(["CIVIC_EARLY_EMPIRE"])],
            GAME_STATE,
        )
        tech_off = conv.action_space.category_offsets[TECH_RANGE]
        civic_off = conv.action_space.category_offsets[CIVIC_RANGE]
        added = {int(i) for i in np.nonzero(full != base)[0]}
        assert added == {tech_off + 0, civic_off + 2}
        end_idx = conv.action_space.category_offsets[ActionCategory.END_TURN]
        assert full[end_idx] == 1.0

    def test_unmapped_names_are_counted_and_set_nothing(self, reg: Any) -> None:
        conv = action_converter(reg)
        mask = conv.generate_action_mask(
            [*END_ONLY, tech_action(["TECH_WRITING", "TECH_NOPE"], [2, 9])], GAME_STATE
        )
        assert research_bits(conv, mask, TECH_RANGE) == {0}
        assert conv.research_stats == {"unmapped_research_items": 1, "research_out_of_range": 0}

    @pytest.mark.parametrize("names", [None, []])
    def test_missing_or_empty_item_names_set_nothing_and_count_each_item(
        self, reg: Any, names: list[str] | None
    ) -> None:
        conv = action_converter(reg)
        mask = conv.generate_action_mask(
            [*END_ONLY, tech_action(names, [1, 2, 3]), civic_action(names, [0])], GAME_STATE
        )
        assert research_bits(conv, mask, TECH_RANGE) == set()
        assert research_bits(conv, mask, CIVIC_RANGE) == set()
        assert conv.research_stats["unmapped_research_items"] == 4
        assert conv.research_stats["research_out_of_range"] == 0

    def test_tech_slot_at_or_above_num_techs_is_out_of_range(self, tr: ModuleType) -> None:
        conv = action_converter(big_registry(tr, 80, 60))
        num_techs = conv.action_space.category_sizes[TECH_RANGE]
        assert num_techs == 73
        mask = conv.generate_action_mask(
            [*END_ONLY, tech_action(["TECH_T72", "TECH_T73", "TECH_T75"], [0, 1, 2])], GAME_STATE
        )
        assert research_bits(conv, mask, TECH_RANGE) == {72}
        assert conv.research_stats == {"unmapped_research_items": 0, "research_out_of_range": 2}

    def test_civic_slot_at_or_above_num_civics_is_out_of_range(self, tr: ModuleType) -> None:
        conv = action_converter(big_registry(tr, 80, 60))
        num_civics = conv.action_space.category_sizes[CIVIC_RANGE]
        assert num_civics == 56
        mask = conv.generate_action_mask(
            [*END_ONLY, civic_action(["CIVIC_C55", "CIVIC_C56", "CIVIC_C57"], [0, 1, 2])],
            GAME_STATE,
        )
        assert research_bits(conv, mask, CIVIC_RANGE) == {55}
        assert conv.research_stats == {"unmapped_research_items": 0, "research_out_of_range": 2}

    def test_no_research_action_means_no_research_bit(self, reg: Any) -> None:
        conv = action_converter(reg)
        mask = conv.generate_action_mask(END_ONLY, GAME_STATE)
        assert research_bits(conv, mask, TECH_RANGE) == set()
        assert research_bits(conv, mask, CIVIC_RANGE) == set()
        assert conv.research_stats == {"unmapped_research_items": 0, "research_out_of_range": 0}

    def test_accepts_legal_action_objects(self, reg: Any) -> None:
        from alpha_zero.bridge.protocol import LegalAction

        conv = action_converter(reg)
        legal = [
            LegalAction(action_type="end_turn"),
            LegalAction(action_type="research_tech", items=[2], item_names=["TECH_WRITING"]),
        ]
        mask = conv.generate_action_mask(legal, GAME_STATE)
        assert research_bits(conv, mask, TECH_RANGE) == {0}


class TestMaskWithoutRegistry:
    def test_row_indexes_are_the_selectors(self) -> None:
        conv = action_converter()
        assert conv.research_registry is None
        mask = conv.generate_action_mask(
            [*END_ONLY, tech_action(["TECH_WRITING"], [2, 5]), civic_action(["CIVIC_X"], [1])],
            GAME_STATE,
        )
        assert research_bits(conv, mask, TECH_RANGE) == {2, 5}
        assert research_bits(conv, mask, CIVIC_RANGE) == {1}
        assert conv.research_stats["research_out_of_range"] == 0

    def test_items_outside_the_range_are_counted(self) -> None:
        conv = action_converter()
        mask = conv.generate_action_mask(
            [*END_ONLY, tech_action(None, [0, 72, 73, 100, -1])], GAME_STATE
        )
        assert research_bits(conv, mask, TECH_RANGE) == {0, 72}
        assert conv.research_stats == {"unmapped_research_items": 0, "research_out_of_range": 3}

    def test_civic_range_uses_num_civics(self) -> None:
        conv = action_converter()
        mask = conv.generate_action_mask([*END_ONLY, civic_action(None, [55, 56])], GAME_STATE)
        assert research_bits(conv, mask, CIVIC_RANGE) == {55}
        assert conv.research_stats["research_out_of_range"] == 1

    def test_names_are_ignored_without_a_registry(self) -> None:
        conv = action_converter()
        mask = conv.generate_action_mask(
            [*END_ONLY, tech_action(["TECH_WHATEVER"], [4])], GAME_STATE
        )
        assert research_bits(conv, mask, TECH_RANGE) == {4}
        assert conv.research_stats["unmapped_research_items"] == 0

    def test_no_research_action_means_no_research_bit(self) -> None:
        conv = action_converter()
        mask = conv.generate_action_mask(END_ONLY, GAME_STATE)
        assert research_bits(conv, mask, TECH_RANGE) == set()
        assert research_bits(conv, mask, CIVIC_RANGE) == set()


class TestActionConverterStats:
    def test_exact_keys_at_zero(self) -> None:
        assert action_converter().research_stats == {
            "unmapped_research_items": 0,
            "research_out_of_range": 0,
        }

    def test_fresh_dict_each_time(self) -> None:
        conv = action_converter()
        stats = conv.research_stats
        stats["research_out_of_range"] = 5
        stats["extra"] = 1
        assert conv.research_stats == {"unmapped_research_items": 0, "research_out_of_range": 0}

    def test_environment_selects_the_registry(
        self, tr: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(tr.RULESET_ENV, "science-v1-draft")
        conv = ActionConverter(board_size=8)
        assert conv.research_registry == tr.load_registry("science-v1-draft")
        mask = conv.generate_action_mask(
            [*END_ONLY, tech_action(["TECH_WRITING"], [2])], GAME_STATE
        )
        assert research_bits(conv, mask, TECH_RANGE) == {2}  # writing is slot 2 there

    def test_bad_environment_raises_at_construction(
        self, tr: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(tr.RULESET_ENV, "no-such-ruleset")
        with pytest.raises(tr.ResearchRegistryError):
            ActionConverter(board_size=8)


# ---------------------------------------------------------------------------
# ActionConverter: convert()
# ---------------------------------------------------------------------------


class TestConvertWithRegistry:
    def test_tech_command_carries_the_game_type_name(self, reg: Any) -> None:
        conv = action_converter(reg)
        legal = [*END_ONLY, tech_action(["TECH_WRITING"], [2])]
        cmd = conv.convert(flat(conv, TECH_RANGE, 0), GAME_STATE, legal)
        assert cmd.action_type == ActionType.RESEARCH_TECH
        assert cmd.item_id == "TECH_WRITING"
        assert cmd.to_dict()["item_id"] == "TECH_WRITING"

    def test_civic_command_carries_the_game_type_name(self, reg: Any) -> None:
        conv = action_converter(reg)
        legal = [*END_ONLY, civic_action(["CIVIC_FOREIGN_TRADE"], [2])]
        cmd = conv.convert(flat(conv, CIVIC_RANGE, 0), GAME_STATE, legal)
        assert cmd.action_type == ActionType.RESEARCH_CIVIC
        assert cmd.item_id == "CIVIC_FOREIGN_TRADE"

    def test_a_tech_that_is_not_offered_is_refused(self, reg: Any) -> None:
        conv = action_converter(reg)
        legal = [*END_ONLY, tech_action(["TECH_WRITING"], [2])]
        with pytest.raises(InvalidActionError):
            conv.convert(flat(conv, TECH_RANGE, 3), GAME_STATE, legal)  # mining: not offered

    def test_a_civic_that_is_not_offered_is_refused(self, reg: Any) -> None:
        conv = action_converter(reg)
        legal = [*END_ONLY, civic_action(["CIVIC_FOREIGN_TRADE"], [2])]
        with pytest.raises(InvalidActionError):
            conv.convert(flat(conv, CIVIC_RANGE, 3), GAME_STATE, legal)

    def test_research_is_refused_when_no_research_action_is_legal(self, reg: Any) -> None:
        conv = action_converter(reg)
        with pytest.raises(InvalidActionError):
            conv.convert(flat(conv, TECH_RANGE, 0), GAME_STATE, END_ONLY)
        with pytest.raises(InvalidActionError):
            conv.convert(flat(conv, CIVIC_RANGE, 0), GAME_STATE, END_ONLY)

    def test_a_tech_offered_only_as_a_civic_is_refused(self, reg: Any) -> None:
        conv = action_converter(reg)
        legal = [*END_ONLY, civic_action(["CIVIC_FOREIGN_TRADE"], [2])]
        with pytest.raises(InvalidActionError):
            conv.convert(flat(conv, TECH_RANGE, 0), GAME_STATE, legal)

    def test_a_slot_whose_name_is_missing_from_item_names_is_refused(self, reg: Any) -> None:
        conv = action_converter(reg)
        legal = [*END_ONLY, tech_action(None, [2])]
        with pytest.raises(InvalidActionError):
            conv.convert(flat(conv, TECH_RANGE, 0), GAME_STATE, legal)

    def test_validate_false_skips_legality_but_not_the_registry(self, reg: Any) -> None:
        conv = action_converter(reg)
        cmd = conv.convert(flat(conv, TECH_RANGE, 3), GAME_STATE, END_ONLY, validate=False)
        assert cmd.item_id == "TECH_MINING"
        civic = conv.convert(flat(conv, CIVIC_RANGE, 3), GAME_STATE, END_ONLY, validate=False)
        assert civic.item_id == "CIVIC_CRAFTSMANSHIP"

    def test_validate_false_with_a_selector_that_has_no_tech_is_refused(self, reg: Any) -> None:
        conv = action_converter(reg)
        with pytest.raises(InvalidActionError):
            conv.convert(flat(conv, TECH_RANGE, 10), GAME_STATE, END_ONLY, validate=False)
        with pytest.raises(InvalidActionError):
            conv.convert(flat(conv, CIVIC_RANGE, 10), GAME_STATE, END_ONLY, validate=False)


class TestConvertWithoutRegistry:
    def test_command_item_id_is_the_selector_as_an_int(self) -> None:
        conv = action_converter()
        legal = [*END_ONLY, tech_action(["TECH_WRITING"], [2]), civic_action(["CIVIC_X"], [1])]
        tech = conv.convert(flat(conv, TECH_RANGE, 2), GAME_STATE, legal)
        civic = conv.convert(flat(conv, CIVIC_RANGE, 1), GAME_STATE, legal)
        assert (tech.action_type, tech.item_id) == (ActionType.RESEARCH_TECH, 2)
        assert (civic.action_type, civic.item_id) == (ActionType.RESEARCH_CIVIC, 1)
        assert type(tech.item_id) is int
        assert type(civic.item_id) is int

    def test_a_row_that_is_not_offered_is_refused(self) -> None:
        conv = action_converter()
        legal = [*END_ONLY, tech_action(None, [2])]
        with pytest.raises(InvalidActionError):
            conv.convert(flat(conv, TECH_RANGE, 1), GAME_STATE, legal)

    def test_research_is_refused_when_no_research_action_is_legal(self) -> None:
        conv = action_converter()
        with pytest.raises(InvalidActionError):
            conv.convert(flat(conv, TECH_RANGE, 2), GAME_STATE, END_ONLY)

    def test_validate_false_keeps_the_selector(self) -> None:
        conv = action_converter()
        cmd = conv.convert(flat(conv, TECH_RANGE, 7), GAME_STATE, END_ONLY, validate=False)
        assert cmd.item_id == 7
        assert type(cmd.item_id) is int
