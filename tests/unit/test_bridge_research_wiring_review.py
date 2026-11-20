"""GM-B3: research wiring, review delta (D1-D12).

Opt-out, construction logging, counters, warnings and encoder limits. Shared helpers live in
``tests/fixtures/research_wiring.py``.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from alpha_zero.bridge import state_converter as state_converter_module
from alpha_zero.bridge.protocol import (
    ActionType,
    GameStateData,
    LuaCommand,
    ProtocolValidationError,
)
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.core.civ6_actions import ActionCategory
from alpha_zero.rules import parse_ruleset
from tests.fixtures.research_wiring import (
    ACTION_LOGGER,
    CIVIC_RANGE,
    CIVIC_TYPES,
    END_ONLY,
    FACTORIES,
    GAME_STATE,
    REPO_ROOT,
    RULESET,
    STATE_LOGGER,
    TECH_IDS,
    TECH_RANGE,
    big_registry,
    civic_action,
    flat,
    messages,
    new_actions,
    new_state,
    payload,
    registry_from,
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


@pytest.fixture
def reg_other(tr: ModuleType) -> Any:
    return registry_from(tr, list(reversed(TECH_IDS)))


@pytest.fixture
def reg_no_civics(tr: ModuleType) -> Any:
    return registry_from(tr, TECH_IDS, ())


# ===========================================================================
# GM-B3 review delta (D1-D12)
# ===========================================================================


# ---------------------------------------------------------------------------
# D3: research_registry=False opts out of the environment
# ---------------------------------------------------------------------------
class TestOptOut:
    @pytest.mark.parametrize("factory", FACTORIES, ids=["state", "action"])
    def test_false_ignores_a_bad_environment(
        self, tr: ModuleType, monkeypatch: pytest.MonkeyPatch, factory: Any
    ) -> None:
        monkeypatch.setenv(tr.RULESET_ENV, "no-such-ruleset")
        assert factory(False).research_registry is None
        with pytest.raises(tr.ResearchRegistryError):
            factory(None)

    @pytest.mark.parametrize("factory", FACTORIES, ids=["state", "action"])
    def test_false_ignores_a_good_environment(
        self, tr: ModuleType, monkeypatch: pytest.MonkeyPatch, factory: Any
    ) -> None:
        monkeypatch.setenv(tr.RULESET_ENV, "science-v1-draft")
        assert factory(False).research_registry is None
        assert factory(None).research_registry is not None

    def test_a_state_converter_with_false_runs_without_a_registry(
        self, tr: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(tr.RULESET_ENV, "science-v1-draft")
        conv = new_state(False)
        out = conv.convert(payload(techs=("TECH_MINING",)))
        assert out["technologies"] == ["mining"]
        assert conv.research_stats["conversions_without_registry"] == 1


# ---------------------------------------------------------------------------
# D4: construction logs
# ---------------------------------------------------------------------------
class TestConstructionLogging:
    @pytest.mark.parametrize(
        ("factory", "logger"), [(new_state, STATE_LOGGER), (new_actions, ACTION_LOGGER)]
    )
    def test_one_info_line_with_the_description(
        self, reg: Any, caplog: pytest.LogCaptureFixture, factory: Any, logger: str
    ) -> None:
        with caplog.at_level(logging.INFO, logger=logger):
            factory(reg)
        lines = [m for m in messages(caplog, logger, logging.INFO) if reg.describe() in m]
        assert len(lines) == 1

    @pytest.mark.parametrize("registry", [False, None], ids=["false", "none"])
    @pytest.mark.parametrize(
        ("factory", "logger"), [(new_state, STATE_LOGGER), (new_actions, ACTION_LOGGER)]
    )
    def test_one_info_line_without_a_registry(
        self,
        caplog: pytest.LogCaptureFixture,
        factory: Any,
        logger: str,
        registry: Any,
    ) -> None:
        with caplog.at_level(logging.INFO, logger=logger):
            factory(registry)
        lines = [m for m in messages(caplog, logger, logging.INFO) if "no research registry" in m]
        assert len(lines) == 1

    def test_action_converter_warns_when_the_registry_has_no_civics(
        self, reg_no_civics: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=ACTION_LOGGER):
            new_actions(reg_no_civics)
        warnings = messages(caplog, ACTION_LOGGER, logging.WARNING)
        assert len(warnings) == 1
        assert "CIV6AZ_BRIDGE_GAMEINFO" in warnings[0]

    def test_no_civic_warning_with_civics_or_without_a_registry(
        self, reg: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=ACTION_LOGGER):
            new_actions(reg)
            new_actions(False)
        assert messages(caplog, ACTION_LOGGER, logging.WARNING) == []


# ---------------------------------------------------------------------------
# D5: the two converters must agree on the registry
# ---------------------------------------------------------------------------
class TestConvertersAgree:
    def test_state_carries_the_registry_fingerprint(self, reg: Any) -> None:
        assert new_state(reg).convert(payload())["_research_fingerprint"] == reg.fingerprint

    def test_state_without_a_registry_carries_none(self) -> None:
        out = new_state(False).convert(payload())
        assert "_research_fingerprint" in out
        assert out["_research_fingerprint"] is None

    @staticmethod
    def pick(name: str, reg: Any, other: Any) -> Any:
        return {"reg": reg, "other": other, "none": False}[name]

    @pytest.mark.parametrize(
        ("state_side", "action_side"), [("reg", "none"), ("none", "reg"), ("reg", "other")]
    )
    def test_a_mismatch_is_refused_by_mask_and_convert(
        self, tr: ModuleType, reg: Any, reg_other: Any, state_side: str, action_side: str
    ) -> None:
        assert reg.fingerprint != reg_other.fingerprint
        state = new_state(self.pick(state_side, reg, reg_other)).convert(payload())
        conv = new_actions(self.pick(action_side, reg, reg_other))
        with pytest.raises(tr.ResearchRegistryError):
            conv.generate_action_mask(END_ONLY, state)
        end_turn = conv.action_space.category_offsets[ActionCategory.END_TURN]
        with pytest.raises(tr.ResearchRegistryError):
            conv.convert(end_turn, state, END_ONLY)

    @pytest.mark.parametrize("side", ["reg", "none"])
    def test_the_same_registry_is_fine(self, reg: Any, reg_other: Any, side: str) -> None:
        registry = self.pick(side, reg, reg_other)
        state = new_state(registry).convert(payload())
        conv = new_actions(registry)
        mask = conv.generate_action_mask(END_ONLY, state)
        end_turn = conv.action_space.category_offsets[ActionCategory.END_TURN]
        assert mask[end_turn] == 1.0
        assert conv.convert(end_turn, state, END_ONLY).action_type == ActionType.END_TURN

    def test_a_hand_built_state_without_the_key_is_not_checked(self, reg: Any) -> None:
        assert "_research_fingerprint" not in GAME_STATE
        mask = new_actions(reg).generate_action_mask(END_ONLY, GAME_STATE)
        assert mask.sum() == 1.0
        assert new_actions(False).generate_action_mask(END_ONLY, GAME_STATE).sum() == 1.0

    def test_a_hand_built_key_is_checked(self, tr: ModuleType, reg: Any) -> None:
        with pytest.raises(tr.ResearchRegistryError):
            new_actions(reg).generate_action_mask(
                END_ONLY, {**GAME_STATE, "_research_fingerprint": None}
            )
        with pytest.raises(tr.ResearchRegistryError):
            new_actions(False).generate_action_mask(
                END_ONLY, {**GAME_STATE, "_research_fingerprint": reg.fingerprint}
            )
        ok = new_actions(reg).generate_action_mask(
            END_ONLY, {**GAME_STATE, "_research_fingerprint": reg.fingerprint}
        )
        assert ok.sum() == 1.0


# ---------------------------------------------------------------------------
# D6: count once per state
# ---------------------------------------------------------------------------
LOSSES = {
    "techs": ("TECH_NOPE", "TECH_T100"),
    "civics": ("CIVIC_NOPE", "CIVIC_C100"),
}


class TestCountOnce:
    def test_the_same_state_again_does_not_move_any_counter(self, tr: ModuleType) -> None:
        conv = new_state(big_registry(tr, 101, 101))
        conv.convert(payload(LOSSES["techs"], LOSSES["civics"], turn=3))
        first = conv.research_stats
        assert first == {
            "conversions_without_registry": 0,
            "unmapped_techs": 1,
            "unmapped_civics": 1,
            "slots_out_of_range": 2,
        }
        conv.convert(payload(LOSSES["techs"], LOSSES["civics"], turn=3))
        assert conv.research_stats == first

    def test_another_turn_counts_again(self, tr: ModuleType) -> None:
        conv = new_state(big_registry(tr, 101, 101))
        conv.convert(payload(LOSSES["techs"], LOSSES["civics"], turn=3))
        conv.convert(payload(LOSSES["techs"], LOSSES["civics"], turn=4))
        assert conv.research_stats == {
            "conversions_without_registry": 0,
            "unmapped_techs": 2,
            "unmapped_civics": 2,
            "slots_out_of_range": 4,
        }

    def test_other_research_on_the_same_turn_counts(self, reg: Any) -> None:
        conv = new_state(reg)
        conv.convert(payload(("TECH_NOPE",), turn=3))
        conv.convert(payload(("TECH_NOPE", "TECH_NADA"), turn=3))
        assert conv.research_stats["unmapped_techs"] == 3
        conv.convert(payload(("TECH_NOPE", "TECH_NADA"), ("CIVIC_NOPE",), turn=3))
        assert conv.research_stats["unmapped_techs"] == 5
        assert conv.research_stats["unmapped_civics"] == 1

    def test_without_a_registry_the_same_state_counts_once(self) -> None:
        conv = new_state(False)
        conv.convert(payload(("TECH_POTTERY",), turn=3))
        conv.convert(payload(("TECH_POTTERY",), turn=3))
        assert conv.research_stats["conversions_without_registry"] == 1
        conv.convert(payload(("TECH_POTTERY",), turn=4))
        assert conv.research_stats["conversions_without_registry"] == 2

    @pytest.mark.parametrize("registry_kind", ["reg", "none"])
    def test_convert_then_convert_and_encode_counts_once(
        self, reg: Any, registry_kind: str
    ) -> None:
        conv = new_state(reg if registry_kind == "reg" else False)
        state = GameStateData.from_dict(payload(("TECH_NOPE", "TECH_MINING"), ("CIVIC_NOPE",)))
        conv.convert(state)
        conv.convert_and_encode(state)
        stats = conv.research_stats
        if registry_kind == "reg":
            assert (stats["unmapped_techs"], stats["unmapped_civics"]) == (1, 1)
        else:
            assert stats["conversions_without_registry"] == 1

    def test_the_same_offers_again_do_not_move_the_action_counters(self, tr: ModuleType) -> None:
        conv = new_actions(big_registry(tr, 80, 60))
        legal = [*END_ONLY, tech_action(["TECH_T1", "TECH_NOPE", "TECH_T75"], [1, 2, 3])]
        conv.generate_action_mask(legal, GAME_STATE)
        first = conv.research_stats
        assert first == {"unmapped_research_items": 1, "research_out_of_range": 1}
        conv.generate_action_mask(legal, GAME_STATE)
        assert conv.research_stats == first

    def test_different_offers_count(self, reg: Any) -> None:
        conv = new_actions(reg)
        conv.generate_action_mask(
            [*END_ONLY, tech_action(["TECH_WRITING", "TECH_NOPE"], [2, 9])], GAME_STATE
        )
        assert conv.research_stats["unmapped_research_items"] == 1
        # other names
        conv.generate_action_mask(
            [*END_ONLY, tech_action(["TECH_WRITING", "TECH_NADA"], [2, 9])], GAME_STATE
        )
        assert conv.research_stats["unmapped_research_items"] == 2
        # same names, other items
        conv.generate_action_mask(
            [*END_ONLY, tech_action(["TECH_WRITING", "TECH_NADA"], [2, 8])], GAME_STATE
        )
        assert conv.research_stats["unmapped_research_items"] == 3
        # another action type
        conv.generate_action_mask(
            [*END_ONLY, civic_action(["CIVIC_NOPE"], [8])],
            GAME_STATE,
        )
        assert conv.research_stats["unmapped_research_items"] == 4

    def test_no_registry_offers_are_counted_once_too(self) -> None:
        conv = new_actions(False)
        legal = [*END_ONLY, tech_action(None, [3, 100])]
        conv.generate_action_mask(legal, GAME_STATE)
        conv.generate_action_mask(legal, GAME_STATE)
        assert conv.research_stats == {"unmapped_research_items": 0, "research_out_of_range": 1}


# ---------------------------------------------------------------------------
# D7: one warning on the first lost item
# ---------------------------------------------------------------------------
class TestFirstLostItemWarns:
    def test_state_converter_warns_once_naming_the_item(
        self, reg: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        conv = new_state(reg)
        with caplog.at_level(logging.WARNING, logger=STATE_LOGGER):
            caplog.clear()
            conv.convert(payload(("TECH_MINING",), turn=1))
            assert messages(caplog, STATE_LOGGER, logging.WARNING) == []
            conv.convert(payload(("TECH_MINING", "TECH_NOPE"), turn=2))
            first = messages(caplog, STATE_LOGGER, logging.WARNING)
            assert len(first) == 1
            assert "TECH_NOPE" in first[0]
            conv.convert(payload(("TECH_NADA",), ("CIVIC_NOPE",), turn=3))
            conv.convert(payload(("TECH_NADA",), ("CIVIC_NOPE",), turn=4))
            assert len(messages(caplog, STATE_LOGGER, logging.WARNING)) == 1
        assert conv.research_stats["unmapped_techs"] == 3
        assert conv.research_stats["unmapped_civics"] == 2

    def test_a_lost_civic_is_named(self, reg: Any, caplog: pytest.LogCaptureFixture) -> None:
        conv = new_state(reg)
        with caplog.at_level(logging.WARNING, logger=STATE_LOGGER):
            caplog.clear()
            conv.convert(payload((), ("CIVIC_NOPE",)))
        warnings = messages(caplog, STATE_LOGGER, logging.WARNING)
        assert len(warnings) == 1
        assert "CIVIC_NOPE" in warnings[0]

    def test_an_item_the_encoder_cannot_hold_is_named(
        self, tr: ModuleType, caplog: pytest.LogCaptureFixture
    ) -> None:
        conv = new_state(big_registry(tr, 101, 101))
        with caplog.at_level(logging.WARNING, logger=STATE_LOGGER):
            caplog.clear()
            conv.convert(payload(("TECH_T100",)))
        warnings = messages(caplog, STATE_LOGGER, logging.WARNING)
        assert len(warnings) == 1
        assert "TECH_T100" in warnings[0]

    def test_action_converter_warns_once_naming_the_name(
        self, reg: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        conv = new_actions(reg)
        with caplog.at_level(logging.WARNING, logger=ACTION_LOGGER):
            caplog.clear()
            conv.generate_action_mask([*END_ONLY, tech_action(["TECH_WRITING"], [2])], GAME_STATE)
            assert messages(caplog, ACTION_LOGGER, logging.WARNING) == []
            conv.generate_action_mask(
                [*END_ONLY, tech_action(["TECH_WRITING", "TECH_NOPE"], [2, 9])], GAME_STATE
            )
            first = messages(caplog, ACTION_LOGGER, logging.WARNING)
            assert len(first) == 1
            assert "TECH_NOPE" in first[0]
            conv.generate_action_mask([*END_ONLY, civic_action(["CIVIC_NADA"], [5])], GAME_STATE)
            assert len(messages(caplog, ACTION_LOGGER, logging.WARNING)) == 1
        assert conv.research_stats["unmapped_research_items"] == 2

    def test_action_converter_names_the_item_when_there_is_no_name(
        self, reg: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        conv = new_actions(reg)
        with caplog.at_level(logging.WARNING, logger=ACTION_LOGGER):
            caplog.clear()
            conv.generate_action_mask([*END_ONLY, tech_action(None, [7777])], GAME_STATE)
        warnings = messages(caplog, ACTION_LOGGER, logging.WARNING)
        assert len(warnings) == 1
        assert "7777" in warnings[0]

    def test_action_converter_names_a_slot_beyond_the_action_space(
        self, tr: ModuleType, caplog: pytest.LogCaptureFixture
    ) -> None:
        conv = new_actions(big_registry(tr, 80, 60))
        with caplog.at_level(logging.WARNING, logger=ACTION_LOGGER):
            caplog.clear()
            conv.generate_action_mask([*END_ONLY, tech_action(["TECH_T75"], [1])], GAME_STATE)
        warnings = messages(caplog, ACTION_LOGGER, logging.WARNING)
        assert len(warnings) == 1
        assert "TECH_T75" in warnings[0]


# ---------------------------------------------------------------------------
# D8: no-registry mask, items that are not ints
# ---------------------------------------------------------------------------
class TestNoRegistryNonIntItems:
    def test_a_string_item_is_unmapped_and_an_out_of_range_int_is_out_of_range(self) -> None:
        conv = new_actions(False)
        mask = conv.generate_action_mask(
            [*END_ONLY, tech_action(None, ["TECH_WRITING", 3, 100])], GAME_STATE
        )
        assert research_bits(conv, mask, TECH_RANGE) == {3}
        assert conv.research_stats == {"unmapped_research_items": 1, "research_out_of_range": 1}

    def test_civics_likewise(self) -> None:
        conv = new_actions(False)
        mask = conv.generate_action_mask(
            [*END_ONLY, civic_action(None, ["CIVIC_X", 56, 2])], GAME_STATE
        )
        assert research_bits(conv, mask, CIVIC_RANGE) == {2}
        assert conv.research_stats == {"unmapped_research_items": 1, "research_out_of_range": 1}


# ---------------------------------------------------------------------------
# D9: LuaCommand.from_dict, string item ids only for research and only identifiers
# ---------------------------------------------------------------------------
class TestItemIdStringsOnlyForResearch:
    @pytest.mark.parametrize("action", ["research_tech", "research_civic"])
    @pytest.mark.parametrize("name", ["TECH_WRITING", "_x", "a1", "CIVIC_A_B"])
    def test_identifiers_stay_strings_for_research(self, action: str, name: str) -> None:
        assert LuaCommand.from_dict({"action_type": action, "item_id": name}).item_id == name

    @pytest.mark.parametrize("action", ["research_tech", "research_civic"])
    @pytest.mark.parametrize("bad", ["", "-", "TECH-X", "TECH X", "1ABC", "TECH_X "])
    def test_other_strings_are_refused_for_research(self, action: str, bad: str) -> None:
        with pytest.raises(ProtocolValidationError):
            LuaCommand.from_dict({"action_type": action, "item_id": bad})

    @pytest.mark.parametrize("action", ["research_tech", "research_civic"])
    @pytest.mark.parametrize(("text", "number"), [("12", 12), (" 12", 12), ("0", 0)])
    def test_numeric_strings_become_ints_for_research(
        self, action: str, text: str, number: int
    ) -> None:
        cmd = LuaCommand.from_dict({"action_type": action, "item_id": text})
        assert cmd.item_id == number
        assert type(cmd.item_id) is int

    @pytest.mark.parametrize("action", ["city_produce", "city_buy", "unit_build", "end_turn"])
    def test_an_identifier_is_refused_for_any_other_action(self, action: str) -> None:
        with pytest.raises(ProtocolValidationError):
            LuaCommand.from_dict({"action_type": action, "item_id": "UNIT_WARRIOR"})

    def test_numeric_strings_still_become_ints_elsewhere(self) -> None:
        cmd = LuaCommand.from_dict({"action_type": "city_produce", "item_id": "42"})
        assert cmd.item_id == 42
        assert type(cmd.item_id) is int

    def test_to_dict_is_unchanged(self) -> None:
        cmd = LuaCommand(action_type=ActionType.RESEARCH_TECH, item_id="TECH_WRITING")
        assert cmd.to_dict() == {"action_type": "research_tech", "item_id": "TECH_WRITING"}


# ---------------------------------------------------------------------------
# D10: tests/conftest.py keeps the developer's shell out
# ---------------------------------------------------------------------------
class TestEnvIsolation:
    def test_research_env_is_absent_at_test_start(self) -> None:
        assert "CIV6AZ_BRIDGE_RULESET" not in os.environ
        assert "CIV6AZ_BRIDGE_GAMEINFO" not in os.environ

    def test_conftest_removes_a_value_from_the_developers_shell(self) -> None:
        env = {
            **os.environ,
            "CIV6AZ_BRIDGE_RULESET": "no-such-ruleset",
            "CIV6AZ_BRIDGE_GAMEINFO": "no-such-file.json",
        }
        node = f"{Path(__file__).resolve()}::TestEnvIsolation::test_research_env_is_absent_at_test_start"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", node, "-q", "-p", "no:cacheprovider"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=280,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout[-1500:]
        assert "1 passed" in proc.stdout


# ---------------------------------------------------------------------------
# D11: the encoder's research limit
# ---------------------------------------------------------------------------
class TestEncoderLimit:
    def test_position_99_is_set_and_100_is_ignored(self) -> None:
        assert state_converter_module.ENCODER_RESEARCH_SLOTS == 100
        conv = StateConverter()
        assert conv.encoder is not None
        empty = conv.convert(payload())
        empty["technologies"], empty["civics"] = [], []
        _, base = conv.encoder.encode(empty)
        edge = conv.convert(payload())
        edge["technologies"], edge["civics"] = [99, 100], [99, 100]
        _, vector = conv.encoder.encode(edge)
        assert vector[70 + 99] == 1.0
        assert vector[180 + 99] == 1.0
        changed = {int(i) for i in np.nonzero(np.asarray(vector != base))[0]}
        assert changed == {70 + 99, 180 + 99}


class TestLoopLikeTurn:
    def test_each_counter_moves_once_for_one_state(self, reg: Any) -> None:
        legal = [*END_ONLY, tech_action(["TECH_WRITING", "TECH_NOPE"], [2, 9])]
        parsed = GameStateData.from_dict(
            payload(("TECH_MINING", "TECH_GONE"), ("CIVIC_NOPE",), legal=legal)
        )
        states, actions = new_state(reg), new_actions(reg)
        converted = states.convert(parsed)
        states.convert_and_encode(parsed)
        actions.generate_action_mask(parsed.legal_actions, converted)
        cmd = actions.convert(flat(actions, TECH_RANGE, 0), converted, parsed.legal_actions)
        assert cmd.item_id == "TECH_WRITING"
        assert states.research_stats == {
            "conversions_without_registry": 0,
            "unmapped_techs": 1,
            "unmapped_civics": 1,
            "slots_out_of_range": 0,
        }
        assert actions.research_stats == {
            "unmapped_research_items": 1,
            "research_out_of_range": 0,
        }

    def test_without_a_registry_the_loop_counts_the_state_once(self) -> None:
        legal = [*END_ONLY, tech_action(None, [2, 100])]
        parsed = GameStateData.from_dict(payload(("TECH_POTTERY",), legal=legal))
        states, actions = new_state(False), new_actions(False)
        converted = states.convert(parsed)
        states.convert_and_encode(parsed)
        actions.generate_action_mask(parsed.legal_actions, converted)
        cmd = actions.convert(flat(actions, TECH_RANGE, 2), converted, parsed.legal_actions)
        assert cmd.item_id == 2
        assert states.research_stats["conversions_without_registry"] == 1
        assert actions.research_stats == {
            "unmapped_research_items": 0,
            "research_out_of_range": 1,
        }
