"""The mod makes peace and denounces only when allowed, and only when the game's state says so.

Two settings, ``ALLOW_MAKE_PEACE`` and ``ALLOW_DENOUNCE``, default false; only the boolean ``true``
turns one on. Off: ``DIPLOMACY_TURNED_OFF``, nothing reaches the game. On: the game decides, and a
question the game cannot answer (missing API, a raising call, a non-boolean answer) is a refusal
(``EXECUTION_FAILED``, "Cannot ask the game"), never a go-ahead. Neither action is ever offered.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_harness as h

MAKE_PEACE = {"action_type": "diplomacy", "metadata": {"sub_action": 1, "target_player_id": 1}}
DENOUNCE = {"action_type": "diplomacy", "metadata": {"sub_action": 5, "target_player_id": 1}}
SELF_PEACE = {"action_type": "diplomacy", "metadata": {"sub_action": 1, "target_player_id": 0}}
SELF_DENOUNCE = {"action_type": "diplomacy", "metadata": {"sub_action": 5, "target_player_id": 0}}

MODEL = "Civ6Stub.modelDiplomacyActions()\n"
AT_WAR = "Civ6Stub.setWar(0, 1, true)\n"
AT_PEACE = "Civ6Stub.setPeace(0, 1, true)\n"
MET = "Civ6Stub.setMet(0, 1, true)\n"

# (command, setting, setup that makes the game's state allow it)
ACTIONS = {
    "make_peace": (MAKE_PEACE, "ALLOW_MAKE_PEACE", AT_WAR),
    "denounce": (DENOUNCE, "ALLOW_DENOUNCE", AT_PEACE + MET),
}
BOTH = pytest.mark.parametrize("name", list(ACTIONS))


def _load(setting: str | None, value: str = "true", setup: str = "") -> Any:
    runtime = h.load_mod(setup=MODEL + setup)
    if setting is not None:
        runtime.execute(f'AlphaZeroConfig.Set("{setting}", {value})')
    return runtime


def _calls(runtime: Any) -> list[str]:
    count = int(runtime.eval("#Civ6Stub.diplomacyCalls"))
    return [runtime.eval(f"Civ6Stub.diplomacyCalls[{i}].path") for i in range(1, count + 1)]


def _at_war(runtime: Any) -> bool:
    return bool(runtime.eval("Players[0]:GetDiplomacy():IsAtWarWith(1)"))


def _wrap(method: str, expr: str) -> str:
    """Lua: player 0's diplomacy object answers ``method`` with ``expr`` (after the model)."""
    return (
        "do local prev = Players[0].GetDiplomacy\n"
        "Players[0].GetDiplomacy = function(self) local d = prev(self)\n"
        f"d.{method} = {expr}\nreturn d end end\n"
    )


@BOTH
def test_settings_exist_and_default_to_false(name: str) -> None:
    key = ACTIONS[name][1]
    assert h.load_mod().eval(f'AlphaZeroConfig.Get("{key}")') is False


@BOTH
@pytest.mark.parametrize("value", [None, "false", '"true"', "1", '"yes"'])
class TestOff:
    """Unset, false, or anything that is not the boolean true."""

    def _runtime(self, name: str, value: str | None) -> Any:
        _, key, state = ACTIONS[name]
        return _load(None if value is None else key, value or "", state)

    def test_refused_naming_the_setting(self, name: str, value: str | None) -> None:
        success, code, message = h.execute(self._runtime(name, value), ACTIONS[name][0])
        assert (success, code) == (False, "DIPLOMACY_TURNED_OFF")
        assert ACTIONS[name][1] in message

    def test_nothing_reaches_the_game_and_the_record_is_unchanged(
        self, name: str, value: str | None
    ) -> None:
        runtime = self._runtime(name, value)
        before = _at_war(runtime)
        h.execute(runtime, ACTIONS[name][0])
        assert _calls(runtime) == []
        assert h.stub_calls(runtime) == []
        assert _at_war(runtime) is before


def test_each_setting_turns_on_only_its_own_action() -> None:
    runtime = _load("ALLOW_MAKE_PEACE", setup=AT_WAR + MET)
    assert h.execute(runtime, DENOUNCE)[:2] == (False, "DIPLOMACY_TURNED_OFF")
    runtime = _load("ALLOW_DENOUNCE", setup=AT_PEACE + MET)
    assert h.execute(runtime, MAKE_PEACE)[:2] == (False, "DIPLOMACY_TURNED_OFF")


class TestMakePeaceOn:
    def test_at_war_goes_through_with_one_call(self) -> None:
        runtime = _load("ALLOW_MAKE_PEACE", setup=AT_WAR)
        assert _at_war(runtime) is True, "premise: at war before"
        assert h.execute(runtime, MAKE_PEACE) == (True, None, "MAKE_PEACE against player 1")
        assert _calls(runtime) == ["DoAction"]
        assert _at_war(runtime) is False

    def test_already_at_peace_is_refused_and_nothing_is_sent(self) -> None:
        runtime = _load("ALLOW_MAKE_PEACE", setup=AT_PEACE)
        success, code, _ = h.execute(runtime, MAKE_PEACE)
        assert (success, code) == (False, "OPERATION_REFUSED")
        assert _calls(runtime) == []
        assert _at_war(runtime) is False


class TestDenounceOn:
    def test_met_and_at_peace_goes_through_with_one_call(self) -> None:
        runtime = _load("ALLOW_DENOUNCE", setup=AT_PEACE + MET)
        assert h.execute(runtime, DENOUNCE) == (True, None, "DENOUNCE against player 1")
        assert _calls(runtime) == ["DoAction"]
        assert _at_war(runtime) is False

    def test_unmet_is_refused_and_nothing_is_sent(self) -> None:
        runtime = _load("ALLOW_DENOUNCE", setup=AT_PEACE)
        assert h.execute(runtime, DENOUNCE)[:2] == (False, "OPERATION_REFUSED")
        assert _calls(runtime) == []

    def test_at_war_is_refused_and_nothing_is_sent(self) -> None:
        runtime = _load("ALLOW_DENOUNCE", setup=AT_WAR + MET)
        assert h.execute(runtime, DENOUNCE)[:2] == (False, "OPERATION_REFUSED")
        assert _calls(runtime) == []
        assert _at_war(runtime) is True


# Ways the game cannot be asked. Each is (id, Lua that breaks one method, or the whole object).
BREAKS = {
    "missing method": "nil",
    "raises": "function() error('no answer') end",
    "returns nil": "function() return nil end",
    "returns 1": "function() return 1 end",
    "returns string true": "function() return 'true' end",
}


def _cannot_ask(runtime: Any, command: dict[str, Any]) -> None:
    success, code, message = h.execute(runtime, command)
    assert (success, code) == (False, "EXECUTION_FAILED")
    assert message.startswith("Cannot ask the game")
    assert _calls(runtime) == []


@pytest.mark.parametrize("how", list(BREAKS))
class TestCannotAsk:
    def test_make_peace_when_is_at_war_with_cannot_be_answered(self, how: str) -> None:
        setup = AT_WAR + _wrap("IsAtWarWith", BREAKS[how])
        _cannot_ask(_load("ALLOW_MAKE_PEACE", setup=setup), MAKE_PEACE)

    def test_denounce_when_has_met_cannot_be_answered(self, how: str) -> None:
        setup = AT_PEACE + MET + _wrap("HasMet", BREAKS[how])
        _cannot_ask(_load("ALLOW_DENOUNCE", setup=setup), DENOUNCE)

    def test_denounce_when_is_at_war_with_cannot_be_answered(self, how: str) -> None:
        setup = MET + _wrap("IsAtWarWith", BREAKS[how])
        _cannot_ask(_load("ALLOW_DENOUNCE", setup=setup), DENOUNCE)


@BOTH
def test_no_diplomacy_object_at_all_is_cannot_ask(name: str) -> None:
    command, key, state = ACTIONS[name]
    runtime = _load(key, setup=state + "Civ6Stub.removeDiplomacy(0)\n")
    _cannot_ask(runtime, command)


@BOTH
def test_a_diplomacy_object_that_is_not_a_table_is_cannot_ask(name: str) -> None:
    command, key, state = ACTIONS[name]
    runtime = _load(key, setup=state + "Players[0].GetDiplomacy = function() return 7 end\n")
    _cannot_ask(runtime, command)


@BOTH
def test_a_target_equal_to_the_acting_player_is_invalid(name: str) -> None:
    command, key, state = ACTIONS[name]
    runtime = _load(key, setup=state)
    self_command = SELF_PEACE if command is MAKE_PEACE else SELF_DENOUNCE
    assert h.execute(runtime, self_command)[:2] == (False, "INVALID_TARGET")
    assert _calls(runtime) == []


@pytest.mark.parametrize("settings_on", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("state", [AT_WAR, AT_PEACE + MET], ids=["war", "peace-met"])
def test_neither_action_is_ever_offered(settings_on: bool, state: str) -> None:
    runtime = _load(None, setup=state)
    if settings_on:
        runtime.execute('AlphaZeroConfig.Set("ALLOW_MAKE_PEACE", true)')
        runtime.execute('AlphaZeroConfig.Set("ALLOW_DENOUNCE", true)')
    exported = h.export_state(runtime)
    assert exported["legal_actions"], "premise: a state with actions was exported"
    assert [a for a in exported["legal_actions"] if a["action"] == "diplomacy"] == []
    text = str(exported["legal_actions"]).lower()
    assert "peace" not in text
    assert "denounce" not in text


# --- The target must be a major civilization (review follow-up) -------------------------------

#: (target id, setup that makes the player that kind). 10 is the stand-in's city-state, 63 the
#: barbarians; a free city is player 1 turned into one.
NOT_MAJOR = {
    "barbarian": (63, ""),
    "city-state": (10, ""),
    "free city": (1, "Civ6Stub.kinds[1].major = false\nCiv6Stub.addFreeCities(1, true)\n"),
}


def _aimed(name: str, target: int) -> tuple[dict[str, Any], str, str]:
    """The action at ``target``, its setting, and setup that makes the game's state allow it."""
    command, key, _ = ACTIONS[name]
    command = {"action_type": "diplomacy", "metadata": {**command["metadata"]}}
    command["metadata"]["target_player_id"] = target
    if name == "make_peace":
        state = f"Civ6Stub.setWar(0, {target}, true)\n"
    else:
        state = f"Civ6Stub.setPeace(0, {target}, true)\nCiv6Stub.setMet(0, {target}, true)\n"
    return command, key, state


@BOTH
@pytest.mark.parametrize("kind", list(NOT_MAJOR))
def test_a_target_that_is_not_a_major_civilization_is_refused(name: str, kind: str) -> None:
    target, make = NOT_MAJOR[kind]
    command, key, state = _aimed(name, target)
    runtime = _load(key, setup=make + state)
    assert runtime.eval(f"Players[{target}]:IsMajor()") is False, "premise: not a major"
    success, code, message = h.execute(runtime, command)
    assert (success, code) == (False, "OPERATION_REFUSED")
    assert "major" in message.lower()
    assert _calls(runtime) == []


IS_MAJOR_BREAKS = {
    "player missing": "Players[1] = nil",
    "method missing": "Players[1].IsMajor = nil",
    "raises": "Players[1].IsMajor = function() error('no answer') end",
    "returns nil": "Players[1].IsMajor = function() return nil end",
    "returns 1": "Players[1].IsMajor = function() return 1 end",
    "returns string true": "Players[1].IsMajor = function() return 'true' end",
}


@BOTH
@pytest.mark.parametrize("how", list(IS_MAJOR_BREAKS))
def test_is_major_that_cannot_be_answered_is_cannot_ask(name: str, how: str) -> None:
    command, key, state = ACTIONS[name]
    runtime = _load(key, setup=state)
    runtime.execute(IS_MAJOR_BREAKS[how])
    _cannot_ask(runtime, command)


# --- Self-target is refused for every sub-action, before anything else ------------------------


@pytest.mark.parametrize("allow", [False, True], ids=["war off", "war on"])
def test_a_self_targeted_declare_war_is_invalid(allow: bool) -> None:
    runtime = _load("ALLOW_DECLARE_WAR", value=str(allow).lower())
    command = {"action_type": "diplomacy", "metadata": {"sub_action": 0, "target_player_id": 0}}
    assert h.execute(runtime, command)[:2] == (False, "INVALID_TARGET")
    assert _calls(runtime) == []
    assert h.stub_calls(runtime) == []


def test_a_self_targeted_deferred_sub_action_is_invalid() -> None:
    runtime = _load(None)
    command = {"action_type": "diplomacy", "metadata": {"sub_action": 2, "target_player_id": 0}}
    assert h.execute(runtime, command)[:2] == (False, "INVALID_TARGET")
    assert _calls(runtime) == []
    assert h.stub_calls(runtime) == []
