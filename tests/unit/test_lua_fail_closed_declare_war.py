"""GM-B9 / B3: the mod does not declare war unless the configuration allows it.

New configuration ``ALLOW_DECLARE_WAR`` (default false). With it false the diplomacy command whose
sub-action is DECLARE_WAR answers ``WOULD_START_WAR`` and nothing reaches the game: no ``DoAction``,
no ``UI.RequestDiplomacyMove``, no change to the stand-in's war record. With it true behaviour is as
before. Make peace and denounce have their own settings (``ALLOW_MAKE_PEACE`` / ``ALLOW_DENOUNCE``,
default false, see ``test_lua_fail_closed_diplomacy.py``): whatever ``ALLOW_DECLARE_WAR`` says they
answer ``DIPLOMACY_TURNED_OFF`` here. A declare-war action is never offered.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_harness as h

DECLARE_WAR = {"action_type": "diplomacy", "metadata": {"sub_action": 0, "target_player_id": 1}}
MAKE_PEACE = {"action_type": "diplomacy", "metadata": {"sub_action": 1, "target_player_id": 1}}
DENOUNCE = {"action_type": "diplomacy", "metadata": {"sub_action": 5, "target_player_id": 1}}

#: The stand-in models what DoAction / RequestDiplomacyMove do to the war record.
MODEL = "Civ6Stub.modelDiplomacyActions()\n"
AT_PEACE = "Civ6Stub.setPeace(0, 1, true)\n"


def _load(*, allow: bool | None, setup: str = MODEL + AT_PEACE) -> Any:
    runtime = h.load_mod(setup=setup)
    if allow is not None:
        runtime.execute(f'AlphaZeroConfig.Set("ALLOW_DECLARE_WAR", {str(allow).lower()})')
    return runtime


def _at_war(runtime: Any) -> bool:
    return bool(runtime.eval("Players[0]:GetDiplomacy():IsAtWarWith(1)"))


def _calls(runtime: Any) -> list[str]:
    count = int(runtime.eval("#Civ6Stub.diplomacyCalls"))
    return [runtime.eval(f"Civ6Stub.diplomacyCalls[{i}].path") for i in range(1, count + 1)]


@pytest.mark.parametrize("allow", [None, False], ids=["default", "explicitly false"])
class TestNotAllowed:
    def test_the_answer_is_would_start_war(self, allow: bool | None) -> None:
        runtime = _load(allow=allow)
        success, code, message = h.execute(runtime, DECLARE_WAR)
        assert (success, code) == (False, "WOULD_START_WAR")
        assert message

    def test_nothing_reaches_the_game_and_the_war_record_is_unchanged(
        self, allow: bool | None
    ) -> None:
        runtime = _load(allow=allow)
        assert _at_war(runtime) is False, "premise: at peace before"
        h.execute(runtime, DECLARE_WAR)
        assert _calls(runtime) == []
        assert h.stub_calls(runtime) == []
        assert _at_war(runtime) is False

    def test_the_older_ui_path_is_not_a_way_round(self, allow: bool | None) -> None:
        # DoAction absent: the old code fell through to UI.RequestDiplomacyMove.
        setup = MODEL + AT_PEACE + "Players[0].GetDiplomacy = function() return {} end\n"
        runtime = _load(allow=allow, setup=setup)
        assert h.execute(runtime, DECLARE_WAR)[:2] == (False, "WOULD_START_WAR")
        assert _calls(runtime) == []

    def test_the_other_sub_actions_are_refused_by_their_own_settings(
        self, allow: bool | None
    ) -> None:
        runtime = _load(allow=allow, setup=MODEL)  # at war
        assert h.execute(runtime, MAKE_PEACE)[:2] == (False, "DIPLOMACY_TURNED_OFF")
        assert h.execute(runtime, DENOUNCE)[:2] == (False, "DIPLOMACY_TURNED_OFF")
        assert _at_war(runtime) is True
        assert _calls(runtime) == []


class TestAllowed:
    def test_declare_war_goes_through_and_the_pair_is_at_war(self) -> None:
        runtime = _load(allow=True)
        assert _at_war(runtime) is False, "premise: at peace before"
        assert h.execute(runtime, DECLARE_WAR) == (True, None, "DECLARE_WAR against player 1")
        assert _calls(runtime) == ["DoAction"]
        assert _at_war(runtime) is True

    def test_allowing_war_does_not_allow_peace_or_denouncing(self) -> None:
        runtime = _load(allow=True, setup=MODEL)
        assert h.execute(runtime, MAKE_PEACE)[:2] == (False, "DIPLOMACY_TURNED_OFF")
        assert h.execute(runtime, DENOUNCE)[:2] == (False, "DIPLOMACY_TURNED_OFF")
        assert _at_war(runtime) is True


def test_the_setting_exists_and_defaults_to_false() -> None:
    assert h.load_mod().eval('AlphaZeroConfig.Get("ALLOW_DECLARE_WAR")') is False


@pytest.mark.parametrize("allow", [None, False, True], ids=["default", "false", "true"])
def test_declare_war_is_never_offered(allow: bool | None) -> None:
    state = h.export_state(_load(allow=allow, setup=""))
    assert state["legal_actions"], "premise: a state with actions was exported"
    assert [a for a in state["legal_actions"] if a["action"] == "diplomacy"] == []
    assert "declare" not in str(state["legal_actions"]).lower()
