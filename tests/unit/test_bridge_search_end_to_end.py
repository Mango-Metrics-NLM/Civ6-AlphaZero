"""GM-B6b (C): the real search drives the real loop in the stand-in game (the mod's real Lua).

Real: ``BridgeAgentLoop``, ``StateConverter``, ``ActionConverter``, the translator,
``Civ6MctsPolicy`` over a real ``MicroCivNetwork`` and ``GumbelMCTS`` at 8 simulations, and the
mod's Lua, run by ``LuaStandInBridge``. Every outcome is read from the STAND-IN GAME afterwards
(a unit's tile, the city list, a city's build, the game turn, the notification), never from a
recorded call. How each scenario is made independent of an untrained network's luck is said in
the test (``Restricted`` hands the real search an extra ``allowed`` rule; the search and the
game still decide everything else).

What the stand-in cannot show: that the real game carries these requests out the same way; it is
a reading of the game's UI code, written from the same reading as the mod (``docs/LIVE_CIV6.md``).
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from alpha_zero.bridge.agent_loop import BridgeAgentLoop
from alpha_zero.bridge.turn_ledger import command_key
from tests.fixtures import lua_commands as c
from tests.fixtures.bridge_search_standin import (
    END,
    IGNORE_UNIT_REQUESTS,
    NO_UNITS_NO_BUILDS,
    NOTHING_RESEARCHABLE,
    RESEARCH_BLOCKER,
    Restricted,
    failed_acks,
    only,
    own_city_count,
    play,
    unit_xy,
)

CAP = 30
MOVES = only("unit_move")


def kinds(bridge: Any) -> list[str]:
    return [command["action_type"] for command in bridge.commands_sent()]


def game_turn(runtime: Any) -> int:
    return int(runtime.eval("Civ6Stub.turn"))


def end_turn_requests(runtime: Any) -> int:
    return int(
        runtime.eval(
            "(function() local n = 0 for _, call in ipairs(Civ6Stub.calls) do"
            " if call.name == 'UI.RequestAction' then n = n + 1 end end return n end)()"
        )
    )


class TestWhatTheSearchDoes:
    def test_a_unit_moved(self) -> None:
        """Only unit 102's moves may be named, one command per turn (the cap), so the tile it
        ends on is one of the two the game offered it, whichever the network prefers."""
        start = c.loaded_turn()
        offered = [
            tuple(t)
            for action in c.states(start)[0]["legal_actions"]
            if action["action"] == "move" and action["unit_id"] == 102
            for t in action["targets"]
        ]
        assert offered == [(1, 1), (0, 1)], "premise: the stand-in offers 102 two tiles"
        runtime, bridge, loop, _, code = play(
            keep=lambda cmd: cmd["action_type"] == "unit_move" and cmd["unit_id"] == 102,
            max_game_turns=1,
            max_commands_per_turn=1,
        )
        assert code == BridgeAgentLoop.EXIT_OK
        assert unit_xy(runtime, 102) in offered
        assert unit_xy(runtime, 102) != (1, 0), "the game moved it"
        assert unit_xy(runtime, 101) == (0, 0), "nothing else was moved"
        assert game_turn(runtime) == 13
        assert kinds(bridge) == ["unit_move", "end_turn"]
        assert loop.stats()["search_failures"] == 0

    def test_a_city_was_founded(self) -> None:
        """Only ``unit_found_city`` may be named: the settler is the single unit offered it, so
        the search has one root action; afterwards nothing may be named and the turn ends."""
        runtime, bridge, loop, policy, code = play(
            keep=only("unit_found_city"), max_game_turns=1, max_commands_per_turn=3
        )
        assert code == BridgeAgentLoop.EXIT_OK
        assert own_city_count(runtime) == 2, "the game has a second city"
        assert unit_xy(runtime, 101) is None, "the settler was used up"
        assert runtime.eval("Civ6Stub.cityRecords[0][2].x") == 0
        assert runtime.eval("Civ6Stub.cityRecords[0][2].y") == 0
        assert bridge.commands_sent() == [{"action_type": "unit_found_city", "unit_id": 101}, END]
        assert game_turn(runtime) == 13
        assert loop.stats()["search_fallbacks"] == 1, "then nothing could be named"
        assert policy.returned[0] == bridge.commands_sent()[0]

    def test_a_build_was_started(self) -> None:
        """Only ``city_produce`` of a warrior may be named, so the build is that warrior."""
        assert c.loaded_turn().eval("Civ6Stub.city(301).building") is None, "premise: idle city"
        runtime, bridge, _, _, code = play(
            keep=lambda cmd: (
                cmd["action_type"] == "city_produce" and cmd["item_id"] == "UNIT_WARRIOR"
            ),
            max_game_turns=1,
            max_commands_per_turn=3,
        )
        assert code == BridgeAgentLoop.EXIT_OK
        assert runtime.eval("Civ6Stub.city(301).building") == "UNIT_WARRIOR"
        assert kinds(bridge) == ["city_produce", "end_turn"]
        assert game_turn(runtime) == 13

    def test_a_turn_was_ended_by_the_unrestricted_search(self) -> None:
        """No unit has moves and the city can build nothing: the only translatable command is
        ``end_turn`` (research is offered, and a search in the simulator cannot express it)."""
        runtime, bridge, loop, _, code = play(NO_UNITS_NO_BUILDS, max_game_turns=1)
        offered = {a["action"] for a in c.states(runtime)[0]["legal_actions"]}
        assert "research_tech" in offered, "premise: research was on offer and not taken"
        assert code == BridgeAgentLoop.EXIT_OK
        assert bridge.commands_sent() == [END]
        assert game_turn(runtime) == 13
        assert end_turn_requests(runtime) == 1
        assert c.requests(runtime, "UI.RequestPlayerOperation") == []
        assert loop.stats()["search_decisions"] == 1


class TestBlockedTurn:
    def test_a_research_blocker_is_answered_and_the_turn_then_ends(self) -> None:
        """Nothing else to do, so the search ends the turn; the game blocks it; the search can
        then name nothing, and the loop answers the blocker with the research the game offers."""
        runtime, bridge, loop, _, code = play(
            RESEARCH_BLOCKER + NO_UNITS_NO_BUILDS, max_game_turns=1
        )
        stats = loop.stats()
        assert stats["blocked_end_turns"] == 1, "premise: the game really blocked the turn"
        assert code == BridgeAgentLoop.EXIT_OK
        assert kinds(bridge) == ["end_turn", "research_tech", "end_turn"]
        assert runtime.eval("Civ6Stub.notifications[0][1].dismissed") is True
        assert len(c.requests(runtime, "UI.RequestPlayerOperation")) == 1
        assert end_turn_requests(runtime) == 1
        assert game_turn(runtime) == 13
        assert stats["stalled_turns"] == 0
        assert stats["search_fallbacks"] == 1
        assert stats["search_failures"] == 0

    def test_without_the_blocker_no_research_is_sent(self) -> None:
        """Control: the loop does not research on its own."""
        _, bridge, _, _, _ = play(NO_UNITS_NO_BUILDS, max_game_turns=1)
        assert "research_tech" not in kinds(bridge)


class TestStall:
    def test_a_game_that_accepts_moves_and_ignores_them_ends_in_a_declared_stall(self) -> None:
        """The game accepts every unit and city request and does nothing; research is blocked and
        nothing can be researched. Each command is tried at most twice (the ledger's limit), then
        end_turn is refused three times in a row: a stall, well inside the turn's cap."""
        runtime, bridge, loop, _, code = play(
            RESEARCH_BLOCKER + NOTHING_RESEARCHABLE + IGNORE_UNIT_REQUESTS,
            max_commands_per_turn=CAP,
        )
        stats = loop.stats()
        accepted = [a for a in c.acks(runtime) if a["success"]]
        assert len(accepted) >= 3, "premise: the game accepted several commands"
        assert unit_xy(runtime, 101) == (0, 0) and unit_xy(runtime, 102) == (1, 0), (
            "premise: it did nothing with them"
        )
        assert code == BridgeAgentLoop.EXIT_STALLED
        assert stats["stalled_turns"] == 1
        assert game_turn(runtime) == 12
        sent = bridge.commands_sent()
        assert len(sent) < CAP, "declared before the cap, not by it"
        assert sent[-3:] == [END] * 3
        assert failed_acks(runtime) == ["END_TURN_BLOCKED"] * stats["blocked_end_turns"]
        assert end_turn_requests(runtime) == 0
        assert stats["no_effect_commands"] >= 1, "the ledger saw that nothing changed"
        tries = Counter(command_key(command) for command in sent if command != END)
        assert max(tries.values()) <= 2, tries


class TestRuledOutCommands:
    def _first_move(self) -> dict[str, Any]:
        _, bridge, _, _, _ = play(keep=MOVES, max_game_turns=1, max_commands_per_turn=5)
        first = bridge.commands_sent()[0]
        assert first["action_type"] == "unit_move", "premise: the search names a move first"
        return first

    def test_a_refused_command_is_not_returned_by_the_search_again_that_turn(self) -> None:
        first = self._first_move()
        unit = first["unit_id"]
        spoil = {
            json.dumps(first, sort_keys=True): (
                f"Civ6Stub.savedMoves = Civ6Stub.unit({unit}).moves\n"
                f"Civ6Stub.unit({unit}).moves = 0",
                f"Civ6Stub.unit({unit}).moves = Civ6Stub.savedMoves",
            )
        }
        runtime, bridge, loop, policy, code = play(
            keep=MOVES, spoil=spoil, max_game_turns=1, max_commands_per_turn=5
        )
        assert isinstance(policy, Restricted)
        assert bridge.commands_sent()[0] == first, "premise: the same first decision"
        assert failed_acks(runtime) == ["NOT_ENOUGH_MOVEMENT"], "premise: the game refused it"
        assert loop.stats()["refused_commands"] == 1
        assert bridge.commands_sent().count(first) == 1
        assert policy.returned.count(first) == 1, "the search itself did not name it again"
        assert len(bridge.commands_sent()) >= 3, "premise: the turn went on"
        assert code == BridgeAgentLoop.EXIT_OK
        assert bridge.commands_sent()[-1] == END
        assert game_turn(runtime) == 13

    def test_control_the_same_search_names_it_again_when_it_is_not_refused(self) -> None:
        """Without the refusal the first decision is the same, so the test above measures the
        ledger and not a different search."""
        first = self._first_move()
        _, again, _, _, _ = play(keep=MOVES, max_game_turns=1, max_commands_per_turn=5)
        assert again.commands_sent()[0] == first


class TestSeed:
    def test_the_same_base_seed_over_the_same_stand_in_gives_the_same_commands(self) -> None:
        first = play(seed=21, max_game_turns=2, max_commands_per_turn=6)
        second = play(seed=21, max_game_turns=2, max_commands_per_turn=6)
        sequence = first[1].commands_sent()
        assert len(sequence) >= 6, "premise: a real sequence"
        assert len({command_key(command) for command in sequence}) >= 4
        assert second[1].commands_sent() == sequence
        assert game_turn(first[0]) == game_turn(second[0]) == 14

    def test_the_sequence_does_not_depend_on_what_ran_before(self) -> None:
        first = play(seed=21, max_game_turns=1, max_commands_per_turn=5)[1].commands_sent()
        play(seed=3, max_game_turns=1, max_commands_per_turn=5)
        again = play(seed=21, max_game_turns=1, max_commands_per_turn=5)[1].commands_sent()
        assert again == first
