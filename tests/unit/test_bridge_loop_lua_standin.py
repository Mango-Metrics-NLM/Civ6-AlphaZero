"""GM-B7c-2 (B): the real agent loop against the real mod Lua, in the stand-in game.

Nothing is mocked: ``BridgeAgentLoop``, ``StateConverter`` and ``ActionConverter`` are the real
ones, and the commands go through the mod's own Lua (``LuaStandInBridge``) into the stand-in
game, which advances a turn when an end of turn was carried out. The policy is a small scripted
one that ranks the legal commands by a rule; it never sees an action index it did not decode.

What the stand-in cannot show: that the real game ends a turn the way it does. The refusals
here are provoked by changing the stand-in game between the mod's offer and the mod's execution.

Every test first checks that the mod really carried something out (the game's recorded
requests, the mod's acknowledgements, the game turn), so "not repeated" cannot pass emptily.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable
from itertools import pairwise
from typing import Any

import numpy as np

from alpha_zero.bridge.action_converter import ActionConverter
from alpha_zero.bridge.agent_loop import AgentLoopConfig, BridgeAgentLoop
from alpha_zero.bridge.file_bridge import StateTimeoutError
from alpha_zero.bridge.state_converter import StateConverter
from tests.fixtures import lua_commands as c
from tests.fixtures.lua_bridge import LuaStandInBridge

BOARD = 16
END = {"action_type": "end_turn"}
MOVE_101 = {"action_type": "unit_move", "unit_id": 101, "target": {"x": 0, "y": 1}}
RESEARCH_BLOCKER = (
    "Civ6Stub.notifications[0] = { { id = 5, blocking = EndTurnBlockingTypes."
    "ENDTURN_BLOCKING_RESEARCH, message = 'Choose a technology',"
    " typeName = 'NOTIFICATION_CHOOSE_TECH', dismissed = false } }"
)
#: Nothing the player can do but end the turn: no moves, nothing to research or produce.
NOTHING_TO_DO = (
    "Civ6Stub.unit(101).moves = 0\nCiv6Stub.unit(102).moves = 0\n"
    "Civ6Stub.players[0].tech.can = {}\nCiv6Stub.players[0].civic.can = {}\n"
    "Civ6Stub.cities[301].canProduce = {}\n"
)
SAFETY_STEPS = 400  # a regression must fail on an assertion, not spin to the suite timeout


def kind(command: dict[str, Any]) -> str:
    return str(command["action_type"])


def end_turn_last(command: dict[str, Any]) -> int:
    """Research is never the policy's own choice; end_turn only when nothing else is left."""
    return {"end_turn": 0, "research_tech": -1, "research_civic": -1}.get(kind(command), 1)


def end_turn_first(command: dict[str, Any]) -> int:
    """A policy that always prefers to end the turn; ties go to the lowest action index."""
    return 2 if kind(command) == "end_turn" else 1


class ScriptedPolicy:
    """Picks the highest-ranked legal command (ties: lowest index); records what it picked."""

    def __init__(self, action_converter: ActionConverter, rank: Callable[[dict[str, Any]], int]):
        self.action_converter = action_converter
        self.rank = rank
        self.picked: list[dict[str, Any]] = []

    async def select_action(self, obs: Any, mask: np.ndarray, converted_state: Any = None) -> Any:
        best: tuple[int, int] | None = None
        for idx in np.flatnonzero(mask):
            command = self.action_converter.convert(
                int(idx),
                converted_state,
                legal_actions=converted_state["_legal_actions"],
                validate=False,
            ).to_dict()
            if best is None or self.rank(command) > best[0]:
                best = (self.rank(command), int(idx))
        assert best is not None, "the policy was asked with an empty mask"
        chosen = self.action_converter.convert(
            best[1],
            converted_state,
            legal_actions=converted_state["_legal_actions"],
            validate=False,
        ).to_dict()
        self.picked.append(chosen)
        return best[1], {}


class TracedBridge(LuaStandInBridge):
    """Records the game turn each command was sent in; can spoil a command's execution.

    ``spoil`` maps a command to Lua run just before the mod reads it, and ``restore`` is the Lua
    that undoes it as soon as the mod has answered: the mod refuses a command it offered, and
    offers it again in the next export.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        spoil: dict[str, tuple[str, str]] | None = None,
        on_timeout: Callable[[int, Any], None] | None = None,
    ) -> None:
        super().__init__(runtime)
        self.turns: list[int] = []
        self.timeouts = 0
        self._spoil = spoil or {}
        self._on_timeout = on_timeout

    async def send_command(self, command: Any) -> bool:
        (single,) = json.loads(command.to_json())["commands"]
        self.turns.append(int(self.runtime.eval("Civ6Stub.turn")))
        before, after = self._spoil.get(json.dumps(single, sort_keys=True), ("", ""))
        if before:
            self.runtime.execute(before)
        sent = await super().send_command(command)
        if after:
            self.runtime.execute(after)
        return sent

    async def wait_for_state(self, timeout: float | None = None, require_new: bool = True) -> Any:
        try:
            return await super().wait_for_state(timeout, require_new)
        except StateTimeoutError:
            self.timeouts += 1
            if self._on_timeout is not None:
                self._on_timeout(self.timeouts, self.runtime)
            raise

    def per_turn(self) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for turn, command in zip(self.turns, self.commands_sent(), strict=True):
            grouped.setdefault(turn, []).append(command)
        return grouped


def play(
    runtime: Any,
    rank: Callable[[dict[str, Any]], int],
    *,
    bridge: TracedBridge | None = None,
    **config: Any,
) -> tuple[BridgeAgentLoop, TracedBridge, ScriptedPolicy, int]:
    bridge = bridge or TracedBridge(runtime)
    action_converter = ActionConverter(board_size=BOARD)
    policy = ScriptedPolicy(action_converter, rank)
    config.setdefault("max_turns", SAFETY_STEPS)
    loop = BridgeAgentLoop(
        bridge=bridge,
        policy=policy,
        state_converter=StateConverter(board_size=BOARD),
        action_converter=action_converter,
        config=AgentLoopConfig(per_turn_timeout_s=5.0, **config),
    )
    return loop, bridge, policy, asyncio.run(loop.run())


def requests_made(runtime: Any) -> int:
    """Game-changing requests the mod made (unit, city and research)."""
    return int(
        runtime.eval(
            "(function() local n = 0 for _, call in ipairs(Civ6Stub.calls) do"
            " if call.name ~= 'UI.RequestAction' then n = n + 1 end end return n end)()"
        )
    )


def end_turn_requests(runtime: Any) -> int:
    return int(
        runtime.eval(
            "(function() local n = 0 for _, call in ipairs(Civ6Stub.calls) do"
            " if call.name == 'UI.RequestAction' then n = n + 1 end end return n end)()"
        )
    )


def failed_acks(runtime: Any) -> list[str]:
    """The per-command error code of every acknowledgement that was a failure."""
    return [a["executed"][0]["error_code"] for a in c.acks(runtime) if not a["success"]]


# ---------------------------------------------------------------------------
# Game turns are counted and bounded
# ---------------------------------------------------------------------------
class TestGameTurnLimit:
    def test_three_game_turns_with_several_commands_each_and_nothing_on_the_fourth(self) -> None:
        runtime = c.loaded_turn()
        loop, bridge, _, code = play(runtime, end_turn_last, max_game_turns=3)
        assert requests_made(runtime) >= 3, "premise: the mod carried commands out"
        assert any(a["success"] for a in c.acks(runtime))
        assert code == BridgeAgentLoop.EXIT_OK
        per_turn = bridge.per_turn()
        assert sorted(per_turn) == [12, 13, 14]
        assert all(len(commands) >= 2 for commands in per_turn.values())
        assert all(len(commands) <= 40 for commands in per_turn.values()), "a turn is bounded"
        assert all(commands[-1] == END for commands in per_turn.values())
        assert runtime.eval("Civ6Stub.turn") == 15
        assert loop.stats()["game_turns_played"] == 3

    def test_control_without_the_limit_the_loop_plays_on(self) -> None:
        runtime = c.loaded_turn()
        _, bridge, _, code = play(runtime, end_turn_last, max_turns=40)
        assert code == BridgeAgentLoop.EXIT_OK
        assert len(bridge.per_turn()) >= 4

    def test_the_command_cap_ends_a_turn_the_policy_never_would(self) -> None:
        runtime = c.loaded_turn()
        # prefers anything to end_turn, and research is chosen at most once (so it cannot loop)
        _, bridge, _, code = play(runtime, end_turn_last, max_game_turns=1, max_commands_per_turn=3)
        assert code == BridgeAgentLoop.EXIT_OK
        commands = bridge.per_turn()[12]
        assert commands[:3] != [END] * 3 and END not in commands[:3]
        assert len(commands) == 4 and commands[-1] == END
        assert runtime.eval("Civ6Stub.turn") == 13


# ---------------------------------------------------------------------------
# What the mod refused is not sent again
# ---------------------------------------------------------------------------
class TestRefusedCommand:
    def _spoiled_move(self) -> dict[str, tuple[str, str]]:
        # Unit 101 has no moves while the mod reads the command, and its moves back afterwards.
        return {
            json.dumps(MOVE_101, sort_keys=True): (
                "Civ6Stub.moves101 = Civ6Stub.unit(101).moves\nCiv6Stub.unit(101).moves = 0",
                "Civ6Stub.unit(101).moves = Civ6Stub.moves101",
            )
        }

    def test_it_reaches_the_mod_once_and_the_turn_is_still_ended(self) -> None:
        def move_101_first(command: dict[str, Any]) -> int:
            return 3 if command == MOVE_101 else end_turn_last(command)

        runtime = c.loaded_turn()
        bridge = TracedBridge(runtime, spoil=self._spoiled_move())
        loop, bridge, policy, code = play(runtime, move_101_first, bridge=bridge, max_game_turns=1)
        assert failed_acks(runtime) == ["NOT_ENOUGH_MOVEMENT"], "premise: the mod refused it once"
        assert requests_made(runtime) >= 1, "premise: other commands were carried out"
        offered_again = [
            state
            for state in c.states(runtime)[1:]
            if any(
                action.get("action") == "move"
                and action.get("unit_id") == 101
                and [0, 1] in action.get("targets", [])
                for action in state["legal_actions"]
            )
        ]
        assert offered_again, "premise: the mod offered the refused move again"
        # Since the review round the refused move is taken out of the mask
        # before the policy is asked, so the policy names it only once.
        assert policy.picked.count(MOVE_101) == 1
        assert bridge.commands_sent().count(MOVE_101) == 1
        assert code == BridgeAgentLoop.EXIT_OK
        assert runtime.eval("Civ6Stub.turn") == 13
        assert bridge.commands_sent()[-1] == END
        assert loop.stats()["refused_commands"] == 1

    def test_control_the_same_policy_sends_it_every_time_when_the_mod_accepts_it(self) -> None:
        def move_101_first(command: dict[str, Any]) -> int:
            return 3 if command == MOVE_101 else end_turn_last(command)

        runtime = c.loaded_turn()
        _, bridge, _, _ = play(runtime, move_101_first, max_game_turns=1)
        assert failed_acks(runtime) == []
        assert bridge.commands_sent().count(MOVE_101) == 1, "carried out once, then it is gone"
        assert c.unit_in(c.states(runtime)[0], 101)["x"] == 0


# ---------------------------------------------------------------------------
# A turn the game blocks
# ---------------------------------------------------------------------------
class TestBlockedTurn:
    def test_research_is_chosen_for_a_policy_that_only_wants_to_end_the_turn(self) -> None:
        runtime = c.loaded_turn(extra=RESEARCH_BLOCKER)
        loop, bridge, _, code = play(runtime, end_turn_first, max_game_turns=1)
        stats = loop.stats()
        assert stats["blocked_end_turns"] >= 1, "premise: the game did block the turn"
        assert any(kind(cmd) == "research_tech" for cmd in bridge.commands_sent())
        assert runtime.eval("Civ6Stub.notifications[0][1].dismissed") is True
        assert code == BridgeAgentLoop.EXIT_OK
        assert runtime.eval("Civ6Stub.turn") == 13
        assert end_turn_requests(runtime) == 1
        assert stats["stalled_turns"] == 0
        assert stats["game_turns_played"] == 1

    def test_a_blocked_end_turn_is_not_repeated_while_nothing_has_changed(self) -> None:
        runtime = c.loaded_turn(extra=RESEARCH_BLOCKER)
        _, bridge, _, _ = play(runtime, end_turn_first, max_game_turns=1)
        commands = bridge.commands_sent()
        for earlier, later in pairwise(commands):
            assert not (earlier == END and later == END), "end_turn sent twice in a row"

    def test_research_and_civic_are_chosen_at_most_once_per_game_turn(self) -> None:
        def research_first(command: dict[str, Any]) -> int:
            return {"research_tech": 3, "research_civic": 2, "end_turn": 0}.get(kind(command), 1)

        runtime = c.loaded_turn()
        _, bridge, _, code = play(runtime, research_first, max_game_turns=2)
        assert code == BridgeAgentLoop.EXIT_OK
        research_requests = c.requests(runtime, "UI.RequestPlayerOperation")
        assert len(research_requests) == 4, "premise: the mod carried out one of each per turn"
        by_turn = bridge.per_turn()
        assert sorted(by_turn) == [12, 13]
        for turn, commands in by_turn.items():
            counts = Counter(kind(cmd) for cmd in commands)
            assert counts["research_tech"] == 1, (turn, counts)
            assert counts["research_civic"] == 1, (turn, counts)
            assert len(commands) <= 40, "bounded"

    def test_a_turn_that_cannot_be_unblocked_ends_in_a_stall_after_three_end_turns(self) -> None:
        runtime = c.loaded_turn(extra=RESEARCH_BLOCKER + "\n" + NOTHING_TO_DO)
        loop, bridge, _, code = play(runtime, end_turn_first)
        assert failed_acks(runtime) == ["END_TURN_BLOCKED"] * 3, "premise: the game refused 3"
        assert end_turn_requests(runtime) == 0
        assert code == BridgeAgentLoop.EXIT_STALLED == 3
        assert bridge.commands_sent() == [END] * 3
        stats = loop.stats()
        assert stats["stalled_turns"] == 1
        assert stats["blocked_end_turns"] == 3
        assert runtime.eval("Civ6Stub.turn") == 12

    def test_a_stall_with_things_still_to_do_is_bounded_too(self) -> None:
        unresearchable = "Civ6Stub.players[0].tech.can = {}\nCiv6Stub.players[0].civic.can = {}\n"
        runtime = c.loaded_turn(extra=RESEARCH_BLOCKER + "\n" + unresearchable)
        loop, bridge, _, code = play(runtime, end_turn_first)
        assert requests_made(runtime) >= 1, "premise: the units were really ordered about"
        assert code == BridgeAgentLoop.EXIT_STALLED
        commands = bridge.commands_sent()
        assert commands.count(END) >= 3
        assert len(commands) <= 40
        assert loop.stats()["stalled_turns"] == 1
        assert runtime.eval("Civ6Stub.turn") == 12


# ---------------------------------------------------------------------------
# on_stall="wait"
# ---------------------------------------------------------------------------
class TestStallWait:
    SETUP = RESEARCH_BLOCKER + "\n" + NOTHING_TO_DO

    def test_nothing_more_is_sent_on_the_stalled_turn_and_the_wait_ends_in_exit_three(
        self,
    ) -> None:
        runtime = c.loaded_turn(extra=self.SETUP)
        bridge = TracedBridge(runtime)
        loop, bridge, _, code = play(
            runtime, end_turn_first, bridge=bridge, on_stall="wait", stall_wait_s=0.2
        )
        assert failed_acks(runtime) == ["END_TURN_BLOCKED"] * 3, "premise: three were refused"
        assert bridge.timeouts >= 1, "the loop did wait for a new turn"
        assert bridge.commands_sent() == [END] * 3, "nothing was sent while waiting"
        assert code == BridgeAgentLoop.EXIT_STALLED
        assert loop.stats()["stalled_turns"] == 1

    def test_a_new_game_turn_resumes_the_loop(self) -> None:
        def someone_ends_the_turn(timeouts: int, runtime: Any) -> None:
            if timeouts == 2:
                runtime.execute(
                    "Civ6Stub.notifications[0][1].dismissed = true\n"
                    "Civ6Stub.nextTurn()\n"
                    "AlphaZeroBridge.OnTurnEnd(0)\nAlphaZeroBridge.OnTurnStart(0)"
                )

        runtime = c.loaded_turn(extra=self.SETUP)
        bridge = TracedBridge(runtime, on_timeout=someone_ends_the_turn)
        loop, bridge, _, code = play(
            runtime,
            end_turn_first,
            bridge=bridge,
            on_stall="wait",
            stall_wait_s=30.0,
            max_game_turns=1,
        )
        assert failed_acks(runtime) == ["END_TURN_BLOCKED"] * 3, "premise: it stalled first"
        assert runtime.eval("Civ6Stub.turn") == 13, "premise: the game moved on without the loop"
        assert code == BridgeAgentLoop.EXIT_OK
        assert bridge.commands_sent() == [END] * 3, "nothing was sent on the stalled turn"
        assert loop.stats()["game_turns_played"] == 1
