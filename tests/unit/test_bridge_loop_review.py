"""GM-B7c-2 review round (loop with fakes): the cap trap, filtered offers, failed sends.

Small fake classes, not mocks, so each test reads what the bridge was really given. The
ledger sees a state as ``SimpleNamespace`` without a summary, so what a test needs from a
state change is spelled out in the acknowledgements and the ``end_turn_blocking`` field.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np

from alpha_zero.bridge.action_converter import InvalidActionError
from alpha_zero.bridge.agent_loop import AgentLoopConfig, BridgeAgentLoop
from alpha_zero.bridge.file_bridge import AckTimeoutError
from alpha_zero.bridge.protocol import AckMessage, LuaCommand

END = {"action_type": "end_turn"}
RESEARCH = {"action_type": "research_tech", "item_id": 2}
BLOCKER = "ENDTURN_BLOCKING_RESEARCH"


def move(n: int) -> dict[str, Any]:
    return {"action_type": "unit_move", "unit_id": n, "target": {"x": n, "y": 0}}


def game_state(
    turn: int = 5, *, over: bool = False, blocking: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        turn_id=turn,
        player_id=0,
        is_game_over=over,
        winner_id=None,
        legal_actions=[],
        end_turn_blocking=blocking,
    )


def success() -> AckMessage:
    return AckMessage(msg_id=1, success=True)


def failure(code: str) -> AckMessage:
    executed = [{"success": False, "error_code": code, "message": code}]
    return AckMessage(msg_id=1, success=False, error_code="COMMAND_FAILED", executed=executed)


class Bridge:
    def __init__(self, states: list[Any], ack_for: Callable[[dict[str, Any]], Any] | None = None):
        self._states = list(states)
        self._ack_for = ack_for or (lambda _c: success())
        self._id = 0
        self.sent: list[dict[str, Any]] = []

    def next_msg_id(self) -> int:
        self._id += 1
        return self._id

    def check_heartbeat(self) -> bool:
        return True

    async def wait_for_state(self, timeout: float | None = None, require_new: bool = True) -> Any:
        return self._states.pop(0)

    def _record(self, command: Any) -> dict[str, Any]:
        (single,) = json.loads(command.to_json())["commands"]
        self.sent.append(single)
        return single

    async def send_command(self, command: Any) -> bool:
        self._record(command)
        return True

    async def send_and_wait_with_retry(self, command: Any, *_: Any, **__: Any) -> Any:
        result = self._ack_for(self._record(command))
        if isinstance(result, Exception):
            raise result
        return result


class StateConv:
    def convert(self, state: Any) -> dict[str, Any]:
        obs = {"spatial": np.zeros((13, 8, 8), np.float32), "vector": np.zeros(64, np.float32)}
        return {"_legal_actions": [], "_obs": obs}


class TableConv:
    """The offered actions are the table's keys; ``convert`` counts its calls."""

    def __init__(self, table: dict[int, dict[str, Any]]) -> None:
        self.table = table
        self.converted: list[int] = []

    def generate_action_mask(self, legal_actions: Any, converted_state: Any) -> np.ndarray:
        mask = np.zeros(max(self.table) + 3, dtype=np.float32)
        for index in self.table:
            mask[index] = 1.0
        return mask

    def convert(
        self, idx: int, state: Any, legal_actions: Any = None, validate: bool = True
    ) -> Any:
        self.converted.append(idx)
        if idx not in self.table:
            raise InvalidActionError(f"action {idx} is not offered")
        return LuaCommand.from_dict(self.table[idx])


class Policy:
    """The first of ``preference`` the mask allows; ``blind`` answers without reading the mask."""

    def __init__(
        self, preference: list[int], *, blind: bool = False, script: list[int] | None = None
    ) -> None:
        self.preference = preference
        self.script = list(script or [])
        self.blind = blind
        self.masks: list[np.ndarray] = []

    async def select_action(self, obs: Any, mask: np.ndarray, converted_state: Any = None) -> Any:
        self.masks.append(mask.copy())
        if self.script:
            return self.script.pop(0), {}
        if self.blind:
            return self.preference[0], {}
        for idx in self.preference:
            if idx < len(mask) and mask[idx] > 0:
                return idx, {}
        raise AssertionError(f"nothing the policy wants is in the mask: {mask}")


def play(
    states: list[Any],
    table: dict[int, dict[str, Any]],
    policy: Policy,
    ack_for: Callable[[dict[str, Any]], Any] | None = None,
    **config: Any,
) -> tuple[BridgeAgentLoop, Bridge, TableConv, int]:
    bridge = Bridge(states, ack_for)
    converter = TableConv(table)
    config.setdefault("max_turns", 0)
    loop = BridgeAgentLoop(
        bridge=bridge,  # type: ignore[arg-type]
        policy=policy,
        state_converter=StateConv(),  # type: ignore[arg-type]
        action_converter=converter,  # type: ignore[arg-type]
        config=AgentLoopConfig(per_turn_timeout_s=0.5, **config),
    )
    return loop, bridge, converter, asyncio.run(loop.run())


def same_turn(count: int) -> list[Any]:
    return [game_state() for _ in range(count)] + [game_state(over=True)]


# ---------------------------------------------------------------------------
# 1. The cap must not trap a turn that is blocked late
# ---------------------------------------------------------------------------
class TestCapAndBlocker:
    TABLE = {
        0: END,
        2: move(1),
        4: move(2),
        3: RESEARCH,
        5: {"action_type": "unit_skip", "unit_id": 9},
    }

    @staticmethod
    def acks(research_done: list[bool]) -> Callable[[dict[str, Any]], Any]:
        def ack_for(command: dict[str, Any]) -> Any:
            if command == RESEARCH:
                research_done.append(True)
            if command == END and not research_done:
                return failure("END_TURN_BLOCKED")
            return success()

        return ack_for

    def test_research_is_chosen_past_the_cap_and_the_turn_ends(self) -> None:
        done: list[bool] = []
        table = {k: v for k, v in self.TABLE.items() if k != 5}
        policy = Policy([0, 3], script=[2, 4])  # two moves, then end_turn before research
        loop, bridge, _, code = play(
            same_turn(5), table, policy, self.acks(done), max_commands_per_turn=2
        )
        assert done, "premise: the research really went out"
        assert bridge.sent == [move(1), move(2), END, RESEARCH, END]
        assert code == BridgeAgentLoop.EXIT_OK
        assert loop.stats()["stalled_turns"] == 0

    def test_past_the_cap_only_unblocking_kinds_are_on_offer(self) -> None:
        done: list[bool] = []
        policy = Policy([0, 5, 3], script=[2, 4])
        _, bridge, _, _ = play(
            same_turn(4), self.TABLE, policy, self.acks(done), max_commands_per_turn=2
        )
        # the step after the blocked end_turn: moves and the blocked end_turn are not offered
        offered = list(np.flatnonzero(policy.masks[2]))
        assert offered == [3, 5]
        assert bridge.sent[:3] == [move(1), move(2), END]
        assert bridge.sent[3] in ({"action_type": "unit_skip", "unit_id": 9}, RESEARCH)

    def test_control_without_a_blocker_only_end_turn_goes_past_the_cap(self) -> None:
        policy = Policy([3], script=[2, 4])
        _, bridge, _, _ = play(same_turn(5), self.TABLE, policy, max_commands_per_turn=2)
        assert bridge.sent == [move(1), move(2), END, END, END]
        assert len(policy.masks) == 2, "not asked past the cap"

    def test_twice_the_cap_is_the_limit_and_the_turn_then_stalls(self) -> None:
        # cap 1: the move is command 1, the blocked end_turn command 2 == 2 * cap: no room left
        done: list[bool] = []
        table = {0: END, 2: move(1), 3: RESEARCH}
        _, bridge, _, code = play(
            same_turn(10),
            table,
            Policy([0, 3], script=[2]),
            self.acks(done),
            max_commands_per_turn=1,
        )
        assert not done, "research was not sent past twice the cap"
        assert bridge.sent == [move(1), END, END, END]
        assert code == BridgeAgentLoop.EXIT_STALLED

    def test_just_inside_twice_the_cap_research_is_still_sent(self) -> None:
        done: list[bool] = []
        table = {0: END, 2: move(1), 3: RESEARCH}
        _, bridge, _, code = play(
            same_turn(4),
            table,
            Policy([0, 3], script=[2]),
            self.acks(done),
            max_commands_per_turn=2,
        )
        assert done
        assert bridge.sent == [move(1), END, RESEARCH, END]
        assert RESEARCH in bridge.sent
        assert bridge.sent[-1] == END and code == BridgeAgentLoop.EXIT_OK


# ---------------------------------------------------------------------------
# 2. Suppressed actions are filtered out before the policy is asked
# ---------------------------------------------------------------------------
class TestOffersAreFiltered:
    def test_many_refused_actions_do_not_use_up_the_reselections(self) -> None:
        refused_moves = [move(n) for n in range(1, 13)]
        table = {n: refused_moves[n - 1] for n in range(1, 13)}
        table[20] = {"action_type": "unit_skip", "unit_id": 77}
        table[0] = END
        refused = {json.dumps(m, sort_keys=True) for m in refused_moves}

        def ack_for(command: dict[str, Any]) -> Any:
            return (
                failure("OPERATION_REFUSED")
                if json.dumps(command, sort_keys=True) in refused
                else success()
            )

        policy = Policy([*range(1, 13), 20, 0])
        loop, bridge, _, _ = play(same_turn(14), table, policy, ack_for)
        assert bridge.sent[:12] == refused_moves, "premise: each was tried and refused once"
        assert bridge.sent[12] == table[20], "the useful action is sent as soon as it is the best"
        assert loop.stats()["reselections"] == 0
        assert loop.stats()["refused_commands"] == 12

    def test_a_policy_that_insists_on_a_blocked_end_turn_gets_something_else(self) -> None:
        table = {0: END, 1: {"action_type": "unit_skip", "unit_id": 3}}

        def ack_for(command: dict[str, Any]) -> Any:
            return failure("END_TURN_BLOCKED") if command == END else success()

        policy = Policy([0], blind=True)
        _, bridge, _, _ = play(same_turn(2), table, policy, ack_for)
        assert bridge.sent == [END, table[1]]

    def test_so_does_a_policy_that_answers_outside_the_mask(self) -> None:
        table = {0: END, 1: {"action_type": "unit_skip", "unit_id": 3}}

        def ack_for(command: dict[str, Any]) -> Any:
            return failure("END_TURN_BLOCKED") if command == END else success()

        policy = Policy([0], blind=True)
        loop, bridge, _, _ = play(same_turn(2), table, policy, ack_for)
        # step 2: the mask no longer holds end_turn (index 0), the policy still says 0
        assert policy.masks[1][0] == 0.0
        assert bridge.sent == [END, table[1]]
        assert len(policy.masks) == 2, "asked once per step"
        assert loop.stats()["reselections"] == 0

    def test_when_nothing_is_ruled_out_nothing_is_converted_but_the_choice(self) -> None:
        table = {0: END, 1: {"action_type": "unit_skip", "unit_id": 3}, 2: move(1)}
        policy = Policy([1])
        _, bridge, converter, _ = play(same_turn(1), table, policy)
        assert bridge.sent == [table[1]]
        assert converter.converted == [1]
        np.testing.assert_array_equal(policy.masks[0], converter.generate_action_mask([], {}))


# ---------------------------------------------------------------------------
# 7. An index outside the mask ends the asking
# ---------------------------------------------------------------------------
class TestIndexOutsideTheMask:
    def test_the_loop_stops_asking_after_it(self) -> None:
        table = {0: END, 1: {"action_type": "unit_skip", "unit_id": 3}}
        policy = Policy([3], blind=True)  # 3 is in range but not offered
        loop, bridge, _, _ = play(same_turn(1), table, policy)
        assert len(policy.masks) == 1
        assert loop.stats()["reselections"] == 0
        assert bridge.sent == [END]


# ---------------------------------------------------------------------------
# 5. A send that raises is not a blocked end_turn
# ---------------------------------------------------------------------------
class TestFailedSends:
    def test_three_end_turns_that_time_out_are_not_a_stall(self) -> None:
        def ack_for(command: dict[str, Any]) -> Any:
            return AckTimeoutError(1, 0.5)

        states = [game_state(blocking=BLOCKER) for _ in range(4)] + [game_state(over=True)]
        loop, bridge, _, code = play(states, {0: END}, Policy([0]), ack_for)
        assert bridge.sent == [END] * 4, "premise: every send reached the bridge and failed"
        assert loop.stats()["ack_failures"] == 4
        assert code == BridgeAgentLoop.EXIT_OK
        assert loop.stats()["stalled_turns"] == 0
        assert loop.stats()["blocked_end_turns"] == 0


# ---------------------------------------------------------------------------
# 6. The fallback end_turn seen blocked in the next state is counted in stats
# ---------------------------------------------------------------------------
class TestFallbackBlockedStat:
    def test_blocked_end_turns_includes_it(self) -> None:
        class Broken:
            async def select_action(self, *_: Any, **__: Any) -> Any:
                raise RuntimeError("boom")

        states = [game_state(), game_state(blocking=BLOCKER), game_state(over=True)]
        loop, bridge, _, _ = play(states, {0: END}, Broken())  # type: ignore[arg-type]
        assert bridge.sent == [END, END], "premise: both were fallbacks"
        stats = loop.stats()
        assert stats["fallback_end_turns"] == 2
        assert stats["blocked_end_turns"] == 1
