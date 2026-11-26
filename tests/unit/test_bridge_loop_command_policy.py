"""GM-B6b (B): the real agent loop with fake command policies and the loop fakes.

A command policy answers ``select_command`` with ``(command, record)``; the loop must ask it
through ``command_policy.choose`` and never call its ``select_action``. The fakes are the
fixtures in ``tests/fixtures/bridge_loop_fakes.py`` (the reselect tests' own shape).
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.bridge.agent_loop import BridgeAgentLoop
from alpha_zero.utils.seeding import derive_seed
from tests.fixtures.bridge_loop_fakes import (
    END,
    MOVE_1,
    RESEARCH,
    SKIP_1,
    SKIP_2,
    Harness,
    blocking_end_turn,
    game_state,
    play,
    refusing,
)

PROBES = {"end": END, "skip_1": SKIP_1, "skip_2": SKIP_2, "move": MOVE_1, "research": RESEARCH}


class CommandPolicy:
    """Answers from ``script`` in order (the last answer repeats); an exception is raised."""

    def __init__(self, script: list[Any], *, seed: int | None = None) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.select_action_calls = 0
        self.config = type("Config", (), {"seed": seed})()

    async def select_command(
        self, converted_state: Any, *, allowed: Any = None, decision_seed: Any = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # ``allowed`` reads the live ledger, so what it says is recorded at the time of asking.
        verdicts = {name: allowed(cmd) for name, cmd in PROBES.items()} if allowed else {}
        self.calls.append(
            {
                "state": converted_state,
                "allowed": allowed,
                "decision_seed": decision_seed,
                "verdicts": verdicts,
            }
        )
        answer = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(answer, BaseException):
            raise answer
        command, record = answer
        return dict(command), dict(record)

    async def select_action(self, *_: Any, **__: Any) -> Any:
        self.select_action_calls += 1
        raise AssertionError("select_action must not be called on a command policy")


def says(command: dict[str, Any], **record: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return command, {"fallback": None, **record}


def falls_back(why: str = "nothing translatable") -> tuple[dict[str, Any], dict[str, Any]]:
    return END, {"fallback": why}


def turn(n: int = 5, count: int = 1) -> list[Any]:
    return [game_state(n) for _ in range(count)] + [game_state(over=True)]


class TestCommandIsSent:
    def test_the_policy_is_asked_through_select_command_and_its_command_is_sent(self) -> None:
        policy = CommandPolicy([says(SKIP_1)])
        loop, bridge, converter, _ = play(turn(count=2), policy)
        assert bridge.sent == [SKIP_1, SKIP_1]
        assert policy.select_action_calls == 0
        assert len(policy.calls) == 2
        assert policy.calls[0]["state"] is converter.converted[0]
        assert loop.stats()["commands_sent"] == 2

    def test_it_goes_through_the_same_path_as_any_command(self) -> None:
        """The ledger sees it: a refusal is remembered and the policy is told it is ruled out."""
        policy = CommandPolicy([says(SKIP_1), says(SKIP_1), says(SKIP_2)])
        loop, bridge, _, _ = play(turn(count=3), policy, ack_for=refusing(SKIP_1))
        assert bridge.sent[0] == SKIP_1
        assert bridge.sent.count(SKIP_1) == 1, "the refused command is not sent again"
        assert loop.stats()["refused_commands"] == 1
        verdicts = policy.calls[1]["verdicts"]
        assert verdicts["skip_1"] is False
        assert verdicts["skip_2"] is True

    def test_a_command_the_ledger_ruled_out_is_not_sent_even_when_the_policy_returns_it(
        self,
    ) -> None:
        policy = CommandPolicy([says(SKIP_1)])
        loop, bridge, _, _ = play(turn(count=3), policy, ack_for=refusing(SKIP_1))
        assert bridge.sent == [SKIP_1, END, END]
        stats = loop.stats()
        assert stats["search_decisions"] == 3
        assert stats["search_fallbacks"] == 2
        assert stats["search_failures"] == 0

    def test_each_game_turn_starts_with_a_clean_ledger(self) -> None:
        policy = CommandPolicy([says(SKIP_1)])
        states = [game_state(5), game_state(5), game_state(6), game_state(over=True)]
        _, bridge, _, _ = play(states, policy, ack_for=refusing(SKIP_1))
        assert bridge.sent == [SKIP_1, END, SKIP_1]


class TestStatistics:
    def test_a_decision_is_counted_each_time_the_policy_is_asked(self) -> None:
        policy = CommandPolicy([says(SKIP_1)])
        loop, _, _, _ = play(turn(count=4), policy)
        stats = loop.stats()
        assert stats["search_decisions"] == 4
        assert stats["search_fallbacks"] == 0
        assert stats["search_failures"] == 0
        assert all(isinstance(stats[k], int) for k in stats if k.startswith("search_"))

    def test_a_select_action_policy_reports_zeros_for_all_three(self) -> None:
        class Plain:
            def __init__(self) -> None:
                self.asked = 0

            async def select_action(self, obs: Any, mask: Any, converted_state: Any = None) -> Any:
                self.asked += 1
                return 1, {}

        plain = Plain()
        loop, bridge, _, _ = play(turn(count=2), plain)
        assert plain.asked == 2, "premise: the old path ran"
        assert bridge.sent == [SKIP_1, SKIP_1]
        stats = loop.stats()
        for key in ("search_decisions", "search_fallbacks", "search_failures"):
            assert stats[key] == 0, key
        assert stats["reselections"] == 0

    def test_the_keys_exist_before_any_decision(self) -> None:
        loop, _, _, _ = play([game_state(over=True)], CommandPolicy([says(END)]))
        stats = loop.stats()
        assert (stats["search_decisions"], stats["search_fallbacks"], stats["search_failures"]) == (
            0,
            0,
            0,
        )

    def test_a_fallback_is_counted_as_a_fallback_and_not_as_a_failure(self) -> None:
        loop, _, _, _ = play(turn(count=2), CommandPolicy([falls_back()]))
        stats = loop.stats()
        assert (stats["search_decisions"], stats["search_fallbacks"], stats["search_failures"]) == (
            2,
            2,
            0,
        )

    def test_a_failure_is_counted_as_a_failure_and_not_as_a_fallback(self) -> None:
        loop, _, _, _ = play(turn(count=2), CommandPolicy([RuntimeError("boom")]))
        stats = loop.stats()
        assert (stats["search_decisions"], stats["search_fallbacks"], stats["search_failures"]) == (
            2,
            0,
            2,
        )


class TestFallbackAndFailure:
    def test_a_fallback_with_nothing_blocking_sends_end_turn(self) -> None:
        policy = CommandPolicy([falls_back()])
        loop, bridge, _, _ = play(turn(), policy)
        assert bridge.sent == [END]
        assert loop.stats()["fallback_end_turns"] == 0, "an ordinary send, not the safe fallback"
        assert policy.select_action_calls == 0

    def test_a_policy_that_raises_is_not_fatal_and_does_not_use_the_safe_end_turn(self) -> None:
        policy = CommandPolicy([RuntimeError("boom")])
        loop, bridge, _, code = play(turn(count=2), policy)
        assert code == BridgeAgentLoop.EXIT_OK
        assert bridge.sent == [END, END], "the loop went on to the next state"
        stats = loop.stats()
        assert stats["fallback_end_turns"] == 0, "_send_safe_end_turn was not used"
        assert stats["commands_sent"] == 2
        assert stats["search_failures"] == 2

    def test_a_raise_after_a_good_decision_keeps_the_loop_running(self) -> None:
        policy = CommandPolicy([says(SKIP_1), RuntimeError("boom"), says(SKIP_2)])
        _, bridge, _, code = play(turn(count=3), policy)
        assert code == BridgeAgentLoop.EXIT_OK
        assert bridge.sent == [SKIP_1, END, SKIP_2]

    def test_a_raise_with_the_end_of_turn_blocked_sends_the_offered_research(self) -> None:
        # the game blocks end_turn; the simulator cannot express research, so the policy raises
        policy = CommandPolicy([says(END), RuntimeError("boom")])
        loop, bridge, _, _ = play(turn(count=2), policy, ack_for=blocking_end_turn, offered=(0, 4))
        assert bridge.sent == [END, RESEARCH], "not a second end_turn"
        stats = loop.stats()
        assert stats["blocked_end_turns"] == 1
        assert stats["search_failures"] == 1
        assert stats["fallback_end_turns"] == 0

    def test_a_fallback_with_the_end_of_turn_blocked_sends_the_offered_research(self) -> None:
        policy = CommandPolicy([says(END), falls_back()])
        loop, bridge, _, _ = play(turn(count=2), policy, ack_for=blocking_end_turn, offered=(0, 4))
        assert bridge.sent == [END, RESEARCH]
        assert loop.stats()["search_fallbacks"] == 1

    def test_the_command_sent_instead_is_one_that_may_still_be_sent(self) -> None:
        """skip 1 was refused earlier this turn, so the first sendable offer is research."""
        policy = CommandPolicy([says(SKIP_1), says(END), falls_back()])

        def ack_for(command: dict[str, Any]) -> Any:
            return refusing(SKIP_1)(command) if command != END else blocking_end_turn(command)

        _, bridge, _, _ = play(turn(count=3), policy, ack_for=ack_for, offered=(0, 1, 4))
        assert bridge.sent == [SKIP_1, END, RESEARCH]

    def test_when_nothing_else_is_offered_a_blocked_turn_still_gets_an_end_turn(self) -> None:
        policy = CommandPolicy([says(END), falls_back()])
        _, bridge, _, _ = play(turn(count=2), policy, ack_for=blocking_end_turn, offered=(0,))
        assert bridge.sent == [END, END]

    def test_the_command_sent_instead_is_never_end_turn_while_something_else_is_sendable(
        self,
    ) -> None:
        policy = CommandPolicy([says(END), RuntimeError("boom")])
        _, bridge, _, _ = play(
            turn(count=2), policy, ack_for=blocking_end_turn, offered=(0, 2, 3, 4)
        )
        assert bridge.sent[0] == END
        assert bridge.sent[1] != END
        assert bridge.sent[1] in (SKIP_2, MOVE_1, RESEARCH)


class TestCapAndStall:
    def test_past_the_cap_only_end_turn_is_sent_and_the_policy_is_not_asked(self) -> None:
        policy = CommandPolicy([says(SKIP_1)])
        loop, bridge, _, _ = play(turn(count=5), policy, max_commands_per_turn=2)
        assert bridge.sent == [SKIP_1, SKIP_1, END, END, END]
        assert len(policy.calls) == 2
        assert loop.stats()["search_decisions"] == 2
        assert policy.select_action_calls == 0

    def test_the_cap_is_per_game_turn(self) -> None:
        policy = CommandPolicy([says(SKIP_1)])
        states = [game_state(5), game_state(5), game_state(5), game_state(6), game_state(over=True)]
        _, bridge, _, _ = play(states, policy, max_commands_per_turn=2)
        assert bridge.sent == [SKIP_1, SKIP_1, END, SKIP_1]

    def test_past_the_cap_with_a_blocked_end_turn_the_policy_may_only_unblock(self) -> None:
        policy = CommandPolicy([says(END), says(MOVE_1)])
        _, bridge, _, _ = play(
            turn(count=2),
            policy,
            ack_for=blocking_end_turn,
            offered=(0, 3, 4),
            max_commands_per_turn=1,
        )
        assert len(policy.calls) == 2, "asked once more: the allowance for unblocking"
        verdicts = policy.calls[1]["verdicts"]
        assert verdicts["move"] is False
        assert verdicts["skip_1"] is True
        assert verdicts["research"] is True
        assert bridge.sent == [END, RESEARCH], "the move it named anyway was not sent"

    def test_a_stall_is_declared_as_before(self) -> None:
        policy = CommandPolicy([says(END)])
        states = [game_state(5) for _ in range(8)] + [game_state(over=True)]
        loop, bridge, _, code = play(states, policy, ack_for=blocking_end_turn, offered=(0,))
        assert code == BridgeAgentLoop.EXIT_STALLED
        assert bridge.sent == [END, END, END]
        stats = loop.stats()
        assert stats["stalled_turns"] == 1
        assert stats["blocked_end_turns"] == 3


class TestDecisionSeed:
    def test_the_base_seed_is_the_policy_s_and_the_command_number_counts_the_turn(self) -> None:
        policy = CommandPolicy([says(SKIP_1)], seed=9)
        states = [game_state(5), game_state(5), game_state(6), game_state(over=True)]
        play(states, policy)
        seeds = [call["decision_seed"] for call in policy.calls]
        assert seeds == [
            derive_seed(9, "bridge_search", 5, 0),
            derive_seed(9, "bridge_search", 5, 1),
            derive_seed(9, "bridge_search", 6, 0),
        ]

    def test_without_a_base_seed_there_is_no_decision_seed(self) -> None:
        policy = CommandPolicy([says(SKIP_1)], seed=None)
        play(turn(count=2), policy)
        assert [call["decision_seed"] for call in policy.calls] == [None, None]


class TestEvalHarness:
    def test_one_report_per_decision_with_float_values(self) -> None:
        policy = CommandPolicy([says(SKIP_1, root_value=0.25), says(SKIP_2, root_value=-1)])
        harness = Harness()
        play(turn(count=2), policy, harness=harness)
        assert [turn_id for turn_id, _ in harness.turn_ends] == [5, 5]
        first, second = (diagnostics for _, diagnostics in harness.turn_ends)
        assert first["value_estimate"] == pytest.approx(0.25)
        assert second["value_estimate"] == pytest.approx(-1.0)
        for diagnostics in (first, second):
            assert diagnostics["policy_kind"] == 1.0
            assert all(type(value) is float for value in diagnostics.values()), diagnostics

    def test_a_record_without_a_root_value_reports_zero(self) -> None:
        harness = Harness()
        play(turn(), CommandPolicy([says(SKIP_1)]), harness=harness)
        ((_, diagnostics),) = harness.turn_ends
        assert diagnostics["value_estimate"] == 0.0
        assert type(diagnostics["value_estimate"]) is float
        assert diagnostics["policy_kind"] == 1.0

    def test_a_select_action_policy_is_not_reported_as_a_search(self) -> None:
        class Plain:
            async def select_action(self, obs: Any, mask: Any, converted_state: Any = None) -> Any:
                return 1, {"value_estimate": 0.5, "policy_kind": 0.0}

        harness = Harness()
        play(turn(), Plain(), harness=harness)
        ((_, diagnostics),) = harness.turn_ends
        assert diagnostics == {"value_estimate": 0.5, "policy_kind": 0.0}
