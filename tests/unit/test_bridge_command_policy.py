"""GM-B6b (A): ``alpha_zero.bridge.command_policy``, every rule, with a real ``TurnLedger``.

The policies here are small fakes: the policy is the collaborator, ``choose`` is the unit.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from alpha_zero.bridge.command_policy import (
    Choice,
    allowed_by,
    choose,
    decision_seed,
    is_command_policy,
)
from alpha_zero.bridge.offer_filter import vetoed
from alpha_zero.bridge.protocol import LuaCommand
from alpha_zero.bridge.turn_ledger import TurnLedger
from alpha_zero.utils.seeding import derive_seed
from tests.fixtures.ledger_states import (
    END_TURN,
    MOVE_101,
    MOVE_102,
    PRODUCE,
    UPGRADE,
    blocked,
    exchange,
    ok,
    refused,
    state,
)

SKIP_101 = {"action_type": "unit_skip", "unit_id": 101}
RESEARCH = {"action_type": "research_tech", "item_id": "TECH_WRITING"}
PROBES = [END_TURN, MOVE_101, MOVE_102, PRODUCE, UPGRADE, SKIP_101, RESEARCH]


def fresh() -> TurnLedger:
    ledger = TurnLedger()
    ledger.observe(state())
    return ledger


def with_refused() -> TurnLedger:
    ledger = fresh()
    exchange(ledger, MOVE_101, state(), refused())
    return ledger


def with_chosen() -> TurnLedger:
    ledger = fresh()
    exchange(ledger, PRODUCE, state(), ok())
    return ledger


def with_withheld_end_turn() -> TurnLedger:
    ledger = fresh()
    exchange(ledger, END_TURN, state(), blocked())
    return ledger


LEDGERS = {
    "fresh": (fresh, set()),
    "refused": (with_refused, {"move_101"}),
    "once_per_turn": (with_chosen, {"produce"}),
    "withheld_end_turn": (with_withheld_end_turn, {"end_turn"}),
}
NAMES = {
    "end_turn": END_TURN,
    "move_101": MOVE_101,
    "produce": PRODUCE,
}


class Policy:
    """Answers with ``answer`` (a ``(command, record)`` pair) or raises it; records its calls."""

    def __init__(self, answer: Any, *, seed: Any = "none") -> None:
        self.answer = answer
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        if seed != "none":
            self.config = SimpleNamespace(seed=seed)

    async def select_command(
        self, converted_state: Any, *, allowed: Any = None, decision_seed: Any = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append((converted_state, {"allowed": allowed, "decision_seed": decision_seed}))
        if isinstance(self.answer, BaseException):
            raise self.answer
        command, record = self.answer
        return dict(command), dict(record)


def run_choose(
    policy: Any, ledger: TurnLedger, *, unblocking_only: bool = False, turn_id: Any = 12
) -> Choice:
    return asyncio.run(
        choose(
            policy, ledger, {"converted": True}, unblocking_only=unblocking_only, turn_id=turn_id
        )
    )


class TestIsCommandPolicy:
    def test_an_object_with_a_callable_select_command(self) -> None:
        assert is_command_policy(Policy((END_TURN, {}))) is True

    def test_a_select_action_policy_is_not_one(self) -> None:
        class Plain:
            async def select_action(self, *_: Any) -> Any:
                return 0, {}

        assert is_command_policy(Plain()) is False

    def test_a_non_callable_attribute_does_not_count(self) -> None:
        assert is_command_policy(SimpleNamespace(select_command="no")) is False

    def test_something_else_entirely_is_not_one(self) -> None:
        assert is_command_policy(None) is False
        assert is_command_policy(object()) is False


class TestAllowedBy:
    @pytest.mark.parametrize("unblocking_only", [False, True])
    @pytest.mark.parametrize("name", sorted(LEDGERS))
    def test_it_is_not_vetoed_over_every_probe(self, name: str, unblocking_only: bool) -> None:
        build, _ = LEDGERS[name]
        ledger = build()
        allowed = allowed_by(ledger, unblocking_only)
        for command in PROBES:
            expected = not vetoed(ledger, command, unblocking_only, others=True)
            assert allowed(command) is expected, (name, unblocking_only, command)

    @pytest.mark.parametrize("name", sorted(LEDGERS))
    def test_the_ledger_cases_really_rule_something_out(self, name: str) -> None:
        """Without this, the loop above passes for an ``allowed_by`` that allows everything."""
        build, ruled_out = LEDGERS[name]
        allowed = allowed_by(build(), False)
        assert {key for key, command in NAMES.items() if not allowed(command)} == ruled_out

    def test_a_refused_command_is_ruled_out_and_a_sibling_is_not(self) -> None:
        allowed = allowed_by(with_refused(), False)
        assert allowed(MOVE_101) is False
        assert allowed(MOVE_102) is True

    def test_a_choice_made_this_turn_is_ruled_out(self) -> None:
        allowed = allowed_by(with_chosen(), False)
        assert allowed(PRODUCE) is False
        assert allowed(MOVE_101) is True

    def test_end_turn_is_ruled_out_even_alone_while_the_game_blocks_it(self) -> None:
        """``others=True``: the policy is told what it may not name, whatever else is on offer."""
        assert allowed_by(with_withheld_end_turn(), False)(END_TURN) is False
        assert allowed_by(fresh(), False)(END_TURN) is True

    def test_unblocking_only_keeps_only_what_can_answer_a_blocker(self) -> None:
        allowed = allowed_by(fresh(), True)
        assert allowed(MOVE_101) is False
        assert allowed(UPGRADE) is False
        assert allowed(SKIP_101) is True
        assert allowed(RESEARCH) is True
        assert allowed(PRODUCE) is True
        assert allowed(END_TURN) is True

    def test_the_ledger_is_read_when_asked_not_when_built(self) -> None:
        ledger = fresh()
        allowed = allowed_by(ledger, False)
        assert allowed(MOVE_101) is True
        exchange(ledger, MOVE_101, state(), refused())
        assert allowed(MOVE_101) is False


class TestDecisionSeed:
    def test_none_stays_none(self) -> None:
        assert decision_seed(None, 12, 0) is None
        assert decision_seed(None, 13, 5) is None

    def test_an_int_that_is_the_same_for_the_same_inputs(self) -> None:
        first = decision_seed(7, 12, 3)
        assert isinstance(first, int)
        assert decision_seed(7, 12, 3) == first

    def test_it_differs_with_the_turn_the_command_number_and_the_base(self) -> None:
        base = decision_seed(7, 12, 3)
        assert decision_seed(7, 13, 3) != base
        assert decision_seed(7, 12, 4) != base
        assert decision_seed(8, 12, 3) != base

    def test_it_is_the_named_stream_of_the_run_seed(self) -> None:
        assert decision_seed(7, 12, 3) == derive_seed(7, "bridge_search", 12, 3)

    def test_a_run_gets_distinct_seeds_over_a_few_turns_and_commands(self) -> None:
        seeds = {decision_seed(7, turn, n) for turn in range(10, 16) for n in range(8)}
        assert len(seeds) == 48


class TestChooseOutcomes:
    def test_a_command_the_game_may_take(self) -> None:
        record = {"fallback": None, "root_value": 0.25}
        choice = run_choose(Policy((MOVE_101, record)), fresh())
        assert isinstance(choice, Choice)
        assert isinstance(choice.command, LuaCommand)
        assert choice.command.to_dict() == MOVE_101
        assert choice.record == record
        assert choice.failed is False
        assert choice.reason is None

    def test_the_policy_raises(self) -> None:
        policy = Policy(RuntimeError("the search blew up"))
        choice = run_choose(policy, fresh())
        assert len(policy.calls) == 1, "premise: it was asked"
        assert choice.command is None
        assert choice.record == {}
        assert choice.failed is True
        assert "the search blew up" in str(choice.reason)

    def test_any_exception_type_is_caught(self) -> None:
        choice = run_choose(Policy(ValueError("bad")), fresh())
        assert choice.failed is True
        assert choice.command is None

    def test_a_fallback_names_nothing_and_says_why(self) -> None:
        record = {"fallback": "nothing the search could choose translates", "root_actions": []}
        choice = run_choose(Policy((END_TURN, record)), fresh())
        assert choice.command is None
        assert choice.failed is False
        assert choice.reason == "nothing the search could choose translates"
        assert choice.record == record

    def test_a_falsy_fallback_is_not_a_fallback(self) -> None:
        for falsy in (None, "", 0):
            choice = run_choose(Policy((END_TURN, {"fallback": falsy})), fresh())
            assert choice.command is not None, falsy
            assert choice.command.to_dict() == END_TURN
            assert choice.reason is None

    def test_a_dict_that_is_not_a_command(self) -> None:
        for bad in ({"action_type": "no_such_action"}, {"unit_id": 3}):
            choice = run_choose(Policy((bad, {"fallback": None})), fresh())
            assert choice.command is None, bad
            assert choice.failed is False
            assert isinstance(choice.reason, str) and choice.reason
            assert choice.record == {"fallback": None}

    def test_a_command_the_ledger_ruled_out_is_not_handed_on(self) -> None:
        policy = Policy((MOVE_101, {"fallback": None}))
        choice = run_choose(policy, with_refused())
        assert len(policy.calls) == 1
        assert choice.command is None
        assert choice.failed is False
        assert isinstance(choice.reason, str) and choice.reason
        assert choice.record == {"fallback": None}

    def test_a_choice_already_made_this_turn_is_not_handed_on_either(self) -> None:
        choice = run_choose(Policy((PRODUCE, {"fallback": None})), with_chosen())
        assert choice.command is None and choice.failed is False

    def test_end_turn_while_the_game_blocks_it_is_ruled_out(self) -> None:
        choice = run_choose(Policy((END_TURN, {"fallback": None})), with_withheld_end_turn())
        assert choice.command is None and choice.failed is False
        assert isinstance(choice.reason, str) and choice.reason

    def test_past_the_cap_a_command_that_cannot_unblock_is_ruled_out(self) -> None:
        policy = Policy((MOVE_101, {"fallback": None}))
        assert run_choose(policy, fresh(), unblocking_only=True).command is None
        control = run_choose(Policy((SKIP_101, {"fallback": None})), fresh(), unblocking_only=True)
        assert control.command is not None and control.command.to_dict() == SKIP_101

    def test_the_same_move_is_sent_when_nothing_rules_it_out(self) -> None:
        """Control for the ledger cases above: the policy and the command are the same."""
        choice = run_choose(Policy((MOVE_101, {"fallback": None})), fresh())
        assert choice.command is not None


class TestChooseArguments:
    def test_the_policy_gets_the_converted_state_itself(self) -> None:
        policy = Policy((END_TURN, {"fallback": None}))
        converted = {"the": "state"}
        asyncio.run(choose(policy, fresh(), converted, unblocking_only=False, turn_id=12))
        assert policy.calls[0][0] is converted

    def test_allowed_is_the_ledger_filter(self) -> None:
        ledger = with_refused()
        policy = Policy((END_TURN, {"fallback": None}))
        run_choose(policy, ledger, unblocking_only=False)
        allowed = policy.calls[0][1]["allowed"]
        expected = allowed_by(ledger, False)
        assert allowed(MOVE_101) is False
        for command in PROBES:
            assert allowed(command) is expected(command), command

    def test_allowed_carries_the_unblocking_only_flag(self) -> None:
        policy = Policy((END_TURN, {"fallback": None}))
        run_choose(policy, fresh(), unblocking_only=True)
        allowed = policy.calls[0][1]["allowed"]
        assert allowed(MOVE_101) is False
        assert allowed(SKIP_101) is True

    def test_the_seed_comes_from_the_policy_config_the_turn_and_the_command_number(self) -> None:
        ledger = fresh()
        policy = Policy((END_TURN, {"fallback": None}), seed=11)
        run_choose(policy, ledger, turn_id=12)
        ledger.sent(MOVE_101, state())
        ledger.sent(MOVE_102, state())
        assert ledger.commands_this_turn == 2, "premise"
        run_choose(policy, ledger, turn_id=12)
        run_choose(policy, ledger, turn_id=13)
        seeds = [call[1]["decision_seed"] for call in policy.calls]
        assert seeds == [
            decision_seed(11, 12, 0),
            decision_seed(11, 12, 2),
            decision_seed(11, 13, 2),
        ]
        assert len(set(seeds)) == 3

    def test_no_base_seed_means_no_decision_seed(self) -> None:
        for policy in (
            Policy((END_TURN, {"fallback": None}), seed=None),  # a config with seed None
            Policy((END_TURN, {"fallback": None})),  # no config at all
        ):
            run_choose(policy, fresh())
            assert policy.calls[0][1]["decision_seed"] is None

    def test_a_config_without_a_seed_field_is_a_policy_without_one(self) -> None:
        policy = Policy((END_TURN, {"fallback": None}))
        policy.config = SimpleNamespace()
        run_choose(policy, fresh())
        assert policy.calls[0][1]["decision_seed"] is None

    def test_choose_asks_exactly_once(self) -> None:
        policy = Policy((END_TURN, {"fallback": None}))
        run_choose(policy, fresh())
        run_choose(Policy(RuntimeError("x")), fresh())
        assert len(policy.calls) == 1
