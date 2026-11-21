"""GM-B7c-2 (C): the agent loop with fakes: re-selection, the command cap, the new config.

A choice that cannot be sent is not turned into ``end_turn`` any more: the action is masked
out and the policy is asked again. The fakes here are small classes (not mocks) so a test can
read what the bridge was really given.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from alpha_zero.bridge import agent_loop, agent_loop_config
from alpha_zero.bridge.action_converter import InvalidActionError
from alpha_zero.bridge.agent_loop import ENV_PREFIX, AgentLoopConfig, BridgeAgentLoop
from alpha_zero.bridge.protocol import AckMessage, LuaCommand

SIZE = 8
END = {"action_type": "end_turn"}
SKIP_1 = {"action_type": "unit_skip", "unit_id": 1}
SKIP_2 = {"action_type": "unit_skip", "unit_id": 2}
SKIP_3 = {"action_type": "unit_skip", "unit_id": 3}
#: What each fake action index means. Index 7 and above are not in the table: invalid.
TABLE = {0: END, 1: SKIP_1, 2: SKIP_2, 3: SKIP_3}


def game_state(turn: int = 5, *, over: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        turn_id=turn,
        player_id=0,
        is_game_over=over,
        winner_id=None,
        legal_actions=[],
        end_turn_blocking=None,
    )


class FakeBridge:
    """Hands out scripted states, records every command, answers by ``ack_for``."""

    def __init__(
        self,
        states: list[SimpleNamespace],
        ack_for: Callable[[dict[str, Any]], AckMessage] | None = None,
    ) -> None:
        self._states = list(states)
        self._ack_for = ack_for or (lambda _command: AckMessage(msg_id=1, success=True))
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

    async def send_and_wait_with_retry(self, command: Any, *_: Any, **__: Any) -> AckMessage:
        return self._ack_for(self._record(command))


class FakeStateConverter:
    def convert(self, state: Any) -> dict[str, Any]:
        return {
            "_legal_actions": [],
            "_obs": {
                "spatial": np.zeros((13, SIZE, SIZE), dtype=np.float32),
                "vector": np.zeros(64, dtype=np.float32),
            },
        }


class FakeActionConverter:
    """Index -> ``TABLE``; any other index is an ``InvalidActionError``."""

    def generate_action_mask(self, legal_actions: Any, converted_state: Any) -> np.ndarray:
        return np.ones(SIZE, dtype=np.float32)

    def convert(
        self, idx: int, state: Any, legal_actions: Any = None, validate: bool = True
    ) -> Any:
        if idx not in TABLE:
            raise InvalidActionError(f"action {idx} is not offered")
        return LuaCommand.from_dict(TABLE[idx])


class ScriptedPolicy:
    """Picks, from ``preference``, the first index the mask still allows; records each call."""

    def __init__(self, preference: list[int]) -> None:
        self.preference = preference
        self.masks: list[np.ndarray] = []

    async def select_action(self, obs: Any, mask: np.ndarray, converted_state: Any = None) -> Any:
        self.masks.append(mask.copy())
        for idx in self.preference:
            if mask[idx] > 0:
                return idx, {}
        raise AssertionError(f"the policy was asked with nothing it wants: {mask}")


def play(
    states: list[SimpleNamespace],
    policy: Any,
    *,
    ack_for: Callable[[dict[str, Any]], AckMessage] | None = None,
    **config: Any,
) -> tuple[BridgeAgentLoop, FakeBridge, int]:
    import asyncio

    bridge = FakeBridge(states, ack_for)
    config.setdefault("max_turns", 0)
    loop = BridgeAgentLoop(
        bridge=bridge,  # type: ignore[arg-type]
        policy=policy,
        state_converter=FakeStateConverter(),  # type: ignore[arg-type]
        action_converter=FakeActionConverter(),  # type: ignore[arg-type]
        config=AgentLoopConfig(per_turn_timeout_s=0.5, **config),
    )
    code = asyncio.run(loop.run())
    return loop, bridge, code


def refuse_command(target: dict[str, Any]) -> Callable[[dict[str, Any]], AckMessage]:
    def ack_for(command: dict[str, Any]) -> AckMessage:
        if command == target:
            executed = [{"success": False, "error_code": "OPERATION_REFUSED", "message": "no"}]
            return AckMessage(
                msg_id=1, success=False, error_code="COMMAND_FAILED", executed=executed
            )
        return AckMessage(msg_id=1, success=True)

    return ack_for


# ---------------------------------------------------------------------------
# An invalid choice is chosen again, not turned into end_turn
# ---------------------------------------------------------------------------
class TestInvalidChoiceIsReselected:
    def test_the_policy_is_asked_again_with_that_action_masked_out(self) -> None:
        policy = ScriptedPolicy([6, 1, 0])  # 6 is invalid, then a real skip
        _, bridge, _ = play([game_state(), game_state(over=True)], policy)
        assert bridge.sent == [SKIP_1], "the invalid choice must not become an end_turn"
        assert len(policy.masks) == 2
        assert policy.masks[0][6] == 1.0
        assert policy.masks[1][6] == 0.0
        assert policy.masks[1][1] == 1.0, "only the rejected action is masked"

    def test_it_is_counted(self) -> None:
        policy = ScriptedPolicy([6, 5, 1])
        loop, bridge, _ = play([game_state(), game_state(over=True)], policy)
        assert bridge.sent == [SKIP_1]
        stats = loop.stats()
        assert stats["reselections"] == 2
        assert stats["invalid_actions"] == 2
        assert stats["commands_sent"] == 1

    def test_control_a_valid_first_choice_is_not_a_reselection(self) -> None:
        policy = ScriptedPolicy([1])
        loop, bridge, _ = play([game_state(), game_state(over=True)], policy)
        assert bridge.sent == [SKIP_1]
        assert loop.stats()["reselections"] == 0
        assert len(policy.masks) == 1

    def test_the_mask_of_the_next_state_is_fresh(self) -> None:
        policy = ScriptedPolicy([6, 1])
        _, bridge, _ = play([game_state(), game_state(), game_state(over=True)], policy)
        assert bridge.sent == [SKIP_1, SKIP_1]
        assert policy.masks[2][6] == 1.0, "a masked-out action is masked for one step only"

    def test_after_the_reselections_are_used_up_end_turn_is_sent(self) -> None:
        policy = ScriptedPolicy([4, 5, 6, 0])  # three invalid choices: 1 + 2 re-selections
        loop, bridge, _ = play([game_state(), game_state(over=True)], policy, max_reselections=2)
        assert bridge.sent == [END]
        assert len(policy.masks) == 3
        assert loop.stats()["reselections"] == 2

    def test_zero_reselections_sends_end_turn_at_the_first_invalid_choice(self) -> None:
        policy = ScriptedPolicy([4, 1])
        loop, bridge, _ = play([game_state(), game_state(over=True)], policy, max_reselections=0)
        assert bridge.sent == [END]
        assert len(policy.masks) == 1
        assert loop.stats()["reselections"] == 0

    def test_nothing_else_in_the_mask_sends_end_turn_without_asking_again(self) -> None:
        class OnlyInvalid(FakeActionConverter):
            def generate_action_mask(self, legal_actions: Any, converted_state: Any) -> Any:
                mask = np.zeros(SIZE, dtype=np.float32)
                mask[6] = 1.0
                return mask

        import asyncio

        bridge = FakeBridge([game_state(), game_state(over=True)])
        policy = ScriptedPolicy([6])
        loop = BridgeAgentLoop(
            bridge=bridge,  # type: ignore[arg-type]
            policy=policy,
            state_converter=FakeStateConverter(),  # type: ignore[arg-type]
            action_converter=OnlyInvalid(),  # type: ignore[arg-type]
            config=AgentLoopConfig(max_turns=0, per_turn_timeout_s=0.5),
        )
        asyncio.run(loop.run())
        assert bridge.sent == [END]
        assert len(policy.masks) == 1


# ---------------------------------------------------------------------------
# Refused and withheld choices are chosen again too
# ---------------------------------------------------------------------------
class TestRefusedAndBlockedChoices:
    def test_a_command_the_mod_refused_is_not_sent_again_this_turn(self) -> None:
        policy = ScriptedPolicy([1, 2, 0])
        states = [game_state(), game_state(), game_state(), game_state(over=True)]
        loop, bridge, _ = play(states, policy, ack_for=refuse_command(SKIP_1))
        assert bridge.sent.count(SKIP_1) == 1
        assert bridge.sent[:2] == [SKIP_1, SKIP_2]
        assert loop.stats()["refused_commands"] == 1

    def test_the_refusal_is_forgotten_on_the_next_game_turn(self) -> None:
        policy = ScriptedPolicy([1, 0])
        states = [game_state(5), game_state(6), game_state(over=True)]
        _, bridge, _ = play(states, policy, ack_for=refuse_command(SKIP_1))
        assert bridge.sent == [SKIP_1, SKIP_1]

    def test_end_turn_is_chosen_again_while_the_game_blocks_it(self) -> None:
        def ack_for(command: dict[str, Any]) -> AckMessage:
            if command == END:
                executed = [{"success": False, "error_code": "END_TURN_BLOCKED", "message": "B"}]
                return AckMessage(
                    msg_id=1, success=False, error_code="COMMAND_FAILED", executed=executed
                )
            return AckMessage(msg_id=1, success=True)

        policy = ScriptedPolicy([0, 1])  # always prefers end_turn
        states = [game_state(), game_state(), game_state(over=True)]
        loop, bridge, _ = play(states, policy, ack_for=ack_for)
        assert bridge.sent == [END, SKIP_1], "the second end_turn was chosen again, not sent"
        assert loop.stats()["blocked_end_turns"] == 1

    def test_end_turn_alone_in_the_mask_is_sent_even_when_blocked(self) -> None:
        def ack_for(command: dict[str, Any]) -> AckMessage:
            executed = [{"success": False, "error_code": "END_TURN_BLOCKED", "message": "B"}]
            return AckMessage(
                msg_id=1, success=False, error_code="COMMAND_FAILED", executed=executed
            )

        class EndOnly(FakeActionConverter):
            def generate_action_mask(self, legal_actions: Any, converted_state: Any) -> Any:
                mask = np.zeros(SIZE, dtype=np.float32)
                mask[0] = 1.0
                return mask

        import asyncio

        bridge = FakeBridge([game_state(), game_state(), game_state(over=True)], ack_for)
        loop = BridgeAgentLoop(
            bridge=bridge,  # type: ignore[arg-type]
            policy=ScriptedPolicy([0]),
            state_converter=FakeStateConverter(),  # type: ignore[arg-type]
            action_converter=EndOnly(),  # type: ignore[arg-type]
            config=AgentLoopConfig(max_turns=0, per_turn_timeout_s=0.5),
        )
        asyncio.run(loop.run())
        assert bridge.sent == [END, END]


# ---------------------------------------------------------------------------
# The cap on commands per game turn
# ---------------------------------------------------------------------------
class TestCommandCap:
    def test_past_the_cap_only_end_turn_is_sent_and_the_policy_is_not_asked(self) -> None:
        policy = ScriptedPolicy([1])
        states = [game_state() for _ in range(5)] + [game_state(over=True)]
        _, bridge, _ = play(states, policy, max_commands_per_turn=2)
        assert bridge.sent == [SKIP_1, SKIP_1, END, END, END]
        assert len(policy.masks) == 2

    def test_the_cap_is_per_game_turn(self) -> None:
        policy = ScriptedPolicy([1])
        states = [game_state(5), game_state(5), game_state(5), game_state(6), game_state(over=True)]
        _, bridge, _ = play(states, policy, max_commands_per_turn=2)
        assert bridge.sent == [SKIP_1, SKIP_1, END, SKIP_1]

    def test_the_commands_up_to_the_cap_are_not_replaced(self) -> None:
        policy = ScriptedPolicy([1])
        states = [game_state() for _ in range(3)] + [game_state(over=True)]
        _, bridge, _ = play(states, policy)  # default cap 100
        assert bridge.sent == [SKIP_1] * 3


# ---------------------------------------------------------------------------
# The fallback end_turn is accounted for
# ---------------------------------------------------------------------------
class TestFallbackEndTurn:
    def test_a_failing_policy_sends_a_fallback_that_counts_as_a_command(self) -> None:
        class Broken:
            async def select_action(self, *_: Any, **__: Any) -> Any:
                raise RuntimeError("boom")

        loop, bridge, _ = play([game_state(), game_state(over=True)], Broken())
        assert bridge.sent == [END]
        stats = loop.stats()
        assert stats["fallback_end_turns"] == 1
        assert stats["commands_sent"] == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
class TestStats:
    def test_the_new_keys_are_there_and_the_old_ones_stay(self) -> None:
        loop, _, _ = play([game_state(over=True)], ScriptedPolicy([0]))
        stats = loop.stats()
        for key in (
            "game_turns_played",
            "commands_sent",
            "refused_commands",
            "no_effect_commands",
            "reselections",
            "stalled_turns",
            "turns_played",
            "invalid_actions",
            "blocked_end_turns",
            "fallback_end_turns",
        ):
            assert key in stats, key
        assert all(stats[k] == 0 for k in ("commands_sent", "reselections", "stalled_turns"))

    def test_turns_played_counts_loop_steps_and_game_turns_played_counts_game_turns(self) -> None:
        states = [game_state(5), game_state(5), game_state(6), game_state(over=True)]
        loop, _, _ = play(states, ScriptedPolicy([1]))
        stats = loop.stats()
        assert stats["turns_played"] == 3
        assert stats["game_turns_played"] == 1
        assert stats["commands_sent"] == 3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_the_defaults(self) -> None:
        cfg = AgentLoopConfig()
        assert cfg.max_game_turns == 0
        assert cfg.max_commands_per_turn == 100
        assert cfg.max_reselections == 8
        assert cfg.on_stall == "exit"
        assert cfg.stall_wait_s == 300.0
        assert cfg.max_turns == 500

    def test_the_config_lives_in_its_own_module_and_is_re_exported(self) -> None:
        assert agent_loop.AgentLoopConfig is agent_loop_config.AgentLoopConfig
        assert agent_loop.ENV_PREFIX == agent_loop_config.ENV_PREFIX == "CIV6AZ_AGENT_"

    def test_the_exit_code_for_a_stall_is_three(self) -> None:
        assert BridgeAgentLoop.EXIT_STALLED == 3

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_game_turns": -1}, "max_game_turns"),
            ({"max_commands_per_turn": 0}, "max_commands_per_turn"),
            ({"max_commands_per_turn": -3}, "max_commands_per_turn"),
            ({"max_reselections": -1}, "max_reselections"),
            ({"on_stall": "retry"}, "on_stall"),
            ({"stall_wait_s": 0}, "stall_wait_s"),
            ({"stall_wait_s": -1.0}, "stall_wait_s"),
        ],
    )
    def test_invalid_values_raise(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            AgentLoopConfig(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_game_turns": 0},
            {"max_commands_per_turn": 1},
            {"max_reselections": 0},
            {"on_stall": "wait"},
            {"stall_wait_s": 0.1},
        ],
    )
    def test_the_boundary_values_are_accepted(self, kwargs: dict[str, Any]) -> None:
        AgentLoopConfig(**kwargs)

    def test_env_parsing_of_the_five_fields(self) -> None:
        cfg = AgentLoopConfig.from_env(
            env={
                f"{ENV_PREFIX}MAX_GAME_TURNS": "7",
                f"{ENV_PREFIX}MAX_COMMANDS_PER_TURN": "30",
                f"{ENV_PREFIX}MAX_RESELECTIONS": "2",
                f"{ENV_PREFIX}ON_STALL": "wait",
                f"{ENV_PREFIX}STALL_WAIT_S": "12.5",
            }
        )
        assert cfg.max_game_turns == 7
        assert cfg.max_commands_per_turn == 30
        assert cfg.max_reselections == 2
        assert cfg.on_stall == "wait"
        assert cfg.stall_wait_s == 12.5

    def test_env_values_are_validated(self) -> None:
        with pytest.raises(ValueError, match="max_commands_per_turn"):
            AgentLoopConfig.from_env(env={f"{ENV_PREFIX}MAX_COMMANDS_PER_TURN": "0"})
        with pytest.raises(ValueError, match="on_stall"):
            AgentLoopConfig.from_env(env={f"{ENV_PREFIX}ON_STALL": "hang"})

    def test_an_empty_environment_changes_nothing(self) -> None:
        assert AgentLoopConfig.from_env(env={}) == AgentLoopConfig()

    def test_on_stall_is_case_insensitive_like_on_disconnect(self) -> None:
        cfg = AgentLoopConfig.from_env(env={f"{ENV_PREFIX}ON_STALL": "WAIT"})
        assert cfg.on_stall == "wait"


# ---------------------------------------------------------------------------
# The game-turn limit with fakes
# ---------------------------------------------------------------------------
class TestMaxGameTurns:
    def test_the_loop_stops_before_acting_on_the_state_that_begins_the_next_turn(self) -> None:
        states = [game_state(5), game_state(5), game_state(6), game_state(6)]
        loop, bridge, code = play(states, ScriptedPolicy([1]), max_game_turns=1)
        assert code == BridgeAgentLoop.EXIT_OK
        assert bridge.sent == [SKIP_1, SKIP_1]
        assert loop.stats()["game_turns_played"] == 1
        assert len(bridge._states) == 1, "nothing was read past the first state of turn 6"

    def test_control_without_the_limit_the_loop_acts_on_the_next_turn(self) -> None:
        states = [game_state(5), game_state(6), game_state(over=True)]
        _, bridge, _ = play(states, ScriptedPolicy([1]))
        assert len(bridge.sent) == 2
