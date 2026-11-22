"""Tests for what the independent review of GM-B6b found.

1. While the game blocked the end of turn and the search named nothing, the
   loop sent the first offered command by action index, which is a unit action
   nobody chose, not the research that would unblock the turn.
2. ``build_search_policy`` accepted a checkpoint of the full-game network (84
   channels), or one with another action count, and called it trained.
3. A policy whose ``select_command`` is set on the instance was taken for an
   index policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from alpha_zero.bridge.command_policy import is_command_policy
from alpha_zero.bridge.offer_filter import first_to_unblock
from alpha_zero.bridge.policy_factory import build_search_policy
from alpha_zero.core.civ6_network import MicroCivNetwork
from tests.fixtures.bridge_loop_fakes import (
    END,
    MOVE_1,
    RESEARCH,
    SKIP_2,
    blocking_end_turn,
    game_state,
    play,
)

PRODUCE = {"action_type": "city_produce", "city_id": 7, "item_id": "UNIT_WARRIOR"}
CIVIC = {"action_type": "research_civic", "item_id": "CIVIC_CRAFTSMANSHIP"}


class _NamesNothing:
    """A command policy: ends the turn first, then says it has nothing."""

    def __init__(self) -> None:
        self.asked = 0

    async def select_command(
        self, converted_state: Any, *, allowed: Any = None, decision_seed: Any = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.asked += 1
        if self.asked == 1:
            return END, {"fallback": None}
        return END, {"fallback": "nothing translatable"}


class TestFallbackOrder:
    def test_research_goes_before_a_unit_action_nobody_chose(self) -> None:
        states = [game_state(5), game_state(5), game_state(over=True)]
        # Offered: end_turn (0), a move (3), research (4). The move has the lower index.
        _, bridge, _, _ = play(
            states, _NamesNothing(), ack_for=blocking_end_turn, offered=(0, 3, 4)
        )
        assert bridge.sent == [END, RESEARCH]

    def test_without_research_the_move_is_still_better_than_a_refused_end_turn(self) -> None:
        states = [game_state(5), game_state(5), game_state(over=True)]
        _, bridge, _, _ = play(states, _NamesNothing(), ack_for=blocking_end_turn, offered=(0, 3))
        assert bridge.sent == [END, MOVE_1]

    @pytest.mark.parametrize(
        ("commands", "expected"),
        [
            ([MOVE_1, SKIP_2, PRODUCE, RESEARCH], RESEARCH),
            ([MOVE_1, PRODUCE, CIVIC], CIVIC),
            ([MOVE_1, SKIP_2, PRODUCE], SKIP_2),
            ([MOVE_1, PRODUCE], PRODUCE),
            ([END, MOVE_1], MOVE_1),
            ([END], None),
            ([], None),
        ],
    )
    def test_first_to_unblock(self, commands: list[dict[str, Any]], expected: Any) -> None:
        assert first_to_unblock(commands) == expected


class TestFactoryRefusesWhatTheSearchCannotUse:
    def _save(self, path: Path, **shape: Any) -> Path:
        network = MicroCivNetwork(board_size=8, **shape)
        network.save(str(path), metadata={"max_turns": 30, "board_size": 8})
        return path

    def test_the_full_game_network_is_refused(self, tmp_path: Path) -> None:
        path = self._save(
            tmp_path / "full.pt", spatial_channels=84, vector_dim=512, action_space_size=88562
        )
        with pytest.raises(ValueError, match="84 channels"):
            build_search_policy(path)

    def test_another_action_count_is_refused(self, tmp_path: Path) -> None:
        path = self._save(tmp_path / "odd.pt", action_space_size=500)
        with pytest.raises(ValueError, match="500 actions"):
            build_search_policy(path)

    def test_control_a_simulator_network_is_accepted(self, tmp_path: Path) -> None:
        untrained = build_search_policy(allow_untrained=True, board_size=8, max_turns=30)
        actions = untrained.policy.network.config.action_space_size
        path = self._save(tmp_path / "sim.pt", action_space_size=actions)
        built = build_search_policy(path)
        assert built.trained and built.board_size == 8 and built.max_turns == 30

    @pytest.mark.parametrize(("board", "turns"), [(0, 30), (8, 0), (-1, 30)])
    def test_untrained_needs_positive_sizes(self, board: int, turns: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            build_search_policy(allow_untrained=True, board_size=board, max_turns=turns)


class TestIsCommandPolicy:
    def test_a_coroutine_set_on_the_instance_counts(self) -> None:
        async def select_command(converted_state: Any, **_: Any) -> tuple[dict, dict]:
            return END, {}

        policy = type("Plain", (), {})()
        policy.select_command = select_command
        assert is_command_policy(policy)

    def test_mocks_are_still_not_command_policies(self) -> None:
        assert not is_command_policy(MagicMock())
        assert not is_command_policy(AsyncMock())

    def test_a_plain_function_on_the_instance_does_not_count(self) -> None:
        policy = type("Plain", (), {})()
        policy.select_command = lambda *_a, **_k: (END, {})
        assert not is_command_policy(policy)
