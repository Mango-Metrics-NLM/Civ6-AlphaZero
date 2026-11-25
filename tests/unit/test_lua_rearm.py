"""The kick re-arms the mod and never ends the turn: the mod side (contract PR 1).

``ExposedMembers.AlphaZeroRearm`` is written by ``SocketTunerBridge.request_rearm`` (the Lua
string is the module constant ``REARM_LUA`` with a ``{nonce}`` placeholder; these tests run that
exact text). ``PollCommands`` reads it before its early return.

"Turn active" is not something the mod asks today (it learns turns from events). The stand-in
has no ``Player:IsTurnActive``, so ``TURN_API`` below adds one the tests steer with
``Civ6Stub.activePlayer``; the contract's implementation is assumed to ask
``Players[Game.GetLocalPlayer()]:IsTurnActive()``. Time is the harness clock (``c.at``).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import gameinfo_snapshot as g
from tests.fixtures import lua_commands as c

TURN_API = """
Civ6Stub.activePlayer = 0
for id, player in pairs(Players) do
    player.IsTurnActive = function(self) return Civ6Stub.activePlayer == id end
end
"""
NONCE = "nonce-7f3a"


def rearm_lua(nonce: str = NONCE) -> str:
    from alpha_zero.bridge.tuner_socket_bridge import REARM_LUA

    assert "{nonce}" in REARM_LUA
    return REARM_LUA.replace("{nonce}", nonce)


def new_game(active: int | None = 0) -> Any:
    """The mod on the AI player's turn; ``active`` is whose turn the game says it is."""
    game = c.loaded_turn(TURN_API)
    game.execute(f"Civ6Stub.activePlayer = {'nil' if active is None else active}")
    return game


def lapse(game: Any) -> None:
    """Let the command window run out (31 s after the turn began)."""
    c.at(game, 31)
    c.tick(game)
    assert status(game)["waitingForCommand"] is False


def status(game: Any) -> dict[str, Any]:
    return dict(game.eval("AlphaZeroBridge.GetStatus()").items())


def flag(game: Any) -> Any:
    return game.eval("ExposedMembers.AlphaZeroRearm")


def log_text(game: Any) -> str:
    return "\n".join(g.printed(game))


def ui_requests(game: Any) -> int:
    return int(
        game.eval(
            "(function() local n = 0 for _, k in ipairs(Civ6Stub.calls) do"
            " if k.name == 'UI.RequestAction' then n = n + 1 end end return n end)()"
        )
    )


class TestAccepted:
    def test_a_lapsed_window_is_reopened_and_the_state_exported_once(self) -> None:
        game = new_game()
        lapse(game)
        before = len(c.states(game))
        game.execute(rearm_lua())
        assert flag(game) == NONCE
        c.tick(game)
        assert len(c.states(game)) == before + 1
        assert status(game)["waitingForCommand"] is True
        assert status(game)["isMyTurn"] is True
        assert flag(game) is None
        assert f"Re-arm accepted {NONCE}" in log_text(game)
        assert f"AZ_REARM:{NONCE}:accepted" in g.printed(game)
        c.tick(game, 3)
        assert len(c.states(game)) == before + 1, "one export per accepted re-arm"

    def test_it_never_asks_the_game_to_end_the_turn(self) -> None:
        game = new_game()
        lapse(game)
        game.execute(rearm_lua())
        c.tick(game, 4)
        assert ui_requests(game) == 0

    def test_the_window_restarts_from_the_re_arm(self) -> None:
        game = new_game()
        lapse(game)  # clock 1031
        game.execute(rearm_lua())
        c.tick(game)
        c.at(game, 55)  # 24 s after the re-arm: still open
        c.tick(game)
        assert status(game)["waitingForCommand"] is True
        c.at(game, 63)  # 32 s after it: lapsed again
        c.tick(game)
        assert status(game)["waitingForCommand"] is False

    def test_a_command_sent_after_the_re_arm_is_carried_out(self) -> None:
        game = new_game()
        lapse(game)
        game.execute(rearm_lua())
        c.tick(game, 3)
        c.send(game, [c.unit_move(102, 0, 1)])
        c.tick(game, 5)
        assert [a["msg_id"] for a in c.acks(game)] == [1]


def ignored_line(reason: str, nonce: str = NONCE) -> str:
    return f"Re-arm ignored {nonce}: {reason}"


def assert_ignored(game: Any, reason: str, nonce: str = NONCE) -> None:
    assert ignored_line(reason, nonce) in log_text(game)
    assert f"AZ_REARM:{nonce}:ignored:{reason}" in g.printed(game)
    assert "Re-arm accepted" not in log_text(game)


class TestIgnored:
    # Review M2: the full reason is asserted; ``activePlayer = 1`` (our player is the local
    # player, the game says another player's turn is active) is "turn not active".
    @pytest.mark.parametrize(
        ("setup", "reason"),
        [
            ("Civ6Stub.activePlayer = 1", "turn not active"),
            ("Civ6Stub.localPlayerId = 1", "not our turn"),
            ("Civ6Stub.localPlayerId = 1; Civ6Stub.activePlayer = 1", "not our turn"),
            ("Civ6Stub.activePlayer = nil", "turn not active"),
            ("for _, p in pairs(Players) do p.IsTurnActive = nil end", "cannot ask"),
            (
                "for _, p in pairs(Players) do p.IsTurnActive = function() error('boom') end end",
                "cannot ask",
            ),
            ("Game.GetLocalPlayer = function() error('boom') end", "cannot ask"),
            ("Game.GetLocalPlayer = function() return '0' end", "cannot ask"),
        ],
    )
    def test_when_the_turn_is_not_ours_and_active(self, setup: str, reason: str) -> None:
        game = new_game()
        lapse(game)
        game.execute(setup)
        before = len(c.states(game))
        game.execute(rearm_lua())
        c.tick(game, 3)
        assert len(c.states(game)) == before
        assert status(game)["waitingForCommand"] is False
        assert flag(game) is None
        assert_ignored(game, reason)
        assert ui_requests(game) == 0

    def test_without_a_bridge_player(self) -> None:
        game = new_game()
        game.execute("AlphaZeroBridge.Reset()")
        game.execute(rearm_lua())
        game.execute("AlphaZeroBridge.ManualPoll()")
        assert flag(game) is None
        assert_ignored(game, "cannot ask")

    def test_while_a_command_is_in_flight(self) -> None:
        game = new_game()
        c.send(game, [c.unit_move(102, 0, 1)])
        c.tick(game)  # processed; the export is pending
        before = len(c.states(game))
        game.execute(rearm_lua())
        c.tick(game)
        assert flag(game) is None
        assert len(c.states(game)) == before, "the re-arm exported while a command was in flight"
        assert_ignored(game, "command in flight")
        c.tick(game, 6)
        assert len(c.states(game)) == before + 1, "only the command's own export"

    def test_while_a_command_is_queued_and_unread(self) -> None:
        # Review L1: the window lapsed with a command waiting in ExposedMembers.
        game = new_game()
        lapse(game)
        c.send(game, [c.unit_move(102, 0, 1)])
        before = len(c.states(game))
        game.execute(rearm_lua())
        c.tick(game, 3)
        assert flag(game) is None
        assert len(c.states(game)) == before
        assert_ignored(game, "command in flight")

    def test_while_the_window_is_open(self) -> None:
        # Review M1: a second export while the agent already has the state would be stale.
        game = new_game()
        before = len(c.states(game))
        game.execute(rearm_lua())
        c.tick(game, 3)
        assert flag(game) is None
        assert len(c.states(game)) == before
        assert_ignored(game, "window open")

    def test_after_end_turn_was_executed(self) -> None:
        # Review L2: the turn is ending; reopening it would export a state for a turn that is over.
        game = new_game()
        c.send(game, [{"action_type": "end_turn"}])
        c.tick(game, 3)
        assert c.acks(game)[0]["executed"][0]["success"] is True
        before = len(c.states(game))
        game.execute(rearm_lua())
        c.tick(game, 3)
        assert flag(game) is None
        assert len(c.states(game)) == before
        assert status(game)["waitingForCommand"] is False
        assert_ignored(game, "turn ending")

    @pytest.mark.parametrize("value", ["''", "42", "true", "{}"])
    def test_an_empty_or_non_string_flag(self, value: str) -> None:
        game = new_game()
        lapse(game)
        before = len(c.states(game))
        game.execute(f"ExposedMembers.AlphaZeroRearm = {value}")
        c.tick(game, 3)
        assert len(c.states(game)) == before
        assert status(game)["waitingForCommand"] is False
        assert flag(game) is None
        assert_ignored(game, "not a non-empty string", nonce="-")


class TestTurnBoundaries:
    def test_a_flag_left_across_turns_is_dropped_at_turn_start(self) -> None:
        # Review M1: exactly one export for the new turn, the turn start's own.
        game = new_game()
        game.execute(rearm_lua())
        game.execute("AlphaZeroBridge.OnTurnEnd(0)")
        before = len(c.states(game))
        game.execute("AlphaZeroBridge.OnTurnStart(0)")
        c.tick(game, 4)
        assert len(c.states(game)) == before + 1
        assert flag(game) is None
        assert f"Re-arm dropped {NONCE}: the turn start exports the state" in log_text(game)
        assert "Re-arm accepted" not in log_text(game)

    def test_the_command_count_survives_a_re_arm_of_our_own_turn(self) -> None:
        # Review L3.
        game = new_game()
        c.send(game, [c.unit_move(102, 0, 1)])
        c.tick(game, 5)
        assert status(game)["actionsThisTurn"] == 1
        lapse(game)
        game.execute(rearm_lua())
        c.tick(game)
        assert "Re-arm accepted" in log_text(game)
        assert status(game)["actionsThisTurn"] == 1


class TestLoadedMidTurn:
    def load(self, active: int | None) -> Any:
        game = g.load_printing(
            c.CLOCK + TURN_API + f"Civ6Stub.activePlayer = {'nil' if active is None else active}"
        )
        return game

    def test_our_active_turn_is_started(self) -> None:
        game = self.load(0)
        status_now = status(game)
        assert status_now["isMyTurn"] is True
        assert status_now["waitingForCommand"] is True
        assert len(c.states(game)) == 1

    def test_the_real_turn_event_after_a_mid_turn_load_changes_nothing(self) -> None:
        # Review L4: one export for the turn, and the command count kept.
        game = self.load(0)
        c.send(game, [c.unit_move(102, 0, 1)])
        c.tick(game, 5)
        exported, count = len(c.states(game)), status(game)["actionsThisTurn"]
        assert count == 1
        game.execute("AlphaZeroBridge.OnTurnStart(0)")
        c.tick(game, 2)
        assert len(c.states(game)) == exported
        assert status(game)["actionsThisTurn"] == count

    def test_a_flag_pending_at_load_is_dropped(self) -> None:
        # Review M1: a flag from before the mod was (re)initialised is never read later.
        game = g.load_printing(
            c.CLOCK
            + TURN_API
            + "Civ6Stub.activePlayer = nil\nExposedMembers.AlphaZeroRearm = 'old'"
        )
        assert flag(game) is None
        assert "Re-arm dropped old" in log_text(game)

    @pytest.mark.parametrize("active", [1, None])
    def test_another_players_turn_or_no_active_turn_is_not_started(
        self, active: int | None
    ) -> None:
        game = self.load(active)
        assert status(game)["isMyTurn"] is False
        assert status(game)["waitingForCommand"] is False
        assert c.states(game) == []


class TestReset:
    def test_reset_clears_the_flag(self) -> None:
        game = new_game()
        game.execute("ExposedMembers.AlphaZeroRearm = 'pending'")
        game.execute("AlphaZeroBridge.Reset()")
        assert flag(game) is None
