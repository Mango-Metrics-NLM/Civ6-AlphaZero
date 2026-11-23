"""GM-B9 / B5: which player the bridge plays, found the way the game's own scripts find it.

Order: the configured ``AI_PLAYER_ID``; else ``Game.GetLocalPlayer()`` when it is a number >= 0;
else the first human, alive player among ``Game.GetPlayers{ Alive = true }`` (fallback: ids
``0 .. GameDefines.MAX_PLAYERS - 1``); else 0. ``PlayerManager.GetWasEverAliveCount`` is not
called: in the stand-in it answers 2 (and counts the calls), so a mod that enumerates by it can
never find a human with a higher id, such as 3.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import gameinfo_snapshot as g
from tests.fixtures import lua_harness as h

#: Player 3 is the only human; player 0 (human by default in the stand-in) is not.
HUMAN_IS_3 = (
    "Players[3] = Civ6Stub.makePlayer(3)\n"
    "Civ6Stub.players[3] = Civ6Stub.players[1]\n"
    "Players[3].IsMajor = function() return true end\n"
    "Players[3].IsBarbarian = function() return false end\n"
    "Players[0].IsHuman = function() return false end\n"
    "Players[3].IsHuman = function() return true end\n"
    # makePlayer's IsAlive is always true; Civ6Stub.kill only affects the players the others
    # stand-in builds, so a player made here must read the same record.
    "Players[3].IsAlive = function() return not Civ6Stub.dead[3] end\n"
)
#: Counts calls to the enumeration the stand-in keeps at 2.
COUNT_EVER_ALIVE = (
    "Civ6Stub.everAliveCalls = 0\n"
    "PlayerManager.GetWasEverAliveCount = function()"
    " Civ6Stub.everAliveCalls = Civ6Stub.everAliveCalls + 1 return 2 end\n"
)
NO_LOCAL = "Game.GetLocalPlayer = nil\n"


def _found(setup: str = "", *, configured: int | None = None) -> tuple[int, int]:
    """Initialise the bridge afresh; ``(id found, calls to GetWasEverAliveCount)``."""
    runtime = h.load_mod(setup=COUNT_EVER_ALIVE + setup)
    config = ""
    if configured is not None:
        config = (
            "local init = AlphaZeroConfig.Init\n"
            "AlphaZeroConfig.Init = function() local c = init()"
            f" AlphaZeroConfig.Set('AI_PLAYER_ID', {configured}) return c end\n"
        )
    runtime.execute(
        config
        + "Civ6Stub.everAliveCalls = 0\nAlphaZeroBridge.Reset()\nAlphaZeroBridge.Initialize()"
    )
    return (
        int(runtime.eval("AlphaZeroBridge.GetStatus().aiPlayerId")),
        int(runtime.eval("Civ6Stub.everAliveCalls")),
    )


def _id(setup: str = "", *, configured: int | None = None) -> int:
    return _found(setup, configured=configured)[0]


class TestConfigured:
    def test_the_configured_id_wins_over_the_local_player(self) -> None:
        assert _id(configured=5) == 5

    def test_the_configured_id_wins_over_a_human_elsewhere(self) -> None:
        assert _id(HUMAN_IS_3, configured=5) == 5


class TestLocalPlayer:
    @pytest.mark.parametrize("local", [2, 3, 10])
    def test_the_local_player_is_used_when_it_is_a_number_from_zero(self, local: int) -> None:
        assert _id(f"Game.GetLocalPlayer = function() return {local} end\n") == local

    def test_the_local_player_wins_over_the_first_human(self) -> None:
        setup = HUMAN_IS_3 + "Game.GetLocalPlayer = function() return 1 end\n"
        assert _id(setup) == 1

    @pytest.mark.parametrize("answer", ["-1", "nil", "'3'", "true"])
    def test_an_unusable_local_player_falls_through_to_the_first_human(self, answer: str) -> None:
        setup = HUMAN_IS_3 + f"Game.GetLocalPlayer = function() return {answer} end\n"
        assert _id(setup) == 3

    def test_a_raising_local_player_falls_through_to_the_first_human(self) -> None:
        setup = HUMAN_IS_3 + "Game.GetLocalPlayer = function() error('not in a game') end\n"
        assert _id(setup) == 3


class TestFirstHuman:
    def test_the_human_is_player_3_and_no_local_player_exists(self) -> None:
        assert _id(HUMAN_IS_3 + NO_LOCAL) == 3

    def test_without_game_get_players_the_ids_are_walked_up_to_max_players(self) -> None:
        setup = HUMAN_IS_3 + NO_LOCAL + "Civ6Stub.removeGetPlayers()\n"
        assert _id(setup) == 3

    def test_a_raising_game_get_players_is_walked_by_id_too(self) -> None:
        setup = HUMAN_IS_3 + NO_LOCAL + "Civ6Stub.breakGetPlayers()\n"
        assert _id(setup) == 3

    @pytest.mark.parametrize("fallback", [False, True], ids=["game list", "id walk"])
    def test_a_dead_human_is_skipped_for_the_next_one(self, fallback: bool) -> None:
        setup = (
            HUMAN_IS_3
            + "Players[4] = Civ6Stub.makePlayer(4)\n"
            + "Civ6Stub.players[4] = Civ6Stub.players[1]\n"
            + "Players[4].IsHuman = function() return true end\n"
            + "Players[4].IsAlive = function() return not Civ6Stub.dead[4] end\n"
            + "Civ6Stub.kill(3)\n"
            + NO_LOCAL
            + ("Civ6Stub.removeGetPlayers()\n" if fallback else "")
        )
        assert _id(setup) == 4

    def test_the_lowest_human_wins(self) -> None:
        setup = (
            HUMAN_IS_3
            + "Players[7] = Civ6Stub.makePlayer(7)\n"
            + "Civ6Stub.players[7] = Civ6Stub.players[1]\n"
            + "Players[7].IsHuman = function() return true end\n"
            + NO_LOCAL
        )
        assert _id(setup) == 3

    def test_no_human_at_all_is_player_0(self) -> None:
        setup = "Players[0].IsHuman = function() return false end\n" + NO_LOCAL
        assert _id(setup) == 0

    def test_nothing_to_ask_is_player_0(self) -> None:
        setup = NO_LOCAL + "Civ6Stub.removeGetPlayers()\nCiv6Stub.removeMaxPlayers()\n"
        assert _id(setup + "Players[0].IsHuman = function() return false end\n") == 0


class TestNeverCountsPlayersTheOldWay:
    @pytest.mark.parametrize(
        "setup",
        ["", HUMAN_IS_3 + NO_LOCAL, "Game.GetLocalPlayer = function() return 2 end\n"],
        ids=["stand-in defaults", "human is 3", "local is 2"],
    )
    def test_get_was_ever_alive_count_is_not_called(self, setup: str) -> None:
        assert _found(setup)[1] == 0

    def test_get_was_ever_alive_count_is_not_called_with_a_configured_id(self) -> None:
        assert _found(configured=5)[1] == 0

    def test_a_raising_get_was_ever_alive_count_does_not_stop_the_search(self) -> None:
        setup = (
            HUMAN_IS_3
            + NO_LOCAL
            + ("PlayerManager.GetWasEverAliveCount = function() error('do not ask') end\n")
        )
        runtime = h.load_mod(setup=setup)
        runtime.execute("AlphaZeroBridge.Reset()\nAlphaZeroBridge.Initialize()")
        assert runtime.eval("AlphaZeroBridge.GetStatus().aiPlayerId") == 3


def test_the_id_found_at_load_is_the_one_the_state_is_exported_for() -> None:
    """End to end at load time (no re-initialisation): the human is 3, there is no local player."""
    runtime: Any = h.load_mod(setup=HUMAN_IS_3 + NO_LOCAL)
    assert runtime.eval("AlphaZeroBridge.GetStatus().aiPlayerId") == 3


def _decision_lines(setup: str = "", *, configured: int | None = None) -> list[str]:
    """The ``AI player <id>: <rule>`` log lines of a fresh initialisation."""
    runtime = g.load_printing(COUNT_EVER_ALIVE + setup)
    config = ""
    if configured is not None:
        config = (
            "local init = AlphaZeroConfig.Init\n"
            "AlphaZeroConfig.Init = function() local c = init()"
            f" AlphaZeroConfig.Set('AI_PLAYER_ID', {configured}) return c end\n"
        )
    g.clear_printed(runtime)
    runtime.execute(config + "AlphaZeroBridge.Reset()\nAlphaZeroBridge.Initialize()")
    return [line for line in g.printed(runtime) if "AI player " in line]


class TestTheRuleThatDecidedIsLogged:
    """M5: a wrong player here makes every later check ask about the wrong player."""

    def test_configured(self) -> None:
        (line,) = _decision_lines(configured=5)
        assert line.endswith("AI player 5: configured (AI_PLAYER_ID)"), line

    def test_the_games_local_player(self) -> None:
        (line,) = _decision_lines("Game.GetLocalPlayer = function() return 2 end\n")
        assert line.endswith("AI player 2: the game's local player"), line

    def test_the_first_living_human_player(self) -> None:
        (line,) = _decision_lines(HUMAN_IS_3 + NO_LOCAL)
        assert line.endswith("AI player 3: the first living human player"), line

    def test_the_fallback(self) -> None:
        setup = "Players[0].IsHuman = function() return false end\n" + NO_LOCAL
        (line,) = _decision_lines(setup)
        assert line.endswith("AI player 0: fallback: no local player and no human found"), line
