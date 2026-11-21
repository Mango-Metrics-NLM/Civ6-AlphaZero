"""GM-B7a (X1-X4): the mod's commands are ones the game can carry out.

Executed against the stand-in game (lupa, Lua 5.1; ``civ6_stub_commands.lua``). A request
changes nothing when it returns, so every test that claims an effect plays the game's next
frame (``Civ6Stub.applyRequests()``) and reads the game's state. The stand-in drops a request
the game would not carry out and lists it in ``Civ6Stub.dropped``.

Tests marked CONTROL pass on the tree before the fix (they pin behaviour that must not change).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_commands as c
from tests.fixtures import lua_harness as h

SKIP, FORTIFY, SLEEP, FOUND = (
    "UNITOPERATION_SKIP_TURN",
    "UNITOPERATION_FORTIFY",
    "UNITOPERATION_SLEEP",
    "UNITOPERATION_FOUND_CITY",
)
WARRIOR_MOVE = c.unit_move(102, 0, 1)


def unit_cmd(action: str, unit: int = 102, **extra: Any) -> dict[str, Any]:
    return {"action_type": action, "unit_id": unit, **extra}


def attack(action: str = "unit_attack", x: int = 2, y: int = 0) -> dict[str, Any]:
    return {"action_type": action, "unit_id": 102, "target": {"x": x, "y": y}}


def produce(item: Any, item_type: str | None = None) -> dict[str, Any]:
    command: dict[str, Any] = {"action_type": "city_produce", "city_id": 301, "item_id": item}
    if item_type is not None:
        command["item_type"] = item_type
    return command


def run(runtime: Any, command: dict[str, Any]) -> tuple[Any, Any, str]:
    success, code, message = h.execute(runtime, command)
    return success, code, str(message)


def units(runtime: Any, unit_id: int, field: str) -> Any:
    return runtime.eval(f"Civ6Stub.unit({unit_id}) and Civ6Stub.unit({unit_id}).{field}")


def nothing_requested(runtime: Any) -> bool:
    return (
        c.queued(runtime) == 0
        and not c.requests(runtime, "UnitManager.RequestOperation")
        and not c.requests(runtime, "UnitManager.RequestCommand")
        and not c.requests(runtime, "CityManager.RequestOperation")
    )


@pytest.fixture
def game() -> Any:
    return c.load(c.CHECK_LOG)


# ---------------------------------------------------------------------------
# X1: one way to request a unit action, by the row's Hash
# ---------------------------------------------------------------------------
class TestRequestedThroughTheRowHash:
    @pytest.mark.parametrize(
        ("command", "row", "moves_modifier", "has_target"),
        [
            (WARRIOR_MOVE, "UNITOPERATION_MOVE_TO", "NONE", True),
            (attack(), "UNITOPERATION_MOVE_TO", "ATTACK", True),
            (unit_cmd("unit_skip"), SKIP, None, False),
            (unit_cmd("unit_fortify"), FORTIFY, None, False),
            (unit_cmd("unit_sleep"), SLEEP, None, False),
            (unit_cmd("unit_found_city", 101), FOUND, None, False),
        ],
        ids=["move", "attack", "skip", "fortify", "sleep", "found_city"],
    )
    def test_the_request_carries_the_rows_hash_and_parameters(
        self,
        game: Any,
        command: dict[str, Any],
        row: str,
        moves_modifier: str | None,
        has_target: bool,
    ) -> None:
        success, code, message = run(game, command)
        assert (success, code) == (True, None), message
        want = row_of(game, row)
        ((op, params),) = c.requests(game, "UnitManager.RequestOperation")
        assert op == want
        px, py, pm = (
            game.eval(f"UnitOperationTypes.{k}") for k in ("PARAM_X", "PARAM_Y", "PARAM_MODIFIERS")
        )
        if has_target:
            target = command["target"]
            assert params[px] == target["x"] and params[py] == target["y"]
            assert params[pm] == game.eval(f"UnitOperationMoveModifiers.{moves_modifier}")
        else:
            assert params == {}

    def test_the_ranged_attack_carries_the_target_and_no_modifiers(self) -> None:
        game = c.load(c.CHECK_LOG + c.ARCHER)
        assert run(game, attack("unit_ranged_attack"))[:2] == (True, None)
        ((op, params),) = c.requests(game, "UnitManager.RequestOperation")
        assert op == row_of(game, "UNITOPERATION_RANGE_ATTACK")
        assert params == {
            game.eval("UnitOperationTypes.PARAM_X"): 2,
            game.eval("UnitOperationTypes.PARAM_Y"): 0,
        }

    def test_the_message_says_requested_and_does_not_claim_the_action_happened(
        self, game: Any
    ) -> None:
        for command in (unit_cmd("unit_fortify"), unit_cmd("unit_skip", 101), WARRIOR_MOVE):
            success, _, message = run(game, command)
            assert success is True
            assert "request" in message.lower(), message
        for old in ("Unit fortified", "Unit skipped", "City founded", "Unit sleeping"):
            assert old not in message

    def test_a_request_made_for_the_game_still_changes_nothing_by_itself(self, game: Any) -> None:
        run(game, WARRIOR_MOVE)
        assert (units(game, 102, "x"), units(game, 102, "y")) == (1, 0)
        assert c.queued(game) == 1

    def test_a_target_action_asks_the_strict_check_with_the_parameters(self, game: Any) -> None:
        run(game, WARRIOR_MOVE)
        (check,) = [
            k for k in iter_checks(game) if k["hash"] == row_of(game, "UNITOPERATION_MOVE_TO")
        ]
        assert isinstance(check["fourth"], dict)
        assert check["fourth"][game.eval("UnitOperationTypes.PARAM_X")] == 0

    def test_an_action_without_a_target_asks_the_strict_check_with_no_targets(
        self, game: Any
    ) -> None:
        run(game, unit_cmd("unit_skip"))
        checks = [k for k in iter_checks(game) if k["hash"] == row_of(game, SKIP)]
        assert checks, "the strict check was not asked for the skip row"
        assert all(k["fourth"] is False for k in checks)
        assert all(k["fifth"] == game.eval("OperationResultsTypes.NO_TARGETS") for k in checks)

    def test_the_hash_comes_from_the_game_row_not_from_a_constant(self) -> None:
        decoys = "UnitOperationTypes.SKIP_TURN = 12345\nUnitOperationTypes.MOVE_TO = 12346\n"
        game = c.load(decoys)
        assert run(game, unit_cmd("unit_skip"))[:2] == (True, None)
        assert run(game, WARRIOR_MOVE)[:2] == (True, None)
        c.frame(game)
        assert units(game, 102, "skipped") is True
        assert (units(game, 102, "x"), units(game, 102, "y")) == (0, 1)
        assert c.dropped(game) == 0

    def test_a_refusal_carries_the_games_reason_and_requests_nothing(self, game: Any) -> None:
        game.execute("Civ6Stub.unit(102).moves = 0")
        success, code, message = run(game, unit_cmd("unit_skip"))
        assert (success, code) == (False, "OPERATION_REFUSED")
        assert "no moves left" in message
        assert nothing_requested(game)

    def test_the_loose_check_is_not_what_decides(self, game: Any) -> None:
        # The stand-in's loose check passes a unit with no moves; the strict one does not.
        game.execute("Civ6Stub.unit(102).moves = 0")
        assert run(game, unit_cmd("unit_fortify"))[1] == "OPERATION_REFUSED"
        assert all(k["fourth"] is not True for k in iter_checks(game))

    def test_a_missing_row_names_the_type_and_requests_nothing(self) -> None:
        game = c.load("table.remove(Civ6Stub.unitOperationRows, 4)")
        success, code, message = run(game, unit_cmd("unit_skip"))
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert SKIP in message
        assert nothing_requested(game)

    def test_a_row_in_the_command_table_is_requested_as_a_command(self) -> None:
        setup = (
            "local row = table.remove(Civ6Stub.unitOperationRows, 4)\n"
            "table.insert(Civ6Stub.unitCommandRows,"
            " { Hash = row.Hash, CommandType = 'UNITOPERATION_SKIP_TURN' })\n"
            "UnitManager.CanStartCommand = function() return true, {} end\n"
        )
        game = c.load(setup)
        success, code, message = run(game, unit_cmd("unit_skip"))
        assert (success, code) == (True, None), message
        ((command, _),) = c.requests(game, "UnitManager.RequestCommand")
        assert command == 91004
        assert not c.requests(game, "UnitManager.RequestOperation")

    def test_a_command_row_asks_the_strict_command_check(self) -> None:
        setup = (
            "local row = table.remove(Civ6Stub.unitOperationRows, 4)\n"
            "table.insert(Civ6Stub.unitCommandRows,"
            " { Hash = row.Hash, CommandType = 'UNITOPERATION_SKIP_TURN' })\n"
            + c.CHECK_LOG
            + "\nUnitManager.CanStartCommand = function(u, hash, loose, want)"
            " table.insert(Civ6Stub.commandChecks, { hash = hash, loose = loose, want = want })"
            " return true, {} end\n"
        )
        game = c.load(setup)
        run(game, unit_cmd("unit_skip"))
        assert game.eval("#Civ6Stub.commandChecks") == 1
        assert game.eval("Civ6Stub.commandChecks[1].loose") is False
        assert game.eval("Civ6Stub.commandChecks[1].want") is True

    def test_a_command_the_game_refuses_is_not_requested(self) -> None:
        setup = (
            "local row = table.remove(Civ6Stub.unitOperationRows, 4)\n"
            "table.insert(Civ6Stub.unitCommandRows,"
            " { Hash = row.Hash, CommandType = 'UNITOPERATION_SKIP_TURN' })\n"
            "UnitManager.CanStartCommand = function() return false, {} end\n"
        )
        game = c.load(setup)
        assert run(game, unit_cmd("unit_skip"))[:2] == (False, "OPERATION_REFUSED")
        assert nothing_requested(game)

    def test_the_row_in_neither_table_is_a_failure_naming_it(self) -> None:
        game = c.load("table.remove(Civ6Stub.unitOperationRows, 4)\nCiv6Stub.unitCommandRows = {}")
        game.execute("GameInfo.UnitCommands = function() return function() return nil end end")
        success, code, message = run(game, unit_cmd("unit_skip"))
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert SKIP in message

    # CONTROL: passes on the old tree (it already answers EXECUTION_FAILED without UnitManager).
    def test_control_without_unit_manager_nothing_is_requested(self) -> None:
        game = c.load("UnitManager = nil")
        for command in (WARRIOR_MOVE, unit_cmd("unit_skip"), unit_cmd("unit_fortify")):
            assert run(game, command)[:2] == (False, "EXECUTION_FAILED")
        assert nothing_requested(game)

    @pytest.mark.parametrize("missing", ["CanStartOperation", "RequestOperation"])
    def test_a_missing_function_is_a_failure_and_requests_nothing(self, missing: str) -> None:
        game = c.load(f"UnitManager.{missing} = nil")
        for command in (WARRIOR_MOVE, unit_cmd("unit_skip")):
            assert run(game, command)[:2] == (False, "EXECUTION_FAILED")
        assert nothing_requested(game)


def row_of(runtime: Any, type_name: str) -> int:
    for table, column in (("UnitOperations", "OperationType"), ("UnitCommands", "CommandType")):
        value = runtime.eval(
            f"(function() for r in GameInfo.{table}() do if r.{column} == '{type_name}'"
            " then return r.Hash end end end)()"
        )
        if value is not None:
            return int(value)
    raise AssertionError(type_name)


def iter_checks(runtime: Any) -> list[dict[str, Any]]:
    found = []
    for i in range(1, int(runtime.eval("#Civ6Stub.checks")) + 1):
        entry = runtime.eval(f"Civ6Stub.checks[{i}]")
        fourth = entry["fourth"]
        found.append(
            {
                "hash": entry["hash"],
                "fourth": dict(fourth.items()) if hasattr(fourth, "items") else fourth,
                "fifth": entry["fifth"],
            }
        )
    return found


# ---------------------------------------------------------------------------
# X2: what must be true in the game afterwards
# ---------------------------------------------------------------------------
class TestMove:
    def test_a_move_to_a_free_neighbouring_tile_puts_the_unit_there(self) -> None:
        game = c.load()
        assert run(game, WARRIOR_MOVE)[:2] == (True, None)
        c.frame(game)
        assert (units(game, 102, "x"), units(game, 102, "y")) == (0, 1)
        assert c.dropped(game) == 0

    def test_a_move_onto_the_enemy_is_refused_and_never_attacks(self) -> None:
        game = c.load()
        success, code, _ = run(game, c.unit_move(102, 2, 0))
        assert (success, code) == (False, "OPERATION_REFUSED")
        c.frame(game)
        assert not units(game, 201, "damage")
        assert (units(game, 102, "x"), units(game, 102, "y")) == (1, 0)
        assert nothing_requested(game)

    def test_a_move_onto_an_own_unit_is_refused(self) -> None:
        game = c.load()
        assert run(game, c.unit_move(102, 0, 0))[:2] == (False, "OPERATION_REFUSED")
        assert nothing_requested(game)

    # CONTROL: existing behaviour.
    def test_control_a_unit_with_no_moves_answers_not_enough_movement(self) -> None:
        game = c.load("Civ6Stub.unit(102).moves = 0")
        assert run(game, WARRIOR_MOVE)[:2] == (False, "NOT_ENOUGH_MOVEMENT")
        assert nothing_requested(game)


class TestAttack:
    def test_an_attack_on_the_enemy_damages_it(self) -> None:
        game = c.load()
        assert run(game, attack())[:2] == (True, None)
        c.frame(game)
        assert units(game, 201, "damage") > 0
        assert c.dropped(game) == 0

    @pytest.mark.parametrize(("x", "y"), [(3, 0), (0, 0)], ids=["empty tile", "own unit"])
    def test_an_attack_with_nothing_to_attack_is_an_invalid_target(self, x: int, y: int) -> None:
        game = c.load()
        success, code, _ = run(game, attack(x=x, y=y))
        assert (success, code) == (False, "INVALID_TARGET")
        assert nothing_requested(game)

    def test_a_ranged_attack_in_range_damages_the_target(self) -> None:
        # CONTROL on the old tree: the old code already sends the ranged operation.
        game = c.load(c.ARCHER)
        assert run(game, attack("unit_ranged_attack"))[:2] == (True, None)
        c.frame(game)
        assert units(game, 201, "damage") > 0

    def test_a_ranged_attack_out_of_range_is_refused(self) -> None:
        game = c.load(c.ARCHER + "Civ6Stub.unit(102).range = 1\nCiv6Stub.unit(201).x = 3\n")
        success, code, _ = run(game, attack("unit_ranged_attack", x=3))
        assert (success, code) == (False, "OPERATION_REFUSED")
        assert nothing_requested(game)

    def test_a_unit_without_ranged_strength_cannot_make_a_ranged_attack(self) -> None:
        game = c.load()
        assert run(game, attack("unit_ranged_attack"))[:2] == (False, "OPERATION_REFUSED")
        assert nothing_requested(game)


class TestSkipFortifySleep:
    def test_skip_marks_the_unit_and_clears_the_end_of_turn_blocker(self) -> None:
        game = c.load("Civ6Stub.unit(101).moves = 0\nCiv6Stub.unitsNeedOrders = true")
        assert game.eval("UI.CanEndTurn()") is False, "premise: the unit blocks the turn"
        assert run(game, unit_cmd("unit_skip"))[:2] == (True, None)
        c.frame(game)
        assert units(game, 102, "skipped") is True
        assert game.eval("UI.CanEndTurn()") is True
        assert game.eval("NotificationManager.GetFirstEndTurnBlocking(0)") == 0
        assert run(game, {"action_type": "end_turn"})[:2] == (True, None)

    def test_fortify_fortifies_and_a_second_fortify_is_refused(self) -> None:
        game = c.load()
        assert run(game, unit_cmd("unit_fortify"))[:2] == (True, None)
        c.frame(game)
        assert units(game, 102, "activity") == "fortified"
        success, code, _ = run(game, unit_cmd("unit_fortify"))
        assert (success, code) == (False, "OPERATION_REFUSED")
        assert c.queued(game) == 0

    def test_sleep_puts_the_unit_to_sleep(self) -> None:
        game = c.load()
        assert run(game, unit_cmd("unit_sleep"))[:2] == (True, None)
        c.frame(game)
        assert units(game, 102, "activity") == "sleeping"
        assert c.dropped(game) == 0

    def test_a_settler_cannot_fortify(self) -> None:
        game = c.load()
        assert run(game, unit_cmd("unit_fortify", 101))[:2] == (False, "OPERATION_REFUSED")
        assert nothing_requested(game)

    def test_a_unit_with_no_moves_cannot_be_skipped(self) -> None:
        game = c.load("Civ6Stub.unit(102).moves = 0")
        assert run(game, unit_cmd("unit_skip"))[:2] == (False, "OPERATION_REFUSED")
        assert nothing_requested(game)


class TestFoundCity:
    def test_the_settler_is_replaced_by_a_city_on_its_tile(self) -> None:
        game = c.load()
        before = game.eval("#Civ6Stub.cityRecords[0]")
        assert run(game, unit_cmd("unit_found_city", 101))[:2] == (True, None)
        c.frame(game)
        assert units(game, 101, "id") is None
        assert game.eval("#Civ6Stub.cityRecords[0]") == before + 1
        assert game.eval("Civ6Stub.cityRecords[0][2].x") == 0
        assert game.eval("Civ6Stub.cityRecords[0][2].y") == 0

    def test_where_no_city_can_be_founded_the_command_is_refused(self) -> None:
        game = c.load("Civ6Stub.canFoundCity = false")
        success, code, _ = run(game, unit_cmd("unit_found_city", 101))
        assert (success, code) == (False, "OPERATION_REFUSED")
        c.frame(game)
        assert units(game, 101, "id") == 101
        assert game.eval("#Civ6Stub.cityRecords[0]") == 1

    def test_a_warrior_cannot_found_a_city(self) -> None:
        game = c.load()
        assert run(game, unit_cmd("unit_found_city"))[:2] == (False, "OPERATION_REFUSED")


# ---------------------------------------------------------------------------
# X3: no command starts a war
# ---------------------------------------------------------------------------
PEACE = "Civ6Stub.setPeace(0, 1, true)\n"
ENEMY_CITY = (
    "Civ6Stub.cityRecords[1] = { { id = 401, name = 'Rival', x = 2, y = 1, population = 2 } }\n"
)


class TestNoWar:
    @pytest.mark.parametrize("action", ["unit_attack", "unit_ranged_attack"])
    def test_an_attack_at_peace_starts_no_war_and_hurts_nobody(self, action: str) -> None:
        game = c.load(c.CHECK_LOG + c.ARCHER + PEACE)
        success, code, _ = run(game, attack(action))
        assert (success, code) == (False, "WOULD_START_WAR")
        assert nothing_requested(game)
        c.frame(game)
        assert game.eval("#Civ6Stub.wars") == 0
        assert not units(game, 201, "damage")

    def test_a_move_into_a_city_of_a_player_at_peace_is_refused(self) -> None:
        game = c.load(PEACE + ENEMY_CITY)
        success, code, _ = run(game, c.unit_move(102, 2, 1))
        assert (success, code) == (False, "WOULD_START_WAR")
        assert nothing_requested(game)
        c.frame(game)
        assert (units(game, 102, "x"), units(game, 102, "y")) == (1, 0)

    def test_control_at_war_the_same_move_into_a_city_goes_through(self) -> None:
        game = c.load(ENEMY_CITY)
        assert run(game, c.unit_move(102, 2, 1))[:2] == (True, None)
        c.frame(game)
        assert (units(game, 102, "x"), units(game, 102, "y")) == (2, 1)

    def test_control_at_war_the_attack_goes_through_and_starts_no_war(self) -> None:
        game = c.load()
        assert run(game, attack())[:2] == (True, None)
        c.frame(game)
        assert units(game, 201, "damage") > 0
        assert game.eval("#Civ6Stub.wars") == 0

    @pytest.mark.parametrize(
        "command",
        [WARRIOR_MOVE, attack(), attack("unit_ranged_attack")],
        ids=["move", "attack", "ranged"],
    )
    def test_the_game_is_asked_about_the_target_tile_of_the_unit(
        self, command: dict[str, Any]
    ) -> None:
        game = c.load(c.CHECK_LOG + c.ARCHER)
        run(game, command)
        assert game.eval("#Civ6Stub.warChecks") == 1
        target = command["target"]
        assert game.eval("Civ6Stub.warChecks[1].player") == 0
        assert game.eval("Civ6Stub.warChecks[1].id") == 102
        assert game.eval("Civ6Stub.warChecks[1].x") == target["x"]
        assert game.eval("Civ6Stub.warChecks[1].y") == target["y"]

    @pytest.mark.parametrize(
        "setup",
        [
            "CombatManager = nil",
            "CombatManager.IsAttackChangeWarState = nil",
            "CombatManager.IsAttackChangeWarState = function() error('no answer') end",
        ],
        ids=["no manager", "no function", "raises"],
    )
    @pytest.mark.parametrize(
        "command",
        [WARRIOR_MOVE, attack(), attack("unit_ranged_attack")],
        ids=["move", "attack", "ranged"],
    )
    def test_when_the_game_cannot_be_asked_nothing_is_attacked(
        self, setup: str, command: dict[str, Any]
    ) -> None:
        game = c.load(c.ARCHER + setup)
        success, code, _ = run(game, command)
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert nothing_requested(game)


# ---------------------------------------------------------------------------
# X4: production by hash
# ---------------------------------------------------------------------------
PRODUCTION_BLOCKER = (
    "Civ6Stub.notifications[0] = { { id = 7,"
    " blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_PRODUCTION,"
    " message = 'Choose production', typeName = 'NOTIFICATION_PRODUCTION', dismissed = false } }\n"
)


def building(runtime: Any) -> Any:
    return runtime.eval("Civ6Stub.city(301).building")


class TestProduction:
    def test_a_unit_by_name_replaces_the_production_and_clears_the_blocker(self) -> None:
        game = c.load(
            c.CHECK_LOG + PRODUCTION_BLOCKER + "Civ6Stub.city(301).building = 'BUILDING_GRANARY'"
        )
        assert game.eval("UI.CanEndTurn()") is False, "premise"
        success, code, message = run(game, produce("UNIT_WARRIOR"))
        assert (success, code) == (True, None), message
        c.frame(game)
        assert building(game) == "UNIT_WARRIOR"
        assert game.eval("UI.CanEndTurn()") is True
        assert c.dropped(game) == 0

    def test_the_request_has_the_hash_and_the_exclusive_insert_mode(self) -> None:
        game = c.load(c.CHECK_LOG)
        run(game, produce("UNIT_WARRIOR"))
        ((op, params),) = c.requests(game, "CityManager.RequestOperation")
        assert op == game.eval("CityOperationTypes.BUILD")
        assert params == {
            game.eval("CityOperationTypes.PARAM_UNIT_TYPE"): c.row_hash(
                game, "Units", "UnitType", "UNIT_WARRIOR"
            ),
            game.eval("CityOperationTypes.PARAM_INSERT_MODE"): game.eval(
                "CityOperationTypes.VALUE_EXCLUSIVE"
            ),
        }

    def test_the_strict_check_is_asked_with_the_hash(self) -> None:
        game = c.load(c.CHECK_LOG)
        run(game, produce("UNIT_WARRIOR"))
        assert game.eval("#Civ6Stub.produceChecks") >= 1
        entry = game.eval("Civ6Stub.produceChecks[1]")
        assert entry["ref"] == c.row_hash(game, "Units", "UnitType", "UNIT_WARRIOR")
        assert entry["loose"] is False and entry["strict"] is True

    def test_a_building_by_name_uses_the_building_parameter(self) -> None:
        game = c.load()
        assert run(game, produce("BUILDING_GRANARY", "building"))[:2] == (True, None)
        ((_, params),) = c.requests(game, "CityManager.RequestOperation")
        assert params[game.eval("CityOperationTypes.PARAM_BUILDING_TYPE")] == c.row_hash(
            game, "Buildings", "BuildingType", "BUILDING_GRANARY"
        )
        c.frame(game)
        assert building(game) == "BUILDING_GRANARY"

    def test_a_row_index_builds_the_row_of_the_table_item_type_names(self) -> None:
        game = c.load()
        assert run(game, produce(1))[:2] == (True, None)
        c.frame(game)
        assert building(game) == "UNIT_WARRIOR"
        assert run(game, produce(1, "building"))[:2] == (True, None)
        c.frame(game)
        assert building(game) == "BUILDING_GRANARY"

    @pytest.mark.parametrize("item_type", [None, "unit", "building"])
    def test_a_name_is_found_whatever_item_type_says(self, item_type: str | None) -> None:
        # The contract: a name is looked up in every table. (This replaced a case that
        # expected INVALID_COMMAND for a building named without item_type.)
        game = c.load()
        assert run(game, produce("BUILDING_GRANARY", item_type))[:2] == (True, None)
        c.frame(game)
        assert building(game) == "BUILDING_GRANARY"

    def test_a_city_that_can_start_it_only_later_is_refused(self) -> None:
        game = c.load("Civ6Stub.cities[301].canProduce = { UNIT_WARRIOR = 'later' }")
        success, code, _ = run(game, produce("UNIT_WARRIOR"))
        assert (success, code) == (False, "OPERATION_REFUSED")
        assert nothing_requested(game)

    def test_a_settler_the_city_cannot_start_now_is_refused(self) -> None:
        game = c.load("Civ6Stub.city(301).building = 'UNIT_WARRIOR'")
        assert run(game, produce("UNIT_SETTLER"))[:2] == (False, "OPERATION_REFUSED")
        assert nothing_requested(game)
        c.frame(game)
        assert building(game) == "UNIT_WARRIOR"

    @pytest.mark.parametrize(
        "command",
        [
            produce("DISTRICT_CAMPUS", "district"),
            produce(1, "district"),
            produce("BUILDING_GRANARY", "wonder"),
        ],
        ids=["district by name", "district by index", "wonder"],
    )
    def test_districts_and_wonders_are_not_implemented(self, command: dict[str, Any]) -> None:
        game = c.load()
        success, code, message = run(game, command)
        assert (success, code) == (False, "NOT_IMPLEMENTED")
        assert "placement" in message.lower()
        assert nothing_requested(game)

    @pytest.mark.parametrize(
        ("item", "item_type", "named"),
        [
            ("UNIT_NOPE", None, "UNIT_NOPE"),
            (99, None, "99"),
            (99, "building", "99"),
        ],
        ids=["unknown name", "unknown index", "unknown building"],
    )
    def test_an_item_with_no_row_is_an_invalid_command_naming_it(
        self, item: Any, item_type: str | None, named: str
    ) -> None:
        game = c.load()
        success, code, message = run(game, produce(item, item_type))
        assert (success, code) == (False, "INVALID_COMMAND")
        assert named in message
        assert nothing_requested(game)

    def test_a_row_the_city_does_not_offer_at_all_is_refused(self) -> None:
        game = c.load()
        assert run(game, produce(0, "building"))[:2] == (False, "OPERATION_REFUSED")
        assert nothing_requested(game)

    def test_without_the_city_manager_the_command_fails(self) -> None:
        # CONTROL: the old code also fails here (its fallback finds no queue method).
        game = c.load("CityManager = nil")
        assert run(game, produce("UNIT_WARRIOR"))[:2] == (False, "EXECUTION_FAILED")

    def test_control_an_unknown_city_is_still_not_found(self) -> None:
        game = c.load()
        command = produce("UNIT_WARRIOR") | {"city_id": 999}
        assert run(game, command)[:2] == (False, "INVALID_COMMAND")

    def test_a_missing_item_is_still_an_invalid_command(self) -> None:
        # CONTROL
        game = c.load()
        assert run(game, {"action_type": "city_produce", "city_id": 301})[1] == "INVALID_COMMAND"


# ---------------------------------------------------------------------------
# Review of GM-B7a: a wonder by name, reasons for a targeted refusal, fog, item_type
# ---------------------------------------------------------------------------
GRANARY_NEEDS_A_PLOT = (
    "for row in GameInfo.Buildings() do"
    " if row.BuildingType == 'BUILDING_GRANARY' then row.RequiresPlacement = true end end\n"
)


class TestReviewFindings:
    @pytest.mark.parametrize("command", [produce("BUILDING_GRANARY"), produce(1, "building")])
    def test_a_building_that_needs_a_plot_is_refused_however_it_is_named(
        self, command: dict[str, Any]
    ) -> None:
        game = c.load(GRANARY_NEEDS_A_PLOT)
        success, code, message = run(game, command)
        assert (success, code) == (False, "NOT_IMPLEMENTED")
        assert "placement" in message.lower()
        assert nothing_requested(game)

    @pytest.mark.parametrize("item_type", ["project", "improvement", ""])
    def test_an_unknown_item_type_is_not_read_as_a_unit(self, item_type: str) -> None:
        game = c.load()
        success, code, _ = run(game, produce(1, item_type))
        assert (success, code) == (False, "INVALID_COMMAND")
        assert nothing_requested(game)

    def test_a_targeted_check_asks_for_the_games_reasons(self) -> None:
        # The game's UI passes `true` after a parameter table when it wants results back.
        game = c.load(c.CHECK_LOG)
        run(game, WARRIOR_MOVE)
        targeted = [check for check in iter_checks(game) if isinstance(check["fourth"], dict)]
        assert targeted, "premise: the move asked a check with its parameters"
        assert all(check["fifth"] is True for check in targeted)

    def test_a_move_into_a_tile_the_player_cannot_see_does_not_ask_about_war(self) -> None:
        # The game's own move routine asks only for a visible plot.
        fog = (
            "PlayersVisibility = { [0] = { IsVisible = function() return false end } }\n"
            "CombatManager.IsAttackChangeWarState = function() error('asked about fog') end\n"
        )
        game = c.load(fog)
        assert run(game, WARRIOR_MOVE)[:2] == (True, None)
        c.frame(game)
        assert (units(game, 102, "x"), units(game, 102, "y")) == (0, 1)

    def test_an_attack_always_asks_about_war_even_in_fog(self) -> None:
        fog = "PlayersVisibility = { [0] = { IsVisible = function() return false end } }\n" + PEACE
        game = c.load(fog)
        assert run(game, attack())[:2] == (False, "WOULD_START_WAR")
        assert nothing_requested(game)


class TestProductionNeedsAClearYes:
    """Review of #158: a city that cannot be asked was treated as one that said yes."""

    @pytest.mark.parametrize(
        "setup",
        [
            "Players[0]:GetCities():FindID(301):GetBuildQueue().CanProduce = nil\n",
            "Players[0]:GetCities():FindID(301):GetBuildQueue().CanProduce ="
            " function() error('not in this context') end\n",
            "Players[0]:GetCities():FindID(301):GetBuildQueue().CanProduce ="
            " function() return nil end\n",
            "Players[0]:GetCities():FindID(301).GetBuildQueue = function() return nil end\n",
        ],
        ids=["no CanProduce", "CanProduce raises", "CanProduce answers nil", "no build queue"],
    )
    def test_nothing_is_requested_without_an_explicit_yes(self, setup: str) -> None:
        game = c.load(setup)
        success, code, _ = run(game, produce("UNIT_WARRIOR"))
        assert (success, code) == (False, "OPERATION_REFUSED")
        assert nothing_requested(game)
        c.frame(game)
        assert building(game) is None
