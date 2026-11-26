"""Lua 5.1 stand-in execution tests for secondary commands and fail-closed unit_delete.

Covers the five handlers previously tested only at the dispatch-key level:
``unit_delete`` (fail-closed behind ``ALLOW_UNIT_DELETE`` until live-tested),
``unit_build``, ``unit_promote``, ``city_buy``, and ``city_citizen``.

Executed against the real ``AlphaZeroAI`` bundle in ``lupa`` Lua 5.1 via
:mod:`tests.fixtures.lua_commands` and :mod:`tests.fixtures.lua_harness`.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_commands as c
from tests.fixtures import lua_harness as h


def _run(runtime: Any, command: dict[str, Any]) -> tuple[Any, Any, str]:
    success, code, message = h.execute(runtime, command)
    return success, code, str(message)


def _nothing_requested(runtime: Any) -> bool:
    return (
        c.queued(runtime) == 0
        and not c.requests(runtime, "UnitManager.RequestOperation")
        and not c.requests(runtime, "UnitManager.RequestCommand")
        and not c.requests(runtime, "CityManager.RequestOperation")
    )


class TestUnitDeleteFailClosed:
    """``unit_delete`` must refuse unless ``ALLOW_UNIT_DELETE`` is boolean ``true``."""

    def test_setting_exists_and_defaults_to_false(self) -> None:
        assert c.load().eval('AlphaZeroConfig.Get("ALLOW_UNIT_DELETE")') is False

    @pytest.mark.parametrize("value", [None, "false", '"true"', "1", '"yes"'])
    def test_refused_unless_boolean_true_and_unit_survives(self, value: str | None) -> None:
        runtime = c.load()
        if value is not None:
            runtime.execute(f'AlphaZeroConfig.Set("ALLOW_UNIT_DELETE", {value})')
        success, code, message = _run(runtime, {"action_type": "unit_delete", "unit_id": 102})
        assert (success, code) == (False, "OPERATION_REFUSED")
        assert "ALLOW_UNIT_DELETE" in message
        assert _nothing_requested(runtime)
        c.frame(runtime)
        assert runtime.eval("Civ6Stub.unit(102) ~= nil") is True

    def test_when_enabled_deletes_unit_on_next_frame(self) -> None:
        runtime = c.load()
        runtime.execute('AlphaZeroConfig.Set("ALLOW_UNIT_DELETE", true)')
        success, code, message = _run(runtime, {"action_type": "unit_delete", "unit_id": 102})
        assert (success, code, message) == (True, None, "Unit deleted")
        calls = c.requests(runtime, "UnitManager.RequestCommand")
        assert len(calls) == 1
        assert calls[0][0] == int(runtime.eval("UnitCommandTypes.DELETE"))
        c.frame(runtime)
        assert c.dropped(runtime) == 0
        assert runtime.eval("Civ6Stub.unit(102) == nil") is True

    def test_when_enabled_validates_unit_id_and_existence(self) -> None:
        runtime = c.load()
        runtime.execute('AlphaZeroConfig.Set("ALLOW_UNIT_DELETE", true)')
        assert _run(runtime, {"action_type": "unit_delete"})[:2] == (False, "INVALID_COMMAND")
        assert _run(runtime, {"action_type": "unit_delete", "unit_id": 999})[:2] == (
            False,
            "UNIT_NOT_FOUND",
        )
        assert _nothing_requested(runtime)

    @pytest.mark.parametrize("allow", [None, False, True], ids=["default", "false", "true"])
    def test_unit_delete_is_never_offered(self, allow: bool | None) -> None:
        runtime = c.load()
        if allow is not None:
            runtime.execute(f'AlphaZeroConfig.Set("ALLOW_UNIT_DELETE", {str(allow).lower()})')
        state = h.export_state(runtime)
        offered = [a.get("action_type") for a in state.get("legal_actions", [])]
        assert "unit_delete" not in offered


class TestUnitBuild:
    """Tests for ``executeUnitBuild`` in Lua 5.1."""

    def test_validates_missing_args_and_unknown_unit(self) -> None:
        runtime = c.load()
        assert _run(runtime, {"action_type": "unit_build", "item_id": "IMPROVEMENT_FARM"})[:2] == (
            False,
            "INVALID_COMMAND",
        )
        assert _run(runtime, {"action_type": "unit_build", "unit_id": 102})[:2] == (
            False,
            "INVALID_COMMAND",
        )
        success, code, msg = _run(
            runtime,
            {"action_type": "unit_build", "unit_id": 999, "item_id": "IMPROVEMENT_FARM"},
        )
        # Validate() rejects missing units before the handler runs under VALIDATE_COMMANDS=true.
        assert (success, code) == (False, "INVALID_COMMAND")
        assert "not found" in msg.lower()
        assert _nothing_requested(runtime)

    def test_requests_build_improvement_and_applies_on_next_frame(self) -> None:
        runtime = c.load()
        success, code, message = _run(
            runtime,
            {"action_type": "unit_build", "unit_id": 102, "item_id": "IMPROVEMENT_FARM"},
        )
        assert (success, code, message) == (True, None, "Build started")
        calls = c.requests(runtime, "UnitManager.RequestOperation")
        assert len(calls) == 1
        op, params = calls[0]
        assert op == int(runtime.eval("UnitOperationTypes.BUILD_IMPROVEMENT"))
        param_key = runtime.eval("UnitOperationTypes.PARAM_IMPROVEMENT_TYPE")
        assert params == {param_key: "IMPROVEMENT_FARM"}
        c.frame(runtime)
        assert c.dropped(runtime) == 0
        assert runtime.eval("#Civ6Stub.builtImprovements") == 1
        assert runtime.eval("Civ6Stub.builtImprovements[1].unitId") == 102
        assert runtime.eval("Civ6Stub.builtImprovements[1].itemId") == "IMPROVEMENT_FARM"

    def test_without_unit_manager_returns_execution_failed(self) -> None:
        runtime = c.load("UnitManager = nil")
        assert _run(
            runtime,
            {"action_type": "unit_build", "unit_id": 102, "item_id": "IMPROVEMENT_FARM"},
        )[:2] == (False, "EXECUTION_FAILED")
        assert _nothing_requested(runtime)


class TestUnitPromote:
    """Tests for ``executeUnitPromote`` in Lua 5.1."""

    def test_validates_missing_args_unknown_unit_and_cannot_promote(self) -> None:
        runtime = c.load()
        assert _run(runtime, {"action_type": "unit_promote"})[:2] == (False, "INVALID_COMMAND")
        assert _run(runtime, {"action_type": "unit_promote", "unit_id": 999})[:2] == (
            False,
            "UNIT_NOT_FOUND",
        )
        success, code, msg = _run(
            runtime, {"action_type": "unit_promote", "unit_id": 102, "item_id": 0}
        )
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert "cannot promote" in msg.lower()
        assert _nothing_requested(runtime)

    @pytest.mark.parametrize(
        ("item_id", "expected_param"),
        [(0, None), (7, 7)],
        ids=["first_available", "explicit_promotion_id"],
    )
    def test_requests_promotion_and_applies_on_next_frame(
        self, item_id: int, expected_param: int | None
    ) -> None:
        runtime = c.load("Civ6Stub.unit(102).canPromote = true")
        success, code, message = _run(
            runtime,
            {"action_type": "unit_promote", "unit_id": 102, "item_id": item_id},
        )
        assert (success, code, message) == (True, None, "Promotion requested")
        calls = c.requests(runtime, "UnitManager.RequestOperation")
        assert len(calls) == 1
        op, params = calls[0]
        assert op == int(runtime.eval("UnitOperationTypes.PROMOTE"))
        if expected_param is None:
            assert params == {}
        else:
            param_key = runtime.eval("UnitOperationTypes.PARAM_PROMOTION_TYPE")
            assert params == {param_key: expected_param}
        c.frame(runtime)
        assert c.dropped(runtime) == 0
        assert runtime.eval("#Civ6Stub.promotions") == 1
        assert runtime.eval("Civ6Stub.promotions[1].unitId") == 102
        assert runtime.eval("Civ6Stub.promotions[1].promotionId") == expected_param


class TestCityBuy:
    """Tests for ``executeCityBuy`` primary and fallback paths in Lua 5.1."""

    def test_validates_missing_args_and_unknown_city(self) -> None:
        runtime = c.load()
        assert _run(runtime, {"action_type": "city_buy", "item_id": 6111})[:2] == (
            False,
            "INVALID_COMMAND",
        )
        assert _run(runtime, {"action_type": "city_buy", "city_id": 301})[:2] == (
            False,
            "INVALID_COMMAND",
        )
        success, code, msg = _run(
            runtime, {"action_type": "city_buy", "city_id": 999, "item_id": 6111}
        )
        assert (success, code) == (False, "INVALID_COMMAND")
        assert "not found" in msg.lower()
        assert _nothing_requested(runtime)

    @pytest.mark.parametrize(
        ("item_type", "item_id", "expected_param_key"),
        [
            (None, 6111, "PARAM_UNIT_TYPE"),
            ("unit", 6111, "PARAM_UNIT_TYPE"),
            ("building", 6200, "PARAM_BUILDING_TYPE"),
            ("district", 6311, "PARAM_DISTRICT_TYPE"),
        ],
    )
    def test_primary_city_manager_purchase_applies(
        self, item_type: str | None, item_id: int, expected_param_key: str
    ) -> None:
        runtime = c.load()
        cmd: dict[str, Any] = {"action_type": "city_buy", "city_id": 301, "item_id": item_id}
        if item_type is not None:
            cmd["item_type"] = item_type
        success, code, _ = _run(runtime, cmd)
        assert (success, code) == (True, None)
        calls = c.requests(runtime, "CityManager.RequestOperation")
        assert len(calls) == 1
        op, params = calls[0]
        assert op == int(runtime.eval("CityOperationTypes.PURCHASE"))
        param_key = runtime.eval(f"CityOperationTypes.{expected_param_key}")
        assert params == {param_key: item_id}
        c.frame(runtime)
        assert c.dropped(runtime) == 0
        assert runtime.eval("#Civ6Stub.purchases") == 1
        assert runtime.eval("Civ6Stub.purchases[1].cityId") == 301

    def test_fallback_purchase_production_and_failure(self) -> None:
        runtime = c.load("CityManager = nil")
        success, code, msg = _run(
            runtime,
            {"action_type": "city_buy", "city_id": 301, "item_id": 6111, "item_type": "unit"},
        )
        assert (success, code) == (True, None)
        assert "fallback" in msg.lower()
        assert runtime.eval("#Civ6Stub.fallbackPurchases") == 1
        assert runtime.eval("Civ6Stub.fallbackPurchases[1].itemId") == 6111

        fail_runtime = c.load("CityManager = nil\nCiv6Stub.fallbackPurchaseResult = false")
        assert _run(
            fail_runtime,
            {"action_type": "city_buy", "city_id": 301, "item_id": 6111, "item_type": "unit"},
        )[:2] == (False, "EXECUTION_FAILED")


class TestCityCitizen:
    """Tests for ``executeCityCitizen`` void-API caveat and failure modes."""

    def test_validates_missing_args_unknown_city_and_unavailable_citizens(self) -> None:
        runtime = c.load()
        assert _run(runtime, {"action_type": "city_citizen", "item_id": 4})[:2] == (
            False,
            "INVALID_COMMAND",
        )
        assert _run(runtime, {"action_type": "city_citizen", "city_id": 301})[:2] == (
            False,
            "INVALID_COMMAND",
        )
        success, code, msg = _run(
            runtime, {"action_type": "city_citizen", "city_id": 999, "item_id": 4}
        )
        assert (success, code) == (False, "INVALID_COMMAND")
        assert "not found" in msg.lower()
        no_cit = c.load("Civ6Stub.citizensUnavailable = true")
        assert _run(no_cit, {"action_type": "city_citizen", "city_id": 301, "item_id": 4})[:2] == (
            False,
            "EXECUTION_FAILED",
        )

    @pytest.mark.parametrize(
        ("metadata", "expected_lock"),
        [(None, False), ({"lock": True}, True), ({"lock": False}, False)],
        ids=["default_unlock", "explicit_lock", "explicit_unlock"],
    )
    def test_calls_set_citizen_locked_and_notes_void_contract(
        self, metadata: dict[str, Any] | None, expected_lock: bool
    ) -> None:
        runtime = c.load()
        cmd: dict[str, Any] = {"action_type": "city_citizen", "city_id": 301, "item_id": 12}
        if metadata is not None:
            cmd["metadata"] = metadata
        success, code, message = _run(runtime, cmd)
        assert (success, code) == (True, None)
        assert "void api" in message.lower()
        assert runtime.eval("#Civ6Stub.citizenLocks") == 1
        assert runtime.eval("Civ6Stub.citizenLocks[1].cityId") == 301
        assert runtime.eval("Civ6Stub.citizenLocks[1].plotIndex") == 12
        assert runtime.eval("Civ6Stub.citizenLocks[1].lock") is expected_lock

    def test_pcall_error_returns_execution_failed(self) -> None:
        runtime = c.load("Civ6Stub.citizenLockRaises = true")
        assert _run(
            runtime,
            {
                "action_type": "city_citizen",
                "city_id": 301,
                "item_id": 12,
                "metadata": {"lock": True},
            },
        )[:2] == (False, "EXECUTION_FAILED")
