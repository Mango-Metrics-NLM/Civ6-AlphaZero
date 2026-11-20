"""Structural rules: each fixture violates exactly one, and each rule is load-bearing."""

from __future__ import annotations

import types

import pytest

import alpha_zero.rules as rules_pkg
import alpha_zero.rules.validate as validate_module
from alpha_zero.rules.spec import (
    BuildingSpec,
    ModifierSpec,
    RulesetSpec,
    TechSpec,
    UnitSpec,
    VictorySpec,
)
from alpha_zero.rules.validate import (
    RULES,
    Issue,
    RulesetValidationError,
    ensure_valid,
    validate,
)


def tech(name: str, *prereqs: str, cost: int = 10, **kw: object) -> TechSpec:
    return TechSpec(id=name, era=1, cost=cost, prerequisites=tuple(prereqs), **kw)  # type: ignore[arg-type]


def ruleset(*techs: TechSpec, **kw: object) -> RulesetSpec:
    return RulesetSpec(id="t", techs=tuple(techs), **kw)  # type: ignore[arg-type]


FIXTURES: dict[str, RulesetSpec] = {
    "duplicate-id": ruleset(tech("a"), tech("a")),
    "unresolved-prerequisite": ruleset(tech("a", "ghost")),
    "prerequisite-cycle": ruleset(tech("a", "b"), tech("b", "a")),
    "non-positive-cost": ruleset(tech("a", cost=0)),
    "unresolved-unlock": ruleset(tech("a", unlocks_units=("ghost",))),
    "unreachable-capstone": ruleset(tech("a"), victory=VictorySpec(capstone_tech="ghost")),
}

# A fixture function name per rule, for the negative control.
RULE_FUNCTIONS = {
    "duplicate-id": "_duplicate_ids",
    "unresolved-prerequisite": "_unresolved_prerequisites",
    "prerequisite-cycle": "_cycles",
    "non-positive-cost": "_non_positive_costs",
    "unresolved-unlock": "_unresolved_unlocks",
    "unreachable-capstone": "_unreachable_capstone",
}


def rule_names(spec: RulesetSpec) -> set[str]:
    return {issue.rule for issue in validate(spec)}


def test_every_rule_has_a_fixture_and_a_function() -> None:
    assert set(FIXTURES) == set(RULES) == set(RULE_FUNCTIONS)
    assert len(RULES) == 6
    for name in RULE_FUNCTIONS.values():
        assert callable(getattr(validate_module, name))


@pytest.mark.parametrize("rule", RULES)
def test_fixture_violates_only_its_rule(rule: str) -> None:
    assert rule_names(FIXTURES[rule]) == {rule}


@pytest.mark.parametrize("rule", RULES)
def test_negative_control_disabling_the_rule_lets_its_fixture_pass(
    rule: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert validate(FIXTURES[rule]), "fixture must fail before the rule is disabled"
    monkeypatch.setattr(validate_module, RULE_FUNCTIONS[rule], lambda spec: [])
    assert validate_module.validate(FIXTURES[rule]) == ()
    # and the other rules' fixtures still fail: the disabled function was the only guard
    for other in RULES:
        if other != rule:
            assert rule_names(FIXTURES[other]) == {other}


def test_valid_spec_has_no_issues_and_ensure_valid_returns_it() -> None:
    spec = ruleset(tech("a"), tech("b", "a"), victory=VictorySpec(capstone_tech="b"))
    assert validate(spec) == ()
    assert ensure_valid(spec) is spec


def test_ensure_valid_raises_with_every_issue() -> None:
    spec = ruleset(tech("a", "ghost", cost=0))
    with pytest.raises(RulesetValidationError) as excinfo:
        ensure_valid(spec)
    assert excinfo.value.ruleset_id == "t"
    assert {i.rule for i in excinfo.value.issues} == {
        "unresolved-prerequisite",
        "non-positive-cost",
    }
    assert "[non-positive-cost]" in str(excinfo.value)


def test_issue_str_names_its_rule() -> None:
    assert str(Issue("duplicate-id", "x")) == "[duplicate-id] x"


def test_cycle_names_both_members() -> None:
    (issue,) = validate(FIXTURES["prerequisite-cycle"])
    assert "'a'" in issue.message and "'b'" in issue.message


def test_self_loop_is_a_cycle() -> None:
    (issue,) = validate(ruleset(tech("a", "a")))
    assert issue.rule == "prerequisite-cycle" and "'a'" in issue.message


def test_unresolved_prerequisite_is_not_reported_as_a_cycle() -> None:
    issues = validate(ruleset(tech("a", "ghost"), tech("b", "a")))
    assert {i.rule for i in issues} == {"unresolved-prerequisite"}
    assert "'ghost'" in issues[0].message


def test_tech_downstream_of_a_cycle_is_not_named() -> None:
    spec = ruleset(tech("a", "b"), tech("b", "a"), tech("c", "a"), tech("d", "c"))
    (issue,) = validate(spec)
    assert issue.rule == "prerequisite-cycle"
    assert "'a'" in issue.message and "'b'" in issue.message
    assert "'c'" not in issue.message and "'d'" not in issue.message


def test_tech_upstream_of_a_cycle_is_not_named() -> None:
    spec = ruleset(tech("root"), tech("a", "b", "root"), tech("b", "a"))
    (issue,) = validate(spec)
    assert "'root'" not in issue.message


def test_tech_on_a_path_between_two_cycles_is_not_named() -> None:
    # x is on no cycle: it merely sits between cycle {a, b} and cycle {c, d}.
    spec = ruleset(
        tech("a", "b"),
        tech("b", "a"),
        tech("x", "a"),
        tech("c", "d", "x"),
        tech("d", "c"),
    )
    (issue,) = validate(spec)
    assert issue.rule == "prerequisite-cycle"
    assert "'x'" not in issue.message, issue.message


def test_capstone_behind_a_cycle_is_unreachable() -> None:
    spec = ruleset(tech("a", "b"), tech("b", "a"), tech("cap", "a"), victory=VictorySpec("cap"))
    assert rule_names(spec) == {"prerequisite-cycle", "unreachable-capstone"}


def test_negative_cost_and_unit_building_costs_are_flagged() -> None:
    spec = RulesetSpec(
        id="t",
        techs=(tech("a"),),
        units=(UnitSpec("u", 0, 1, 1),),
        buildings=(BuildingSpec("b", -1),),
    )
    messages = [i.message for i in validate(spec)]
    assert any("unit 'u'" in m for m in messages) and any("building 'b'" in m for m in messages)


def test_duplicate_unit_and_building_ids_are_flagged() -> None:
    spec = RulesetSpec(
        id="t",
        units=(UnitSpec("u", 1, 1, 1), UnitSpec("u", 1, 1, 1)),
        buildings=(BuildingSpec("b", 1), BuildingSpec("b", 1)),
    )
    messages = [i.message for i in validate(spec)]
    assert any("unit id 'u'" in m for m in messages)
    assert any("building id 'b'" in m for m in messages)


def test_unlock_and_modifier_refs_must_resolve() -> None:
    bad_mod = ModifierSpec("science", 1, per="building", ref="ghost")
    spec = RulesetSpec(
        id="t",
        techs=(tech("a", unlocks_buildings=("nope",), modifiers=(bad_mod,)),),
        buildings=(BuildingSpec("b", 1, modifiers=(bad_mod,)),),
    )
    messages = [i.message for i in validate(spec)]
    assert {i.rule for i in validate(spec)} == {"unresolved-unlock"}
    assert any("unknown building 'nope'" in m for m in messages)
    assert any("tech 'a'" in m and "'ghost'" in m for m in messages)
    assert any("building 'b'" in m and "'ghost'" in m for m in messages)


def test_resolving_modifier_ref_is_clean() -> None:
    mod = ModifierSpec("science", 1, per="building", ref="b")
    spec = RulesetSpec(
        id="t", techs=(tech("a", modifiers=(mod,)),), buildings=(BuildingSpec("b", 1),)
    )
    assert validate(spec) == ()


def test_package_attribute_validate_is_the_module_not_the_function() -> None:
    assert isinstance(rules_pkg.validate, types.ModuleType)
    assert rules_pkg.validate is validate_module
    assert validate_module.validate is validate


def test_three_cycle_names_exactly_its_members() -> None:
    spec = ruleset(tech("a", "c"), tech("b", "a"), tech("c", "b"), tech("d", "c"))
    (issue,) = validate(spec)
    assert issue.rule == "prerequisite-cycle"
    for name in ("'a'", "'b'", "'c'"):
        assert name in issue.message
    assert "'d'" not in issue.message


def test_self_loop_beside_an_acyclic_tech_names_only_the_loop() -> None:
    (issue,) = validate(ruleset(tech("ok"), tech("a", "a", "ok")))
    assert "'a'" in issue.message and "'ok'" not in issue.message
