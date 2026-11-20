"""Queries over the built-in tree, and the index registry."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from alpha_zero.rules import available, build_registry, prerequisite_closure, research_cost
from alpha_zero.rules.builtin import load_builtin
from alpha_zero.rules.loader import content_sha256, parse_ruleset
from alpha_zero.rules.spec import RulesetSpec, TechSpec
from tests.unit.test_rules._fixtures import BASE, mutate

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def science() -> RulesetSpec:
    return load_builtin("science-v1-draft")


def test_available_at_the_start_is_the_prerequisite_free_techs_in_order(
    science: RulesetSpec,
) -> None:
    free = tuple(t.id for t in science.techs if not t.prerequisites)
    assert free == ("agriculture", "mining", "writing")
    assert available(science, ()) == free


def test_available_after_researching(science: RulesetSpec) -> None:
    done = ["agriculture", "mining", "writing", "mathematics", "construction"]
    assert available(science, done) == ("irrigation", "bronze_working", "education", "engineering")
    assert available(science, set(done)) == available(science, done)


def test_available_needs_every_prerequisite(science: RulesetSpec) -> None:
    got = available(science, ["mining", "writing", "mathematics", "construction"])
    assert "engineering" in got  # both prerequisites done
    assert "steel" not in got
    assert "agriculture" in got
    assert "mining" not in got  # already researched


def test_available_is_empty_when_everything_is_done(science: RulesetSpec) -> None:
    assert available(science, [t.id for t in science.techs]) == ()


def test_prerequisite_closure_of_the_capstone(science: RulesetSpec) -> None:
    closure = prerequisite_closure(science, "scientific_theory")
    assert "scientific_theory" not in closure
    total = sum(science.tech(n).cost for n in closure) + science.tech("scientific_theory").cost
    assert total == 580
    assert len(closure) == len(set(closure)) == 7


def test_all_techs_sum_to_780(science: RulesetSpec) -> None:
    assert sum(t.cost for t in science.techs) == 780


def test_prerequisite_closure_is_in_declared_order(science: RulesetSpec) -> None:
    assert prerequisite_closure(science, "steel") == (
        "mining",
        "writing",
        "bronze_working",
        "mathematics",
        "construction",
        "engineering",
    )
    assert prerequisite_closure(science, "agriculture") == ()


def test_prerequisite_closure_unknown_tech_raises(science: RulesetSpec) -> None:
    with pytest.raises(KeyError):
        prerequisite_closure(science, "ghost")


def _boosted(cost: int, pct: int) -> RulesetSpec:
    return RulesetSpec(id="b", techs=(TechSpec("t", 1, cost, boost_percent=pct),))


def test_research_cost_boost_rounds_the_discount_down() -> None:
    assert research_cost(_boosted(21, 40), "t", boosted=True) == 13
    assert research_cost(_boosted(20, 40), "t", boosted=True) == 12
    assert research_cost(_boosted(21, 40), "t") == 21
    assert research_cost(_boosted(7, 50), "t", boosted=True) == 4  # 7 - 3
    assert research_cost(_boosted(10, 100), "t", boosted=True) == 0


def test_research_cost_boost_on_a_tech_without_one_raises(science: RulesetSpec) -> None:
    with pytest.raises(ValueError, match="agriculture"):
        research_cost(science, "agriculture", boosted=True)
    assert research_cost(science, "agriculture") == 20


@pytest.mark.parametrize("boosted", [False, True])
def test_research_cost_unknown_tech_raises_key_error(science: RulesetSpec, boosted: bool) -> None:
    with pytest.raises(KeyError):
        research_cost(science, "ghost", boosted=boosted)


# ---- registry -------------------------------------------------------------------


def test_registry_indices_follow_declared_order(science: RulesetSpec) -> None:
    reg = build_registry(science)
    assert reg.ruleset_id == "science-v1-draft"
    assert list(reg.tech_index.items()) == [(t.id, i) for i, t in enumerate(science.techs)]
    assert reg.tech_names == tuple(t.id for t in science.techs)
    assert dict(reg.unit_index) == {"swordsman": 0, "knight": 1}
    assert reg.building_index["observatory"] == 4


def test_registry_maps_are_read_only(science: RulesetSpec) -> None:
    reg = build_registry(science)
    for mapping in (reg.tech_index, reg.unit_index, reg.building_index):
        with pytest.raises(TypeError):
            mapping["x"] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        reg.fingerprint = "x"  # type: ignore[misc]


def test_fingerprint_ignores_costs_but_not_order_or_names() -> None:
    base = parse_ruleset(BASE)
    cost_changed = parse_ruleset(mutate("cost: 20", "cost: 21"))
    assert content_sha256(cost_changed) != content_sha256(base)
    assert build_registry(cost_changed).fingerprint == build_registry(base).fingerprint
    reordered = RulesetSpec(id=base.id, techs=tuple(reversed(base.techs)))
    assert content_sha256(reordered) != content_sha256(RulesetSpec(id=base.id, techs=base.techs))
    assert (
        build_registry(reordered).fingerprint
        != build_registry(RulesetSpec(id=base.id, techs=base.techs)).fingerprint
    )
    renamed = parse_ruleset(mutate("id: beta", "id: gamma"))
    assert build_registry(renamed).fingerprint != build_registry(base).fingerprint


def test_fingerprint_moves_when_a_unit_is_reordered(science: RulesetSpec) -> None:
    swapped = RulesetSpec(
        id=science.id,
        techs=science.techs,
        units=tuple(reversed(science.units)),
        buildings=science.buildings,
    )
    assert build_registry(swapped).fingerprint != build_registry(science).fingerprint


_CHILD = (
    "from alpha_zero.rules import build_registry, content_sha256, load_builtin\n"
    "s = load_builtin('science-v1-draft')\n"
    "print(build_registry(s).fingerprint, content_sha256(s))\n"
)


def _in_child(hashseed: str) -> list[str]:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONHASHSEED=hashseed)
    done = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=True,
        timeout=60,
    )
    return done.stdout.split()


def test_fingerprint_is_equal_across_separate_processes(science: RulesetSpec) -> None:
    first, second = _in_child("1"), _in_child("987")
    assert first == second
    assert first[0] == build_registry(science).fingerprint
    assert first[1] == content_sha256(science)
    assert len(first[0]) == 64


def test_registries_are_hashable_and_equal_specs_hash_equal(science: RulesetSpec) -> None:
    a, b = build_registry(science), build_registry(load_builtin("science-v1-draft"))
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1
    other = build_registry(parse_ruleset(BASE))
    assert other != a and len({a, other}) == 2
