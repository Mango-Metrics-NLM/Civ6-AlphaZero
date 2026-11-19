"""Strict loading, and the content identity of a rule set."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_zero.rules.builtin import read_builtin_text
from alpha_zero.rules.loader import (
    RulesetLoadError,
    canonical_json,
    content_sha256,
    load_ruleset,
    parse_ruleset,
)
from tests.unit.test_rules._fixtures import BASE, mutate, tiny

# (id, old, new, substrings the message must contain)
BAD_CASES = [
    ("unknown-key", "    cost: 10\n", "    cost: 10\n    colour: red\n", ["$.techs[0]", "colour"]),
    (
        "duplicate-key",
        "    cost: 10\n",
        "    cost: 10\n    cost: 11\n",
        ["duplicate key", "'cost'", "line 7"],
    ),
    ("float", "    cost: 10\n", "    cost: 10.5\n", ["$.techs[0].cost", "float"]),
    ("bool", "    cost: 10\n", "    cost: yes\n", ["$.techs[0].cost", "bool"]),
    (
        "negative-cost",
        "    cost: 10\n",
        "    cost: -5\n",
        ["$.techs[0].cost", "non-negative", "-5"],
    ),
    ("missing-key", "    era: 1\n", "", ["$.techs[0]", "missing key", "'era'"]),
    ("octal-looking", "    cost: 10\n", "    cost: 020\n", ["'020'", "line 6", "plain decimal"]),
    ("hex", "    cost: 10\n", "    cost: 0x14\n", ["'0x14'", "line 6", "plain decimal"]),
    ("underscore", "    cost: 10\n", "    cost: 1_000\n", ["'1_000'", "line 6", "plain decimal"]),
    ("anchor", "    cost: 10\n", "    cost: &c 10\n", ["anchor", "&c", "line 6"]),
    ("alias", "prerequisites: [alpha]", "prerequisites: [*x]", ["alias", "*x", "line 10"]),
    (
        "merge-key",
        "  - {id: sword, cost: 5, strength: 3, moves: 2}",
        "  - {<<: {id: sword}, cost: 5, strength: 3, moves: 2}",
        ["merge key", "line 12"],
    ),
    ("explicit-tag", "    cost: 10\n", "    cost: !!int 10\n", ["explicit tag", "line 6"]),
    ("bad-item-id", "  - id: alpha\n", "  - id: Alpha\n", ["$.techs[0].id", "'Alpha'"]),
    ("digit-item-id", "  - id: alpha\n", "  - id: 1alpha\n", ["$.techs[0].id", "'1alpha'"]),
    ("bad-ruleset-id", "id: tiny-test", "id: Tiny_Test", ["$.id", "'Tiny_Test'"]),
    ("ruleset-id-path", "id: tiny-test", "id: tiny.test", ["$.id", "'tiny.test'"]),
    ("schema-version", "schema_version: 1", "schema_version: 2", ["$.schema_version", "2"]),
    (
        "bad-terrain-ref",
        "      - {yield: food, amount: 1}",
        "      - {yield: food, amount: 1, per: worked_terrain, ref: swamp}",
        ["$.buildings[0].modifiers[0].ref", "terrain", "'swamp'"],
    ),
    (
        "city-with-ref",
        "      - {yield: food, amount: 1}",
        "      - {yield: food, amount: 1, ref: plains}",
        ["$.buildings[0].modifiers[0].ref"],
    ),
    (
        "building-without-ref",
        "      - {yield: food, amount: 1}",
        "      - {yield: food, amount: 1, per: building}",
        ["$.buildings[0].modifiers[0].ref"],
    ),
    (
        "bad-yield",
        "      - {yield: food, amount: 1}",
        "      - {yield: gold, amount: 1}",
        ["$.buildings[0].modifiers[0].yield", "'gold'"],
    ),
    (
        "bad-scope",
        "      - {yield: food, amount: 1}",
        "      - {yield: food, amount: 1, per: empire}",
        ["$.buildings[0].modifiers[0].per", "'empire'"],
    ),
    (
        "boost-over-100",
        "    cost: 10\n",
        "    cost: 10\n    boost_percent: 101\n",
        ["boost_percent", "101"],
    ),
    ("null-cost", "    cost: 10\n", "    cost:\n", ["$.techs[0].cost", "NoneType"]),
    ("unit-missing", "strength: 3, ", "", ["$.units[0]", "'strength'"]),
    ("prereq-bad-name", "prerequisites: [alpha]", "prerequisites: [Alpha]", ["prerequisites[0]"]),
    ("unknown-top-key", "units:\n", "extra: 1\nunits:\n", ["$", "'extra'"]),
    ("capstone-bad-id", "capstone_tech: beta", "capstone_tech: Beta", ["$.victory.capstone_tech"]),
    ("bool-key-text", "id: tiny-test", "id: true", ["$.id", "bool"]),
]


@pytest.mark.parametrize(
    ("old", "new", "needles"),
    [pytest.param(o, n, s, id=name) for name, o, n, s in BAD_CASES],
)
def test_malformed_file_is_refused_and_named(old: str, new: str, needles: list[str]) -> None:
    text = mutate(old, new)
    with pytest.raises(RulesetLoadError) as excinfo:
        parse_ruleset(text)
    message = str(excinfo.value)
    for needle in needles:
        assert needle in message, f"{needle!r} missing from {message!r}"


@pytest.mark.parametrize(
    "text",
    ["", "- a\n- b\n", "just a string", "key: [unclosed", "a: 1\n---\nb: 2\n", "\tid: x\n"],
)
def test_non_rulesets_are_refused_as_load_errors(text: str) -> None:
    with pytest.raises(RulesetLoadError):
        parse_ruleset(text)


def test_unhashable_mapping_key_is_a_load_error_not_a_crash() -> None:
    # `? [a]` makes a list the key; the loader must refuse it as a RulesetLoadError.
    text = mutate("units:\n", "? [a]\n: 1\nunits:\n")
    with pytest.raises(RulesetLoadError):
        parse_ruleset(text)


def test_valid_base_loads_with_expected_values() -> None:
    spec = tiny()
    assert spec.id == "tiny-test"
    assert [t.id for t in spec.techs] == ["alpha", "beta"]
    assert spec.techs[1].prerequisites == ("alpha",)
    assert spec.units[0].strength == 3
    assert spec.buildings[0].modifiers[0].yield_ == "food"
    assert spec.buildings[0].modifiers[0].per == "city"
    assert spec.victory.capstone_tech == "beta"


def test_optional_sections_may_be_absent_or_null() -> None:
    spec = parse_ruleset("schema_version: 1\nid: empty\ntechs:\nvictory:\n")
    assert spec.techs == () and spec.units == () and spec.victory.capstone_tech is None


def test_valid_terrain_modifier_loads() -> None:
    text = mutate(
        "      - {yield: food, amount: 1}",
        "      - {yield: food, amount: 1, per: worked_terrain, ref: hills}",
    )
    modifier = parse_ruleset(text).buildings[0].modifiers[0]
    assert (modifier.per, modifier.ref) == ("worked_terrain", "hills")


def test_load_ruleset_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    path.write_text(BASE, encoding="utf-8")
    assert load_ruleset(path) == tiny()


# ---- identity -------------------------------------------------------------------


def test_hash_is_hex_sha256_and_canonical_json_is_golden() -> None:
    spec = parse_ruleset("schema_version: 1\nid: g\ntechs:\n  - {id: alpha, era: 1, cost: 10}\n")
    assert canonical_json(spec) == '{"techs":[{"cost":10,"era":1,"id":"alpha"}]}'
    assert len(content_sha256(spec)) == 64


def test_hash_ignores_line_endings() -> None:
    assert content_sha256(parse_ruleset(BASE.replace("\n", "\r\n"))) == content_sha256(tiny())


def test_hash_ignores_the_builtin_files_line_endings() -> None:
    text = read_builtin_text("science-v1-draft")
    crlf = text.replace("\r\n", "\n").replace("\n", "\r\n")
    assert content_sha256(parse_ruleset(crlf)) == content_sha256(parse_ruleset(text))


def test_hash_ignores_key_order() -> None:
    reordered = """\
victory:
  capstone_tech: beta
buildings:
  - modifiers:
      - {amount: 1, yield: food}
    cost: 5
    id: hut
units:
  - {moves: 2, strength: 3, cost: 5, id: sword}
techs:
  - cost: 10
    id: alpha
    era: 1
  - {prerequisites: [alpha], cost: 20, era: 2, id: beta}
id: tiny-test
schema_version: 1
"""
    assert content_sha256(parse_ruleset(reordered)) == content_sha256(tiny())


def test_hash_ignores_a_default_written_out() -> None:
    explicit = mutate(
        "    cost: 10\n", "    cost: 10\n    boost_percent: 0\n    prerequisites: []\n"
    )
    explicit = mutate(
        "      - {yield: food, amount: 1}",
        "      - {yield: food, amount: 1, per: city, ref: null}",
        explicit,
    )
    assert content_sha256(parse_ruleset(explicit)) == content_sha256(tiny())
    assert canonical_json(parse_ruleset(explicit)) == canonical_json(tiny())


def test_hash_ignores_default_victory_and_schema_version_spelling() -> None:
    a = parse_ruleset("schema_version: 1\nid: a\ntechs:\n  - {id: x, era: 1, cost: 1}\n")
    b = parse_ruleset(
        "schema_version: 1\nid: a\ntechs:\n  - {id: x, era: 1, cost: 1}\n"
        "victory: {capstone_tech: null}\nunits: []\nbuildings: []\n"
    )
    assert content_sha256(a) == content_sha256(b)


def test_hash_ignores_only_the_id() -> None:
    other = mutate("id: tiny-test", "id: another-name")
    assert content_sha256(parse_ruleset(other)) == content_sha256(tiny())
    assert "tiny-test" not in canonical_json(tiny())


def test_hash_moves_with_one_tech_cost() -> None:
    assert content_sha256(parse_ruleset(mutate("cost: 20", "cost: 21"))) != content_sha256(tiny())


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("strength: 3", "strength: 4"),
        ("moves: 2", "moves: 3"),
        ("    cost: 5\n    modifiers:", "    cost: 6\n    modifiers:"),
        ("amount: 1}", "amount: 2}"),
        ("yield: food", "yield: production"),
        ("capstone_tech: beta", "capstone_tech: alpha"),
        ("    era: 1\n", "    era: 2\n"),
        ("    cost: 10\n", "    cost: 10\n    boost_percent: 50\n"),
    ],
)
def test_hash_moves_with_any_value(old: str, new: str) -> None:
    assert content_sha256(parse_ruleset(mutate(old, new))) != content_sha256(tiny())


def test_hash_moves_when_two_techs_are_reordered() -> None:
    a = "schema_version: 1\nid: a\ntechs:\n  - {id: x, era: 1, cost: 1}\n  - {id: y, era: 1, cost: 1}\n"
    b = "schema_version: 1\nid: a\ntechs:\n  - {id: y, era: 1, cost: 1}\n  - {id: x, era: 1, cost: 1}\n"
    assert content_sha256(parse_ruleset(a)) != content_sha256(parse_ruleset(b))


@pytest.mark.parametrize("section", ["techs", "units", "buildings"])
def test_a_section_that_is_not_a_list_is_refused(section: str) -> None:
    with pytest.raises(RulesetLoadError, match=rf"\$\.{section}: expected a list"):
        parse_ruleset(f"schema_version: 1\nid: x\n{section}: 5\n")


@pytest.mark.parametrize("key", ["[a, b]", "{x: 1}"])
def test_non_scalar_key_message_names_the_line(key: str) -> None:
    text = mutate("units:\n", f"? {key}\n: 1\nunits:\n")
    with pytest.raises(RulesetLoadError, match=r"line 11.*not a scalar"):
        parse_ruleset(text)


def test_non_utf8_file_is_a_load_error_naming_the_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"schema_version: 1\nid: \xff\xfe\n")
    with pytest.raises(RulesetLoadError, match=r"bad\.yaml.*UTF-8"):
        load_ruleset(path)


def test_a_missing_file_stays_an_os_error(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_ruleset(tmp_path / "nope.yaml")
