"""Built-in rule sets: pinned by content hash, loaded through importlib.resources."""

from __future__ import annotations

from importlib import resources

import pytest

from alpha_zero.rules import builtin
from alpha_zero.rules.builtin import (
    BUILTIN,
    BuiltinRuleset,
    RulesetHashMismatchError,
    builtin_ids,
    frozen_pins,
    load_builtin,
    read_builtin_text,
)
from alpha_zero.rules.loader import content_sha256, parse_ruleset
from alpha_zero.rules.validate import RulesetValidationError

HEX64 = "0" * 64


def suffix_problems(table: dict[str, BuiltinRuleset]) -> list[str]:
    return [p for name, e in table.items() if (p := builtin.entry_problem(name, e))]


def test_builtin_table_obeys_the_suffix_rule() -> None:
    assert suffix_problems(BUILTIN) == []


def test_suffix_rule_checker_catches_both_directions() -> None:
    draft = builtin.STATUS_DRAFT
    frozen = builtin.STATUS_FROZEN
    assert suffix_problems({"a-v1": BuiltinRuleset("a.yaml", HEX64, draft)}) != []
    assert suffix_problems({"a-v1-draft": BuiltinRuleset("a.yaml", HEX64, frozen)}) != []
    assert suffix_problems({"a-v1": BuiltinRuleset("a.yaml", HEX64, "wip")}) != []
    assert suffix_problems({"a-v1-draft": BuiltinRuleset("a.yaml", HEX64, draft)}) == []
    assert suffix_problems({"a-v1": BuiltinRuleset("a.yaml", HEX64, frozen)}) == []


def test_science_v1_draft_is_registered_as_a_draft() -> None:
    assert builtin_ids() == ("science-v1-draft",)
    assert BUILTIN["science-v1-draft"].status == "draft"
    assert "science-v1" not in BUILTIN  # reserved for the calibrated tree


def test_load_builtin_matches_its_pin_and_validates() -> None:
    spec = load_builtin("science-v1-draft")
    assert spec.id == "science-v1-draft"
    assert content_sha256(spec) == BUILTIN["science-v1-draft"].sha256
    assert len(spec.techs) == 12


def test_unknown_builtin_is_a_key_error() -> None:
    with pytest.raises(KeyError):
        load_builtin("science-v1")
    with pytest.raises(KeyError):
        read_builtin_text("nope")


def test_editing_the_data_without_a_repin_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    text = read_builtin_text("science-v1-draft")
    edited = text.replace("cost: 180", "cost: 181", 1)
    assert edited != text
    monkeypatch.setattr(builtin, "read_builtin_text", lambda ruleset_id: edited)
    with pytest.raises(RulesetHashMismatchError, match="science-v1-draft"):
        load_builtin("science-v1-draft")


def test_repinning_makes_the_edited_data_load(monkeypatch: pytest.MonkeyPatch) -> None:
    text = read_builtin_text("science-v1-draft")
    edited = text.replace("cost: 180", "cost: 181", 1)
    new_pin = content_sha256(parse_ruleset(edited))
    entry = BuiltinRuleset("science_v1_draft.yaml", new_pin, "draft")
    monkeypatch.setattr(builtin, "read_builtin_text", lambda ruleset_id: edited)
    monkeypatch.setitem(BUILTIN, "science-v1-draft", entry)
    assert load_builtin("science-v1-draft").tech("scientific_theory").cost == 181


def test_a_file_declaring_another_id_than_the_table_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = read_builtin_text("science-v1-draft")
    renamed = text.replace("id: science-v1-draft", "id: science-v1", 1)
    assert renamed != text
    monkeypatch.setattr(builtin, "read_builtin_text", lambda ruleset_id: renamed)
    with pytest.raises(RulesetHashMismatchError, match="declares id"):
        load_builtin("science-v1-draft")


def test_a_pinned_but_invalid_ruleset_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = read_builtin_text("science-v1-draft").replace(
        "prerequisites: [agriculture]", "prerequisites: [ghost]", 1
    )
    pin = content_sha256(parse_ruleset(text))
    monkeypatch.setattr(builtin, "read_builtin_text", lambda ruleset_id: text)
    monkeypatch.setitem(BUILTIN, "science-v1-draft", BuiltinRuleset("x.yaml", pin, "draft"))
    with pytest.raises(RulesetValidationError, match="unresolved-prerequisite"):
        load_builtin("science-v1-draft")


def test_frozen_pins_lists_frozen_entries_only(monkeypatch: pytest.MonkeyPatch) -> None:
    assert frozen_pins() == {}
    table = {
        "a-v1": BuiltinRuleset("a.yaml", "a" * 64, "frozen"),
        "b-v1-draft": BuiltinRuleset("b.yaml", "b" * 64, "draft"),
    }
    monkeypatch.setattr(builtin, "BUILTIN", table)
    assert frozen_pins() == {"a-v1": "a" * 64}
    assert builtin.builtin_ids() == ("a-v1", "b-v1-draft")


def test_every_entry_has_a_resource_file_and_every_file_an_entry() -> None:
    data = resources.files("alpha_zero.rules").joinpath("data")
    on_disk = {p.name for p in data.iterdir() if p.name.endswith(".yaml")}
    referenced = {entry.resource for entry in BUILTIN.values()}
    for entry in BUILTIN.values():
        assert data.joinpath(entry.resource).is_file(), entry.resource
        assert len(entry.sha256) == 64 and entry.sha256 == entry.sha256.lower()
    assert on_disk, "no yaml data shipped"
    assert on_disk == referenced, (
        f"unreferenced: {on_disk - referenced}; missing: {referenced - on_disk}"
    )


def test_the_pinned_hash_is_literal_and_stable() -> None:
    # A literal, so an edit to data plus a silent re-pin needs two deliberate changes.
    assert content_sha256(load_builtin("science-v1-draft")) == (
        "0c336c21e209a76a63fd9b70d2c2e8dff18fb0d326562e8b362989e159d18c3c"
    )


@pytest.mark.parametrize(
    ("name", "status"),
    [("a-v1", "draft"), ("a-v1-draft", "frozen"), ("a-v1", "wip"), ("a-v1-draft", "")],
)
def test_load_builtin_refuses_an_entry_that_breaks_the_status_rule(
    name: str, status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = read_builtin_text("science-v1-draft")
    pin = content_sha256(parse_ruleset(text))
    monkeypatch.setitem(BUILTIN, name, BuiltinRuleset("science_v1_draft.yaml", pin, status))
    with pytest.raises(ValueError, match=name):
        load_builtin(name)


def test_entry_problem_messages_name_the_id_and_status() -> None:
    unknown = builtin.entry_problem("a-v1", BuiltinRuleset("x", HEX64, "wip"))
    assert unknown is not None and "'a-v1'" in unknown and "'wip'" in unknown
    wrong = builtin.entry_problem("a-v1", BuiltinRuleset("x", HEX64, "draft"))
    assert wrong is not None and "'a-v1'" in wrong and "-draft" in wrong
    assert builtin.entry_problem("a-v1-draft", BuiltinRuleset("x", HEX64, "draft")) is None
