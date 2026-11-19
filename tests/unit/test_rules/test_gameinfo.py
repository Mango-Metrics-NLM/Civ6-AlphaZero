"""``alpha_zero.rules.gameinfo`` and ``import-gameinfo``: conversion and output paths (GM-K2)."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.rules import gameinfo
from alpha_zero.rules.__main__ import main
from alpha_zero.rules.gameinfo import (
    GameInfoError,
    RefusedPathError,
    checkout_root,
    convert,
    import_gameinfo,
    resolve_output,
)
from alpha_zero.rules.loader import parse_ruleset
from alpha_zero.rules.validate import RulesetValidationError, validate

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "gameinfo" / "synthetic_export.json"


def export() -> dict[str, Any]:
    """A fresh, mutable copy of the synthetic export."""
    data: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return copy.deepcopy(data)


def write_export(tmp_path: Path, document: object, name: str = "export.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# ---- convert: the happy path ------------------------------------------------------


def test_the_synthetic_export_converts_to_a_valid_rule_set() -> None:
    summary = convert(export(), "synthetic-test")
    spec = parse_ruleset(summary.text)
    assert validate(spec) == ()
    assert spec.id == "synthetic-test"
    assert (summary.techs, summary.units, summary.buildings) == (5, 2, 2)
    assert summary.ruleset_id == "synthetic-test"


def test_ids_are_lowercased_and_eras_mapped() -> None:
    spec = parse_ruleset(convert(export()).text)
    assert [t.id for t in spec.techs] == [
        "tech_alpha",
        "tech_beta",
        "tech_gamma",
        "tech_delta",
        "tech_epsilon",
    ]
    # ERA_ANCIENT -> 1, ERA_CLASSICAL -> 2, integer Era 3 as given, ERA_MEDIEVAL -> 3
    assert [t.era for t in spec.techs] == [1, 1, 2, 3, 3]
    assert [u.id for u in spec.units] == ["unit_scout", "unit_guard"]
    assert [b.id for b in spec.buildings] == ["building_hall", "building_wall"]
    assert all(t.id == t.id.lower() for t in spec.techs)


def test_prerequisites_costs_and_unit_numbers_are_carried() -> None:
    spec = parse_ruleset(convert(export()).text)
    tech = spec.tech
    assert tech("tech_delta").prerequisites == ("tech_beta", "tech_gamma")
    assert tech("tech_gamma").prerequisites == ("tech_alpha",)
    assert tech("tech_alpha").prerequisites == ()
    assert tech("tech_epsilon").cost == 80
    units = {u.id: u for u in spec.units}
    assert (units["unit_guard"].cost, units["unit_guard"].strength) == (40, 15)
    assert units["unit_guard"].moves == 2
    assert (units["unit_scout"].strength, units["unit_scout"].moves) == (0, 3)


def test_unlocks_are_derived_from_prereq_tech() -> None:
    spec = parse_ruleset(convert(export()).text)
    assert spec.tech("tech_beta").unlocks_buildings == ("building_hall",)
    assert spec.tech("tech_gamma").unlocks_units == ("unit_guard",)
    # rows with a null PrereqTech unlock nothing
    for t in spec.techs:
        assert "building_wall" not in t.unlocks_buildings
        assert "unit_scout" not in t.unlocks_units


def test_non_positive_cost_rows_are_skipped_and_counted() -> None:
    summary = convert(export())
    assert summary.skipped_non_positive_cost == 1
    assert "tech_unused" not in summary.text
    document = export()
    document["Buildings"].append({"BuildingType": "BUILDING_ZERO", "Cost": 0})
    document["Units"].append({"UnitType": "UNIT_NEG", "Cost": -1})
    again = convert(document)
    assert again.skipped_non_positive_cost == 3
    assert (again.techs, again.units, again.buildings) == (5, 2, 2)


def test_civics_are_counted_not_imported() -> None:
    summary = convert(export())
    assert summary.civics_not_imported == 1
    assert "civic" not in summary.text.lower().replace("schema version 1 has no civic", "")
    document = export()
    document["Civics"].append({"CivicType": "CIVIC_TWO", "Cost": 20})
    assert convert(document).civics_not_imported == 2


def test_no_modifier_and_no_victory_is_imported() -> None:
    spec = parse_ruleset(convert(export()).text)
    assert spec.victory.capstone_tech is None
    assert all(not t.modifiers for t in spec.techs)
    assert all(not b.modifiers for b in spec.buildings)


def test_conversion_is_deterministic() -> None:
    assert convert(export()).text == convert(export()).text


# ---- convert: errors name the place -----------------------------------------------


def _error(document: object, ruleset_id: str = "x-test") -> str:
    with pytest.raises(GameInfoError) as excinfo:
        convert(document, ruleset_id)
    return str(excinfo.value)


def test_unknown_table_is_named() -> None:
    document = export()
    document["Feats"] = []
    assert "unknown table 'Feats'" in _error(document)


def test_the_document_must_be_an_object() -> None:
    assert "JSON object" in _error([])


def test_a_table_must_be_a_list() -> None:
    document = export()
    document["Technologies"] = {"TechnologyType": "TECH_X"}
    assert "Technologies: expected a list" in _error(document)


def test_a_row_must_be_an_object() -> None:
    document = export()
    document["Buildings"][1] = "BUILDING_WALL"
    assert "Buildings[1]: expected an object" in _error(document)


@pytest.mark.parametrize(
    ("value", "fragment"),
    [(2.5, "2.5"), (True, "True"), ("free text", "'free text'"), ("", "''"), (["a"], "['a']")],
)
def test_a_non_identifier_non_integer_value_names_table_row_column(
    value: object, fragment: str
) -> None:
    document = export()
    document["Technologies"][2]["Cost"] = value
    message = _error(document)
    assert "Technologies[2].Cost" in message
    assert fragment in message


def test_a_text_value_with_a_space_in_a_name_column_is_refused() -> None:
    document = export()
    document["Units"][0]["UnitType"] = "Unit Scout"
    assert "Units[0].UnitType" in _error(document)


def test_a_float_flagged_even_in_a_row_that_would_be_skipped() -> None:
    document = export()
    document["Technologies"][5]["Cost"] = -1.0
    assert "Technologies[5].Cost" in _error(document)


def test_a_bool_is_not_an_integer_era() -> None:
    document = export()
    document["Technologies"][3]["Era"] = True
    assert "Technologies[3].Era" in _error(document)


@pytest.mark.parametrize(
    ("table", "index", "column"),
    [
        ("Technologies", 0, "Cost"),
        ("Technologies", 1, "TechnologyType"),
        ("Technologies", 2, "EraType"),
        ("TechnologyPrereqs", 0, "Technology"),
        ("TechnologyPrereqs", 1, "PrereqTech"),
        ("Buildings", 0, "Cost"),
        ("Buildings", 1, "BuildingType"),
        ("Units", 0, "Cost"),
        ("Units", 1, "UnitType"),
    ],
)
def test_a_missing_required_column_names_table_row_column(
    table: str, index: int, column: str
) -> None:
    document = export()
    del document[table][index][column]
    message = _error(document)
    assert f"{table}[{index}]" in message
    assert column in message
    assert "missing column" in message


def test_a_null_required_column_is_missing_too() -> None:
    document = export()
    document["Technologies"][0]["Cost"] = None
    assert "Technologies[0]: missing column 'Cost'" in _error(document)


def test_an_unknown_era_names_the_column() -> None:
    document = export()
    document["Technologies"][0]["EraType"] = "ERA_NEVER"
    message = _error(document)
    assert "Technologies[0].EraType" in message and "ERA_NEVER" in message


@pytest.mark.parametrize(
    ("table", "index", "column"),
    [
        ("TechnologyPrereqs", 0, "Technology"),
        ("TechnologyPrereqs", 1, "PrereqTech"),
        ("Buildings", 0, "PrereqTech"),
        ("Units", 1, "PrereqTech"),
    ],
)
def test_every_technology_reference_to_an_unknown_tech_names_table_row_column(
    table: str, index: int, column: str
) -> None:
    document = export()
    document[table][index][column] = "TECH_GHOST"
    message = _error(document)
    assert f"{table}[{index}].{column}" in message
    assert "unknown technology 'tech_ghost'" in message


@pytest.mark.parametrize(
    ("table", "index", "column"),
    [
        ("TechnologyPrereqs", 0, "Technology"),
        ("TechnologyPrereqs", 1, "PrereqTech"),
        ("Buildings", 0, "PrereqTech"),
        ("Units", 1, "PrereqTech"),
    ],
)
def test_a_reference_to_a_tech_skipped_for_its_cost_says_so(
    table: str, index: int, column: str
) -> None:
    document = export()
    document[table][index][column] = "TECH_UNUSED"  # the Cost -1 row
    message = _error(document)
    assert f"{table}[{index}].{column}" in message
    assert "was skipped for its non-positive cost" in message


def test_a_ghost_prerequisite_is_a_game_info_error_not_a_validation_error() -> None:
    document = export()
    document["TechnologyPrereqs"][0]["PrereqTech"] = "TECH_GHOST"
    assert "TechnologyPrereqs[0].PrereqTech" in _error(document)


def test_a_ghost_prerequisite_exits_one_through_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = export()
    document["TechnologyPrereqs"][1]["PrereqTech"] = "TECH_GHOST"
    out = tmp_path / "r.yaml"
    assert main(["import-gameinfo", str(write_export(tmp_path, document)), "--out", str(out)]) == 1
    assert "error: TechnologyPrereqs[1].PrereqTech: unknown technology" in capsys.readouterr().err
    assert not out.exists()


# ---- duplicates, repeated prerequisites, eras, clamping ---------------------------


@pytest.mark.parametrize(
    ("table", "column", "template"),
    [
        ("Technologies", "TechnologyType", {"EraType": "ERA_ANCIENT", "Cost": 999}),
        ("Buildings", "BuildingType", {"Cost": 999}),
        ("Units", "UnitType", {"Cost": 999}),
    ],
)
@pytest.mark.parametrize("spelling", ["same", "case"])
def test_a_duplicate_id_is_refused_naming_the_row(
    table: str, column: str, template: dict[str, object], spelling: str
) -> None:
    document = export()
    first = document[table][0][column]
    duplicate = first if spelling == "same" else first.lower()
    assert (duplicate == first) == (spelling == "same")
    document[table].append({column: duplicate, **template})
    index = len(document[table]) - 1
    message = _error(document)
    assert f"{table}[{index}].{column}: duplicate id" in message
    assert first.lower() in message


def test_a_skipped_row_still_claims_its_id() -> None:
    document = export()
    document["Technologies"].append(
        {"TechnologyType": "tech_unused", "EraType": "ERA_ANCIENT", "Cost": 5}
    )
    assert "Technologies[6].TechnologyType: duplicate id 'tech_unused'" in _error(document)
    document = export()
    document["Buildings"].append({"BuildingType": "BUILDING_ZERO", "Cost": 0})
    document["Buildings"].append({"BuildingType": "building_zero", "Cost": 9})
    assert "Buildings[3].BuildingType: duplicate id" in _error(document)


def test_the_same_name_in_different_tables_is_not_a_duplicate() -> None:
    document = export()
    document["Units"][0]["UnitType"] = "TECH_ALPHA"
    spec = parse_ruleset(convert(document).text)
    assert "tech_alpha" in {t.id for t in spec.techs} and "tech_alpha" in {u.id for u in spec.units}


def test_a_repeated_prerequisite_row_is_dropped() -> None:
    document = export()
    document["TechnologyPrereqs"].append(dict(document["TechnologyPrereqs"][0]))
    document["TechnologyPrereqs"].append(dict(document["TechnologyPrereqs"][2]))
    spec = parse_ruleset(convert(document).text)
    assert spec.tech("tech_gamma").prerequisites == ("tech_alpha",)
    assert spec.tech("tech_delta").prerequisites == ("tech_beta", "tech_gamma")
    assert convert(document).text == convert(export()).text


def test_era_and_era_type_together_are_refused() -> None:
    document = export()
    document["Technologies"][0]["Era"] = 2  # row 0 also has EraType
    message = _error(document)
    assert "Technologies[0]" in message and "not both" in message


def test_a_negative_era_is_refused_naming_the_column() -> None:
    document = export()
    document["Technologies"][3]["Era"] = -1
    assert "Technologies[3].Era" in _error(document)


def test_era_zero_is_accepted() -> None:
    document = export()
    document["Technologies"][3]["Era"] = 0
    assert parse_ruleset(convert(document).text).tech("tech_delta").era == 0


def test_negative_combat_and_moves_import_as_zero() -> None:
    document = export()
    document["Units"][1].update({"Combat": -1, "BaseMoves": -1})
    units = {u.id: u for u in parse_ruleset(convert(document).text).units}
    assert (units["unit_guard"].strength, units["unit_guard"].moves) == (0, 0)
    assert units["unit_guard"].cost == 40


def test_a_prerequisite_cycle_in_the_export_still_raises_the_validation_error() -> None:
    document = export()
    document["TechnologyPrereqs"].append({"Technology": "TECH_ALPHA", "PrereqTech": "TECH_ALPHA"})
    with pytest.raises(RulesetValidationError) as excinfo:
        convert(document)
    assert [i.rule for i in excinfo.value.issues] == ["prerequisite-cycle"]


@pytest.mark.parametrize("ruleset_id", ["Bad_Id", "has space", "../x", "", "UPPER"])
def test_a_bad_ruleset_id_is_refused(ruleset_id: str) -> None:
    message = _error(export(), ruleset_id)
    assert "rule-set id" in message


def test_a_prerequisite_cycle_in_the_export_raises_the_validation_error() -> None:
    document = export()
    document["TechnologyPrereqs"].append({"Technology": "TECH_ALPHA", "PrereqTech": "TECH_EPSILON"})
    with pytest.raises(RulesetValidationError) as excinfo:
        convert(document)
    assert [i.rule for i in excinfo.value.issues] == ["prerequisite-cycle"]
    assert "tech_alpha" in str(excinfo.value)


# ---- checkout_root and resolve_output ---------------------------------------------


def test_checkout_root_is_the_repository_root() -> None:
    assert checkout_root() == REPO_ROOT


def _assert_refused(out: str | None) -> None:
    """``resolve_output`` must refuse ``out``."""
    with pytest.raises(RefusedPathError, match="inside the repository"):
        resolve_output(out, "x-test")


def test_a_path_inside_the_checkout_is_refused() -> None:
    _assert_refused(str(REPO_ROOT / "gameinfo" / "x.yaml"))
    _assert_refused(str(REPO_ROOT / "out.yaml"))
    _assert_refused(str(REPO_ROOT))
    _assert_refused(str(REPO_ROOT / "tests" / ".." / "x.yaml"))


def test_a_dotdot_that_climbs_out_of_the_checkout_is_allowed(tmp_path: Path) -> None:
    sibling = REPO_ROOT.parent / "gm-k2-outside" / "x.yaml"
    assert (
        resolve_output(str(REPO_ROOT / ".." / "gm-k2-outside" / "x.yaml"), "x") == sibling.resolve()
    )


def test_a_sibling_with_a_common_name_prefix_is_not_inside() -> None:
    sibling = Path(str(REPO_ROOT) + "-data") / "x.yaml"
    assert resolve_output(str(sibling), "x") == sibling.resolve()


def test_an_outside_path_is_returned_resolved(tmp_path: Path) -> None:
    target = tmp_path / "sub" / ".." / "r.yaml"
    assert resolve_output(str(target), "x") == (tmp_path / "r.yaml").resolve()


def test_a_relative_path_is_judged_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    _assert_refused("relative-probe.yaml")
    monkeypatch.chdir(tmp_path)
    assert resolve_output("relative-probe.yaml", "x") == tmp_path.resolve() / "relative-probe.yaml"


def test_a_symlink_into_the_checkout_is_refused(tmp_path: Path) -> None:
    link = tmp_path / "link"
    try:
        link.symlink_to(REPO_ROOT / "tests", target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform without symlinks
        pytest.fail("this platform cannot create a symlink; the case is untested")
    _assert_refused(str(link / "probe.yaml"))


def test_no_out_and_no_data_root_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CIV6AZ_DATA_ROOT", raising=False)
    with pytest.raises(RefusedPathError, match="CIV6AZ_DATA_ROOT"):
        resolve_output(None, "x-test")
    monkeypatch.setenv("CIV6AZ_DATA_ROOT", "   ")
    with pytest.raises(RefusedPathError, match="CIV6AZ_DATA_ROOT"):
        resolve_output(None, "x-test")


def test_the_data_root_names_the_default_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CIV6AZ_DATA_ROOT", str(tmp_path))
    assert resolve_output(None, "my-set") == tmp_path.resolve() / "gameinfo" / "my-set.yaml"


def test_a_data_root_inside_the_checkout_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_DATA_ROOT", str(REPO_ROOT / "data-root"))
    with pytest.raises(RefusedPathError, match="inside the repository"):
        resolve_output(None, "x")


def test_without_a_checkout_any_path_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gameinfo, "checkout_root", lambda: None)
    inside = REPO_ROOT / "would-be-refused.yaml"
    assert resolve_output(str(inside), "x") == inside.resolve()


def test_checkout_root_is_none_when_no_pyproject_is_above(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "pkg" / "gameinfo.py"
    fake.parent.mkdir()
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(gameinfo, "__file__", str(fake))
    if any((p / "pyproject.toml").is_file() for p in fake.resolve().parents):
        pytest.fail("an ancestor of the temp directory holds a pyproject.toml")
    assert checkout_root() is None
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    assert checkout_root() == tmp_path.resolve()


# ---- import_gameinfo and the CLI: writing -----------------------------------------


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


def test_cli_import_to_an_outside_out_writes_a_file_that_validates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "result" / "synth.yaml"
    code = main(["import-gameinfo", str(FIXTURE), "--out", str(out), "--id", "synth-test"])
    assert code == 0
    assert out.is_file()
    assert _leftovers(out.parent) == ["synth.yaml"]  # no .tmp left behind
    said = capsys.readouterr().out
    assert "5 techs, 2 units, 2 buildings" in said
    assert "skipped 1 rows" in said
    assert "1 civics not imported" in said
    assert main(["validate", str(out)]) == 0
    assert "synth-test: valid" in capsys.readouterr().out
    assert main(["hash", str(out)]) == 0


def test_the_written_text_is_what_convert_produces(tmp_path: Path) -> None:
    out = tmp_path / "r.yaml"
    path, summary = import_gameinfo(FIXTURE, str(out), "synth-test")
    assert path == out.resolve()
    assert out.read_text(encoding="utf-8") == summary.text
    assert b"\r" not in out.read_bytes()


def test_default_destination_under_the_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CIV6AZ_DATA_ROOT", str(tmp_path))
    assert main(["import-gameinfo", str(FIXTURE), "--id", "from-root"]) == 0
    written = tmp_path / "gameinfo" / "from-root.yaml"
    assert written.is_file()
    assert main(["validate", str(written)]) == 0
    assert str(written) in capsys.readouterr().out


def test_default_id_names_the_default_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_DATA_ROOT", str(tmp_path))
    assert main(["import-gameinfo", str(FIXTURE)]) == 0
    assert (tmp_path / "gameinfo" / "gameinfo-import.yaml").is_file()


def test_no_out_and_no_data_root_exits_two_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CIV6AZ_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(["import-gameinfo", str(FIXTURE)]) == 2
    assert "CIV6AZ_DATA_ROOT" in capsys.readouterr().err
    assert _leftovers(tmp_path) == []


def assert_refused_and_nothing_written(
    out: Path, capsys: pytest.CaptureFixture[str], watch: Path
) -> None:
    """Drive the CLI at ``out``: exit 2, a refusal message, no file and no ``.tmp`` anywhere."""
    leftovers_before = _leftovers(watch)
    code = main(["import-gameinfo", str(FIXTURE), "--out", str(out)])
    assert code == 2, f"expected a refusal, got exit {code}"
    assert "refused" in capsys.readouterr().err
    assert not out.exists()
    assert not out.with_name(out.name + ".tmp").exists()
    assert _leftovers(watch) == leftovers_before


def test_cli_refuses_an_output_inside_the_repository(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = REPO_ROOT / "tests" / "gm_k2_probe_inside_repo.yaml"
    try:
        assert_refused_and_nothing_written(out, capsys, out.parent)
    finally:  # only runs the cleanup if the guard failed
        out.unlink(missing_ok=True)
        out.with_name(out.name + ".tmp").unlink(missing_ok=True)


def test_cli_refuses_a_relative_path_inside_the_repository(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPO_ROOT / "tests")
    out = Path("gm_k2_probe_relative.yaml")
    try:
        assert_refused_and_nothing_written(out, capsys, REPO_ROOT / "tests")
    finally:
        out.unlink(missing_ok=True)
        out.with_name(out.name + ".tmp").unlink(missing_ok=True)


def test_cli_refuses_a_symlink_into_the_repository(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    link = tmp_path / "link"
    try:
        link.symlink_to(REPO_ROOT / "tests", target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover
        pytest.fail("this platform cannot create a symlink; the case is untested")
    real = REPO_ROOT / "tests" / "gm_k2_probe_symlink.yaml"
    try:
        assert_refused_and_nothing_written(link / real.name, capsys, REPO_ROOT / "tests")
        assert not real.exists()
    finally:
        real.unlink(missing_ok=True)
        real.with_name(real.name + ".tmp").unlink(missing_ok=True)


def test_cli_refuses_a_data_root_inside_the_repository(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CIV6AZ_DATA_ROOT", str(REPO_ROOT / "tests" / "gm_k2_probe_root"))
    probe = REPO_ROOT / "tests" / "gm_k2_probe_root"
    try:
        assert main(["import-gameinfo", str(FIXTURE)]) == 2
        assert "refused" in capsys.readouterr().err
        assert not probe.exists()
    finally:
        if probe.exists():  # pragma: no cover - only if the guard failed
            import shutil

            shutil.rmtree(probe)


def test_control_the_refusal_test_fails_without_the_checkout_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Negative control (a): ``resolve_output`` that skips the checkout check.

    The "inside the repository" assertion is run against a fake checkout in
    ``tmp_path`` so a broken guard writes nowhere near the real repository.
    """
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    monkeypatch.setattr(gameinfo, "checkout_root", lambda: fake_repo.resolve())
    out = fake_repo / "probe.yaml"
    assert_refused_and_nothing_written(out, capsys, fake_repo)  # the guard works

    def unchecked(out_arg: str | None, ruleset_id: str) -> Path:
        assert out_arg is not None
        return Path(out_arg).resolve()

    monkeypatch.setattr(gameinfo, "resolve_output", unchecked)
    with pytest.raises(AssertionError):
        assert_refused_and_nothing_written(out, capsys, fake_repo)
    out.unlink(missing_ok=True)


def test_a_malformed_export_exits_one_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = export()
    document["Technologies"][0]["Cost"] = 1.5
    source = write_export(tmp_path, document)
    out = tmp_path / "out" / "r.yaml"
    assert main(["import-gameinfo", str(source), "--out", str(out)]) == 1
    assert "Technologies[0].Cost" in capsys.readouterr().err
    assert not out.exists() and not out.parent.exists()


def test_an_invalid_converted_rule_set_exits_one_listing_the_issue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI's RulesetValidationError branch: only the validator can see a cycle."""
    document = export()
    document["TechnologyPrereqs"].append({"Technology": "TECH_ALPHA", "PrereqTech": "TECH_EPSILON"})
    source = write_export(tmp_path, document)
    out = tmp_path / "r.yaml"
    assert main(["import-gameinfo", str(source), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "the imported rule set is invalid" in err
    assert "[prerequisite-cycle]" in err and "tech_alpha" in err
    assert not out.exists()


def test_a_bad_id_exits_one_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    monkeypatch.setenv("CIV6AZ_DATA_ROOT", str(root))
    assert main(["import-gameinfo", str(FIXTURE), "--id", "../../evil"]) == 1
    assert not root.exists()
    assert not (tmp_path / "evil.yaml").exists()


@pytest.mark.parametrize(
    "payload", [b"this is not json", b"", b"\xff\xfe\x00{", b'{"Technologies": [}'], ids=str
)
def test_non_json_or_non_utf8_source_exits_one(
    tmp_path: Path, payload: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "bad.json"
    source.write_bytes(payload)
    out = tmp_path / "r.yaml"
    assert main(["import-gameinfo", str(source), "--out", str(out)]) == 1
    assert "not UTF-8 JSON" in capsys.readouterr().err
    assert not out.exists()


def test_a_json_array_source_exits_one(tmp_path: Path) -> None:
    source = write_export(tmp_path, [1, 2])
    assert main(["import-gameinfo", str(source), "--out", str(tmp_path / "r.yaml")]) == 1


def test_a_missing_source_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "r.yaml"
    assert main(["import-gameinfo", str(tmp_path / "absent.json"), "--out", str(out)]) == 2
    assert "absent.json" in capsys.readouterr().err
    assert not out.exists()


def test_a_directory_source_exits_two(tmp_path: Path) -> None:
    assert main(["import-gameinfo", str(tmp_path), "--out", str(tmp_path / "r.yaml")]) == 2


def test_the_refusal_comes_before_the_source_is_read(tmp_path: Path) -> None:
    """A bad destination wins over a bad source: nothing is read first (docstring promise)."""
    source = tmp_path / "bad.json"
    source.write_bytes(b"not json")
    out = REPO_ROOT / "tests" / "gm_k2_probe_order.yaml"
    try:
        assert main(["import-gameinfo", str(source), "--out", str(out)]) == 2
        assert not out.exists()
    finally:
        out.unlink(missing_ok=True)


def test_a_failed_conversion_leaves_an_existing_destination_untouched(
    tmp_path: Path,
) -> None:
    out = tmp_path / "keep.yaml"
    out.write_bytes(b"precious: bytes\r\n\xff")
    before = out.read_bytes()
    document = export()
    document["Units"][0]["Cost"] = "two words"
    source = write_export(tmp_path, document)
    assert main(["import-gameinfo", str(source), "--out", str(out)]) == 1
    assert out.read_bytes() == before
    assert _leftovers(tmp_path) == ["export.json", "keep.yaml"]


def test_a_failed_write_leaves_an_existing_destination_and_no_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "keep.yaml"
    out.write_bytes(b"old\n")

    def boom(src: object, dst: object) -> None:
        raise PermissionError("replace-denied-by-test")

    monkeypatch.setattr(os, "replace", boom)
    assert main(["import-gameinfo", str(FIXTURE), "--out", str(out)]) == 1
    assert "replace-denied-by-test" in capsys.readouterr().err
    assert out.read_bytes() == b"old\n"
    assert _leftovers(tmp_path) == ["keep.yaml"]


def test_a_second_import_replaces_the_first(tmp_path: Path) -> None:
    out = tmp_path / "r.yaml"
    assert main(["import-gameinfo", str(FIXTURE), "--out", str(out), "--id", "first-id"]) == 0
    assert main(["import-gameinfo", str(FIXTURE), "--out", str(out), "--id", "second-id"]) == 0
    assert parse_ruleset(out.read_text(encoding="utf-8")).id == "second-id"


def test_import_usage_errors_exit_two() -> None:
    for argv in (["import-gameinfo"], ["import-gameinfo", "a", "--bogus"]):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2


# ---- the remaining column and summary branches ------------------------------------


def test_an_identifier_string_where_an_integer_is_expected_names_the_column() -> None:
    document = export()
    document["Technologies"][3]["Era"] = "ERA_ANCIENT"
    assert "Technologies[3].Era: expected an integer" in _error(document)


def test_an_integer_where_an_identifier_is_expected_names_the_column() -> None:
    document = export()
    document["Technologies"][1]["TechnologyType"] = 7
    assert "Technologies[1].TechnologyType: expected an identifier" in _error(document)


def test_a_clean_export_prints_no_skipped_or_civic_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = export()
    del document["Technologies"][5], document["Civics"], document["CivicPrereqs"]
    source = write_export(tmp_path, document)
    assert main(["import-gameinfo", str(source), "--out", str(tmp_path / "r.yaml")]) == 0
    said = capsys.readouterr().out
    assert "5 techs" in said and "skipped" not in said and "civics" not in said
