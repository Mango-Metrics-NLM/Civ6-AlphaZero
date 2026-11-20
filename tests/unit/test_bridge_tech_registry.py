"""GM-B3: ``alpha_zero.bridge.tech_registry``, the tech and civic slot registry.

The module is imported inside the ``tr`` fixture, so while it does not exist only the
tests that use it fail; nothing here is a collection error. No Lua runtime is needed.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from alpha_zero.rules import build_registry, load_builtin, parse_ruleset

REPO_ROOT = Path(__file__).resolve().parents[2]
RULES_DIR = REPO_ROOT / "alpha_zero" / "rules"

_RULESET_TEMPLATE = """\
schema_version: 1
id: {rid}
techs:
{techs}
units:
  - {{id: sword, cost: 5, strength: 3, moves: 2}}
buildings:
  - id: hut
    cost: 5
    modifiers:
      - {{yield: food, amount: 1}}
"""


def ruleset_text(ids: list[str], *, cost: int = 10, rid: str = "research-test") -> str:
    """A valid rule set whose techs are ``ids`` in that order, none with prerequisites."""
    techs = "\n".join(f"  - id: {i}\n    era: 1\n    cost: {cost}" for i in ids)
    return _RULESET_TEMPLATE.format(rid=rid, techs=techs)


TECH_IDS = ["writing", "bronze_working", "pottery", "mining"]
CIVIC_TYPES = (
    "CIVIC_EARLY_EMPIRE",
    "CIVIC_FOREIGN_TRADE",
    "CIVIC_CRAFTSMANSHIP",
    "CIVIC_CODE_OF_LAWS",
)


def write_gameinfo(path: Path, civics: list[Any]) -> Path:
    path.write_text(json.dumps({"Technologies": [], "Civics": civics}), encoding="utf-8")
    return path


@pytest.fixture
def tr() -> ModuleType:
    return importlib.import_module("alpha_zero.bridge.tech_registry")


def make(tr: ModuleType, tech_ids: tuple[str, ...], civic_ids: tuple[str, ...] = ()) -> Any:
    return tr.ResearchRegistry(
        ruleset_id="direct", fingerprint="0" * 64, tech_ids=tech_ids, civic_ids=civic_ids
    )


class TestFixturePremises:
    """The rule-set text below is valid for the kernel, so a failure elsewhere is not its fault."""

    def test_the_fixture_rule_set_loads_and_validates(self) -> None:
        from alpha_zero.rules import ensure_valid

        spec = parse_ruleset(ruleset_text(TECH_IDS))
        ensure_valid(spec)
        assert tuple(build_registry(spec).tech_index) == tuple(TECH_IDS)

    def test_the_dangling_prerequisite_is_a_kernel_error(self) -> None:
        from alpha_zero.rules import RulesetValidationError, ensure_valid

        text = ruleset_text(["a", "b"]).replace(
            "  - id: b\n    era: 1\n    cost: 10",
            "  - id: b\n    era: 1\n    cost: 10\n    prerequisites: [ghost]",
        )
        assert "ghost" in text
        with pytest.raises(RulesetValidationError):
            ensure_valid(parse_ruleset(text))


# ---------------------------------------------------------------------------
# constants and the dataclass
# ---------------------------------------------------------------------------
class TestConstantsAndShape:
    def test_env_names(self, tr: ModuleType) -> None:
        assert tr.RULESET_ENV == "CIV6AZ_BRIDGE_RULESET"
        assert tr.GAMEINFO_ENV == "CIV6AZ_BRIDGE_GAMEINFO"

    def test_error_is_a_value_error(self, tr: ModuleType) -> None:
        assert issubclass(tr.ResearchRegistryError, ValueError)

    def test_frozen_hashable_and_equal_by_value(self, tr: ModuleType) -> None:
        a = make(tr, ("mining", "writing"), ("code_of_laws",))
        b = make(tr, ("mining", "writing"), ("code_of_laws",))
        c = make(tr, ("writing", "mining"), ("code_of_laws",))
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert len({a, b, c}) == 2
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.ruleset_id = "other"  # type: ignore[misc]


class TestDescribe:
    def reg(self, tr: ModuleType) -> Any:
        return tr.ResearchRegistry(
            ruleset_id="demo-rules",
            fingerprint="abcdef012345" + "f" * 52,
            tech_ids=tuple(f"t{i}" for i in range(37)),
            civic_ids=tuple(f"c{i}" for i in range(11)),
        )

    def test_one_line_with_id_fingerprint_prefix_and_counts(self, tr: ModuleType) -> None:
        reg = self.reg(tr)
        text = reg.describe()
        assert isinstance(text, str)
        assert "\n" not in text
        assert "demo-rules" in text
        assert "abcdef012345" in text
        assert "abcdef012345f" not in text, "more than 12 characters of the fingerprint"
        assert "37" in text
        assert "11" in text

    def test_no_civics_is_visible(self, tr: ModuleType) -> None:
        reg = tr.ResearchRegistry(
            ruleset_id="demo-rules", fingerprint="abcdef012345" + "f" * 52, tech_ids=("a",)
        )
        assert re.search(r"\b0\b", reg.describe())


# ---------------------------------------------------------------------------
# slot lookup
# ---------------------------------------------------------------------------
class TestSlotLookup:
    def test_tech_slot_matches_bare_and_prefixed_ids_case_insensitively(
        self, tr: ModuleType
    ) -> None:
        reg = make(tr, ("agriculture", "tech_mining", "bronze_working"))
        assert reg.tech_slot("TECH_AGRICULTURE") == 0
        assert reg.tech_slot("tech_agriculture") == 0
        assert reg.tech_slot("Tech_Mining") == 1
        assert reg.tech_slot("TECH_BRONZE_WORKING") == 2
        assert reg.tech_slot("agriculture") == 0

    def test_exact_lowercase_id_wins_over_prefix_stripped_id(self, tr: ModuleType) -> None:
        both = make(tr, ("mining", "tech_mining"))
        assert both.tech_slot("TECH_MINING") == 1
        reverse = make(tr, ("tech_mining", "mining"))
        assert reverse.tech_slot("TECH_MINING") == 0

    @pytest.mark.parametrize("bad", [None, "", 3, 1.5, True, b"TECH_MINING", ["TECH_MINING"]])
    def test_non_string_or_empty_gives_none(self, tr: ModuleType, bad: Any) -> None:
        reg = make(tr, ("mining", "writing"), ("code_of_laws", "foreign_trade"))
        assert reg.tech_slot(bad) is None
        assert reg.civic_slot(bad) is None

    def test_unknown_names_give_none(self, tr: ModuleType) -> None:
        reg = make(tr, ("mining",), ("code_of_laws",))
        assert reg.tech_slot("TECH_POTTERY") is None
        assert reg.civic_slot("CIVIC_FOREIGN_TRADE") is None
        # A tech name is not a civic name and the other way round.
        assert reg.civic_slot("TECH_MINING") is None
        assert reg.tech_slot("CIVIC_CODE_OF_LAWS") is None

    def test_civic_slot_matches_bare_and_prefixed_ids(self, tr: ModuleType) -> None:
        reg = make(tr, (), ("code_of_laws", "civic_craftsmanship", "foreign_trade"))
        assert reg.civic_slot("CIVIC_CODE_OF_LAWS") == 0
        assert reg.civic_slot("civic_code_of_laws") == 0
        assert reg.civic_slot("CIVIC_CRAFTSMANSHIP") == 1
        assert reg.civic_slot("Civic_Foreign_Trade") == 2

    def test_civic_exact_lowercase_id_wins(self, tr: ModuleType) -> None:
        reg = make(tr, (), ("code_of_laws", "civic_code_of_laws"))
        assert reg.civic_slot("CIVIC_CODE_OF_LAWS") == 1


class TestTypeLookup:
    def test_tech_type_has_the_prefix_and_is_upper_case(self, tr: ModuleType) -> None:
        reg = make(tr, ("mining", "tech_writing", "bronze_working"))
        assert reg.tech_type(0) == "TECH_MINING"
        assert reg.tech_type(1) == "TECH_WRITING"
        assert reg.tech_type(2) == "TECH_BRONZE_WORKING"

    def test_civic_type_has_the_prefix_and_is_upper_case(self, tr: ModuleType) -> None:
        reg = make(tr, (), ("code_of_laws", "civic_foreign_trade"))
        assert reg.civic_type(0) == "CIVIC_CODE_OF_LAWS"
        assert reg.civic_type(1) == "CIVIC_FOREIGN_TRADE"

    @pytest.mark.parametrize("bad", [-1, 2, 99, "0", "1", 0.0, 1.0, None, True, False])
    def test_bad_slots_give_none(self, tr: ModuleType, bad: Any) -> None:
        reg = make(tr, ("mining", "writing"), ("code_of_laws", "foreign_trade"))
        assert reg.tech_type(bad) is None
        assert reg.civic_type(bad) is None

    def test_type_and_slot_round_trip(self, tr: ModuleType) -> None:
        reg = make(tr, ("mining", "tech_writing", "bronze_working"), ("a", "civic_b"))
        for slot in range(3):
            assert reg.tech_slot(reg.tech_type(slot)) == slot
        for slot in range(2):
            assert reg.civic_slot(reg.civic_type(slot)) == slot


# ---------------------------------------------------------------------------
# from_ruleset
# ---------------------------------------------------------------------------
class TestFromRuleset:
    def test_tech_slots_are_the_rules_kernel_index(self, tr: ModuleType) -> None:
        spec = parse_ruleset(ruleset_text(TECH_IDS))
        reg = tr.from_ruleset(spec)
        assert reg.tech_ids == tuple(build_registry(spec).tech_index)
        assert reg.tech_ids == tuple(TECH_IDS)
        assert reg.ruleset_id == "research-test"
        assert reg.civic_ids == ()

    def test_civic_slots_follow_the_given_order_lowercased(self, tr: ModuleType) -> None:
        reg = tr.from_ruleset(parse_ruleset(ruleset_text(TECH_IDS)), CIVIC_TYPES)
        assert reg.civic_ids == tuple(c.lower() for c in CIVIC_TYPES)
        assert reg.civic_slot("CIVIC_CODE_OF_LAWS") == 3
        assert reg.civic_type(0) == "CIVIC_EARLY_EMPIRE"

    def test_civics_may_be_given_as_bare_ids(self, tr: ModuleType) -> None:
        reg = tr.from_ruleset(parse_ruleset(ruleset_text(TECH_IDS)), ("code_of_laws", "writing"))
        assert reg.civic_ids == ("code_of_laws", "writing")
        assert reg.civic_type(0) == "CIVIC_CODE_OF_LAWS"
        assert reg.civic_slot("CIVIC_CODE_OF_LAWS") == 0

    def test_a_civic_repeated_after_lowercasing_raises(self, tr: ModuleType) -> None:
        spec = parse_ruleset(ruleset_text(TECH_IDS))
        with pytest.raises(tr.ResearchRegistryError):
            tr.from_ruleset(spec, ("CIVIC_CODE_OF_LAWS", "civic_code_of_laws"))
        with pytest.raises(tr.ResearchRegistryError):
            tr.from_ruleset(spec, ("CIVIC_A", "CIVIC_B", "CIVIC_A"))

    def test_two_techs_naming_the_same_game_type_are_refused(self, tr: ModuleType) -> None:
        for ids in (["mining", "tech_mining"], ["tech_mining", "writing", "mining"]):
            spec = parse_ruleset(ruleset_text(ids))
            with pytest.raises(tr.ResearchRegistryError):
                tr.from_ruleset(spec)

    def test_distinct_techs_with_and_without_prefix_are_fine(self, tr: ModuleType) -> None:
        spec = parse_ruleset(ruleset_text(["mining", "tech_writing", "pottery"]))
        assert tr.from_ruleset(spec).tech_ids == ("mining", "tech_writing", "pottery")

    @pytest.mark.parametrize(
        "civics",
        [("CIVIC_A", "a"), ("a", "civic_a"), ("civic_a", "CIVIC_B", "A")],
    )
    def test_two_civics_naming_the_same_game_type_are_refused(
        self, tr: ModuleType, civics: tuple[str, ...]
    ) -> None:
        spec = parse_ruleset(ruleset_text(TECH_IDS))
        with pytest.raises(tr.ResearchRegistryError):
            tr.from_ruleset(spec, civics)

    def test_distinct_civics_are_fine(self, tr: ModuleType) -> None:
        spec = parse_ruleset(ruleset_text(TECH_IDS))
        assert tr.from_ruleset(spec, ("CIVIC_A", "b")).civic_ids == ("civic_a", "b")

    def test_fingerprint_is_64_hex(self, tr: ModuleType) -> None:
        reg = tr.from_ruleset(parse_ruleset(ruleset_text(TECH_IDS)), CIVIC_TYPES)
        assert re.fullmatch(r"[0-9a-f]{64}", reg.fingerprint)

    def test_fingerprint_changes_with_tech_order(self, tr: ModuleType) -> None:
        a = tr.from_ruleset(parse_ruleset(ruleset_text(TECH_IDS)), CIVIC_TYPES)
        b = tr.from_ruleset(parse_ruleset(ruleset_text(list(reversed(TECH_IDS)))), CIVIC_TYPES)
        assert a.fingerprint != b.fingerprint
        assert a != b

    def test_fingerprint_changes_with_the_civic_list(self, tr: ModuleType) -> None:
        spec = parse_ruleset(ruleset_text(TECH_IDS))
        base = tr.from_ruleset(spec, CIVIC_TYPES)
        reordered = tr.from_ruleset(spec, tuple(reversed(CIVIC_TYPES)))
        shorter = tr.from_ruleset(spec, CIVIC_TYPES[:-1])
        none = tr.from_ruleset(spec)
        prints = {base.fingerprint, reordered.fingerprint, shorter.fingerprint, none.fingerprint}
        assert len(prints) == 4

    def test_fingerprint_ignores_costs(self, tr: ModuleType) -> None:
        cheap = tr.from_ruleset(parse_ruleset(ruleset_text(TECH_IDS, cost=10)), CIVIC_TYPES)
        dear = tr.from_ruleset(parse_ruleset(ruleset_text(TECH_IDS, cost=99)), CIVIC_TYPES)
        assert cheap.fingerprint == dear.fingerprint
        assert cheap == dear

    def test_fingerprint_is_deterministic(self, tr: ModuleType) -> None:
        spec = parse_ruleset(ruleset_text(TECH_IDS))
        assert tr.from_ruleset(spec, CIVIC_TYPES) == tr.from_ruleset(spec, CIVIC_TYPES)

    def test_builtin_ruleset_slots_match_its_declared_order(self, tr: ModuleType) -> None:
        spec = load_builtin("science-v1-draft")
        reg = tr.from_ruleset(spec)
        order = tuple(build_registry(spec).tech_index)
        assert reg.tech_ids == order
        assert order[:3] == ("agriculture", "mining", "writing")
        assert reg.tech_slot("TECH_WRITING") == 2
        assert reg.tech_type(order.index("bronze_working")) == "TECH_BRONZE_WORKING"


# ---------------------------------------------------------------------------
# civic_types_from_gameinfo
# ---------------------------------------------------------------------------
class TestCivicTypesFromGameinfo:
    def test_reads_civic_types_in_row_order(self, tr: ModuleType, tmp_path: Path) -> None:
        path = write_gameinfo(
            tmp_path / "gi.json",
            [
                {"CivicType": "CIVIC_Z", "Index": 0},
                {"CivicType": "CIVIC_A", "Index": 1},
                {"CivicType": "CIVIC_M"},
            ],
        )
        result = tr.civic_types_from_gameinfo(path)
        assert result == ("CIVIC_Z", "CIVIC_A", "CIVIC_M")
        assert isinstance(result, tuple)

    def test_accepts_a_string_path(self, tr: ModuleType, tmp_path: Path) -> None:
        path = write_gameinfo(tmp_path / "gi.json", [{"CivicType": "CIVIC_A"}])
        assert tr.civic_types_from_gameinfo(str(path)) == ("CIVIC_A",)

    def test_missing_file(self, tr: ModuleType, tmp_path: Path) -> None:
        path = tmp_path / "absent.json"
        with pytest.raises(tr.ResearchRegistryError) as err:
            tr.civic_types_from_gameinfo(path)
        assert str(path) in str(err.value)

    def test_invalid_json(self, tr: ModuleType, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(tr.ResearchRegistryError) as err:
            tr.civic_types_from_gameinfo(path)
        assert str(path) in str(err.value)

    @pytest.mark.parametrize("document", [[], [{"CivicType": "CIVIC_A"}], "Civics", 3, None])
    def test_a_document_that_is_not_an_object(
        self, tr: ModuleType, tmp_path: Path, document: Any
    ) -> None:
        path = tmp_path / "doc.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(tr.ResearchRegistryError) as err:
            tr.civic_types_from_gameinfo(path)
        assert str(path) in str(err.value)

    def test_missing_civics_table(self, tr: ModuleType, tmp_path: Path) -> None:
        path = tmp_path / "nocivics.json"
        path.write_text(json.dumps({"Technologies": [{"TechnologyType": "TECH_A"}]}), "utf-8")
        with pytest.raises(tr.ResearchRegistryError) as err:
            tr.civic_types_from_gameinfo(path)
        assert str(path) in str(err.value)

    @pytest.mark.parametrize(
        "rows",
        [
            [{"CivicType": "CIVIC_A"}, {}],
            [{"CivicType": "CIVIC_A"}, {"CivicType": 7}],
            [{"CivicType": None}],
            [{"CivicType": "CIVIC_A"}, "CIVIC_B"],
        ],
    )
    def test_a_row_without_a_string_civic_type(
        self, tr: ModuleType, tmp_path: Path, rows: list[Any]
    ) -> None:
        path = write_gameinfo(tmp_path / "rows.json", rows)
        with pytest.raises(tr.ResearchRegistryError) as err:
            tr.civic_types_from_gameinfo(path)
        assert str(path) in str(err.value)


# ---------------------------------------------------------------------------
# load_registry
# ---------------------------------------------------------------------------
class TestLoadRegistry:
    def test_builtin_id(self, tr: ModuleType) -> None:
        reg = tr.load_registry("science-v1-draft")
        expected = tuple(build_registry(load_builtin("science-v1-draft")).tech_index)
        assert reg.ruleset_id == "science-v1-draft"
        assert reg.tech_ids == expected
        assert reg.civic_ids == ()
        assert reg == tr.load_registry("science-v1-draft")

    @pytest.mark.parametrize("as_path", [False, True])
    def test_file_path(self, tr: ModuleType, tmp_path: Path, as_path: bool) -> None:
        path = tmp_path / "rules.yaml"
        path.write_text(ruleset_text(TECH_IDS), encoding="utf-8")
        reg = tr.load_registry(path if as_path else str(path))
        assert reg.ruleset_id == "research-test"
        assert reg.tech_ids == tuple(TECH_IDS)

    def test_gameinfo_supplies_the_civic_order(self, tr: ModuleType, tmp_path: Path) -> None:
        rules = tmp_path / "rules.yaml"
        rules.write_text(ruleset_text(TECH_IDS), encoding="utf-8")
        gameinfo = write_gameinfo(tmp_path / "gi.json", [{"CivicType": c} for c in CIVIC_TYPES])
        reg = tr.load_registry(str(rules), gameinfo=str(gameinfo))
        assert reg.civic_ids == tuple(c.lower() for c in CIVIC_TYPES)
        assert reg == tr.from_ruleset(parse_ruleset(ruleset_text(TECH_IDS)), CIVIC_TYPES)

    def test_unknown_id_that_is_not_a_file(self, tr: ModuleType, tmp_path: Path) -> None:
        with pytest.raises(tr.ResearchRegistryError):
            tr.load_registry("no-such-ruleset")
        with pytest.raises(tr.ResearchRegistryError):
            tr.load_registry(str(tmp_path / "missing.yaml"))

    def test_a_file_that_does_not_parse(self, tr: ModuleType, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("techs: [unclosed", encoding="utf-8")
        with pytest.raises(tr.ResearchRegistryError):
            tr.load_registry(str(path))

    def test_a_file_that_does_not_validate(self, tr: ModuleType, tmp_path: Path) -> None:
        text = ruleset_text(TECH_IDS).replace(
            "  - id: mining\n    era: 1\n    cost: 10",
            "  - id: mining\n    era: 1\n    cost: 10\n    prerequisites: [ghost]",
        )
        assert "ghost" in text, "fixture is vacuous"
        path = tmp_path / "invalid.yaml"
        path.write_text(text, encoding="utf-8")
        # It parses (the loader does not check structure) but names a tech that is not there.
        parse_ruleset(text)
        with pytest.raises(tr.ResearchRegistryError):
            tr.load_registry(str(path))

    def test_a_bad_gameinfo_raises(self, tr: ModuleType, tmp_path: Path) -> None:
        with pytest.raises(tr.ResearchRegistryError):
            tr.load_registry("science-v1-draft", gameinfo=str(tmp_path / "absent.json"))


# ---------------------------------------------------------------------------
# resolve_registry
# ---------------------------------------------------------------------------
class TestResolveRegistry:
    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_unset_or_blank_ruleset_is_none(self, tr: ModuleType, value: str | None) -> None:
        env = {} if value is None else {tr.RULESET_ENV: value}
        assert tr.resolve_registry(env) is None

    def test_gameinfo_without_a_ruleset_is_an_error(self, tr: ModuleType, tmp_path: Path) -> None:
        gameinfo = write_gameinfo(tmp_path / "gi.json", [{"CivicType": "CIVIC_A"}])
        with pytest.raises(tr.ResearchRegistryError):
            tr.resolve_registry({tr.GAMEINFO_ENV: str(gameinfo)})

    def test_a_builtin_id(self, tr: ModuleType) -> None:
        reg = tr.resolve_registry({tr.RULESET_ENV: "science-v1-draft"})
        assert reg == tr.load_registry("science-v1-draft")

    def test_ruleset_and_gameinfo(self, tr: ModuleType, tmp_path: Path) -> None:
        gameinfo = write_gameinfo(tmp_path / "gi.json", [{"CivicType": c} for c in CIVIC_TYPES])
        env = {tr.RULESET_ENV: "science-v1-draft", tr.GAMEINFO_ENV: str(gameinfo)}
        reg = tr.resolve_registry(env)
        assert reg == tr.load_registry("science-v1-draft", gameinfo=str(gameinfo))
        assert reg.civic_ids == tuple(c.lower() for c in CIVIC_TYPES)

    def test_a_configured_registry_that_cannot_load_fails_loudly(
        self, tr: ModuleType, tmp_path: Path
    ) -> None:
        with pytest.raises(tr.ResearchRegistryError):
            tr.resolve_registry({tr.RULESET_ENV: "no-such-ruleset"})
        with pytest.raises(tr.ResearchRegistryError):
            tr.resolve_registry(
                {tr.RULESET_ENV: "science-v1-draft", tr.GAMEINFO_ENV: str(tmp_path / "x.json")}
            )

    def test_default_env_is_os_environ(
        self, tr: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert tr.resolve_registry() is None
        monkeypatch.setenv(tr.RULESET_ENV, "science-v1-draft")
        reg = tr.resolve_registry()
        assert reg is not None
        assert reg.ruleset_id == "science-v1-draft"
        monkeypatch.setenv(tr.RULESET_ENV, "no-such-ruleset")
        with pytest.raises(tr.ResearchRegistryError):
            tr.resolve_registry()


# ---------------------------------------------------------------------------
# layering
# ---------------------------------------------------------------------------
class TestLayering:
    def test_rules_package_has_no_bridge_import(self) -> None:
        files = sorted(RULES_DIR.rglob("*.py"))
        assert files, "no rules sources found; the check below would be vacuous"
        offenders: list[str] = []
        for file in files:
            for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                    if node.module == "alpha_zero":
                        names += [f"alpha_zero.{a.name}" for a in node.names]
                offenders += [
                    f"{file.name}: {n}" for n in names if n.startswith("alpha_zero.bridge")
                ]
        assert offenders == []

    def test_tech_registry_source_imports_neither_torch_nor_numpy(self) -> None:
        source = REPO_ROOT / "alpha_zero" / "bridge" / "tech_registry.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        assert "alpha_zero" in roots or "dataclasses" in roots, "no imports read; vacuous"
        assert not roots & {"torch", "numpy"}

    def test_tech_registry_pulls_in_neither_torch_nor_numpy(self) -> None:
        code = (
            "import sys\n"
            "import alpha_zero.bridge\n"
            "before = set(sys.modules)\n"
            "import alpha_zero.bridge.tech_registry\n"
            "new = set(sys.modules) - before\n"
            "bad = sorted(m for m in new if m.split('.')[0] in ('torch', 'numpy'))\n"
            "print('LOADED' if 'alpha_zero.bridge.tech_registry' in sys.modules else 'NOT', bad)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "LOADED []"
