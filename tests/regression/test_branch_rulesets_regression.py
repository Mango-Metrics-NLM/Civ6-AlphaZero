"""The repository rulesets (``.github/rulesets/*.json``, plan S3.3) against the workflows.

A required status check that no job produces is a branch nobody can merge into;
one produced by a ``continue-on-error`` job or a path-filtered workflow is a
check that can be green while the thing it names failed or never ran. So every
context any ruleset requires must be the name of a real ``pull_request`` job --
matrix legs expanded the way GitHub expands them (``ci.yml``'s ``Test (Python
${{ matrix.python-version }})``), evaluated per leg for ``continue-on-error``.

The rest pins the shape S3.1/S3.3 decide: who may bypass (nobody on qa/main;
admins on dev in ``pull_request`` mode only; the maintainer as a ``User`` on the
merge and tag rulesets, never in ``pull_request`` mode on a tag ruleset, where
GitHub rejects it), strictness (qa off -- its own merge commits would make every
later promotion "out of date" -- dev/main on), and merge methods.

Field names follow the GitHub REST rulesets schema (``github/docs``
``src/rest/data/fpt-2026-03-10/repos.json``); ``integration_id`` is deliberately
omitted -- see ``scripts/maintainer/rulesets.py``.

Negative controls live in ``TestThePredicatesDiscriminate`` on in-memory YAML.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.regression.branch_policy import protected_branches

RULESET_DIR = Path(".github/rulesets")
WORKFLOW_DIR = Path(".github/workflows")
SENTINEL = "{{MAINTAINER_USER_ID}}"
LEGACY_DEFAULT = "claude/civ6-alphazero-implementation-vMnun"

#: Plan S3.3, verbatim. Release Evidence joins main.json only in T14.
REQUIRED_CHECKS = frozenset(
    {
        "Lint & Type Check",
        "Test (Python 3.11)",
        "Test (Python 3.12)",
        "Cloud Unit Tests",
        "Security Scan",
        "Governance Gates",
        "LLM Eval Harness",
        "Coverage Report",
        "Branch Direction",
    }
)
NEVER_REQUIRED = (
    "Test (Python 3.13)",
    "contributor-smoke",
    "CodeQL",
    "Scorecard",
    "Dependency review",
)
EXPECTED_FILES = {"dev", "qa", "main", "qa-main-merge", "release-tags", "legacy-name"}
RULE_TYPES = {
    "creation",
    "update",
    "deletion",
    "required_linear_history",
    "merge_queue",
    "required_deployments",
    "required_signatures",
    "pull_request",
    "required_status_checks",
    "non_fast_forward",
}
PULL_REQUEST_REQUIRED = {
    "dismiss_stale_reviews_on_push",
    "require_code_owner_review",
    "require_last_push_approval",
    "required_approving_review_count",
    "required_review_thread_resolution",
}


def _load(name: str) -> dict[str, Any]:
    return json.loads((RULESET_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _rulesets() -> dict[str, dict[str, Any]]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(RULESET_DIR.glob("*.json"))
    }


def _rule(doc: dict[str, Any], rule_type: str) -> dict[str, Any] | None:
    return next((r for r in doc["rules"] if r["type"] == rule_type), None)


def required_contexts(doc: dict[str, Any]) -> list[str]:
    rule = _rule(doc, "required_status_checks")
    if rule is None:
        return []
    return [c["context"] for c in rule["parameters"]["required_status_checks"]]


# ---------------------------------------------------------------------------
# Workflow job enumeration, with GitHub's matrix semantics
# ---------------------------------------------------------------------------


def _triggers(doc: dict[str, Any]) -> dict[str, Any]:
    on = doc.get(True, doc.get("on", {}))  # PyYAML reads the bare key `on` as True
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list):
        return dict.fromkeys(on)
    return on or {}


def _runs_on_every_protected_pr(doc: dict[str, Any], protected: tuple[str, ...]) -> bool:
    """A ``pull_request`` trigger that fires for every PR into a protected branch."""
    triggers = _triggers(doc)
    if "pull_request" not in triggers:
        return False
    pr = triggers["pull_request"] or {}
    if pr.get("paths") or pr.get("paths-ignore"):
        return False
    if pr.get("branches-ignore"):
        return False
    types = pr.get("types")
    if types and not {"opened", "synchronize", "reopened"} <= set(types):
        return False
    branches = pr.get("branches")
    return not branches or all(b in branches for b in protected)


def expand_matrix(strategy: dict[str, Any] | None) -> list[dict[str, Any]]:
    """GitHub's expansion: the cartesian product, then ``include``/``exclude``.

    An ``include`` entry extends every combination whose ORIGINAL values it does
    not contradict; if it extends none, it becomes a new combination.
    """
    matrix = (strategy or {}).get("matrix")
    if not isinstance(matrix, dict):
        return [{}]
    axes = {k: v for k, v in matrix.items() if k not in ("include", "exclude")}
    if not axes:  # an include-only matrix: each entry is one leg
        return [dict(inc) for inc in matrix.get("include") or []] or [{}]
    keys = list(axes)
    combos = [dict(zip(keys, values, strict=True)) for values in itertools.product(*axes.values())]
    for exclusion in matrix.get("exclude") or []:
        combos = [c for c in combos if any(c.get(k) != v for k, v in exclusion.items())]
    originals = [dict(c) for c in combos]
    extra: list[dict[str, Any]] = []
    for inc in matrix.get("include") or []:
        matched = False
        for combo, original in zip(combos, originals, strict=True):
            if all(original.get(k, v) == v for k, v in inc.items() if k in original):
                combo.update({k: v for k, v in inc.items() if k not in original})
                matched = True
        if not matched:
            extra.append(dict(inc))
    return combos + extra


_EXPR = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}")


def _render(template: str, leg: dict[str, Any]) -> str:
    return _EXPR.sub(lambda m: str(leg.get(m.group(1), m.group(0))), template)


def _continue_on_error(value: Any, leg: dict[str, Any]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        m = _EXPR.fullmatch(value.strip())
        if m:
            return bool(leg.get(m.group(1)))
        return value.strip().lower() != "false"  # any other expression: assume it can be true
    return False


def pr_check_names(doc: dict[str, Any], protected: tuple[str, ...]) -> dict[str, bool]:
    """Check-run name -> ``continue-on-error`` for every leg of every PR job.

    A job with an ``if:`` is excluded (it may be skipped, and a skipped required
    check is a merge nobody can make or one made on nothing).
    """
    if not _runs_on_every_protected_pr(doc, protected):
        return {}
    out: dict[str, bool] = {}
    for job_id, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict) or "if" in job:
            continue
        name = str(job.get("name", job_id))
        for leg in expand_matrix(job.get("strategy")):
            out[_render(name, leg)] = _continue_on_error(job.get("continue-on-error", False), leg)
    return out


def all_pr_checks() -> dict[str, bool]:
    names: dict[str, bool] = {}
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, coe in pr_check_names(doc, ("dev", "qa", "main")).items():
            names[name] = names.get(name, False) or coe
    return names


# ---------------------------------------------------------------------------
# The real rulesets
# ---------------------------------------------------------------------------


def test_the_expected_rulesets_exist() -> None:
    assert set(_rulesets()) == EXPECTED_FILES


@pytest.mark.parametrize("name", sorted(EXPECTED_FILES))
def test_every_required_check_is_a_real_blocking_pr_job(name: str) -> None:
    checks = all_pr_checks()
    assert checks, "no pull_request job found at all; this test would be vacuous"
    contexts = required_contexts(_load(name))
    missing = [c for c in contexts if c not in checks]
    advisory = [c for c in contexts if checks.get(c)]
    assert not missing, (
        f"{name}.json requires {missing}, which no unfiltered pull_request job produces"
    )
    assert not advisory, f"{name}.json requires {advisory}, produced by a continue-on-error job"


@pytest.mark.parametrize("name", ["dev", "qa"])
def test_dev_and_qa_require_exactly_the_plan_set(name: str) -> None:
    assert set(required_contexts(_load(name))) == REQUIRED_CHECKS


def test_main_requires_the_dev_qa_set_and_at_most_release_evidence() -> None:
    contexts = set(required_contexts(_load("main")))
    assert contexts >= REQUIRED_CHECKS
    assert contexts - REQUIRED_CHECKS <= {"Release Evidence"}


@pytest.mark.parametrize("name", sorted(EXPECTED_FILES))
def test_never_required_checks_stay_unrequired(name: str) -> None:
    contexts = required_contexts(_load(name))
    assert not [
        c for c in contexts if c in NEVER_REQUIRED or c.lower() in map(str.lower, NEVER_REQUIRED)
    ]


@pytest.mark.parametrize(("name", "strict"), [("dev", True), ("qa", False), ("main", True)])
def test_strictness(name: str, strict: bool) -> None:
    rule = _rule(_load(name), "required_status_checks")
    assert rule is not None
    assert rule["parameters"]["strict_required_status_checks_policy"] is strict


@pytest.mark.parametrize(
    ("name", "methods"), [("dev", {"squash", "merge"}), ("qa", {"merge"}), ("main", {"merge"})]
)
def test_merge_methods_and_review_settings(name: str, methods: set[str]) -> None:
    rule = _rule(_load(name), "pull_request")
    assert rule is not None
    params = rule["parameters"]
    assert set(params["allowed_merge_methods"]) == methods
    assert "rebase" not in params["allowed_merge_methods"]
    assert params["required_approving_review_count"] == 0
    assert params["required_review_thread_resolution"] is True
    assert PULL_REQUEST_REQUIRED <= set(params)


@pytest.mark.parametrize("name", ["dev", "qa", "main"])
def test_branch_rulesets_target_exactly_their_branch(name: str) -> None:
    doc = _load(name)
    assert doc["target"] == "branch"
    assert doc["enforcement"] == "active"
    assert doc["conditions"]["ref_name"]["include"] == [f"refs/heads/{name}"]
    types = {r["type"] for r in doc["rules"]}
    assert {"deletion", "non_fast_forward", "pull_request", "required_status_checks"} <= types
    assert "required_linear_history" not in types  # S3.1: there is no linear-history rule


@pytest.mark.parametrize("name", ["qa", "main"])
def test_qa_and_main_have_no_bypass(name: str) -> None:
    assert _load(name)["bypass_actors"] == []


def test_dev_bypass_is_admin_in_pull_request_mode_only() -> None:
    actors = _load("dev")["bypass_actors"]
    assert actors == [
        {"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "pull_request"}
    ]


def test_qa_main_merge_lets_only_the_maintainer_update() -> None:
    doc = _load("qa-main-merge")
    assert doc["target"] == "branch"
    assert sorted(doc["conditions"]["ref_name"]["include"]) == ["refs/heads/main", "refs/heads/qa"]
    assert doc["rules"] == [
        {"type": "update", "parameters": {"update_allows_fetch_and_merge": False}}
    ]
    assert doc["bypass_actors"] == [
        {"actor_id": SENTINEL, "actor_type": "User", "bypass_mode": "always"}
    ]


def test_release_tags_ruleset() -> None:
    doc = _load("release-tags")
    assert doc["target"] == "tag"
    assert doc["conditions"]["ref_name"]["include"] == ["refs/tags/v*"]
    assert {r["type"] for r in doc["rules"]} == {
        "creation",
        "update",
        "deletion",
        "non_fast_forward",
    }
    assert doc["bypass_actors"] == [
        {"actor_id": SENTINEL, "actor_type": "User", "bypass_mode": "always"}
    ]


@pytest.mark.parametrize("name", sorted(EXPECTED_FILES))
def test_no_tag_ruleset_uses_pull_request_bypass(name: str) -> None:
    doc = _load(name)
    if doc["target"] == "tag":
        assert all(a.get("bypass_mode") != "pull_request" for a in doc["bypass_actors"])


def test_legacy_name_blocks_recreating_the_old_default() -> None:
    doc = _load("legacy-name")
    assert doc["target"] == "branch"
    assert doc["conditions"]["ref_name"]["include"] == [f"refs/heads/{LEGACY_DEFAULT}"]
    assert {r["type"] for r in doc["rules"]} >= {"creation"}
    assert doc["bypass_actors"] == []


@pytest.mark.parametrize("name", sorted(EXPECTED_FILES))
def test_schema_shape(name: str) -> None:
    doc = _load(name)
    assert set(doc) == {"name", "target", "enforcement", "bypass_actors", "conditions", "rules"}
    assert doc["name"] == name
    assert doc["target"] in {"branch", "tag"}
    assert doc["enforcement"] == "active"
    assert set(doc["conditions"]) == {"ref_name"}
    assert set(doc["conditions"]["ref_name"]) <= {"include", "exclude"}
    for rule in doc["rules"]:
        assert rule["type"] in RULE_TYPES
    rsc = _rule(doc, "required_status_checks")
    if rsc is not None:
        params = rsc["parameters"]
        assert {"required_status_checks", "strict_required_status_checks_policy"} <= set(params)
        for check in params["required_status_checks"]:
            # integration_id omitted on purpose (see scripts/maintainer/rulesets.py)
            assert set(check) == {"context"}
    for actor in doc["bypass_actors"]:
        assert set(actor) == {"actor_id", "actor_type", "bypass_mode"}
        assert actor["actor_type"] in {"RepositoryRole", "User"}
        assert actor["bypass_mode"] in {"always", "pull_request"}
        if actor["actor_type"] == "User":
            # never commit a personal id; branch_cutover.sh substitutes it at apply time
            assert actor["actor_id"] == SENTINEL
        else:
            assert isinstance(actor["actor_id"], int)


def test_every_protected_release_branch_has_a_ruleset() -> None:
    covered = {
        ref.removeprefix("refs/heads/")
        for doc in _rulesets().values()
        if doc["target"] == "branch"
        for ref in doc["conditions"]["ref_name"]["include"]
    }
    assert {"dev", "qa", "main"} <= covered
    assert {"dev", "qa", "main"} <= set(protected_branches())


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


class TestThePredicatesDiscriminate:
    _CI = """
on:
  pull_request:
jobs:
  test:
    name: Test (Python ${{ matrix.python-version }})
    continue-on-error: ${{ matrix.experimental }}
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
        experimental: [false]
        include:
          - python-version: "3.13"
            experimental: true
  lint:
    name: Lint
  maybe:
    name: Maybe
    if: github.event_name == 'push'
"""

    def test_matrix_legs_expand_with_include_as_a_new_leg(self) -> None:
        names = pr_check_names(yaml.safe_load(self._CI), ("dev", "qa", "main"))
        assert names == {
            "Test (Python 3.11)": False,
            "Test (Python 3.12)": False,
            "Test (Python 3.13)": True,
            "Lint": False,
        }

    def test_include_extending_an_existing_leg_does_not_add_one(self) -> None:
        strategy = {"matrix": {"py": ["3.11", "3.12"], "include": [{"py": "3.12", "extra": 1}]}}
        assert expand_matrix(strategy) == [{"py": "3.11"}, {"py": "3.12", "extra": 1}]

    def test_a_path_filtered_workflow_produces_no_required_checks(self) -> None:
        doc = yaml.safe_load(
            self._CI.replace("pull_request:\n", "pull_request:\n    paths: ['src/**']\n")
        )
        assert pr_check_names(doc, ("dev", "qa", "main")) == {}

    def test_a_branch_filter_missing_qa_produces_no_required_checks(self) -> None:
        doc = yaml.safe_load(
            self._CI.replace("pull_request:\n", "pull_request:\n    branches: [dev, main]\n")
        )
        assert pr_check_names(doc, ("dev", "qa", "main")) == {}

    def test_types_without_synchronize_produce_no_required_checks(self) -> None:
        doc = yaml.safe_load(
            self._CI.replace("pull_request:\n", "pull_request:\n    types: [opened]\n")
        )
        assert pr_check_names(doc, ("dev", "qa", "main")) == {}

    def test_a_push_only_workflow_produces_no_required_checks(self) -> None:
        doc = yaml.safe_load(self._CI.replace("pull_request:", "push:"))
        assert pr_check_names(doc, ("dev", "qa", "main")) == {}

    def test_a_ruleset_naming_a_continue_on_error_leg_would_fail(self) -> None:
        checks = pr_check_names(yaml.safe_load(self._CI), ("dev", "qa", "main"))
        planted = {"rules": [{"type": "required_status_checks", "parameters": {
            "strict_required_status_checks_policy": True,
            "required_status_checks": [{"context": "Test (Python 3.13)"}, {"context": "Ghost"}],
        }}]}  # fmt: skip
        contexts = required_contexts(planted)
        assert [c for c in contexts if c not in checks] == ["Ghost"]
        assert [c for c in contexts if checks.get(c)] == ["Test (Python 3.13)"]
