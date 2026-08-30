"""AQA meta-gate: a test suite that nothing runs is not a gate.

This generalises the worst finding of the PR #115 review. ``tests/regression/``
existed, passed when invoked by hand, and was enumerated by **no** CI job and
**no** make target -- so its invariants could regress with CI green. That was
fixed for one directory; the *category* was not, and five more suites were in
exactly the same position:

===================== ============================================
Suite                  What running it revealed
===================== ============================================
``tests/scripts/``     **3 real failures** -- optional-dependency
                       imports (``pandas``, ``mlflow``) written as
                       bare imports rather than ``importorskip``,
                       so they failed instead of skipping
``tests/smoke/``       30 passing tests, gating nothing
``tests/cloud/smoke/`` 36 passing tests, gating nothing
``tests/e2e/``         70 passing tests, gating nothing
``tests/inference/``   skip-guarded, gating nothing
===================== ============================================

So this file asserts the invariant directly: every directory under ``tests/``
that holds test modules is named by some pytest invocation in CI, or is listed
in :data:`NOT_GATED` with a stated reason. Adding a suite and forgetting to wire
it now fails here instead of passing silently for months.

The CI workflow is parsed with ``yaml``, not regex. That is deliberate: a hand
parser over YAML is how the skills validator ended up reducing a folded block
scalar to the literal ``">"`` and validating it vacuously. ``pyyaml`` is a hard
project dependency (CLAUDE.md 2.1), not an optional extra.
"""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = REPO_ROOT / "Makefile"
TESTS_ROOT = REPO_ROOT / "tests"

# Suites that deliberately do not gate, each with the reason. Same burndown
# pattern as ruff's PLR0915 per-file-ignores: an entry here is a visible debt,
# not an exemption to be added casually. Empty is the goal state.
NOT_GATED: dict[str, str] = {}

# `make test` mirrors CI's `test` + `cloud-tests` jobs. The Makefile says so
# itself ("The suites CI runs, in CI's order and with CI's marker filters"), and
# the two silently drifted on four axes before -- coverage threshold, coverage
# scope, mypy strictness and a validator flag -- which is why passing locally
# stopped meaning anything. These are the jobs it must stay in step with.
_MIRRORED_CI_JOBS = ("test", "cloud-tests")


def _defines_tests(path: Path) -> bool:
    """Does this module actually define collectible tests?

    Matching ``test_*.py`` is not enough, and neither is a ``Test``-prefixed
    class: ``tests/test_constants.py`` is a shared constants module holding
    ``class TestDefaults``, which pytest collects and then finds nothing in.
    Either shortcut would have demanded a CI wiring for a directory with no
    tests to run -- a gate whose subject does not exist.

    So this mirrors pytest's own rule: a module-level ``test_*`` function, or a
    ``Test*`` class with at least one ``test_*`` method.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - unreadable module
        return False

    def _is_test_callable(node: ast.AST) -> bool:
        return isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
            "test"
        )

    for node in tree.body:
        if _is_test_callable(node):
            return True
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            if any(_is_test_callable(child) for child in ast.walk(node)):
                return True
    return False


def _suites_on_disk() -> set[str]:
    """Every directory under ``tests/`` that directly contains real test modules.

    Directories holding only fixtures, mocks, constants or ``__init__.py`` are
    not suites and are not required to be gated.
    """
    suites = set()
    for path in TESTS_ROOT.rglob("test_*.py"):
        if "__pycache__" in path.parts:
            continue
        if not _defines_tests(path):
            continue
        suites.add(path.parent.relative_to(REPO_ROOT).as_posix())
    return suites


def _run_scripts_for_jobs(job_names: tuple[str, ...] | None = None) -> list[str]:
    """Every ``run:`` script in the CI workflow, optionally filtered to jobs."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    scripts: list[str] = []
    for name, job in (workflow.get("jobs") or {}).items():
        if job_names is not None and name not in job_names:
            continue
        for step in job.get("steps") or []:
            run = step.get("run")
            if isinstance(run, str):
                scripts.append(run)
    return scripts


def _test_paths_in(script: str) -> set[str]:
    """Extract the ``tests/...`` paths a pytest command actually runs.

    ``--ignore=`` values are excluded -- naming a path in order to skip it is
    the opposite of gating it. A file argument (``-c tests/x/pytest.ini``)
    counts for its directory, since that is how the governance job selects its
    suite.
    """
    found: set[str] = set()
    for line in script.splitlines():
        stripped = line.strip().rstrip("\\").strip()
        if "pytest" not in stripped:
            continue
        try:
            tokens = shlex.split(stripped, comments=True)
        except ValueError:  # pragma: no cover - unbalanced quotes in a run block
            continue
        for token in tokens:
            if token.startswith("--ignore"):
                continue
            match = re.fullmatch(r"(tests/[\w./-]*?)/?", token)
            if match is None:
                continue
            candidate = match.group(1)
            resolved = REPO_ROOT / candidate
            if resolved.is_file():
                candidate = resolved.parent.relative_to(REPO_ROOT).as_posix()
            found.add(candidate)
    return found


def _gated_paths(job_names: tuple[str, ...] | None = None) -> set[str]:
    paths: set[str] = set()
    for script in _run_scripts_for_jobs(job_names):
        paths |= _test_paths_in(script)
    return paths


def _make_recipe(target: str) -> str:
    """Return the recipe body of a Makefile target."""
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(target)}:.*?\n((?:\t.*\n)+)", text, re.M)
    return match.group(1) if match else ""


def _covers(gated: set[str], suite: str) -> bool:
    """Is ``suite`` run by any of the ``gated`` path arguments?"""
    return any(suite == path or suite.startswith(f"{path}/") for path in gated)


class TestEverySuiteIsGated:
    """The invariant this file exists for."""

    def test_every_test_directory_is_run_by_ci(self) -> None:
        gated = _gated_paths()
        orphaned = sorted(
            suite
            for suite in _suites_on_disk()
            if not _covers(gated, suite) and suite not in NOT_GATED
        )
        assert not orphaned, (
            f"test suites that no CI job runs: {orphaned}. A suite nothing invokes "
            "is not a gate -- its tests pass by hand and can regress with CI green. "
            "Add a pytest step in .github/workflows/ci.yml, or add an entry to "
            "NOT_GATED in this file stating why it deliberately does not gate."
        )

    def test_not_gated_entries_are_real_and_justified(self) -> None:
        """An exemption must name a suite that exists and say why."""
        on_disk = _suites_on_disk()
        for suite, reason in NOT_GATED.items():
            assert suite in on_disk, (
                f"NOT_GATED names {suite!r}, which holds no test modules -- "
                "stale exemptions hide the suites that really are ungated"
            )
            assert len(reason.strip()) >= 20, (
                f"NOT_GATED[{suite!r}] needs a real reason, not {reason!r}"
            )

    def test_make_test_mirrors_the_ci_jobs_it_claims_to(self) -> None:
        """`make test` says it runs "the suites CI runs" -- hold it to that."""
        ci_suites = _gated_paths(_MIRRORED_CI_JOBS)
        make_suites = _test_paths_in(_make_recipe("test"))
        missing = sorted(ci_suites - make_suites)
        extra = sorted(make_suites - ci_suites)
        assert not missing and not extra, (
            f"`make test` has drifted from CI's {list(_MIRRORED_CI_JOBS)} jobs — "
            f"missing {missing}, unexpected {extra}. When they drift, passing "
            "locally stops meaning anything, which is the drift this target exists "
            "to prevent."
        )


class TestTheGateItselfDiscriminates:
    """Negative controls. A meta-gate that cannot fail is the bug it checks for."""

    def test_parser_finds_the_suites_ci_visibly_runs(self) -> None:
        """Guards against a parser that returns nothing and proves nothing."""
        gated = _gated_paths()
        for expected in ("tests/unit", "tests/regression", "tests/integration"):
            assert expected in gated, (
                f"{expected} is run by a visible CI step but the parser missed it; "
                "the extraction is broken, so any pass here is vacuous"
            )

    def test_parser_does_not_match_the_tests_root_wholesale(self) -> None:
        """A bare ``tests`` argument would mark every suite covered vacuously."""
        assert "tests" not in _gated_paths(), (
            "a CI step runs the whole tests/ tree; that makes this gate unable to "
            "distinguish a wired suite from an unwired one"
        )

    def test_an_unwired_suite_is_detected(self) -> None:
        """The control the string-matching version of this idea would fail.

        Removes a suite from the parsed set and proves the coverage predicate
        reports it, without editing the repository.
        """
        gated = _gated_paths() - {"tests/regression"}
        assert not _covers(gated, "tests/regression"), (
            "dropping a suite from the gated set must make it uncovered, or this "
            "file cannot detect the defect it was written for"
        )

    def test_ignore_arguments_do_not_count_as_gating(self) -> None:
        """``--ignore=tests/x`` names a path in order *not* to run it."""
        script = "pytest tests/unit/ --ignore=tests/unit/test_distributed"
        paths = _test_paths_in(script)
        assert paths == {"tests/unit"}, f"--ignore= value leaked into the gated set: {paths}"

    def test_config_file_argument_counts_for_its_directory(self) -> None:
        """How the governance job selects its suite: ``-c <dir>/pytest.ini``."""
        script = "pytest -c tests/governance/pytest.ini"
        assert "tests/governance" in _test_paths_in(script)


class TestOptionalDependencyImportsAreGuarded:
    """`tests/scripts/` failed, rather than skipped, on absent optional extras.

    Three tests imported ``pandas``/``mlflow`` directly. Both arrive with the
    ``monitoring`` extra, which CI's ``.[dev]`` install does not provide, so they
    were failures -- invisible only because nothing ran the suite. The repo's
    convention everywhere else is ``pytest.importorskip`` (see
    ``tests/unit/test_serving/``).
    """

    @pytest.mark.parametrize("module", ["pandas", "mlflow"])
    def test_optional_modules_are_not_imported_bare(self, module: str) -> None:
        source = (REPO_ROOT / "tests" / "scripts" / "test_run_baseline_eval.py").read_text(
            encoding="utf-8"
        )
        bare = re.search(rf"^\s*import {module}\b", source, re.M)
        assert bare is None, (
            f"tests/scripts/test_run_baseline_eval.py imports {module} directly; it "
            "is an optional extra absent from CI's [dev] install, so this is a "
            "failure rather than a skip. Use pytest.importorskip."
        )
