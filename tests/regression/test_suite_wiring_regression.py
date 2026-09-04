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
import sys
from pathlib import Path

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


def _is_module_level_skip_guarded(path: Path) -> bool:
    """Is the whole module skipped at import time by ``pytest.importorskip``?

    A module-level ``importorskip`` aborts collection for the file, so a suite
    where *every* module is guarded collects nothing when its optional extra is
    absent -- and pytest exits **5** ("no tests collected"), failing the build.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - unreadable module
        return False
    for node in tree.body:
        if not isinstance(node, ast.Expr | ast.Assign):
            continue
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "importorskip"
        ):
            return True
    return False


def _collects_nothing_without_extras(suite: str) -> bool:
    """Would this suite collect zero tests on a plain ``[dev]`` install?"""
    modules = [path for path in (REPO_ROOT / suite).glob("test_*.py") if _defines_tests(path)]
    return bool(modules) and all(_is_module_level_skip_guarded(path) for path in modules)


_MIN_REASON_CHARS = 20


def _not_gated_problems(entries: dict[str, str], on_disk: set[str]) -> list[str]:
    """Why each ``NOT_GATED`` entry is unacceptable, if any.

    Two rules. The suite must exist -- a stale exemption silently covers for a
    suite that no longer has that name, and hides the ones really ungated. And
    the reason must be a real sentence: an exemption whose justification is
    ``"todo"`` is a wired-off gate with a comment on it.
    """
    problems = []
    for suite, reason in sorted(entries.items()):
        if suite not in on_disk:
            problems.append(
                f"NOT_GATED names {suite!r}, which holds no test modules -- "
                "stale exemptions hide the suites that really are ungated"
            )
        if len(reason.strip()) < _MIN_REASON_CHARS:
            problems.append(f"NOT_GATED[{suite!r}] needs a real reason, not {reason!r}")
    return problems


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
        """An exemption must name a suite that exists and say why.

        ``NOT_GATED`` is empty today, so this loop runs zero times and passes
        vacuously -- which is the exact defect this file exists to catch. The
        rule is therefore extracted into :func:`_not_gated_problems` and
        exercised on synthetic entries in ``TestTheGateItselfDiscriminates``,
        so it is proven to discriminate before anyone relies on it.
        """
        problems = _not_gated_problems(NOT_GATED, _suites_on_disk())
        assert not problems, "\n".join(problems)

    def test_no_ci_step_runs_only_fully_skip_guarded_suites(self) -> None:
        """A step whose every suite is optional-extra-guarded exits 5, not 0.

        ``pytest tests/inference/`` collected 0 items and returned pytest's
        "no tests collected" code, turning CI red for a suite behaving exactly
        as designed. Locally the output read ``1 skipped`` and looked fine --
        the failure was in the exit code, which is the only part CI reads.

        The fix is to pair such a suite with a populated one; the point of this
        assertion is that the next one cannot be given its own step by mistake.
        """
        offenders = []
        for script in _run_scripts_for_jobs():
            for line in script.splitlines():
                if "pytest" not in line:
                    continue
                suites = _test_paths_in(line)
                if suites and all(_collects_nothing_without_extras(s) for s in suites):
                    offenders.append(sorted(suites))
        assert not offenders, (
            f"CI steps run only fully skip-guarded suites: {offenders}. Without "
            "the optional extra these collect nothing and pytest exits 5, so the "
            "step fails. Pair each with a suite that always collects."
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

    def test_a_stale_exemption_is_rejected(self) -> None:
        """``NOT_GATED`` is empty, so its own check would pass on anything."""
        problems = _not_gated_problems(
            {"tests/deleted-long-ago": "a perfectly adequate sentence of reasoning"},
            _suites_on_disk(),
        )
        assert problems, "an exemption naming a nonexistent suite must be rejected"

    def test_an_unjustified_exemption_is_rejected(self) -> None:
        real_suite = next(iter(sorted(_suites_on_disk())))
        problems = _not_gated_problems({real_suite: "todo"}, _suites_on_disk())
        assert problems, "an exemption with a placeholder reason must be rejected"

    def test_a_well_formed_exemption_is_accepted(self) -> None:
        """The positive half: the rule must not reject everything either."""
        real_suite = next(iter(sorted(_suites_on_disk())))
        problems = _not_gated_problems(
            {real_suite: "deliberately not gated because it needs live GCP credentials"},
            _suites_on_disk(),
        )
        assert not problems, f"a valid exemption was rejected: {problems}"


class TestOptionalDependencyImportsAreGuarded:
    """A test module must not import an optional extra at module scope.

    Doing so turns an absent extra into a COLLECTION ERROR for the whole module
    and, because pytest aborts the run, for every other module in the same
    invocation. Measured: with ``chess`` absent -- it is not in the ``dev`` extra
    because its sdist fails to build on Debian/Ubuntu setuptools -- ``pytest
    tests/unit`` collected **0** tests instead of 9,916.

    The previous version read ONE file and grepped for TWO module names, so it
    could not see the eleven modules with the same defect. It now walks every
    test module with the AST and derives the allowed set from ``pyproject.toml``,
    so a new base dependency is permitted automatically and a new optional one is
    caught automatically -- no hand-kept list to drift.
    """

    #: Import name -> distribution name, only where they differ. Anything absent
    #: falls back to the import name, which is right for most packages.
    IMPORT_TO_DISTRIBUTION = {
        "yaml": "pyyaml",
        "sklearn": "scikit-learn",
        "PIL": "pillow",
        "cv2": "opencv-python",
        "dotenv": "python-dotenv",
        "attr": "attrs",
    }

    #: The test runner itself. Everything else that is always importable is
    #: DERIVED from the repository layout by :meth:`_repo_local_roots`, so adding
    #: a top-level module or package does not require editing this file.
    RUNNER_MODULES = frozenset({"pytest", "_pytest", "conftest"})

    @staticmethod
    def _repo_local_roots() -> set[str]:
        """Import roots that resolve inside this repository, not from PyPI.

        A test importing ``demos`` or ``run_civ6_ai`` is importing repo-local
        code that is present by construction; only a *distribution* can be
        absent. Computed from the tree so a new top-level module is covered
        automatically.
        """
        roots: set[str] = set()
        for entry in REPO_ROOT.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_dir() and (entry / "__init__.py").exists():
                roots.add(entry.name)
            elif entry.is_dir() and any(entry.glob("*.py")):
                # A namespace-style directory (no __init__) is still importable
                # when the repo root is on sys.path, which conftest arranges.
                roots.add(entry.name)
            elif entry.suffix == ".py":
                roots.add(entry.stem)

        # Modules one level down that tests import after a sys.path insert --
        # `scripts/gcp_diagnose.py` is imported as bare `gcp_diagnose`. Only
        # direct children, so this stays a bounded, derived set rather than a
        # walk of the whole tree.
        for parent in ("scripts", "infra", "demos", "examples"):
            directory = REPO_ROOT / parent
            if directory.is_dir():
                roots.update(f.stem for f in directory.glob("*.py"))
        return roots

    @staticmethod
    def _base_distributions() -> set[str]:
        """Distributions CI guarantees: base dependencies plus the ``dev`` extra."""
        import tomllib

        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = data.get("project", {})
        specs = list(project.get("dependencies", []))
        specs += list(project.get("optional-dependencies", {}).get("dev", []))
        names: set[str] = set()
        for spec in specs:
            name = re.split(r"[<>=!\[;~ ]", spec, maxsplit=1)[0].strip().lower()
            if name:
                names.add(name.replace("_", "-"))
        return names

    @staticmethod
    def _module_scope_imports(tree: ast.Module) -> set[str]:
        """Top-level import roots only.

        A function-local import is deferred and cannot break collection, so only
        ``tree.body`` is inspected rather than the whole AST.
        """
        roots: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        return roots

    @staticmethod
    def _guarded_names(tree: ast.Module) -> set[str]:
        """Modules named by any ``pytest.importorskip("...")`` call in the module."""
        guarded: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "importorskip"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                guarded.add(node.args[0].value.split(".")[0])
        return guarded

    def test_no_test_module_imports_an_optional_extra_bare(self) -> None:
        base = self._base_distributions()
        allowed_roots = self._repo_local_roots() | self.RUNNER_MODULES
        assert base, "pyproject declared no dependencies; this check would be vacuous"
        offenders: list[str] = []

        for path in sorted((REPO_ROOT / "tests").rglob("test_*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
                continue
            guarded = self._guarded_names(tree)
            for root in self._module_scope_imports(tree):
                if root in allowed_roots or root in guarded:
                    continue
                if root in sys.stdlib_module_names:
                    continue
                distribution = self.IMPORT_TO_DISTRIBUTION.get(root, root).lower()
                if distribution.replace("_", "-") in base:
                    continue
                offenders.append(f"{path.relative_to(REPO_ROOT)}: import {root}")

        assert not offenders, (
            "test modules import an optional extra at module scope, which turns an "
            "absent extra into a collection error for the whole pytest run:\n  "
            + "\n  ".join(offenders)
            + "\nGuard each with pytest.importorskip('<module>', reason=...) before the "
            "import, and add the file to pyproject's per-file E402 ignores."
        )

    def test_the_check_detects_a_planted_bare_import(self, tmp_path: Path) -> None:
        """Negative control: an unguarded optional import must be visible."""
        planted = tmp_path / "test_planted.py"
        planted.write_text("import chess\n\n\ndef test_x():\n    assert chess\n")
        tree = ast.parse(planted.read_text())
        assert "chess" in self._module_scope_imports(tree)
        assert "chess" not in self._guarded_names(tree)

    def test_the_check_accepts_the_guarded_form(self, tmp_path: Path) -> None:
        """And must not flag the guarded form, or it would forbid its own fix."""
        planted = tmp_path / "test_guarded.py"
        planted.write_text("import pytest\n\npytest.importorskip('chess')\n\nimport chess\n")
        tree = ast.parse(planted.read_text())
        assert "chess" in self._guarded_names(tree)


class TestSearchTestsAssertSomething:
    """A test on the search path must assert something.

    ``test_performance_better_than_random`` in ``tests/unit/test_gumbel_mcts.py``
    contained zero assert statements -- its own comment read "just checking it
    doesn't crash" -- while sitting on ``DEFAULT_MCTS_TYPE``. It could not have
    detected the value-backup sign defect, the one-hot policy target, or an
    illegal action, and a repository-wide census counted it as test-quality debt
    without anyone noticing what it was guarding.

    Scoped to the MCTS suites rather than all of ``tests/``: the repo-wide count
    is 166 and burning that down is a separate piece of work, but the search is
    where a silent pass is most expensive.

    Negative control: emptying any test body in these files must fail
    ``test_no_search_test_is_assertion_free``. Verified by stripping the body of
    ``test_plays_a_complete_game_producing_valid_policies`` and observing the
    failure name that test.
    """

    #: Helper names that assert on the caller's behalf. A test whose body calls
    #: one of these is doing real checking even with no bare ``assert``.
    ASSERTING_CALLS = frozenset(
        {
            "raises",
            "warns",
            "approx",
            "fail",
            "assert_called",
            "assert_called_once",
            "assert_called_with",
            "assert_called_once_with",
            "assert_any_call",
            "assert_has_calls",
            "assert_not_called",
            "assert_array_equal",
            "assert_allclose",
            "assert_array_almost_equal",
        }
    )

    #: Tests whose contract genuinely IS "this must not raise", listed with the
    #: reason so the absence of an assertion is a recorded decision rather than
    #: an oversight -- the same pattern as NOT_GATED and the per-file lint
    #: ignores. A new entry needs a reason; an empty reason fails the check.
    ASSERTION_FREE_BY_DESIGN: dict[str, str] = {
        "tests/unit/test_jax_mcts_adapter.py::test_eval_is_noop": (
            "the adapter's eval() is a documented no-op; the contract is that "
            "calling it neither raises nor mutates state"
        ),
        "tests/unit/test_jax_mcts_adapter.py::test_train_is_noop": (
            "same contract as test_eval_is_noop, for train()"
        ),
        "tests/unit/test_mcts_backup_correctness.py::"
        "test_backpropagate_path_is_a_noop_on_an_empty_path": (
            "an empty search path must be tolerated silently; there is no state "
            "left to assert on afterwards"
        ),
    }

    @staticmethod
    def _search_test_files() -> list[Path]:
        return sorted((REPO_ROOT / "tests" / "unit").glob("test_*mcts*.py"))

    @staticmethod
    def _is_fixture(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """A fixture named ``test_*`` is not a test, despite the name."""
        for decorator in func.decorator_list:
            node = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(node, ast.Attribute) and node.attr == "fixture":
                return True
            if isinstance(node, ast.Name) and node.id == "fixture":
                return True
        return False

    @classmethod
    def _asserts_something(cls, func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        for node in ast.walk(func):
            if isinstance(node, ast.Assert):
                return True
            if isinstance(node, ast.Raise):
                return True
            if isinstance(node, ast.Call):
                name = ""
                if isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    name = node.func.id
                if name in cls.ASSERTING_CALLS or name.startswith("assert"):
                    return True
            # `with pytest.raises(...)` is an ast.Call, already covered above.
        return False

    def test_search_suites_are_discoverable(self) -> None:
        """Guard against the whole class passing because it matched no files."""
        files = self._search_test_files()
        assert files, "no tests/unit/test_*mcts*.py files found; this class is vacuous"

    def test_no_search_test_is_assertion_free(self) -> None:
        offenders: list[str] = []
        checked = 0
        for path in self._search_test_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if not node.name.startswith("test_") or self._is_fixture(node):
                    continue
                checked += 1
                qualified = f"{path.relative_to(REPO_ROOT)}::{node.name}"
                if qualified in self.ASSERTION_FREE_BY_DESIGN:
                    continue
                if not self._asserts_something(node):
                    offenders.append(qualified)

        assert checked > 0, "no test functions were inspected; the assertion below is vacuous"
        assert not offenders, (
            "search tests with no assertion (they pass whatever the search does):\n  "
            + "\n  ".join(offenders)
            + "\nAssert the contract the test's name promises, or delete it."
        )

    def test_every_assertion_free_exemption_has_a_reason(self) -> None:
        """An exemption without a reason is an unrecorded decision."""
        empty = [name for name, why in self.ASSERTION_FREE_BY_DESIGN.items() if not why.strip()]
        assert not empty, f"exemptions with no reason: {empty}"

    def test_no_exemption_is_stale(self) -> None:
        """An exemption for a test that now asserts, or no longer exists, must go."""
        stale: list[str] = []
        for qualified in self.ASSERTION_FREE_BY_DESIGN:
            rel, _, func_name = qualified.partition("::")
            path = REPO_ROOT / rel
            if not path.exists():
                stale.append(f"{qualified} (file is gone)")
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            found = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == func_name
            ]
            if not found:
                stale.append(f"{qualified} (test is gone)")
            elif any(self._asserts_something(n) for n in found):
                stale.append(f"{qualified} (it asserts now -- drop the exemption)")
        assert not stale, "stale assertion-free exemptions: " + "; ".join(stale)
