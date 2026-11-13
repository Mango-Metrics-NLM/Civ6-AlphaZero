"""The release play path covers what the A1 training and benchmark code actually imports (I-4).

``scripts/release_play_path.txt`` lists the code whose change makes an evidence run
stale. It was written by hand and missed modules ``training_civ6`` imports directly
(the value targets in ``multiplayer/scoring.py``, ``load_network`` in
``utils/checkpoint.py``, ...), so an ``rcfix`` to one of them would have left a
release bundle "fresh" while the trained loop differed from the one judged.

This test derives the set instead of trusting the list: it computes the static
``alpha_zero`` import closure of the A1 entry points and requires every module in it
to be either covered by a play-path pathspec or named in :data:`EXEMPT` with a reason.
A new import on the play path therefore fails here until someone decides which side
of the line it is on.

How the closure is computed (reusing ``scripts/audit_import_graph.py``'s module
table, relative-import resolver and package re-export tables):

* ``import a.b.c`` is an edge to the deepest existing module of that name;
* ``from pkg import name`` is an edge to ``pkg.name`` when that is a module, else to
  the submodule ``pkg/__init__`` re-exports ``name`` from, else to ``pkg``;
* literal ``importlib.import_module("...")`` calls count;
* function-local imports count (they run when the function does);
* imports under ``if TYPE_CHECKING:`` do not (they never run).

What it does not see: a package ``__init__`` executed only because a submodule was
imported is not followed (the re-export tables already route a name to the module
that defines it), and dispatch by a non-literal string is invisible. Both are why
the registries (``mcts_factory``, ``encoder_registry``) are listed explicitly.
"""

from __future__ import annotations

import ast
import re
from collections import deque
from collections.abc import Iterator
from functools import cache
from pathlib import Path

from scripts import audit_import_graph as aig
from scripts import verify_release_evidence as vre

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The A1 entry points: the registered training command and the benchmark
#: (``scripts/benchmark_suite.py`` is a thin wrapper over it, and is on the play path).
SEEDS = ("alpha_zero.training_civ6", "alpha_zero.evaluation.benchmark")

#: Modules in the closure that cannot change how a fixed network trains or plays.
EXEMPT: dict[str, str] = {
    "alpha_zero.cloud.gcp.exceptions": "exception types only",
    "alpha_zero.core.muzero_network": (
        "reached only through the MuZero search variant; section 7.1 registers "
        "--mcts gumbel, and core/mcts/** already makes any MuZero search change stale"
    ),
    "alpha_zero.core.losses": "MuZero losses, reached only through core.muzero_network",
    "alpha_zero.monitoring.config": "MLflow configuration: tracking only",
    "alpha_zero.monitoring.metrics": "diagnostic statistics that are logged, never fed back",
    "alpha_zero.monitoring.mlflow_logger": "MLflow logging only",
    "alpha_zero.storage.models": (
        "result record dataclass; the verifier recomputes every statistic from W/D/L"
    ),
    "alpha_zero.training.diagnostics": "per-step diagnostic records; written, never read back",
    "alpha_zero.training.run_record": (
        "iterations.jsonl identity record; the verifier checks its fields directly"
    ),
    "alpha_zero.utils.compat": "datetime.UTC shim",
    "alpha_zero.utils.data_root": "where files are written, not what is computed",
    "alpha_zero.utils.experiment_tracker": "metric logging only",
    "alpha_zero.utils.game_budget": (
        "prices and refuses before a run starts; it cannot change a run that starts"
    ),
    "alpha_zero.utils.logging_config": "logging only",
}


@cache
def _graph() -> tuple[dict[str, Path], dict[str, ast.Module], dict[str, dict[str, str]]]:
    modules: dict[str, Path] = {}
    for path in aig._python_files(REPO_ROOT / "alpha_zero"):
        name = aig._module_name(path, REPO_ROOT)
        if name:
            modules[name] = path
    trees = aig._parse_all(modules)
    return modules, trees, aig._reexport_tables(modules, trees)


def _deepest(name: str, modules: dict[str, Path]) -> str | None:
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in modules:
            return candidate
    return None


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Every descendant, skipping the body of ``if TYPE_CHECKING:`` blocks."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If) and _is_type_checking(child.test):
            for other in child.orelse:
                yield other
                yield from _runtime_nodes(other)
            continue
        yield child
        yield from _runtime_nodes(child)


def _edges(name: str) -> set[str]:
    modules, trees, tables = _graph()
    path = modules[name]
    out: set[str] = set()
    for node in _runtime_nodes(trees[name]):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _deepest(alias.name, modules)
                if target:
                    out.add(target)
        elif isinstance(node, ast.ImportFrom):
            base = (
                aig._resolve_relative(name, path, node.level, node.module)
                if node.level
                else (node.module or "")
            )
            for alias in node.names:
                qualified = f"{base}.{alias.name}"
                if qualified in modules:
                    out.add(qualified)
                elif alias.name in tables.get(base, {}):
                    out.add(tables[base][alias.name])
                elif base in modules:
                    out.add(base)
        elif isinstance(node, ast.Call):
            func = node.func
            func_name = getattr(func, "attr", None) or getattr(func, "id", None)
            if (
                func_name in {"import_module", "__import__"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                target = _deepest(node.args[0].value, modules)
                if target:
                    out.add(target)
    out.discard(name)
    return out


def closure(seeds: tuple[str, ...] = SEEDS) -> dict[str, str | None]:
    """``{module: the module that first imported it}`` over ``alpha_zero`` only."""
    parent: dict[str, str | None] = dict.fromkeys(seeds)
    queue = deque(seeds)
    while queue:
        module = queue.popleft()
        for target in sorted(_edges(module)):
            if target.startswith("alpha_zero") and target not in parent:
                parent[target] = module
                queue.append(target)
    return parent


def _pathspec_regex(spec: str) -> re.Pattern[str]:
    """``:(glob)`` semantics: ``**`` spans directories, ``*`` does not."""
    out = []
    i = 0
    while i < len(spec):
        if spec.startswith("**", i):
            out.append(".*")
            i += 2
        elif spec[i] == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(spec[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _covered(rel_path: str, specs: list[str]) -> bool:
    return any(_pathspec_regex(spec).match(rel_path) for spec in specs)


def _chain(module: str, parent: dict[str, str | None]) -> str:
    links: list[str] = []
    current: str | None = module
    while current is not None:
        links.append(current)
        current = parent[current]
    return " <- ".join(links)


def test_the_closure_reaches_the_modules_the_review_found() -> None:
    """Non-vacuity: the walker sees the direct imports the hand-written list missed."""
    found = closure()
    for module in (
        "alpha_zero.multiplayer.scoring",
        "alpha_zero.multiplayer.network_heads",
        "alpha_zero.utils.checkpoint",
        "alpha_zero.utils.device",
        "alpha_zero.core.mcts.gumbel",
        "alpha_zero.cloud.gcp.arena",
    ):
        assert module in found, module


def test_type_checking_imports_are_not_followed() -> None:
    tree = ast.parse(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    import hidden\nelse:\n    import shown\nimport plain\n"
    )
    seen = {
        alias.name
        for node in _runtime_nodes(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert seen == {"shown", "plain"}


def test_pathspec_matching_follows_git_glob() -> None:
    assert _covered("alpha_zero/core/mcts/gumbel.py", ["alpha_zero/core/mcts/**"])
    assert not _covered("alpha_zero/core/mcts_factory.py", ["alpha_zero/core/mcts/**"])
    assert _covered("alpha_zero/utils/device.py", ["alpha_zero/utils/device.py"])
    assert not _covered("alpha_zero/utils/device.pyc", ["alpha_zero/utils/device.py"])


def test_every_module_on_the_a1_path_is_on_the_play_path_or_exempt() -> None:
    modules, _, _ = _graph()
    specs = vre.read_play_path()
    found = closure()
    uncovered = []
    for module in sorted(found):
        rel = modules[module].relative_to(REPO_ROOT).as_posix()
        if module in EXEMPT or _covered(rel, specs):
            continue
        uncovered.append(f"{rel}  ({_chain(module, found)})")
    assert not uncovered, (
        "modules the A1 training/benchmark path imports are neither on "
        "scripts/release_play_path.txt nor in EXEMPT (add each to one, with a reason):\n  "
        + "\n  ".join(uncovered)
    )


def test_exemptions_are_not_stale_or_contradictory() -> None:
    """An exemption that is no longer imported, or that is also on the play path, is noise."""
    modules, _, _ = _graph()
    specs = vre.read_play_path()
    found = closure()
    for module, reason in EXEMPT.items():
        assert reason.strip(), f"{module}: an exemption needs a reason"
        assert module in found, f"{module} is exempt but no longer on the A1 path; drop it"
        rel = modules[module].relative_to(REPO_ROOT).as_posix()
        assert not _covered(rel, specs), f"{module} is both exempt and on the play path"


def test_the_registered_arena_entry_point_is_on_the_play_path() -> None:
    """Section 7.1 runs ``scripts/benchmark_suite.py``; the closure starts one step later."""
    assert _covered("scripts/benchmark_suite.py", vre.read_play_path())
