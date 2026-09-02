#!/usr/bin/env python3
"""Find ``alpha_zero`` modules that nothing outside tests imports.

Stdlib only. Builds the import graph with ``ast`` (absolute imports, relative
imports, ``from pkg import submodule``, and ``importlib.import_module`` /
``__import__`` with a literal string), then reports every package module with
no *live* importer. A live importer is a non-test module that is not the
target's own package ``__init__`` -- re-exporting a module from its package
``__init__`` does not make it used, it only hides that nobody uses it.

Modules that are entry points (named ``__main__`` or carrying an
``if __name__ == "__main__"`` guard) and modules named as a string in a
build or deploy file (Dockerfiles, ``cloudbuild*.yaml``, ``Makefile``,
``k8s/``, ``*.sh``, ``pyproject.toml``) are reported separately: they have no
importer by design, and a grep of their dotted name is how you decide whether
the entry point is still wired to anything.

Deadness is **not transitive**: a dead module still counts as an importer of
what it imports, so a cluster of mutually-importing dead modules hides its
members. Expect a second wave after any deletion -- this branch found 79,
deleted 44, and re-measured 54.

Exit codes: 0 always for a report; 2 on a usage error. This is an audit,
not a gate -- the gate is ``tests/regression/test_dead_module_ratchet.py``
once it exists (see ``kb/specs/20260902-tech-debt-hardening.md`` WS-M).

Usage::

    python scripts/audit_import_graph.py            # table to stdout
    python scripts/audit_import_graph.py --json     # machine-readable
    python scripts/audit_import_graph.py --package alpha_zero.mechanics
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("audit_import_graph")

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "alpha_zero"

# Files whose string mention of a module counts as "wired" (a build or deploy
# reference), as opposed to prose in docs/ or kb/ that merely talks about it.
DEPLOY_GLOBS = (
    "Dockerfile*",
    "dockerfiles/*",
    "cloudbuild*.yaml",
    "Makefile",
    "k8s/**/*.yaml",
    "scripts/*.sh",
    "scripts/*.ps1",
    "docker-compose.yml",
    "configs/**/*.yaml",
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "infra/**/*.yaml",
)
SKIP_PARTS = {".git", "__pycache__", "venv", ".venv", "node_modules", "alpha_zero.egg-info"}
# Prefix on an importer entry recording that the import was a side-effect import
# (``import pkg.mod`` / ``importlib.import_module``) rather than a name re-export.
SIDE_EFFECT_MARKER = "!"
DOTTED_OR_PATH = re.compile(
    r"alpha_zero(?:\.[A-Za-z_][A-Za-z0-9_]*)+|alpha_zero(?:/[A-Za-z_][A-Za-z0-9_]*)+\.py"
)


@dataclass
class ModuleReport:
    """One package module with no live importer."""

    module: str
    path: str
    lines: int
    test_importers: int
    entry_point: bool
    deploy_refs: list[str] = field(default_factory=list)


def _module_name(path: Path, root: Path) -> str | None:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _is_test(path: Path) -> bool:
    return "tests" in path.parts or path.name.startswith("test_") or path.name == "conftest.py"


def _python_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if not (SKIP_PARTS & set(p.parts))]


def _resolve_relative(src_mod: str, src_path: Path, level: int, mod: str | None) -> str:
    base = src_mod.split(".")
    if src_path.name != "__init__.py":
        base = base[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join(base + ([mod] if mod else []))


def _parse_all(modules: dict[str, Path]) -> dict[str, ast.Module]:
    """Parse every module, warning loudly about any that cannot be read.

    A file dropped here loses *its* imports from the graph, so whatever it
    imports can be reported dead. Silently skipping is therefore an
    under-count, not a no-op: a UTF-8 BOM alone used to be enough (``ast.parse``
    rejects U+FEFF), which is why the read is ``utf-8-sig``.
    """
    trees: dict[str, ast.Module] = {}
    for name, path in modules.items():
        try:
            trees[name] = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except (SyntaxError, OSError) as exc:
            logger.warning(
                "could not parse %s (%s); its imports are missing from the graph, "
                "so this run may over-report dead modules",
                path,
                exc,
            )
    return trees


def _reexport_tables(
    modules: dict[str, Path], trees: dict[str, ast.Module]
) -> dict[str, dict[str, str]]:
    """For each package ``__init__``: exported name -> submodule that defines it.

    ``from .mcts import NPlayerMCTS`` in ``alpha_zero/multiplayer/__init__.py``
    yields ``tables["alpha_zero.multiplayer"]["NPlayerMCTS"] == "alpha_zero.multiplayer.mcts"``.
    A PEP 562 lazy ``__getattr__`` with a literal ``{name: "module"}`` dict is
    also read, because ``alpha_zero/__init__.py`` uses one.
    """
    tables: dict[str, dict[str, str]] = defaultdict(dict)
    for name, path in modules.items():
        if path.name != "__init__.py" or name not in trees:
            continue
        for node in ast.walk(trees[name]):
            if isinstance(node, ast.ImportFrom):
                base = (
                    _resolve_relative(name, path, node.level, node.module)
                    if node.level
                    else (node.module or "")
                )
                for alias in node.names:
                    target = f"{base}.{alias.name}" if f"{base}.{alias.name}" in modules else base
                    if target in modules and target != name:
                        tables[name][alias.asname or alias.name] = target
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=False):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                        and value.value in modules
                    ):
                        tables[name][key.value] = value.value
    return tables


def build_import_graph(root: Path) -> tuple[dict[str, Path], dict[str, set[str]]]:
    """Return ``(modules, importers)`` where ``importers[target]`` is a set of source modules.

    Names imported from a package are credited to the submodule the package's
    ``__init__`` re-exports them from, so ``from alpha_zero.multiplayer import
    NPlayerMCTS`` counts as an import of ``alpha_zero.multiplayer.mcts``.
    Attribute access on a package alias (``import alpha_zero.multiplayer as mp;
    mp.NPlayerMCTS``) is resolved the same way.
    """
    modules: dict[str, Path] = {}
    for path in _python_files(root):
        name = _module_name(path, root)
        if name:
            modules[name] = path

    trees = _parse_all(modules)
    tables = _reexport_tables(modules, trees)
    importers: dict[str, set[str]] = defaultdict(set)

    def record(target: str, source: str, *, side_effect: bool = False) -> None:
        parts = target.split(".")
        for i in range(1, len(parts) + 1):
            candidate = ".".join(parts[:i])
            if candidate in modules:
                importers[candidate].add(source)
                if side_effect:
                    # A plain ``import pkg.mod`` (or importlib call) executes the
                    # module for its side effects; it is a use even when the
                    # importer is the module's own package ``__init__``.
                    importers[candidate].add(SIDE_EFFECT_MARKER + source)

    def record_reexport(package: str, name: str, source: str) -> None:
        target = tables.get(package, {}).get(name)
        if target:
            record(target, source)

    for source, tree in trees.items():
        path = modules[source]
        package_aliases: dict[str, str] = {}  # local name -> package it is bound to
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    record(alias.name, source, side_effect=True)
                    if alias.name in tables:
                        package_aliases[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                base = (
                    _resolve_relative(source, path, node.level, node.module)
                    if node.level
                    else (node.module or "")
                )
                record(base, source)
                for alias in node.names:
                    qualified = f"{base}.{alias.name}"
                    # `from pkg import submodule` binds and executes the module
                    # itself -- semantically identical to `import pkg.submodule`
                    # and the usual shape of a registry-registration import.
                    record(qualified, source, side_effect=qualified in modules)
                    record_reexport(base, alias.name, source)
                    if f"{base}.{alias.name}" in tables:
                        package_aliases[alias.asname or alias.name] = f"{base}.{alias.name}"
            elif isinstance(node, ast.Call):
                func = node.func
                func_name = getattr(func, "attr", None) or getattr(func, "id", None)
                if (
                    func_name in {"import_module", "__import__"}
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    record(node.args[0].value, source, side_effect=True)
        if package_aliases:
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    package = package_aliases.get(node.value.id)
                    if package:
                        record_reexport(package, node.attr, source)
    return modules, importers


def deploy_references(root: Path, modules: dict[str, Path]) -> dict[str, set[str]]:
    """Map module -> deploy/build files that name it as a string."""
    refs: dict[str, set[str]] = defaultdict(set)
    seen: set[Path] = set()
    for pattern in DEPLOY_GLOBS:
        for path in root.glob(pattern):
            if not path.is_file() or path in seen or (SKIP_PARTS & set(path.parts)):
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for hit in set(DOTTED_OR_PATH.findall(text)):
                dotted = hit.replace("/", ".").removesuffix(".py")
                if dotted in modules:
                    refs[dotted].add(str(path.relative_to(root)))
    return refs


def _is_entry_point(module: str, path: Path) -> bool:
    if module.rsplit(".", 1)[-1] == "__main__":
        return True
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return '__name__ == "__main__"' in text or "__name__ == '__main__'" in text


def _live_importers(target: str, sources: set[str], modules: dict[str, Path]) -> set[str]:
    package = target.rsplit(".", 1)[0]
    side_effect_sources = {
        s.removeprefix(SIDE_EFFECT_MARKER) for s in sources if s.startswith(SIDE_EFFECT_MARKER)
    }
    live: set[str] = set()
    for source in sources:
        if source.startswith(SIDE_EFFECT_MARKER):
            continue
        if source == target or source.startswith(target + "."):
            continue  # a module importing its own children
        if source == package and source not in side_effect_sources:
            continue  # the owning package __init__ merely re-exporting it
        if _is_test(modules[source]):
            continue
        live.add(source)
    return live


def audit(root: Path, package_filter: str | None = None) -> list[ModuleReport]:
    """Return every ``PACKAGE`` module with no live importer, largest first."""
    modules, importers = build_import_graph(root)
    refs = deploy_references(root, modules)
    reports: list[ModuleReport] = []
    for module, path in modules.items():
        if not module.startswith(PACKAGE + "."):
            continue
        if package_filter and not (
            module == package_filter or module.startswith(package_filter + ".")
        ):
            continue
        if path.name == "__init__.py":
            continue
        if _live_importers(module, importers.get(module, set()), modules):
            continue
        try:
            with path.open(encoding="utf-8-sig", errors="replace") as handle:
                lines = sum(1 for _ in handle)
        except OSError:
            # The tree can change under a long walk (a concurrent test run
            # creating and removing a probe file, an editor save). Skip rather
            # than crash: this script feeds a CI gate.
            continue
        test_count = sum(
            1
            for s in importers.get(module, set())
            if not s.startswith(SIDE_EFFECT_MARKER) and _is_test(modules[s])
        )
        reports.append(
            ModuleReport(
                module=module,
                path=str(path.relative_to(root)),
                lines=lines,
                test_importers=test_count,
                entry_point=_is_entry_point(module, path),
                deploy_refs=sorted(refs.get(module, set())),
            )
        )
    reports.sort(key=lambda r: (-r.lines, r.module))
    return reports


def _render_table(reports: list[ModuleReport]) -> str:
    dead = [r for r in reports if not r.entry_point and not r.deploy_refs]
    wired = [r for r in reports if r.entry_point or r.deploy_refs]
    out = [f"{'lines':>6}  {'module':<58} {'tests':>5}  kind"]
    for r in reports:
        kind = "entry-point" if r.entry_point else ("deploy-ref" if r.deploy_refs else "DEAD")
        if r.deploy_refs:
            kind += f" ({', '.join(r.deploy_refs)})"
        out.append(f"{r.lines:6d}  {r.module:<58} {r.test_importers:5d}  {kind}")
    out.append("")
    out.append(
        f"dead: {len(dead)} modules / {sum(r.lines for r in dead)} lines; "
        f"no-importer entry points or deploy-referenced: {len(wired)} modules / "
        f"{sum(r.lines for r in wired)} lines"
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--package", help="restrict to one dotted package, e.g. alpha_zero.mechanics"
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if not (args.root / PACKAGE).is_dir():
        parser.error(f"{args.root} does not contain a {PACKAGE}/ package")
        return 2
    reports = audit(args.root, args.package)
    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2))
    else:
        print(_render_table(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
