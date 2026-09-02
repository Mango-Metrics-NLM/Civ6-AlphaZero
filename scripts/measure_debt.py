#!/usr/bin/env python3
"""Measure the tech-debt metrics the plan tracks, as JSON.

Every number in ``kb/specs/20260902-tech-debt-hardening.md`` §2 is produced
here so that the baseline is a command, not a hand count, and so that
``tests/regression/test_debt_ratchet.py`` can hold each metric to "never
worse than the committed baseline".

Every metric is "lower is better". Metrics that need a tool this
environment lacks (``ruff`` for statement counts) are reported as ``null``
and skipped by the ratchet rather than guessed.

Usage::

    python scripts/measure_debt.py                 # JSON to stdout
    python scripts/measure_debt.py --write PATH    # also write the file
    python scripts/measure_debt.py --compare PATH  # exit 1 if any metric regressed
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import logging
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_import_graph import audit  # noqa: E402

logger = logging.getLogger("measure_debt")

PACKAGE_DIR = "alpha_zero"
TESTS_DIR = "tests"
GOD_FILE_LINES = 1000
MAX_STATEMENTS = 100
SKIP_SITE_PATTERN = re.compile(r"@pytest\.mark\.(skip|skipif|xfail)|pytest\.(skip|importorskip)\(")
SKIP_PARTS = {"__pycache__", ".git", "alpha_zero.egg-info"}


def _py_files(root: Path, subdir: str) -> list[Path]:
    return sorted(p for p in (root / subdir).rglob("*.py") if not (SKIP_PARTS & set(p.parts)))


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _pyproject(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def coverage_omit_patterns(pyproject: dict[str, Any]) -> list[str]:
    """The ``alpha_zero/`` patterns in ``[tool.coverage.run].omit``."""
    omit = pyproject.get("tool", {}).get("coverage", {}).get("run", {}).get("omit", [])
    return [p for p in omit if p.startswith(PACKAGE_DIR + "/")]


def is_omitted(rel_path: str, patterns: list[str]) -> bool:
    """The rule the plan states: fnmatch of each pattern against the repo-relative path."""
    return any(
        fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(rel_path, "*/" + p) for p in patterns
    )


def _ruff_statement_violations(root: Path) -> int | None:
    ruff = shutil.which("ruff")
    if ruff is None:
        logger.warning("ruff not on PATH; statement-count metric reported as null")
        return None
    cmd = [
        ruff,
        "check",
        PACKAGE_DIR,
        "scripts",
        "--select",
        "PLR0915",
        "--config",
        f"lint.pylint.max-statements={MAX_STATEMENTS}",
        "--output-format",
        "json",
        "--no-cache",
    ]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
    try:
        return len(json.loads(result.stdout or "[]"))
    except json.JSONDecodeError:
        logger.warning("could not parse ruff output; statement-count metric reported as null")
        return None


def _ast_counts(files: list[Path]) -> dict[str, int]:
    """Runtime asserts, ``except: pass`` handlers, and unsafe ``torch.load`` calls."""
    asserts = except_pass = unsafe_loads = 0
    for path in files:
        try:
            tree = ast.parse(_read(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                asserts += 1
            elif isinstance(node, ast.ExceptHandler):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    except_pass += 1
            elif isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "load"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "torch"
                ):
                    weights_only = next(
                        (kw for kw in node.keywords if kw.arg == "weights_only"), None
                    )
                    if weights_only is None or (
                        isinstance(weights_only.value, ast.Constant)
                        and weights_only.value.value is False
                    ):
                        unsafe_loads += 1
    return {
        "runtime_asserts": asserts,
        "except_pass_handlers": except_pass,
        "unsafe_torch_load_calls": unsafe_loads,
    }


def measure(root: Path = REPO_ROOT) -> dict[str, int | None]:
    """Compute every tracked metric. Keys are stable; the ratchet test depends on them."""
    pkg_files = _py_files(root, PACKAGE_DIR)
    test_files = _py_files(root, TESTS_DIR)
    pyproject = _pyproject(root)
    patterns = coverage_omit_patterns(pyproject)

    pkg_lines = {p: _line_count(p) for p in pkg_files}
    omitted = [p for p in pkg_files if is_omitted(str(p.relative_to(root)), patterns)]

    reports = audit(root)
    dead = [r for r in reports if not r.entry_point and not r.deploy_refs]
    dead_measured = [r for r in dead if not is_omitted(r.path, patterns)]

    pkg_text = {p: _read(p) for p in pkg_files}
    ruff_cfg = pyproject.get("tool", {}).get("ruff", {}).get("lint", {})

    metrics: dict[str, int | None] = {
        "package_files": len(pkg_files),
        "package_lines": sum(pkg_lines.values()),
        "files_over_1000_lines": sum(1 for n in pkg_lines.values() if n > GOD_FILE_LINES),
        "test_files_over_1000_lines": sum(1 for p in test_files if _line_count(p) > GOD_FILE_LINES),
        "functions_over_100_statements": _ruff_statement_violations(root),
        "dead_modules": len(dead),
        "dead_module_lines": sum(r.lines for r in dead),
        "dead_modules_inside_coverage": len(dead_measured),
        "coverage_omit_patterns": len(patterns),
        "coverage_omitted_files": len(omitted),
        "coverage_omitted_lines": sum(pkg_lines[p] for p in omitted),
        "type_ignore_comments": sum(t.count("# type: ignore") for t in pkg_text.values()),
        "noqa_comments": sum(len(re.findall(r"#\s*noqa", t)) for t in pkg_text.values()),
        "ruff_per_file_ignores": len(ruff_cfg.get("per-file-ignores", {})),
        "pytest_skip_sites": sum(len(SKIP_SITE_PATTERN.findall(_read(p))) for p in test_files),
        "dependency_manifests": sum(
            1
            for name in (
                "pyproject.toml",
                "requirements.txt",
                "requirements-container.txt",
                "requirements-inference.txt",
                "setup.py",
                "uv.lock",
            )
            if (root / name).exists()
        ),
    }
    metrics.update(_ast_counts(pkg_files))
    return metrics


def compare(current: dict[str, int | None], baseline: dict[str, int | None]) -> list[str]:
    """Return one message per metric that is worse than the baseline (higher)."""
    regressions: list[str] = []
    for key, base in baseline.items():
        now = current.get(key)
        if base is None or now is None:
            continue
        if now > base:
            regressions.append(f"{key}: {now} > baseline {base}")
    return regressions


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--write", type=Path, help="also write the JSON to this path")
    parser.add_argument("--compare", type=Path, help="exit 1 if any metric regressed vs this file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    metrics = measure(args.root)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n", encoding="utf-8")
        logger.info("wrote %s", args.write)
    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        regressions = compare(metrics, baseline)
        for line in regressions:
            print("REGRESSION", line, file=sys.stderr)
        return 1 if regressions else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
