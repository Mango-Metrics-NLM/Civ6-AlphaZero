#!/usr/bin/env python3
"""Enforce the per-package coverage floors in ``coverage-floors.json``.

The global gate (``--cov-fail-under=85`` in CI's coverage job) omits whole
subtrees -- ``alpha_zero/cloud/*`` among them, which is the deployed path. A
global percentage cannot express "this package must not get worse" for a
package the global run does not measure at all, so each such package gets its
own floor, measured in the job that actually exercises it.

A floor is a ratchet: it may be raised, never lowered. Lowering one is a
reviewed change to ``coverage-floors.json`` and should say why in the PR.

Reads a ``coverage json`` report (``coverage json -o FILE``) and compares the
covered-statement percentage of every file under each configured package
prefix against that package's ``minimum``.

Exit codes: 0 all floors met; 1 a floor was breached; 2 usage or input error.

Usage::

    coverage json -o coverage.json
    python scripts/check_coverage_floors.py --report coverage.json
    python scripts/check_coverage_floors.py --report coverage.json --package alpha_zero/cloud
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FLOORS = REPO_ROOT / "coverage-floors.json"
DEFAULT_REPORT = REPO_ROOT / "coverage.json"

logger = logging.getLogger("check_coverage_floors")


@dataclass(frozen=True)
class PackageResult:
    """Measured coverage for one configured package prefix."""

    package: str
    minimum: float
    percent: float
    covered: int
    total: int
    files: int

    @property
    def ok(self) -> bool:
        """True when the measured percentage meets the floor.

        An empty result is deliberately **not** ok. A report with no files
        under the prefix means the package was not measured at all, which is
        the exact failure this gate exists to catch: drop ``--rcfile`` from
        the ``coverage json`` step and pyproject's omit list silently removes
        every ``alpha_zero/cloud`` file, leaving the only job that measures
        the deployed path measuring nothing and reporting green.
        """
        return self.files > 0 and self.percent >= self.minimum

    def describe(self) -> str:
        """One line for the CLI, naming the shortfall when there is one."""
        if self.files == 0:
            return (
                f"{self.package}: NOT MEASURED — no files under this prefix in the report. "
                "Did the coverage run omit the package (missing --rcfile), or was it renamed?"
            )
        verdict = "OK" if self.ok else "BELOW FLOOR"
        return (
            f"{self.package}: {self.percent:.1f}% of {self.total} statements "
            f"across {self.files} files (floor {self.minimum:.0f}%) [{verdict}]"
        )


def load_floors(path: Path) -> dict[str, float]:
    """Return ``{package prefix: minimum percent}`` from the floors file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"floors file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"floors file is not valid JSON: {path}: {exc}") from None
    floors = raw.get("floors")
    if not isinstance(floors, dict) or not floors:
        raise SystemExit(f"floors file has no non-empty 'floors' object: {path}")
    out: dict[str, float] = {}
    for package, spec in floors.items():
        if not isinstance(spec, dict) or "minimum" not in spec:
            raise SystemExit(f"floors['{package}'] must be an object with a 'minimum'")
        out[package.rstrip("/")] = float(spec["minimum"])
    return out


def measure(report: dict[str, object], package: str, minimum: float) -> PackageResult:
    """Aggregate every file in the coverage report under ``package``."""
    files = report.get("files")
    if not isinstance(files, dict):
        raise SystemExit(
            "coverage report has no 'files' object; was it produced by `coverage json`?"
        )
    prefix = package.rstrip("/") + "/"
    covered = total = count = 0
    for filename, entry in files.items():
        normalized = filename.replace("\\", "/")
        if not normalized.startswith(prefix):
            continue
        summary = entry.get("summary", {}) if isinstance(entry, dict) else {}
        statements = int(summary.get("num_statements", 0))
        missing = int(summary.get("missing_lines", 0))
        covered += statements - missing
        total += statements
        count += 1
    percent = 100.0 * covered / total if total else 0.0
    return PackageResult(package, minimum, percent, covered, total, count)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="coverage json output")
    parser.add_argument("--floors", type=Path, default=DEFAULT_FLOORS)
    parser.add_argument("--package", help="check only this configured package prefix")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="treat a package with no files in the report as a pass (default: hard failure)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    floors = load_floors(args.floors)
    if args.package:
        key = args.package.rstrip("/")
        if key not in floors:
            raise SystemExit(f"no floor configured for '{key}'; known: {sorted(floors)}")
        floors = {key: floors[key]}

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(
            f"coverage report not found: {args.report}. Produce it with `coverage json -o {args.report}`."
        ) from None

    results = [measure(report, package, minimum) for package, minimum in sorted(floors.items())]
    for result in results:
        logger.info("%s", result.describe())

    empty = [r for r in results if r.files == 0]
    if empty and not args.allow_empty:
        for result in empty:
            logger.error(
                "NOT MEASURED %s: the report contains no files under this prefix, so the floor "
                "proves nothing. Check that the coverage run used --rcfile with a config that "
                "does not omit the package. Pass --allow-empty only when the package is "
                "deliberately absent from this run.",
                result.package,
            )
        return 2

    breached = [r for r in results if not r.ok]
    if breached:
        for result in breached:
            logger.error(
                "FLOOR BREACHED %s: %.1f%% < %.0f%%. Add tests, or if the drop is deliberate and "
                "reviewed, change the floor in %s and say why in the PR.",
                result.package,
                result.percent,
                result.minimum,
                args.floors.name,
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
