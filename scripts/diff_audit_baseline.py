#!/usr/bin/env python3
"""Compare pip-audit JSON output against an accepted-findings baseline.

Usage::

    python scripts/diff_audit_baseline.py audit.json .security/pip-audit-baseline.toml
    python scripts/diff_audit_baseline.py a.json b.json [baseline.toml] [--strict]

Positional arguments are pip-audit JSON reports; a ``.toml`` argument is the
baseline (default ``.security/pip-audit-baseline.toml``). The baseline holds
``[[ignore]]`` tables, each with exactly these keys::

    [[ignore]]
    id = "GHSA-xxxx-xxxx-xxxx"   # the finding's id, or any of its aliases
    package = "example"          # distribution name as pip-audit reports it
    reason = "why this is accepted, and where it is tracked"
    expires = 2027-01-01         # a TOML date; after it the entry is stale

Exit codes, in the same shape as the other validators in ``scripts/``:

  0  every finding is covered by an unexpired baseline entry (or there are none)
  1  at least one finding is NOT in the baseline -- or, with ``--strict``, at
     least one finding is covered only by an EXPIRED entry
  2  usage or input error (unreadable report, malformed baseline entry)

Without ``--strict`` an expired entry still silences its finding but is
reported as a warning, so a stale baseline is visible without turning every
scheduled run red the day an entry lapses. Unused entries are warned about
too: an entry nothing matches any more is debt the next reader has to re-derive.

Stdlib only (``tomllib`` is 3.11+, which ``requires-python`` already demands).
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

DEFAULT_BASELINE = Path(".security/pip-audit-baseline.toml")
_REQUIRED_KEYS = frozenset({"id", "package", "reason", "expires"})


def _normalise(name: str) -> str:
    """PEP 503 name normalisation, so ``PyYAML`` and ``pyyaml`` match."""
    return name.strip().lower().replace("_", "-").replace(".", "-")


@dataclass(frozen=True)
class Finding:
    package: str
    version: str
    vuln_id: str
    aliases: frozenset[str]
    fix_versions: tuple[str, ...]
    source: str

    @property
    def ids(self) -> frozenset[str]:
        return frozenset({self.vuln_id, *self.aliases})

    def render(self) -> str:
        fix = (
            f" (fix: {', '.join(self.fix_versions)})" if self.fix_versions else " (no fix released)"
        )
        return f"{self.package} {self.version}: {self.vuln_id}{fix} [{self.source}]"


@dataclass(frozen=True)
class BaselineEntry:
    vuln_id: str
    package: str
    reason: str
    expires: date
    index: int

    def covers(self, finding: Finding) -> bool:
        return _normalise(self.package) == _normalise(finding.package) and self.vuln_id in {
            i.upper() for i in finding.ids
        }

    def expired(self, today: date) -> bool:
        return self.expires < today


@dataclass
class Report:
    new: list[Finding] = field(default_factory=list)
    covered: list[tuple[Finding, BaselineEntry]] = field(default_factory=list)
    expired: list[tuple[Finding, BaselineEntry]] = field(default_factory=list)
    unused: list[BaselineEntry] = field(default_factory=list)


class InputError(Exception):
    """A report or baseline that cannot be used; maps to exit 2."""


def load_findings(path: Path) -> list[Finding]:
    """Parse one pip-audit JSON report (``--format json``).

    Accepts both the current shape (``{"dependencies": [...], "fixes": [...]}``)
    and the older bare-list shape. A dependency pip-audit skipped (``skip_reason``)
    carries no ``vulns`` and contributes nothing.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"{path} is not valid JSON ({exc.msg})") from exc

    deps: Any = data.get("dependencies") if isinstance(data, dict) else data
    if not isinstance(deps, list):
        raise InputError(f"{path}: expected a pip-audit report with a 'dependencies' list")

    findings: list[Finding] = []
    for dep in deps:
        if not isinstance(dep, dict):
            continue
        name = str(dep.get("name", "?"))
        version = str(dep.get("version", "?"))
        for vuln in dep.get("vulns") or []:
            if not isinstance(vuln, dict) or not vuln.get("id"):
                continue
            findings.append(
                Finding(
                    package=name,
                    version=version,
                    vuln_id=str(vuln["id"]),
                    aliases=frozenset(str(a) for a in vuln.get("aliases") or []),
                    fix_versions=tuple(str(v) for v in vuln.get("fix_versions") or []),
                    source=path.name,
                )
            )
    return findings


def _coerce_date(value: Any, where: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise InputError(f"{where}: 'expires' {value!r} is not an ISO date") from exc
    raise InputError(f"{where}: 'expires' must be a date, got {type(value).__name__}")


def load_baseline(path: Path) -> list[BaselineEntry]:
    """Parse the baseline, refusing any entry that is not fully specified.

    A silent entry with no reason or no expiry is how an accepted finding
    becomes a permanent blind spot, so a malformed entry is an input error
    (exit 2), not a warning.
    """
    if not path.exists():
        return []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise InputError(f"{path} is not valid TOML ({exc})") from exc

    entries: list[BaselineEntry] = []
    for index, raw in enumerate(data.get("ignore") or [], start=1):
        where = f"{path} [[ignore]] #{index}"
        if not isinstance(raw, dict):
            raise InputError(f"{where}: entry is not a table")
        missing = sorted(_REQUIRED_KEYS - set(raw))
        extra = sorted(set(raw) - _REQUIRED_KEYS)
        if missing:
            raise InputError(f"{where}: missing key(s) {missing}")
        if extra:
            raise InputError(f"{where}: unknown key(s) {extra}")
        if not str(raw["reason"]).strip():
            raise InputError(f"{where}: 'reason' is empty")
        entries.append(
            BaselineEntry(
                vuln_id=str(raw["id"]).strip().upper(),
                package=str(raw["package"]),
                reason=str(raw["reason"]).strip(),
                expires=_coerce_date(raw["expires"], where),
                index=index,
            )
        )
    return entries


def diff(findings: list[Finding], baseline: list[BaselineEntry], today: date) -> Report:
    report = Report()
    used: set[int] = set()
    for finding in findings:
        match = next((e for e in baseline if e.covers(finding)), None)
        if match is None:
            report.new.append(finding)
            continue
        used.add(match.index)
        if match.expired(today):
            report.expired.append((finding, match))
        else:
            report.covered.append((finding, match))
    report.unused = [e for e in baseline if e.index not in used]
    return report


def _print_report(report: Report, *, strict: bool) -> None:
    for finding, entry in report.covered:
        print(f"accepted  {finding.render()} -- {entry.reason} (until {entry.expires})")
    for finding, entry in report.expired:
        level = "ERROR" if strict else "WARNING"
        print(
            f"{level} expired baseline entry #{entry.index} ({entry.vuln_id}, {entry.package}) "
            f"lapsed {entry.expires}; still matches {finding.render()}",
            file=sys.stderr,
        )
    for entry in report.unused:
        print(
            f"WARNING baseline entry #{entry.index} ({entry.vuln_id}, {entry.package}) "
            "matched no finding; remove it",
            file=sys.stderr,
        )
    for finding in report.new:
        print(f"ERROR new finding not in baseline: {finding.render()}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail on pip-audit findings that are not in the accepted baseline."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="pip-audit JSON report(s); a .toml argument is the baseline",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail on findings covered only by an expired entry",
    )
    parser.add_argument(
        "--today", help="override today's date (ISO) for the expiry check; tests use this"
    )
    args = parser.parse_args(argv)

    reports = [Path(p) for p in args.paths if not p.endswith(".toml")]
    baselines = [Path(p) for p in args.paths if p.endswith(".toml")]
    if len(baselines) > 1:
        print(f"ERROR: more than one baseline given: {baselines}", file=sys.stderr)
        return 2
    if not reports:
        print("ERROR: no pip-audit JSON report given", file=sys.stderr)
        return 2
    baseline_path = baselines[0] if baselines else DEFAULT_BASELINE

    try:
        today = date.fromisoformat(args.today) if args.today else date.today()
    except ValueError:
        print(f"ERROR: --today {args.today!r} is not an ISO date", file=sys.stderr)
        return 2

    try:
        findings = [f for path in reports for f in load_findings(path)]
        baseline = load_baseline(baseline_path)
    except InputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = diff(findings, baseline, today)
    _print_report(report, strict=args.strict)

    failed = bool(report.new) or (args.strict and bool(report.expired))
    summary = (
        f"pip-audit baseline: {len(findings)} finding(s) across {len(reports)} report(s) -- "
        f"{len(report.new)} new, {len(report.covered)} accepted, "
        f"{len(report.expired)} expired-but-matching, {len(report.unused)} unused entries"
    )
    print(summary, file=sys.stderr if failed else sys.stdout)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
