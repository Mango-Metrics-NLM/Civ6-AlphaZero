#!/usr/bin/env python3
"""Deterministic validator for the ``kb/`` knowledge-base workflow.

Checks (error codes are stable and referenced by ``tests/governance/``):

  K1  every line of ``kb/ledger/ledger.jsonl`` parses as JSON
  K2  every entry has an ``event`` and an ISO-8601 ``timestamp``
  K3  ``event`` is in the current set, or in the legacy set for a pre-cutoff entry
  K4  ``session_id`` is present on post-cutoff entries (optional before the cutoff)
  K5  any ``spec_file`` / ``pr_draft`` path referenced by an entry exists;
      an ``openspec/changes/<id>/...`` path also resolves through the archive
      (``openspec/changes/archive/<date>-<id>/...``), since the ledger is
      append-only and archiving a merged change moves its directory
  K6  every ``kb/sessions/*.state.json`` file parses as JSON
  K7  no commit reachable from HEAD modifies, deletes, or renames a file under
      ``kb/handoffs/`` (immutability), except entries in ``ALLOWED_AMENDMENTS``

Grandfathering: a frozen ``LEGACY_CUTOFF`` lets the legacy event names and the
early session-key-less entries pass while blocking new drift. This keeps the 12
historical ledger lines valid without weakening the schema going forward.

Modes:
  (default)     K1-K6 + full-history K7 (skipped on a shallow clone)
  --git         K1-K6 + full-history K7 (for CI; checkout must be unshallowed)
  --no-git      K1-K6 only (skip K7) -- used by the fast PostToolUse hook path
  --path FILE   validate a single file in isolation: the ledger (K1-K4) or a
                session-state file (K6); no cross-file (K5) or git (K7) checks

Exit codes: 0 = pass, 1 = validation errors, 2 = usage / internal error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER = REPO_ROOT / "kb" / "ledger" / "ledger.jsonl"
SESSIONS_DIR = REPO_ROOT / "kb" / "sessions"
HANDOFFS_PREFIX = "kb/handoffs/"

# Frozen grandfathering boundary. Entries dated on or after this may only use the
# current event names and must carry a session_id; earlier entries are exempt.
LEGACY_CUTOFF = date(2026, 7, 25)

# Exactly the events emitted by the three handoff skills.
CURRENT_EVENTS = frozenset(
    {
        "SESSION-END",
        "PLANNING-HANDOFF",
        "DEV-SQE-HANDOFF",
        "PR-READY",
        "DESIGN-GAP",
        "QA-GAP",
        "SCOPE-CHANGE",
        # A training/evaluation run that produced committed evidence. The three
        # workflow roles only ever recorded process events, so the ledger could
        # not answer "was this code ever run, and what happened?" -- the gap that
        # docs/evidence/validation_iteration_zero/ was created to close.
        "VALIDATION-RUN",
    }
)
# Retired event names, tolerated only before the cutoff.
LEGACY_EVENTS = frozenset({"PLAN-LOCKED", "DEV-COMPLETE"})

# Reviewed, deliberate handoff amendments, keyed by (repo-relative path, blob sha).
# Empty by default: amending a handoff requires a reviewed change to this list.
ALLOWED_AMENDMENTS: frozenset[tuple[str, str]] = frozenset()

_PATH_KEYS = ("spec_file", "pr_draft")


@dataclass
class Issue:
    code: str
    message: str
    file: Path
    line: int | None = None

    def render(self) -> str:
        rel = _relpath(self.file)
        loc = f"{rel}:{self.line}" if self.line is not None else str(rel)
        return f"ERROR {loc}: {self.code} {self.message}"


def _relpath(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def _ref_exists(ref: str) -> bool:
    """K5 existence check, tolerant of archived OpenSpec changes.

    The ledger is append-only, so a ``spec_file`` recorded as
    ``openspec/changes/<id>/...`` must stay valid after the merged change is
    archived to ``openspec/changes/archive/<date>-<id>/...``.
    """
    if (REPO_ROOT / ref).exists():
        return True
    parts = Path(ref).parts
    if len(parts) >= 4 and parts[:2] == ("openspec", "changes") and parts[2] != "archive":
        change_id, rest = parts[2], Path(*parts[3:])
        archive_dir = REPO_ROOT / "openspec" / "changes" / "archive"
        return any((candidate / rest).exists() for candidate in archive_dir.glob(f"*-{change_id}"))
    return False


def _parse_timestamp(value: str) -> datetime | None:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _validate_ledger(path: Path, issues: list[Issue], *, cross_file: bool) -> None:
    if not path.is_file():
        issues.append(Issue("K1", "ledger file is missing", path))
        return
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            issues.append(Issue("K1", f"line is not valid JSON ({exc.msg})", path, lineno))
            continue
        if not isinstance(entry, dict):
            issues.append(Issue("K1", "line is not a JSON object", path, lineno))
            continue
        _validate_ledger_entry(entry, path, lineno, issues, cross_file=cross_file)


def _validate_ledger_entry(
    entry: dict, path: Path, lineno: int, issues: list[Issue], *, cross_file: bool
) -> None:
    event = entry.get("event")
    ts_raw = entry.get("timestamp")
    if not event:
        issues.append(Issue("K2", "entry has no 'event'", path, lineno))
    if not ts_raw:
        issues.append(Issue("K2", "entry has no 'timestamp'", path, lineno))
    parsed = _parse_timestamp(ts_raw) if isinstance(ts_raw, str) else None
    if ts_raw and parsed is None:
        issues.append(Issue("K2", f"timestamp '{ts_raw}' is not ISO-8601", path, lineno))

    entry_date = parsed.date() if parsed is not None else None
    pre_cutoff = entry_date is not None and entry_date < LEGACY_CUTOFF

    if event and event not in CURRENT_EVENTS:
        if event in LEGACY_EVENTS and pre_cutoff:
            pass  # grandfathered
        else:
            issues.append(Issue("K3", f"unknown event '{event}'", path, lineno))

    if entry_date is not None and not pre_cutoff:
        if not entry.get("session_id"):
            issues.append(Issue("K4", "post-cutoff entry has no 'session_id'", path, lineno))

    if cross_file:
        for key in _PATH_KEYS:
            ref = entry.get(key)
            if ref and not _ref_exists(ref):
                issues.append(Issue("K5", f"referenced {key} '{ref}' does not exist", path, lineno))


def _validate_sessions(issues: list[Issue]) -> None:
    if not SESSIONS_DIR.is_dir():
        return
    for state in sorted(SESSIONS_DIR.glob("*.state.json")):
        try:
            json.loads(state.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(Issue("K6", f"session state is not valid JSON ({exc.msg})", state))


def _is_shallow_clone() -> bool:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return out.stdout.strip() == "true"


def _validate_handoff_immutability(issues: list[Issue], *, quiet: bool) -> None:
    """K7: no reachable commit may modify/delete/rename a kb/handoffs/ file."""
    if _is_shallow_clone():
        if not quiet:
            print(
                "note: shallow clone -- skipping handoff-immutability check (K7)",
                file=sys.stderr,
            )
        return
    try:
        out = subprocess.run(
            [
                "git",
                "log",
                "--diff-filter=MDR",
                "--name-status",
                "--format=",
                "--",
                HANDOFFS_PREFIX,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        issues.append(Issue("K7", f"could not inspect handoff history ({exc})", REPO_ROOT))
        return
    for row in out.stdout.splitlines():
        row = row.strip()
        if not row:
            continue
        parts = row.split("\t")
        status = parts[0][0]
        touched = parts[-1]
        if any(touched == amend_path for amend_path, _ in ALLOWED_AMENDMENTS):
            continue
        issues.append(
            Issue("K7", f"handoff '{touched}' was altered ({status}) after creation", REPO_ROOT)
        )


def _validate_path(path: Path, issues: list[Issue]) -> None:
    if not path.is_file():
        issues.append(Issue("K1", "path does not exist", path))
        return
    if path.name.endswith(".state.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(Issue("K6", f"session state is not valid JSON ({exc.msg})", path))
    elif path.name == "ledger.jsonl":
        _validate_ledger(path, issues, cross_file=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the kb/ workflow.")
    parser.add_argument(
        "--git", action="store_true", help="run the git-based K7 immutability check"
    )
    parser.add_argument("--no-git", action="store_true", help="skip the git-based K7 check")
    parser.add_argument("--path", help="validate a single file in isolation (hook mode)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    issues: list[Issue] = []
    try:
        if args.path:
            _validate_path(Path(args.path).resolve(), issues)
        else:
            _validate_ledger(LEDGER, issues, cross_file=True)
            _validate_sessions(issues)
            if not args.no_git:
                _validate_handoff_immutability(issues, quiet=args.quiet)
    except OSError as exc:  # pragma: no cover - defensive I/O guard
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = [
            {"code": i.code, "message": i.message, "file": str(_relpath(i.file)), "line": i.line}
            for i in issues
        ]
        print(json.dumps(payload, indent=2))
    else:
        for issue in issues:
            print(issue.render(), file=sys.stderr)

    if issues:
        return 1
    if not args.quiet and not args.json:
        print("kb: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
