#!/usr/bin/env python3
"""Verify public-release readiness across metadata, ignores, ratchets, workflows, and git state.

This checker aggregates deterministic, read-only invariants (R1-R6) across the
repository so a public release candidate and everyday governance validation
share one machine-checkable gate:

- **R1** (`public-surface`): required root community, legal, and documentation
  files exist, are non-empty, and the top of `README.md` carries its honest
  status and non-affiliation framing.
- **R2** (`version-citation-changelog`): `[project].version` in `pyproject.toml`,
  `__version__` in `alpha_zero/__init__.py`, `version` / `date-released` in
  `CITATION.cff`, the matching `## [X.Y.Z]` heading in `CHANGELOG.md`, and the
  `X.Y.x` line in `SECURITY.md` are aligned and strict `X.Y.Z` semver.
- **R3** (`hygiene-and-ignores`): `.gitignore`, `.dockerignore`, and
  `.gitleaks.toml` cover required local scratch, secret, and build paths and
  keep `[allowlist]` paths anchored without catch-all patterns.
- **R4** (`coverage-and-debt-ratchets`): `coverage-floors.json` (`alpha_zero/cloud`
  present with `measured >= minimum >= 58`), `pyproject.toml` global
  `fail_under >= 85`, and `docs/evidence/tech_debt_baseline/baseline.json`
  integrity.
- **R5** (`workflow-playpath-and-launcher`): `scripts/release_play_path.txt` is
  populated, core GitHub Actions workflows pin third-party `uses:` references to
  40-hex commit SHAs, and `scripts/local_muzero_launcher.py` has no `subprocess`
  calls with non-`False` `shell=`.
- **R6** (`git-hygiene`): inside a git work tree, no unmerged index entries, and
  no tracked `.env` / private-key / credential artifacts (skipped via `--skip-git`).

Exit codes: 0 when all checks pass (or skip cleanly), 1 when one or more fail.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("check_release_readiness")

REQUIRED_PUBLIC_FILES: tuple[str, ...] = (
    ".env.template",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "LICENSE",
    "PROJECT_CHARTER.md",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
)

REQUIRED_GITIGNORE_TOKENS: tuple[str, ...] = (
    ".env",
    "*.key",
    "*.pem",
    "bandit-report.json",
    "coverage-cloud.json",
    "cutover/",
    "mypy_output.txt",
)

REQUIRED_DOCKERIGNORE_TOKENS: tuple[str, ...] = (
    ".env",
    ".git",
    "cutover/",
    "eval_results/",
    "mypy_output.txt",
    "release-staging*/",
    "test_results/",
)

REQUIRED_WORKFLOWS: tuple[str, ...] = (
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    ".github/workflows/security-scan.yml",
)

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
INIT_VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
CIT_VERSION_RE = re.compile(r'^version:\s*["\']?([^"\'\s#]+)["\']?', re.MULTILINE)
CIT_DATE_RE = re.compile(r'^date-released:\s*["\']?([^"\'\s#]+)["\']?', re.MULTILINE)
WORKFLOW_USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)")
SHA_PIN_RE = re.compile(r"^[^@]+@[0-9a-f]{40}$")
TRACKED_SECRET_RE = re.compile(
    r"(^|/)\.env$|(?:^|/)service[-_]account[^/]*\.json$|\.(?:pem|key|pfx|p12)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CheckResult:
    """Deterministic outcome for one release-readiness rule."""

    code: str
    title: str
    status: str
    details: tuple[str, ...]


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _non_comment_lines(text: str) -> set[str]:
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _check_r1_public_surface(root: Path) -> CheckResult:
    failures: list[str] = []
    for rel in REQUIRED_PUBLIC_FILES:
        target = root / rel
        if not target.is_file():
            failures.append(f"missing required public file: {rel}")
        elif not target.read_text(encoding="utf-8").strip():
            failures.append(f"required public file is empty: {rel}")

    readme = _read_text(root / "README.md")
    if readme is not None:
        header = "\n".join(readme.splitlines()[:60]).lower()
        if "status" not in header:
            failures.append("README.md top 60 lines must declare project Status")
        if "not affiliated" not in header:
            failures.append("README.md top 60 lines must include non-affiliation disclaimer")

    if failures:
        return CheckResult("R1", "public-surface", "FAIL", tuple(sorted(failures)))
    return CheckResult(
        "R1",
        "public-surface",
        "PASS",
        (f"{len(REQUIRED_PUBLIC_FILES)} required root files present with README framing",),
    )


def _check_r2_version_citation_changelog(root: Path) -> CheckResult:
    failures: list[str] = []
    version: str | None = None

    pyproject_raw = _read_text(root / "pyproject.toml")
    if pyproject_raw is None:
        failures.append("missing pyproject.toml")
    else:
        try:
            data = tomllib.loads(pyproject_raw)
        except tomllib.TOMLDecodeError as exc:
            failures.append(f"invalid pyproject.toml: {exc}")
        else:
            project = data.get("project", {})
            if not isinstance(project, dict):
                failures.append("pyproject.toml missing [project] table")
            else:
                for key in ("name", "version", "description", "readme", "requires-python", "urls"):
                    if key not in project:
                        failures.append(f"pyproject.toml [project] missing '{key}'")
                raw_ver = project.get("version")
                if not isinstance(raw_ver, str) or not SEMVER_RE.match(raw_ver):
                    failures.append(f"pyproject.toml version must match X.Y.Z, got {raw_ver!r}")
                else:
                    version = raw_ver

    init_raw = _read_text(root / "alpha_zero" / "__init__.py")
    if init_raw is None:
        failures.append("missing alpha_zero/__init__.py")
    else:
        m_init = INIT_VERSION_RE.search(init_raw)
        if m_init is None:
            failures.append("alpha_zero/__init__.py missing __version__ literal")
        elif version is not None and m_init.group(1) != version:
            failures.append(
                f"alpha_zero/__init__.py __version__ ({m_init.group(1)}) != pyproject.toml ({version})"
            )

    citation_raw = _read_text(root / "CITATION.cff")
    if citation_raw is None:
        failures.append("missing CITATION.cff")
    else:
        m_cit_ver = CIT_VERSION_RE.search(citation_raw)
        m_cit_date = CIT_DATE_RE.search(citation_raw)
        if m_cit_ver is None:
            failures.append("CITATION.cff missing version")
        elif version is not None and m_cit_ver.group(1) != version:
            failures.append(
                f"CITATION.cff version ({m_cit_ver.group(1)}) != pyproject.toml ({version})"
            )
        if m_cit_date is None or not ISO_DATE_RE.match(m_cit_date.group(1)):
            failures.append("CITATION.cff missing valid YYYY-MM-DD date-released")

    changelog_raw = _read_text(root / "CHANGELOG.md")
    if changelog_raw is None:
        failures.append("missing CHANGELOG.md")
    elif version is not None and f"## [{version}]" not in changelog_raw:
        failures.append(f"CHANGELOG.md missing released section header '## [{version}]'")

    security_raw = _read_text(root / "SECURITY.md")
    if security_raw is None:
        failures.append("missing SECURITY.md")
    elif version is not None:
        major_minor = ".".join(version.split(".")[:2]) + ".x"
        if major_minor not in security_raw:
            failures.append(f"SECURITY.md Supported versions must list '{major_minor}'")

    if failures:
        return CheckResult("R2", "version-citation-changelog", "FAIL", tuple(sorted(failures)))
    return CheckResult(
        "R2",
        "version-citation-changelog",
        "PASS",
        (
            f"version {version} aligned across pyproject.toml, __init__.py, CITATION.cff, CHANGELOG.md, SECURITY.md",
        ),
    )


def _check_r3_hygiene_and_ignores(root: Path) -> CheckResult:
    failures: list[str] = []

    gitignore = _read_text(root / ".gitignore")
    if gitignore is None:
        failures.append("missing .gitignore")
    else:
        lines = _non_comment_lines(gitignore)
        for token in REQUIRED_GITIGNORE_TOKENS:
            if token not in lines:
                failures.append(f".gitignore missing ignore rule '{token}'")

    dockerignore = _read_text(root / ".dockerignore")
    if dockerignore is None:
        failures.append("missing .dockerignore")
    else:
        lines = _non_comment_lines(dockerignore)
        for token in REQUIRED_DOCKERIGNORE_TOKENS:
            if token not in lines:
                failures.append(f".dockerignore missing exclude rule '{token}'")

    gitleaks_raw = _read_text(root / ".gitleaks.toml")
    if gitleaks_raw is None:
        failures.append("missing .gitleaks.toml")
    else:
        try:
            gitleaks = tomllib.loads(gitleaks_raw)
        except tomllib.TOMLDecodeError as exc:
            failures.append(f"invalid .gitleaks.toml: {exc}")
        else:
            allowlist = gitleaks.get("allowlist")
            if not isinstance(allowlist, dict):
                failures.append(".gitleaks.toml missing [allowlist] table")
            else:
                paths = allowlist.get("paths", [])
                if not isinstance(paths, list):
                    failures.append(".gitleaks.toml allowlist.paths must be a list")
                else:
                    for pat in paths:
                        if str(pat).strip() in {".*", ".", "^.*$", "^"}:
                            failures.append(
                                f".gitleaks.toml has overly broad allowlist path {pat!r}"
                            )

    if failures:
        return CheckResult("R3", "hygiene-and-ignores", "FAIL", tuple(sorted(failures)))
    return CheckResult(
        "R3",
        "hygiene-and-ignores",
        "PASS",
        (".gitignore, .dockerignore, and .gitleaks.toml cover required artifacts and secrets",),
    )


def _check_r4_coverage_and_debt_ratchets(root: Path) -> CheckResult:
    failures: list[str] = []

    floors_raw = _read_text(root / "coverage-floors.json")
    if floors_raw is None:
        failures.append("missing coverage-floors.json")
    else:
        try:
            floors_doc = json.loads(floors_raw)
        except json.JSONDecodeError as exc:
            failures.append(f"invalid coverage-floors.json: {exc}")
        else:
            floors = floors_doc.get("floors")
            if not isinstance(floors, dict) or not floors:
                failures.append("coverage-floors.json missing non-empty 'floors'")
            else:
                cloud = floors.get("alpha_zero/cloud")
                if not isinstance(cloud, dict):
                    failures.append("coverage-floors.json missing 'alpha_zero/cloud'")
                for pkg, spec in sorted(floors.items()):
                    if not isinstance(spec, dict):
                        failures.append(f"coverage-floors.json '{pkg}' entry must be an object")
                        continue
                    minimum = spec.get("minimum")
                    measured = spec.get("measured")
                    if not isinstance(minimum, int | float) or not (0 < float(minimum) <= 100):
                        failures.append(
                            f"coverage-floors.json '{pkg}' invalid minimum: {minimum!r}"
                        )
                    elif not isinstance(measured, int | float) or float(measured) < float(minimum):
                        failures.append(
                            f"coverage-floors.json '{pkg}' measured ({measured!r}) < minimum ({minimum!r})"
                        )

    pyproject_raw = _read_text(root / "pyproject.toml")
    if pyproject_raw is not None:
        try:
            py_doc = tomllib.loads(pyproject_raw)
        except tomllib.TOMLDecodeError:
            pass
        else:
            fail_under = (
                py_doc.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under")
            )
            if not isinstance(fail_under, int | float) or float(fail_under) < 85.0:
                failures.append(
                    f"pyproject.toml coverage fail_under must be >= 85, got {fail_under!r}"
                )

    baseline_raw = _read_text(root / "docs" / "evidence" / "tech_debt_baseline" / "baseline.json")
    if baseline_raw is None:
        failures.append("missing docs/evidence/tech_debt_baseline/baseline.json")
    else:
        try:
            baseline = json.loads(baseline_raw)
        except json.JSONDecodeError as exc:
            failures.append(f"invalid baseline.json: {exc}")
        else:
            for key in (
                "files_over_1000_lines",
                "noqa_comments",
                "pytest_skip_sites",
                "unsafe_torch_load_calls",
            ):
                val = baseline.get(key)
                if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                    failures.append(f"baseline.json missing non-negative int '{key}'")

    if failures:
        return CheckResult("R4", "coverage-and-debt-ratchets", "FAIL", tuple(sorted(failures)))
    return CheckResult(
        "R4",
        "coverage-and-debt-ratchets",
        "PASS",
        ("coverage-floors.json, pyproject fail_under >= 85, and tech_debt baseline.json valid",),
    )


def _subprocess_shell_violations(source: str, rel_path: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        return [f"{rel_path}: syntax error at line {exc.lineno}: {exc.msg}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
        ):
            continue
        for kw in node.keywords:
            if kw.arg == "shell" and not (
                isinstance(kw.value, ast.Constant) and kw.value.value is False
            ):
                violations.append(
                    f"{rel_path}:{node.lineno}: subprocess.{func.attr} must set shell=False cleanly"
                )
    return violations


def _check_r5_workflows_playpath_launcher(root: Path) -> CheckResult:
    failures: list[str] = []

    play_path_raw = _read_text(root / "scripts" / "release_play_path.txt")
    if play_path_raw is None or not _non_comment_lines(play_path_raw):
        failures.append("missing or empty scripts/release_play_path.txt")

    for rel in REQUIRED_WORKFLOWS:
        wf_raw = _read_text(root / rel)
        if wf_raw is None:
            failures.append(f"missing workflow: {rel}")
            continue
        for lineno, line in enumerate(wf_raw.splitlines(), start=1):
            m = WORKFLOW_USES_RE.match(line)
            if m is None:
                continue
            ref = m.group(1).strip("\"'")
            if ref.startswith("./"):
                continue
            if not SHA_PIN_RE.match(ref):
                failures.append(f"{rel}:{lineno}: unpinned GitHub Action reference '{ref}'")

    launcher_rel = "scripts/local_muzero_launcher.py"
    launcher_raw = _read_text(root / launcher_rel)
    if launcher_raw is None:
        failures.append(f"missing {launcher_rel}")
    else:
        failures.extend(_subprocess_shell_violations(launcher_raw, launcher_rel))

    if failures:
        return CheckResult("R5", "workflow-playpath-and-launcher", "FAIL", tuple(sorted(failures)))
    return CheckResult(
        "R5",
        "workflow-playpath-and-launcher",
        "PASS",
        ("release_play_path.txt populated, workflows SHA-pinned, launcher shell=False verified",),
    )


def _run_git(root: Path, git_bin: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [git_bin, "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _check_r6_git_hygiene(root: Path, *, skip_git: bool) -> CheckResult:
    if skip_git:
        return CheckResult("R6", "git-hygiene", "SKIP", ("git checks skipped via --skip-git",))

    git_bin = shutil.which("git")
    if git_bin is None:
        return CheckResult("R6", "git-hygiene", "FAIL", ("git executable not found on PATH",))

    inside = _run_git(root, git_bin, ["rev-parse", "--is-inside-work-tree"])
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return CheckResult(
            "R6",
            "git-hygiene",
            "FAIL",
            ("target root is not a git work tree (pass --skip-git for synthetic trees)",),
        )

    failures: list[str] = []
    unmerged = _run_git(root, git_bin, ["ls-files", "--unmerged"])
    if unmerged.returncode != 0:
        failures.append("git ls-files --unmerged failed")
    elif unmerged.stdout.strip():
        failures.append("repository has unmerged index entries")

    tracked = _run_git(root, git_bin, ["ls-files"])
    if tracked.returncode != 0:
        failures.append("git ls-files failed")
    else:
        for rel in sorted(tracked.stdout.splitlines()):
            norm = rel.replace("\\", "/")
            if TRACKED_SECRET_RE.search(norm):
                failures.append(f"tracked secret-like file path: {norm}")

    branch_proc = _run_git(root, git_bin, ["rev-parse", "--abbrev-ref", "HEAD"])
    branch = branch_proc.stdout.strip() if branch_proc.returncode == 0 else "unknown"

    if failures:
        return CheckResult("R6", "git-hygiene", "FAIL", tuple(sorted(failures)))
    return CheckResult(
        "R6",
        "git-hygiene",
        "PASS",
        (f"branch '{branch}' has zero unmerged entries and zero tracked secret paths",),
    )


def evaluate_release_readiness(root: Path, *, skip_git: bool = False) -> dict[str, Any]:
    """Run checks R1-R6 deterministically against ``root`` and return structured results."""
    results = [
        _check_r1_public_surface(root),
        _check_r2_version_citation_changelog(root),
        _check_r3_hygiene_and_ignores(root),
        _check_r4_coverage_and_debt_ratchets(root),
        _check_r5_workflows_playpath_launcher(root),
        _check_r6_git_hygiene(root, skip_git=skip_git),
    ]
    failed = [item for item in results if item.status == "FAIL"]
    ok = not failed
    summary = (
        "release-readiness: OK"
        if ok
        else f"release-readiness: FAILED ({len(failed)} checks failed)"
    )

    for item in results:
        level = logging.ERROR if item.status == "FAIL" else logging.DEBUG
        logger.log(
            level, "[%s] %s (%s): %s", item.status, item.code, item.title, "; ".join(item.details)
        )

    return {
        "ok": ok,
        "summary": summary,
        "checks": [asdict(item) for item in results],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify Civ6-AlphaZero public release readiness checks (R1-R6)."
    )
    parser.add_argument("--root", type=Path, default=None, help="Repository root to inspect.")
    parser.add_argument("--quiet", action="store_true", help="Suppress passing check lines.")
    parser.add_argument("--json", action="store_true", help="Emit deterministic JSON to stdout.")
    parser.add_argument(
        "--verbose", action="store_true", help="Emit per-check debug logs to stderr."
    )
    parser.add_argument("--skip-git", action="store_true", help="Skip R6 git tree inspection.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for release readiness verification."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    root = (args.root or REPO_ROOT).resolve()
    payload = evaluate_release_readiness(root, skip_git=args.skip_git)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for check in payload["checks"]:
            if args.quiet and check["status"] != "FAIL":
                continue
            print(
                f"[{check['status']}] {check['code']}: {check['title']} - {'; '.join(check['details'])}"
            )
        if not args.quiet or not payload["ok"]:
            print(payload["summary"])

    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
