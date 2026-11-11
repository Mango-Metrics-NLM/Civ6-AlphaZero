#!/usr/bin/env python3
"""The release gates (plan S3.5), shared by ``release.yml`` and ``tag_release.py``.

``verify-tag`` (job ``verify-tag`` in ``release.yml``; also every pre-check in
``scripts/maintainer/tag_release.py``):

* the tag's base version (``${version%%-*}``) equals ``pyproject.toml`` at the
  tagged commit;
* an rc is an ancestor of ``origin/qa`` and has ``origin/main`` as an ancestor
  (a squashed or missing back-merge is caught here as well as at promotion);
* a final is an ancestor of ``origin/main``;
* final-tag rule: ``git diff <highest rc>..<final>`` touches only
  ``docs/evidence/**``. A final with no rc is allowed only when the tagged
  commit is the merge of a ``hotfix/*`` PR.

``wait-for-ci`` (job ``wait-for-ci``; the publish job ``needs`` it):

* push checks -- the eight CI jobs that run on push -- must succeed on the
  tagged commit itself;
* PR-only checks are read from the PR that produced the commit:
  ``GET /commits/{sha}/pulls`` must yield exactly one merged PR whose
  ``merge_commit_sha`` is the tagged commit, and that PR's HEAD must carry a
  successful ``Branch Direction`` -- and ``Release Evidence`` for a non-hotfix
  final.

Check runs count only when produced by GitHub Actions (``app.slug``), and the
newest run of a name wins. Every API call pins REST API version 2022-11-28:
2026-03-10 removed ``merge_commit_sha`` from pull request payloads, and a
missing field fails closed rather than matching nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import quote

from scripts.check_branch_direction import (
    EVIDENCE_PREFIX,
    Api,
    ApiError,
    Git,
    GitError,
    gh_api,
)
from scripts.release_version import ReleaseError, ReleaseTag, parse_tag, pyproject_version

#: CI jobs that run on push to qa and main: the ruleset's required checks minus
#: the PR-only ones. tests/scripts/test_release_scripts.py holds this list to
#: .github/rulesets/main.json.
PUSH_CHECKS = (
    "Lint & Type Check",
    "Test (Python 3.11)",
    "Test (Python 3.12)",
    "Cloud Unit Tests",
    "Security Scan",
    "Governance Gates",
    "LLM Eval Harness",
    "Coverage Report",
)
BRANCH_DIRECTION = "Branch Direction"
RELEASE_EVIDENCE = "Release Evidence"
ACTIONS_APP_SLUG = "github-actions"
RELEASE_WORKFLOW = ".github/workflows/release.yml"
WAIT_JOB = "wait-for-ci"


# ---------------------------------------------------------------------------
# Tags and the final-tag rule
# ---------------------------------------------------------------------------


def rc_tags(git: Git, base: str) -> list[ReleaseTag]:
    """Every ``v<base>-rc.N`` tag, ordered by N (numerically)."""
    tags = []
    for name in git.lines("tag", "--list", f"v{base}-rc.*"):
        try:
            tags.append(parse_tag(name))
        except ReleaseError:
            continue
    return sorted(tags, key=lambda t: t.rc or 0)


def highest_rc(git: Git, base: str) -> ReleaseTag | None:
    tags = rc_tags(git, base)
    return tags[-1] if tags else None


def offending_final_paths(git: Git, rc_name: str, sha: str) -> list[str]:
    """Paths ``git diff <rc>..<final>`` changes outside ``docs/evidence/``."""
    return [
        p
        for p in git.lines("diff", "--name-only", f"{rc_name}..{sha}")
        if not p.startswith(EVIDENCE_PREFIX)
    ]


def merged_pr_for_commit(api: Api, repo: str, sha: str) -> dict[str, Any]:
    """The one merged PR whose merge commit is ``sha``; anything else fails."""
    prs = api(f"repos/{repo}/commits/{sha}/pulls?per_page=100")
    if not isinstance(prs, list):
        raise ApiError(f"commits/{sha}/pulls did not return a list")
    merged = [p for p in prs if isinstance(p, dict) and p.get("merged_at")]
    if any("merge_commit_sha" not in p for p in merged):
        raise ApiError(
            "pull request payload has no merge_commit_sha (REST API version must be 2022-11-28)"
        )
    matching = [p for p in merged if p["merge_commit_sha"] == sha]
    if len(matching) != 1:
        raise ApiError(
            f"expected exactly one merged PR whose merge commit is {sha}, found {len(matching)}; "
            "release tags go on the merge commit a PR produced"
        )
    pr: dict[str, Any] = matching[0]
    return pr


def is_hotfix_pr(pr: dict[str, Any]) -> bool:
    return str(pr.get("head", {}).get("ref", "")).startswith("hotfix/")


def verify_tag(
    git: Git, tag: ReleaseTag, sha: str, *, api: Api, repo: str, remote: str = "origin"
) -> list[str]:
    """Every S3.5 rule for tagging ``sha`` as ``tag``; an empty list means pass."""
    problems: list[str] = []
    try:
        declared = pyproject_version(git.out("show", f"{sha}:pyproject.toml"))
    except (GitError, ReleaseError) as exc:
        return [f"cannot read pyproject.toml at {sha[:12]}: {exc}"]
    if declared != tag.base:
        problems.append(
            f"{tag.name} is version {tag.base}, but pyproject.toml at {sha[:12]} says {declared}"
        )
    main, qa = f"{remote}/main", f"{remote}/qa"
    if tag.is_rc:
        if not git.is_ancestor(sha, qa):
            problems.append(f"an rc must be an ancestor of {qa}; {sha[:12]} is not")
        if not git.is_ancestor(main, sha):
            problems.append(
                f"{main} is not an ancestor of {sha[:12]}: the last back-merge was squashed or is missing"
            )
        return problems
    if not git.is_ancestor(sha, main):
        problems.append(f"a final must be an ancestor of {main}; {sha[:12]} is not")
    rc = highest_rc(git, tag.base)
    if rc is not None:
        if not git.is_ancestor(rc.name, sha):
            problems.append(f"{rc.name} is not an ancestor of {sha[:12]}")
        else:
            problems.extend(
                f"{path} changed after {rc.name}: a final may differ from its highest rc only "
                f"under {EVIDENCE_PREFIX}"
                for path in offending_final_paths(git, rc.name, sha)
            )
        return problems
    try:
        pr = merged_pr_for_commit(api, repo, sha)
    except ApiError as exc:
        problems.append(f"no rc for {tag.base} and the producing PR cannot be established: {exc}")
        return problems
    if not is_hotfix_pr(pr):
        problems.append(
            f"no rc for {tag.base}: a final without an rc must be the merge of a hotfix/* PR, "
            f"not {pr.get('head', {}).get('ref')!r}"
        )
    return problems


def is_r2_lineage(git: Git, sha: str) -> bool:
    """Does ``sha`` carry the R2 release workflow (a ``wait-for-ci`` gate)?

    Older lineage has a release workflow without it, so tagging there publishes
    without waiting for CI. Checked on content, not on a SHA: the workflow that a
    tag push runs is the one in the tagged commit.
    """
    import yaml  # local: release.yml does not need pyyaml for the other commands

    try:
        doc = yaml.safe_load(git.out("show", f"{sha}:{RELEASE_WORKFLOW}"))
        git.out("cat-file", "-e", f"{sha}:scripts/release_checks.py")
    except (GitError, yaml.YAMLError):
        return False
    jobs = doc.get("jobs") if isinstance(doc, dict) else None
    if not isinstance(jobs, dict) or WAIT_JOB not in jobs:
        return False
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        perms = job.get("permissions") or {}
        needs = job.get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        if isinstance(perms, dict) and perms.get("contents") == "write" and WAIT_JOB not in needs:
            return False
    return True


# ---------------------------------------------------------------------------
# wait-for-ci
# ---------------------------------------------------------------------------


def pr_checks_for(tag: ReleaseTag, hotfix: bool) -> tuple[str, ...]:
    if tag.is_rc or hotfix:
        return (BRANCH_DIRECTION,)
    return (BRANCH_DIRECTION, RELEASE_EVIDENCE)


def latest_run(api: Api, repo: str, sha: str, name: str) -> dict[str, Any] | None:
    data = api(
        f"repos/{repo}/commits/{sha}/check-runs?check_name={quote(name)}&filter=latest&per_page=100"
    )
    runs = [
        r
        for r in (data.get("check_runs") if isinstance(data, dict) else None) or []
        if isinstance(r, dict) and (r.get("app") or {}).get("slug") == ACTIONS_APP_SLUG
    ]
    return max(runs, key=lambda r: int(r.get("id") or 0)) if runs else None


def run_state(check_run: dict[str, Any] | None) -> str:
    if check_run is None:
        return "missing"
    if check_run.get("status") != "completed":
        return "pending"
    conclusion = check_run.get("conclusion")
    return "success" if conclusion == "success" else f"failure ({conclusion})"


def wait_for_ci(
    api: Api,
    repo: str,
    sha: str,
    tag: ReleaseTag,
    *,
    timeout_s: float,
    interval_s: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[str]:
    """Poll until every required check is green; an empty list means pass."""
    try:
        pr = merged_pr_for_commit(api, repo, sha)
    except ApiError as exc:
        return [f"cannot find the PR that produced {sha[:12]}: {exc}"]
    head = str(pr.get("head", {}).get("sha", ""))
    if not head:
        return [f"PR #{pr.get('number')} has no head sha"]
    targets = [(sha, name) for name in PUSH_CHECKS]
    targets += [(head, name) for name in pr_checks_for(tag, is_hotfix_pr(pr))]
    deadline = clock() + timeout_s
    while True:
        states = {
            (commit, name): run_state(latest_run(api, repo, commit, name))
            for commit, name in targets
        }
        failed = [
            f"{name} on {commit[:12]}: {state}"
            for (commit, name), state in states.items()
            if state.startswith("failure")
        ]
        if failed:
            return [f"required check failed: {item}" for item in failed]
        waiting = [
            f"{name} on {commit[:12]} ({state})"
            for (commit, name), state in states.items()
            if state != "success"
        ]
        if not waiting:
            return []
        if clock() >= deadline:
            return [f"timed out waiting for: {', '.join(waiting)}"]
        print(f"waiting for {len(waiting)} check(s): {', '.join(waiting)}", flush=True)
        sleep(interval_s)


# ---------------------------------------------------------------------------
# CLI (release.yml)
# ---------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None, *, api: Api | None = None, git: Git | None = None
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("verify-tag", "wait-for-ci"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True)
        p.add_argument("--sha", required=True)
        p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
        p.add_argument("--remote", default="origin")
        if name == "wait-for-ci":
            p.add_argument("--timeout-minutes", type=float, default=95)
            p.add_argument("--interval-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    api = api or gh_api
    git = git or Git(".")
    try:
        tag = parse_tag(args.tag)
        sha = git.resolve(args.sha)
        if not args.repo:
            raise ReleaseError("--repo (or GITHUB_REPOSITORY) is required")
        if args.cmd == "verify-tag":
            problems = verify_tag(git, tag, sha, api=api, repo=args.repo, remote=args.remote)
        else:
            problems = wait_for_ci(
                api,
                args.repo,
                sha,
                tag,
                timeout_s=args.timeout_minutes * 60,
                interval_s=args.interval_seconds,
            )
    except (ReleaseError, GitError, ApiError) as exc:
        print(f"::error::{args.cmd}: {exc}", file=sys.stderr)
        return 2
    for problem in problems:
        print(f"::error::{args.cmd}: {problem}", file=sys.stderr)
    if not problems:
        print(f"{args.cmd}: {tag.name} at {sha[:12]} passes")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
