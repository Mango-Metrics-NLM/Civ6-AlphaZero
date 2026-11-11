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
  ``merge_commit_sha`` is the tagged commit and whose base is ``qa`` (rc) or
  ``main`` (final). On that PR's HEAD, the newest ``Branch Direction`` run
  completed at or before the PR's ``merged_at`` -- the run that gated the merge,
  not a later re-run against refs the merge itself moved -- must have succeeded,
  and so must ``Release Evidence`` for a non-hotfix final. PR-scoped checks
  cannot arrive after the merge, so they are judged once, without polling.

Check runs count only when produced by GitHub Actions (``app.slug``). Push
checks are polled until the timeout; a transient API failure (5xx, connection)
is retried with bounded exponential backoff, anything else fails at once. Every
API call pins REST API version 2022-11-28: 2026-03-10 removed
``merge_commit_sha`` from pull request payloads, and a missing field fails
closed rather than matching nothing.
"""

from __future__ import annotations

import argparse
import datetime as dt
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
#: Consecutive transient API failures tolerated at one wait-for-ci call site.
MAX_TRANSIENT_RETRIES = 6
MAX_BACKOFF_S = 300.0


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
    """Paths ``git diff <rc>..<final>`` changes outside ``docs/evidence/``.

    ``--no-renames``: a move from ``alpha_zero/`` into ``docs/evidence/`` must
    report the source path too, or it would read as an evidence-only change.
    """
    return [
        p
        for p in git.lines("diff", "--name-only", "--no-renames", f"{rc_name}..{sha}")
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


def expected_pr_base(tag: ReleaseTag) -> str:
    """An rc is a qa state, a final a main state: the producing PR targets that branch."""
    return "qa" if tag.is_rc else "main"


def pr_base_problem(pr: dict[str, Any], tag: ReleaseTag) -> str | None:
    base = str((pr.get("base") or {}).get("ref", ""))
    want = expected_pr_base(tag)
    if base == want:
        return None
    return (
        f"{tag.name} must be the merge of a PR into {want}; PR #{pr.get('number')} "
        f"was merged into {base or 'an unknown base'}"
    )


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
    if (base_problem := pr_base_problem(pr, tag)) is not None:
        problems.append(base_problem)
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


def _actions_runs(data: Any) -> list[dict[str, Any]]:
    return [
        r
        for r in (data.get("check_runs") if isinstance(data, dict) else None) or []
        if isinstance(r, dict) and (r.get("app") or {}).get("slug") == ACTIONS_APP_SLUG
    ]


def latest_run(api: Api, repo: str, sha: str, name: str) -> dict[str, Any] | None:
    """The newest Actions run of ``name`` on ``sha`` (push checks)."""
    data = api(
        f"repos/{repo}/commits/{sha}/check-runs?check_name={quote(name)}&filter=latest&per_page=100"
    )
    runs = _actions_runs(data)
    return max(runs, key=lambda r: int(r.get("id") or 0)) if runs else None


def _timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def run_before(
    api: Api, repo: str, sha: str, name: str, cutoff: dt.datetime
) -> dict[str, Any] | None:
    """The newest completed Actions run of ``name`` on ``sha`` finished by ``cutoff``.

    For a PR-scoped check this is the run that gated the merge: an ``edited``
    event after the merge re-judges against refs the merge moved, and must not
    decide the release.
    """
    data = api(
        f"repos/{repo}/commits/{sha}/check-runs?check_name={quote(name)}&filter=all&per_page=100"
    )
    dated = []
    for r in _actions_runs(data):
        when = _timestamp(r.get("completed_at"))
        if r.get("status") == "completed" and when is not None and when <= cutoff:
            dated.append((when, int(r.get("id") or 0), r))
    if not dated:
        return None
    return max(dated, key=lambda item: (item[0], item[1]))[2]


def run_state(check_run: dict[str, Any] | None) -> str:
    if check_run is None:
        return "missing"
    if check_run.get("status") != "completed":
        return "pending"
    conclusion = check_run.get("conclusion")
    return "success" if conclusion == "success" else f"failure ({conclusion})"


def pr_check_problems(api: Api, repo: str, pr: dict[str, Any], tag: ReleaseTag) -> list[str]:
    """Judge the PR-scoped checks once, as they stood when the PR merged."""
    number = pr.get("number")
    head = str(pr.get("head", {}).get("sha", ""))
    merged_at = _timestamp(pr.get("merged_at"))
    if not head or merged_at is None:
        return [f"PR #{number} has no head sha or merged_at"]
    problems = []
    for name in pr_checks_for(tag, is_hotfix_pr(pr)):
        state = run_state(run_before(api, repo, head, name, merged_at))
        if state == "success":
            continue
        if name == RELEASE_EVIDENCE and state == "missing":
            problems.append(
                f"{RELEASE_EVIDENCE} never ran on PR #{number} (head {head[:12]}) before it "
                "merged. That job arrives with T14; until it exists and the maintainer adds it "
                "to main.json (action 9a), a non-hotfix final cannot publish."
            )
        else:
            problems.append(
                f"{name} on PR #{number} (head {head[:12]}) was {state} when it merged at "
                f"{pr.get('merged_at')}; PR-scoped checks cannot arrive after the merge"
            )
    return problems


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
    """Every required check green; an empty list means pass."""
    deadline = clock() + timeout_s

    def with_retry(call: Callable[[], Any]) -> Any:
        failures = 0
        while True:
            try:
                return call()
            except ApiError as exc:
                if not exc.transient or failures >= MAX_TRANSIENT_RETRIES or clock() >= deadline:
                    raise
                delay = min(interval_s * 2**failures, MAX_BACKOFF_S)
                failures += 1
                print(f"transient GitHub API error, retrying in {delay:.0f}s: {exc}", flush=True)
                sleep(delay)

    try:
        pr = with_retry(lambda: merged_pr_for_commit(api, repo, sha))
    except ApiError as exc:
        return [f"cannot find the PR that produced {sha[:12]}: {exc}"]
    base_problem = pr_base_problem(pr, tag)
    if base_problem is not None:
        return [base_problem]
    try:
        problems = with_retry(lambda: pr_check_problems(api, repo, pr, tag))
    except ApiError as exc:
        return [f"GitHub API error reading PR checks: {exc}"]
    if problems:
        return [f"required check failed: {p}" for p in problems]

    def poll() -> dict[str, str]:
        return {name: run_state(latest_run(api, repo, sha, name)) for name in PUSH_CHECKS}

    while True:
        try:
            states = with_retry(poll)
        except ApiError as exc:
            return [f"GitHub API error while polling push checks: {exc}"]
        failed = [f"{n} on {sha[:12]}: {s}" for n, s in states.items() if s.startswith("failure")]
        if failed:
            return [f"required check failed: {item}" for item in failed]
        waiting = [f"{n} on {sha[:12]} ({s})" for n, s in states.items() if s != "success"]
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
