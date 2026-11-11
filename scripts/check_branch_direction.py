#!/usr/bin/env python3
"""Branch Direction: may this pull request's head merge into its base? (plan S3.4)

Run by ``.github/workflows/branch-policy.yml`` on every ``pull_request`` event
(opened, synchronize, reopened, edited). The verdict is a required status check
on ``dev``, ``qa`` and ``main`` once the maintainer PATCHes it into the rulesets
(cut-over runbook step 5).

Inputs come from the environment, never from the command line or the event
payload: ``PR_NUMBER`` and ``REPO`` (or ``GITHUB_REPOSITORY``). Base and head
are read from ``GET repos/{REPO}/pulls/{PR_NUMBER}`` at run time, so a re-run
after the PR's base was edited judges the current base, not the one in a stale
payload. The checkout must be a full clone (``fetch-depth: 0``): every rule is
ancestry over ``origin/<branch>`` refs.

Directions (S3.1):

* ``dev``  accepts any same-repo head except ``main``, ``qa``, ``hotfix/*``,
  ``evidence/*`` and ``rcfix/*``; ``backmerge/*`` gets the back-merge rule.
* ``qa``   accepts ``dev`` (promotion), ``evidence/*``, ``rcfix/*``,
  ``backmerge/*``.
* ``main`` accepts ``qa`` (release) and ``hotfix/*``.
* any other base has no direction policy and is allowed.

Rules (S3.4):

* promotion ``dev -> qa``: ``origin/main`` is an ancestor of the head;
* release ``qa -> main``: ``origin/main`` is an ancestor of the head (main's tip
  is always in qa, so this cannot deadlock; it catches a missing back-merge);
* ``backmerge/*``: with R = ``origin/main`` (into qa) or ``origin/main`` and
  ``origin/qa`` (into dev), every non-merge commit in base..head is reachable
  from R, and every parent of every merge commit is reachable from R, from
  ``origin/<base>``, or lies inside the range -- so any number of
  "Update branch" merges is fine. Each merge's ``git show --remerge-diff`` is
  written to the job summary;
* ``hotfix/*``: ``origin/main`` is an ancestor of the head, and no commit in
  main..head is reachable from ``origin/dev`` or ``origin/qa`` (a hotfix carries
  main plus new commits, never unreleased work);
* ``evidence/<rc tag>``: only ``docs/evidence/**`` changes and the rc tag is an
  ancestor of the head;
* ``rcfix/*``: no merge commits, and ``git cherry origin/dev <head> <base>``
  prints only ``-`` lines (every commit is patch-equivalent to one on dev).

Exit status: 0 allowed, 1 denied, 2 could not decide (bad input, API or git
failure). Anything but 0 fails the check: a gate that cannot inspect its input
must deny.

This module is deliberately stdlib-only and self-contained. The workflow runs
the copy from the default branch, not the pull request's own copy (which would
let a PR rewrite its own judge), and that copy is fetched as a single file.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: The GitHub REST API version every call pins. 2026-03-10 removed
#: ``merge_commit_sha`` from pull request payloads, which the release workflow's
#: ``wait-for-ci`` needs; one pin for every caller keeps the payloads alike.
API_VERSION = "2022-11-28"
CHECK_NAME = "Branch Direction"
EVIDENCE_PREFIX = "docs/evidence/"
RC_TAG_RE = re.compile(r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)-rc\.([1-9]\d*)$")
_PR_NUMBER_RE = re.compile(r"^[1-9]\d{0,9}$")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: Heads ``dev`` refuses outright: release-line branches flow the other way.
DEV_REFUSED_EXACT = ("main", "qa")
DEV_REFUSED_PREFIXES = ("hotfix/", "evidence/", "rcfix/")


class GitError(RuntimeError):
    """A git command failed; the caller must fail closed."""


class ApiError(RuntimeError):
    """A GitHub API call failed or returned an unusable payload."""


Api = Callable[[str], Any]


@dataclass(frozen=True)
class GitResult:
    returncode: int
    stdout: str
    stderr: str


class Git:
    """A thin, injectable wrapper over the ``git`` CLI in one working tree."""

    def __init__(self, cwd: Path | str = ".", *, env: dict[str, str] | None = None) -> None:
        self.cwd = Path(cwd)
        self.env = env

    def run(self, *args: str, check: bool = True) -> GitResult:
        proc = subprocess.run(
            ["git", *args],
            cwd=self.cwd,
            env=self.env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        result = GitResult(proc.returncode, proc.stdout, proc.stderr)
        if check and proc.returncode != 0:
            raise GitError(
                f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        return result

    def out(self, *args: str) -> str:
        return self.run(*args).stdout.strip()

    def lines(self, *args: str) -> list[str]:
        return [line for line in self.run(*args).stdout.splitlines() if line.strip()]

    def resolve(self, rev: str) -> str:
        """Full SHA of ``rev`` as a commit; raises :class:`GitError` if absent."""
        return self.out("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")

    def exists(self, rev: str) -> bool:
        return (
            self.run(
                "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", check=False
            ).returncode
            == 0
        )

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """``git merge-base --is-ancestor``; any status but 0/1 is an error."""
        result = self.run("merge-base", "--is-ancestor", ancestor, descendant, check=False)
        if result.returncode in (0, 1):
            return result.returncode == 0
        raise GitError(f"is-ancestor {ancestor} {descendant}: {result.stderr.strip()}")

    def rev_list(self, range_spec: str) -> list[str]:
        return self.lines("rev-list", range_spec)

    def parents(self, sha: str) -> list[str]:
        return self.out("rev-list", "--parents", "-n", "1", sha).split()[1:]

    def changed_paths(self, base: str, head: str) -> list[str]:
        """Paths the head changes relative to its merge base with ``base``."""
        return self.lines("diff", "--name-only", f"{base}...{head}")

    def ensure_commit(self, sha: str, *, pr_number: int, remote: str = "origin") -> None:
        """Make the PR head available; fetch ``refs/pull/N/head`` when it is not."""
        if self.exists(sha):
            return
        self.run("fetch", "--no-tags", remote, f"+refs/pull/{pr_number}/head")
        if not self.exists(sha):
            raise GitError(f"head commit {sha} is not reachable after fetching pull/{pr_number}")


@dataclass
class Decision:
    allowed: bool
    rule: str
    base: str
    head_ref: str
    reasons: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def markdown(self) -> str:
        verdict = "ALLOWED" if self.allowed else "DENIED"
        lines = [
            f"## {CHECK_NAME}: {verdict}",
            "",
            f"`{self.head_ref}` -> `{self.base}` (rule: `{self.rule}`)",
            "",
        ]
        lines += [f"- {reason}" for reason in self.reasons]
        for detail in self.details:
            lines += [
                "",
                "<details><summary>remerge-diff</summary>",
                "",
                "```diff",
                detail,
                "```",
                "</details>",
            ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _short(sha: str) -> str:
    return sha[:12]


def _reachable_from_any(git: Git, sha: str, refs: Iterable[str]) -> bool:
    return any(git.is_ancestor(sha, ref) for ref in refs)


def _promotion(git: Git, d: Decision, head: str, remote: str) -> Decision:
    d.rule = "promotion"
    if git.is_ancestor(f"{remote}/main", head):
        d.allowed = True
        d.reasons.append(f"{remote}/main is an ancestor of dev")
    else:
        d.reasons.append(
            f"{remote}/main is not an ancestor of dev: the last back-merge into dev was "
            "squashed or is missing. Recreate backmerge/<version>-to-dev from main and merge "
            "it with scripts/maintainer/merge_backmerge.sh (merge commit, never squash)."
        )
    return d


def _release(git: Git, d: Decision, head: str, remote: str) -> Decision:
    d.rule = "release"
    if git.is_ancestor(f"{remote}/main", head):
        d.allowed = True
        d.reasons.append(f"{remote}/main is an ancestor of qa")
    else:
        d.reasons.append(
            f"{remote}/main is not an ancestor of qa: a hotfix on main was not back-merged "
            "into qa (backmerge/<version>-to-qa)."
        )
    return d


def _backmerge(git: Git, d: Decision, head: str, base: str, remote: str) -> Decision:
    d.rule = "backmerge"
    if base not in ("dev", "qa"):
        d.reasons.append(f"backmerge/* may target dev or qa, not {base}")
        return d
    sources = [f"{remote}/main"] if base == "qa" else [f"{remote}/main", f"{remote}/qa"]
    base_ref = f"{remote}/{base}"
    commits = git.rev_list(f"{base_ref}..{head}")
    if not commits:
        d.reasons.append(f"nothing to merge: {d.head_ref} adds no commits to {base}")
        return d
    in_range = set(commits)
    problems: list[str] = []
    for sha in commits:
        parents = git.parents(sha)
        if len(parents) <= 1:
            if not _reachable_from_any(git, sha, sources):
                problems.append(
                    f"commit {_short(sha)} is not reachable from {' or '.join(sources)}: a "
                    "back-merge may carry only released commits"
                )
            continue
        for parent in parents:
            if parent in in_range or _reachable_from_any(git, parent, [*sources, base_ref]):
                continue
            problems.append(
                f"merge {_short(sha)} has parent {_short(parent)} reachable from neither "
                f"{' / '.join(sources)} nor {base_ref}"
            )
        remerge = git.run("show", "--remerge-diff", "--format=%H %s", sha, check=False)
        body = remerge.stdout.strip() if remerge.returncode == 0 else remerge.stderr.strip()
        d.details.append(f"remerge-diff {_short(sha)}\n{body}")
    if problems:
        d.reasons.extend(problems)
    else:
        d.allowed = True
        d.reasons.append(
            f"{len(commits)} commit(s), every one released ({' / '.join(sources)}) or an "
            f"update merge of {base_ref}"
        )
    return d


def _hotfix(git: Git, d: Decision, head: str, remote: str) -> Decision:
    d.rule = "hotfix"
    if not git.is_ancestor(f"{remote}/main", head):
        d.reasons.append(f"{remote}/main is not an ancestor of the hotfix: branch it from main")
        return d
    unreleased = [f"{remote}/dev", f"{remote}/qa"]
    missing = [ref for ref in unreleased if not git.exists(ref)]
    if missing:
        # Fail closed: without the ref there is nothing to prove the hotfix against.
        raise GitError(f"cannot check the hotfix: {', '.join(missing)} is not available")
    leaked = [
        sha
        for sha in git.rev_list(f"{remote}/main..{head}")
        if _reachable_from_any(git, sha, unreleased)
    ]
    if leaked:
        d.reasons.extend(
            f"commit {_short(sha)} comes from unreleased dev/qa history, not from main"
            for sha in leaked
        )
        return d
    d.allowed = True
    d.reasons.append(f"based on {remote}/main and carries only new commits")
    return d


def _evidence(git: Git, d: Decision, head: str, head_ref: str, remote: str) -> Decision:
    d.rule = "evidence"
    tag = head_ref.removeprefix("evidence/")
    if not RC_TAG_RE.match(tag):
        d.reasons.append(f"evidence branches are named evidence/vX.Y.Z-rc.N; got {head_ref}")
        return d
    if not git.exists(f"refs/tags/{tag}"):
        d.reasons.append(f"rc tag {tag} does not exist")
        return d
    if not git.is_ancestor(f"refs/tags/{tag}", head):
        d.reasons.append(
            f"rc tag {tag} is not an ancestor of the evidence branch: cut it from the tag"
        )
        return d
    paths = git.changed_paths(f"{remote}/qa", head)
    outside = [p for p in paths if not p.startswith(EVIDENCE_PREFIX)]
    if not paths:
        d.reasons.append("the evidence branch changes nothing")
    elif outside:
        d.reasons.extend(f"{p} is outside {EVIDENCE_PREFIX}" for p in outside)
    else:
        d.allowed = True
        d.reasons.append(f"{len(paths)} path(s), all under {EVIDENCE_PREFIX}; {tag} is an ancestor")
    return d


def _rcfix(git: Git, d: Decision, head: str, remote: str) -> Decision:
    d.rule = "rcfix"
    base_ref = f"{remote}/qa"
    commits = git.rev_list(f"{base_ref}..{head}")
    if not commits:
        d.reasons.append("the rcfix branch adds no commits")
        return d
    merges = [sha for sha in commits if len(git.parents(sha)) > 1]
    if merges:
        d.reasons.extend(
            f"merge commit {_short(sha)}: an rcfix is cherry-picks only" for sha in merges
        )
        return d
    cherry = git.lines("cherry", f"{remote}/dev", head, base_ref)
    unmatched = [line for line in cherry if not line.startswith("-")]
    if unmatched:
        d.reasons.extend(
            f"{line[2:14]} is not patch-equivalent to any commit on dev (hand-edited "
            "cherry-pick?). Fix it on dev first and `git cherry-pick -x` it with no conflict "
            "edits, or promote a new rc."
            for line in unmatched
        )
        return d
    d.allowed = True
    d.reasons.append(f"{len(cherry)} commit(s), each patch-equivalent to a commit on dev")
    return d


def evaluate(
    git: Git, *, base: str, head_ref: str, head_sha: str, remote: str = "origin"
) -> Decision:
    """Judge ``head_ref`` (at ``head_sha``) merging into ``base``."""
    d = Decision(allowed=False, rule="direction", base=base, head_ref=head_ref)
    if base == "dev":
        if head_ref in DEV_REFUSED_EXACT or head_ref.startswith(DEV_REFUSED_PREFIXES):
            d.reasons.append(
                f"{head_ref} may not merge into dev (S3.1: release-line branches flow away from dev)"
            )
            return d
        if head_ref.startswith("backmerge/"):
            return _backmerge(git, d, head_sha, base, remote)
        d.rule = "feature"
        d.allowed = True
        d.reasons.append(f"{head_ref} -> dev is a feature merge")
        return d
    if base == "qa":
        if head_ref == "dev":
            return _promotion(git, d, head_sha, remote)
        if head_ref.startswith("evidence/"):
            return _evidence(git, d, head_sha, head_ref, remote)
        if head_ref.startswith("rcfix/"):
            return _rcfix(git, d, head_sha, remote)
        if head_ref.startswith("backmerge/"):
            return _backmerge(git, d, head_sha, base, remote)
        d.reasons.append(
            f"{head_ref} may not merge into qa: qa accepts dev, evidence/*, rcfix/* and backmerge/*"
        )
        return d
    if base == "main":
        if head_ref == "qa":
            return _release(git, d, head_sha, remote)
        if head_ref.startswith("hotfix/"):
            return _hotfix(git, d, head_sha, remote)
        d.reasons.append(f"{head_ref} may not merge into main: main accepts qa and hotfix/*")
        return d
    d.rule = "unprotected-base"
    d.allowed = True
    d.reasons.append(f"{base} is not a protected branch; no direction policy applies")
    return d


# ---------------------------------------------------------------------------
# GitHub API and CLI
# ---------------------------------------------------------------------------


def gh_api_argv(path: str) -> list[str]:
    """The read-only ``gh api`` call for ``path`` (no method or field flags)."""
    return [
        "gh",
        "api",
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        f"X-GitHub-Api-Version: {API_VERSION}",
        path,
    ]


def gh_api(path: str) -> Any:
    """Default :data:`Api`: ``gh api`` with ``GH_TOKEN`` from the environment."""
    proc = subprocess.run(
        gh_api_argv(path), capture_output=True, text=True, encoding="utf-8", check=False
    )
    if proc.returncode != 0:
        raise ApiError(f"gh api {path} failed: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ApiError(f"gh api {path} returned invalid JSON: {exc}") from exc


def run(*, pr_number: int, repo: str, api: Api, git: Git, remote: str = "origin") -> Decision:
    """Fetch the PR from the API, then :func:`evaluate` it."""
    pr = api(f"repos/{repo}/pulls/{pr_number}")
    try:
        base = str(pr["base"]["ref"])
        head_ref = str(pr["head"]["ref"])
        head_sha = str(pr["head"]["sha"])
        head_repo = pr["head"].get("repo")
    except (KeyError, TypeError) as exc:
        raise ApiError(f"pull request payload is missing {exc}") from exc
    if not _SHA_RE.match(head_sha):
        raise ApiError(f"head sha {head_sha!r} is not a full commit SHA")
    head_full_name = head_repo.get("full_name") if isinstance(head_repo, dict) else None
    if head_full_name != repo:
        return Decision(
            allowed=False,
            rule="same-repo",
            base=base,
            head_ref=head_ref,
            reasons=[
                f"the head must be a branch of {repo}; it is in {head_full_name or 'a deleted repository'}"
            ],
        )
    git.ensure_commit(head_sha, pr_number=pr_number, remote=remote)
    return evaluate(git, base=base, head_ref=head_ref, head_sha=head_sha, remote=remote)


def _write_summary(text: str) -> None:
    target = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(text)


def main(
    argv: Sequence[str] | None = None, *, api: Api | None = None, git: Git | None = None
) -> int:
    del argv  # inputs come only from the environment
    pr_raw = os.environ.get("PR_NUMBER", "").strip()
    repo = (os.environ.get("REPO") or os.environ.get("GITHUB_REPOSITORY") or "").strip()
    if not _PR_NUMBER_RE.match(pr_raw) or not _REPO_RE.match(repo):
        print(
            f"::error::{CHECK_NAME}: PR_NUMBER={pr_raw!r} REPO={repo!r} are not valid",
            file=sys.stderr,
        )
        return 2
    try:
        decision = run(pr_number=int(pr_raw), repo=repo, api=api or gh_api, git=git or Git("."))
    except (ApiError, GitError) as exc:
        message = f"{CHECK_NAME} could not decide (failing closed): {exc}"
        print(f"::error::{message}", file=sys.stderr)
        _write_summary(f"## {CHECK_NAME}: ERROR\n\n{message}\n")
        return 2
    text = decision.markdown()
    print(text)
    _write_summary(text)
    if not decision.allowed:
        print(
            f"::error::{CHECK_NAME}: {decision.head_ref} -> {decision.base} denied", file=sys.stderr
        )
    return 0 if decision.allowed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
