#!/usr/bin/env python3
"""Tag a release candidate or a final release -- after every gate passes (plan S3.5).

MAINTAINER ONLY: the ``release-tags`` ruleset lets only the maintainer create
``v*`` tags, and with ``--execute`` this script refuses inside a Claude Code
session. Run it from your own terminal at the repository root:

    python -m scripts.maintainer.tag_release v2.1.0-rc.1            # dry run
    python -m scripts.maintainer.tag_release v2.1.0-rc.1 --execute  # tag + push

What it checks before tagging (any failure: exit 1, nothing tagged):

* the tag parses and does not exist locally or on the remote; an rc number is
  above every existing rc for that version, and no rc follows its final;
* the commit -- ``origin/qa`` for an rc, ``origin/main`` for a final, or
  ``--sha`` -- descends from the LF-R2 merge: its ``release.yml`` gates
  publishing on ``wait-for-ci`` (older lineage publishes with an unpinned,
  write-capable workflow), and it contains ``CIV6AZ_R2_MERGE_SHA`` /
  ``--r2-merge`` when the maintainer has recorded that merge;
* every ``release.yml`` rule (``scripts/release_checks.verify_tag``): version,
  ancestry (rc: on qa with main as an ancestor; final: on main), and the
  final-tag rule (diff from the highest rc only under ``docs/evidence/``, or a
  ``hotfix/*`` merge when there is no rc).

Then it creates an annotated tag and pushes exactly that ref. The release
workflow re-checks all of it on the tag push and waits for CI before
publishing.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence

from scripts.check_branch_direction import Api, ApiError, Git, GitError, gh_api
from scripts.release_checks import is_r2_lineage, rc_tags, verify_tag
from scripts.release_version import ReleaseError, ReleaseTag, parse_tag

_REMOTE_URL_RE = re.compile(
    r"github\.com[:/](?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)


def repo_from_remote(git: Git, remote: str) -> str:
    url = git.out("remote", "get-url", remote)
    match = _REMOTE_URL_RE.search(url)
    if not match:
        raise ReleaseError(f"cannot read owner/repo from {remote} URL {url!r}; pass --repo")
    return match.group("repo")


def tag_exists(git: Git, name: str, remote: str) -> bool:
    if git.exists(f"refs/tags/{name}"):
        return True
    return bool(git.out("ls-remote", "--tags", remote, f"refs/tags/{name}"))


def preflight(
    git: Git, tag: ReleaseTag, sha: str, *, api: Api, repo: str, remote: str, r2_merge: str | None
) -> list[str]:
    problems: list[str] = []
    if tag_exists(git, tag.name, remote):
        problems.append(f"{tag.name} already exists")
    if tag.is_rc:
        if tag_exists(git, f"v{tag.base}", remote):
            problems.append(f"v{tag.base} is already released; an rc cannot follow its final")
        existing = [t for t in rc_tags(git, tag.base) if t.name != tag.name]
        if existing and (existing[-1].rc or 0) >= (tag.rc or 0):
            problems.append(
                f"{tag.name} is not above the highest existing rc ({existing[-1].name})"
            )
    if not is_r2_lineage(git, sha):
        problems.append(
            f"{sha[:12]} does not descend from the LF-R2 merge: its release.yml publishes without "
            "a wait-for-ci gate. Tag only commits that carry the R2 release workflow."
        )
    if r2_merge:
        try:
            if not git.is_ancestor(r2_merge, sha):
                problems.append(
                    f"{sha[:12]} does not contain the recorded LF-R2 merge {r2_merge[:12]}"
                )
        except GitError as exc:
            problems.append(f"cannot check the recorded LF-R2 merge: {exc}")
    problems.extend(verify_tag(git, tag, sha, api=api, repo=repo, remote=remote))
    return problems


def main(
    argv: Sequence[str] | None = None, *, api: Api | None = None, git: Git | None = None
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.maintainer.tag_release",
        description="Tag vX.Y.Z or vX.Y.Z-rc.N after every release gate passes (maintainer only).",
    )
    parser.add_argument("tag", help="vX.Y.Z or vX.Y.Z-rc.N")
    parser.add_argument(
        "--sha", help="commit to tag (default: origin/qa for an rc, origin/main for a final)"
    )
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--repo", help="owner/repo (default: read from the remote URL)")
    parser.add_argument("--r2-merge", default=os.environ.get("CIV6AZ_R2_MERGE_SHA") or None)
    parser.add_argument(
        "--execute", action="store_true", help="create and push the tag (default: dry run)"
    )
    args = parser.parse_args(argv)
    git = git or Git(".")
    try:
        tag = parse_tag(args.tag)
    except ReleaseError as exc:
        print(f"tag_release: {exc}", file=sys.stderr)
        return 2
    if args.execute and os.environ.get("CLAUDECODE"):
        print(
            "tag_release: refusing --execute inside a Claude Code session; release tags are "
            "maintainer-only. Run this from your own terminal.",
            file=sys.stderr,
        )
        return 2
    try:
        git.run("fetch", "--quiet", "--prune", "--tags", args.remote)
        sha = git.resolve(args.sha or f"{args.remote}/{'qa' if tag.is_rc else 'main'}")
        repo = args.repo or repo_from_remote(git, args.remote)
        problems = preflight(
            git, tag, sha, api=api or gh_api, repo=repo, remote=args.remote, r2_merge=args.r2_merge
        )
    except (GitError, ApiError, ReleaseError) as exc:
        print(f"tag_release: could not check {tag.name}: {exc}", file=sys.stderr)
        return 2
    if problems:
        print(f"tag_release: refusing to tag {tag.name} at {sha[:12]}:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    tag_cmd = ["tag", "-a", tag.name, sha, "-m", f"Release {tag.name}"]
    push_cmd = ["push", args.remote, f"refs/tags/{tag.name}"]
    print(f"{tag.name} -> {sha} passes every release gate.")
    print("git " + " ".join(tag_cmd))
    print("git " + " ".join(push_cmd))
    if not args.execute:
        print("dry run: pass --execute to create and push the tag")
        return 0
    try:
        git.run(*tag_cmd)
        git.run(*push_cmd)
    except GitError as exc:
        print(f"tag_release: {exc}", file=sys.stderr)
        return 2
    print(f"pushed {tag.name}; release.yml re-checks it and waits for CI before publishing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
