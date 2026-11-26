"""H1: the push guard asks before a push that would re-create a merged branch.

Slip: after a PR merged and GitHub deleted its branch, the agent pushed the old
branch name again; it came back on the remote, pointing at commits already on the
default branch. The rule is a convenience net (local git only, never turns a deny
into an allow), so these tests drive the real hook in real throwaway repositories
that have a bare ``origin``.

Rule under test: ask when the pushed commit is on a protected branch that has a
local remote-tracking ref (dev, qa, main, the resolved default, and every name in
``CIV6AZ_PROTECTED_BRANCHES``) AND the destination's tracking ref is absent or is
itself fully merged into that protected branch's tracking ref (a stale ref that
``git`` never pruned after the remote branch was deleted).

Truth table (source commit x destination tracking ref ``refs/remotes/origin/<dst>``):

=================  ========  ===============  ==================
source merged?     absent    present, merged  present, unmerged
=================  ========  ===============  ==================
yes                ask       ask              allow
no                 allow     allow            allow
=================  ========  ===============  ==================
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.governance._bash import resolve_bash

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / ".claude" / "hooks" / "pretooluse_guard.sh"
SETTINGS = REPO_ROOT / ".claude" / "settings.json"
BASH = resolve_bash()
TRUNK = "trunk"  # deliberately not `main`
MERGED = "claude/merged-work"
KNOBS = ("CIV6AZ_DEFAULT_BRANCH", "CIV6AZ_PROTECTED_BRANCHES")
_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None
    or not any(shutil.which(tool) for tool in ("jq", "python3", "python")),
    reason="guard needs git plus jq or python on PATH",
)


def _shipped_env() -> dict[str, str]:
    """The ``env`` block of the real settings.json, as Claude Code exports it."""
    env = json.loads(SETTINGS.read_text(encoding="utf-8"))["env"]
    return {str(k): str(v) for k, v in env.items()}


def _shipped_trunk() -> str:
    """The real default branch's long name: the one protected name that is not dev/qa/main."""
    names = _shipped_env()["CIV6AZ_PROTECTED_BRANCHES"].split()
    return next(n for n in names if n not in ("dev", "qa", "main"))


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env={**os.environ, **_ENV},
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _decide(
    repo: Path,
    command: str,
    *,
    shipped: bool = False,
    env: dict[str, str] | None = None,
    bash: str | None = None,
) -> tuple[str | None, str]:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    base = {k: v for k, v in os.environ.items() if k not in KNOBS}
    if shipped:
        base.update(_shipped_env())
    proc = subprocess.run(
        [bash or BASH, GUARD.as_posix()],
        cwd=repo,
        input=payload,
        env=env if env is not None else base,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    if not proc.stdout.strip():
        return None, ""
    out = json.loads(proc.stdout)["hookSpecificOutput"]  # the JSON decision must parse
    assert out["hookEventName"] == "PreToolUse"
    return str(out["permissionDecision"]), str(out["permissionDecisionReason"])


def _build(tmp_path: Path, trunk: str) -> Path:
    """A clone whose ``origin/<trunk>`` exists; HEAD is a branch at trunk's tip."""
    bare = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", trunk, str(bare))
    work = tmp_path / "work"
    _git(tmp_path, "init", "-q", "-b", trunk, str(work))
    _git(work, "commit", "-q", "--allow-empty", "-m", "init")
    _git(work, "remote", "add", "origin", str(bare))
    _git(work, "push", "-q", "origin", trunk)
    _git(work, "fetch", "-q", "origin")
    _git(work, "remote", "set-head", "origin", trunk)
    # A merged branch: points at a commit already on origin/<trunk>, never pushed.
    _git(work, "checkout", "-q", "-b", MERGED)
    return work


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _build(tmp_path, TRUNK)


@pytest.fixture
def shipped_repo(tmp_path: Path) -> Path:
    """The shipped setup: default branch is the long name, ``origin/dev`` does not exist."""
    work = _build(tmp_path, _shipped_trunk())
    assert (
        subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", "refs/remotes/origin/dev"],
            cwd=work,
            check=False,
        ).returncode
        != 0
    )
    return work


def _commit_on(repo: Path, branch: str, start: str = TRUNK) -> None:
    _git(repo, "checkout", "-q", "-b", branch, start)
    _git(repo, "commit", "-q", "--allow-empty", "-m", f"work on {branch}")


def _publish(repo: Path, branch: str) -> None:
    """Make ``branch`` exist on origin and as a remote-tracking ref (a protected one)."""
    _git(repo, "push", "-q", "origin", f"{branch}:refs/heads/{branch}")
    _git(repo, "fetch", "-q", "origin")


# --- the slip, and the original three conditions ------------------------------------


def test_slip_pushing_a_merged_branch_name_asks(repo: Path) -> None:
    decision, reason = _decide(repo, f"git push origin {MERGED}")
    assert decision == "ask"
    low = reason.lower()
    assert "default branch" in low
    assert "merged" in low


def test_new_feature_branch_with_unmerged_commit_is_allowed(repo: Path) -> None:
    _commit_on(repo, "feature/new")
    assert _decide(repo, "git push origin feature/new") == (None, "")


def test_push_u_origin_head_from_merged_untracked_branch_asks(repo: Path) -> None:
    decision, reason = _decide(repo, "git push -u origin HEAD")
    assert decision == "ask"
    assert "default branch" in reason.lower()


def test_push_u_origin_head_from_new_work_is_allowed(repo: Path) -> None:
    _commit_on(repo, "feature/new")
    assert _decide(repo, "git push -u origin HEAD") == (None, "")


def test_refspec_judged_on_source_merged_source_asks(repo: Path) -> None:
    _commit_on(repo, "feature/new")  # HEAD carries unmerged work; the refspec does not
    assert _decide(repo, f"git push origin {MERGED}:claude/other-name")[0] == "ask"


def test_refspec_judged_on_source_unmerged_source_is_allowed(repo: Path) -> None:
    _commit_on(repo, "feature/new")
    _git(repo, "checkout", "-q", MERGED)  # HEAD is merged; the refspec pushes new work
    assert _decide(repo, "git push origin feature/new:feature/target") == (None, "")


@pytest.mark.parametrize("dst", [TRUNK, "qa", "main", "dev"])
def test_protected_destination_is_still_denied_not_asked(repo: Path, dst: str) -> None:
    assert _decide(repo, f"git push origin HEAD:{dst}")[0] == "deny"


@pytest.mark.parametrize(
    "command",
    [
        f"git push --force origin {MERGED}",
        f"git push origin +{MERGED}",
        f"git push -f origin {MERGED}",
    ],
)
def test_forced_push_of_a_merged_branch_is_still_denied(repo: Path, command: str) -> None:
    assert _decide(repo, command)[0] == "deny"


def test_without_any_protected_tracking_ref_the_guard_allows(repo: Path) -> None:
    _git(repo, "update-ref", "-d", f"refs/remotes/origin/{TRUNK}")
    assert _decide(repo, f"git push origin {MERGED}") == (None, "")


def test_repository_without_any_remote_is_allowed(tmp_path: Path) -> None:
    solo = tmp_path / "solo"
    _git(tmp_path, "init", "-q", "-b", "feature/solo", str(solo))
    _git(solo, "commit", "-q", "--allow-empty", "-m", "init")
    assert _decide(solo, "git push origin feature/solo") == (None, "")


# --- B1: the shipped settings (default dev, protected list names the real default) --


def test_shipped_settings_the_slip_asks(shipped_repo: Path) -> None:
    decision, reason = _decide(shipped_repo, f"git push origin {MERGED}", shipped=True)
    assert decision == "ask"
    assert "merged" in reason.lower()


def test_shipped_settings_commit_on_dev_with_origin_dev_asks(shipped_repo: Path) -> None:
    _commit_on(shipped_repo, "dev", _shipped_trunk())
    _publish(shipped_repo, "dev")
    _git(shipped_repo, "checkout", "-q", "-b", "claude/on-dev", "dev")
    assert _decide(shipped_repo, "git push origin claude/on-dev", shipped=True)[0] == "ask"


def test_shipped_settings_commit_only_on_qa_asks_though_default_dev_is_missing(
    shipped_repo: Path,
) -> None:
    _commit_on(shipped_repo, "qa", _shipped_trunk())
    _publish(shipped_repo, "qa")
    _git(shipped_repo, "checkout", "-q", "-b", "claude/on-qa", "qa")
    assert _decide(shipped_repo, "git push origin claude/on-qa", shipped=True)[0] == "ask"


def test_shipped_settings_commit_on_no_protected_branch_is_allowed(shipped_repo: Path) -> None:
    _commit_on(shipped_repo, "feature/new", _shipped_trunk())
    assert _decide(shipped_repo, "git push origin feature/new", shipped=True) == (None, "")


def test_name_from_protected_list_alone_counts(tmp_path: Path) -> None:
    work = _build(tmp_path, "release-line")
    # `release-line` is neither dev/qa/main nor the default (env says dev): only the list names it.
    decision, _ = _decide(
        work,
        f"git push origin {MERGED}",
        env={
            **{k: v for k, v in os.environ.items() if k not in KNOBS},
            "CIV6AZ_DEFAULT_BRANCH": "dev",
            "CIV6AZ_PROTECTED_BRANCHES": "dev qa main release-line",
        },
    )
    assert decision == "ask"


# --- B2: a stale, never-pruned tracking ref must not hide the slip -------------------


def test_stale_tracking_ref_after_remote_delete_asks(repo: Path) -> None:
    _commit_on(repo, "feature/stale")
    _git(repo, "push", "-q", "origin", "feature/stale")
    _git(repo, "fetch", "-q", "origin")
    _git(repo, "checkout", "-q", TRUNK)
    _git(repo, "merge", "-q", "--ff-only", "feature/stale")  # the PR merges
    _git(repo, "push", "-q", "origin", TRUNK)
    _git(
        repo, "-C", str(repo.parent / "origin.git"), "branch", "-D", "feature/stale"
    )  # remote delete
    _git(repo, "fetch", "-q", "origin")  # no --prune: git keeps the stale ref
    _git(repo, "show-ref", "--verify", "refs/remotes/origin/feature/stale")  # precondition
    _git(repo, "checkout", "-q", "feature/stale")
    decision, reason = _decide(repo, "git push origin feature/stale")
    assert decision == "ask"
    assert "merged" in reason.lower()


def _arrange(repo: Path, src_merged: bool, dst_state: str) -> str:
    """Set up one cell of the truth table; return the push command."""
    tip = _git(repo, "rev-parse", TRUNK)
    src = MERGED
    if not src_merged:
        _commit_on(repo, "feature/src")
        src = "feature/src"
    if dst_state == "merged":
        _git(repo, "update-ref", "refs/remotes/origin/feature/dst", tip)
    elif dst_state == "unmerged":
        tree = _git(repo, "rev-parse", f"{TRUNK}^{{tree}}")
        other = _git(repo, "commit-tree", tree, "-p", tip, "-m", "remote-only work")
        _git(repo, "update-ref", "refs/remotes/origin/feature/dst", other)
    return f"git push origin {src}:feature/dst"


@pytest.mark.parametrize(
    ("src_merged", "dst_state", "expected"),
    [
        (True, "absent", "ask"),
        (True, "merged", "ask"),
        (True, "unmerged", None),
        (False, "absent", None),
        (False, "merged", None),
        (False, "unmerged", None),
    ],
)
def test_truth_table(repo: Path, src_merged: bool, dst_state: str, expected: str | None) -> None:
    command = _arrange(repo, src_merged, dst_state)
    assert _decide(repo, command)[0] == expected


def test_merged_branch_with_tracking_ref_at_its_own_tip_asks(repo: Path) -> None:
    # Inverts the first version's "tracked, merged: allowed": the ref is merged, so it asks.
    _git(repo, "update-ref", f"refs/remotes/origin/{MERGED}", "HEAD")
    assert _decide(repo, f"git push origin {MERGED}")[0] == "ask"


# --- behaviours the review found untested --------------------------------------------


def test_git_dash_c_from_another_directory_asks(repo: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    decision, _ = _decide(elsewhere, f"git -C {repo.as_posix()} push origin {MERGED}")
    assert decision == "ask"


def test_two_refspecs_one_merged_one_new_asks_naming_the_merged_one(repo: Path) -> None:
    _commit_on(repo, "feature/new")
    decision, reason = _decide(repo, f"git push origin feature/new {MERGED}")
    assert decision == "ask"
    assert MERGED in reason


@pytest.mark.parametrize(
    "command",
    [
        f"git push origin :{MERGED}",
        f"git push --delete origin {MERGED}",
        f"git push -d origin {MERGED}",
    ],
)
def test_deleting_a_remote_branch_does_not_ask(repo: Path, command: str) -> None:
    # A branch that exists on origin, with its remote-tracking ref (PR #171: an unqualified
    # deletion needs one, since a local branch of that name does not prove it is a branch there).
    _publish(repo, MERGED)
    assert _decide(repo, command) == (None, "")


def test_other_remote_does_not_ask(repo: Path, tmp_path: Path) -> None:
    other = tmp_path / "fork.git"
    _git(tmp_path, "init", "-q", "--bare", str(other))
    _git(repo, "remote", "add", "fork", str(other))
    assert _decide(repo, f"git push fork {MERGED}") == (None, "")


@pytest.mark.parametrize("command", ["git push", "git push origin"])
def test_bare_push_from_the_merged_branch_asks(repo: Path, command: str) -> None:
    assert _decide(repo, command)[0] == "ask"


def test_detached_head_does_not_crash(repo: Path) -> None:
    _git(repo, "checkout", "-q", "--detach")
    assert _decide(repo, "git push origin HEAD") == (None, "")


@pytest.mark.parametrize(
    "spec", ["HEAD:refs/heads/claude/new-name", f"refs/heads/{MERGED}:refs/heads/claude/new-name"]
)
def test_full_ref_forms_are_judged(repo: Path, spec: str) -> None:
    assert _decide(repo, f"git push origin {spec}")[0] == "ask"


def test_a_deny_in_another_segment_wins_over_the_ask(repo: Path) -> None:
    decision, _ = _decide(repo, f"git push origin {MERGED} && git push --force origin x")
    assert decision == "deny"


@pytest.mark.parametrize(
    "command",
    [
        f'pwsh -c "git push origin {MERGED}"',
        f"env X=1 git push origin {MERGED}",
        f"X=1 git push origin {MERGED}",
    ],
)
def test_wrapped_forms_ask(repo: Path, command: str) -> None:
    assert _decide(repo, command)[0] == "ask"


def test_non_ascii_destination_yields_parseable_json(repo: Path) -> None:
    name = "claude/café-日本"
    _git(repo, "checkout", "-q", "-b", name)
    decision, reason = _decide(repo, f"git push origin {name}")  # _decide parses the JSON
    assert decision == "ask"
    assert reason


def test_not_a_repository_does_not_ask_or_crash(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    env = {
        **{k: v for k, v in os.environ.items() if k not in KNOBS},
        "GIT_CEILING_DIRECTORIES": str(tmp_path),
    }
    assert _decide(empty, f"git push origin {MERGED}", env=env) == (None, "")


def test_git_missing_from_path_does_not_ask_or_crash(repo: Path, tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    if sys.platform == "win32":
        # Git\bin\bash.exe is a launcher that prepends /mingw64/bin (holding git);
        # invoke the underlying usr\bin\bash.exe directly so PATH stays isolated.
        resolved = Path(BASH).resolve()
        if resolved.parent.name.lower() == "bin" and resolved.parent.parent.name.lower() == "usr":
            msys_bash = resolved
        else:
            msys_bash = resolved.parents[1] / "usr" / "bin" / "bash.exe"
        assert msys_bash.is_file(), f"expected MSYS bash at {msys_bash}"
        clean_path = os.pathsep.join(
            [str(msys_bash.parent), str(Path(sys.executable).resolve().parent)]
        )
        assert shutil.which("git", path=clean_path) is None
        env["PATH"] = clean_path
        assert _decide(repo, f"git push origin {MERGED}", env=env, bash=str(msys_bash)) == (
            None,
            "",
        )
        return

    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in ("bash", "cat", "jq", "python3", "python"):
        found = shutil.which(tool)
        if found is None:
            continue
        target = bindir / Path(found).name
        try:
            target.symlink_to(found)
        except OSError:
            shutil.copy2(found, target)
    assert not (bindir / "git").exists()
    env["PATH"] = str(bindir)
    assert _decide(repo, f"git push origin {MERGED}", env=env) == (None, "")
