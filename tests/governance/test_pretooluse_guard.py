"""Behavioural tests for ``.claude/hooks/pretooluse_guard.sh``.

The guard is a bash script that reads a Claude Code PreToolUse payload on
stdin and either prints a deny decision or prints nothing (allow). These
tests drive it exactly the way the hook runner does -- ``bash <script>`` with
the JSON on stdin -- from inside a throwaway git repository whose
``refs/remotes/origin/HEAD`` symbolic ref points at a branch that is *not*
``main``. That is the shape of this repository (its default branch is not
``main``), and it is the case the guard historically missed.

Stdlib only; run via ``pytest -c tests/governance/pytest.ini``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / ".claude" / "hooks" / "pretooluse_guard.sh"

DEFAULT_BRANCH = "trunk"  # deliberately not `main`
FEATURE_BRANCH = "feature/work"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("jq") is None,
    reason="guard needs git and jq on PATH (both present on ubuntu-latest)",
)


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        # Do not let the developer's global config (e.g. init.defaultBranch,
        # commit signing) leak into the fixture repo.
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repo on a feature branch whose origin/HEAD resolves to ``trunk``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", FEATURE_BRANCH)
    _git(repo, "commit", "-q", "--allow-empty", "-m", "init")
    # `git symbolic-ref` happily creates a symref to a ref that does not exist
    # yet, which is all the guard reads (`--short` resolves the name only).
    _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", f"refs/remotes/origin/{DEFAULT_BRANCH}")
    assert _git(repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD") == (
        f"origin/{DEFAULT_BRANCH}"
    )
    return repo


def _run_guard(repo: Path, command: str, env_extra: dict[str, str] | None = None) -> str | None:
    """Return the deny reason, or None when the guard allowed the command."""
    # Start from a clean slate for the knob under test so a developer's own
    # CIV6AZ_DEFAULT_BRANCH cannot leak into the fixture's expectations.
    env = {k: v for k, v in os.environ.items() if k != "CIV6AZ_DEFAULT_BRANCH"}
    env.update(env_extra or {})
    payload = json.dumps({"tool_input": {"command": command}})
    proc = subprocess.run(
        ["bash", str(GUARD)],
        cwd=repo,
        input=payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    if not proc.stdout.strip():
        return None
    decision = json.loads(proc.stdout)["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"
    return str(decision["permissionDecisionReason"])


# --- push destinations -------------------------------------------------------


def test_push_to_main_is_denied(repo: Path) -> None:
    reason = _run_guard(repo, "git push origin main")
    assert reason is not None and "S3.2" in reason


def test_push_to_resolved_default_branch_is_denied(repo: Path) -> None:
    """The case the literal-`main` guard missed: origin/HEAD is `trunk`."""
    reason = _run_guard(repo, f"git push origin {DEFAULT_BRANCH}")
    assert reason is not None and DEFAULT_BRANCH in reason


@pytest.mark.parametrize(
    "refspec",
    [f"HEAD:{DEFAULT_BRANCH}", f"refs/heads/{DEFAULT_BRANCH}", f"'{DEFAULT_BRANCH}'"],
)
def test_default_branch_as_refspec_destination_is_denied(repo: Path, refspec: str) -> None:
    assert _run_guard(repo, f"git push origin {refspec}") is not None


def test_push_to_feature_branch_is_allowed(repo: Path) -> None:
    assert _run_guard(repo, f"git push origin {FEATURE_BRANCH}") is None


def test_default_branch_prefix_is_not_over_matched(repo: Path) -> None:
    """`trunk-fix` must not be caught by the `trunk` rule (whitespace/EOL anchored)."""
    assert _run_guard(repo, f"git push origin {DEFAULT_BRANCH}-fix") is None


def test_mention_inside_quoted_argument_is_allowed(repo: Path) -> None:
    """Anchored matching: prose that names the command is not the command."""
    assert _run_guard(repo, f"echo 'git push origin {DEFAULT_BRANCH}'") is None


# --- force pushes ------------------------------------------------------------


def test_force_push_is_denied(repo: Path) -> None:
    reason = _run_guard(repo, f"git push --force origin {FEATURE_BRANCH}")
    assert reason is not None and "force" in reason


def test_force_with_lease_is_allowed(repo: Path) -> None:
    assert _run_guard(repo, f"git push --force-with-lease origin {FEATURE_BRANCH}") is None


# --- checked-out branch ------------------------------------------------------


def test_commit_on_feature_branch_is_allowed(repo: Path) -> None:
    assert _run_guard(repo, "git commit -m 'x'") is None


@pytest.mark.parametrize("protected", ["main", DEFAULT_BRANCH])
def test_commit_while_checked_out_on_protected_branch_is_denied(repo: Path, protected: str) -> None:
    """Regression: the old `A || B && C` parsed as `(A || B) && C`, so a commit
    on `main` was denied only when the segment also *mentioned* main."""
    _git(repo, "checkout", "-q", "-b", protected)
    reason = _run_guard(repo, "git commit -m 'x'")
    assert reason is not None and protected in reason


# --- resolution order --------------------------------------------------------


def test_env_override_takes_precedence_over_origin_head(repo: Path) -> None:
    override = {"CIV6AZ_DEFAULT_BRANCH": "develop"}
    assert _run_guard(repo, "git push origin develop", override) is not None
    # With the override set, origin/HEAD's answer is no longer consulted...
    assert _run_guard(repo, f"git push origin {DEFAULT_BRANCH}", override) is None
    # ...but the literal `main` stays protected regardless.
    assert _run_guard(repo, "git push origin main", override) is not None


def test_falls_back_to_main_when_origin_head_is_unset(repo: Path) -> None:
    _git(repo, "symbolic-ref", "--delete", "refs/remotes/origin/HEAD")
    assert _run_guard(repo, "git push origin main") is not None
    assert _run_guard(repo, f"git push origin {DEFAULT_BRANCH}") is None


def test_default_branch_with_regex_metacharacters_is_escaped(repo: Path) -> None:
    """A branch named `release.1` must match literally, not as `release?1`."""
    override = {"CIV6AZ_DEFAULT_BRANCH": "release.1"}
    assert _run_guard(repo, "git push origin release.1", override) is not None
    assert _run_guard(repo, "git push origin releaseX1", override) is None
