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
# Resolved through PATH, not left to the OS: on Windows, CreateProcess searches
# System32 before PATH and finds WSL's bash.exe, which cannot see this checkout.
BASH = shutil.which("bash") or "bash"

DEFAULT_BRANCH = "trunk"  # deliberately not `main`
FEATURE_BRANCH = "feature/work"
# The value `.claude/settings.json` ships (plan §3.6). The old default branch
# stays protected until the rename task (R4) drops it.
OLD_DEFAULT = "claude/civ6-alphazero-implementation-vMnun"
SETTINGS_PROTECTED = f"dev qa main {OLD_DEFAULT}"
# Every knob the guard reads from the environment. Stripped before each run so
# the developer's own session (which exports the settings.json values) cannot
# leak into a fixture's expectations.
GUARD_ENV_KNOBS = ("CIV6AZ_DEFAULT_BRANCH", "CIV6AZ_PROTECTED_BRANCHES")

# The guard parses its payload with jq, else python3, else python. It used to
# skip unless jq was present, which meant it never ran on a Windows workstation
# (no jq there) -- the very machine the PowerShell rules exist for.
_HAS_PARSER = any(shutil.which(tool) for tool in ("jq", "python3", "python"))

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not _HAS_PARSER,
    reason="guard needs git plus jq or python on PATH",
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


def _raw_guard(
    repo: Path,
    payload: str,
    env_extra: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Run the guard on a raw stdin payload; return (decision, reason)."""
    env = {k: v for k, v in os.environ.items() if k not in GUARD_ENV_KNOBS}
    env.update(env_extra or {})
    proc = subprocess.run(
        [BASH, GUARD.as_posix()],
        cwd=repo,
        input=payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    if not proc.stdout.strip():
        return None, None
    decision = json.loads(proc.stdout)["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] in ("deny", "ask"), decision
    return str(decision["permissionDecision"]), str(decision["permissionDecisionReason"])


def _decide(
    repo: Path,
    command: str,
    env_extra: dict[str, str] | None = None,
    tool_name: str = "Bash",
) -> tuple[str | None, str | None]:
    payload = json.dumps({"tool_name": tool_name, "tool_input": {"command": command}})
    return _raw_guard(repo, payload, env_extra)


def _run_guard(repo: Path, command: str, env_extra: dict[str, str] | None = None) -> str | None:
    """Return the deny reason, or None when the guard allowed the command.

    An ``ask`` here is a test failure: callers of this helper expect a plain
    allow-or-deny, and an unexpected prompt is a behaviour change to look at.
    """
    decision, reason = _decide(repo, command, env_extra)
    assert decision != "ask", f"unexpected ask for {command!r}: {reason}"
    return reason


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
    [
        f"HEAD:{DEFAULT_BRANCH}",
        f"refs/heads/{DEFAULT_BRANCH}",
        f"'{DEFAULT_BRANCH}'",
        # `refs/heads/` on the DESTINATION side of a refspec. The pattern
        # allowed it only on the source side, so this idiomatic, legal push
        # slipped through while the header comment claimed it was covered.
        f"HEAD:refs/heads/{DEFAULT_BRANCH}",
        f"{DEFAULT_BRANCH}:refs/heads/{DEFAULT_BRANCH}",
    ],
)
def test_default_branch_as_refspec_destination_is_denied(repo: Path, refspec: str) -> None:
    assert _run_guard(repo, f"git push origin {refspec}") is not None


@pytest.mark.parametrize(
    "git_option",
    ["--no-pager", "-P", "-c user.name=x", "--literal-pathspecs", "-c k=v --no-pager"],
)
def test_global_git_options_do_not_bypass_the_guard(repo: Path, git_option: str) -> None:
    """A git global option before the subcommand must not defeat the rule.

    The option run was enumerated by name (-C, --git-dir, --work-tree), so
    every other one -- `--no-pager`, `-c key=val`, `-P` -- made the whole
    protected-branch rule a no-op. Probed 2026-09-02: `git push origin trunk`
    denied, `git --no-pager push origin trunk` allowed.
    """
    assert _run_guard(repo, f"git {git_option} push origin {DEFAULT_BRANCH}") is not None


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


# --- protected-branch list (plan §3.6, LF-G6) --------------------------------

SETTINGS_ENV = {"CIV6AZ_DEFAULT_BRANCH": "dev", "CIV6AZ_PROTECTED_BRANCHES": SETTINGS_PROTECTED}


@pytest.mark.parametrize("protected", ["dev", "qa", "main", OLD_DEFAULT])
def test_commit_and_push_on_every_settings_protected_branch_are_denied(
    repo: Path, protected: str
) -> None:
    """With the shipped settings, the OLD default stays protected even though
    CIV6AZ_DEFAULT_BRANCH already says `dev`: the list, not the default, drives
    protection, so the pre-rename window has no hole."""
    _git(repo, "checkout", "-q", "-b", protected)
    assert _run_guard(repo, "git commit -m 'x'", SETTINGS_ENV) is not None
    assert _run_guard(repo, f"git push origin {FEATURE_BRANCH}", SETTINGS_ENV) is not None


@pytest.mark.parametrize("protected", ["dev", "qa", "main"])
def test_dev_qa_main_are_protected_without_any_configuration(repo: Path, protected: str) -> None:
    """An unset CIV6AZ_PROTECTED_BRANCHES must not mean 'nothing is protected'."""
    _git(repo, "checkout", "-q", "-b", protected)
    assert _run_guard(repo, "git commit -m 'x'") is not None
    _git(repo, "checkout", "-q", FEATURE_BRANCH)
    assert _run_guard(repo, f"git push origin HEAD:{protected}") is not None


@pytest.mark.parametrize(
    "command",
    [
        "git push origin HEAD:qa",
        "git push origin feat:qa",
        "git push origin HEAD:refs/heads/qa",
        "git push origin qa",
        f"git push origin HEAD:{OLD_DEFAULT}",
        # tags, by refspec or by flag
        "git push origin HEAD:refs/tags/v9",
        "git push origin refs/tags/v9",
        "git push origin v9:refs/tags/v9",
        "git push --tags origin",
        f"git push origin {FEATURE_BRANCH} --tags",
        f"git push --follow-tags origin {FEATURE_BRANCH}",
        # deleting a protected branch
        "git push origin :dev",
        "git push origin :refs/heads/main",
        "git push origin --delete dev",
        "git push -d origin qa",
        # pushes that sweep protected refs along with everything else
        "git push --mirror origin",
        "git push --all origin",
    ],
)
def test_protected_and_tag_pushes_are_denied(repo: Path, command: str) -> None:
    assert _run_guard(repo, command, SETTINGS_ENV) is not None, command


@pytest.mark.parametrize(
    "command",
    [
        f"git push origin {FEATURE_BRANCH}",
        f"git push -u origin {FEATURE_BRANCH}",
        f"git push origin HEAD:{FEATURE_BRANCH}",
        f"git push origin --delete {FEATURE_BRANCH}",
        "git fetch --tags origin",
        "git log --tags --oneline",
        "git push origin qa-notes",
    ],
)
def test_ordinary_feature_branch_work_is_allowed(repo: Path, command: str) -> None:
    assert _run_guard(repo, command, SETTINGS_ENV) is None, command


# --- git tag -----------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git tag v1",
        "git tag -a v1 -m 'release'",
        "git tag -s v1",
        "git tag -f v1",
        "git tag -d v1",
        "git tag v1 HEAD~1",
        "git -C . tag v1",
        "git --no-pager tag v1",
    ],
)
def test_tag_creation_and_mutation_are_denied(repo: Path, command: str) -> None:
    assert _run_guard(repo, command) is not None, command


@pytest.mark.parametrize(
    "command",
    [
        "git tag",
        "git tag -l 'v2.1.0-rc.*'",
        "git tag --list",
        "git tag --list 'v*' --sort=-v:refname",
        "git tag --contains HEAD",
        "git tag --points-at HEAD",
        "git tag --merged main --format '%(refname)'",
        "git tag -n5 -l",
        "git tag -n -l v2*",
        "git tag --sort -creatordate -l",
    ],
)
def test_tag_listing_is_allowed(repo: Path, command: str) -> None:
    assert _run_guard(repo, command) is None, command


# --- gh ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "gh pr merge 1",
        "gh pr merge 1 --squash --delete-branch",
        "gh release create v9",
        "gh release delete v9",
        "gh repo edit --default-branch dev",
        "gh ruleset list",
        "gh api -X POST repos/o/r/issues",
        "gh api repos/o/r/issues -X POST",
        "gh api -XPOST repos/o/r/issues",
        "gh api --method PATCH repos/o/r",
        "gh api --method=DELETE repos/o/r/git/refs/heads/dev",
        "gh api repos/o/r/git/refs -f ref=refs/heads/x -f sha=abc",
        "gh api repos/o/r/rulesets -F enforcement=disabled",
        "gh api repos/o/r/issues --field title=x",
        "gh api repos/o/r/issues --raw-field title=x",
        "gh api repos/o/r/rulesets --input body.json",
        "gh api graphql -f query='mutation { x }'",
        "gh api graphql",
    ],
)
def test_mutating_gh_commands_are_denied(repo: Path, command: str) -> None:
    assert _run_guard(repo, command) is not None, command


@pytest.mark.parametrize(
    "command",
    [
        "gh api repos/o/r/pulls",
        "gh api repos/o/r/pulls --paginate --jq '.[].number'",
        "gh api -H 'Accept: application/vnd.github+json' repos/o/r/branches",
        "gh pr view 1",
        "gh pr list --state open",
        "gh pr checks 1",
        "gh pr create --title t --body b",
        "echo 'gh pr merge 1'",
    ],
)
def test_read_only_gh_commands_are_allowed(repo: Path, command: str) -> None:
    assert _run_guard(repo, command) is None, command


# --- unwrapping (pwsh / powershell / cmd / bash / iex / &) -------------------


@pytest.mark.parametrize(
    "command",
    [
        'pwsh -c "git push origin HEAD:qa"',
        'pwsh -NoProfile -Command "git push origin HEAD:qa"',
        "powershell.exe -NoProfile -ExecutionPolicy Bypass -Command 'git tag v1'",
        'cmd /c "git push origin HEAD:qa"',
        "cmd.exe /d /c gh pr merge 1",
        'cmd /c "git status & git push origin HEAD:qa"',
        "bash -c 'git push origin HEAD:qa'",
        'bash -lc "gh api -X POST repos/o/r/issues"',
        "sh -c 'git push --tags origin'",
        'iex "git push origin HEAD:qa"',
        "Invoke-Expression 'gh pr merge 1'",
        "& git push origin HEAD:qa",
        "& 'git' push origin HEAD:qa",
        "git status; & gh pr merge 1",
        "bash -c \"pwsh -c 'git push origin HEAD:qa'\"",
    ],
)
def test_wrapped_commands_are_unwrapped_before_matching(repo: Path, command: str) -> None:
    assert _run_guard(repo, command, SETTINGS_ENV) is not None, command


def test_powershell_tool_payload_is_guarded(repo: Path) -> None:
    """The PowerShell tool sends its command in ``tool_input.command`` too."""
    decision, _ = _decide(repo, "git push origin HEAD:qa", SETTINGS_ENV, tool_name="PowerShell")
    assert decision == "deny"
    decision, _ = _decide(repo, "Get-ChildItem", SETTINGS_ENV, tool_name="PowerShell")
    assert decision is None


def test_wrapped_harmless_command_is_allowed(repo: Path) -> None:
    assert _run_guard(repo, 'pwsh -c "git status"', SETTINGS_ENV) is None
    assert _run_guard(repo, "bash -c 'ls -la'", SETTINGS_ENV) is None


# --- game cap: deny without approval_ref, ask with it ------------------------


def test_over_cap_without_approval_ref_is_denied(repo: Path) -> None:
    reason = _run_guard(repo, "python -m alpha_zero.training_civ6 --num_games 5000")
    assert reason is not None and "1000" in reason


@pytest.mark.parametrize(
    "command",
    [
        "GAMES_PER_WORKER=2000 make cloud-train",
        "NUM_GAMES=5000 python -m alpha_zero.training_civ6",
        "bash -c 'NUM_GAMES=5000 make train'",
    ],
)
def test_env_var_game_count_over_cap_is_denied(repo: Path, command: str) -> None:
    """The env-var form (Makefile / run_cloud_training.sh) must survive the
    leading-assignment stripping the unwrapper does."""
    reason = _run_guard(repo, command)
    assert reason is not None and "1000" in reason, command


@pytest.mark.parametrize(
    "command",
    [
        "python -m alpha_zero.training_civ6 --num_games 5000 --approval_ref 123",
        "python -m alpha_zero.training_civ6 --num_games 10 --approval_ref=123",
        "python scripts/run_baseline_eval.py --num_games 2000 --approval_ref #41",
    ],
)
def test_approval_ref_asks_instead_of_denying(repo: Path, command: str) -> None:
    """The ruling: --approval_ref is recorded, not verified, so a human confirms."""
    decision, reason = _decide(repo, command)
    assert decision == "ask", (command, reason)
    assert reason and "approval_ref" in reason


@pytest.mark.parametrize(
    "command",
    [
        "python -m alpha_zero.orchestrator --preset production",
        "python -m alpha_zero.orchestrator --preset=production --dry_run",
        "python -m alpha_zero.training_civ6 --config configs/civ6_config.yaml",
        "python scripts/run_sweep.py --config configs/sweep_muzero.yaml",
        # unknown runners: shaped like a game runner, not on the priced list
        "python scripts/run_mystery_selfplay.py --games 10",
        "python scripts/benchmark_mystery.py --n_games 10",
        "python -m alpha_zero.cloud.gcp.selfplay_worker --num_games 4",
        'pwsh -c "python scripts/run_mystery.py"',
    ],
)
def test_unpriceable_runs_ask(repo: Path, command: str) -> None:
    decision, reason = _decide(repo, command)
    assert decision == "ask", (command, reason)


@pytest.mark.parametrize(
    "command",
    [
        "python -m alpha_zero.training_civ6 --dry_run",
        "python -m alpha_zero.orchestrator --preset fast --dry_run",
        "python -m pytest tests/governance -q",
        "python scripts/validate_openspec.py",
        "python scripts/check_doc_sync.py",
        "python scripts/run_llm_eval.py",
        "python scripts/run_mystery_selfplay.py --help",
        "OMP_NUM_THREADS=2 python -m pytest tests/unit -q",
        "python -m ruff check --config pyproject.toml .",
        # priced through alpha_zero.evaluation.benchmark (fix round 1)
        "python scripts/benchmark_suite.py --num_games 10",
    ],
)
def test_priced_or_non_runner_commands_are_allowed(repo: Path, command: str) -> None:
    assert _decide(repo, command) == (None, None), command


def test_deny_beats_ask_across_segments(repo: Path) -> None:
    decision, _ = _decide(
        repo,
        "python -m alpha_zero.training_civ6 --approval_ref 1 && git push origin HEAD:qa",
        SETTINGS_ENV,
    )
    assert decision == "deny"


# --- payload parsing fails closed --------------------------------------------


def test_unparseable_payload_is_denied(repo: Path) -> None:
    """A payload no parser can read must not fall through to allow."""
    decision, reason = _raw_guard(repo, "{not json")
    assert decision == "deny", reason


def test_non_command_payload_is_allowed(repo: Path) -> None:
    """A tool input with no command (not a shell tool) is not the guard's business."""
    decision, _ = _raw_guard(repo, json.dumps({"tool_name": "Read", "tool_input": {}}))
    assert decision is None


# --- fix round 1: bypasses found by review ------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # embedded quotes and escapes inside a refspec
        'git push origin HEAD:"qa"',
        "git push origin HEAD:'qa'",
        'git push origin "HEAD:qa"',
        "git push origin HEAD:q\\a",
        "git push origin HEAD:refs//heads/qa",
        # program basenames are case-insensitive on Windows
        "GIT push origin HEAD:qa",
        "Git push origin HEAD:qa",
        "git.EXE push origin HEAD:qa",
        "GH pr merge 1",
        "Gh.exe release create v9",
    ],
)
def test_quoting_and_case_do_not_bypass_the_push_and_gh_rules(repo: Path, command: str) -> None:
    assert _run_guard(repo, command, SETTINGS_ENV) is not None, command


@pytest.mark.parametrize(
    "command",
    [
        "g`it push origin HEAD:qa",
        "git push origin HEAD:`qa",
        "git push origin HEAD:q`a",
        "& 'C:\\Program Files\\Git\\cmd\\git.exe' push origin HEAD:qa",
        '& "C:\\Program Files\\Git\\cmd\\git.exe" push origin HEAD:qa',
        '& "git" push origin HEAD:qa',
    ],
)
def test_powershell_escapes_and_quoted_program_paths_are_denied(repo: Path, command: str) -> None:
    decision, reason = _decide(repo, command, SETTINGS_ENV, tool_name="PowerShell")
    assert decision == "deny", (command, reason)


@pytest.mark.parametrize(
    "command",
    [
        "env git push origin HEAD:qa",
        "env -i FOO=1 git push origin HEAD:qa",
        "command git push origin HEAD:qa",
        "sudo git push origin HEAD:qa",
        "nohup gh pr merge 1",
        "git push origin tag v9",
        "python -m alpha_zero.training_civ6 --eval_games 5000",
    ],
)
def test_launcher_prefixes_tag_keyword_and_eval_games_are_covered(repo: Path, command: str) -> None:
    assert _run_guard(repo, command, SETTINGS_ENV) is not None, command


def test_bash_command_substitution_is_still_split(repo: Path) -> None:
    """Deleting backticks for PowerShell must not stop bash `...` being inspected."""
    assert _run_guard(repo, "echo `git push origin HEAD:qa`", SETTINGS_ENV) is not None


@pytest.mark.parametrize(
    "command",
    [
        "python -m alpha_zero.training_civ6 --num_games 99999999999999999999",
        "python -m alpha_zero.training_civ6 --num_games 0002000",
        "python -m alpha_zero.training_civ6 --num_games=1001",
        "python -m alpha_zero.orchestrator --games-per-iteration 5000",
        "NUM_GAMES=99999999999999999999 make train",
    ],
)
def test_oversized_or_padded_game_counts_are_denied(repo: Path, command: str) -> None:
    """An integer `[ -gt ]` cannot parse used to error inside the `if` -- and
    read as false, so the run was allowed."""
    reason = _run_guard(repo, command)
    assert reason is not None and "1000" in reason, command


def test_oversized_game_count_with_approval_ref_asks(repo: Path) -> None:
    decision, _ = _decide(
        repo, "python -m alpha_zero.training_civ6 --num_games 99999999999999999999 --approval_ref 7"
    )
    assert decision == "ask"


def test_padded_small_game_count_is_allowed(repo: Path) -> None:
    assert _run_guard(repo, "python -m alpha_zero.training_civ6 --num_games 0000000000010") is None


def test_control_characters_in_a_reason_still_produce_valid_json(repo: Path) -> None:
    """`git tag v1\\x01` put a raw control character into the reason; the
    output was invalid JSON, which Claude Code ignores -- a fail-open."""
    decision, reason = _decide(repo, "git tag v1\x01\x1b[31m")
    assert decision == "deny"
    assert reason is not None and all(ord(ch) >= 0x20 for ch in reason)


def test_an_unexpected_guard_failure_denies(repo: Path) -> None:
    """Any crash inside the guard must surface as a deny decision (exit 0 +
    JSON), never as a non-zero exit, which Claude Code treats as non-blocking."""
    decision, reason = _decide(repo, "git status", {"CIV6AZ_GUARD_SELFTEST_FAIL": "1"})
    assert decision == "deny"
    assert reason is not None and "failed unexpectedly" in reason
