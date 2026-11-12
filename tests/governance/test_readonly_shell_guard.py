"""Behavioural tests for the two subagent-scoped hooks (LF-G13, T15).

* ``.claude/hooks/readonly_shell_guard.sh`` -- an ALLOW list over the shell for the
  read-only SDLC roles. Driven the way the hook runner does: ``bash <script>`` with a
  PreToolUse JSON payload on stdin. No output means allow; a JSON decision means deny.
* ``.claude/hooks/tests_only_write_guard.sh`` -- confines ``sdlc-qa-sdet`` to ``tests/**``.

Negative control: ``CIV6AZ_TEST_READONLY_GUARD`` / ``CIV6AZ_TEST_WRITE_GUARD`` repoint the
tests at a stand-in script. Pointing them at a script that always exits 0 (allow
everything) must make every deny test below fail; the T15 report records the run.

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
_HOOKS = REPO_ROOT / ".claude" / "hooks"
READONLY_GUARD = Path(
    os.environ.get("CIV6AZ_TEST_READONLY_GUARD", str(_HOOKS / "readonly_shell_guard.sh"))
)
WRITE_GUARD = Path(
    os.environ.get("CIV6AZ_TEST_WRITE_GUARD", str(_HOOKS / "tests_only_write_guard.sh"))
)
# Through PATH, not the OS: on Windows CreateProcess finds WSL's bash.exe first.
BASH = shutil.which("bash") or "bash"
_HAS_PARSER = any(shutil.which(tool) for tool in ("jq", "python3", "python"))

pytestmark = pytest.mark.skipif(not _HAS_PARSER, reason="guards need jq or python on PATH")


def _run(
    script: Path,
    payload: str,
    env_extra: dict[str, str] | None = None,
    path: str | None = None,
) -> tuple[str | None, str]:
    env = dict(os.environ)
    env.update(env_extra or {})
    if path is not None:
        env["PATH"] = path
    proc = subprocess.run(
        [BASH, script.as_posix()],
        cwd=REPO_ROOT,
        input=payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    if not proc.stdout.strip():
        return None, ""
    decision = json.loads(proc.stdout)["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    return str(decision["permissionDecision"]), str(decision["permissionDecisionReason"])


def shell(command: str, tool: str = "Bash") -> tuple[str | None, str]:
    payload = json.dumps({"tool_name": tool, "tool_input": {"command": command}})
    return _run(READONLY_GUARD, payload)


def write(file_path: str, env_extra: dict[str, str] | None = None) -> tuple[str | None, str]:
    payload = json.dumps({"tool_name": "Edit", "tool_input": {"file_path": file_path}})
    return _run(WRITE_GUARD, payload, env_extra)


ALLOWED = [
    "git log --oneline -5",
    "git log --pretty=oneline --since=2.days -n 3 -- scripts",
    "git show HEAD --stat",
    "git show HEAD:README.md",
    "git diff --stat dev...HEAD",
    "git diff --cached --name-only",
    "git status --porcelain",
    "git status -sb",
    "git rev-parse --abbrev-ref HEAD",
    "git merge-base --is-ancestor HEAD dev",
    "git ls-files -o --exclude-standard",
    "git cat-file -p HEAD",
    "git tag -l v2.0.0",
    "git tag --list --sort=-v:refname",
    "gh pr view 12 --json state",
    "gh pr list --state open --limit 5",
    "gh pr checks 12",
    "gh api repos/o/r/pulls/1",
    "gh api repos/o/r/pulls --paginate --jq .number",
    "gh api repos/o/r -H X-GitHub-Api-Version:2022-11-28",
    "python scripts/validate_claude_agents.py",
    "python scripts/validate_kb.py --git",
    "python3 scripts/check_doc_sync.py --quiet",
    "python scripts/verify_release_evidence.py --scope",
    "GIT LOG --oneline",
    "git.exe status",
]

DENIED = [
    # --- not on the list at all
    "echo hello",
    "ls",
    "cat README.md",
    "rm -rf build",
    "git commit -m x",
    "git push origin HEAD",
    "git checkout -b x",
    "git reset --hard",
    "git branch -D x",
    "git config user.name x",
    "git tag v1",
    "git tag -d v1",
    "git tag -l -a v1",
    "gh pr merge 1",
    "gh pr create",
    "gh pr comment 1 --body x",
    "gh release create v1",
    "python -c print(1)",
    "python -m pytest",
    "python scripts/run_baseline_eval.py",
    "python scripts/check_build.py",
    "python scripts/validate_gcp_credentials.py",
    "python ./scripts/check_doc_sync.py",
    "python scripts/../scripts/check_doc_sync.py",
    "python scripts/sub/check_doc_sync.py",
    # --- wrappers and launchers
    "bash -c git log",
    "sh -c ls",
    "pwsh -c git log",
    "powershell -Command git log",
    "cmd /c git log",
    "env git log",
    "sudo git log",
    "FOO=bar git log",
    "xargs git log",
    # --- chaining, pipes, substitution, redirection (the smuggling cases)
    "git log; rm -rf x",
    "git log && rm -rf x",
    "git log || rm -rf x",
    "git log | tee out.txt",
    "git log & rm x",
    "git log $(rm x)",
    "git log `rm x`",
    "git log ${HOME}",
    "git log > out.txt",
    "git log >> out.txt",
    "git log 2> err.txt",
    "git log < in.txt",
    "git log\nrm x",
    "(git log)",
    "{ git log; }",
    "git status;git push",
    # --- flags that write or run a configured program
    "git log --output=x",
    "git diff --output x",
    "git show --output=x HEAD",
    "git diff --ext-diff",
    "git diff --textconv",
    "git cat-file --textconv HEAD:f",
    "git cat-file --filters HEAD:f",
    "git log -c core.pager=x",
    "git -c core.pager=x log",
    "git --git-dir=x log",
    "git diff --no-such-flag",
    # --- quoting and escaping must not hide a flag from the guard
    'git diff "--output=/tmp/x"',
    "git diff '--output=/tmp/x'",
    "git diff --out\\put=x",
    'git diff --out"put=x"',
    "git diff --out'put'=x",
    "git diff --o\\utput=x",
    'git "diff" --stat',
    "git log --format='%h %s'",
    # --- PowerShell argument smuggling (a comma array is two arguments to git)
    "git diff x,--output=/tmp/f",
    "git log HEAD,--output=/tmp/f",
    "git log -- --stat,--output=/tmp/f",
    "gh api repos/o/r,-X,POST",
    "git log --% --output=/tmp/f",
    "git diff --%",
    "git diff @args",
    "git diff @{u}",
    "git log # ; rm x",
    "git log --pretty=%h",
    # --- globs in a positional argument could expand to a flag-like file name
    "git diff *",
    "git diff -- *.py",
    "git log ?",
    "git ls-files [a-z]*",
    "git tag -l v2*",
    "git show HEAD:docs/*",
    "gh api repos/o/*",
    "gh api repos/o/r?per_page=100",
    "python scripts/check_doc_sync.py *",
    "gh api repos/o/r --jq .[].number",
    # --- gh api: GET only, path + a short flag list
    "gh api -X POST repos/o/r/issues",
    "gh api --method POST repos/o/r/issues",
    "gh api --method=PATCH repos/o/r",
    "gh api -XDELETE repos/o/r",
    "gh api -X GET repos/o/r",
    "gh api -X 'POST' repos/o/r",
    'gh api -X "POST" repos/o/r',
    "gh api repos/o/r -f title=x",
    "gh api repos/o/r -F title=x",
    "gh api repos/o/r --field a=b",
    "gh api repos/o/r --raw-field a=b",
    "gh api repos/o/r --input body.json",
    "gh api graphql",
    "gh api /graphql",
    "gh api repos/o/r -H Accept:application/vnd.github+json",
    "gh api repos/o/r -H",
    "gh api repos/o/r --hostname example.com",
    "gh api repos/o/r --cache 1h",
    # --- gh pr flags
    "gh pr view 1 --web",
    "gh pr view 1 -w",
    "gh pr list --no-such-flag",
    # --- python checkers must not be made to write (argparse accepts abbreviations)
    "python scripts/check_learning_trend.py --out report.md",
    "python scripts/check_learning_trend.py --ou report.md",
    "python scripts/check_learning_trend.py --out=report.md",
    "python scripts/check_doc_sync.py --fix",
    "python scripts/check_doc_sync.py --wr x",
]


@pytest.mark.parametrize("command", ALLOWED)
def test_read_only_commands_are_allowed(command: str) -> None:
    decision, reason = shell(command)
    assert decision is None, f"{command!r} was denied: {reason}"


@pytest.mark.parametrize("command", DENIED)
def test_everything_else_is_denied(command: str) -> None:
    decision, _ = shell(command)
    assert decision == "deny", f"{command!r} was allowed"


def test_powershell_tool_is_judged_identically() -> None:
    assert shell("git log --oneline", tool="PowerShell")[0] is None
    assert shell("git log; Remove-Item x", tool="PowerShell")[0] == "deny"
    assert shell("& git log", tool="PowerShell")[0] == "deny"
    assert shell("git log `n Remove-Item x", tool="PowerShell")[0] == "deny"


def test_deny_reason_tells_the_role_what_to_do() -> None:
    _, reason = shell("rm -rf x")
    assert "read-only" in reason and "Return the text" in reason


def test_empty_command_runs_nothing_and_is_allowed() -> None:
    assert shell("")[0] is None
    assert shell("   ")[0] is None


@pytest.mark.parametrize("payload", ["", "not json"])
def test_unparsable_payload_fails_closed(payload: str) -> None:
    decision, _ = _run(READONLY_GUARD, payload)
    assert decision == "deny"


@pytest.mark.parametrize("payload", ["[]", '{"tool_input": 3}'])
def test_payload_without_a_string_command_is_denied(payload: str) -> None:
    decision, _ = _run(READONLY_GUARD, payload)
    assert decision == "deny"


def test_denies_when_no_parser_is_available(tmp_path: Path) -> None:
    """With neither jq nor python on PATH the guard cannot see the command: deny."""
    empty = tmp_path / "empty"
    empty.mkdir()
    payload = json.dumps({"tool_input": {"command": "git push --force"}})
    decision, reason = _run(READONLY_GUARD, payload, path=empty.as_posix())
    assert decision == "deny", reason


def test_internal_failure_becomes_a_deny() -> None:
    payload = json.dumps({"tool_input": {"command": "git log"}})
    decision, reason = _run(READONLY_GUARD, payload, {"CIV6AZ_GUARD_SELFTEST_FAIL": "1"})
    assert decision == "deny" and "unexpectedly" in reason


# --- tests-only write guard --------------------------------------------------


@pytest.mark.parametrize(
    "file_path",
    [
        "tests/unit/test_x.py",
        "./tests/unit/test_x.py",
        "tests\\unit\\test_x.py",
        (REPO_ROOT / "tests" / "unit" / "test_x.py").as_posix(),
    ],
)
def test_qa_may_edit_under_tests(file_path: str) -> None:
    decision, reason = write(file_path, {"CLAUDE_PROJECT_DIR": REPO_ROOT.as_posix()})
    assert decision is None, reason


@pytest.mark.parametrize(
    "file_path",
    [
        "alpha_zero/core/civ6_network.py",
        "scripts/validate_kb.py",
        ".claude/settings.json",
        "CLAUDE.md",
        "tests/../scripts/x.py",
        "tests/../../etc/passwd",
        "../tests/x.py",
        "mytests/x.py",
        "tests_extra/x.py",
        "/tmp/tests/x.py",
        "",
        (REPO_ROOT / "alpha_zero" / "x.py").as_posix(),
    ],
)
def test_qa_may_not_edit_outside_tests(file_path: str) -> None:
    decision, _ = write(file_path, {"CLAUDE_PROJECT_DIR": REPO_ROOT.as_posix()})
    assert decision == "deny", f"{file_path!r} was allowed"


def test_write_guard_fails_closed() -> None:
    decision, _ = _run(WRITE_GUARD, "garbage")
    assert decision == "deny"
    decision, reason = _run(
        WRITE_GUARD,
        json.dumps({"tool_input": {"file_path": "tests/x.py"}}),
        {"CIV6AZ_GUARD_SELFTEST_FAIL": "1"},
    )
    assert decision == "deny" and "unexpectedly" in reason
