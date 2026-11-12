"""AQA: the governance validators must behave alike at their CLI boundary.

``scripts/validate_agents.py --path <missing>`` exited **0**. The PostToolUse
hook invokes validators through ``--path``, so a stale or mistyped path read as
a clean validation -- a gate reporting success for work it never looked at. Its
three siblings got that right, which is the point: the defect was not a hard
problem, it was an unenforced convention.

So rather than pinning the one validator that had the hole, this file asserts
the convention across every validator that accepts ``--path``. The next one
added inherits the check.

Fixing A0 then surfaced a second defect in the fix itself: the renderer called
``Path.relative_to(REPO_ROOT)`` unguarded, so reporting a path *outside* the
repository -- precisely the ``--path /nonexistent/...`` case -- died with a
``ValueError`` traceback instead of printing its own diagnostic. The exit code
was still non-zero, so a coarser test would have called that a pass.

These run as subprocesses. That is deliberate: the hook and CI invoke these as
programs, and importing ``main()`` would not exercise ``SystemExit`` handling,
argument parsing, or the stderr rendering where that traceback lived.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
MAKEFILE = REPO_ROOT / "Makefile"

# Validators exposing `--path` for single-file validation, which is how the
# PostToolUse hook calls them.
PATH_VALIDATORS = (
    "validate_agents.py",
    "validate_claude_agents.py",
    "validate_kb.py",
    "validate_openspec.py",
    "validate_skills.py",
)

# Every deterministic validator, `--path` or not. All are expected to run
# whole-repo with no arguments and exit 0 on a healthy tree.
ALL_VALIDATORS = (*PATH_VALIDATORS, "check_doc_sync.py")

# `ERROR <where>: <CODE> <message>` -- the shape the hook and CI logs are read
# through. Codes are per-validator (A0, K1, OS001, S1), so match the shape.
_DIAGNOSTIC = re.compile(r"^(ERROR|WARNING) \S+: [A-Z]+\d+ ", re.M)


@pytest.fixture(scope="module")
def makefile() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


class TestMissingPathIsAnError:
    """The F8 defect, generalised to every validator that takes ``--path``."""

    @pytest.mark.parametrize("script", PATH_VALIDATORS)
    def test_nonexistent_path_exits_nonzero(self, script: str) -> None:
        result = _run(script, "--path", "does/not/exist/anywhere.md")
        assert result.returncode != 0, (
            f"{script} --path <missing> exited 0. The PostToolUse hook calls "
            "validators this way, so a stale path would read as a clean "
            "validation of a file that was never opened."
        )

    @pytest.mark.parametrize("script", PATH_VALIDATORS)
    def test_nonexistent_path_reports_a_diagnostic_not_a_traceback(self, script: str) -> None:
        """Non-zero is necessary but not sufficient -- a crash is also non-zero.

        ``validate_agents.py`` exited 1 by way of an unhandled ``ValueError``
        from ``Path.relative_to``. Only checking the code would have passed.
        """
        result = _run(script, "--path", "does/not/exist/anywhere.md")
        combined = result.stdout + result.stderr
        assert "Traceback (most recent call last)" not in combined, (
            f"{script} crashed rather than reporting the bad path:\n{combined}"
        )
        assert _DIAGNOSTIC.search(combined), (
            f"{script} gave no `ERROR <path>: <CODE> <message>` diagnostic for a "
            f"missing --path; output was:\n{combined}"
        )

    @pytest.mark.parametrize("script", PATH_VALIDATORS)
    def test_path_outside_the_repository_is_handled(self, script: str) -> None:
        """The exact input that produced the traceback: an absolute outside path."""
        result = _run(script, "--path", "/nonexistent/definitely-missing")
        combined = result.stdout + result.stderr
        assert "Traceback (most recent call last)" not in combined, (
            f"{script} crashed on an absolute path outside the repo:\n{combined}"
        )
        assert result.returncode != 0


class TestMisnamedPathIsAnError:
    """The half of the ``--path`` contract that the first fix missed.

    ``validate_agents.py`` was fixed to reject a *missing* target, and
    ``validate_skills.py`` was left with an ``if/elif`` and no ``else``: an
    existing file that is simply not a ``SKILL.md`` recorded no issue and exited
    0. The hook passes whatever file was just edited, so a misnamed skill
    manifest read as validated without ever being opened -- the same defect as
    the missing-path one, reached by a different route.

    Each validator here owns a filename; hand it a real file with the wrong name
    and it must object.
    """

    # validator -> a tracked file that exists but is not what it validates
    WRONG_FILE = {
        "validate_agents.py": "README.md",
        "validate_claude_agents.py": "README.md",
        "validate_skills.py": "README.md",
    }

    @pytest.mark.parametrize("script", sorted(WRONG_FILE))
    def test_existing_file_with_the_wrong_name_is_rejected(self, script: str) -> None:
        result = _run(script, "--path", self.WRONG_FILE[script])
        combined = result.stdout + result.stderr
        assert result.returncode != 0, (
            f"{script} --path {self.WRONG_FILE[script]} exited 0. The file exists but is "
            "not what this validator validates, so passing means reporting a clean "
            "validation of something never inspected."
        )
        assert _DIAGNOSTIC.search(combined), (
            f"{script} gave no structured diagnostic for a misnamed target:\n{combined}"
        )

    @pytest.mark.parametrize("script", sorted(WRONG_FILE))
    def test_the_right_file_still_validates(self, script: str) -> None:
        """The positive half: a validator that rejected everything would also
        satisfy the assertions above."""
        targets = {
            "validate_agents.py": "tests/Agent.md",
            "validate_claude_agents.py": ".claude/agents/sdlc-architect.md",
            "validate_skills.py": ".claude/skills/verify/SKILL.md",
        }
        result = _run(script, "--path", targets[script])
        assert result.returncode == 0, (
            f"{script} rejects its own valid target:\n{result.stdout}\n{result.stderr}"
        )


class TestValidatorsShareOneContract:
    """The documented exit-code contract (CLAUDE.md 5.2: exit 0/1/2)."""

    @pytest.mark.parametrize("script", ALL_VALIDATORS)
    def test_clean_tree_exits_zero(self, script: str) -> None:
        result = _run(script)
        assert result.returncode == 0, (
            f"{script} fails on the current tree:\n{result.stdout}\n{result.stderr}"
        )

    @pytest.mark.parametrize("script", ALL_VALIDATORS)
    def test_help_is_available_and_exits_zero(self, script: str) -> None:
        result = _run(script, "--help")
        assert result.returncode == 0, f"{script} --help exited {result.returncode}"
        assert result.stdout.strip(), f"{script} --help printed nothing"

    def test_every_path_validator_is_listed_here(self) -> None:
        """Keeps PATH_VALIDATORS honest as validators are added.

        Without this, a new validator with the same hole simply would not be
        covered -- and the file would still be green, which is the failure mode
        this whole suite exists to prevent.
        """
        discovered = {
            script.name
            for script in SCRIPTS.glob("validate_*.py")
            if re.search(r'["\']--path["\']', script.read_text(encoding="utf-8"))
        }
        missing = sorted(discovered - set(PATH_VALIDATORS))
        assert not missing, (
            f"validators accept --path but are untested here: {missing}. Add them "
            "to PATH_VALIDATORS."
        )


class TestSecretScanningToolchainIsPinned:
    """Regression cover for two findings that shipped without any.

    F7: ``GITLEAKS_BIN`` defaulted to ``$(shell command -v gitleaks)``, so a dev
    machine with any gitleaks on PATH silently ran a different ruleset from CI --
    defeating the version pin declared two lines above it.

    S1: the installer hardcoded ``linux_x64``, against CLAUDE.md 2.1's
    cross-platform requirement, so ``make verify`` on macOS or ARM installed a
    binary that cannot run.
    """

    def test_binary_defaults_to_the_managed_pinned_copy(self, makefile: str) -> None:
        match = re.search(r"^GITLEAKS_BIN \?=(.*)$", makefile, re.M)
        assert match, "GITLEAKS_BIN must stay an overridable Makefile variable"
        default = match.group(1).strip()
        assert ".tools/gitleaks" in default, (
            f"GITLEAKS_BIN defaults to {default!r}. Resolving from PATH means a "
            "local run can use a different gitleaks version than CI, which makes "
            "the GITLEAKS_VERSION pin decorative."
        )
        assert "command -v" not in default, (
            "GITLEAKS_BIN resolves from PATH again; the pin is only real if the "
            "managed binary is the default"
        )
        assert "$(GITLEAKS_VERSION)" in default, (
            f"GITLEAKS_BIN is {default!r}, with no version in the path. The "
            "installer only fetches when the target is missing, so a binary left "
            "over from an earlier pin is reused for ever -- the pin defeated one "
            "step further along than the PATH default. Stamping the version into "
            "the name makes a pin change a cache miss."
        )

    @pytest.mark.parametrize(
        ("os_name", "arch"),
        [
            ("Linux", "x86_64"),
            ("Linux", "aarch64"),
            ("Darwin", "x86_64"),
            ("Darwin", "arm64"),
        ],
    )
    def test_installer_maps_every_supported_platform(
        self, makefile: str, os_name: str, arch: str
    ) -> None:
        """The uname mapping must name each platform it claims to support.

        A string check, not a download: the artifact names were verified live
        against the pinned release once (HTTP 200 each), and a gate that reaches
        the network is a gate that fails on a bad day for reasons of its own.
        """
        install = re.search(r"^gitleaks-install:.*?\n((?:\t.*\n)+)", makefile, re.M)
        assert install, "gitleaks-install target not found"
        recipe = install.group(1)
        assert os_name in recipe, (
            f"the installer's uname mapping does not mention {os_name}; on that "
            "platform `make verify` installs an unusable binary or nothing"
        )
        assert arch in recipe, f"the installer's uname mapping does not mention {arch}"

    def test_unsupported_platform_fails_loudly(self, makefile: str) -> None:
        install = re.search(r"^gitleaks-install:.*?\n((?:\t.*\n)+)", makefile, re.M)
        assert install
        recipe = install.group(1)
        assert "unsupported" in recipe.lower(), (
            "an unrecognised platform must fail with a message naming GITLEAKS_BIN "
            "as the escape hatch, never silently install something unrunnable"
        )
        assert "GITLEAKS_BIN" in recipe, (
            "the failure path must name the override that lets a user proceed"
        )
