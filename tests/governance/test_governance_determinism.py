"""The governance validators must be byte-for-byte reproducible.

Why this gate exists
--------------------
A validator whose output order depends on filesystem iteration order produces a
CI failure that cannot be reproduced locally: the same tree yields a different
report, so the engineer who tries to confirm the finding sees a different set of
lines (or a different count) and concludes the gate is flaky rather than that
the tree is wrong. A flaky gate is ignored, and an ignored gate is not a gate.

``validate_agents.py`` grew a recursive ``rglob`` scan in this change. ``rglob``
returns entries in filesystem order, so the sort applied to it is load-bearing
rather than cosmetic, and this suite is what holds it in place.

What is asserted
----------------
Each validator is run repeatedly in a fresh subprocess -- a subprocess, not an
in-process call, so process-level nondeterminism (hash randomisation, module
import order) is inside the measurement rather than outside it. stdout, stderr
and the exit code must be identical across every run, and across several values
of ``PYTHONHASHSEED``, which is the knob that exposes ``set``/``dict`` iteration
order.

Negative control (recorded per the ``negative-control`` skill)
--------------------------------------------------------------
``test_negative_control_detects_a_nondeterministic_validator`` runs the same
comparison helper against a deliberately nondeterministic script that shuffles
its own output lines. It fails the comparison, which is what proves the helper
above can fail at all. Verified by running that script through
``_run_repeatedly``: the shuffling script is reported as non-identical, while
both real validators are reported identical.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"

# Seeds chosen to span the space rather than to be pretty: 0 disables
# randomisation, the rest are arbitrary distinct values.
HASH_SEEDS = ("0", "1", "42", "12345", "99999")

# All six enterprise governance validators, with the offline/deterministic flags
# for repeated and cross-HASH_SEED verification (text and JSON modes).
VALIDATOR_INVOCATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("validate_skills.py", ()),
    ("validate_skills.py", ("--json",)),
    ("validate_skills.py", ("--verbose",)),
    ("validate_agents.py", ()),
    ("validate_agents.py", ("--json",)),
    ("validate_agents.py", ("--verbose",)),
    ("validate_claude_agents.py", ()),
    ("validate_claude_agents.py", ("--json",)),
    ("validate_claude_agents.py", ("--verbose",)),
    ("validate_kb.py", ("--no-git",)),
    ("validate_kb.py", ("--no-git", "--json")),
    ("validate_kb.py", ("--no-git", "--verbose")),
    ("validate_openspec.py", ()),
    ("validate_openspec.py", ("--json",)),
    ("validate_openspec.py", ("--verbose",)),
    ("check_doc_sync.py", ()),
    ("check_doc_sync.py", ("--json",)),
    ("check_doc_sync.py", ("--verbose",)),
)


def _deterministic_env(hash_seed: str) -> dict[str, str]:
    """Minimal cross-platform env isolating PYTHONHASHSEED while keeping git/Windows DLLs working."""
    path_parts = [os.path.dirname(sys.executable)]
    git_bin = shutil.which("git")
    if git_bin:
        path_parts.append(os.path.dirname(git_bin))
    path_parts.extend(["/usr/bin", "/bin", "/usr/local/bin"])
    env = {"PATH": os.pathsep.join(path_parts), "PYTHONHASHSEED": hash_seed}
    for key in ("SystemRoot", "SYSTEMROOT"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _run(script: Path, *args: str, hash_seed: str = "0") -> tuple[int, str, str]:
    """Run one validator in a fresh interpreter and capture everything it emits."""
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=_deterministic_env(hash_seed),
        timeout=120,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _run_repeatedly(script: Path, *args: str, runs: int = 3) -> list[tuple[int, str, str]]:
    return [_run(script, *args) for _ in range(runs)]


def _all_identical(results: list[tuple[int, str, str]]) -> bool:
    return all(r == results[0] for r in results)


@pytest.mark.parametrize(
    "script_name", ("validate_skills.py", "validate_agents.py", "validate_claude_agents.py")
)
def test_skill_and_agent_validators_are_deterministic(script_name: str) -> None:
    """Five fresh-process runs of each skill/agent validator must agree byte for byte."""
    results = _run_repeatedly(SCRIPTS / script_name, runs=5)
    assert _all_identical(results), f"{script_name} output varied across runs: {results}"


@pytest.mark.parametrize(
    ("script_name", "extra_args"),
    VALIDATOR_INVOCATIONS,
    ids=[f"{s}[{' '.join(a) or 'default'}]" for s, a in VALIDATOR_INVOCATIONS],
)
def test_validators_are_deterministic_across_hash_seeds(
    script_name: str, extra_args: tuple[str, ...]
) -> None:
    """Set/dict iteration order must not reach text or JSON validator reports."""
    script = SCRIPTS / script_name
    results = [_run(script, *extra_args, hash_seed=seed) for seed in HASH_SEEDS]
    assert _all_identical(results), (
        f"{script_name} {' '.join(extra_args)} output depends on PYTHONHASHSEED -- "
        f"something iterates a set or dict without sorting: {results}"
    )
    if "--json" in extra_args:
        parsed = json.loads(results[0][1])
        assert isinstance(parsed, (list, dict))
        assert json.dumps(parsed, indent=2, sort_keys=True) == results[0][1].strip(), (
            f"{script_name} --json output is not canonically sorted"
        )
    if "--verbose" in extra_args:
        assert results[0][0] == 0 and bool(results[0][2].strip()), (
            f"{script_name} --verbose failed or emitted empty stderr: {results[0]}"
        )


def test_negative_control_detects_a_nondeterministic_validator(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: the comparison above must be able to fail.

    A validator that emits its findings in shuffled order is exactly the defect
    this suite guards against. Running it through the same ``_run_repeatedly`` /
    ``_all_identical`` path must report a difference -- if it did not, the two
    passing tests above would be vacuous.
    """
    shuffler = tmp_path / "nondeterministic_validator.py"
    shuffler.write_text(
        textwrap.dedent(
            """\
            import random
            lines = [f"ERROR file{i}.md: X{i} something is wrong" for i in range(40)]
            random.shuffle(lines)
            print("\\n".join(lines))
            """
        ),
        encoding="utf-8",
    )

    results = _run_repeatedly(shuffler, runs=5)
    assert not _all_identical(results), (
        "the shuffling control script produced identical output across 5 runs, "
        "so _all_identical cannot detect nondeterminism and the tests above prove nothing"
    )
