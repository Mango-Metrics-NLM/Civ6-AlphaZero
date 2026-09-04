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

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"

# Seeds chosen to span the space rather than to be pretty: 0 disables
# randomisation, the rest are arbitrary distinct values.
HASH_SEEDS = ("0", "1", "42", "12345", "99999")


def _run(script: Path, hash_seed: str = "0") -> tuple[int, str, str]:
    """Run one validator in a fresh interpreter and capture everything it emits."""
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "PYTHONHASHSEED": hash_seed},
        timeout=120,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _run_repeatedly(script: Path, runs: int = 3) -> list[tuple[int, str, str]]:
    return [_run(script) for _ in range(runs)]


def _all_identical(results: list[tuple[int, str, str]]) -> bool:
    return all(r == results[0] for r in results)


def test_validate_skills_is_deterministic() -> None:
    """Five in-repo runs of validate_skills.py must agree byte for byte."""
    results = _run_repeatedly(SCRIPTS / "validate_skills.py", runs=5)
    assert _all_identical(results), f"validate_skills.py output varied across runs: {results}"


def test_validate_agents_is_deterministic() -> None:
    """Five in-repo runs of validate_agents.py must agree byte for byte."""
    results = _run_repeatedly(SCRIPTS / "validate_agents.py", runs=5)
    assert _all_identical(results), f"validate_agents.py output varied across runs: {results}"


def test_validators_are_deterministic_across_hash_seeds() -> None:
    """Set/dict iteration order must not reach the report.

    ``PYTHONHASHSEED`` is the only lever that changes ``set`` ordering between
    processes, so varying it is the discriminating test for "we iterated a set".
    """
    for script_name in ("validate_skills.py", "validate_agents.py"):
        script = SCRIPTS / script_name
        results = [_run(script, hash_seed=seed) for seed in HASH_SEEDS]
        assert _all_identical(results), (
            f"{script_name} output depends on PYTHONHASHSEED -- "
            f"something iterates a set or dict without sorting: {results}"
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
