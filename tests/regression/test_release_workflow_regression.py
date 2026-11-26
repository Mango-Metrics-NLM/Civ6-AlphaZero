"""``release.yml`` keeps its gates (plan S3.5).

The workflow a tag push runs is the one in the tagged commit, so a release
workflow that publishes without ``verify-tag`` and ``wait-for-ci`` is a release
that can ship a commit CI never passed. ``tag_release.py`` refuses to tag a
commit whose ``release.yml`` lacks the wait gate (``is_r2_lineage``); this test
holds the current file to the same shape, plus least privilege per job, and the
push checks ``wait-for-ci`` polls to jobs that actually run on push to qa and
main.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from scripts.release_checks import PUSH_CHECKS, WAIT_JOB
from tests.regression.test_branch_rulesets_regression import expand_matrix

RELEASE = Path(".github/workflows/release.yml")
CI = Path(".github/workflows/ci.yml")


def _doc(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return doc


def _jobs() -> dict[str, Any]:
    jobs: dict[str, Any] = _doc(RELEASE)["jobs"]
    return jobs


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs") or []
    return [needs] if isinstance(needs, str) else list(needs)


def test_publishing_needs_both_gates() -> None:
    writers = {
        n: j for n, j in _jobs().items() if (j.get("permissions") or {}).get("contents") == "write"
    }
    assert list(writers) == ["release"], "exactly one job may hold contents: write"
    assert {"verify-tag", WAIT_JOB} <= set(_needs(writers["release"]))


def test_wait_for_ci_is_least_privilege_and_bounded() -> None:
    job = _jobs()[WAIT_JOB]
    assert job["permissions"] == {"contents": "read", "checks": "read", "pull-requests": "read"}
    assert job["timeout-minutes"] == 100
    run = " ".join(str(s.get("run", "")) for s in job["steps"])
    assert "scripts.release_checks wait-for-ci" in run
    assert '"${GITHUB_REF}^{commit}"' in run


def test_verify_tag_runs_the_shared_checks_read_only() -> None:
    job = _jobs()["verify-tag"]
    assert job["permissions"] == {"contents": "read", "pull-requests": "read"}
    run = " ".join(str(s.get("run", "")) for s in job["steps"])
    assert "scripts.release_checks verify-tag" in run
    checkout = next(
        s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout@")
    )
    assert checkout["with"]["fetch-depth"] == 0


def test_the_workflow_default_is_read_only() -> None:
    assert _doc(RELEASE)["permissions"] == {"contents": "read"}


def test_rc_builds_are_stamped_and_notes_use_the_base_version() -> None:
    steps = " ".join(str(s.get("run", "")) for s in _jobs()["release"]["steps"])
    assert "scripts.release_version check" in steps
    assert "scripts.release_version stamp" in steps
    assert "scripts.release_version notes" in steps
    names = [s.get("name", "") for s in _jobs()["release"]["steps"]]
    assert names.index("Check and stamp the build version") < names.index("Build sdist and wheel")


def test_push_checks_run_on_push_to_qa_and_main() -> None:
    doc = _doc(CI)
    on = doc.get(True, doc.get("on"))
    branches = on["push"]["branches"]
    assert "qa" in branches and "main" in branches
    produced: set[str] = set()
    for job_id, job in doc["jobs"].items():
        if "if" in job:
            continue
        for leg in expand_matrix(job.get("strategy")):
            name = str(job.get("name", job_id))
            for key, value in leg.items():
                name = name.replace("${{ matrix." + key + " }}", str(value))
            produced.add(name)
    assert set(PUSH_CHECKS) <= produced
