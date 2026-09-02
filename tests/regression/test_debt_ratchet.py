"""The tech-debt ratchet: no tracked metric may get worse than the committed baseline.

``scripts/measure_debt.py`` produces every number the plan in
``kb/specs/20260902-tech-debt-hardening.md`` tracks (dead modules, files over
1,000 lines, coverage-omit patterns, ``type: ignore`` counts, unsafe
``torch.load`` calls, ...). The baseline lives at
``docs/evidence/tech_debt_baseline/baseline.json``. A PR may lower a metric and
should then lower the baseline in the same PR; it may never raise one. That is
the whole gate: it turns "we will pay this down" into something CI enforces.

Negative control (recorded 2026-09-02): with the baseline at
``dead_modules: 79``, adding an unimported module
``alpha_zero/_ratchet_probe.py`` made this test fail with
``dead_modules: 80 > baseline 79``; removing it restored the pass. The test
also fails when a metric is missing from either side, so the key set cannot
drift silently.

Runtime: ~3 s (one AST walk of ``alpha_zero/`` and ``tests/`` plus one ruff
invocation when ruff is installed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import measure_debt  # noqa: E402

BASELINE_PATH = REPO_ROOT / "docs" / "evidence" / "tech_debt_baseline" / "baseline.json"


@pytest.fixture(scope="module")
def baseline() -> dict[str, int | None]:
    assert BASELINE_PATH.exists(), f"missing baseline: {BASELINE_PATH}"
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict[str, int | None]:
    return measure_debt.measure(REPO_ROOT)


def test_metric_keys_match_baseline(
    baseline: dict[str, int | None], current: dict[str, int | None]
) -> None:
    missing = set(baseline) - set(current)
    extra = set(current) - set(baseline)
    assert not missing and not extra, (
        f"metric set drifted; missing from measurement: {sorted(missing)}, "
        f"not in baseline: {sorted(extra)}. Regenerate with "
        "`python scripts/measure_debt.py --write docs/evidence/tech_debt_baseline/baseline.json`"
    )


def test_no_metric_regressed(
    baseline: dict[str, int | None], current: dict[str, int | None]
) -> None:
    regressions = measure_debt.compare(current, baseline)
    assert not regressions, (
        "tech-debt ratchet: "
        + "; ".join(regressions)
        + ". Lower the metric, or if the increase is deliberate and reviewed, "
        "regenerate the baseline in the same PR and say why in the PR body."
    )


def test_compare_reports_only_increases() -> None:
    base = {"a": 5, "b": None, "c": 3}
    assert measure_debt.compare({"a": 5, "b": 9, "c": 2}, base) == []
    assert measure_debt.compare({"a": 6, "b": 9, "c": 2}, base) == ["a: 6 > baseline 5"]
    assert measure_debt.compare({"a": 5, "c": None}, base) == []


def test_omit_rule_is_the_documented_one() -> None:
    patterns = ["alpha_zero/cloud/*", "alpha_zero/utils/llm_client.py"]
    assert measure_debt.is_omitted("alpha_zero/cloud/gcp/x.py", patterns)
    assert measure_debt.is_omitted("alpha_zero/utils/llm_client.py", patterns)
    assert not measure_debt.is_omitted("alpha_zero/utils/config.py", patterns)
