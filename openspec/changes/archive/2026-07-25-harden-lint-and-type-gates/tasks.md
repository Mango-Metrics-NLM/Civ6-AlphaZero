# Tasks — harden-lint-and-type-gates

## 1. Pin alignment (measured)

- [x] 1.1 (S) Bump `ruff==0.8.0` → `ruff==0.15.8` and `mypy==1.13.0` →
      `mypy==1.19.1` in ci.yml lint job, ci.yml governance job,
      `.pre-commit-config.yaml`, pyproject `dev` extra, and `requirements.txt`
      (pin the currently-unpinned entries) — one commit
- [x] 1.2 (S) Measure `ruff check` / `ruff format --check` on `alpha_zero/` at
      0.15.8; fix surfaced violations; drop the then-dead `UP038` ignore

## 2. Lint scope expansion

- [x] 2.1 (M) `ruff format` + `ruff check --fix` the drifted files under
      `tests/` (logic-free commits, ≤20 files each)
- [x] 2.2 (M) Same for `scripts/` and root runners
- [x] 2.3 (S) ci.yml lint job → `ruff check .` / `ruff format --check .`;
      widen the pre-commit `&maintained` anchor to match
- [x] 2.4 (S) Measured: all 12 remaining `__init__.py` `noqa: F401`s are
      availability probes inside try/except ImportError blocks (or side-effect
      imports) — none are plain re-exports, so all are kept as-is

## 3. Mypy gate

- [x] 3.1 (S) ci.yml lint job installs `-e ".[dev]"`; add `types-requests` to
      the dev extra; drop the `--ignore-missing-imports` CLI flag (pyproject
      global covers it)
- [x] 3.2 (M) Measure `mypy alpha_zero/` at the new pin in the canonical env;
      apply the decision gate (≤150 burn down fully / 150–300 mechanical first /
      >300 stop and report)
- [x] 3.3 (M) Burn down mechanical errors (`unused-ignore`, bare ignores)
- [x] 3.4 (M) Burn down semantic errors per package, tests run per commit
- [x] 3.5 (S) Removed `continue-on-error: true` after the advisory step ran
      green in the canonical env (CI run 30169685310); CLAUDE.md §4.1 updated

## 4. Test-scope + eval-harness (measured)

- [x] 4.1 (S) Measure `tests/e2e tests/inference tests/regression tests/scripts
      tests/smoke` locally; record runtime and pass state
- [x] 4.2 (S) Measured 2026-07-25: 263 passed / 12 failed / 18 skipped in
      2m03s under the [dev] install. All 12 failures are environment-dependent
      (need .env, GCP ADC, or MLflow run history — test_deployment_infra 6,
      test_gcp_diagnose 1, test_run_baseline_eval 3, +2), not code bugs.
      Wiring into CI therefore needs env_required-style markers on those
      suites first — DEFERRED to a follow-up; no CI job added
- [x] 4.3 (S) Check eval-harness job pass rate; flip to enforcing if
      consistently green (carried from add-enterprise-docs-and-gates task 6.3);
      else document why not
- [x] 4.4 (S) Update the flaky `test_concurrent_submit_and_sample` skip reason
      with a tracking pointer
