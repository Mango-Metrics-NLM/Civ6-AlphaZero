# Tasks — harden-lint-and-type-gates

## 1. Pin alignment (measured)

- [ ] 1.1 (S) Bump `ruff==0.8.0` → `ruff==0.15.8` and `mypy==1.13.0` →
      `mypy==1.19.1` in ci.yml lint job, ci.yml governance job,
      `.pre-commit-config.yaml`, pyproject `dev` extra, and `requirements.txt`
      (pin the currently-unpinned entries) — one commit
- [ ] 1.2 (S) Measure `ruff check` / `ruff format --check` on `alpha_zero/` at
      0.15.8; fix surfaced violations; drop the then-dead `UP038` ignore

## 2. Lint scope expansion

- [ ] 2.1 (M) `ruff format` + `ruff check --fix` the drifted files under
      `tests/` (logic-free commits, ≤20 files each)
- [ ] 2.2 (M) Same for `scripts/` and root runners
- [ ] 2.3 (S) ci.yml lint job → `ruff check .` / `ruff format --check .`;
      widen the pre-commit `&maintained` anchor to match
- [ ] 2.4 (S) Convert genuine `__init__.py` re-export `noqa: F401`s to
      `__all__`; keep availability-probe noqas

## 3. Mypy gate

- [ ] 3.1 (S) ci.yml lint job installs `-e ".[dev]"`; add `types-requests` to
      the dev extra; drop the `--ignore-missing-imports` CLI flag (pyproject
      global covers it)
- [ ] 3.2 (M) Measure `mypy alpha_zero/` at the new pin in the canonical env;
      apply the decision gate (≤150 burn down fully / 150–300 mechanical first /
      >300 stop and report)
- [ ] 3.3 (M) Burn down mechanical errors (`unused-ignore`, bare ignores)
- [ ] 3.4 (M) Burn down semantic errors per package, tests run per commit
- [ ] 3.5 (S) Remove `continue-on-error: true` from the mypy step after one
      green advisory CI run; update CLAUDE.md §4.1 invocation

## 4. Test-scope + eval-harness (measured)

- [ ] 4.1 (S) Measure `tests/e2e tests/inference tests/regression tests/scripts
      tests/smoke` locally; record runtime and pass state
- [ ] 4.2 (S) If green and fast: add advisory `extended-tests` CI job with a
      dated observation window; else document and defer
- [ ] 4.3 (S) Check eval-harness job pass rate; flip to enforcing if
      consistently green (carried from add-enterprise-docs-and-gates task 6.3);
      else document why not
- [ ] 4.4 (S) Update the flaky `test_concurrent_submit_and_sample` skip reason
      with a tracking pointer
