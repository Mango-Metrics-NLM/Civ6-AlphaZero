# PR: Code hygiene sweep: dead-code deletions, whole-repo lint gate, blocking mypy

## Summary
- Executes the approved six-phase code-hygiene plan: doc-truth fixes, OpenSpec archive housekeeping, ~3,350 lines of verified-orphaned code deleted, ruff/mypy pins aligned at 0.15.8/1.19.1 across all five pin locations, lint gate extended to the whole repository, and mypy taken from decorative (advisory, bare env) to a blocking gate at zero errors.
- eval-harness CI job flipped to enforcing (observation window closed green); never-in-CI test dirs measured and documented (263 passed / 12 env-dependent failures), CI wiring deferred until env-required markers exist.
- Independent isolated-context review ran as the review of record; both blocking findings (validate_kb K5 archive fallback, ruff extend-exclude anchoring) fixed on-branch.

## Changes
- `pyproject.toml` — dead/orphaned ignores removed, mlflow<3.0 bound, pins, anchored `extend-exclude` + `force-exclude`, dev extra += types-requests
- `.github/workflows/ci.yml` — whole-repo ruff, canonical mypy env, blocking mypy, enforcing eval-harness, pin bumps (both jobs)
- `.pre-commit-config.yaml` — pin bump, widened `&maintained` anchor
- `Makefile` — lint/lint-check/typecheck now reproduce CI exactly
- Deletions: `alpha_zero/cloud/gcp/cost_optimization.py`, `alpha_zero/cloud/gcp/lifecycle_cli.py`, `alpha_zero/inference/api_server.py`, `run_debug_selfplay.py`, `scripts/analyze_training_run.py`, `scripts/validate_v130_training.py`, 6 expired skipped tests; `inspect_db.py` → `scripts/`
- Types: 130→0 mypy errors across 52 files (stale-ignore removals + annotations/casts/value-preserving wrappers)
- Format-only: 133 files reformatted at ruff 0.15.8 (tests/, scripts/, root, infra/, 2 in alpha_zero/)
- Governance: `scripts/validate_kb.py` K5 archive fallback (+2 tests); archived `add-enterprise-docs-and-gates`; new `harden-lint-and-type-gates` change
- Docs: CLAUDE.md (§4.1, §5.3, §7 counts), MODULE_REFERENCE, INFERENCE_SERVICE, README, CONTRIBUTING, CHANGELOG

## Test Results
- **Unit/Integration/Cloud:** 11,728 pass, 0 fail (57 skipped, 1 xfailed) — CI-equivalent sweep
- **Governance:** 50 tests pass (incl. 2 new K5 tests)
- **Coverage:** delegated to CI job (85% gate; deleted modules already omitted; local `--cov` blocked by documented numpy-2.4 tracer issue)
- **Lint:** clean — `ruff check .` / `ruff format --check .` (863 files)
- **Type Check:** clean — `mypy alpha_zero/` exits 0 (342 files)
- **Smoke Tests:** CI run 30169685310: lint/test/cloud/security/eval-harness jobs all green

## Breaking Changes
None. Contract verified by the independent review: deletions are zero-importer, format commits byte-derivable via `ruff format`, type fixes are annotations/casts/value-preserving wrappers.

## Checklist
- [x] All verification commands pass
- [x] Coverage >= 85% on changed modules (deleted modules were coverage-omitted; CI gate authoritative)
- [x] CLAUDE.md updated
- [x] No hardcoded values (review criterion 6: zero hits)
- [x] Type hints on all new public interfaces
