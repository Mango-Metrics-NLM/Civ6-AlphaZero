# Harden lint and type gates

## Why

Three gate weaknesses were measured during the 2026-07-25 hygiene sweep:

1. **Mypy is decorative.** The CI mypy step is `continue-on-error: true`, runs in
   a bare environment (no project dependencies), and contradicts CLAUDE.md §1.2's
   success criterion ("mypy exits 0"). Baseline at CI's exact invocation
   (mypy 1.13.0, real deps installed): 144 errors in 54 files.
2. **Lint scope covers only `alpha_zero/`.** `tests/`, `scripts/`, and the root
   runners are never ruff-checked or format-checked in CI; 26 files fail
   `ruff format --check`. The pre-commit `&maintained` anchor mirrors the gap.
3. **Tool pins drift across five locations.** CI lint job (`ruff==0.8.0`,
   `mypy==1.13.0`), CI governance job, `.pre-commit-config.yaml`, the pyproject
   `dev` extra, and `requirements.txt` (unpinned `>=`) do not agree with each
   other or with current releases (ruff 0.15.8 / mypy 1.19.1).

## What Changes

- Upgrade and align ruff/mypy pins across all five locations (measured: fix any
  new violations the 0.15.8 rules surface; drop the then-dead `UP038` ignore).
- Reformat the drifted files and extend the CI lint job and pre-commit anchor to
  the whole repository (`ruff check .` / `ruff format --check .`).
- Give the CI mypy step a real environment (`pip install -e ".[dev]"`, stubs
  included), burn down the measured errors, then remove `continue-on-error`.
- Measure the never-in-CI test directories (`tests/e2e`, `tests/inference`,
  `tests/regression`, `tests/scripts`, `tests/smoke`) and wire them into CI
  behind a dated observation window if green; check the eval-harness job's pass
  rate and flip it to enforcing if consistently green (task 6.3 carried from
  the archived `add-enterprise-docs-and-gates` change).

## Impact

- Affected capability: `validation-gates` (MODIFIED: Installable git hooks with
  CI-aligned pins; ADDED: Whole-repository lint gate, Blocking type-check gate).
- Affected code: `.github/workflows/ci.yml`, `.pre-commit-config.yaml`,
  `pyproject.toml`, `requirements.txt`, formatting-only edits across `tests/`,
  `scripts/`, and root runners, and `type: ignore` adjustments in `alpha_zero/`.
- No runtime behavior changes: every code edit is formatting or annotation-level.
