# Validation Gates — deltas for harden-lint-and-type-gates

## ADDED Requirements

### Requirement: Whole-repository lint gate

The CI lint job SHALL run `ruff check` and `ruff format --check` over the whole
repository — including `tests/`, `scripts/`, and root-level runners — not just
`alpha_zero/`, and the pre-commit hooks MUST cover the same file set.

#### Scenario: A drifted script fails CI

- **WHEN** a file under `scripts/` is committed with formatting that
  `ruff format` would change
- **THEN** the CI lint job fails on `ruff format --check .`

### Requirement: Blocking type-check gate

The CI mypy step SHALL run with the project and its `dev` extra installed (so
type stubs and real dependencies are visible) and MUST be blocking — not
`continue-on-error` — once the measured error count is zero.

#### Scenario: A type error blocks the build

- **WHEN** a commit introduces a mypy error in `alpha_zero/`
- **THEN** the CI lint job fails instead of reporting the error as advisory

## MODIFIED Requirements

### Requirement: Installable git hooks with CI-aligned pins

The repository SHALL provide installable git hooks whose tool versions match CI,
via ``.pre-commit-config.yaml`` and a ``make hooks-install`` target. The same
ruff and mypy versions MUST be pinned identically in the CI lint job, the CI
governance job, ``.pre-commit-config.yaml``, the pyproject ``dev`` extra, and
``requirements.txt``.

#### Scenario: The ruff pin matches CI

- **WHEN** a contributor installs the git hooks and runs them
- **THEN** the ruff version used by the hooks equals the version used by the CI lint job

#### Scenario: The dev extra matches CI

- **WHEN** a contributor installs ``pip install -e ".[dev]"``
- **THEN** the installed ruff and mypy versions equal the versions pinned in the CI lint job
