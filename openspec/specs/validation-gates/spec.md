# Validation Gates

## Purpose

Defines the deterministic quality gates that guard the repository: the
stdlib-only governance validators and their exit-code contract, OpenSpec
structural validity, the CI governance job, the lint/format and type-check
gates, and the installable git hooks whose tool pins mirror CI. The intent is
that every gate is reproducible locally, versioned with the code, and cannot
drift silently from what CI enforces.

## Requirements

### Requirement: Deterministic exit-code gates

Every governance gate SHALL be a stdlib-only script with a stable exit-code
contract — 0 for pass, 1 for violations, 2 for usage error — and MUST NOT depend
on network access, wall-clock time, or randomness.

#### Scenario: A gate reports violations deterministically

- **WHEN** a governance validator is run against input that violates a rule
- **THEN** it prints ``ERROR <file>[:<line>]: <CODE> <msg>`` and exits 1

### Requirement: OpenSpec structural validity

The ``openspec/`` tree SHALL be structurally valid at all times, as enforced by
``scripts/validate_openspec.py``. Deltas MUST use only ADDED/MODIFIED/REMOVED
Requirements sections, and MODIFIED/REMOVED requirements MUST name an existing
baseline requirement.

#### Scenario: An invalid delta section is rejected

- **WHEN** a delta file introduces an ``## UPDATED Requirements`` heading
- **THEN** ``validate_openspec.py`` reports an OS020 violation and exits 1

### Requirement: CI governance job

Continuous integration SHALL run all governance validators in a dedicated job, and
that job MUST run on pushes to development branches and on pull requests. The job
SHALL also run a workflow linter (``zizmor``) over ``.github/workflows``, and
every ``actions/checkout`` step in every workflow MUST set
``persist-credentials: false``.

#### Scenario: Governance runs on a development-branch push

- **WHEN** a commit is pushed to a ``claude/**`` branch
- **THEN** the CI ``governance`` job runs the validators against that commit

#### Scenario: A checkout persists credentials

- **WHEN** a workflow step uses ``actions/checkout`` without ``persist-credentials: false``
- **THEN** ``pytest tests/regression/test_workflow_pinning_regression.py`` fails naming the workflow

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

### Requirement: Public-readiness regression gate

Continuous integration SHALL run ``tests/regression/test_public_readiness_regression.py``
on every push and pull request, and that test MUST read its patterns and
allowlists from data files so that adding a check does not require editing the
test, and every allowlist entry MUST carry a reason and MUST fail when stale.

#### Scenario: A stale allowlist entry is detected

- **WHEN** an allowlisted file no longer contains the pattern it was allowlisted for
- **THEN** the public-readiness regression test fails naming the entry

#### Scenario: The gate is enumerated by CI

- **WHEN** ``tests/regression/test_suite_wiring_regression.py`` runs
- **THEN** the regression directory is named by a CI pytest invocation and the gate is counted as wired

### Requirement: Secret scanning over tree and history

The repository SHALL be scanned for committed secrets by a maintained,
rule-driven scanner covering both the tracked working tree and the full commit
history, because a secret removed in a later commit remains published in the
commit that introduced it. The scanner version MUST have exactly one definition,
which CI reads rather than restates.

#### Scenario: A credential is added to a tracked file

- **WHEN** a commit adds a value matching a secret rule to any non-allowlisted path
- **THEN** the secret-scanning gate fails and names the rule and file, with the value redacted

#### Scenario: The allowlist cannot neuter the gate

- **WHEN** the scanner configuration contains a catch-all allowlist pattern
- **THEN** the repository-hygiene regression suite fails, because a gate that
  allowlists everything reports success while detecting nothing

### Requirement: Deterministic validation of Agent.md persona documents

Every **top-level** package under `alpha_zero/` SHALL carry an `Agent.md` whose
structure is machine-checkable: a heading naming a real module, a persona
statement, the required sections, coverage literals matching the canonical gate,
and file references that resolve.

Scoped to top-level deliberately. `alpha_zero/` holds 29 top-level packages and 44
importable packages in total; the nested 15 are implementation subdivisions
(`hierarchical/goals`, `rag/stores`, and so on) whose ownership is already stated
by their parent's persona document. Requiring a persona doc per nested package
would produce thin files that restate the parent. A substantial new subpackage
still warrants one — `alpha_zero/core/mcts/` has its own.

#### Scenario: A persona document references a file that no longer exists

- **WHEN** an `Agent.md` names a backticked `*.py` path that is absent from both the
  documented module directory and the repository root
- **THEN** the agents validator reports A6 and exits non-zero, unless that line
  carries the `<!-- validate-agents: allow -->` marker

#### Scenario: A new top-level package ships without a persona document

- **WHEN** a direct child directory of `alpha_zero/` contains `__init__.py` but no `Agent.md`
- **THEN** the agents validator reports A5 and exits non-zero

#### Scenario: A `--path` target does not exist

- **WHEN** the validator is invoked with `--path` naming a missing or non-`Agent.md` file
- **THEN** it reports A0 and exits non-zero, rather than silently validating nothing

### Requirement: Both skill trees are gated

Skill validation SHALL cover every skill tree the harness routes to it, and MUST
parse YAML block scalars so a folded value is validated on its content.

#### Scenario: A skill outside `.claude/skills/` violates a rule

- **WHEN** a `SKILL.md` under `.agents/skills/` declares a name that differs from its
  directory, or a coverage literal that differs from the canonical gate
- **THEN** the skills validator reports the violation and exits non-zero

### Requirement: Function size is mechanically bounded

A lint gate SHALL bound statements per function. Functions already over the limit
MUST be enumerated individually as a burndown list rather than accommodated by
raising the limit.

#### Scenario: A new oversized function is introduced

- **WHEN** a function outside the burndown list exceeds the configured statement limit
- **THEN** `ruff check .` fails with `PLR0915`

