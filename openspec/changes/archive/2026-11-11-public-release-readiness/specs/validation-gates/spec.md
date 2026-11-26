# validation-gates — deltas

## ADDED Requirements

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

## MODIFIED Requirements

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
