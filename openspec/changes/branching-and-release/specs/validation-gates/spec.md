# validation-gates — deltas

## ADDED Requirements

### Requirement: Gating workflows record every protected-branch state

The ``ci``, ``codeql``, ``security-scan`` and ``scorecard`` workflows SHALL run on
push to every protected branch (``dev``, ``qa``, ``main`` and any branch in
``CIV6AZ_PROTECTED_BRANCHES``), and a newer push MUST NOT cancel an in-progress
run on a protected branch. The workflow-pinning regression suite MUST read the
protected list from the same source as the agent guard and fail when either
property is broken.

#### Scenario: A workflow drops a protected branch from its push trigger

- **WHEN** ``scorecard.yml`` no longer lists ``qa`` under ``on.push.branches``
- **THEN** ``test_gating_workflows_run_on_every_protected_branch`` fails

#### Scenario: A workflow cancels runs on a protected branch

- **WHEN** ``codeql.yml`` sets ``cancel-in-progress: true`` while pushing on ``dev``
- **THEN** ``test_runs_on_protected_branches_are_never_cancelled`` fails

#### Scenario: A push to qa is ratcheted against qa

- **WHEN** the coverage-floor ratchet runs on a push event with ``GITHUB_REF_NAME=qa``
- **THEN** its first base candidate is ``origin/qa``
