# PRD: Fix Stale Tournament Test

## User Story

**As a** developer maintaining the Civ6 AlphaZero pipeline  
**I want** the `test_no_command_field_set` test to match the current `VertexTournamentJob.launch()` behavior  
**So that** the test suite accurately validates the tournament job specification

## Background

The `VertexTournamentJob.launch()` method (line 1023 of `vertex_jobs.py`) sets:

```python
command=["python", "-m", "alpha_zero.cloud.gcp.entrypoint"]
```

But the test at line 996 of `test_tournament_worker.py` asserts:

```python
assert submitted_spec.command == []
```

This is a stale assertion from when tournament jobs relied on the container's ENTRYPOINT directive.

## Acceptance Criteria

### AC1: Test Matches Production Code

**Given** a `VertexTournamentJob` is launched  
**When** the submitted spec is inspected  
**Then** `spec.command` equals `["python", "-m", "alpha_zero.cloud.gcp.entrypoint"]`

### AC2: No Regression in Suite

**Given** the test is updated  
**When** the full tournament test suite is run  
**Then** all tests pass (0 failures, 0 errors)

## Out of Scope

- Changing the actual `VertexTournamentJob.launch()` behavior
- Migrating to container ENTRYPOINT architecture
- Adding new test cases for tournament job specs

## Success Metrics

- `pytest tests/cloud/unit/test_tournament_worker.py -v` → 100% pass
- No regression in existing 110+ cloud unit tests
