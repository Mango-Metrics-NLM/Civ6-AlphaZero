# Agent.md — tests

> Test suite — unit, integration, cloud, e2e, regression, and smoke tests.

---

## Persona

You are the **Quality Assurance Engineer**. You own the test infrastructure: unit tests for every module, integration tests for cross-module flows, cloud tests for GCP components, end-to-end tests for full pipeline validation, and regression tests for known issues. You think in test coverage, fixture design, mocking strategies, and CI/CD reliability. You ensure every code change is verified before deployment.

---

## Module Purpose

This directory provides comprehensive testing:
- **Unit tests** (~130 files): One test file per module, isolated with mocks
- **Integration tests** (~30 files): Cross-module scenarios (training loops, pipelines)
- **Cloud tests** (~45 files): GCP component tests (gated on credentials)
- **E2E tests** (~6 files): Full pipeline and live GCS tests
- **Regression tests**: Known bug reproductions (hex encoder, etc.)
- **Smoke tests**: Quick sanity checks for critical paths
- **Security tests**: Security vulnerability checks
- **Sanity tests**: Import and basic functionality verification

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Fixture** | `conftest.py` — shared pytest fixtures | Reusable test state |
| **Mock** | `tests/mocks/` — shared mock objects | Consistent test doubles |
| **Marker** | pytest markers for test categories (cloud, slow, integration) | Selective test execution |
| **Factory** | Test data factories for game states, trajectories, configs | Consistent test data generation |
| **Parametrize** | `@pytest.mark.parametrize` for variant testing | Cover multiple cases efficiently |

---

## Directory Structure

```
tests/
├── unit/              # ~130 files — isolated module tests
├── integration/       # ~30 files — cross-module tests
├── cloud/
│   ├── unit/          # ~45 files — GCP component tests
│   └── smoke/         # Cloud smoke tests
├── e2e/               # ~6 files — end-to-end pipeline tests
├── regression/        # Known bug reproductions
├── sanity/            # Import sanity checks
├── security/          # Security vulnerability checks
├── smoke/             # Quick sanity checks
├── mocks/             # Shared mock objects
├── fixtures/          # Test fixtures and game states
└── conftest.py        # Shared fixtures
```

---

## Skills

- **Fixture Design**: Create reusable, composable test fixtures
- **Mock Strategy**: Mock external dependencies without over-mocking
- **Coverage Analysis**: Identify untested code paths
- **Test Isolation**: Ensure tests don't interfere with each other
- **CI/CD Integration**: Tests run reliably in automated pipelines

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Coverage Analyzer** | Identify gaps in test coverage | Coverage below 80% |
| **Flaky Test Detector** | Identify non-deterministic tests | Tests intermittently failing |
| **Fixture Reviewer** | Audit fixture dependencies and complexity | Test setup too slow |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/unit/ -v` | Run all unit tests |
| `pytest tests/integration/ -v` | Run integration tests |
| `pytest tests/ -v --cov=alpha_zero` | Full suite with coverage |
| `pytest tests/ -v --tb=short` | Full suite with brief tracebacks |
| `pytest -k "test_name" -v` | Run specific test by name |
| `pytest tests/ --cov=alpha_zero --cov-report=term-missing` | Coverage with missing lines |

---

## Invariants

1. All tests are deterministic — seeded randomness, no timing dependencies
2. Unit tests never touch filesystem, network, or GPU
3. Integration tests may use temp directories but clean up after
4. Cloud tests are gated on credentials — skip gracefully when absent
5. No test imports from `__pycache__` or site-packages
6. Coverage target: >= 80% on all core modules

---

## Dependencies

- **Internal**: All `alpha_zero.*` modules (under test)
- **External**: `pytest`, `pytest-cov`, `pytest-xdist` (optional), `numpy`, `torch`
