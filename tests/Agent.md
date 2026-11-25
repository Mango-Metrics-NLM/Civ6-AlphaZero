# Agent.md — tests

> Test suite — unit, integration, cloud, e2e, regression, and smoke tests.

---

## Persona

You are the **Quality Assurance Engineer**. You own the test infrastructure: unit tests for every module, integration tests for cross-module flows, cloud tests for GCP components, end-to-end tests for full pipeline validation, and regression tests for known issues. You think in test coverage, fixture design, mocking strategies, and CI/CD reliability. You ensure every code change is verified before deployment.

---

## Module Purpose

This directory provides comprehensive testing:
- **Unit tests** (325 files): One test file per module, isolated with mocks
- **Integration tests** (49 files): Cross-module scenarios (training loops, pipelines)
- **Cloud tests** (46 unit + 2 smoke): GCP component tests — these import cleanly with no
  Google SDK present, so CI runs them on `.[dev]` alone
- **E2E tests** (8 files): Full pipeline and live GCS tests
- **Regression tests** (23 files): known bug reproductions (hex encoder, CUDA determinism, dual-mode GPU/E2E & worktree AQA), repository-hygiene
  invariants (tracked-but-ignored files, secret-scan allowlist scoping), the AQA meta-gate that
  every suite under `tests/` is actually run by CI, debt/coverage ratchets, and the
  governance validators' CLI contract
- **Structural / contract tests**: pin an interface rather than a behaviour — import-shim
  identity, `mock.patch` reachability through a shim, and package layering asserted by parsing
  ASTs (`tests/unit/test_mcts_split_backcompat.py`)
- **Governance tests** (19 files): the deterministic validators (`openspec`, `kb`, `skills`, `agents`,
  `doc-sync`), harness hook guards, and cross-file tool pins. Run with their own `pytest.ini` and `--confcutdir` so they load neither the root
  `addopts` nor the torch-importing `tests/conftest.py`
- **Script tests** (30 files): entry points under `scripts/`
- **Smoke tests** (3 files): Quick sanity checks for critical paths
- **Security tests** (3 files): Security vulnerability checks + secret scanning (`make security-secrets`)
- **Sanity tests** (1 file): Import and basic functionality verification

---

## Key Files

| Path | Responsibility |
|------|---------------|
| `conftest.py` | Root pytest fixtures and torch environment setup |
| `governance/pytest.ini` | Isolated governance suite pytest config (`--confcutdir=tests/governance`) |
| `governance/agent_warning_baseline.json` | Ratchet baseline (`max_warnings`) for `scripts/validate_agents.py` |
| `regression/test_gpu_e2e_and_worktree_aqa.py` | Dual-mode (live GPU + mocked) AQA suite & worktree/venv/RAG hardening |
| `regression/test_gpu_skip_gate_aqa.py` | AQA gate verifying opt-in `CIV6AZ_REQUIRE_GPU` skip-to-failure plugin behavior |
| `regression/test_post_sync_defects_regression.py` | Regression tests for post-sync defects including bidirectional `cd -P` physical resolution across NTFS junctions |
| `fixtures/gpu_skip_gate.py` | Opt-in pytest plugin (`CIV6AZ_REQUIRE_GPU`) enforcing device count and failing CUDA/GPU/device skips |
| `regression/test_suite_wiring_regression.py` | Enforces CI + `Makefile` wiring across all test directories |
| `regression/test_repo_hygiene_regression.py` | Validates `.gitignore` and `.gitleaks.toml` invariants |

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
├── unit/              # 325 files — isolated module tests
├── integration/       # 49 files — cross-module tests
├── cloud/
│   ├── unit/          # 46 files — GCP component tests
│   └── smoke/         # 2 files — cloud smoke tests
├── e2e/               # 8 files — end-to-end pipeline tests
├── governance/        # 19 files — validator & harness tests (own pytest.ini, --confcutdir)
├── scripts/           # 30 files — scripts/ entry-point tests
├── inference/         # 1 file — inference service tests
├── regression/        # 23 files — Known bug reproductions, repo-hygiene, ratchets & AQA gates
├── sanity/            # 1 file — Import sanity checks (auto-discovers new modules)
├── security/          # 3 files — Security vulnerability checks
├── smoke/             # 3 files — Quick sanity checks
├── mocks/             # Shared mock objects
├── fixtures/          # Test fixtures and game states
└── conftest.py        # Shared fixtures (imports torch at module scope)
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
| **Coverage Analyzer** | Identify gaps in test coverage | Coverage below the 85% gate |
| **Flaky Test Detector** | Identify non-deterministic tests | Tests intermittently failing |
| **Fixture Reviewer** | Audit fixture dependencies and complexity | Test setup too slow |

---

## Tools

| Tool | Purpose |
|------|---------|
| `make verify` | **The gate.** Every CI job in one pass — lint, format, types, four suites, security, governance, validators, coverage, eval |
| `make test` | Every suite CI's `test` + `cloud-tests` jobs run, in CI's order and marker filters (pinned by `tests/regression/test_suite_wiring_regression.py`) |
| `make test-all` | CI's coverage job, gate 85% (measured 89.30% on 2026-08-30) |
| `make test-governance` | Validator tests, isolated from the root conftest |
| `make security-secrets` | gitleaks over the tracked tree and full history |
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
6. Coverage target: >= 85% on all core modules

---

## Dependencies

- **Internal**: All `alpha_zero.*` modules (under test)
- **External**: `pytest`, `pytest-cov`, `pytest-xdist` (optional), `numpy`, `torch`
