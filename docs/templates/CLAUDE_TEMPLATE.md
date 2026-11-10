# Agentic Coding System Prompt Template

> **Design Principle:** This template treats prompting as constraint programming, not instruction writing. Define the feasible region, objective function, and search parameters—then let the agent solve.

---

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am building: [System type in one sentence]
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
- [ ] All tests pass: `pytest tests/ -v` exits with code 0
- [ ] Coverage >= {COVERAGE_THRESHOLD}%: `pytest --cov={PACKAGE_NAME}` shows >= {COVERAGE_THRESHOLD}%
- [ ] Type checking passes: `mypy {PACKAGE_NAME}/` exits with code 0
- [ ] Linting passes: `ruff check {PACKAGE_NAME}/` exits with code 0
- [ ] Smoke test: `python -m {PACKAGE_NAME} --dry_run` completes
- [ ] [Custom criterion 1]
- [ ] [Custom criterion 2]
```

### 1.3 Problem Description (The "Three Paragraphs")

**Paragraph 1 - Core Problem:**
```
[What problem does this solve? What's the core coordination logic? What are the
primary use cases and user workflows?]
```

**Paragraph 2 - Data Flows:**
```
[What are the key data flows? What state must be maintained and synchronized?
Describe: Input sources → Processing stages → Output destinations → Feedback loops]
```

**Paragraph 3 - Failure Modes:**
```
[What are the failure modes? What invariants must hold under adversarial conditions?
List critical invariants, potential failure scenarios, and recovery strategies.]
```

---

## SECTION 2: C4 ARCHITECTURE

### 2.1 Level 1: System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SYSTEM CONTEXT                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌──────────┐         ┌─────────────────────┐         ┌──────────────┐   │
│    │  User /  │         │                     │         │   External   │   │
│    │  Client  │────────▶│   {SYSTEM_NAME}     │◀───────▶│   Services   │   │
│    │          │         │                     │         │   (APIs)     │   │
│    └──────────┘         └─────────────────────┘         └──────────────┘   │
│         │                        │                             │            │
│         │                        ▼                             │            │
│         │               ┌─────────────────┐                    │            │
│         └──────────────▶│   Data Store    │◀───────────────────┘            │
│                         │   (Database)    │                                 │
│                         └─────────────────┘                                 │
│                                                                              │
│  Legend:                                                                     │
│  ────▶  Request/Response    ◀───▶  Bidirectional                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Actors:
- [Actor 1]: [Description and interaction pattern]
- [Actor 2]: [Description and interaction pattern]

External Systems:
- [System 1]: [Purpose and integration method]
- [System 2]: [Purpose and integration method]
```

### 2.2 Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CONTAINER DIAGRAM                                 │
│                            {SYSTEM_NAME}                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   API Gateway   │    │  Core Engine    │    │   Worker Pool   │         │
│  │   [Container]   │───▶│   [Container]   │───▶│   [Container]   │         │
│  │                 │    │                 │    │                 │         │
│  │ Tech: FastAPI   │    │ Tech: Python    │    │ Tech: Celery/   │         │
│  │ Port: 8000      │    │ AsyncIO         │    │ Multiprocessing │         │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘         │
│           │                      │                      │                   │
│           │                      ▼                      │                   │
│           │             ┌─────────────────┐             │                   │
│           │             │  Message Queue  │             │                   │
│           └────────────▶│   [Container]   │◀────────────┘                   │
│                         │ Tech: Redis/    │                                 │
│                         │ RabbitMQ        │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                          │
│                                  ▼                                          │
│                         ┌─────────────────┐                                 │
│                         │   Data Store    │                                 │
│                         │   [Container]   │                                 │
│                         │ Tech: PostgreSQL│                                 │
│                         │ / SQLite        │                                 │
│                         └─────────────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Containers:
| Container     | Technology    | Purpose                      | Port/Protocol |
|---------------|---------------|------------------------------|---------------|
| API Gateway   | {FRAMEWORK}   | External interface           | {PORT}        |
| Core Engine   | Python/Async  | Business logic orchestration | Internal      |
| Worker Pool   | {WORKER_TECH} | Parallel task execution      | Internal      |
| Message Queue | {QUEUE_TECH}  | Async communication          | {QUEUE_PORT}  |
| Data Store    | {DB_TECH}     | Persistence layer            | {DB_PORT}     |
```

### 2.3 Level 3: Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPONENT DIAGRAM                                   │
│                          Core Engine Container                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         {PACKAGE_NAME}/                                │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   config/   │  │    core/    │  │  services/  │  │   utils/    │  │  │
│  │  │             │  │             │  │             │  │             │  │  │
│  │  │ - settings  │  │ - models    │  │ - handlers  │  │ - logging   │  │  │
│  │  │ - env_vars  │  │ - schemas   │  │ - workers   │  │ - helpers   │  │  │
│  │  │ - constants │  │ - entities  │  │ - pipelines │  │ - validators│  │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │  │
│  │         │                │                │                │         │  │
│  │         └────────────────┴────────┬───────┴────────────────┘         │  │
│  │                                   │                                   │  │
│  │                          ┌────────▼────────┐                         │  │
│  │                          │   orchestrator/ │                         │  │
│  │                          │                 │                         │  │
│  │                          │ - coordinator   │                         │  │
│  │                          │ - scheduler     │                         │  │
│  │                          │ - state_machine │                         │  │
│  │                          └─────────────────┘                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                              tests/                                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │    unit/    │  │ integration/│  │     e2e/    │  │  fixtures/  │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Components:
| Component    | Responsibility                    | Dependencies           |
|--------------|-----------------------------------|------------------------|
| config       | Configuration management          | pydantic, pyyaml       |
| core         | Domain models and business logic  | config                 |
| services     | Service layer and handlers        | core, utils            |
| utils        | Cross-cutting utilities           | None (leaf module)     |
| orchestrator | Workflow coordination             | All above              |
```

### 2.4 Level 4: Code Diagram (Key Classes)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             CODE DIAGRAM                                     │
│                        Key Classes and Interfaces                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        <<interface>>                                 │    │
│  │                        IComponent                                    │    │
│  │  ─────────────────────────────────────────────────────────────────  │    │
│  │  + initialize() -> None                                             │    │
│  │  + execute(context: Context) -> Result                              │    │
│  │  + shutdown() -> None                                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    △                                         │
│                                    │                                         │
│          ┌─────────────────────────┼─────────────────────────┐              │
│          │                         │                         │              │
│  ┌───────┴───────┐        ┌───────┴───────┐        ┌───────┴───────┐       │
│  │  BaseComponent │        │ AsyncComponent │        │ PooledComponent│       │
│  │───────────────│        │───────────────│        │───────────────│       │
│  │ - config      │        │ - event_loop  │        │ - pool_size   │       │
│  │ - logger      │        │ - semaphore   │        │ - workers     │       │
│  │───────────────│        │───────────────│        │───────────────│       │
│  │ + validate()  │        │ + await_ready │        │ + scale()     │       │
│  │ + health()    │        │ + cancel()    │        │ + distribute()│       │
│  └───────────────┘        └───────────────┘        └───────────────┘       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          Factory<T>                                  │    │
│  │  ─────────────────────────────────────────────────────────────────  │    │
│  │  + register(name: str, cls: Type[T]) -> None                        │    │
│  │  + create(name: str, **kwargs) -> T                                 │    │
│  │  + list_registered() -> List[str]                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## SECTION 3: FEASIBLE REGION (Constraints)

### 3.1 Hard Constraints (Violations = Failure)

```yaml
language:
  runtime: "Python 3.10+"
  type_hints: required

dependencies:
  required:
    - pydantic >= 2.0        # Configuration and validation
    - pyyaml >= 6.0          # YAML configuration files
    - pytest >= 7.0          # Testing framework
    - structlog >= 23.0      # Structured logging
  optional:
    - redis >= 4.0           # Caching and queues
    - sqlalchemy >= 2.0      # Database ORM

security:
  - NO hardcoded secrets (use environment variables or secret managers)
  - ALL user inputs validated via Pydantic models
  - API keys loaded from environment only
  - Sensitive data never logged

compatibility:
  - Must run on CPU (GPU optional)
  - Cross-platform: Windows, Linux, macOS
  - Docker-compatible
  - CI/CD pipeline ready
```

### 3.2 Soft Constraints (Preferences)

```yaml
style:
  linter: ruff
  line_length: 100
  formatter: black

architecture:
  - Prefer composition over inheritance
  - Use dataclasses/Pydantic for data objects
  - Async I/O for all external calls
  - Type hints on ALL public interfaces
  - Dependency injection for testability

performance:
  - Batch operations where possible
  - Connection pooling for databases
  - Lazy loading for expensive resources
  - Caching with TTL for repeated queries

testing:
  framework: pytest
  coverage_threshold: 80
  patterns:
    - Unit tests for all public methods
    - Integration tests for cross-module flows
    - Fixtures for shared test data
    - Mocks for external dependencies

logging:
  library: structlog
  levels: [DEBUG, INFO, WARNING, ERROR, CRITICAL]
  format: JSON in production, human-readable in development
  context: Request ID, user ID, operation name
```

### 3.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Restructure existing file organization for better modularity
- Add dependencies from PyPI (update requirements.txt and pyproject.toml)
- Refactor adjacent code for consistency
- Choose implementation patterns not explicitly specified
- Create new test files and fixtures
- Add new configuration parameters
- Implement helper functions and utilities
- Add logging statements for debugging
- Create new submodules within the package
- Choose between sync/async based on context
- Select appropriate data structures
```

---

## SECTION 4: PERMISSION ARCHITECTURE

### 4.1 Scope (What You Can Touch)

```
IN SCOPE:
├── {PACKAGE_NAME}/          # All source code
│   ├── __init__.py
│   ├── config/              # Configuration management
│   ├── core/                # Core business logic
│   ├── services/            # Service layer
│   ├── utils/               # Utilities
│   └── orchestrator/        # Orchestration layer
├── tests/                   # All test code
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   ├── fixtures/
│   └── conftest.py
├── configs/                 # Configuration files
├── scripts/                 # Utility scripts
├── requirements.txt         # Dependencies
├── pyproject.toml          # Package configuration
├── setup.py                # Legacy setup (if needed)
├── CLAUDE.md               # This file
└── README.md               # Documentation

OUT OF SCOPE:
├── .git/                   # Git internals
├── __pycache__/            # Python cache
├── .env                    # Secrets (read-only)
├── venv/                   # Virtual environment
├── .venv/                  # Virtual environment (alternate)
├── dist/                   # Build artifacts
├── *.egg-info/             # Package metadata
└── node_modules/           # Node dependencies (if any)
```

### 4.2 Autonomy Level

```yaml
AUTONOMOUS (proceed without asking):
  - File creation/deletion within scope
  - Dependency installation via pip
  - Running tests and linters
  - Code refactoring within existing patterns
  - Adding logging and debug statements
  - Creating new test files
  - Updating configuration defaults
  - Documentation updates within scope
  - Bug fixes in implementation
  - Adding type hints
  - Creating utility functions

CONFIRM FIRST (ask before proceeding):
  - Architectural changes affecting >3 modules
  - Breaking changes to public APIs
  - Deletions of >100 lines of code
  - Changes to security-sensitive code
  - Database schema modifications
  - External service integrations
  - Changes to CI/CD configuration
  - Modifications to deployment configs

PROHIBITED (do not attempt):
  - Commits directly to main/master branch
  - Modifying .env with real credentials
  - External API calls with side effects (except designated services)
  - Deleting production data or configs
  - Force pushing to any branch
  - Disabling security features
  - Bypassing authentication/authorization
```

### 4.3 Resource Budget

```yaml
iteration_limits:
  max_iterations_before_guidance: 5
  max_files_per_modification: 20
  max_lines_per_edit: 500

time_limits:
  exploration_timeout_minutes: 10
  test_execution_timeout_seconds: 300

compute_limits:
  max_parallel_tests: 4
  max_memory_mb: 4096
```

---

## SECTION 5: SUB-AGENT SPECIFICATIONS

### 5.1 Available Sub-Agents

```yaml
agents:
  Explore:
    description: "Fast codebase exploration and search"
    capabilities:
      - File pattern matching (glob)
      - Code content search (grep)
      - Codebase structure analysis
      - Dependency tracing
    use_when:
      - "Finding files by pattern"
      - "Searching for code keywords"
      - "Understanding codebase structure"
      - "Tracing imports and dependencies"
    thoroughness_levels:
      - quick: "Basic searches, first matches"
      - medium: "Moderate exploration, related files"
      - very thorough: "Comprehensive analysis, all locations"

  Plan:
    description: "Software architect for implementation planning"
    capabilities:
      - Design implementation strategies
      - Identify critical files and dependencies
      - Consider architectural trade-offs
      - Create step-by-step plans
    use_when:
      - "Complex multi-file changes"
      - "New feature implementation"
      - "Architectural decisions"
      - "Refactoring strategies"
    output: "Step-by-step plan with file list and considerations"

  Bash:
    description: "Command execution specialist"
    capabilities:
      - Git operations
      - Package management (pip, npm)
      - Running tests and linters
      - System commands
    use_when:
      - "Git commits and pushes"
      - "Installing dependencies"
      - "Running verification commands"
      - "Build and deploy operations"
    restrictions:
      - "No interactive commands"
      - "No commands requiring user input"

  general-purpose:
    description: "Complex multi-step task handler"
    capabilities:
      - Multi-step research
      - Code search and analysis
      - File reading and exploration
      - Complex query resolution
    use_when:
      - "Multi-step investigations"
      - "Complex code searches"
      - "When unsure of file locations"
      - "Research tasks"
```

### 5.2 Agent Orchestration Patterns

```python
# Pattern 1: Parallel Exploration
# Use when gathering independent information
"""
Launch multiple Explore agents in parallel:
- Agent 1: Find all configuration files
- Agent 2: Find all test files
- Agent 3: Search for usage patterns
"""

# Pattern 2: Plan then Execute
# Use for complex implementations
"""
1. Launch Plan agent to design approach
2. Review plan output
3. Execute implementation steps
4. Launch Bash agent for verification
"""

# Pattern 3: Iterative Refinement
# Use when initial approach needs adjustment
"""
1. Explore to understand current state
2. Implement initial solution
3. Test and gather feedback
4. Refine based on results
"""

# Pattern 4: Research then Implement
# Use when domain knowledge is needed
"""
1. general-purpose agent for research
2. Plan agent for strategy
3. Direct implementation
4. Bash for verification
"""
```

### 5.3 Agent Selection Matrix

```
┌─────────────────────────────────┬──────────┬──────┬──────┬─────────────┐
│ Task Type                       │ Explore  │ Plan │ Bash │ General-Purp│
├─────────────────────────────────┼──────────┼──────┼──────┼─────────────┤
│ Find specific file              │    ✓     │      │      │             │
│ Search code patterns            │    ✓     │      │      │             │
│ Understand architecture         │    ✓     │  ✓   │      │             │
│ Design new feature              │          │  ✓   │      │             │
│ Plan refactoring                │          │  ✓   │      │             │
│ Run tests                       │          │      │  ✓   │             │
│ Git operations                  │          │      │  ✓   │             │
│ Install dependencies            │          │      │  ✓   │             │
│ Complex investigation           │    ✓     │      │      │      ✓      │
│ Multi-step research             │          │      │      │      ✓      │
│ Unknown file locations          │    ✓     │      │      │      ✓      │
└─────────────────────────────────┴──────────┴──────┴──────┴─────────────┘
```

---

## SECTION 6: FEEDBACK LOOP SPECIFICATION

### 6.1 Verification Commands

```bash
# After writing code, run in this order:

# 1. Format check (optional auto-fix)
ruff format --check {PACKAGE_NAME}/ tests/

# 2. Lint check
ruff check {PACKAGE_NAME}/ tests/

# 3. Type check
mypy {PACKAGE_NAME}/ --ignore-missing-imports

# 4. Unit tests (fast feedback)
pytest tests/unit/ -v --tb=short

# 5. Integration tests
pytest tests/integration/ -v --tb=short

# 6. Full test suite with coverage
pytest tests/ -v --cov={PACKAGE_NAME} --cov-report=term-missing

# 7. Coverage threshold check
pytest tests/ --cov={PACKAGE_NAME} --cov-fail-under={COVERAGE_THRESHOLD}

# 8. Smoke test
python -m {PACKAGE_NAME} --dry_run
```

### 6.2 Error Handling Protocol

```yaml
ON_LINT_FAILURE:
  steps:
    - Read error output carefully
    - Apply ruff --fix for auto-fixable issues
    - Manually fix remaining issues
    - Re-run: ruff check {PACKAGE_NAME}/
  max_attempts: 3

ON_TYPE_ERROR:
  steps:
    - Analyze mypy error message
    - Check if Optional or Union type needed
    - Add/fix type annotations
    - Consider if TYPE_CHECKING import needed
    - Re-run: mypy {PACKAGE_NAME}/
  max_attempts: 3

ON_TEST_FAILURE:
  steps:
    - Read full traceback
    - Identify root cause (implementation vs test bug)
    - If implementation bug: fix source code
    - If test bug: fix test (document justification)
    - Re-run specific test: pytest path/to/test.py::test_name -v
  max_attempts: 3

ON_IMPORT_ERROR:
  steps:
    - Check if dependency in requirements.txt
    - If missing: add and pip install -r requirements.txt
    - If version conflict: resolve in requirements.txt
    - Check for circular imports
  max_attempts: 2

ON_REPEATED_FAILURE:
  condition: "Same error 3 times"
  steps:
    - Stop attempting automatic fix
    - Document what was tried
    - Report analysis with hypothesis
    - Request human guidance
```

### 6.3 Success Verification Checklist

```yaml
before_reporting_completion:
  verification:
    - All lint checks pass
    - All type checks pass
    - All tests pass
    - Coverage meets threshold
    - Smoke test succeeds

  documentation:
    - CLAUDE.md updated if new patterns discovered
    - Code comments for non-obvious logic
    - Docstrings for public APIs

  quality:
    - No hardcoded values (check for magic numbers, paths, keys)
    - No security vulnerabilities introduced
    - Error handling for edge cases
    - Logging for debugging

  deliverables:
    - Summary of changes made
    - List of files modified
    - Known limitations documented
    - Follow-up tasks identified (if any)
```

---

## SECTION 7: CONTEXT PERSISTENCE

### 7.1 Session Memory Updates

When making changes, update the relevant sections below:

#### Build Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/unit/ -v` | Run unit tests |
| `pytest tests/integration/ -v` | Run integration tests |
| `pytest tests/ -v --tb=short` | Run all tests |
| `pytest tests/ -v --cov={PACKAGE_NAME}` | Run tests with coverage |
| `ruff check {PACKAGE_NAME}/` | Lint Python code |
| `ruff format {PACKAGE_NAME}/` | Format Python code |
| `mypy {PACKAGE_NAME}/` | Type check |
| `python -m {PACKAGE_NAME} --dry_run` | Smoke test |
| `pip install -e .` | Install package in editable mode |
| `pip install -e ".[dev]"` | Install with dev dependencies |

#### Test Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/unit/test_{module}.py -v` | Test specific module |
| `pytest -k "test_name" -v` | Run test by name pattern |
| `pytest -x` | Stop on first failure |
| `pytest --lf` | Run last failed tests |
| `pytest --pdb` | Debug on failure |

### 7.2 Architecture Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| {DATE} | {Decision description} | {Why this choice was made} |

### 7.3 Known Issues

| Issue | Workaround | Status |
|-------|------------|--------|
| {Issue description} | {How to work around it} | {Open/Resolved} |

### 7.4 Information to Preserve

```
- Build/test commands that work
- Non-obvious environment setup steps
- Architectural decisions and their rationale
- Gotchas discovered during implementation
- API keys location (never the keys themselves)
- External service dependencies
```

---

## SECTION 8: EXECUTION PROTOCOL

### 8.1 Initial Actions (Always Do First)

```bash
# 1. Read project configuration
cat CLAUDE.md  # or equivalent

# 2. Check environment
python --version  # Verify Python 3.10+

# 3. Install dependencies if needed
pip install -r requirements.txt
pip install -e ".[dev]"

# 4. Scan project structure
find . -type f -name "*.py" | grep -v __pycache__ | grep -v venv | head -50

# 5. Run existing tests to establish baseline
pytest tests/ -v --tb=line | head -50

# 6. Check for any failing tests
pytest tests/ -v --tb=short 2>&1 | grep -E "(PASSED|FAILED|ERROR)"
```

### 8.2 Implementation Order

```
1. UNDERSTAND: Read existing code before modifying
   - Identify patterns used in similar code
   - Check for existing utilities that can be reused
   - Review related tests for expected behavior
   - Use Explore agent for codebase navigation

2. PLAN: Design approach before coding
   - Use Plan agent for complex changes
   - Break into small, testable increments
   - Identify dependencies and order

3. IMPLEMENT: Write smallest working version first
   - Start with function signature and docstring
   - Implement core logic without edge cases
   - Add type hints immediately
   - Follow existing code patterns

4. TEST: Write tests alongside implementation
   - Unit test for each public method
   - Test edge cases and error conditions
   - Mock external dependencies
   - Use existing fixtures where possible

5. VERIFY: Run verification loop
   - ruff check (lint)
   - mypy (types)
   - pytest (tests)
   - Smoke test

6. REFACTOR: Clean up only if needed
   - Only refactor if tests pass
   - Keep changes minimal
   - Preserve existing APIs

7. DOCUMENT: Update context
   - Add new commands to build table
   - Document architectural decisions
   - Note any gotchas discovered
   - Update CLAUDE.md
```

### 8.3 Completion Checklist

```
□ All success criteria from Section 1.2 met
□ All verification commands from Section 6.1 pass
□ Coverage threshold met
□ CLAUDE.md updated with new commands/decisions
□ Summary of changes provided to user
□ Known limitations documented
□ No hardcoded values
□ No security vulnerabilities
□ Tests added for new functionality
□ Type hints on all new public interfaces
□ Logging added where appropriate
```

---

## SECTION 9: TESTING FRAMEWORK

### 9.1 Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests (isolated, fast)
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_core.py
│   ├── test_services.py
│   └── test_utils.py
├── integration/             # Integration tests (cross-module)
│   ├── __init__.py
│   ├── test_pipelines.py
│   └── test_workflows.py
├── e2e/                     # End-to-end tests (full system)
│   ├── __init__.py
│   └── test_scenarios.py
└── fixtures/                # Test data and mocks
    ├── __init__.py
    ├── sample_data.py
    └── mock_services.py
```

### 9.2 Testing Best Practices

```python
# 1. Use descriptive test names
def test_user_creation_with_valid_email_succeeds():
    pass

def test_user_creation_with_invalid_email_raises_validation_error():
    pass

# 2. Follow Arrange-Act-Assert pattern
def test_calculate_total():
    # Arrange
    items = [Item(price=10), Item(price=20)]
    calculator = Calculator()

    # Act
    total = calculator.calculate_total(items)

    # Assert
    assert total == 30

# 3. Use fixtures for shared setup
@pytest.fixture
def sample_config():
    return Config(
        debug=True,
        log_level="DEBUG",
    )

# 4. Parametrize for multiple cases
@pytest.mark.parametrize("input,expected", [
    ("user@example.com", True),
    ("invalid", False),
    ("", False),
])
def test_email_validation(input, expected):
    assert validate_email(input) == expected

# 5. Mark slow/integration tests
@pytest.mark.slow
@pytest.mark.integration
def test_full_pipeline():
    pass
```

### 9.3 Coverage Requirements

```yaml
coverage:
  minimum_total: 80
  minimum_per_module:
    core: 90
    services: 85
    utils: 75
    config: 80

  exclude_patterns:
    - "*/tests/*"
    - "*/__init__.py"
    - "*/migrations/*"
```

---

## SECTION 10: LOGGING AND DEBUGGING

### 10.1 Logging Configuration

```python
# Structured logging setup
import structlog

def configure_logging(
    level: str = "INFO",
    format: str = "json",  # "json" for production, "console" for development
    context: dict = None,
):
    """Configure structured logging for the application."""

    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

### 10.2 Logging Patterns

```python
# Get logger for module
logger = structlog.get_logger(__name__)

# Basic logging
logger.info("operation_started", operation="create_user", user_id=user_id)
logger.error("operation_failed", operation="create_user", error=str(e))

# Context binding
log = logger.bind(request_id=request_id, user_id=user_id)
log.info("processing_request")

# Timing operations
import time
start = time.perf_counter()
result = expensive_operation()
duration = time.perf_counter() - start
logger.info("operation_completed", duration_ms=duration * 1000)

# Exception logging
try:
    risky_operation()
except Exception as e:
    logger.exception("unexpected_error", error_type=type(e).__name__)
    raise
```

### 10.3 Debug Commands

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m {PACKAGE_NAME}

# Run with verbose output
pytest tests/ -v -s  # -s shows print statements

# Debug specific test
pytest tests/unit/test_core.py::test_specific -v --pdb

# Profile code
python -m cProfile -s cumtime -m {PACKAGE_NAME}

# Memory profiling
python -m memory_profiler script.py
```

---

## SECTION 11: CONFIGURATION MANAGEMENT

### 11.1 Configuration Hierarchy

```
Priority (highest to lowest):
1. Command-line arguments
2. Environment variables
3. .env file (local development)
4. Config files (YAML/JSON)
5. Default values in code
```

### 11.2 Configuration Schema

```python
from pydantic import BaseSettings, Field
from typing import Optional

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = Field(default="{PACKAGE_NAME}", env="APP_NAME")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    db_pool_size: int = Field(default=5, env="DB_POOL_SIZE")

    # External Services
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    api_timeout: int = Field(default=30, env="API_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
```

### 11.3 Environment Variables

```bash
# .env.template - Copy to .env and fill in values

# Application
APP_NAME={PACKAGE_NAME}
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/db

# External Services (DO NOT COMMIT REAL VALUES)
API_KEY=your_api_key_here
API_TIMEOUT=30

# Feature Flags
FEATURE_NEW_UI=false
FEATURE_BETA_API=false
```

---

## SECTION 12: PROJECT STRUCTURE

```
{PROJECT_NAME}/
├── {PACKAGE_NAME}/
│   ├── __init__.py
│   ├── __main__.py              # CLI entry point
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Pydantic settings
│   │   └── constants.py         # Application constants
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py            # Domain models
│   │   ├── schemas.py           # Pydantic schemas
│   │   └── exceptions.py        # Custom exceptions
│   ├── services/
│   │   ├── __init__.py
│   │   ├── handlers.py          # Request handlers
│   │   └── workers.py           # Background workers
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py           # Logging utilities
│   │   ├── helpers.py           # General helpers
│   │   └── validators.py        # Validation utilities
│   └── orchestrator/
│       ├── __init__.py
│       ├── coordinator.py       # Main orchestration
│       ├── scheduler.py         # Task scheduling
│       └── factories.py         # Component factories
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── configs/
│   ├── default.yaml
│   ├── development.yaml
│   └── production.yaml
├── scripts/
│   ├── setup.sh
│   └── run_tests.sh
├── .env.template
├── .gitignore
├── CLAUDE.md
├── README.md
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

---

## SECTION 13: TEMPLATE VARIABLES

Replace these placeholders when instantiating the template:

| Variable | Description | Example |
|----------|-------------|---------|
| `{SYSTEM_NAME}` | Full system name | "Distributed Task Queue" |
| `{PROJECT_NAME}` | Project directory name | "task-queue" |
| `{PACKAGE_NAME}` | Python package name | "task_queue" |
| `{COVERAGE_THRESHOLD}` | Minimum test coverage | "80" |
| `{FRAMEWORK}` | Web framework | "FastAPI" |
| `{PORT}` | Default port | "8000" |
| `{WORKER_TECH}` | Worker technology | "Celery" |
| `{QUEUE_TECH}` | Queue technology | "Redis" |
| `{QUEUE_PORT}` | Queue port | "6379" |
| `{DB_TECH}` | Database technology | "PostgreSQL" |
| `{DB_PORT}` | Database port | "5432" |
| `{DATE}` | Current date | "2026-01-23" |

---

## SECTION 14: QUICK REFERENCE

### Common Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"

# Development
ruff check --fix .
mypy {PACKAGE_NAME}/
pytest tests/ -v

# Running
python -m {PACKAGE_NAME}
python -m {PACKAGE_NAME} --dry_run
python -m {PACKAGE_NAME} --help
```

### File Templates

```python
# New module template
"""
Module description.

This module provides [functionality].
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

logger = logging.getLogger(__name__)


class ClassName:
    """Class description."""

    def __init__(self, config: Config) -> None:
        """Initialize with configuration."""
        self.config = config
        self._logger = logger.bind(component=self.__class__.__name__)

    def method_name(self, param: str) -> str:
        """Method description.

        Args:
            param: Parameter description.

        Returns:
            Return value description.

        Raises:
            ValueError: When param is invalid.
        """
        self._logger.debug("method_called", param=param)
        return param
```

---

*Template Version: 1.0.0*
*Last Updated: {DATE}*
