# Agentic Framework for Claude Code

A dynamic, reusable framework for agentic coding workflows. Implements the Agentic Coding System Prompt Template with full support for constraint programming, specialized sub-agents, and iterative feedback loops.

## Features

- **Template Parsing**: Parse markdown or YAML templates defining objectives, constraints, and execution protocols
- **Constraint Management**: Hard constraints (violations = failure), soft constraints (preferences), and anti-constraints (explicit permissions)
- **Permission Architecture**: Scope checking, autonomy levels, and resource budgeting
- **Specialized Agents**: Planning, Implementation, Testing, Verification, and Documentation agents
- **Feedback Loop**: Verification command execution with error detection and auto-fixing
- **Session Memory**: Persistent memory across sessions via CLAUDE.md

## Quick Start

```python
from pathlib import Path
from agentic_framework import Orchestrator

# Create orchestrator from configuration file
orchestrator = Orchestrator(
    config_path=Path("config.yaml"),
    working_dir=Path.cwd(),
)

# Execute the workflow
result = orchestrator.execute(
    objective="Build a REST API for user management",
)

# Check results
if result.success:
    print("All tasks completed successfully!")
    print(result.change_summary)
else:
    print(f"Failed with errors: {result.errors}")
```

## Installation

```bash
# Install dependencies
pip install pyyaml

# Install the framework
pip install -e .
```

## Configuration

Create a `config.yaml` in your project root. See `config/example_project.yaml` for a complete example.

### Minimal Configuration

```yaml
version: "1.0"

objective:
  system_intent: "Build feature X"
  success_criteria:
    - criterion: "All tests pass"

permissions:
  scope:
    in_scope:
      - "src/**/*.py"
      - "tests/**/*.py"

feedback:
  verification_commands:
    - name: "tests"
      command: "pytest tests/ -v"
```

## Architecture

```
agentic_framework/
├── config/               # Configuration management
│   ├── config_loader.py  # YAML loading with env var substitution
│   └── default_config.yaml
├── core/                 # Core components
│   ├── template_parser.py    # Parse prompt templates
│   ├── constraint_manager.py # Manage constraints
│   ├── permission_manager.py # Manage permissions
│   ├── feedback_loop.py      # Verification loop
│   └── session_memory.py     # CLAUDE.md management
├── agents/               # Specialized agents
│   ├── base_agent.py         # Abstract base class
│   ├── planning_agent.py     # Codebase analysis
│   ├── implementation_agent.py # Code writing
│   ├── testing_agent.py      # Test execution
│   ├── verification_agent.py # Linting/type checking
│   └── documentation_agent.py # Documentation updates
├── utils/                # Utilities
│   ├── logging_config.py     # Structured logging
│   └── command_runner.py     # Shell execution
└── orchestrator.py       # Main orchestrator
```

## Agents

### Planning Agent
Analyzes codebase structure, identifies critical files, and creates implementation plans.

```python
from agentic_framework import PlanningAgent

agent = PlanningAgent(timeout_seconds=300)
result = agent.execute(context)

plan = result.data["plan"]
print(plan.render_markdown())
```

### Implementation Agent
Writes and modifies code with change tracking and rollback support.

```python
from agentic_framework import ImplementationAgent

agent = ImplementationAgent()

# Create a file
agent.create_file(context, path, content, description)

# Modify a file
agent.modify_file(context, path, transform_fn, description)

# Rollback all changes
agent.rollback_changes()
```

### Testing Agent
Discovers tests, runs test suites, and analyzes coverage.

```python
from agentic_framework import TestingAgent

agent = TestingAgent(coverage_threshold=80)
result = agent.execute(context)

print(f"Passed: {result.data['passed']}")
print(f"Coverage: {result.data['coverage']['total_coverage']}%")
```

### Verification Agent
Runs linting, type checking, and custom verification commands.

```python
from agentic_framework import VerificationAgent

agent = VerificationAgent(auto_fix=True)
result = agent.execute(context)

print(agent.get_issues_summary())
```

### Documentation Agent
Updates CLAUDE.md with build commands, architecture decisions, and known issues.

```python
from agentic_framework import DocumentationAgent

agent = DocumentationAgent()
agent.record_decision(context, "Use pytest", "Industry standard")
agent.record_issue(context, "Slow tests", "Use pytest-xdist")
```

## Callbacks

Register callbacks for progress tracking:

```python
orchestrator.set_callbacks(
    on_agent_start=lambda name: print(f"Starting: {name}"),
    on_agent_complete=lambda name, result: print(f"Completed: {name}"),
    on_progress=lambda name, pct, msg: print(f"{name}: {pct*100:.0f}% - {msg}"),
    on_confirmation_needed=lambda action, details: input(f"Confirm {action}? ") == "y",
)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest agentic_framework/tests/ -v

# Run unit tests only
pytest agentic_framework/tests/unit/ -v

# Run integration tests only
pytest agentic_framework/tests/integration/ -v

# Run with coverage
pytest agentic_framework/tests/ --cov=agentic_framework --cov-report=term-missing
```

## Logging

Configure logging via the config file or programmatically:

```python
from agentic_framework import setup_logging, get_logger

setup_logging(
    level="DEBUG",
    handlers_config=[
        {"type": "console", "level": "INFO"},
        {"type": "json", "level": "DEBUG", "destination": "logs/debug.json"},
    ],
)

logger = get_logger("my_module")
logger.info("Starting process", extra={"task_id": "123"})
```

## Environment Variables

Configuration supports environment variable substitution:

```yaml
database:
  host: "${DB_HOST:-localhost}"
  password: "${DB_PASSWORD}"  # Required
```

## Best Practices

1. **Start with Planning**: Always let the planning agent analyze before implementing
2. **Small Changes**: Prefer multiple small changes over large refactors
3. **Test First**: Ensure existing tests pass before making changes
4. **Document Decisions**: Record architecture decisions in session memory
5. **Use Constraints**: Define clear constraints to guide implementation

## License

MIT License
