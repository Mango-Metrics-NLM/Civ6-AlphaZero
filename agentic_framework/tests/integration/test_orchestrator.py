"""
Integration tests for the orchestrator.
"""

import pytest

from agentic_framework import (
    Orchestrator,
    OrchestratorResult,
)


@pytest.fixture
def sample_project(tmp_path):
    """Create a sample project structure."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()

    # Create source files
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "main.py").write_text("""
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b

def multiply(a: int, b: int) -> int:
    \"\"\"Multiply two numbers.\"\"\"
    return a * b

if __name__ == "__main__":
    print(add(1, 2))
""")

    # Create test files
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "tests" / "test_main.py").write_text("""
from src.main import add, multiply

def test_add():
    assert add(1, 2) == 3

def test_multiply():
    assert multiply(2, 3) == 6
""")

    # Create config file
    config_content = """
version: "1.0"

objective:
  system_intent: "Test the sample project"
  success_criteria:
    - criterion: "All tests pass"
      verifiable: true
    - criterion: "No lint errors"
      verifiable: true

constraints:
  hard:
    - name: "python_version"
      type: "language"
      value: ">=3.10"
  soft:
    - name: "type_hints"
      type: "style"
      value: "Use type hints"
      priority: 8
  anti:
    - permission: "Restructure files"

permissions:
  scope:
    in_scope:
      - "src/**/*.py"
      - "tests/**/*.py"
    out_of_scope:
      - "__pycache__/**"
  autonomy:
    autonomous:
      - "running_tests"
      - "file_creation_in_scope"
    confirm_first:
      - "architectural_changes"
    prohibited:
      - "commits_to_main"
  resource_budget:
    max_iterations: 3
    max_files_per_pass: 10
    exploration_timeout_seconds: 300

feedback:
  verification_commands:
    - name: "unit_tests"
      command: "python -m pytest tests/ -v"
      timeout_seconds: 60
      retry_count: 1
  error_handling:
    auto_fix_patterns:
      - error_type: "test_failure"
        action: "analyze_fix"
        max_retries: 2
  success_verification:
    - check: "all_tests_pass"

session:
  memory_file: "CLAUDE.md"
  sections:
    - name: "Build Commands"
      persist: true
      format: "markdown_list"
    - name: "Test Commands"
      persist: true
      format: "markdown_list"

execution:
  initial_actions:
    - action: "read_session_memory"
      required: false
    - action: "scan_project_structure"
      command: "find . -type f -name '*.py' | head -20"
      required: true
  implementation_order:
    - phase: "understand"
      description: "Understand existing code"
      agent: "planning"
    - phase: "verify"
      description: "Run verifications"
      agent: "verification"
    - phase: "document"
      description: "Update documentation"
      agent: "documentation"
  completion_checklist:
    - item: "All verifications pass"
      verification: "run_all_verifications"
    - item: "Session memory updated"
      verification: "check_memory_updated"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - type: "console"
      level: "INFO"
"""

    (tmp_path / "config.yaml").write_text(config_content)

    return tmp_path


class TestOrchestratorBasic:
    """Basic tests for Orchestrator."""

    def test_orchestrator_initialization(self, sample_project):
        """Test orchestrator can be initialized."""
        orchestrator = Orchestrator(
            config_path=sample_project / "config.yaml",
            working_dir=sample_project,
        )

        assert orchestrator.config is not None
        assert orchestrator.working_dir == sample_project

    def test_orchestrator_from_template(self, tmp_path):
        """Test creating orchestrator from template file."""
        template_content = """
## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

I am building: A simple calculator module

### 1.2 Success Criteria

- [ ] All tests pass
- [ ] Code is type-checked
"""

        template_file = tmp_path / "template.md"
        template_file.write_text(template_content)

        orchestrator = Orchestrator.from_template(
            template_path=template_file,
            working_dir=tmp_path,
        )

        assert orchestrator is not None

    def test_orchestrator_registers_agents(self, sample_project):
        """Test orchestrator has all agents registered."""
        orchestrator = Orchestrator(
            config_path=sample_project / "config.yaml",
            working_dir=sample_project,
        )

        assert orchestrator.get_agent("planning") is not None
        assert orchestrator.get_agent("implementation") is not None
        assert orchestrator.get_agent("testing") is not None
        assert orchestrator.get_agent("verification") is not None
        assert orchestrator.get_agent("documentation") is not None


class TestOrchestratorExecution:
    """Tests for Orchestrator execution."""

    def test_dry_run(self, sample_project):
        """Test dry run mode only executes planning."""
        orchestrator = Orchestrator(
            config_path=sample_project / "config.yaml",
            working_dir=sample_project,
        )

        result = orchestrator.execute(dry_run=True)

        assert isinstance(result, OrchestratorResult)
        # In dry run, only planning should run
        agent_names = [r.agent_name for r in result.agent_results]
        assert "planning" in agent_names

    def test_run_single_agent(self, sample_project):
        """Test running a single agent."""
        orchestrator = Orchestrator(
            config_path=sample_project / "config.yaml",
            working_dir=sample_project,
        )

        result = orchestrator.run_single_agent("planning")

        assert result is not None
        assert result.agent_name == "planning"

    def test_callbacks_are_called(self, sample_project):
        """Test that callbacks are invoked during execution."""
        orchestrator = Orchestrator(
            config_path=sample_project / "config.yaml",
            working_dir=sample_project,
        )

        started_agents = []
        completed_agents = []

        orchestrator.set_callbacks(
            on_agent_start=lambda name: started_agents.append(name),
            on_agent_complete=lambda name, result: completed_agents.append(name),
        )

        orchestrator.execute(dry_run=True)

        assert len(started_agents) > 0
        assert len(completed_agents) > 0


class TestOrchestratorWithObjective:
    """Tests for Orchestrator with specific objectives."""

    def test_execute_with_objective(self, sample_project):
        """Test execution with custom objective."""
        orchestrator = Orchestrator(
            config_path=sample_project / "config.yaml",
            working_dir=sample_project,
        )

        result = orchestrator.execute(
            objective="Analyze and document the calculator module",
            dry_run=True,
        )

        assert result is not None


class TestOrchestratorResults:
    """Tests for Orchestrator results."""

    def test_result_summary(self, sample_project):
        """Test result summary generation."""
        orchestrator = Orchestrator(
            config_path=sample_project / "config.yaml",
            working_dir=sample_project,
        )

        result = orchestrator.execute(dry_run=True)

        summary = result.summary()

        assert "SUCCESS" in summary or "FAILED" in summary
        assert "Agents run" in summary

    def test_result_to_dict(self, sample_project):
        """Test converting result to dictionary."""
        orchestrator = Orchestrator(
            config_path=sample_project / "config.yaml",
            working_dir=sample_project,
        )

        result = orchestrator.execute(dry_run=True)

        result_dict = result.to_dict()

        assert "success" in result_dict
        assert "agent_results" in result_dict
        assert "total_duration_seconds" in result_dict
