"""
Tests for agents.

Note: basic_config and agent_context fixtures are provided by conftest.py.
"""

from agentic_framework.agents.base_agent import (
    AgentResult,
    AgentStatus,
    BaseAgent,
    CompositeAgent,
)
from agentic_framework.agents.documentation_agent import DocumentationAgent
from agentic_framework.agents.implementation_agent import ImplementationAgent
from agentic_framework.agents.planning_agent import PlanningAgent
from agentic_framework.agents.testing_agent import TestingAgent
from agentic_framework.agents.verification_agent import VerificationAgent


class TestBaseAgent:
    """Tests for BaseAgent functionality."""

    def test_agent_initialization(self):
        """Test basic agent initialization."""

        class TestAgent(BaseAgent):
            def _execute(self, context):
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.COMPLETED,
                    success=True,
                )

        agent = TestAgent(name="test", description="Test agent")

        assert agent.name == "test"
        assert agent.description == "Test agent"
        assert agent.status == AgentStatus.IDLE

    def test_agent_execution_success(self, agent_context):
        """Test successful agent execution."""

        class TestAgent(BaseAgent):
            def _execute(self, context):
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.COMPLETED,
                    success=True,
                    message="Test passed",
                )

        agent = TestAgent(name="test")
        result = agent.execute(agent_context)

        assert result.success
        assert result.status == AgentStatus.COMPLETED
        assert agent.status == AgentStatus.COMPLETED

    def test_agent_execution_failure(self, agent_context):
        """Test agent execution with failure."""

        class TestAgent(BaseAgent):
            def _execute(self, context):
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.FAILED,
                    success=False,
                    message="Test failed",
                )

        agent = TestAgent(name="test")
        result = agent.execute(agent_context)

        assert not result.success
        assert result.status == AgentStatus.FAILED

    def test_agent_exception_handling(self, agent_context):
        """Test agent handles exceptions gracefully."""

        class TestAgent(BaseAgent):
            def _execute(self, context):
                raise ValueError("Test error")

        agent = TestAgent(name="test")
        result = agent.execute(agent_context)

        assert not result.success
        assert agent.status == AgentStatus.FAILED
        assert "Test error" in result.errors[0]


class TestCompositeAgent:
    """Tests for CompositeAgent."""

    def test_composite_runs_all_agents(self, agent_context):
        """Test composite agent runs all sub-agents."""

        class SubAgent(BaseAgent):
            def __init__(self, name, should_succeed=True):
                super().__init__(name)
                self.should_succeed = should_succeed

            def _execute(self, context):
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.COMPLETED if self.should_succeed else AgentStatus.FAILED,
                    success=self.should_succeed,
                )

        agents = [SubAgent("sub1"), SubAgent("sub2"), SubAgent("sub3")]
        composite = CompositeAgent(name="composite", agents=agents)

        result = composite.execute(agent_context)

        assert result.success
        assert len(composite.results) == 3

    def test_composite_stops_on_failure(self, agent_context):
        """Test composite agent stops on failure when configured."""

        class SubAgent(BaseAgent):
            def __init__(self, name, should_succeed=True):
                super().__init__(name)
                self.should_succeed = should_succeed

            def _execute(self, context):
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.COMPLETED if self.should_succeed else AgentStatus.FAILED,
                    success=self.should_succeed,
                )

        agents = [SubAgent("sub1"), SubAgent("sub2", should_succeed=False), SubAgent("sub3")]
        composite = CompositeAgent(name="composite", agents=agents, stop_on_failure=True)

        result = composite.execute(agent_context)

        assert not result.success
        assert len(composite.results) == 2  # Stopped after failure


class TestPlanningAgent:
    """Tests for PlanningAgent."""

    def test_planning_agent_creates_plan(self, agent_context, tmp_path):
        """Test planning agent creates an implementation plan."""
        # Create some test files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main(): pass")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("def test_main(): pass")

        agent = PlanningAgent()
        result = agent.execute(agent_context)

        assert result.success
        assert "plan" in result.data
        assert result.next_agent == "implementation"

    def test_planning_agent_analyzes_structure(self, agent_context, tmp_path):
        """Test planning agent analyzes project structure."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "module.py").write_text("class MyClass: pass")

        agent = PlanningAgent()
        result = agent.execute(agent_context)

        structure = result.data.get("structure", {})
        assert "directories" in structure
        assert "file_types" in structure


class TestImplementationAgent:
    """Tests for ImplementationAgent."""

    def test_create_file(self, agent_context_with_mocks, tmp_path):
        """Test creating a new file."""
        agent = ImplementationAgent()

        new_file = tmp_path / "new_module.py"
        success = agent.create_file(
            agent_context_with_mocks,
            new_file,
            "def hello(): return 'world'",
            "Create hello function",
        )

        assert success
        assert new_file.exists()
        assert "hello" in new_file.read_text()

    def test_modify_file(self, agent_context_with_mocks, tmp_path):
        """Test modifying an existing file."""
        # Create initial file
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        agent = ImplementationAgent()

        success = agent.modify_file(
            agent_context_with_mocks,
            test_file,
            lambda content: content.replace("1", "2"),
            "Change value",
        )

        assert success
        assert "x = 2" in test_file.read_text()

    def test_delete_file(self, agent_context_with_mocks, tmp_path):
        """Test deleting a file."""
        test_file = tmp_path / "to_delete.py"
        test_file.write_text("delete me")

        agent = ImplementationAgent()

        success = agent.delete_file(agent_context_with_mocks, test_file, "Remove unused file")

        assert success
        assert not test_file.exists()

    def test_rollback_changes(self, agent_context_with_mocks, tmp_path):
        """Test rolling back changes."""
        test_file = tmp_path / "test.py"
        test_file.write_text("original")

        agent = ImplementationAgent()

        # Make a change
        agent.modify_file(
            agent_context_with_mocks,
            test_file,
            lambda content: "modified",
            "Modify content",
        )

        assert "modified" in test_file.read_text()

        # Rollback
        rolled_back = agent.rollback_changes()

        assert rolled_back == 1
        assert "original" in test_file.read_text()


class TestTestingAgent:
    """Tests for TestingAgent."""

    def test_discover_tests(self, agent_context, tmp_path):
        """Test discovering test files."""
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_module.py").write_text("def test_foo(): pass")
        (tmp_path / "test_other.py").write_text("def test_bar(): pass")

        agent = TestingAgent()
        test_files = agent._discover_tests(tmp_path)

        assert len(test_files) >= 2

    def test_parse_pytest_output(self):
        """Test parsing pytest output."""
        output = """
test_module.py::test_foo PASSED
test_module.py::test_bar FAILED
test_module.py::test_baz PASSED

=== 2 passed, 1 failed ===
"""

        agent = TestingAgent()
        result = agent._parse_pytest_output(output)

        assert result["passed"] == 2
        assert result["failed"] == 1
        assert not result["success"]


class TestVerificationAgent:
    """Tests for VerificationAgent."""

    def test_parse_lint_output(self):
        """Test parsing lint tool output."""
        output = """
src/module.py:10:5: E101 indentation contains mixed spaces and tabs
src/module.py:15:1: W291 trailing whitespace
"""

        agent = VerificationAgent()
        issues = agent._parse_lint_output(output)

        assert len(issues) == 2
        assert issues[0].severity == "error"
        assert issues[1].severity == "warning"

    def test_parse_type_output(self):
        """Test parsing type checker output."""
        output = """
src/module.py:10: error: Argument 1 to "foo" has incompatible type
src/module.py:15: note: Expected type "int"
"""

        agent = VerificationAgent()
        issues = agent._parse_type_output(output)

        assert len(issues) == 2
        assert issues[0].severity == "error"
        assert issues[1].severity == "info"


class TestDocumentationAgent:
    """Tests for DocumentationAgent."""

    def test_documentation_agent_updates_memory(self, agent_context, tmp_path):
        """Test documentation agent updates session memory."""
        # Initialize session memory
        agent_context.session_memory.create_default()

        agent = DocumentationAgent()
        result = agent.execute(agent_context)

        assert result.success
        assert "updates" in result.data

    def test_record_decision(self, agent_context):
        """Test recording architecture decision."""
        agent_context.session_memory.create_default()

        agent = DocumentationAgent()
        agent.record_decision(
            agent_context,
            "Use pytest for testing",
            "Industry standard, good documentation",
        )

        decisions = agent_context.get_shared("architecture_decisions", [])
        assert len(decisions) == 1

    def test_record_issue(self, agent_context):
        """Test recording known issue."""
        agent_context.session_memory.create_default()

        agent = DocumentationAgent()
        agent.record_issue(
            agent_context,
            "Tests slow on CI",
            "Use pytest-xdist for parallelization",
        )

        issues = agent_context.get_shared("known_issues", [])
        assert len(issues) == 1
