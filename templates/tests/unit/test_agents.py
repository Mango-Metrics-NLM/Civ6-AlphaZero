"""
Unit tests for the agents module.

Tests cover:
- AgentType enum and conversion
- AgentConfig creation and validation
- AgentRegistry functionality
- AgentOrchestrator planning and validation
"""
from __future__ import annotations

import pytest

from template_system.agents import (
    AGENT_SELECTION_MATRIX,
    AgentCapability,
    AgentConfig,
    AgentOrchestrator,
    AgentRegistry,
    AgentResult,
    AgentType,
    ModelType,
    OrchestrationPlan,
    Thoroughness,
    select_agent,
)


class TestAgentType:
    """Tests for AgentType enum."""

    def test_agent_type_values(self) -> None:
        """Test that all agent types have correct values."""
        assert AgentType.EXPLORE.value == "Explore"
        assert AgentType.PLAN.value == "Plan"
        assert AgentType.BASH.value == "Bash"
        assert AgentType.GENERAL_PURPOSE.value == "general-purpose"

    def test_from_string_exact_match(self) -> None:
        """Test creating AgentType from exact string."""
        assert AgentType.from_string("Explore") == AgentType.EXPLORE
        assert AgentType.from_string("Plan") == AgentType.PLAN
        assert AgentType.from_string("Bash") == AgentType.BASH

    def test_from_string_case_insensitive(self) -> None:
        """Test case-insensitive string conversion."""
        assert AgentType.from_string("explore") == AgentType.EXPLORE
        assert AgentType.from_string("EXPLORE") == AgentType.EXPLORE
        assert AgentType.from_string("ExPlOrE") == AgentType.EXPLORE

    def test_from_string_with_underscores(self) -> None:
        """Test string conversion with underscores."""
        assert AgentType.from_string("general_purpose") == AgentType.GENERAL_PURPOSE
        assert AgentType.from_string("GENERAL_PURPOSE") == AgentType.GENERAL_PURPOSE

    def test_from_string_invalid(self) -> None:
        """Test that invalid strings raise ValueError."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            AgentType.from_string("invalid")


class TestThoroughness:
    """Tests for Thoroughness enum."""

    def test_thoroughness_values(self) -> None:
        """Test thoroughness level values."""
        assert Thoroughness.QUICK.value == "quick"
        assert Thoroughness.MEDIUM.value == "medium"
        assert Thoroughness.VERY_THOROUGH.value == "very thorough"


class TestModelType:
    """Tests for ModelType enum."""

    def test_model_type_values(self) -> None:
        """Test model type values."""
        assert ModelType.SONNET.value == "sonnet"
        assert ModelType.OPUS.value == "opus"
        assert ModelType.HAIKU.value == "haiku"


class TestAgentCapability:
    """Tests for AgentCapability dataclass."""

    def test_capability_creation(self) -> None:
        """Test creating a capability."""
        cap = AgentCapability(
            name="test_cap",
            description="Test capability",
            required_tools=["Tool1", "Tool2"],
            constraints=["No interactive"],
        )
        assert cap.name == "test_cap"
        assert cap.description == "Test capability"
        assert len(cap.required_tools) == 2
        assert len(cap.constraints) == 1

    def test_capability_defaults(self) -> None:
        """Test default values."""
        cap = AgentCapability(name="minimal", description="Minimal")
        assert cap.required_tools == []
        assert cap.constraints == []


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_config_creation(self) -> None:
        """Test creating an agent config."""
        config = AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Test task",
            prompt="Find files",
        )
        assert config.agent_type == AgentType.EXPLORE
        assert config.description == "Test task"
        assert config.prompt == "Find files"

    def test_config_with_thoroughness(self) -> None:
        """Test config with thoroughness for Explore agent."""
        config = AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Test",
            prompt="Search",
            thoroughness=Thoroughness.VERY_THOROUGH,
        )
        assert config.thoroughness == Thoroughness.VERY_THOROUGH

    def test_config_thoroughness_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning when thoroughness used with non-Explore agent."""
        import logging
        caplog.set_level(logging.WARNING)

        AgentConfig(
            agent_type=AgentType.PLAN,
            description="Test",
            prompt="Plan",
            thoroughness=Thoroughness.MEDIUM,
        )
        assert "Thoroughness is only applicable" in caplog.text

    def test_to_task_params_basic(self) -> None:
        """Test converting config to task parameters."""
        config = AgentConfig(
            agent_type=AgentType.PLAN,
            description="Plan changes",
            prompt="Design the feature",
        )
        params = config.to_task_params()

        assert params["subagent_type"] == "Plan"
        assert params["description"] == "Plan changes"
        assert params["prompt"] == "Design the feature"
        assert "model" not in params
        assert "max_turns" not in params

    def test_to_task_params_with_model(self) -> None:
        """Test task params with model override."""
        config = AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Quick search",
            prompt="Find file",
            model=ModelType.HAIKU,
        )
        params = config.to_task_params()
        assert params["model"] == "haiku"

    def test_to_task_params_with_max_turns(self) -> None:
        """Test task params with max turns."""
        config = AgentConfig(
            agent_type=AgentType.GENERAL_PURPOSE,
            description="Research",
            prompt="Investigate",
            max_turns=10,
        )
        params = config.to_task_params()
        assert params["max_turns"] == 10

    def test_to_task_params_with_background(self) -> None:
        """Test task params for background execution."""
        config = AgentConfig(
            agent_type=AgentType.BASH,
            description="Run tests",
            prompt="pytest",
            run_in_background=True,
        )
        params = config.to_task_params()
        assert params["run_in_background"] is True

    def test_to_task_params_with_resume(self) -> None:
        """Test task params for resuming agent."""
        config = AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Continue",
            prompt="Continue search",
            resume_id="abc123",
        )
        params = config.to_task_params()
        assert params["resume"] == "abc123"

    def test_to_task_params_explore_with_thoroughness(self) -> None:
        """Test that thoroughness is included in Explore prompt."""
        config = AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Search",
            prompt="Find patterns",
            thoroughness=Thoroughness.VERY_THOROUGH,
        )
        params = config.to_task_params()
        assert "Thoroughness: very thorough" in params["prompt"]
        assert "Find patterns" in params["prompt"]


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_successful_result(self) -> None:
        """Test creating a successful result."""
        result = AgentResult(
            success=True,
            output="Found 5 files",
            agent_id="agent123",
        )
        assert result.success
        assert result.output == "Found 5 files"
        assert result.can_resume

    def test_failed_result(self) -> None:
        """Test creating a failed result."""
        result = AgentResult(
            success=False,
            error="Timeout exceeded",
        )
        assert not result.success
        assert result.error == "Timeout exceeded"
        assert not result.can_resume

    def test_background_result(self) -> None:
        """Test result from background agent."""
        result = AgentResult(
            success=True,
            output_file="/tmp/output.txt",
            agent_id="bg123",
        )
        assert result.output_file == "/tmp/output.txt"
        assert result.can_resume


class TestAgentRegistry:
    """Tests for AgentRegistry singleton."""

    def test_registry_singleton(self) -> None:
        """Test that registry is a singleton."""
        reg1 = AgentRegistry()
        reg2 = AgentRegistry()
        assert reg1 is reg2

    def test_get_capabilities(self) -> None:
        """Test getting capabilities for an agent type."""
        registry = AgentRegistry()
        caps = registry.get_capabilities(AgentType.EXPLORE)
        assert len(caps) > 0
        assert any(c.name == "file_search" for c in caps)

    def test_get_use_cases(self) -> None:
        """Test getting use cases for an agent type."""
        registry = AgentRegistry()
        use_cases = registry.get_use_cases(AgentType.PLAN)
        assert len(use_cases) > 0
        assert any("architectural" in uc.lower() for uc in use_cases)

    def test_register_capability(self) -> None:
        """Test registering a new capability."""
        registry = AgentRegistry()
        cap = AgentCapability(
            name="custom_cap",
            description="Custom capability",
        )
        registry.register_capability(AgentType.BASH, cap)
        caps = registry.get_capabilities(AgentType.BASH)
        assert any(c.name == "custom_cap" for c in caps)

    def test_suggest_agent_explore(self) -> None:
        """Test suggesting Explore agent for search tasks."""
        registry = AgentRegistry()
        agent = registry.suggest_agent("find all configuration files")
        assert agent == AgentType.EXPLORE

    def test_suggest_agent_plan(self) -> None:
        """Test suggesting Plan agent for design tasks."""
        registry = AgentRegistry()
        agent = registry.suggest_agent("design a new authentication system")
        assert agent == AgentType.PLAN

    def test_suggest_agent_bash(self) -> None:
        """Test suggesting Bash agent for execution tasks."""
        registry = AgentRegistry()
        agent = registry.suggest_agent("run the test suite")
        assert agent == AgentType.BASH

    def test_suggest_agent_no_match(self) -> None:
        """Test no suggestion for ambiguous tasks."""
        registry = AgentRegistry()
        agent = registry.suggest_agent("xyz abc 123")
        assert agent is None


class TestOrchestrationPlan:
    """Tests for OrchestrationPlan dataclass."""

    def test_add_step(self) -> None:
        """Test adding steps to a plan."""
        plan = OrchestrationPlan()
        config = AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Step 1",
            prompt="Search",
        )
        idx = plan.add_step(config)
        assert idx == 0
        assert len(plan.steps) == 1

    def test_add_step_with_dependencies(self) -> None:
        """Test adding steps with dependencies."""
        plan = OrchestrationPlan()
        idx1 = plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Step 1",
            prompt="Search",
        ))
        idx2 = plan.add_step(
            AgentConfig(
                agent_type=AgentType.PLAN,
                description="Step 2",
                prompt="Plan",
            ),
            depends_on=[idx1],
        )
        assert plan.dependencies[idx2] == [idx1]

    def test_add_parallel_group(self) -> None:
        """Test marking steps as parallelizable."""
        plan = OrchestrationPlan()
        plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="S1",
            prompt="P1",
        ))
        plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="S2",
            prompt="P2",
        ))
        plan.add_parallel_group([0, 1])
        assert [0, 1] in plan.parallel_groups


class TestAgentOrchestrator:
    """Tests for AgentOrchestrator class."""

    def test_create_plan(self) -> None:
        """Test creating a new plan."""
        orchestrator = AgentOrchestrator()
        plan = orchestrator.create_plan()
        assert isinstance(plan, OrchestrationPlan)
        assert len(plan.steps) == 0

    def test_suggest_workflow_implement(self) -> None:
        """Test workflow suggestion for implementation tasks."""
        orchestrator = AgentOrchestrator()
        plan = orchestrator.suggest_workflow("implement a new login feature")
        assert len(plan.steps) >= 2
        assert plan.steps[0].agent_type == AgentType.EXPLORE
        assert plan.steps[1].agent_type == AgentType.PLAN

    def test_suggest_workflow_search(self) -> None:
        """Test workflow suggestion for search tasks."""
        orchestrator = AgentOrchestrator()
        plan = orchestrator.suggest_workflow("find all test files")
        assert len(plan.steps) >= 1
        assert plan.steps[0].agent_type == AgentType.EXPLORE

    def test_suggest_workflow_run(self) -> None:
        """Test workflow suggestion for execution tasks."""
        orchestrator = AgentOrchestrator()
        plan = orchestrator.suggest_workflow("run pytest")
        assert len(plan.steps) >= 1
        assert plan.steps[0].agent_type == AgentType.BASH

    def test_validate_plan_empty(self) -> None:
        """Test validation fails for empty plan."""
        orchestrator = AgentOrchestrator()
        plan = OrchestrationPlan()
        errors = orchestrator.validate_plan(plan)
        assert "Plan has no steps" in errors

    def test_validate_plan_valid(self) -> None:
        """Test validation passes for valid plan."""
        orchestrator = AgentOrchestrator()
        plan = orchestrator.create_plan()
        plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Test",
            prompt="Search",
        ))
        errors = orchestrator.validate_plan(plan)
        assert len(errors) == 0

    def test_validate_plan_invalid_dependency(self) -> None:
        """Test validation catches invalid dependencies."""
        orchestrator = AgentOrchestrator()
        plan = OrchestrationPlan()
        plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Step 1",
            prompt="Search",
        ))
        plan.dependencies[0] = [1]  # Depends on non-existent step
        errors = orchestrator.validate_plan(plan)
        assert any("Invalid dependency" in e for e in errors)

    def test_validate_plan_no_prompt(self) -> None:
        """Test validation catches missing prompt."""
        orchestrator = AgentOrchestrator()
        plan = OrchestrationPlan()
        plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Test",
            prompt="",  # Empty prompt
        ))
        errors = orchestrator.validate_plan(plan)
        assert any("no prompt" in e for e in errors)

    def test_get_execution_order_sequential(self) -> None:
        """Test execution order for sequential steps."""
        orchestrator = AgentOrchestrator()
        plan = orchestrator.create_plan()
        idx1 = plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="S1",
            prompt="P1",
        ))
        plan.add_step(
            AgentConfig(
                agent_type=AgentType.PLAN,
                description="S2",
                prompt="P2",
            ),
            depends_on=[idx1],
        )
        order = orchestrator.get_execution_order(plan)
        assert len(order) == 2
        assert order[0] == [0]
        assert order[1] == [1]

    def test_get_execution_order_parallel(self) -> None:
        """Test execution order with parallel steps."""
        orchestrator = AgentOrchestrator()
        plan = orchestrator.create_plan()
        plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="S1",
            prompt="P1",
        ))
        plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="S2",
            prompt="P2",
        ))
        plan.add_parallel_group([0, 1])
        order = orchestrator.get_execution_order(plan)
        assert len(order) == 1
        assert set(order[0]) == {0, 1}

    def test_history_tracking(self) -> None:
        """Test execution history tracking."""
        orchestrator = AgentOrchestrator()
        assert len(orchestrator.history) == 0
        orchestrator.clear_history()
        assert len(orchestrator.history) == 0


class TestSelectAgent:
    """Tests for select_agent helper function."""

    def test_select_agent_find_file(self) -> None:
        """Test selecting agent for find_file task."""
        agents = select_agent("find_file")
        assert AgentType.EXPLORE in agents

    def test_select_agent_git_operations(self) -> None:
        """Test selecting agent for git operations."""
        agents = select_agent("git_operations")
        assert AgentType.BASH in agents

    def test_select_agent_unknown(self) -> None:
        """Test selecting agent for unknown task type."""
        agents = select_agent("unknown_task_type")
        assert AgentType.GENERAL_PURPOSE in agents


class TestAgentSelectionMatrix:
    """Tests for the agent selection matrix."""

    def test_matrix_contains_expected_tasks(self) -> None:
        """Test that matrix contains expected task types."""
        expected_tasks = [
            "find_file",
            "search_code",
            "run_tests",
            "git_operations",
        ]
        for task in expected_tasks:
            assert task in AGENT_SELECTION_MATRIX

    def test_matrix_values_are_lists(self) -> None:
        """Test that all matrix values are lists of AgentType."""
        for _task, agents in AGENT_SELECTION_MATRIX.items():
            assert isinstance(agents, list)
            for agent in agents:
                assert isinstance(agent, AgentType)
