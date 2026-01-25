"""
Comprehensive Test Suite for Agentic Template System

Tests all components of the template system including:
- Template generator with variable substitution
- Sub-agent configurations
- Verification pipeline
- Feedback handler
- Template logger
- Constraints
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from agentic_framework.templates import (
    # Template Generator
    TemplateGenerator,
    TemplateContext,
    TemplateVariable,
    VariableType,
    # Sub-Agents
    SubAgentConfig,
    SubAgentType,
    SubAgentRegistry,
    create_sub_agent_config,
    # Verification
    VerificationStep,
    VerificationPipeline,
    VerificationResult,
    VerificationType,
    VerificationStatus,
    create_verification_pipeline,
    # Feedback
    FeedbackHandler,
    FeedbackStrategy,
    ErrorPattern,
    RecoveryAction,
    # Logging
    TemplateLogger,
    LogLevel,
    DebugContext,
    setup_template_logging,
    # Constraints (including ConstraintCategory)
    ConstraintCategory,
    # Constraints
    Constraint,
    ConstraintType,
    ConstraintValidator,
    create_constraint,
)


# =============================================================================
# Template Generator Tests
# =============================================================================


class TestTemplateVariable:
    """Tests for TemplateVariable class."""

    def test_string_variable_validation(self) -> None:
        """Test string variable validation."""
        var = TemplateVariable(
            name="project_name",
            var_type=VariableType.STRING,
            required=True,
        )

        # Valid string
        is_valid, error = var.validate("my_project")
        assert is_valid is True
        assert error == ""

        # None with required=True
        is_valid, error = var.validate(None)
        assert is_valid is False
        assert "required" in error.lower()

        # Wrong type
        is_valid, error = var.validate(123)
        assert is_valid is False
        assert "type" in error.lower()

    def test_integer_variable_validation(self) -> None:
        """Test integer variable validation with custom validator."""
        var = TemplateVariable(
            name="max_iterations",
            var_type=VariableType.INTEGER,
            validator=lambda v: 1 <= v <= 20,
        )

        # Valid integer within range
        is_valid, _ = var.validate(5)
        assert is_valid is True

        # Boolean should not be valid as integer
        is_valid, _ = var.validate(True)
        assert is_valid is False

        # Out of range
        is_valid, _ = var.validate(100)
        assert is_valid is False

    def test_enum_variable_validation(self) -> None:
        """Test enum variable validation."""
        var = TemplateVariable(
            name="log_level",
            var_type=VariableType.ENUM,
            enum_values=["DEBUG", "INFO", "WARNING", "ERROR"],
        )

        is_valid, _ = var.validate("INFO")
        assert is_valid is True

        is_valid, _ = var.validate("INVALID")
        assert is_valid is False

    def test_variable_transform(self) -> None:
        """Test variable transformation."""
        var = TemplateVariable(
            name="path",
            var_type=VariableType.PATH,
            transform=lambda p: str(Path(p).resolve()),
        )

        result = var.apply_transform("./test")
        assert Path(result).is_absolute()

    def test_default_value(self) -> None:
        """Test default value handling."""
        var = TemplateVariable(
            name="optional",
            var_type=VariableType.STRING,
            default="default_value",
        )

        result = var.apply_transform(None)
        assert result == "default_value"


class TestTemplateContext:
    """Tests for TemplateContext class."""

    def test_basic_operations(self) -> None:
        """Test basic get/set operations."""
        ctx = TemplateContext()

        ctx.set("key", "value")
        assert ctx.get("key") == "value"
        assert ctx.get("nonexistent", "default") == "default"

    def test_update_multiple(self) -> None:
        """Test updating multiple values."""
        ctx = TemplateContext()
        ctx.update({"a": 1, "b": 2, "c": 3})

        assert ctx.get("a") == 1
        assert ctx.get("b") == 2
        assert ctx.get("c") == 3

    def test_parent_context_fallback(self) -> None:
        """Test parent context fallback."""
        parent = TemplateContext(variables={"inherited": "parent_value"})
        child = TemplateContext(
            variables={"own": "child_value"},
            parent_context=parent,
        )

        assert child.get("own") == "child_value"
        assert child.get("inherited") == "parent_value"

    def test_merge_contexts(self) -> None:
        """Test merging two contexts."""
        ctx1 = TemplateContext(variables={"a": 1, "b": 2})
        ctx2 = TemplateContext(variables={"b": 3, "c": 4})

        merged = ctx1.merge(ctx2)

        assert merged.get("a") == 1
        assert merged.get("b") == 3  # ctx2 takes precedence
        assert merged.get("c") == 4

    def test_serialization(self) -> None:
        """Test context serialization."""
        ctx = TemplateContext(
            variables={"key": "value"},
            metadata={"version": "1.0"},
        )

        data = ctx.to_dict()
        restored = TemplateContext.from_dict(data)

        assert restored.get("key") == "value"
        assert restored.metadata.get("version") == "1.0"


class TestTemplateGenerator:
    """Tests for TemplateGenerator class."""

    @pytest.fixture
    def generator(self) -> TemplateGenerator:
        """Create a generator instance."""
        return TemplateGenerator()

    def test_variable_substitution(self, generator: TemplateGenerator) -> None:
        """Test basic variable substitution."""
        generator.register_template(
            "test",
            "Project: ${project_name}, Language: ${language_runtime}",
        )

        result = generator.generate(
            "test",
            project_name="MyProject",
            system_intent="Build a test",
            success_criteria=["Tests pass"],
            language_runtime="Python 3.11",
        )

        assert "MyProject" in result
        assert "Python 3.11" in result

    def test_variable_with_default(self, generator: TemplateGenerator) -> None:
        """Test variable substitution with defaults."""
        generator.register_template(
            "test",
            "Value: ${optional_var:default_value}",
        )

        result = generator.generate(
            "test",
            project_name="Test",
            system_intent="Test",
            success_criteria=["Pass"],
        )

        assert "default_value" in result

    def test_conditional_sections(self, generator: TemplateGenerator) -> None:
        """Test conditional section processing."""
        generator.register_template(
            "test",
            "{%if include_debug%}DEBUG MODE{%endif%}Normal content",
        )

        # With condition true
        result = generator.generate(
            "test",
            project_name="Test",
            system_intent="Test",
            success_criteria=["Pass"],
            include_debug=True,
        )
        assert "DEBUG MODE" in result

        # With condition false
        result = generator.generate(
            "test",
            project_name="Test",
            system_intent="Test",
            success_criteria=["Pass"],
            include_debug=False,
        )
        assert "DEBUG MODE" not in result

    def test_loop_sections(self, generator: TemplateGenerator) -> None:
        """Test loop section processing."""
        generator.register_template(
            "test",
            "Items:{%for item in items%} ${item}{%endfor%}",
        )

        result = generator.generate(
            "test",
            project_name="Test",
            system_intent="Test",
            success_criteria=["Pass"],
            items=["one", "two", "three"],
        )

        assert "one" in result
        assert "two" in result
        assert "three" in result

    def test_validation_errors(self, generator: TemplateGenerator) -> None:
        """Test validation catches missing required variables."""
        generator.register_template("test", "Hello ${project_name}")

        with pytest.raises(ValueError) as exc_info:
            generator.generate(
                "test",
                system_intent="Test",
                success_criteria=["Pass"],
                # Missing project_name
            )

        assert "validation failed" in str(exc_info.value).lower()

    def test_custom_variable_registration(self, generator: TemplateGenerator) -> None:
        """Test registering custom variables."""
        generator.register_variable(
            TemplateVariable(
                name="custom_var",
                var_type=VariableType.STRING,
                required=True,
                description="A custom variable",
            )
        )

        generator.register_template("test", "${custom_var}")

        with pytest.raises(ValueError):
            generator.generate(
                "test",
                project_name="Test",
                system_intent="Test",
                success_criteria=["Pass"],
                # Missing custom_var
            )

    def test_helper_registration(self, generator: TemplateGenerator) -> None:
        """Test registering custom helpers."""
        generator.register_helper("reverse", lambda s: s[::-1])

        # Note: This test verifies registration, actual helper invocation
        # would require extended template syntax
        assert "reverse" in generator._helpers


# =============================================================================
# Sub-Agent Tests
# =============================================================================


class TestSubAgentConfig:
    """Tests for SubAgentConfig class."""

    def test_basic_config(self) -> None:
        """Test basic configuration."""
        config = SubAgentConfig(
            agent_type=SubAgentType.EXPLORE,
            name="Test Explorer",
            description="Test description",
            timeout_seconds=600.0,
        )

        assert config.agent_type == SubAgentType.EXPLORE
        assert config.name == "Test Explorer"
        assert config.timeout_seconds == 600.0

    def test_serialization(self) -> None:
        """Test configuration serialization."""
        config = SubAgentConfig(
            agent_type=SubAgentType.PLAN,
            name="Planner",
            tools=["Read", "Grep"],
            metadata={"key": "value"},
        )

        data = config.to_dict()
        restored = SubAgentConfig.from_dict(data)

        assert restored.agent_type == SubAgentType.PLAN
        assert restored.name == "Planner"
        assert "Read" in restored.tools
        assert restored.metadata.get("key") == "value"

    def test_with_overrides(self) -> None:
        """Test creating config with overrides."""
        original = SubAgentConfig(
            agent_type=SubAgentType.BASH,
            name="Original",
            timeout_seconds=300.0,
        )

        modified = original.with_overrides(timeout_seconds=600.0, name="Modified")

        assert modified.timeout_seconds == 600.0
        assert modified.name == "Modified"
        assert original.timeout_seconds == 300.0  # Original unchanged


class TestSubAgentRegistry:
    """Tests for SubAgentRegistry class."""

    @pytest.fixture
    def registry(self) -> SubAgentRegistry:
        """Create a fresh registry."""
        return SubAgentRegistry()

    def test_builtin_agents(self, registry: SubAgentRegistry) -> None:
        """Test builtin agents are registered."""
        agent_types = registry.list_agents()

        assert "explore" in agent_types
        assert "plan" in agent_types
        assert "bash" in agent_types
        assert "general-purpose" in agent_types

    def test_get_agent(self, registry: SubAgentRegistry) -> None:
        """Test retrieving an agent."""
        explore = registry.get(SubAgentType.EXPLORE)

        assert explore is not None
        assert explore.name == "Explore"
        assert "Glob" in explore.tools

    def test_register_custom_agent(self, registry: SubAgentRegistry) -> None:
        """Test registering a custom agent."""
        # Use an existing agent type for the custom config
        custom = SubAgentConfig(
            agent_type=SubAgentType.GENERAL_PURPOSE,
            name="Custom Agent",
            description="A custom agent",
        )

        # Register with a custom key
        registry._agents["custom_agent"] = custom

        retrieved = registry.get("custom_agent")
        assert retrieved is not None
        assert retrieved.name == "Custom Agent"

    def test_factory_function(self) -> None:
        """Test create_sub_agent_config factory."""
        config = create_sub_agent_config("explore", timeout_seconds=900.0)

        assert config.agent_type == SubAgentType.EXPLORE
        assert config.timeout_seconds == 900.0


# =============================================================================
# Verification Tests
# =============================================================================


class TestVerificationStep:
    """Tests for VerificationStep class."""

    def test_step_creation(self) -> None:
        """Test creating a verification step."""
        step = VerificationStep(
            name="lint",
            verification_type=VerificationType.LINT,
            command="ruff check .",
            timeout_seconds=120.0,
        )

        assert step.name == "lint"
        assert step.verification_type == VerificationType.LINT
        assert step.command == "ruff check ."

    def test_serialization(self) -> None:
        """Test step serialization."""
        step = VerificationStep(
            name="test",
            verification_type=VerificationType.UNIT_TEST,
            command="pytest",
            error_patterns=[r"FAILED.*"],
        )

        data = step.to_dict()
        restored = VerificationStep.from_dict(data)

        assert restored.name == "test"
        assert restored.command == "pytest"
        assert r"FAILED.*" in restored.error_patterns


class TestVerificationPipeline:
    """Tests for VerificationPipeline class."""

    @pytest.fixture
    def pipeline(self) -> VerificationPipeline:
        """Create a pipeline with mock steps."""
        # Using simple echo commands that will pass
        steps = [
            VerificationStep(
                name="step1",
                verification_type=VerificationType.CUSTOM,
                command="echo 'step 1 passed'",
                timeout_seconds=10.0,
            ),
            VerificationStep(
                name="step2",
                verification_type=VerificationType.CUSTOM,
                command="echo 'step 2 passed'",
                timeout_seconds=10.0,
            ),
        ]
        return VerificationPipeline(steps=steps)

    def test_add_remove_steps(self, pipeline: VerificationPipeline) -> None:
        """Test adding and removing steps."""
        new_step = VerificationStep(
            name="new_step",
            verification_type=VerificationType.CUSTOM,
            command="echo test",
        )

        pipeline.add_step(new_step)
        assert pipeline.get_step("new_step") is not None

        result = pipeline.remove_step("new_step")
        assert result is True
        assert pipeline.get_step("new_step") is None

    def test_disabled_step_skipped(self) -> None:
        """Test that disabled steps are skipped."""
        step = VerificationStep(
            name="disabled",
            verification_type=VerificationType.CUSTOM,
            command="echo test",
            enabled=False,
        )

        pipeline = VerificationPipeline(steps=[step])
        results = pipeline.run()

        assert len(results) == 1
        assert results[0].status == VerificationStatus.SKIPPED

    def test_summary_generation(self) -> None:
        """Test summary generation."""
        pipeline = VerificationPipeline()
        pipeline.add_step(
            VerificationStep(
                name="test",
                verification_type=VerificationType.CUSTOM,
                command="echo pass",
            )
        )

        pipeline.run()
        summary = pipeline.summary()

        assert "total_steps" in summary
        assert "executed" in summary
        assert "all_passed" in summary


# =============================================================================
# Feedback Handler Tests
# =============================================================================


class TestErrorPattern:
    """Tests for ErrorPattern class."""

    def test_pattern_matching(self) -> None:
        """Test error pattern matching."""
        pattern = ErrorPattern(
            name="lint_error",
            pattern=r"(?P<file>[^\s:]+):(?P<line>\d+):.*error",
            error_type="lint",
        )

        text = "src/main.py:42: error: Type mismatch"
        matches = pattern.match(text)

        assert len(matches) == 1
        assert matches[0]["file"] == "src/main.py"
        assert matches[0]["line"] == "42"

    def test_no_matches(self) -> None:
        """Test when pattern doesn't match."""
        pattern = ErrorPattern(
            name="test",
            pattern=r"FAILED.*",
        )

        matches = pattern.match("All tests passed!")
        assert len(matches) == 0

    def test_serialization(self) -> None:
        """Test pattern serialization."""
        pattern = ErrorPattern(
            name="test",
            pattern=r"error.*",
            strategy=FeedbackStrategy.AUTO_FIX,
            recovery_action=RecoveryAction.FIX_AND_RERUN,
        )

        data = pattern.to_dict()
        restored = ErrorPattern.from_dict(data)

        assert restored.name == "test"
        assert restored.strategy == FeedbackStrategy.AUTO_FIX


class TestFeedbackHandler:
    """Tests for FeedbackHandler class."""

    @pytest.fixture
    def handler(self) -> FeedbackHandler:
        """Create a feedback handler."""
        return FeedbackHandler()

    def test_process_output(self, handler: FeedbackHandler) -> None:
        """Test processing output for errors."""
        output = """
        src/main.py:10: error: Expected type 'int', got 'str'
        FAILED test_main.py::test_something
        """

        events = handler.process_output(output, source="test")

        assert len(events) > 0
        assert any("type_error" in (e.pattern_name or "") for e in events)

    def test_strategy_determination(self, handler: FeedbackHandler) -> None:
        """Test strategy determination for events."""
        handler.add_pattern(
            ErrorPattern(
                name="custom_error",
                pattern=r"CustomError.*",
                strategy=FeedbackStrategy.ESCALATE,
            )
        )

        events = handler.process_output("CustomError: something went wrong")

        assert len(events) == 1
        strategy = handler.determine_strategy(events[0])
        assert strategy == FeedbackStrategy.ESCALATE

    def test_occurrence_tracking(self, handler: FeedbackHandler) -> None:
        """Test that occurrences are tracked."""
        for _ in range(5):
            handler.process_output("src/test.py:1: error: test error")

        assert handler._occurrence_counts.get("type_error", 0) >= 5

    def test_clear_events(self, handler: FeedbackHandler) -> None:
        """Test clearing events."""
        handler.process_output("SyntaxError: invalid syntax")
        assert len(handler.events) > 0

        handler.clear_events()
        assert len(handler.events) == 0


# =============================================================================
# Template Logger Tests
# =============================================================================


class TestTemplateLogger:
    """Tests for TemplateLogger class."""

    @pytest.fixture
    def logger(self) -> TemplateLogger:
        """Create a logger instance."""
        return TemplateLogger(
            name="test",
            level=LogLevel.DEBUG,
            console_output=False,  # Don't clutter test output
        )

    def test_log_levels(self, logger: TemplateLogger) -> None:
        """Test different log levels."""
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")

        assert len(logger.get_entries()) == 4

    def test_level_filtering(self, logger: TemplateLogger) -> None:
        """Test log level filtering."""
        logger.set_level(LogLevel.WARNING)

        logger.debug("should not appear")
        logger.info("should not appear")
        logger.warning("should appear")
        logger.error("should appear")

        assert len(logger.get_entries()) == 2

    def test_context_manager(self, logger: TemplateLogger) -> None:
        """Test context manager for debugging."""
        with logger.context("test_context", template_id="123"):
            logger.info("inside context")

        entries = logger.get_entries()
        assert any("test_context" in e.message for e in entries)

    def test_timed_operation(self, logger: TemplateLogger) -> None:
        """Test timed operation logging."""
        with logger.timed("slow_operation"):
            time.sleep(0.1)

        entries = logger.get_entries()
        assert any(e.duration_ms is not None and e.duration_ms >= 100 for e in entries)

    def test_get_errors(self, logger: TemplateLogger) -> None:
        """Test getting error entries."""
        logger.info("info")
        logger.error("error1")
        logger.critical("critical1")

        errors = logger.get_errors()
        assert len(errors) == 2

    def test_summary(self, logger: TemplateLogger) -> None:
        """Test summary generation."""
        logger.info("info1")
        logger.info("info2")
        logger.error("error1")

        summary = logger.summary()

        assert summary["total_entries"] == 3
        assert summary["by_level"]["INFO"] == 2
        assert summary["by_level"]["ERROR"] == 1

    def test_file_output(self) -> None:
        """Test file output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = Path(f.name)

        try:
            logger = TemplateLogger(
                name="file_test",
                console_output=False,
                file_output=log_path,
            )

            logger.info("test message")
            logger.close()

            content = log_path.read_text()
            assert "test message" in content
        finally:
            log_path.unlink(missing_ok=True)


# =============================================================================
# Constraint Tests
# =============================================================================


class TestConstraint:
    """Tests for Constraint class."""

    def test_pattern_based_constraint(self) -> None:
        """Test pattern-based constraint."""
        constraint = Constraint(
            name="no_secrets",
            constraint_type=ConstraintType.HARD,
            pattern=r"password\s*=\s*['\"].*['\"]",
        )

        # Should fail - contains secret
        is_valid, violation = constraint.check("password = 'secret123'")
        assert is_valid is False
        assert violation is not None

        # Should pass - no secret
        is_valid, violation = constraint.check("password = os.environ['PASSWORD']")
        assert is_valid is True

    def test_function_based_constraint(self) -> None:
        """Test function-based constraint."""
        constraint = Constraint(
            name="min_coverage",
            constraint_type=ConstraintType.SOFT,
            check_function=lambda ctx: (
                ctx.get("coverage", 0) >= 80,
                f"Coverage {ctx.get('coverage', 0)}% is below 80%"
            ),
        )

        # Should pass
        is_valid, _ = constraint.check({"coverage": 90})
        assert is_valid is True

        # Should fail
        is_valid, violation = constraint.check({"coverage": 50})
        assert is_valid is False
        assert "50%" in violation.message

    def test_disabled_constraint(self) -> None:
        """Test disabled constraint."""
        constraint = Constraint(
            name="disabled",
            constraint_type=ConstraintType.HARD,
            pattern=r".*",  # Would match everything
            enabled=False,
        )

        is_valid, _ = constraint.check("any text")
        assert is_valid is True

    def test_serialization(self) -> None:
        """Test constraint serialization."""
        constraint = Constraint(
            name="test",
            constraint_type=ConstraintType.SOFT,
            category=ConstraintCategory.STYLE,
            severity=3,
        )

        data = constraint.to_dict()
        restored = Constraint.from_dict(data)

        assert restored.name == "test"
        assert restored.constraint_type == ConstraintType.SOFT
        assert restored.severity == 3


class TestConstraintValidator:
    """Tests for ConstraintValidator class."""

    @pytest.fixture
    def validator(self) -> ConstraintValidator:
        """Create a validator instance."""
        return ConstraintValidator()

    def test_validate_hard_constraints(self, validator: ConstraintValidator) -> None:
        """Test validating hard constraints."""
        violations = validator.validate(
            "password = 'secret'",
            constraint_type=ConstraintType.HARD,
        )

        # Should find the hardcoded secret
        assert len(violations) > 0
        assert any("no_hardcoded_secrets" in v.constraint_name for v in violations)

    def test_validate_soft_constraints(self, validator: ConstraintValidator) -> None:
        """Test validating soft constraints."""
        violations = validator.validate(
            "some code",
            constraint_type=ConstraintType.SOFT,
        )

        # Soft constraints may or may not produce violations
        # depending on the content - this just tests the filter works
        assert isinstance(violations, list)

    def test_validate_all(self, validator: ConstraintValidator) -> None:
        """Test validating all constraint types."""
        result = validator.validate_all("password = 'secret'")

        assert "hard" in result
        assert "soft" in result
        assert isinstance(result["hard"], list)

    def test_check_permissions(self, validator: ConstraintValidator) -> None:
        """Test checking permissions (anti-constraints)."""
        permissions = validator.check_permissions([
            "can_restructure_files",
            "can_add_dependencies",
            "nonexistent_permission",
        ])

        assert permissions.get("can_restructure_files") is True
        assert permissions.get("can_add_dependencies") is True
        assert permissions.get("nonexistent_permission") is False

    def test_enable_disable_constraint(self, validator: ConstraintValidator) -> None:
        """Test enabling and disabling constraints."""
        # Disable a constraint
        result = validator.enable_constraint("no_hardcoded_secrets", False)
        assert result is True

        # Should now pass validation
        violations = validator.validate("password = 'secret'")
        assert not any("no_hardcoded_secrets" in v.constraint_name for v in violations)

        # Re-enable
        validator.enable_constraint("no_hardcoded_secrets", True)

    def test_summary(self, validator: ConstraintValidator) -> None:
        """Test summary generation."""
        summary = validator.summary()

        assert "total" in summary
        assert "enabled" in summary
        assert "by_type" in summary
        assert "by_category" in summary


class TestConstraintFactory:
    """Tests for constraint factory functions."""

    def test_create_constraint(self) -> None:
        """Test create_constraint factory."""
        constraint = create_constraint(
            name="test",
            constraint_type="hard",
            description="Test constraint",
            severity=8,
        )

        assert constraint.name == "test"
        assert constraint.constraint_type == ConstraintType.HARD
        assert constraint.severity == 8


# =============================================================================
# Integration Tests
# =============================================================================


class TestTemplateSystemIntegration:
    """Integration tests for the complete template system."""

    def test_full_workflow(self) -> None:
        """Test a complete workflow with all components."""
        # 1. Create template generator
        generator = TemplateGenerator()
        generator.register_template(
            "full_test",
            """
Project: ${project_name}
Intent: ${system_intent}
{%if include_tests%}Tests: ${test_command}{%endif%}
""".strip(),
        )

        # 2. Create context
        context = generator.create_project_template(
            project_name="TestProject",
            system_intent="Build a test system",
            success_criteria=["All tests pass"],
            include_tests=True,
            test_command="pytest",
        )

        # 3. Generate template
        output = generator.generate("full_test", context)
        assert "TestProject" in output
        assert "pytest" in output

        # 4. Validate with constraints
        validator = ConstraintValidator()
        violations = validator.validate_all(output)

        # Should have no hard violations for this simple template
        assert len(violations["hard"]) == 0

    def test_sub_agent_with_verification(self) -> None:
        """Test sub-agent configuration with verification pipeline."""
        # Get a sub-agent config
        agent = create_sub_agent_config("test-runner")

        # Create a verification pipeline for the agent
        pipeline = VerificationPipeline()
        pipeline.add_step(
            VerificationStep(
                name="run_tests",
                verification_type=VerificationType.UNIT_TEST,
                command=agent.prompt_template,  # Just for testing
                enabled=False,  # Don't actually run
            )
        )

        results = pipeline.run()
        assert len(results) == 1
        assert results[0].status == VerificationStatus.SKIPPED

    def test_feedback_with_logging(self) -> None:
        """Test feedback handler with logging."""
        logger = TemplateLogger(
            name="feedback_test",
            level=LogLevel.DEBUG,
            console_output=False,
        )

        handler = FeedbackHandler()

        # Process output with pattern that matches (type_error pattern)
        with logger.context("error_processing"):
            # Use format that matches the type_error pattern: file:line: error: message
            events = handler.process_output(
                "src/main.py:42: error: Incompatible types in assignment"
            )
            logger.info(f"Processed {len(events)} events")

        assert len(logger.get_entries()) > 0
        assert len(events) > 0
