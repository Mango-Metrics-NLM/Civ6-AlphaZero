"""
Dynamic Template Generator for Agentic Coding System

Provides dynamic template generation with variable substitution,
ensuring no hardcoded values in the generated prompts.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import yaml

from ..utils.logging_config import get_logger

logger = get_logger("template_generator")

T = TypeVar("T")


class VariableType(Enum):
    """Types of template variables."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    PATH = "path"
    ENUM = "enum"
    CALLABLE = "callable"


@dataclass
class TemplateVariable:
    """Definition of a template variable.

    Attributes:
        name: Variable name used in templates (e.g., 'project_name')
        var_type: Type of the variable
        description: Human-readable description
        default: Default value if not provided
        required: Whether the variable must be provided
        validator: Optional validation function
        enum_values: Valid values for ENUM type
        transform: Optional transformation function
    """

    name: str
    var_type: VariableType
    description: str = ""
    default: Any = None
    required: bool = False
    validator: Optional[Callable[[Any], bool]] = None
    enum_values: list[str] = field(default_factory=list)
    transform: Optional[Callable[[Any], Any]] = None

    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a value against this variable definition.

        Args:
            value: Value to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if value is None:
            if self.required:
                return False, f"Required variable '{self.name}' is not provided"
            return True, ""

        # Type validation
        type_validators = {
            VariableType.STRING: lambda v: isinstance(v, str),
            VariableType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            VariableType.FLOAT: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            VariableType.BOOLEAN: lambda v: isinstance(v, bool),
            VariableType.LIST: lambda v: isinstance(v, list),
            VariableType.DICT: lambda v: isinstance(v, dict),
            VariableType.PATH: lambda v: isinstance(v, (str, Path)),
            VariableType.ENUM: lambda v: str(v) in self.enum_values,
            VariableType.CALLABLE: lambda v: callable(v),
        }

        validator = type_validators.get(self.var_type)
        if validator and not validator(value):
            if self.var_type == VariableType.ENUM:
                return False, f"Variable '{self.name}' must be one of: {self.enum_values}"
            return False, f"Variable '{self.name}' must be of type {self.var_type.value}"

        # Custom validation
        if self.validator and not self.validator(value):
            return False, f"Variable '{self.name}' failed custom validation"

        return True, ""

    def apply_transform(self, value: Any) -> Any:
        """Apply transformation to value if defined.

        Args:
            value: Value to transform.

        Returns:
            Transformed value.
        """
        if value is None:
            return self.default
        if self.transform:
            return self.transform(value)
        return value


@dataclass
class TemplateContext:
    """Context for template generation.

    Holds all variables and metadata needed for template rendering.
    """

    variables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_context: Optional[TemplateContext] = None

    # Standard metadata populated automatically
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    template_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable value with fallback to parent context.

        Args:
            key: Variable name.
            default: Default value if not found.

        Returns:
            Variable value or default.
        """
        if key in self.variables:
            return self.variables[key]
        if self.parent_context:
            return self.parent_context.get(key, default)
        return default

    def set(self, key: str, value: Any) -> None:
        """Set a variable value.

        Args:
            key: Variable name.
            value: Variable value.
        """
        self.variables[key] = value

    def update(self, values: dict[str, Any]) -> None:
        """Update multiple variables.

        Args:
            values: Dictionary of variable values.
        """
        self.variables.update(values)

    def merge(self, other: TemplateContext) -> TemplateContext:
        """Merge with another context, other takes precedence.

        Args:
            other: Context to merge with.

        Returns:
            New merged context.
        """
        return TemplateContext(
            variables={**self.variables, **other.variables},
            metadata={**self.metadata, **other.metadata},
            parent_context=self.parent_context,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "variables": self.variables,
            "metadata": self.metadata,
            "generated_at": self.generated_at,
            "template_id": self.template_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateContext:
        """Create context from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            TemplateContext instance.
        """
        return cls(
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {}),
        )


class TemplateGenerator:
    """Dynamic template generator with variable substitution.

    Generates agentic coding system prompt templates with no hardcoded values.
    All values are configurable through variables and context.

    Features:
    - Variable substitution with ${variable_name} syntax
    - Conditional sections with {%if condition%}...{%endif%}
    - Loop sections with {%for item in items%}...{%endfor%}
    - Template inheritance and composition
    - Validation of all variables
    - Logging and debugging support
    """

    # Regex patterns for template parsing
    VAR_PATTERN = re.compile(r"\$\{(\w+(?:\.\w+)*)\}")
    VAR_DEFAULT_PATTERN = re.compile(r"\$\{(\w+(?:\.\w+)*)\s*:\s*([^}]+)\}")
    CONDITIONAL_PATTERN = re.compile(r"\{%\s*if\s+(\w+)\s*%\}(.*?)\{%\s*endif\s*%\}", re.DOTALL)
    LOOP_PATTERN = re.compile(r"\{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%\}(.*?)\{%\s*endfor\s*%\}", re.DOTALL)

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        variable_definitions: Optional[list[TemplateVariable]] = None,
    ) -> None:
        """Initialize the template generator.

        Args:
            template_dir: Directory containing template files.
            variable_definitions: List of variable definitions for validation.
        """
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self._variable_defs: dict[str, TemplateVariable] = {}
        self._templates: dict[str, str] = {}
        self._helpers: dict[str, Callable[..., str]] = {}

        # Register default variables
        self._register_default_variables()

        # Register provided variable definitions
        if variable_definitions:
            for var_def in variable_definitions:
                self.register_variable(var_def)

        # Register default helpers
        self._register_default_helpers()

    def _register_default_variables(self) -> None:
        """Register default template variables."""
        defaults = [
            TemplateVariable(
                name="project_name",
                var_type=VariableType.STRING,
                description="Name of the project",
                required=True,
            ),
            TemplateVariable(
                name="system_intent",
                var_type=VariableType.STRING,
                description="System intent in one sentence",
                required=True,
            ),
            TemplateVariable(
                name="language_runtime",
                var_type=VariableType.STRING,
                description="Language and runtime requirements",
                default="Python 3.10+",
            ),
            TemplateVariable(
                name="working_directory",
                var_type=VariableType.PATH,
                description="Working directory path",
                transform=lambda p: str(Path(p).resolve()),
            ),
            TemplateVariable(
                name="success_criteria",
                var_type=VariableType.LIST,
                description="List of success criteria",
                required=True,
            ),
            TemplateVariable(
                name="hard_constraints",
                var_type=VariableType.LIST,
                description="List of hard constraints",
                default=[],
            ),
            TemplateVariable(
                name="soft_constraints",
                var_type=VariableType.LIST,
                description="List of soft constraints (preferences)",
                default=[],
            ),
            TemplateVariable(
                name="anti_constraints",
                var_type=VariableType.LIST,
                description="List of explicit permissions",
                default=[],
            ),
            TemplateVariable(
                name="in_scope_paths",
                var_type=VariableType.LIST,
                description="Paths that are in scope",
                default=["src/", "tests/"],
            ),
            TemplateVariable(
                name="out_of_scope_paths",
                var_type=VariableType.LIST,
                description="Paths that are out of scope",
                default=["vendor/", "node_modules/"],
            ),
            TemplateVariable(
                name="max_iterations",
                var_type=VariableType.INTEGER,
                description="Maximum iterations before requesting guidance",
                default=5,
                validator=lambda v: 1 <= v <= 20,
            ),
            TemplateVariable(
                name="max_files_per_pass",
                var_type=VariableType.INTEGER,
                description="Maximum files to modify in single pass",
                default=20,
                validator=lambda v: 1 <= v <= 100,
            ),
            TemplateVariable(
                name="exploration_timeout_minutes",
                var_type=VariableType.INTEGER,
                description="Timeout for exploration (minutes)",
                default=10,
                validator=lambda v: 1 <= v <= 60,
            ),
            TemplateVariable(
                name="verification_commands",
                var_type=VariableType.LIST,
                description="List of verification commands",
                default=[],
            ),
            TemplateVariable(
                name="memory_file",
                var_type=VariableType.STRING,
                description="Session memory file name",
                default="CLAUDE.md",
            ),
            TemplateVariable(
                name="test_command",
                var_type=VariableType.STRING,
                description="Command to run tests",
                default="pytest tests/ -v",
            ),
            TemplateVariable(
                name="lint_command",
                var_type=VariableType.STRING,
                description="Command to run linting",
                default="ruff check .",
            ),
            TemplateVariable(
                name="type_check_command",
                var_type=VariableType.STRING,
                description="Command to run type checking",
                default="mypy src/",
            ),
            TemplateVariable(
                name="sub_agents",
                var_type=VariableType.LIST,
                description="List of sub-agent configurations",
                default=[],
            ),
        ]

        for var_def in defaults:
            self._variable_defs[var_def.name] = var_def

    def _register_default_helpers(self) -> None:
        """Register default helper functions."""
        self._helpers["upper"] = lambda s: str(s).upper()
        self._helpers["lower"] = lambda s: str(s).lower()
        self._helpers["title"] = lambda s: str(s).title()
        self._helpers["join"] = lambda items, sep=", ": sep.join(str(i) for i in items)
        self._helpers["count"] = lambda items: len(items) if items else 0
        self._helpers["first"] = lambda items: items[0] if items else None
        self._helpers["last"] = lambda items: items[-1] if items else None
        self._helpers["indent"] = lambda s, n=2: "\n".join(" " * n + line for line in str(s).split("\n"))

    def register_variable(self, var_def: TemplateVariable) -> None:
        """Register a variable definition.

        Args:
            var_def: Variable definition to register.
        """
        self._variable_defs[var_def.name] = var_def
        logger.debug(f"Registered variable: {var_def.name}")

    def register_helper(self, name: str, func: Callable[..., str]) -> None:
        """Register a helper function.

        Args:
            name: Helper name.
            func: Helper function.
        """
        self._helpers[name] = func
        logger.debug(f"Registered helper: {name}")

    def register_template(self, name: str, template: str) -> None:
        """Register a template string.

        Args:
            name: Template name.
            template: Template content.
        """
        self._templates[name] = template
        logger.debug(f"Registered template: {name}")

    def load_template(self, file_path: Path) -> str:
        """Load a template from a file.

        Args:
            file_path: Path to template file.

        Returns:
            Template content.
        """
        with open(file_path, "r") as f:
            content = f.read()

        name = file_path.stem
        self._templates[name] = content
        logger.info(f"Loaded template: {name} from {file_path}")
        return content

    def validate_context(self, context: TemplateContext) -> list[str]:
        """Validate all variables in a context.

        Args:
            context: Context to validate.

        Returns:
            List of validation errors.
        """
        errors = []

        for name, var_def in self._variable_defs.items():
            value = context.get(name)
            is_valid, error = var_def.validate(value)
            if not is_valid:
                errors.append(error)

        return errors

    def _substitute_variables(self, template: str, context: TemplateContext) -> str:
        """Substitute variables in template.

        Args:
            template: Template string.
            context: Variable context.

        Returns:
            Template with substituted variables.
        """
        result = template

        # Handle variables with defaults first: ${var:default}
        def replace_with_default(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)
            value = context.get(var_name)
            if value is None:
                return default.strip()
            return self._format_value(value)

        result = self.VAR_DEFAULT_PATTERN.sub(replace_with_default, result)

        # Handle regular variables: ${var}
        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            value = context.get(var_name)
            if value is None:
                # Check if it's a nested path
                parts = var_name.split(".")
                if len(parts) > 1:
                    obj = context.get(parts[0])
                    for part in parts[1:]:
                        if isinstance(obj, dict):
                            obj = obj.get(part)
                        elif hasattr(obj, part):
                            obj = getattr(obj, part)
                        else:
                            obj = None
                            break
                    value = obj

            if value is None:
                logger.warning(f"Variable not found: {var_name}")
                return f"${{{var_name}}}"

            return self._format_value(value)

        result = self.VAR_PATTERN.sub(replace_var, result)

        return result

    def _process_conditionals(self, template: str, context: TemplateContext) -> str:
        """Process conditional sections.

        Args:
            template: Template string.
            context: Variable context.

        Returns:
            Template with processed conditionals.
        """
        def replace_conditional(match: re.Match[str]) -> str:
            condition_var = match.group(1)
            content = match.group(2)

            value = context.get(condition_var)
            if value:
                return content
            return ""

        return self.CONDITIONAL_PATTERN.sub(replace_conditional, template)

    def _process_loops(self, template: str, context: TemplateContext) -> str:
        """Process loop sections.

        Args:
            template: Template string.
            context: Variable context.

        Returns:
            Template with processed loops.
        """
        def replace_loop(match: re.Match[str]) -> str:
            item_name = match.group(1)
            list_name = match.group(2)
            content = match.group(3)

            items = context.get(list_name, [])
            if not items:
                return ""

            result_parts = []
            for item in items:
                # Create a sub-context with the loop variable
                loop_context = TemplateContext(
                    variables={item_name: item},
                    parent_context=context,
                )
                rendered = self._substitute_variables(content, loop_context)
                result_parts.append(rendered)

            return "".join(result_parts)

        return self.LOOP_PATTERN.sub(replace_loop, template)

    def _format_value(self, value: Any) -> str:
        """Format a value for template substitution.

        Args:
            value: Value to format.

        Returns:
            Formatted string.
        """
        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value)
        if isinstance(value, dict):
            return yaml.dump(value, default_flow_style=False)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def generate(
        self,
        template_name: str,
        context: Optional[TemplateContext] = None,
        **kwargs: Any,
    ) -> str:
        """Generate output from a template.

        Args:
            template_name: Name of the template to use.
            context: Optional context with variables.
            **kwargs: Additional variables to add to context.

        Returns:
            Generated template output.

        Raises:
            KeyError: If template not found.
            ValueError: If validation fails.
        """
        if template_name not in self._templates:
            raise KeyError(f"Template not found: {template_name}")

        # Create or extend context
        if context is None:
            context = TemplateContext(variables=kwargs)
        else:
            context.update(kwargs)

        # Validate context
        errors = self.validate_context(context)
        if errors:
            raise ValueError(f"Context validation failed: {'; '.join(errors)}")

        # Apply transforms to variables
        for name, var_def in self._variable_defs.items():
            value = context.get(name)
            transformed = var_def.apply_transform(value)
            if transformed != value:
                context.set(name, transformed)

        template = self._templates[template_name]

        # Process in order: loops, conditionals, variables
        result = self._process_loops(template, context)
        result = self._process_conditionals(result, context)
        result = self._substitute_variables(result, context)

        logger.info(f"Generated template: {template_name} (id: {context.template_id})")

        return result

    def generate_from_file(
        self,
        file_path: Path,
        context: Optional[TemplateContext] = None,
        **kwargs: Any,
    ) -> str:
        """Generate output from a template file.

        Args:
            file_path: Path to template file.
            context: Optional context with variables.
            **kwargs: Additional variables to add to context.

        Returns:
            Generated template output.
        """
        self.load_template(file_path)
        return self.generate(file_path.stem, context, **kwargs)

    def create_project_template(
        self,
        project_name: str,
        system_intent: str,
        success_criteria: list[str],
        **kwargs: Any,
    ) -> TemplateContext:
        """Create a standard project template context.

        Args:
            project_name: Name of the project.
            system_intent: System intent description.
            success_criteria: List of success criteria.
            **kwargs: Additional variables.

        Returns:
            Configured TemplateContext.
        """
        context = TemplateContext(
            variables={
                "project_name": project_name,
                "system_intent": system_intent,
                "success_criteria": success_criteria,
                **kwargs,
            },
            metadata={
                "created_by": "TemplateGenerator",
                "version": "1.0",
            },
        )

        return context


def create_template_generator(
    template_dir: Optional[Path] = None,
    extra_variables: Optional[list[TemplateVariable]] = None,
) -> TemplateGenerator:
    """Factory function to create a configured template generator.

    Args:
        template_dir: Directory containing template files.
        extra_variables: Additional variable definitions.

    Returns:
        Configured TemplateGenerator instance.
    """
    generator = TemplateGenerator(
        template_dir=template_dir,
        variable_definitions=extra_variables,
    )

    # Load built-in templates
    builtin_templates_dir = Path(__file__).parent / "builtin"
    if builtin_templates_dir.exists():
        for template_file in builtin_templates_dir.glob("*.md"):
            generator.load_template(template_file)
        for template_file in builtin_templates_dir.glob("*.yaml"):
            generator.load_template(template_file)

    return generator
