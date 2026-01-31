"""
Template Parser for Agentic Framework

Parses the Agentic Coding System Prompt Template format and converts
it into structured configuration that can be used by the framework.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from ..config import AgenticConfig
from ..utils.logging_config import get_logger

logger = get_logger("template_parser")


class TemplateParseError(Exception):
    """Raised when template parsing fails."""

    def __init__(self, message: str, section: str | None = None, line: int | None = None) -> None:
        self.section = section
        self.line = line
        super().__init__(f"{message} (section: {section}, line: {line})")


class TemplateSection(Enum):
    """Sections in the prompt template."""

    OBJECTIVE = "OBJECTIVE FUNCTION"
    CONSTRAINTS = "FEASIBLE REGION"
    PERMISSIONS = "PERMISSION ARCHITECTURE"
    FEEDBACK = "FEEDBACK LOOP SPECIFICATION"
    SESSION = "CONTEXT PERSISTENCE"
    EXECUTION = "EXECUTION PROTOCOL"


@dataclass
class ParsedObjective:
    """Parsed objective function section."""

    system_intent: str = ""
    success_criteria: list[dict[str, Any]] = field(default_factory=list)
    problem_statement: str = ""
    data_flows: str = ""
    failure_modes: str = ""


@dataclass
class ParsedConstraints:
    """Parsed constraints section."""

    hard: list[dict[str, Any]] = field(default_factory=list)
    soft: list[dict[str, Any]] = field(default_factory=list)
    anti: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ParsedPermissions:
    """Parsed permissions section."""

    in_scope: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    autonomous: list[str] = field(default_factory=list)
    confirm_first: list[str] = field(default_factory=list)
    prohibited: list[str] = field(default_factory=list)
    max_iterations: int = 5
    max_files_per_pass: int = 20
    exploration_timeout_seconds: int = 600


@dataclass
class ParsedFeedback:
    """Parsed feedback loop section."""

    verification_commands: list[dict[str, Any]] = field(default_factory=list)
    error_handling: dict[str, Any] = field(default_factory=dict)
    success_verification: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ParsedSession:
    """Parsed session/context persistence section."""

    memory_file: str = "CLAUDE.md"
    sections: list[dict[str, Any]] = field(default_factory=list)
    preserve_info: list[str] = field(default_factory=list)
    derivable_info: list[str] = field(default_factory=list)


@dataclass
class ParsedExecution:
    """Parsed execution protocol section."""

    initial_actions: list[dict[str, Any]] = field(default_factory=list)
    implementation_order: list[dict[str, Any]] = field(default_factory=list)
    completion_checklist: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ParsedTemplate:
    """Complete parsed template structure."""

    objective: ParsedObjective = field(default_factory=ParsedObjective)
    constraints: ParsedConstraints = field(default_factory=ParsedConstraints)
    permissions: ParsedPermissions = field(default_factory=ParsedPermissions)
    feedback: ParsedFeedback = field(default_factory=ParsedFeedback)
    session: ParsedSession = field(default_factory=ParsedSession)
    execution: ParsedExecution = field(default_factory=ParsedExecution)
    raw_content: str = ""
    source_file: Path | None = None

    def to_config(self) -> AgenticConfig:
        """Convert parsed template to AgenticConfig."""
        config_dict = {
            "version": "1.0",
            "objective": {
                "system_intent": self.objective.system_intent,
                "success_criteria": self.objective.success_criteria,
                "problem_description": {
                    "problem_statement": self.objective.problem_statement,
                    "data_flows": self.objective.data_flows,
                    "failure_modes": self.objective.failure_modes,
                },
            },
            "constraints": {
                "hard": self.constraints.hard,
                "soft": self.constraints.soft,
                "anti": self.constraints.anti,
            },
            "permissions": {
                "scope": {
                    "in_scope": self.permissions.in_scope,
                    "out_of_scope": self.permissions.out_of_scope,
                },
                "autonomy": {
                    "autonomous": self.permissions.autonomous,
                    "confirm_first": self.permissions.confirm_first,
                    "prohibited": self.permissions.prohibited,
                },
                "resource_budget": {
                    "max_iterations": self.permissions.max_iterations,
                    "max_files_per_pass": self.permissions.max_files_per_pass,
                    "exploration_timeout_seconds": self.permissions.exploration_timeout_seconds,
                },
            },
            "feedback": {
                "verification_commands": self.feedback.verification_commands,
                "error_handling": self.feedback.error_handling,
                "success_verification": self.feedback.success_verification,
            },
            "session": {
                "memory_file": self.session.memory_file,
                "sections": self.session.sections,
            },
            "execution": {
                "initial_actions": self.execution.initial_actions,
                "implementation_order": self.execution.implementation_order,
                "completion_checklist": self.execution.completion_checklist,
            },
        }
        return AgenticConfig.from_dict(config_dict)


class TemplateParser:
    """Parses Agentic Coding System Prompt Templates.

    Supports both Markdown format (as shown in the template) and
    YAML format for programmatic configuration.
    """

    # Regex patterns for parsing
    SECTION_PATTERN = re.compile(r"^##\s+SECTION\s+\d+:\s+(.+)$", re.MULTILINE)
    SUBSECTION_PATTERN = re.compile(r"^###\s+\d+\.\d+\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
    CHECKBOX_PATTERN = re.compile(r"^-\s*\[\s*[xX ]?\s*\]\s*(.+)$", re.MULTILINE)
    LIST_ITEM_PATTERN = re.compile(r"^-\s+(.+)$", re.MULTILINE)
    BULLET_ITEM_PATTERN = re.compile(r"^\s*[-*]\s+(.+)$", re.MULTILINE)

    def __init__(self) -> None:
        """Initialize the template parser."""
        self._current_section: TemplateSection | None = None
        self._line_number: int = 0

    def parse_file(self, file_path: Path) -> ParsedTemplate:
        """Parse a template from a file.

        Args:
            file_path: Path to the template file.

        Returns:
            ParsedTemplate with extracted content.

        Raises:
            TemplateParseError: If parsing fails.
            FileNotFoundError: If file doesn't exist.
        """
        logger.info(f"Parsing template from {file_path}")

        with open(file_path) as f:
            content = f.read()

        # Detect format based on extension
        if file_path.suffix in [".yaml", ".yml"]:
            return self.parse_yaml(content, source_file=file_path)
        else:
            return self.parse_markdown(content, source_file=file_path)

    def parse_markdown(self, content: str, source_file: Path | None = None) -> ParsedTemplate:
        """Parse a Markdown format template.

        Args:
            content: Markdown content to parse.
            source_file: Optional source file path for error reporting.

        Returns:
            ParsedTemplate with extracted content.
        """
        logger.debug("Parsing Markdown template")

        template = ParsedTemplate(raw_content=content, source_file=source_file)

        # Split into sections
        sections = self._split_into_sections(content)

        for section_name, section_content in sections.items():
            try:
                self._parse_section(template, section_name, section_content)
            except Exception as e:
                logger.warning(f"Error parsing section {section_name}: {e}")
                # Continue parsing other sections

        return template

    def parse_yaml(self, content: str, source_file: Path | None = None) -> ParsedTemplate:
        """Parse a YAML format template.

        Args:
            content: YAML content to parse.
            source_file: Optional source file path for error reporting.

        Returns:
            ParsedTemplate with extracted content.
        """
        logger.debug("Parsing YAML template")

        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise TemplateParseError("YAML content must be a dictionary")

        template = ParsedTemplate(raw_content=content, source_file=source_file)

        # Parse objective
        if "objective" in data:
            obj = data["objective"]
            template.objective = ParsedObjective(
                system_intent=obj.get("system_intent", ""),
                success_criteria=self._parse_criteria_list(obj.get("success_criteria", [])),
                problem_statement=obj.get("problem_description", {}).get("problem_statement", ""),
                data_flows=obj.get("problem_description", {}).get("data_flows", ""),
                failure_modes=obj.get("problem_description", {}).get("failure_modes", ""),
            )

        # Parse constraints
        if "constraints" in data:
            const = data["constraints"]
            template.constraints = ParsedConstraints(
                hard=const.get("hard", []),
                soft=const.get("soft", []),
                anti=const.get("anti", []),
            )

        # Parse permissions
        if "permissions" in data:
            perm = data["permissions"]
            scope = perm.get("scope", {})
            autonomy = perm.get("autonomy", {})
            budget = perm.get("resource_budget", {})

            template.permissions = ParsedPermissions(
                in_scope=scope.get("in_scope", []),
                out_of_scope=scope.get("out_of_scope", []),
                autonomous=autonomy.get("autonomous", []),
                confirm_first=autonomy.get("confirm_first", []),
                prohibited=autonomy.get("prohibited", []),
                max_iterations=budget.get("max_iterations", 5),
                max_files_per_pass=budget.get("max_files_per_pass", 20),
                exploration_timeout_seconds=budget.get("exploration_timeout_seconds", 600),
            )

        # Parse feedback
        if "feedback" in data:
            fb = data["feedback"]
            template.feedback = ParsedFeedback(
                verification_commands=fb.get("verification_commands", []),
                error_handling=fb.get("error_handling", {}),
                success_verification=fb.get("success_verification", []),
            )

        # Parse session
        if "session" in data:
            sess = data["session"]
            template.session = ParsedSession(
                memory_file=sess.get("memory_file", "CLAUDE.md"),
                sections=sess.get("sections", []),
                preserve_info=sess.get("preserve_info", []),
                derivable_info=sess.get("derivable_info", []),
            )

        # Parse execution
        if "execution" in data:
            exec_data = data["execution"]
            template.execution = ParsedExecution(
                initial_actions=exec_data.get("initial_actions", []),
                implementation_order=exec_data.get("implementation_order", []),
                completion_checklist=exec_data.get("completion_checklist", []),
            )

        return template

    def _split_into_sections(self, content: str) -> dict[str, str]:
        """Split content into named sections.

        Args:
            content: Full template content.

        Returns:
            Dictionary mapping section names to content.
        """
        sections: dict[str, str] = {}
        current_section: str | None = None
        current_content: list[str] = []

        for line in content.split("\n"):
            section_match = self.SECTION_PATTERN.match(line)
            if section_match:
                # Save previous section
                if current_section is not None:
                    sections[current_section] = "\n".join(current_content)

                current_section = section_match.group(1).strip()
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section is not None:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _parse_section(self, template: ParsedTemplate, section_name: str, content: str) -> None:
        """Parse a single section and update the template.

        Args:
            template: Template to update.
            section_name: Name of the section.
            content: Section content.
        """
        section_name_upper = section_name.upper()

        if "OBJECTIVE" in section_name_upper:
            self._parse_objective_section(template, content)
        elif "FEASIBLE" in section_name_upper or "CONSTRAINT" in section_name_upper:
            self._parse_constraints_section(template, content)
        elif "PERMISSION" in section_name_upper:
            self._parse_permissions_section(template, content)
        elif "FEEDBACK" in section_name_upper:
            self._parse_feedback_section(template, content)
        elif "CONTEXT" in section_name_upper or "SESSION" in section_name_upper:
            self._parse_session_section(template, content)
        elif "EXECUTION" in section_name_upper:
            self._parse_execution_section(template, content)

    def _parse_objective_section(self, template: ParsedTemplate, content: str) -> None:
        """Parse the objective function section."""
        # Extract system intent from code block or text
        code_blocks = self.CODE_BLOCK_PATTERN.findall(content)

        # Look for "I am building:" pattern
        intent_match = re.search(r"I am building:\s*(.+?)(?:\n|$)", content)
        if intent_match:
            template.objective.system_intent = intent_match.group(1).strip()
        elif code_blocks:
            template.objective.system_intent = code_blocks[0].strip()

        # Extract success criteria from checkboxes
        criteria = self.CHECKBOX_PATTERN.findall(content)
        template.objective.success_criteria = [
            {"criterion": c.strip(), "verifiable": True} for c in criteria
        ]

        # Extract problem description paragraphs
        paragraphs = self._extract_paragraphs(content, after="Problem Description")
        if len(paragraphs) >= 1:
            template.objective.problem_statement = paragraphs[0]
        if len(paragraphs) >= 2:
            template.objective.data_flows = paragraphs[1]
        if len(paragraphs) >= 3:
            template.objective.failure_modes = paragraphs[2]

    def _parse_constraints_section(self, template: ParsedTemplate, content: str) -> None:
        """Parse the constraints/feasible region section."""
        subsections = self._split_subsections(content)

        for name, subcontent in subsections.items():
            name_lower = name.lower()
            items = self._extract_list_items(subcontent)

            if "hard" in name_lower:
                template.constraints.hard = [self._parse_constraint_item(item) for item in items]
            elif "soft" in name_lower:
                template.constraints.soft = [
                    self._parse_constraint_item(item, is_soft=True) for item in items
                ]
            elif "anti" in name_lower:
                template.constraints.anti = [{"permission": item, "scope": ""} for item in items]

    def _parse_permissions_section(self, template: ParsedTemplate, content: str) -> None:
        """Parse the permissions architecture section."""
        subsections = self._split_subsections(content)

        for name, subcontent in subsections.items():
            name_lower = name.lower()
            # Note: items extraction moved inside specific sections as needed

            if "scope" in name_lower:
                # Look for IN SCOPE and OUT OF SCOPE
                in_scope_match = re.search(
                    r"IN\s+SCOPE:?\s*\n((?:\s*-\s*.+\n?)+)", subcontent, re.IGNORECASE
                )
                out_scope_match = re.search(
                    r"OUT\s+OF\s+SCOPE:?\s*\n((?:\s*-\s*.+\n?)+)", subcontent, re.IGNORECASE
                )

                if in_scope_match:
                    template.permissions.in_scope = self._extract_list_items(
                        in_scope_match.group(1)
                    )
                if out_scope_match:
                    template.permissions.out_of_scope = self._extract_list_items(
                        out_scope_match.group(1)
                    )

            elif "autonomy" in name_lower:
                # Look for AUTONOMOUS, CONFIRM FIRST, PROHIBITED
                auto_match = re.search(
                    r"AUTONOMOUS[^:]*:?\s*\n((?:\s*-\s*.+\n?)+)", subcontent, re.IGNORECASE
                )
                confirm_match = re.search(
                    r"CONFIRM\s+FIRST[^:]*:?\s*\n((?:\s*-\s*.+\n?)+)", subcontent, re.IGNORECASE
                )
                prohibit_match = re.search(
                    r"PROHIBITED[^:]*:?\s*\n((?:\s*-\s*.+\n?)+)", subcontent, re.IGNORECASE
                )

                if auto_match:
                    template.permissions.autonomous = self._extract_list_items(auto_match.group(1))
                if confirm_match:
                    template.permissions.confirm_first = self._extract_list_items(
                        confirm_match.group(1)
                    )
                if prohibit_match:
                    template.permissions.prohibited = self._extract_list_items(
                        prohibit_match.group(1)
                    )

            elif "resource" in name_lower or "budget" in name_lower:
                # Extract numeric values
                iter_match = re.search(r"max.*iterations[^:]*:\s*(\d+)", subcontent, re.IGNORECASE)
                files_match = re.search(r"max.*files[^:]*:\s*(\d+)", subcontent, re.IGNORECASE)
                timeout_match = re.search(r"timeout[^:]*:\s*(\d+)", subcontent, re.IGNORECASE)

                if iter_match:
                    template.permissions.max_iterations = int(iter_match.group(1))
                if files_match:
                    template.permissions.max_files_per_pass = int(files_match.group(1))
                if timeout_match:
                    template.permissions.exploration_timeout_seconds = int(timeout_match.group(1))

    def _parse_feedback_section(self, template: ParsedTemplate, content: str) -> None:
        """Parse the feedback loop section."""
        subsections = self._split_subsections(content)

        for name, subcontent in subsections.items():
            name_lower = name.lower()

            if "verification" in name_lower and "command" in name_lower:
                # Extract numbered commands from code blocks
                code_blocks = self.CODE_BLOCK_PATTERN.findall(subcontent)
                if code_blocks:
                    commands = self._parse_verification_commands(code_blocks[0])
                    template.feedback.verification_commands = commands

            elif "error" in name_lower:
                template.feedback.error_handling = self._parse_error_handling(subcontent)

            elif "success" in name_lower:
                items = self._extract_numbered_list(subcontent)
                template.feedback.success_verification = [
                    {"check": item, "command": None} for item in items
                ]

    def _parse_session_section(self, template: ParsedTemplate, content: str) -> None:
        """Parse the session/context persistence section."""
        # Look for memory file specification
        file_match = re.search(
            r"(?:CLAUDE\.md|memory_file)[^:]*:\s*[\"']?(\S+)[\"']?", content, re.IGNORECASE
        )
        if file_match:
            template.session.memory_file = file_match.group(1)

        subsections = self._split_subsections(content)

        for name, subcontent in subsections.items():
            name_lower = name.lower()

            if "preserve" in name_lower:
                template.session.preserve_info = self._extract_list_items(subcontent)
            elif "deriv" in name_lower or "re-deriv" in name_lower:
                template.session.derivable_info = self._extract_list_items(subcontent)
            elif "section" in name_lower:
                template.session.sections = self._parse_memory_sections(subcontent)

    def _parse_execution_section(self, template: ParsedTemplate, content: str) -> None:
        """Parse the execution protocol section."""
        subsections = self._split_subsections(content)

        for name, subcontent in subsections.items():
            name_lower = name.lower()

            if "initial" in name_lower:
                items = self._extract_numbered_list(subcontent)
                template.execution.initial_actions = [
                    {"action": item, "command": None, "required": True} for item in items
                ]

            elif "order" in name_lower or "implementation" in name_lower:
                items = self._extract_numbered_list(subcontent)
                template.execution.implementation_order = [
                    {"phase": item, "description": "", "agent": ""} for item in items
                ]

            elif "checklist" in name_lower or "completion" in name_lower:
                items = self.CHECKBOX_PATTERN.findall(subcontent)
                if not items:
                    items = self._extract_list_items(subcontent)
                template.execution.completion_checklist = [
                    {"item": item, "verification": None} for item in items
                ]

    # Helper methods

    def _split_subsections(self, content: str) -> dict[str, str]:
        """Split content into subsections based on ### headers."""
        subsections: dict[str, str] = {}
        current_name: str | None = None
        current_content: list[str] = []

        for line in content.split("\n"):
            match = self.SUBSECTION_PATTERN.match(line)
            if match:
                if current_name is not None:
                    subsections[current_name] = "\n".join(current_content)
                current_name = match.group(1).strip()
                current_content = []
            else:
                current_content.append(line)

        if current_name is not None:
            subsections[current_name] = "\n".join(current_content)

        return subsections

    def _extract_list_items(self, content: str) -> list[str]:
        """Extract list items from content."""
        items = self.LIST_ITEM_PATTERN.findall(content)
        return [item.strip() for item in items if item.strip()]

    def _extract_numbered_list(self, content: str) -> list[str]:
        """Extract numbered list items from content."""
        pattern = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)
        items = pattern.findall(content)
        return [item.strip() for item in items if item.strip()]

    def _extract_paragraphs(self, content: str, after: str | None = None) -> list[str]:
        """Extract paragraphs from content."""
        if after:
            match = re.search(rf"{after}.*?\n\n", content, re.IGNORECASE | re.DOTALL)
            if match:
                content = content[match.end() :]

        # Split by double newlines and filter empty
        paragraphs = re.split(r"\n\s*\n", content)
        return [p.strip() for p in paragraphs if p.strip() and not p.startswith("#")]

    def _parse_constraint_item(self, text: str, is_soft: bool = False) -> dict[str, Any]:
        """Parse a constraint item into structured form."""
        # Try to extract key: value format
        match = re.match(r"^([^:]+):\s*(.+)$", text)
        if match:
            name = match.group(1).strip()
            value = match.group(2).strip()
        else:
            name = text
            value = ""

        constraint = {
            "name": name,
            "type": "general",
            "value": value,
        }

        if is_soft:
            constraint["priority"] = 5  # Default priority

        return constraint

    def _parse_criteria_list(self, criteria: list[Any]) -> list[dict[str, Any]]:
        """Parse a list of criteria into structured form."""
        result = []
        for c in criteria:
            if isinstance(c, str):
                result.append({"criterion": c, "verifiable": True})
            elif isinstance(c, dict):
                result.append(c)
        return result

    def _parse_verification_commands(self, content: str) -> list[dict[str, Any]]:
        """Parse verification commands from code block content."""
        commands = []
        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Try to extract numbered command
            match = re.match(r"^\d+\.\s*\[?([^\]]+)\]?:?\s*`?([^`]+)`?", line)
            if match:
                name = match.group(1).strip()
                command = match.group(2).strip()
                commands.append(
                    {
                        "name": name.lower().replace(" ", "_"),
                        "command": command,
                        "timeout_seconds": 300,
                        "fail_fast": False,
                        "retry_count": 0,
                    }
                )
            elif "`" in line:
                # Extract command from backticks
                cmd_match = re.search(r"`([^`]+)`", line)
                if cmd_match:
                    commands.append(
                        {
                            "name": f"command_{len(commands)}",
                            "command": cmd_match.group(1),
                            "timeout_seconds": 300,
                            "fail_fast": False,
                            "retry_count": 0,
                        }
                    )

        return commands

    def _parse_error_handling(self, content: str) -> dict[str, Any]:
        """Parse error handling specification."""
        patterns = []

        # Look for ON <ERROR> -> <ACTION> patterns
        on_pattern = re.compile(
            r"ON\s+([A-Z_]+(?:\s+[A-Z_]+)*):\s*\n\s*[→->]+\s*(.+?)(?=\n\s*ON|\Z)",
            re.IGNORECASE | re.DOTALL,
        )

        for match in on_pattern.finditer(content):
            error_type = match.group(1).strip().lower().replace(" ", "_")
            action_text = match.group(2).strip()

            # Determine action type
            action = "analyze_fix"
            if "auto" in action_text.lower() and "fix" in action_text.lower():
                action = "auto_fix"
            elif "report" in action_text.lower() or "stop" in action_text.lower():
                action = "report"
            elif "escalate" in action_text.lower() or "guidance" in action_text.lower():
                action = "escalate"

            patterns.append(
                {
                    "error_type": error_type,
                    "action": action,
                    "max_retries": 3,
                }
            )

        return {"auto_fix_patterns": patterns}

    def _parse_memory_sections(self, content: str) -> list[dict[str, Any]]:
        """Parse session memory sections."""
        sections = []

        # Look for ## headers in code block or list items
        header_pattern = re.compile(r"##\s+(.+)")
        for match in header_pattern.finditer(content):
            section_name = match.group(1).strip()
            sections.append(
                {
                    "name": section_name,
                    "persist": True,
                    "format": "markdown_list",
                }
            )

        # Also check list items
        if not sections:
            items = self._extract_list_items(content)
            for item in items:
                sections.append(
                    {
                        "name": item,
                        "persist": True,
                        "format": "markdown_list",
                    }
                )

        return sections
