"""
Session Memory for Agentic Framework

Manages persistent memory across sessions using CLAUDE.md or similar files.
Handles reading, writing, and updating session state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..config import AgenticConfig
from ..utils.logging_config import get_logger

logger = get_logger("session_memory")


class SectionFormat(Enum):
    """Formats for memory sections."""

    MARKDOWN_LIST = "markdown_list"
    MARKDOWN_TABLE = "markdown_table"
    CODE_BLOCK = "code_block"
    TEXT = "text"


@dataclass
class MemorySection:
    """Represents a section in the session memory."""

    name: str
    content: list[Any] = field(default_factory=list)
    format: SectionFormat = SectionFormat.MARKDOWN_LIST
    persist: bool = True
    header_level: int = 2

    def render(self) -> str:
        """Render the section to markdown.

        Returns:
            Markdown string for the section.
        """
        header = "#" * self.header_level + f" {self.name}"
        lines = [header, ""]

        if not self.content:
            lines.append("*No entries*")
            return "\n".join(lines)

        if self.format == SectionFormat.MARKDOWN_LIST:
            for item in self.content:
                if isinstance(item, dict):
                    # Format as key: value
                    for key, value in item.items():
                        lines.append(f"- **{key}**: {value}")
                else:
                    lines.append(f"- {item}")

        elif self.format == SectionFormat.MARKDOWN_TABLE:
            if self.content and isinstance(self.content[0], dict):
                # Get headers from first item
                headers = list(self.content[0].keys())
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

                for item in self.content:
                    values = [str(item.get(h, "")) for h in headers]
                    lines.append("| " + " | ".join(values) + " |")

        elif self.format == SectionFormat.CODE_BLOCK:
            lines.append("```")
            for item in self.content:
                lines.append(str(item))
            lines.append("```")

        elif self.format == SectionFormat.TEXT:
            for item in self.content:
                lines.append(str(item))

        return "\n".join(lines)

    @classmethod
    def parse(cls, name: str, content: str, format_hint: SectionFormat | None = None) -> MemorySection:
        """Parse a section from markdown content.

        Args:
            name: Section name.
            content: Markdown content for the section.
            format_hint: Hint for the format type.

        Returns:
            Parsed MemorySection.
        """
        section = cls(name=name, format=format_hint or SectionFormat.MARKDOWN_LIST)

        # Detect format from content
        if "```" in content:
            section.format = SectionFormat.CODE_BLOCK
            # Extract code block content
            code_match = re.search(r"```\w*\n(.*?)```", content, re.DOTALL)
            if code_match:
                section.content = code_match.group(1).strip().split("\n")

        elif "|" in content and "---" in content:
            section.format = SectionFormat.MARKDOWN_TABLE
            # Parse table
            lines = content.strip().split("\n")
            if len(lines) >= 2:
                headers = [h.strip() for h in lines[0].split("|") if h.strip()]
                for line in lines[2:]:  # Skip header and separator
                    if "|" in line:
                        values = [v.strip() for v in line.split("|") if v.strip()]
                        if len(values) == len(headers):
                            section.content.append(dict(zip(headers, values, strict=False)))

        else:
            # Parse as list
            list_pattern = re.compile(r"^\s*[-*]\s+(.+)$", re.MULTILINE)
            items = list_pattern.findall(content)
            for item in items:
                # Check for key: value format
                kv_match = re.match(r"\*\*(.+?)\*\*:\s*(.+)", item)
                if kv_match:
                    section.content.append({kv_match.group(1): kv_match.group(2)})
                else:
                    section.content.append(item)

        return section


@dataclass
class ArchitectureDecision:
    """Represents an architecture decision record."""

    date: str
    decision: str
    rationale: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {
            "Date": self.date,
            "Decision": self.decision,
            "Rationale": self.rationale,
        }


@dataclass
class KnownIssue:
    """Represents a known issue with workaround."""

    issue: str
    workaround: str
    status: str = "open"

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {
            "Issue": self.issue,
            "Workaround": self.workaround,
            "Status": self.status,
        }


class SessionMemory:
    """Manages session memory in CLAUDE.md or similar files.

    Features:
    - Read and parse existing memory files
    - Update specific sections
    - Add new entries to sections
    - Preserve formatting and structure
    - Track changes for commit messages
    """

    def __init__(
        self,
        memory_file: Path | None = None,
        config: AgenticConfig | None = None,
    ) -> None:
        """Initialize session memory.

        Args:
            memory_file: Path to memory file (default: CLAUDE.md in cwd).
            config: Configuration for section setup.
        """
        self.memory_file = memory_file or Path.cwd() / "CLAUDE.md"
        self.config = config
        self._sections: dict[str, MemorySection] = {}
        self._raw_content: str = ""
        self._modified: bool = False

        # Load configuration defaults
        if config:
            self._setup_from_config(config)

    def _setup_from_config(self, config: AgenticConfig) -> None:
        """Set up sections from configuration.

        Args:
            config: Configuration with section definitions.
        """
        for section_config in config.session.sections:
            name = section_config.name
            format_str = section_config.format
            try:
                section_format = SectionFormat(format_str)
            except ValueError:
                section_format = SectionFormat.MARKDOWN_LIST

            self._sections[name] = MemorySection(
                name=name,
                format=section_format,
                persist=section_config.persist,
            )

    def load(self) -> bool:
        """Load the memory file.

        Returns:
            True if file was loaded successfully.
        """
        if not self.memory_file.exists():
            logger.info(f"Memory file does not exist: {self.memory_file}")
            return False

        try:
            self._raw_content = self.memory_file.read_text()
            self._parse_content()
            logger.info(f"Loaded session memory from {self.memory_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load memory file: {e}")
            return False

    def _parse_content(self) -> None:
        """Parse the raw content into sections."""
        # Split by headers
        section_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
        matches = list(section_pattern.finditer(self._raw_content))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            name = match.group(2).strip()

            # Get content between this header and next
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(self._raw_content)
            content = self._raw_content[start:end].strip()

            # Get existing section or create new
            if name in self._sections:
                section = self._sections[name]
                parsed = MemorySection.parse(name, content, section.format)
                section.content = parsed.content
            else:
                section = MemorySection.parse(name, content)
                section.header_level = level

            self._sections[name] = section

    def save(self) -> bool:
        """Save the memory file.

        Returns:
            True if file was saved successfully.
        """
        try:
            content = self.render()
            self.memory_file.write_text(content)
            self._modified = False
            logger.info(f"Saved session memory to {self.memory_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save memory file: {e}")
            return False

    def render(self) -> str:
        """Render all sections to markdown.

        Returns:
            Complete markdown content.
        """
        lines = ["# CLAUDE.md", ""]

        # Render in a consistent order
        section_order = [
            "Build Commands",
            "Test Commands",
            "Architecture Decisions",
            "Known Issues",
            "Environment Setup",
            "Gotchas",
        ]

        rendered_sections = set()

        # Render ordered sections first
        for name in section_order:
            if name in self._sections:
                section = self._sections[name]
                if section.persist:
                    lines.append(section.render())
                    lines.append("")
                rendered_sections.add(name)

        # Render remaining sections
        for name, section in self._sections.items():
            if name not in rendered_sections and section.persist:
                lines.append(section.render())
                lines.append("")

        return "\n".join(lines)

    def get_section(self, name: str) -> MemorySection | None:
        """Get a section by name.

        Args:
            name: Section name.

        Returns:
            MemorySection or None if not found.
        """
        return self._sections.get(name)

    def set_section(self, section: MemorySection) -> None:
        """Set or update a section.

        Args:
            section: Section to set.
        """
        self._sections[section.name] = section
        self._modified = True
        logger.debug(f"Updated section: {section.name}")

    def add_to_section(self, section_name: str, item: Any) -> None:
        """Add an item to a section.

        Args:
            section_name: Name of the section.
            item: Item to add.
        """
        if section_name not in self._sections:
            self._sections[section_name] = MemorySection(name=section_name)

        self._sections[section_name].content.append(item)
        self._modified = True
        logger.debug(f"Added item to section: {section_name}")

    def remove_from_section(self, section_name: str, item: Any) -> bool:
        """Remove an item from a section.

        Args:
            section_name: Name of the section.
            item: Item to remove.

        Returns:
            True if item was found and removed.
        """
        if section_name not in self._sections:
            return False

        section = self._sections[section_name]
        if item in section.content:
            section.content.remove(item)
            self._modified = True
            logger.debug(f"Removed item from section: {section_name}")
            return True
        return False

    # Convenience methods for common sections

    def add_build_command(self, command: str, purpose: str) -> None:
        """Add a build command.

        Args:
            command: Command string.
            purpose: Description of what the command does.
        """
        self.add_to_section("Build Commands", {command: purpose})

    def add_test_command(self, command: str, purpose: str) -> None:
        """Add a test command.

        Args:
            command: Command string.
            purpose: Description of what the command does.
        """
        self.add_to_section("Test Commands", {command: purpose})

    def add_architecture_decision(self, decision: ArchitectureDecision) -> None:
        """Add an architecture decision.

        Args:
            decision: Architecture decision to add.
        """
        if "Architecture Decisions" not in self._sections:
            self._sections["Architecture Decisions"] = MemorySection(
                name="Architecture Decisions",
                format=SectionFormat.MARKDOWN_TABLE,
            )

        self._sections["Architecture Decisions"].content.append(decision.to_dict())
        self._modified = True

    def add_known_issue(self, issue: KnownIssue) -> None:
        """Add a known issue.

        Args:
            issue: Known issue to add.
        """
        if "Known Issues" not in self._sections:
            self._sections["Known Issues"] = MemorySection(
                name="Known Issues",
                format=SectionFormat.MARKDOWN_TABLE,
            )

        self._sections["Known Issues"].content.append(issue.to_dict())
        self._modified = True

    def add_gotcha(self, gotcha: str) -> None:
        """Add a gotcha/tip.

        Args:
            gotcha: Gotcha description.
        """
        self.add_to_section("Gotchas", gotcha)

    def add_environment_step(self, step: str) -> None:
        """Add an environment setup step.

        Args:
            step: Setup step description.
        """
        self.add_to_section("Environment Setup", step)

    def get_build_commands(self) -> list[dict[str, str]]:
        """Get all build commands.

        Returns:
            List of command dictionaries.
        """
        section = self.get_section("Build Commands")
        if section:
            return [item for item in section.content if isinstance(item, dict)]
        return []

    def get_test_commands(self) -> list[dict[str, str]]:
        """Get all test commands.

        Returns:
            List of command dictionaries.
        """
        section = self.get_section("Test Commands")
        if section:
            return [item for item in section.content if isinstance(item, dict)]
        return []

    def get_architecture_decisions(self) -> list[dict[str, str]]:
        """Get all architecture decisions.

        Returns:
            List of decision dictionaries.
        """
        section = self.get_section("Architecture Decisions")
        if section:
            return section.content
        return []

    def get_known_issues(self) -> list[dict[str, str]]:
        """Get all known issues.

        Returns:
            List of issue dictionaries.
        """
        section = self.get_section("Known Issues")
        if section:
            return section.content
        return []

    @property
    def is_modified(self) -> bool:
        """Check if memory has been modified since last save."""
        return self._modified

    def get_changes_summary(self) -> str:
        """Get a summary of changes for commit messages.

        Returns:
            Summary string.
        """
        if not self._modified:
            return "No changes to session memory"

        sections_with_content = [
            name for name, section in self._sections.items()
            if section.content and section.persist
        ]

        return f"Updated session memory sections: {', '.join(sections_with_content)}"

    def exists(self) -> bool:
        """Check if the memory file exists.

        Returns:
            True if file exists.
        """
        return self.memory_file.exists()

    def create_default(self) -> None:
        """Create a default memory file with standard sections."""
        self._sections = {
            "Build Commands": MemorySection(
                name="Build Commands",
                format=SectionFormat.MARKDOWN_LIST,
            ),
            "Test Commands": MemorySection(
                name="Test Commands",
                format=SectionFormat.MARKDOWN_LIST,
            ),
            "Architecture Decisions": MemorySection(
                name="Architecture Decisions",
                format=SectionFormat.MARKDOWN_TABLE,
            ),
            "Known Issues": MemorySection(
                name="Known Issues",
                format=SectionFormat.MARKDOWN_TABLE,
            ),
            "Environment Setup": MemorySection(
                name="Environment Setup",
                format=SectionFormat.MARKDOWN_LIST,
            ),
            "Gotchas": MemorySection(
                name="Gotchas",
                format=SectionFormat.MARKDOWN_LIST,
            ),
        }
        self._modified = True
        logger.info("Created default session memory structure")
