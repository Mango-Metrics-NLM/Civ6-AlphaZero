"""
Tests for session memory management.
"""

from agentic_framework.core.session_memory import (
    ArchitectureDecision,
    KnownIssue,
    MemorySection,
    SectionFormat,
    SessionMemory,
)


class TestMemorySection:
    """Tests for MemorySection."""

    def test_render_markdown_list(self):
        """Test rendering section as markdown list."""
        section = MemorySection(
            name="Test Section",
            content=["Item 1", "Item 2", "Item 3"],
            format=SectionFormat.MARKDOWN_LIST,
        )

        rendered = section.render()

        assert "## Test Section" in rendered
        assert "- Item 1" in rendered
        assert "- Item 2" in rendered

    def test_render_markdown_list_with_dicts(self):
        """Test rendering section with dict items."""
        section = MemorySection(
            name="Commands",
            content=[
                {"pytest": "Run tests"},
                {"ruff": "Lint code"},
            ],
            format=SectionFormat.MARKDOWN_LIST,
        )

        rendered = section.render()

        assert "**pytest**" in rendered
        assert "Run tests" in rendered

    def test_render_markdown_table(self):
        """Test rendering section as markdown table."""
        section = MemorySection(
            name="Decisions",
            content=[
                {"Date": "2024-01-01", "Decision": "Use pytest"},
                {"Date": "2024-01-02", "Decision": "Use ruff"},
            ],
            format=SectionFormat.MARKDOWN_TABLE,
        )

        rendered = section.render()

        assert "| Date | Decision |" in rendered
        assert "| --- | --- |" in rendered
        assert "| 2024-01-01 | Use pytest |" in rendered

    def test_render_code_block(self):
        """Test rendering section as code block."""
        section = MemorySection(
            name="Code",
            content=["def hello():", "    return 'world'"],
            format=SectionFormat.CODE_BLOCK,
        )

        rendered = section.render()

        assert "```" in rendered
        assert "def hello():" in rendered

    def test_render_empty_section(self):
        """Test rendering empty section."""
        section = MemorySection(
            name="Empty",
            content=[],
        )

        rendered = section.render()

        assert "*No entries*" in rendered

    def test_parse_markdown_list(self):
        """Test parsing markdown list content."""
        content = """
- Item 1
- Item 2
- **Key**: Value
"""

        section = MemorySection.parse("Test", content)

        assert len(section.content) == 3
        assert "Item 1" in section.content
        assert {"Key": "Value"} in section.content


class TestSessionMemory:
    """Tests for SessionMemory."""

    def test_create_default(self, tmp_path):
        """Test creating default session memory."""
        memory = SessionMemory(memory_file=tmp_path / "CLAUDE.md")
        memory.create_default()

        assert memory.get_section("Build Commands") is not None
        assert memory.get_section("Test Commands") is not None
        assert memory.get_section("Architecture Decisions") is not None

    def test_save_and_load(self, tmp_path):
        """Test saving and loading memory file."""
        memory_file = tmp_path / "CLAUDE.md"

        # Create and save
        memory = SessionMemory(memory_file=memory_file)
        memory.create_default()
        memory.add_build_command("pytest", "Run tests")
        memory.save()

        # Load in new instance
        memory2 = SessionMemory(memory_file=memory_file)
        memory2.load()

        commands = memory2.get_build_commands()
        assert len(commands) > 0

    def test_add_build_command(self, tmp_path):
        """Test adding build command."""
        memory = SessionMemory(memory_file=tmp_path / "CLAUDE.md")
        memory.create_default()

        memory.add_build_command("make build", "Build project")

        commands = memory.get_build_commands()
        assert any("make build" in str(cmd) for cmd in commands)

    def test_add_test_command(self, tmp_path):
        """Test adding test command."""
        memory = SessionMemory(memory_file=tmp_path / "CLAUDE.md")
        memory.create_default()

        memory.add_test_command("pytest -v", "Run tests verbosely")

        commands = memory.get_test_commands()
        assert any("pytest -v" in str(cmd) for cmd in commands)

    def test_add_architecture_decision(self, tmp_path):
        """Test adding architecture decision."""
        memory = SessionMemory(memory_file=tmp_path / "CLAUDE.md")
        memory.create_default()

        decision = ArchitectureDecision(
            date="2024-01-01",
            decision="Use pytest for testing",
            rationale="Industry standard",
        )

        memory.add_architecture_decision(decision)

        decisions = memory.get_architecture_decisions()
        assert len(decisions) > 0
        assert any("pytest" in str(d) for d in decisions)

    def test_add_known_issue(self, tmp_path):
        """Test adding known issue."""
        memory = SessionMemory(memory_file=tmp_path / "CLAUDE.md")
        memory.create_default()

        issue = KnownIssue(
            issue="Tests slow on CI",
            workaround="Use pytest-xdist",
            status="open",
        )

        memory.add_known_issue(issue)

        issues = memory.get_known_issues()
        assert len(issues) > 0

    def test_add_gotcha(self, tmp_path):
        """Test adding gotcha."""
        memory = SessionMemory(memory_file=tmp_path / "CLAUDE.md")
        memory.create_default()

        memory.add_gotcha("Remember to activate virtualenv")

        section = memory.get_section("Gotchas")
        assert section is not None
        assert "virtualenv" in str(section.content)

    def test_is_modified_tracking(self, tmp_path):
        """Test modification tracking."""
        memory = SessionMemory(memory_file=tmp_path / "CLAUDE.md")
        memory.create_default()

        assert memory.is_modified  # Just created

        memory.save()
        assert not memory.is_modified  # Just saved

        memory.add_gotcha("New gotcha")
        assert memory.is_modified  # Modified again

    def test_exists(self, tmp_path):
        """Test checking if memory file exists."""
        memory_file = tmp_path / "CLAUDE.md"

        memory = SessionMemory(memory_file=memory_file)
        assert not memory.exists()

        memory_file.write_text("# Test", encoding="utf-8")
        assert memory.exists()

    def test_render_full(self, tmp_path):
        """Test rendering full memory file."""
        memory = SessionMemory(memory_file=tmp_path / "CLAUDE.md")
        memory.create_default()
        memory.add_build_command("make", "Build all")
        memory.add_test_command("pytest", "Run tests")

        rendered = memory.render()

        assert "# CLAUDE.md" in rendered
        assert "Build Commands" in rendered
        assert "Test Commands" in rendered
