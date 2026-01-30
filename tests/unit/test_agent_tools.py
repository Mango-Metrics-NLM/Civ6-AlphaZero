"""Tests for agent tools module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from alpha_zero.utils.agent_tools import (
    CodeExplorer,
    CodeLocation,
    CodeSymbol,
    DebugHelper,
    FileManager,
    TaskStatus,
    TaskTracker,
    explore,
    find_definition,
    search,
)


class TestCodeLocation:
    """Tests for CodeLocation dataclass."""

    def test_create_location(self):
        """Test creating a code location."""
        loc = CodeLocation(
            file_path=Path("test.py"),
            line_number=42,
        )
        assert loc.file_path == Path("test.py")
        assert loc.line_number == 42
        assert loc.column == 0

    def test_to_reference(self):
        """Test generating reference string."""
        loc = CodeLocation(
            file_path=Path("src/main.py"),
            line_number=100,
        )
        # Path separators vary by OS, so check for the line number part
        ref = loc.to_reference()
        assert ref.endswith(":100")
        assert "main.py" in ref


class TestCodeSymbol:
    """Tests for CodeSymbol dataclass."""

    def test_create_symbol(self):
        """Test creating a code symbol."""
        loc = CodeLocation(file_path=Path("test.py"), line_number=10)
        symbol = CodeSymbol(
            name="my_function",
            kind="function",
            location=loc,
        )
        assert symbol.name == "my_function"
        assert symbol.kind == "function"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        loc = CodeLocation(file_path=Path("test.py"), line_number=10)
        symbol = CodeSymbol(
            name="MyClass",
            kind="class",
            location=loc,
            docstring="A test class",
        )
        data = symbol.to_dict()

        assert data["name"] == "MyClass"
        assert data["kind"] == "class"
        assert data["line"] == 10
        assert data["docstring"] == "A test class"


class TestCodeExplorer:
    """Tests for CodeExplorer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        explorer = CodeExplorer()
        assert explorer.root_path == Path.cwd()

    def test_initialization_with_path(self):
        """Test initialization with custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explorer = CodeExplorer(tmpdir)
            assert explorer.root_path == Path(tmpdir)

    def test_find_by_pattern(self):
        """Test finding files by pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "test1.py").write_text("# test 1")
            (Path(tmpdir) / "test2.py").write_text("# test 2")
            (Path(tmpdir) / "data.json").write_text("{}")

            explorer = CodeExplorer(tmpdir)
            py_files = explorer.find_by_pattern("*.py")

            assert len(py_files) == 2
            assert all(f.suffix == ".py" for f in py_files)

    def test_find_by_pattern_excludes_dirs(self):
        """Test that find_by_pattern excludes specified directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            (Path(tmpdir) / "src").mkdir()
            (Path(tmpdir) / "__pycache__").mkdir()
            (Path(tmpdir) / "src" / "main.py").write_text("# main")
            (Path(tmpdir) / "__pycache__" / "cached.py").write_text("# cached")

            explorer = CodeExplorer(tmpdir)
            files = explorer.find_by_pattern("**/*.py")

            assert len(files) == 1
            assert files[0].name == "main.py"

    def test_find_python_files(self):
        """Test finding Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "script.py").write_text("# script")
            (Path(tmpdir) / "data.txt").write_text("data")

            explorer = CodeExplorer(tmpdir)
            files = explorer.find_python_files()

            assert len(files) == 1
            assert files[0].name == "script.py"

    def test_find_test_files(self):
        """Test finding test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test_main.py").write_text("# test")
            (Path(tmpdir) / "main.py").write_text("# main")
            (Path(tmpdir) / "test_utils.py").write_text("# test utils")

            explorer = CodeExplorer(tmpdir)
            files = explorer.find_test_files()

            assert len(files) == 2
            assert all(f.name.startswith("test_") for f in files)

    def test_grep(self):
        """Test grep functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.py").write_text("def hello():\n    pass\n")
            (Path(tmpdir) / "file2.py").write_text("def world():\n    pass\n")

            explorer = CodeExplorer(tmpdir)
            results = explorer.grep("def hello", "*.py")

            assert len(results) == 1
            assert "def hello" in results[0].context

    def test_grep_case_insensitive(self):
        """Test case-insensitive grep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("class MyClass:\n    pass\n")

            explorer = CodeExplorer(tmpdir)
            results = explorer.grep("myclass", "*.py", ignore_case=True)

            assert len(results) == 1

    def test_extract_symbols(self):
        """Test extracting symbols from Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = '''"""Module docstring."""

class MyClass:
    """A class."""
    pass

def my_function(x: int) -> str:
    """A function."""
    return str(x)
'''
            file_path = Path(tmpdir) / "module.py"
            file_path.write_text(code)

            explorer = CodeExplorer(tmpdir)
            symbols = explorer.extract_symbols(file_path)

            class_symbols = [s for s in symbols if s.kind == "class"]
            func_symbols = [s for s in symbols if s.kind == "function"]

            assert len(class_symbols) >= 1
            assert any(s.name == "MyClass" for s in class_symbols)
            assert len(func_symbols) >= 1
            assert any(s.name == "my_function" for s in func_symbols)

    def test_get_file_structure(self):
        """Test getting file structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "src").mkdir()
            (Path(tmpdir) / "src" / "main.py").write_text("")
            (Path(tmpdir) / "tests").mkdir()
            (Path(tmpdir) / "tests" / "test_main.py").write_text("")

            explorer = CodeExplorer(tmpdir)
            structure = explorer.get_file_structure(max_depth=2)

            assert "src" in structure
            assert "tests" in structure


class TestFileManager:
    """Tests for FileManager class."""

    def test_read_file(self):
        """Test reading a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("Hello, World!")

            manager = FileManager(tmpdir)
            content = manager.read_file("test.txt")

            assert content == "Hello, World!"

    def test_read_lines(self):
        """Test reading specific lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("Line 1\nLine 2\nLine 3\nLine 4\n")

            manager = FileManager(tmpdir)
            lines = manager.read_lines("test.txt", start_line=2, end_line=3)

            assert len(lines) == 2
            assert lines[0] == "Line 2"
            assert lines[1] == "Line 3"

    def test_write_file(self):
        """Test writing a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FileManager(tmpdir)
            manager.write_file("output.txt", "Test content")

            file_path = Path(tmpdir) / "output.txt"
            assert file_path.exists()
            assert file_path.read_text() == "Test content"

    def test_write_file_creates_parents(self):
        """Test that write_file creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FileManager(tmpdir)
            manager.write_file("nested/dir/file.txt", "Nested content")

            file_path = Path(tmpdir) / "nested" / "dir" / "file.txt"
            assert file_path.exists()

    def test_append_file(self):
        """Test appending to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("Initial ")

            manager = FileManager(tmpdir)
            manager.append_file("test.txt", "content")

            assert file_path.read_text() == "Initial content"

    def test_file_exists(self):
        """Test checking file existence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "exists.txt").write_text("")

            manager = FileManager(tmpdir)
            assert manager.file_exists("exists.txt") is True
            assert manager.file_exists("missing.txt") is False

    def test_get_file_info(self):
        """Test getting file information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("Test content")

            manager = FileManager(tmpdir)
            info = manager.get_file_info("test.txt")

            assert info["name"] == "test.txt"
            assert info["extension"] == ".txt"
            assert info["size_bytes"] == 12
            assert info["is_file"] is True


class TestTaskTracker:
    """Tests for TaskTracker class."""

    def test_add_task(self):
        """Test adding a task."""
        tracker = TaskTracker()
        task = tracker.add("Implement feature X")

        assert task.id == "task_1"
        assert task.description == "Implement feature X"
        assert task.status == TaskStatus.PENDING

    def test_add_task_with_tags(self):
        """Test adding a task with tags."""
        tracker = TaskTracker()
        task = tracker.add("Fix bug", tags=["bug", "urgent"])

        assert "bug" in task.tags
        assert "urgent" in task.tags

    def test_start_task(self):
        """Test starting a task."""
        tracker = TaskTracker()
        task = tracker.add("Test task")
        tracker.start(task.id)

        assert task.status == TaskStatus.IN_PROGRESS

    def test_complete_task(self):
        """Test completing a task."""
        tracker = TaskTracker()
        task = tracker.add("Test task")
        tracker.complete(task.id)

        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None

    def test_block_task(self):
        """Test blocking a task."""
        tracker = TaskTracker()
        task = tracker.add("Test task")
        tracker.block(task.id, "Waiting for dependency")

        assert task.status == TaskStatus.BLOCKED
        assert task.notes == "Waiting for dependency"

    def test_list_pending(self):
        """Test listing pending tasks."""
        tracker = TaskTracker()
        tracker.add("Task 1")
        tracker.add("Task 2")
        task3 = tracker.add("Task 3")
        tracker.complete(task3.id)

        pending = tracker.list_pending()
        assert len(pending) == 2

    def test_list_completed(self):
        """Test listing completed tasks."""
        tracker = TaskTracker()
        task1 = tracker.add("Task 1")
        tracker.add("Task 2")
        tracker.complete(task1.id)

        completed = tracker.list_completed()
        assert len(completed) == 1
        assert completed[0].id == task1.id

    def test_summary(self):
        """Test task summary."""
        tracker = TaskTracker()
        tracker.add("Pending task")
        task2 = tracker.add("In progress")
        tracker.start(task2.id)
        task3 = tracker.add("Done")
        tracker.complete(task3.id)

        summary = tracker.summary()
        assert "3 total" in summary
        assert "1 pending" in summary
        assert "1 in progress" in summary
        assert "1 completed" in summary


class TestDebugHelper:
    """Tests for DebugHelper class."""

    def test_inspect_object_primitives(self):
        """Test inspecting primitive values."""
        result = DebugHelper.inspect_object(42)
        assert result == 42

        result = DebugHelper.inspect_object("hello")
        assert result == "hello"

        result = DebugHelper.inspect_object(True)
        assert result is True

    def test_inspect_object_list(self):
        """Test inspecting lists."""
        result = DebugHelper.inspect_object([1, 2, 3])
        assert result == [1, 2, 3]

    def test_inspect_object_dict(self):
        """Test inspecting dictionaries."""
        result = DebugHelper.inspect_object({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_trace_call(self):
        """Test generating trace string."""
        trace = DebugHelper.trace_call("my_func", (1, 2), {"key": "value"})
        assert "my_func" in trace
        assert "1" in trace
        assert "2" in trace
        assert "key" in trace

    def test_check_environment(self):
        """Test environment check."""
        env = DebugHelper.check_environment()

        assert "python_version" in env
        assert "platform" in env
        assert "cwd" in env
        assert "env_vars" in env

    def test_validate_dependencies(self):
        """Test dependency validation."""
        results = DebugHelper.validate_dependencies(["pytest", "nonexistent_package"])

        assert results["pytest"] is True
        assert results["nonexistent_package"] is False


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_explore(self):
        """Test explore convenience function."""
        # Just verify it doesn't raise
        files = explore("**/*.md")
        assert isinstance(files, list)

    def test_search(self):
        """Test search convenience function."""
        # Just verify it doesn't raise
        results = search("def ", "**/*.py")
        assert isinstance(results, list)

    def test_find_definition(self):
        """Test find_definition convenience function."""
        # Just verify it doesn't raise
        results = find_definition("TestClass")
        assert isinstance(results, list)
