"""Agent Tools Module for Claude Code Sub-Agents.

This module provides reusable tools and utilities for Claude Code sub-agents,
including exploration, code analysis, file operations, and task management.

Usage:
    from alpha_zero.utils.agent_tools import (
        CodeExplorer,
        FileManager,
        TaskTracker,
        DebugHelper,
    )

    # Explore codebase
    explorer = CodeExplorer()
    files = explorer.find_by_pattern("**/*.py")

    # Track tasks
    tracker = TaskTracker()
    tracker.add("Implement feature X")
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Code Exploration Tools
# =============================================================================

@dataclass
class CodeLocation:
    """Represents a location in code."""
    file_path: Path
    line_number: int
    column: int = 0
    end_line: int | None = None
    context: str = ""

    def to_reference(self) -> str:
        """Generate a file:line reference string."""
        return f"{self.file_path}:{self.line_number}"


@dataclass
class CodeSymbol:
    """Represents a code symbol (class, function, etc.)."""
    name: str
    kind: str  # "class", "function", "method", "variable"
    location: CodeLocation
    docstring: str | None = None
    signature: str | None = None
    parent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "kind": self.kind,
            "file": str(self.location.file_path),
            "line": self.location.line_number,
            "docstring": self.docstring,
            "signature": self.signature,
            "parent": self.parent,
        }


class CodeExplorer:
    """Utility for exploring and analyzing codebases."""

    def __init__(self, root_path: Path | str | None = None):
        """Initialize explorer.

        Args:
            root_path: Root directory to explore. Defaults to current directory.
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()

    def find_by_pattern(
        self,
        pattern: str,
        exclude_dirs: list[str] | None = None,
    ) -> list[Path]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py").
            exclude_dirs: Directories to exclude.

        Returns:
            List of matching file paths.
        """
        exclude_dirs = exclude_dirs or [
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            ".mypy_cache", ".pytest_cache", "build", "dist",
        ]

        results = []
        for path in self.root_path.glob(pattern):
            if not any(ex in path.parts for ex in exclude_dirs):
                results.append(path)

        return sorted(results)

    def find_python_files(self) -> list[Path]:
        """Find all Python files in the project."""
        return self.find_by_pattern("**/*.py")

    def find_test_files(self) -> list[Path]:
        """Find all test files."""
        return self.find_by_pattern("**/test_*.py")

    def grep(
        self,
        pattern: str,
        file_pattern: str = "**/*.py",
        ignore_case: bool = False,
    ) -> list[CodeLocation]:
        """Search for pattern in files.

        Args:
            pattern: Regex pattern to search for.
            file_pattern: Glob pattern for files to search.
            ignore_case: Whether to ignore case.

        Returns:
            List of locations where pattern was found.
        """
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
        results = []

        for file_path in self.find_by_pattern(file_pattern):
            try:
                content = file_path.read_text(encoding="utf-8")
                for i, line in enumerate(content.split("\n"), 1):
                    if regex.search(line):
                        results.append(CodeLocation(
                            file_path=file_path,
                            line_number=i,
                            context=line.strip(),
                        ))
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")

        return results

    def find_definitions(
        self,
        name: str,
        kind: str | None = None,
    ) -> list[CodeSymbol]:
        """Find definitions of a symbol.

        Args:
            name: Symbol name to find.
            kind: Optional kind filter ("class", "function", "method").

        Returns:
            List of matching symbols.
        """
        results = []

        for file_path in self.find_python_files():
            symbols = self.extract_symbols(file_path)
            for symbol in symbols:
                if symbol.name == name:
                    if kind is None or symbol.kind == kind:
                        results.append(symbol)

        return results

    def extract_symbols(self, file_path: Path) -> list[CodeSymbol]:
        """Extract all symbols from a Python file.

        Args:
            file_path: Path to Python file.

        Returns:
            List of symbols found in the file.
        """
        symbols = []

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            for node in ast.walk(tree):
                symbol = self._node_to_symbol(node, file_path)
                if symbol:
                    symbols.append(symbol)

        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")

        return symbols

    def _node_to_symbol(
        self,
        node: ast.AST,
        file_path: Path,
    ) -> CodeSymbol | None:
        """Convert AST node to CodeSymbol.

        Args:
            node: AST node.
            file_path: Source file path.

        Returns:
            CodeSymbol if node is a definition, else None.
        """
        if isinstance(node, ast.ClassDef):
            return CodeSymbol(
                name=node.name,
                kind="class",
                location=CodeLocation(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    end_line=node.end_lineno,
                ),
                docstring=ast.get_docstring(node),
            )

        elif isinstance(node, ast.FunctionDef):
            # Determine if it's a method (inside a class)
            kind = "function"  # Could detect method by parent analysis
            signature = self._extract_signature(node)

            return CodeSymbol(
                name=node.name,
                kind=kind,
                location=CodeLocation(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    end_line=node.end_lineno,
                ),
                docstring=ast.get_docstring(node),
                signature=signature,
            )

        return None

    def _extract_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature."""
        args = []

        # Handle positional args
        for arg in node.args.args:
            annotation = ""
            if arg.annotation:
                annotation = f": {ast.unparse(arg.annotation)}"
            args.append(f"{arg.arg}{annotation}")

        # Handle *args and **kwargs
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        # Return annotation
        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"

        return f"def {node.name}({', '.join(args)}){returns}"

    def get_file_structure(self, max_depth: int = 3) -> dict[str, Any]:
        """Get project file structure as nested dict.

        Args:
            max_depth: Maximum directory depth.

        Returns:
            Nested dictionary representing file structure.
        """
        def build_tree(path: Path, depth: int) -> dict[str, Any]:
            if depth > max_depth:
                return {"...": None}

            result: dict[str, Any] = {}
            try:
                for item in sorted(path.iterdir()):
                    if item.name.startswith("."):
                        continue
                    if item.is_dir():
                        if item.name not in ["__pycache__", "node_modules", ".git"]:
                            result[item.name] = build_tree(item, depth + 1)
                    else:
                        result[item.name] = None
            except PermissionError:
                pass

            return result

        return build_tree(self.root_path, 0)


# =============================================================================
# File Management Tools
# =============================================================================

class FileManager:
    """Utility for file operations."""

    def __init__(self, root_path: Path | str | None = None):
        """Initialize file manager.

        Args:
            root_path: Root directory. Defaults to current directory.
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()

    def read_file(
        self,
        file_path: Path | str,
        encoding: str = "utf-8",
    ) -> str:
        """Read file contents.

        Args:
            file_path: Path to file.
            encoding: File encoding.

        Returns:
            File contents as string.
        """
        path = self._resolve_path(file_path)
        return path.read_text(encoding=encoding)

    def read_lines(
        self,
        file_path: Path | str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> list[str]:
        """Read specific lines from file.

        Args:
            file_path: Path to file.
            start_line: Starting line (1-indexed).
            end_line: Ending line (inclusive). None for end of file.

        Returns:
            List of lines.
        """
        content = self.read_file(file_path)
        lines = content.split("\n")

        start_idx = max(0, start_line - 1)
        end_idx = end_line if end_line else len(lines)

        return lines[start_idx:end_idx]

    def write_file(
        self,
        file_path: Path | str,
        content: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> None:
        """Write content to file.

        Args:
            file_path: Path to file.
            content: Content to write.
            encoding: File encoding.
            create_parents: Whether to create parent directories.
        """
        path = self._resolve_path(file_path)

        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding=encoding)
        logger.info(f"Wrote {len(content)} bytes to {path}")

    def append_file(
        self,
        file_path: Path | str,
        content: str,
        encoding: str = "utf-8",
    ) -> None:
        """Append content to file.

        Args:
            file_path: Path to file.
            content: Content to append.
            encoding: File encoding.
        """
        path = self._resolve_path(file_path)

        with open(path, "a", encoding=encoding) as f:
            f.write(content)

    def file_exists(self, file_path: Path | str) -> bool:
        """Check if file exists."""
        return self._resolve_path(file_path).exists()

    def get_file_info(self, file_path: Path | str) -> dict[str, Any]:
        """Get file information.

        Args:
            file_path: Path to file.

        Returns:
            Dictionary with file information.
        """
        path = self._resolve_path(file_path)
        stat = path.stat()

        return {
            "path": str(path),
            "name": path.name,
            "extension": path.suffix,
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
        }

    def _resolve_path(self, file_path: Path | str) -> Path:
        """Resolve file path relative to root."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.root_path / path
        return path


# =============================================================================
# Task Tracking Tools
# =============================================================================

class TaskStatus(Enum):
    """Task status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a task to be completed."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    parent_id: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "notes": self.notes,
        }


class TaskTracker:
    """Track and manage tasks."""

    def __init__(self):
        """Initialize task tracker."""
        self._tasks: dict[str, Task] = {}
        self._counter = 0

    def add(
        self,
        description: str,
        parent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Task:
        """Add a new task.

        Args:
            description: Task description.
            parent_id: Optional parent task ID.
            tags: Optional tags.

        Returns:
            Created task.
        """
        self._counter += 1
        task_id = f"task_{self._counter}"

        task = Task(
            id=task_id,
            description=description,
            parent_id=parent_id,
            tags=tags or [],
        )

        self._tasks[task_id] = task
        logger.debug(f"Added task: {task_id} - {description}")
        return task

    def start(self, task_id: str) -> Task:
        """Mark task as in progress.

        Args:
            task_id: Task ID.

        Returns:
            Updated task.
        """
        task = self._tasks[task_id]
        task.status = TaskStatus.IN_PROGRESS
        return task

    def complete(self, task_id: str) -> Task:
        """Mark task as completed.

        Args:
            task_id: Task ID.

        Returns:
            Updated task.
        """
        task = self._tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        return task

    def block(self, task_id: str, reason: str = "") -> Task:
        """Mark task as blocked.

        Args:
            task_id: Task ID.
            reason: Reason for blocking.

        Returns:
            Updated task.
        """
        task = self._tasks[task_id]
        task.status = TaskStatus.BLOCKED
        task.notes = reason
        return task

    def get(self, task_id: str) -> Task:
        """Get task by ID."""
        return self._tasks[task_id]

    def list_all(self) -> list[Task]:
        """Get all tasks."""
        return list(self._tasks.values())

    def list_pending(self) -> list[Task]:
        """Get pending tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]

    def list_in_progress(self) -> list[Task]:
        """Get in-progress tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.IN_PROGRESS]

    def list_completed(self) -> list[Task]:
        """Get completed tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.COMPLETED]

    def summary(self) -> str:
        """Generate task summary."""
        pending = len(self.list_pending())
        in_progress = len(self.list_in_progress())
        completed = len(self.list_completed())
        total = len(self._tasks)

        return (
            f"Tasks: {total} total | "
            f"{pending} pending | "
            f"{in_progress} in progress | "
            f"{completed} completed"
        )


# =============================================================================
# Debug Helper Tools
# =============================================================================

class DebugHelper:
    """Utility for debugging and diagnostics."""

    @staticmethod
    def inspect_object(obj: Any, max_depth: int = 2) -> Any:
        """Inspect an object and return its structure.

        Args:
            obj: Object to inspect.
            max_depth: Maximum recursion depth.

        Returns:
            The object's structure (primitives returned as-is, complex objects as dicts).
        """
        def _inspect(o: Any, depth: int) -> Any:
            if depth > max_depth:
                return f"<{type(o).__name__}>"

            if isinstance(o, (str, int, float, bool, type(None))):
                return o
            elif isinstance(o, (list, tuple)):
                return [_inspect(item, depth + 1) for item in o[:10]]
            elif isinstance(o, dict):
                return {k: _inspect(v, depth + 1) for k, v in list(o.items())[:10]}
            else:
                return {
                    "__type__": type(o).__name__,
                    "__attrs__": {
                        k: _inspect(getattr(o, k), depth + 1)
                        for k in dir(o)
                        if not k.startswith("_") and not callable(getattr(o, k))
                    },
                }

        return _inspect(obj, 0)

    @staticmethod
    def trace_call(func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a trace string for a function call.

        Args:
            func_name: Name of function.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Trace string.
        """
        args_str = ", ".join(repr(a)[:50] for a in args)
        kwargs_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in kwargs.items())

        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        return f"{func_name}({all_args})"

    @staticmethod
    def measure_time(description: str):
        """Context manager for timing code blocks.

        Args:
            description: Description of the timed block.

        Yields:
            Timer object with elapsed property.
        """
        import time

        class Timer:
            def __init__(self):
                self.start = time.perf_counter()
                self.elapsed = 0.0

        timer = Timer()

        try:
            yield timer
        finally:
            timer.elapsed = time.perf_counter() - timer.start
            logger.info(f"{description}: {timer.elapsed:.4f}s")

    @staticmethod
    def check_environment() -> dict[str, Any]:
        """Check and report environment information.

        Returns:
            Dictionary with environment details.
        """
        import platform
        import sys

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "env_vars": {
                k: v for k, v in os.environ.items()
                if k.startswith(("PYTHON", "PATH", "CIV6", "LLM", "OPENAI", "ANTHROPIC"))
            },
        }

    @staticmethod
    def validate_dependencies(required: list[str]) -> dict[str, bool]:
        """Check if required packages are installed.

        Args:
            required: List of package names.

        Returns:
            Dictionary mapping package name to availability.
        """
        import importlib.util

        results = {}
        for package in required:
            spec = importlib.util.find_spec(package)
            results[package] = spec is not None

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def explore(pattern: str = "**/*.py") -> list[Path]:
    """Quick function to find files.

    Args:
        pattern: Glob pattern.

    Returns:
        List of matching paths.
    """
    explorer = CodeExplorer()
    return explorer.find_by_pattern(pattern)


def search(pattern: str, files: str = "**/*.py") -> list[CodeLocation]:
    """Quick function to search code.

    Args:
        pattern: Regex pattern.
        files: File glob pattern.

    Returns:
        List of matching locations.
    """
    explorer = CodeExplorer()
    return explorer.grep(pattern, files)


def find_definition(name: str) -> list[CodeSymbol]:
    """Quick function to find symbol definitions.

    Args:
        name: Symbol name.

    Returns:
        List of matching symbols.
    """
    explorer = CodeExplorer()
    return explorer.find_definitions(name)
