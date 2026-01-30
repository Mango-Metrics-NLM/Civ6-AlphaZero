"""
Implementation Agent for Agentic Framework

Writes and modifies code based on implementation plans,
handles refactoring, and manages file operations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logging_config import get_logger
from .base_agent import AgentContext, AgentResult, AgentStatus, BaseAgent

logger = get_logger("implementation_agent")


@dataclass
class FileChange:
    """Represents a change to a file."""

    path: Path
    change_type: str  # create, modify, delete
    old_content: str | None = None
    new_content: str | None = None
    description: str = ""
    line_changes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "change_type": self.change_type,
            "description": self.description,
            "line_changes": self.line_changes,
        }


@dataclass
class ImplementationTask:
    """A task for the implementation agent."""

    description: str
    target_files: list[str] = field(default_factory=list)
    changes: list[FileChange] = field(default_factory=list)
    completed: bool = False
    error: str | None = None


class ImplementationAgent(BaseAgent):
    """Agent that implements code changes.

    Features:
    - Code generation and modification
    - Safe file operations with permission checking
    - Change tracking and rollback support
    - Pattern-based code transformations
    """

    def __init__(
        self,
        timeout_seconds: float = 600.0,
        max_lines_per_file: int = 500,
        prefer_small_changes: bool = True,
    ) -> None:
        """Initialize the implementation agent.

        Args:
            timeout_seconds: Timeout for implementation operations.
            max_lines_per_file: Maximum lines to write in a single file.
            prefer_small_changes: Prefer smaller, incremental changes.
        """
        super().__init__(
            name="implementation",
            description="Writes and modifies code",
            timeout_seconds=timeout_seconds,
        )
        self.max_lines_per_file = max_lines_per_file
        self.prefer_small_changes = prefer_small_changes
        self._changes: list[FileChange] = []
        self._code_generators: dict[str, Callable[..., str]] = {}

    def register_generator(self, name: str, generator: Callable[..., str]) -> None:
        """Register a code generator function.

        Args:
            name: Name to identify the generator.
            generator: Function that generates code.
        """
        self._code_generators[name] = generator
        logger.debug(f"Registered code generator: {name}")

    def _execute(self, context: AgentContext) -> AgentResult:
        """Execute implementation tasks.

        Args:
            context: Execution context.

        Returns:
            AgentResult with implementation outcome.
        """
        self.report_progress(0.0, "Starting implementation")
        self._changes.clear()

        # Get implementation plan from shared data
        plan = context.get_shared("implementation_plan")
        tasks = context.get_shared("implementation_tasks", [])

        if not tasks and plan:
            # Extract tasks from plan
            tasks = self._extract_tasks_from_plan(plan)

        if not tasks:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                success=True,
                message="No implementation tasks to execute",
            )

        # Execute tasks
        completed_tasks = 0
        failed_tasks = 0
        files_modified: list[Path] = []

        for i, task in enumerate(tasks):
            self.report_progress(i / len(tasks), f"Implementing: {task.description[:50]}")

            try:
                result = self._execute_task(context, task)
                if result:
                    completed_tasks += 1
                    files_modified.extend([c.path for c in task.changes])
                    task.completed = True
                else:
                    failed_tasks += 1
            except Exception as e:
                logger.error(f"Task failed: {task.description} - {e}")
                task.error = str(e)
                failed_tasks += 1

        self.report_progress(1.0, "Implementation complete")

        success = failed_tasks == 0
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.COMPLETED if success else AgentStatus.FAILED,
            success=success,
            message=f"Completed {completed_tasks} tasks, {failed_tasks} failed",
            data={
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "changes": [c.to_dict() for c in self._changes],
            },
            files_modified=list(set(files_modified)),
            next_agent="testing" if success else None,
        )

    def _extract_tasks_from_plan(self, plan: Any) -> list[ImplementationTask]:
        """Extract implementation tasks from a plan.

        Args:
            plan: Implementation plan.

        Returns:
            List of implementation tasks.
        """
        tasks: list[ImplementationTask] = []

        # Get steps designated for implementation agent
        steps = getattr(plan, "steps", [])
        for step in steps:
            agent = getattr(step, "agent", "")
            if agent == "implementation":
                task = ImplementationTask(
                    description=getattr(step, "description", ""),
                    target_files=getattr(step, "files_involved", []),
                )
                tasks.append(task)

        return tasks

    def _execute_task(self, context: AgentContext, task: ImplementationTask) -> bool:
        """Execute a single implementation task.

        Args:
            context: Execution context.
            task: Task to execute.

        Returns:
            True if task succeeded.
        """
        logger.info(f"Executing task: {task.description}")

        for target_file in task.target_files:
            file_path = context.working_dir / target_file

            # Check permissions
            if not self.check_permission(f"modify file {target_file}", file_path):
                logger.warning(f"Permission denied for: {target_file}")
                continue

            # Read existing content
            old_content = None
            if file_path.exists():
                old_content = file_path.read_text()

            # Apply transformations based on task
            new_content = self._transform_content(
                old_content or "",
                task.description,
                context,
            )

            if new_content != old_content:
                # Record the change
                change = FileChange(
                    path=file_path,
                    change_type="modify" if old_content else "create",
                    old_content=old_content,
                    new_content=new_content,
                    description=task.description,
                    line_changes=len(new_content.split("\n"))
                    - len((old_content or "").split("\n")),
                )
                task.changes.append(change)
                self._changes.append(change)

                # Write the file
                self._write_file(file_path, new_content)

        return True

    def _transform_content(
        self,
        content: str,
        task_description: str,
        context: AgentContext,
    ) -> str:
        """Transform content based on task description.

        Args:
            content: Current content.
            task_description: Description of changes to make.
            context: Execution context.

        Returns:
            Transformed content.
        """
        # This is a placeholder - in a real implementation,
        # this would use the LLM or code generation logic
        transformed = content

        # Apply any registered generators
        task_lower = task_description.lower()
        for name, generator in self._code_generators.items():
            if name.lower() in task_lower:
                generated = generator(content, task_description, context)
                if generated:
                    transformed = generated

        return transformed

    def _write_file(self, path: Path, content: str) -> None:
        """Write content to a file safely.

        Args:
            path: Path to write to.
            content: Content to write.
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            temp_path.write_text(content)
            temp_path.rename(path)
            logger.info(f"Wrote file: {path}")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def create_file(
        self,
        context: AgentContext,
        path: Path,
        content: str,
        description: str = "",
    ) -> bool:
        """Create a new file.

        Args:
            context: Execution context.
            path: Path for the new file.
            content: File content.
            description: Description of the file.

        Returns:
            True if file was created.
        """
        if path.exists():
            logger.warning(f"File already exists: {path}")
            return False

        if not self.check_permission(f"create file {path.name}", path):
            return False

        change = FileChange(
            path=path,
            change_type="create",
            new_content=content,
            description=description,
            line_changes=len(content.split("\n")),
        )
        self._changes.append(change)

        self._write_file(path, content)
        return True

    def modify_file(
        self,
        context: AgentContext,
        path: Path,
        transform: Callable[[str], str],
        description: str = "",
    ) -> bool:
        """Modify an existing file.

        Args:
            context: Execution context.
            path: Path to modify.
            transform: Function to transform content.
            description: Description of the change.

        Returns:
            True if file was modified.
        """
        if not path.exists():
            logger.warning(f"File does not exist: {path}")
            return False

        if not self.check_permission(f"modify file {path.name}", path):
            return False

        old_content = path.read_text()
        new_content = transform(old_content)

        if new_content == old_content:
            logger.info(f"No changes needed for: {path}")
            return True

        change = FileChange(
            path=path,
            change_type="modify",
            old_content=old_content,
            new_content=new_content,
            description=description,
            line_changes=len(new_content.split("\n")) - len(old_content.split("\n")),
        )
        self._changes.append(change)

        self._write_file(path, new_content)
        return True

    def delete_file(self, context: AgentContext, path: Path, description: str = "") -> bool:
        """Delete a file.

        Args:
            context: Execution context.
            path: Path to delete.
            description: Description of why.

        Returns:
            True if file was deleted.
        """
        if not path.exists():
            logger.warning(f"File does not exist: {path}")
            return True  # Already deleted

        if not self.check_permission(f"delete file {path.name}", path):
            return False

        old_content = path.read_text()

        change = FileChange(
            path=path,
            change_type="delete",
            old_content=old_content,
            description=description,
            line_changes=-len(old_content.split("\n")),
        )
        self._changes.append(change)

        path.unlink()
        logger.info(f"Deleted file: {path}")
        return True

    def rollback_changes(self) -> int:
        """Rollback all changes made in this session.

        Returns:
            Number of changes rolled back.
        """
        rolled_back = 0

        for change in reversed(self._changes):
            try:
                if change.change_type == "create":
                    if change.path.exists():
                        change.path.unlink()
                        rolled_back += 1
                elif change.change_type == "modify":
                    if change.old_content is not None:
                        change.path.write_text(change.old_content)
                        rolled_back += 1
                elif change.change_type == "delete":
                    if change.old_content is not None:
                        change.path.write_text(change.old_content)
                        rolled_back += 1
            except Exception as e:
                logger.error(f"Failed to rollback change: {change.path} - {e}")

        logger.info(f"Rolled back {rolled_back} changes")
        self._changes.clear()
        return rolled_back

    def get_changes_summary(self) -> str:
        """Get a summary of changes made.

        Returns:
            Markdown summary of changes.
        """
        if not self._changes:
            return "No changes made"

        lines = ["## Changes Made", ""]

        creates = [c for c in self._changes if c.change_type == "create"]
        modifies = [c for c in self._changes if c.change_type == "modify"]
        deletes = [c for c in self._changes if c.change_type == "delete"]

        if creates:
            lines.append(f"### Created ({len(creates)} files)")
            for c in creates:
                lines.append(f"- `{c.path}`: {c.description}")

        if modifies:
            lines.append(f"### Modified ({len(modifies)} files)")
            for c in modifies:
                sign = "+" if c.line_changes >= 0 else ""
                lines.append(f"- `{c.path}` ({sign}{c.line_changes} lines): {c.description}")

        if deletes:
            lines.append(f"### Deleted ({len(deletes)} files)")
            for c in deletes:
                lines.append(f"- `{c.path}`: {c.description}")

        return "\n".join(lines)

    @property
    def changes(self) -> list[FileChange]:
        """Get all recorded changes."""
        return self._changes
