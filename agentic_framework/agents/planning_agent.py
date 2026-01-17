"""
Planning Agent for Agentic Framework

Analyzes the codebase, creates implementation plans, and identifies
critical files and architectural considerations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from ..utils.logging_config import get_logger

logger = get_logger("planning_agent")


@dataclass
class ImplementationStep:
    """A single step in an implementation plan."""

    step_number: int
    description: str
    files_involved: list[str] = field(default_factory=list)
    dependencies: list[int] = field(default_factory=list)  # Step numbers this depends on
    agent: str = "implementation"  # Which agent should handle this
    estimated_complexity: str = "medium"  # low, medium, high
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "files_involved": self.files_involved,
            "dependencies": self.dependencies,
            "agent": self.agent,
            "estimated_complexity": self.estimated_complexity,
            "notes": self.notes,
        }


@dataclass
class ImplementationPlan:
    """A complete implementation plan."""

    objective: str
    steps: list[ImplementationStep] = field(default_factory=list)
    critical_files: list[str] = field(default_factory=list)
    architectural_considerations: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "objective": self.objective,
            "steps": [s.to_dict() for s in self.steps],
            "critical_files": self.critical_files,
            "architectural_considerations": self.architectural_considerations,
            "risks": self.risks,
            "success_criteria": self.success_criteria,
        }

    def render_markdown(self) -> str:
        """Render as markdown."""
        lines = [
            f"# Implementation Plan: {self.objective}",
            "",
            "## Steps",
        ]

        for step in self.steps:
            deps = f" (depends on: {step.dependencies})" if step.dependencies else ""
            lines.append(f"{step.step_number}. {step.description}{deps}")
            if step.files_involved:
                lines.append(f"   - Files: {', '.join(step.files_involved)}")
            if step.notes:
                for note in step.notes:
                    lines.append(f"   - Note: {note}")

        if self.critical_files:
            lines.extend(["", "## Critical Files"])
            for f in self.critical_files:
                lines.append(f"- {f}")

        if self.architectural_considerations:
            lines.extend(["", "## Architectural Considerations"])
            for c in self.architectural_considerations:
                lines.append(f"- {c}")

        if self.risks:
            lines.extend(["", "## Risks"])
            for r in self.risks:
                lines.append(f"- {r}")

        if self.success_criteria:
            lines.extend(["", "## Success Criteria"])
            for s in self.success_criteria:
                lines.append(f"- [ ] {s}")

        return "\n".join(lines)


class PlanningAgent(BaseAgent):
    """Agent that analyzes codebases and creates implementation plans.

    Features:
    - Project structure analysis
    - File discovery and categorization
    - Dependency analysis
    - Implementation plan generation
    - Risk identification
    """

    def __init__(
        self,
        timeout_seconds: float = 300.0,
        max_files_to_scan: int = 100,
    ) -> None:
        """Initialize the planning agent.

        Args:
            timeout_seconds: Timeout for planning operations.
            max_files_to_scan: Maximum files to scan during analysis.
        """
        super().__init__(
            name="planning",
            description="Analyzes codebase and creates implementation plans",
            timeout_seconds=timeout_seconds,
        )
        self.max_files_to_scan = max_files_to_scan

    def _execute(self, context: AgentContext) -> AgentResult:
        """Execute planning analysis.

        Args:
            context: Execution context.

        Returns:
            AgentResult with planning outcome.
        """
        self.report_progress(0.0, "Starting analysis")

        # Get the objective from shared data or config
        objective = context.get_shared("objective", "")
        if not objective:
            objective = context.config.objective.system_intent

        if not objective:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                success=False,
                message="No objective specified for planning",
            )

        # Analyze project structure
        self.report_progress(0.2, "Analyzing project structure")
        structure = self._analyze_structure(context.working_dir)

        # Identify critical files
        self.report_progress(0.4, "Identifying critical files")
        critical_files = self._identify_critical_files(context.working_dir, objective)

        # Generate implementation plan
        self.report_progress(0.6, "Generating implementation plan")
        plan = self._generate_plan(context, objective, structure, critical_files)

        # Identify risks
        self.report_progress(0.8, "Identifying risks")
        plan.risks = self._identify_risks(context, plan)

        # Store plan in shared data
        context.set_shared("implementation_plan", plan)
        context.set_shared("critical_files", critical_files)

        self.report_progress(1.0, "Planning complete")

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.COMPLETED,
            success=True,
            message=f"Created plan with {len(plan.steps)} steps",
            data={
                "plan": plan.to_dict(),
                "structure": structure,
                "plan_markdown": plan.render_markdown(),
            },
            next_agent="implementation",
        )

    def _analyze_structure(self, working_dir: Path) -> dict[str, Any]:
        """Analyze project structure.

        Args:
            working_dir: Directory to analyze.

        Returns:
            Structure dictionary.
        """
        structure = {
            "directories": [],
            "file_types": {},
            "entry_points": [],
            "test_directories": [],
            "config_files": [],
        }

        # Collect file information
        file_count = 0
        for path in working_dir.rglob("*"):
            if file_count >= self.max_files_to_scan:
                break

            if path.is_dir():
                if "__pycache__" not in str(path) and ".git" not in str(path):
                    rel_path = str(path.relative_to(working_dir))
                    if rel_path not in structure["directories"]:
                        structure["directories"].append(rel_path)
                continue

            file_count += 1
            rel_path = path.relative_to(working_dir)
            suffix = path.suffix.lower()

            # Track file types
            if suffix not in structure["file_types"]:
                structure["file_types"][suffix] = []
            structure["file_types"][suffix].append(str(rel_path))

            # Identify entry points
            if path.name in ["main.py", "__main__.py", "app.py", "cli.py"]:
                structure["entry_points"].append(str(rel_path))

            # Identify test directories
            if "test" in str(rel_path).lower():
                parent = str(rel_path.parent)
                if parent not in structure["test_directories"]:
                    structure["test_directories"].append(parent)

            # Identify config files
            if suffix in [".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"]:
                structure["config_files"].append(str(rel_path))

        return structure

    def _identify_critical_files(
        self, working_dir: Path, objective: str
    ) -> list[str]:
        """Identify critical files for the objective.

        Args:
            working_dir: Directory to search.
            objective: Implementation objective.

        Returns:
            List of critical file paths.
        """
        critical: list[str] = []
        objective_lower = objective.lower()

        # Keywords from objective
        keywords = set(re.findall(r"\b\w{3,}\b", objective_lower))
        keywords -= {"the", "and", "for", "with", "that", "this", "from", "have", "will"}

        for path in working_dir.rglob("*.py"):
            if "__pycache__" in str(path):
                continue

            rel_path = str(path.relative_to(working_dir))

            # Check filename matches
            filename_lower = path.stem.lower()
            if any(kw in filename_lower for kw in keywords):
                if rel_path not in critical:
                    critical.append(rel_path)
                continue

            # Check file content
            try:
                content = path.read_text()[:2000].lower()  # Read first 2000 chars
                matches = sum(1 for kw in keywords if kw in content)
                if matches >= 2:
                    if rel_path not in critical:
                        critical.append(rel_path)
            except Exception:
                pass

        return critical[:20]  # Limit to 20 files

    def _generate_plan(
        self,
        context: AgentContext,
        objective: str,
        structure: dict[str, Any],
        critical_files: list[str],
    ) -> ImplementationPlan:
        """Generate a dynamic implementation plan based on the objective.

        Args:
            context: Execution context.
            objective: Implementation objective.
            structure: Project structure analysis.
            critical_files: List of critical files.

        Returns:
            Implementation plan.
        """
        plan = ImplementationPlan(
            objective=objective,
            critical_files=critical_files,
        )

        # Get success criteria from config
        for criterion in context.config.objective.success_criteria:
            criterion_text = criterion.get("criterion", str(criterion))
            plan.success_criteria.append(criterion_text)

        # Analyze objective to determine plan structure dynamically
        objective_lower = objective.lower()
        
        # Determine if this is a feature, bugfix, refactoring, or testing task
        is_feature = any(word in objective_lower for word in ["add", "implement", "create", "new"])
        is_bugfix = any(word in objective_lower for word in ["fix", "bug", "error", "issue"])
        is_refactor = any(word in objective_lower for word in ["refactor", "improve", "optimize", "clean"])
        is_test = any(word in objective_lower for word in ["test", "testing", "coverage"])

        # Generate steps based on objective and structure
        step_num = 1

        # Step 1: Always start with understanding
        plan.steps.append(
            ImplementationStep(
                step_number=step_num,
                description="Review existing code patterns and architecture",
                files_involved=critical_files[:5],
                agent="planning",
                estimated_complexity="low",
            )
        )
        step_num += 1

        # Different paths based on objective type
        if is_test:
            # Test-focused workflow
            plan.steps.append(
                ImplementationStep(
                    step_number=step_num,
                    description=f"Generate tests for: {objective}",
                    files_involved=critical_files,
                    dependencies=[1],
                    agent="testing",
                    estimated_complexity="medium",
                )
            )
            step_num += 1
            
        elif is_bugfix:
            # Bug fix workflow
            plan.steps.append(
                ImplementationStep(
                    step_number=step_num,
                    description=f"Fix bug: {objective}",
                    files_involved=critical_files,
                    dependencies=[1],
                    agent="implementation",
                    estimated_complexity="medium",
                )
            )
            step_num += 1
            
            plan.steps.append(
                ImplementationStep(
                    step_number=step_num,
                    description="Add tests to prevent regression",
                    dependencies=[2],
                    agent="testing",
                    estimated_complexity="low",
                )
            )
            step_num += 1
            
        elif is_refactor:
            # Refactoring workflow
            plan.steps.append(
                ImplementationStep(
                    step_number=step_num,
                    description="Run existing tests to establish baseline",
                    dependencies=[1],
                    agent="testing",
                    estimated_complexity="low",
                )
            )
            step_num += 1
            
            plan.steps.append(
                ImplementationStep(
                    step_number=step_num,
                    description=f"Refactor code: {objective}",
                    files_involved=critical_files,
                    dependencies=[2],
                    agent="implementation",
                    estimated_complexity="high",
                )
            )
            step_num += 1
            
            plan.steps.append(
                ImplementationStep(
                    step_number=step_num,
                    description="Verify all tests still pass after refactoring",
                    dependencies=[3],
                    agent="testing",
                    estimated_complexity="low",
                )
            )
            step_num += 1
            
        else:
            # Default/Feature workflow - Step 2: Core implementation
            plan.steps.append(
                ImplementationStep(
                step_number=step_num,
                description=f"Implement core logic for: {objective}",
                files_involved=critical_files,
                dependencies=[1],
                agent="implementation",
                estimated_complexity="high" if is_feature else "medium",
            )
        )
        step_num += 1

        # Step 3: Error handling
        plan.steps.append(
            ImplementationStep(
                step_number=step_num,
                description="Add error handling and edge case management",
                dependencies=[2],
                agent="implementation",
                estimated_complexity="medium",
            )
        )
        step_num += 1

        # Step 4: Tests (if test directories exist)
        if structure.get("test_directories"):
            plan.steps.append(
                ImplementationStep(
                    step_number=step_num,
                    description="Write unit and integration tests",
                    files_involved=[f"{d}/**" for d in structure["test_directories"][:2]],
                    dependencies=[3],
                    agent="testing",
                    estimated_complexity="medium",
                )
            )
            step_num += 1

        # Final step: Always end with verification
        plan.steps.append(
            ImplementationStep(
                step_number=step_num,
                description="Run linting, type checking, and all tests",
                dependencies=[step_num - 1],
                agent="verification",
                estimated_complexity="low",
            )
        )
        step_num += 1

        # Step 6: Documentation
        plan.steps.append(
            ImplementationStep(
                step_number=step_num,
                description="Update documentation and session memory",
                dependencies=[step_num - 1],
                agent="documentation",
                estimated_complexity="low",
            )
        )

        # Add architectural considerations
        if structure.get("entry_points"):
            plan.architectural_considerations.append(
                f"Entry points: {', '.join(structure['entry_points'][:3])}"
            )

        if ".py" in structure.get("file_types", {}):
            py_count = len(structure["file_types"][".py"])
            plan.architectural_considerations.append(
                f"Python files to consider: {py_count}"
            )

        return plan

    def _identify_risks(
        self, context: AgentContext, plan: ImplementationPlan
    ) -> list[str]:
        """Identify risks in the plan.

        Args:
            context: Execution context.
            plan: Implementation plan.

        Returns:
            List of identified risks.
        """
        risks: list[str] = []

        # Check for breaking changes risk
        if len(plan.critical_files) > 5:
            risks.append(
                f"Modifying {len(plan.critical_files)} files increases risk of breaking changes"
            )

        # Check for test coverage risk
        test_agent_steps = [s for s in plan.steps if s.agent == "testing"]
        if not test_agent_steps:
            risks.append("No testing steps in plan - untested changes risk")

        # Check constraint violations risk
        hard_constraints = context.config.constraints.hard
        if hard_constraints:
            risks.append(
                f"Must adhere to {len(hard_constraints)} hard constraints"
            )

        # Check for scope risks
        out_of_scope = context.config.permissions.scope.out_of_scope
        if out_of_scope:
            risks.append(
                f"Files matching {len(out_of_scope)} patterns are out of scope"
            )

        return risks

    def analyze_file(self, file_path: Path) -> dict[str, Any]:
        """Analyze a single file for planning purposes.

        Args:
            file_path: Path to the file.

        Returns:
            Analysis dictionary.
        """
        analysis = {
            "path": str(file_path),
            "exists": file_path.exists(),
            "size": 0,
            "imports": [],
            "classes": [],
            "functions": [],
            "todos": [],
        }

        if not file_path.exists():
            return analysis

        try:
            content = file_path.read_text()
            analysis["size"] = len(content)

            # Extract imports
            import_pattern = re.compile(r"^(?:from|import)\s+(\S+)", re.MULTILINE)
            analysis["imports"] = import_pattern.findall(content)

            # Extract class definitions
            class_pattern = re.compile(r"^class\s+(\w+)", re.MULTILINE)
            analysis["classes"] = class_pattern.findall(content)

            # Extract function definitions
            func_pattern = re.compile(r"^def\s+(\w+)", re.MULTILINE)
            analysis["functions"] = func_pattern.findall(content)

            # Extract TODOs
            todo_pattern = re.compile(r"#\s*TODO[:\s]+(.+)$", re.MULTILINE | re.IGNORECASE)
            analysis["todos"] = todo_pattern.findall(content)

        except Exception as e:
            analysis["error"] = str(e)

        return analysis
