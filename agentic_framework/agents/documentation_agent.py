"""
Documentation Agent for Agentic Framework

Updates documentation and session memory (CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from .base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from ..core.session_memory import SessionMemory, ArchitectureDecision, KnownIssue
from ..utils.logging_config import get_logger

logger = get_logger("documentation_agent")


@dataclass
class DocumentationUpdate:
    """Represents an update to documentation."""

    section: str
    content: str
    action: str = "add"  # add, replace, remove

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section": self.section,
            "action": self.action,
            "content_preview": self.content[:100] if self.content else "",
        }


class DocumentationAgent(BaseAgent):
    """Agent that updates documentation.

    Features:
    - Session memory (CLAUDE.md) updates
    - Architecture decision recording
    - Known issue tracking
    - Change summary generation
    """

    def __init__(
        self,
        timeout_seconds: float = 120.0,
        auto_update_memory: bool = True,
    ) -> None:
        """Initialize the documentation agent.

        Args:
            timeout_seconds: Timeout for documentation operations.
            auto_update_memory: Automatically update session memory.
        """
        super().__init__(
            name="documentation",
            description="Updates documentation and session memory",
            timeout_seconds=timeout_seconds,
        )
        self.auto_update_memory = auto_update_memory
        self._updates: list[DocumentationUpdate] = []

    def _execute(self, context: AgentContext) -> AgentResult:
        """Execute documentation updates.

        Args:
            context: Execution context.

        Returns:
            AgentResult with documentation outcome.
        """
        self.report_progress(0.0, "Starting documentation")
        self._updates.clear()

        # Load or create session memory
        memory = context.session_memory
        if not memory.exists():
            memory.create_default()

        # Gather information from shared data
        self.report_progress(0.2, "Gathering information")

        # Add any new build/test commands discovered
        self._update_commands(context, memory)

        # Add architecture decisions if any
        self.report_progress(0.4, "Recording decisions")
        self._update_architecture_decisions(context, memory)

        # Add known issues if any
        self.report_progress(0.6, "Recording issues")
        self._update_known_issues(context, memory)

        # Generate change summary
        self.report_progress(0.8, "Generating summary")
        change_summary = self._generate_change_summary(context)

        # Save session memory
        if self.auto_update_memory and memory.is_modified:
            memory.save()
            logger.info("Saved session memory")

        self.report_progress(1.0, "Documentation complete")

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.COMPLETED,
            success=True,
            message=f"Made {len(self._updates)} documentation updates",
            data={
                "updates": [u.to_dict() for u in self._updates],
                "change_summary": change_summary,
                "memory_file": str(memory.memory_file),
            },
            files_modified=[memory.memory_file] if memory.is_modified else [],
        )

    def _update_commands(
        self, context: AgentContext, memory: SessionMemory
    ) -> None:
        """Update build and test commands.

        Args:
            context: Execution context.
            memory: Session memory to update.
        """
        # Add verification commands as test commands
        for vc in context.config.feedback.verification_commands:
            if vc.command and vc.name:
                # Check if already exists
                existing = memory.get_test_commands()
                if not any(vc.command in str(cmd) for cmd in existing):
                    memory.add_test_command(vc.command, vc.name)
                    self._updates.append(DocumentationUpdate(
                        section="Test Commands",
                        content=f"{vc.command}: {vc.name}",
                        action="add",
                    ))

    def _update_architecture_decisions(
        self, context: AgentContext, memory: SessionMemory
    ) -> None:
        """Update architecture decisions.

        Args:
            context: Execution context.
            memory: Session memory to update.
        """
        # Get new decisions from shared data
        new_decisions = context.get_shared("architecture_decisions", [])

        for decision_data in new_decisions:
            if isinstance(decision_data, dict):
                decision = ArchitectureDecision(
                    date=decision_data.get("date", datetime.now().strftime("%Y-%m-%d")),
                    decision=decision_data.get("decision", ""),
                    rationale=decision_data.get("rationale", ""),
                )
            elif isinstance(decision_data, ArchitectureDecision):
                decision = decision_data
            else:
                continue

            memory.add_architecture_decision(decision)
            self._updates.append(DocumentationUpdate(
                section="Architecture Decisions",
                content=decision.decision,
                action="add",
            ))

    def _update_known_issues(
        self, context: AgentContext, memory: SessionMemory
    ) -> None:
        """Update known issues.

        Args:
            context: Execution context.
            memory: Session memory to update.
        """
        # Get issues from shared data
        new_issues = context.get_shared("known_issues", [])

        for issue_data in new_issues:
            if isinstance(issue_data, dict):
                issue = KnownIssue(
                    issue=issue_data.get("issue", ""),
                    workaround=issue_data.get("workaround", ""),
                    status=issue_data.get("status", "open"),
                )
            elif isinstance(issue_data, KnownIssue):
                issue = issue_data
            else:
                continue

            memory.add_known_issue(issue)
            self._updates.append(DocumentationUpdate(
                section="Known Issues",
                content=issue.issue,
                action="add",
            ))

    def _generate_change_summary(self, context: AgentContext) -> str:
        """Generate a summary of all changes.

        Args:
            context: Execution context.

        Returns:
            Markdown summary.
        """
        lines = ["# Change Summary", "", f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]

        # Objective
        objective = context.config.objective.system_intent
        if objective:
            lines.extend(["## Objective", objective, ""])

        # Files modified
        files_modified = context.get_shared("files_modified", [])
        if files_modified:
            lines.append("## Files Modified")
            for f in files_modified[:10]:
                lines.append(f"- `{f}`")
            if len(files_modified) > 10:
                lines.append(f"- ... and {len(files_modified) - 10} more")
            lines.append("")

        # Test results
        test_results = context.get_shared("test_results", {})
        if test_results:
            lines.extend([
                "## Test Results",
                f"- Passed: {test_results.get('passed', 0)}",
                f"- Failed: {test_results.get('failed', 0)}",
                "",
            ])

        # Verification status
        verification_results = context.get_shared("verification_results", [])
        if verification_results:
            lines.append("## Verification")
            for vr in verification_results:
                status = "PASSED" if vr.get("success") else "FAILED"
                lines.append(f"- {vr.get('name', 'unknown')}: {status}")
            lines.append("")

        # Success criteria
        success_criteria = context.config.objective.success_criteria
        if success_criteria:
            lines.append("## Success Criteria")
            for criterion in success_criteria:
                criterion_text = criterion.get("criterion", str(criterion))
                # TODO: Actually check if criteria are met
                lines.append(f"- [ ] {criterion_text}")
            lines.append("")

        return "\n".join(lines)

    def record_decision(
        self,
        context: AgentContext,
        decision: str,
        rationale: str,
    ) -> None:
        """Record an architecture decision.

        Args:
            context: Execution context.
            decision: Description of the decision.
            rationale: Why this decision was made.
        """
        arch_decision = ArchitectureDecision(
            date=datetime.now().strftime("%Y-%m-%d"),
            decision=decision,
            rationale=rationale,
        )

        # Add to shared data for persistence
        decisions = context.get_shared("architecture_decisions", [])
        decisions.append(arch_decision)
        context.set_shared("architecture_decisions", decisions)

        logger.info(f"Recorded architecture decision: {decision}")

    def record_issue(
        self,
        context: AgentContext,
        issue: str,
        workaround: str,
    ) -> None:
        """Record a known issue.

        Args:
            context: Execution context.
            issue: Description of the issue.
            workaround: Workaround or notes.
        """
        known_issue = KnownIssue(
            issue=issue,
            workaround=workaround,
            status="open",
        )

        # Add to shared data for persistence
        issues = context.get_shared("known_issues", [])
        issues.append(known_issue)
        context.set_shared("known_issues", issues)

        logger.info(f"Recorded known issue: {issue}")

    def add_gotcha(self, context: AgentContext, gotcha: str) -> None:
        """Add a gotcha/tip.

        Args:
            context: Execution context.
            gotcha: Description of the gotcha.
        """
        gotchas = context.get_shared("gotchas", [])
        gotchas.append(gotcha)
        context.set_shared("gotchas", gotchas)

        logger.info(f"Added gotcha: {gotcha}")

    def get_documentation_summary(self) -> str:
        """Get a summary of documentation updates.

        Returns:
            Summary string.
        """
        if not self._updates:
            return "No documentation updates made"

        by_section: dict[str, list[DocumentationUpdate]] = {}
        for update in self._updates:
            if update.section not in by_section:
                by_section[update.section] = []
            by_section[update.section].append(update)

        lines = ["## Documentation Updates", ""]
        for section, updates in by_section.items():
            lines.append(f"### {section}")
            for update in updates:
                lines.append(f"- [{update.action}] {update.content[:50]}...")
            lines.append("")

        return "\n".join(lines)

    @property
    def updates(self) -> list[DocumentationUpdate]:
        """Get all documentation updates."""
        return self._updates
