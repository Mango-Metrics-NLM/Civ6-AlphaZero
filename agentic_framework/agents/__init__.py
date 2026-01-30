"""
Agentic Framework Agents Module

Provides specialized agents for different phases of the agentic workflow:
- Planning: Analyzes codebase and creates implementation plans
- Implementation: Writes and modifies code
- Testing: Creates and runs tests
- Verification: Runs linting, type checking, and other verifications
- Documentation: Updates documentation and session memory
"""

from .base_agent import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from .documentation_agent import DocumentationAgent
from .implementation_agent import ImplementationAgent
from .planning_agent import PlanningAgent
from .testing_agent import TestingAgent
from .verification_agent import VerificationAgent

__all__ = [
    # Base
    "BaseAgent",
    "AgentContext",
    "AgentResult",
    "AgentStatus",
    # Specialized Agents
    "PlanningAgent",
    "ImplementationAgent",
    "TestingAgent",
    "VerificationAgent",
    "DocumentationAgent",
]
