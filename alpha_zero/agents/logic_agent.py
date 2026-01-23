"""LogicAgent: A strategic reasoning agent with uncertainty quantification.

This module implements an agent architecture inspired by the "literal android"
archetype - a purely logical entity that:
1. Observes and learns from patterns (style transfer)
2. Enforces strict honesty constraints (no confident hallucinations)
3. Follows hierarchical utility (user/mission success > agent efficiency)
4. Maintains strategic memory for pattern recognition

The agent wraps MCTS/neural network outputs with:
- Confidence thresholding (refuses to act on uncertain information)
- Context observation (tracks game phase, threats, opportunities)
- Sacrifice logic (will recommend costly moves if strategically necessary)
- Pattern imitation (learns from successful strategies via RAG)

Usage:
    from alpha_zero.agents import LogicAgent, LogicAgentConfig

    config = LogicAgentConfig(confidence_threshold=0.85)
    agent = LogicAgent(config, rag_integration=rag)

    # Get decision with uncertainty quantification
    decision = agent.evaluate_position(state, mcts_stats)
    if decision.confidence >= config.confidence_threshold:
        action = decision.recommended_action
    else:
        print(f"Uncertain: {decision.uncertainty_report}")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alpha_zero.orchestrator.rag_integration import RAGIntegration

logger = logging.getLogger(__name__)


class GamePhase(Enum):
    """Game phase detection."""

    EARLY = "early"  # Expansion, scouting
    MID = "mid"  # Development, military buildup
    LATE = "late"  # Victory push, decisive actions
    CRISIS = "crisis"  # Under attack, emergency response


class ThreatLevel(Enum):
    """Threat assessment levels."""

    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


class UtilityPriority(Enum):
    """Hierarchical utility priorities (higher = more important)."""

    VICTORY_CONDITION = 100  # Win the game
    SURVIVAL = 90  # Prevent loss
    STRATEGIC_ADVANTAGE = 70  # Gain long-term advantage
    TACTICAL_EFFICIENCY = 50  # Short-term optimization
    RESOURCE_PRESERVATION = 30  # Minimize losses
    AGENT_COMPUTATION = 10  # Self-efficiency (lowest priority)


@dataclass
class LogicAgentConfig:
    """Configuration for LogicAgent.

    Attributes:
        confidence_threshold: Minimum confidence to state facts (0.0-1.0).
            Below this, agent must disclose uncertainty or refuse.
        sacrifice_threshold: Utility gain required to recommend sacrifice.
        observation_memory_size: Max observations to retain.
        pattern_match_threshold: Similarity score for pattern matching.
        enable_uncertainty_disclosure: Whether to explain uncertainty.
        enable_sacrifice_logic: Whether to consider costly recommendations.
    """

    confidence_threshold: float = 0.85
    sacrifice_threshold: float = 0.3
    observation_memory_size: int = 100
    pattern_match_threshold: float = 0.7
    enable_uncertainty_disclosure: bool = True
    enable_sacrifice_logic: bool = True


@dataclass
class GameContext:
    """Game state context analysis (analogous to SocialContext).

    Represents the agent's "observation" of the game state,
    extracting metadata beyond raw numbers.

    Attributes:
        phase: Detected game phase.
        threat_level: Current threat assessment.
        opportunity_detected: Whether a strategic opportunity exists.
        detected_patterns: Patterns recognized from knowledge base.
        player_position: Relative strength vs opponents.
        turn_pressure: Urgency to act (0.0=no pressure, 1.0=critical).
    """

    phase: GamePhase = GamePhase.EARLY
    threat_level: ThreatLevel = ThreatLevel.NONE
    opportunity_detected: bool = False
    detected_patterns: list[str] = field(default_factory=list)
    player_position: str = "unknown"
    turn_pressure: float = 0.0


@dataclass
class UncertaintyReport:
    """Detailed uncertainty quantification.

    Provides transparency about why the agent is uncertain,
    implementing the "honesty constraint" that prevents
    confident-sounding hallucinations.

    Attributes:
        confidence_score: Overall confidence (0.0-1.0).
        value_variance: Variance in value estimates.
        policy_entropy: Entropy of action distribution.
        visit_ratio: Ratio of visits to best vs total.
        uncertainty_sources: Human-readable uncertainty explanations.
        recommendation: What the agent recommends given uncertainty.
    """

    confidence_score: float
    value_variance: float = 0.0
    policy_entropy: float = 0.0
    visit_ratio: float = 0.0
    uncertainty_sources: list[str] = field(default_factory=list)
    recommendation: str = ""

    def is_confident(self, threshold: float) -> bool:
        """Check if confidence exceeds threshold."""
        return self.confidence_score >= threshold

    def explain(self) -> str:
        """Generate human-readable uncertainty explanation."""
        if self.is_confident(0.85):
            return f"High confidence ({self.confidence_score:.2%})"

        explanations = []
        if self.value_variance > 0.1:
            explanations.append(f"value estimates vary (σ²={self.value_variance:.3f})")
        if self.policy_entropy > 2.0:
            explanations.append(f"many viable actions (H={self.policy_entropy:.2f})")
        if self.visit_ratio < 0.3:
            explanations.append(f"insufficient exploration (ratio={self.visit_ratio:.2%})")

        explanations.extend(self.uncertainty_sources)

        if not explanations:
            explanations.append("general uncertainty in position evaluation")

        return f"Confidence {self.confidence_score:.2%}: " + "; ".join(explanations)


@dataclass
class StrategicDecision:
    """Strategic decision with full transparency.

    This is the agent's output, combining the recommended action
    with uncertainty quantification and strategic reasoning.

    Attributes:
        recommended_action: The action index to take.
        confidence: Confidence in this recommendation.
        uncertainty_report: Detailed uncertainty analysis.
        reasoning: Strategic reasoning behind the choice.
        alternatives: Other considered actions with their merits.
        sacrifice_involved: Whether this involves strategic sacrifice.
        utility_priority: What priority level this serves.
        context: Game context that informed this decision.
    """

    recommended_action: int
    confidence: float
    uncertainty_report: UncertaintyReport
    reasoning: str
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    sacrifice_involved: bool = False
    utility_priority: UtilityPriority = UtilityPriority.TACTICAL_EFFICIENCY
    context: GameContext | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def is_actionable(self, threshold: float) -> bool:
        """Check if decision meets confidence threshold."""
        return self.confidence >= threshold

    def get_disclosure(self) -> str:
        """Get uncertainty disclosure if confidence is low."""
        if self.confidence >= 0.99:
            return ""
        return self.uncertainty_report.explain()


@dataclass
class StrategicObservation:
    """An observation stored in strategic memory.

    Implements the "social observation" pattern where the agent
    learns from interactions to refine future behavior.
    """

    trigger: str
    observation: str
    pattern_type: str
    confidence: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class LogicAgent:
    """Strategic reasoning agent with honesty constraints.

    Implements a "literal android" architecture that:
    1. Never overstates confidence (honesty constraint)
    2. Observes and learns patterns (style transfer)
    3. Prioritizes mission success over efficiency (sacrifice logic)
    4. Maintains strategic memory (social observation)

    This agent wraps MCTS/neural network outputs to provide
    uncertainty-aware strategic recommendations.

    Attributes:
        config: Agent configuration.
        rag: Optional RAG integration for pattern matching.
    """

    def __init__(
        self,
        config: LogicAgentConfig | None = None,
        rag: RAGIntegration | None = None,
        name: str = "LogicAgent-001",
    ):
        """Initialize LogicAgent.

        Args:
            config: Agent configuration. Uses defaults if None.
            rag: RAG integration for knowledge retrieval.
            name: Agent identifier.
        """
        self.name = name
        self.config = config or LogicAgentConfig()
        self.rag = rag

        # Strategic memory (observation log)
        self._strategic_memory: list[StrategicObservation] = []

        # Pattern cache from RAG queries
        self._pattern_cache: dict[str, list[str]] = {}

        # Statistics
        self._decisions_made = 0
        self._sacrifices_recommended = 0
        self._uncertainty_disclosures = 0

    def analyze_context(self, state: dict[str, Any]) -> GameContext:
        """Analyze game state to extract strategic context.

        This is analogous to _analyze_social_cues - extracting
        metadata beyond the raw data to understand the situation.

        Args:
            state: Game state dictionary.

        Returns:
            GameContext with phase, threats, opportunities.
        """
        context = GameContext()

        # Detect game phase based on turn and development
        turn = state.get("turn", 0)
        units = len(state.get("units", []))
        cities = len(state.get("cities", []))

        if turn < 30 or cities < 3:
            context.phase = GamePhase.EARLY
        elif turn < 150 or cities < 8:
            context.phase = GamePhase.MID
        else:
            context.phase = GamePhase.LATE

        # Assess threat level
        enemy_military = state.get("enemy_military_strength", 0)
        our_military = state.get("military_strength", 0)

        if enemy_military > our_military * 2:
            context.threat_level = ThreatLevel.CRITICAL
            context.phase = GamePhase.CRISIS
        elif enemy_military > our_military * 1.5:
            context.threat_level = ThreatLevel.HIGH
        elif enemy_military > our_military:
            context.threat_level = ThreatLevel.MODERATE
        elif enemy_military > our_military * 0.5:
            context.threat_level = ThreatLevel.LOW
        else:
            context.threat_level = ThreatLevel.NONE

        # Detect opportunities
        if state.get("enemy_exposed_city"):
            context.opportunity_detected = True
            context.detected_patterns.append("vulnerable_target")

        if state.get("tech_advantage"):
            context.opportunity_detected = True
            context.detected_patterns.append("timing_attack_window")

        # Calculate turn pressure
        max_turns = state.get("max_turns", 500)
        if max_turns > 0:
            context.turn_pressure = min(1.0, turn / max_turns)

        # Query RAG for relevant patterns
        if self.rag and self.rag.initialized:
            self._enrich_context_from_rag(context, state)

        # Store observation
        self._store_observation(
            trigger=f"turn_{turn}_analysis",
            observation=f"Phase: {context.phase.value}, Threat: {context.threat_level.name}",
            pattern_type="context_analysis",
            confidence=0.8,
        )

        return context

    def _enrich_context_from_rag(
        self, context: GameContext, state: dict[str, Any]
    ) -> None:
        """Enrich context with RAG knowledge.

        This implements "imitation learning" - finding similar
        situations from the knowledge base.
        """
        if not self.rag:
            return

        try:
            # Build query from current situation
            query_parts = []
            if context.phase == GamePhase.CRISIS:
                query_parts.append("defensive strategy under attack")
            elif context.opportunity_detected:
                query_parts.append("offensive timing attack")
            else:
                query_parts.append(f"{context.phase.value} game strategy")

            if context.threat_level.value >= ThreatLevel.HIGH.value:
                query_parts.append("dealing with military threat")

            query = " ".join(query_parts)

            # Query RAG
            results = self.rag.query_strategy(query, game_context=state, top_k=3)

            for result in results:
                if result.score >= self.config.pattern_match_threshold:
                    # Extract pattern name from content
                    pattern = result.content[:50].strip()
                    context.detected_patterns.append(pattern)

        except Exception as e:
            logger.debug(f"RAG enrichment failed: {e}")

    def quantify_uncertainty(
        self,
        mcts_stats: dict[str, Any],
        policy: list[float] | None = None,
    ) -> UncertaintyReport:
        """Quantify uncertainty from MCTS statistics.

        This implements the "honesty constraint" - measuring
        how confident we should actually be.

        Args:
            mcts_stats: MCTS search statistics.
            policy: Policy distribution from network.

        Returns:
            UncertaintyReport with detailed uncertainty analysis.
        """
        sources = []

        # Extract MCTS statistics
        visit_count = mcts_stats.get("visit_count", 0)
        total_visits = mcts_stats.get("total_visits", 1)
        value = mcts_stats.get("value", 0.0)
        value_variance = mcts_stats.get("value_variance", 0.0)

        # Calculate visit ratio (how focused was the search?)
        visit_ratio = visit_count / max(total_visits, 1)
        if visit_ratio < 0.3:
            sources.append("search was widely distributed")

        # Calculate policy entropy (how decisive is the policy?)
        policy_entropy = 0.0
        if policy:
            policy_entropy = self._calculate_entropy(policy)
            if policy_entropy > 2.0:
                sources.append("many actions appear viable")

        # Check value variance
        if value_variance > 0.1:
            sources.append("position evaluation is uncertain")

        # Check visit count (insufficient exploration)
        if visit_count < 50:
            sources.append("insufficient search depth")

        # Calculate overall confidence
        # High confidence requires: focused search, low entropy, low variance
        confidence = self._compute_confidence(
            visit_ratio=visit_ratio,
            entropy=policy_entropy,
            variance=value_variance,
            visits=visit_count,
        )

        # Generate recommendation based on uncertainty
        if confidence >= self.config.confidence_threshold:
            recommendation = "Proceed with recommended action"
        elif confidence >= 0.5:
            recommendation = "Proceed with caution, consider alternatives"
        else:
            recommendation = "Insufficient confidence - gather more information"

        return UncertaintyReport(
            confidence_score=confidence,
            value_variance=value_variance,
            policy_entropy=policy_entropy,
            visit_ratio=visit_ratio,
            uncertainty_sources=sources,
            recommendation=recommendation,
        )

    def _calculate_entropy(self, distribution: list[float]) -> float:
        """Calculate Shannon entropy of probability distribution."""
        entropy = 0.0
        for p in distribution:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _compute_confidence(
        self,
        visit_ratio: float,
        entropy: float,
        variance: float,
        visits: int,
    ) -> float:
        """Compute overall confidence score.

        Combines multiple uncertainty signals into a single score.
        """
        # Base confidence from visit ratio
        conf_visit = min(1.0, visit_ratio * 2)

        # Entropy penalty (lower entropy = more confident)
        max_entropy = 5.0  # Log2 of ~32 actions
        conf_entropy = max(0.0, 1.0 - entropy / max_entropy)

        # Variance penalty
        conf_variance = max(0.0, 1.0 - variance * 5)

        # Visit count bonus
        conf_visits = min(1.0, visits / 100)

        # Weighted combination
        confidence = (
            0.3 * conf_visit
            + 0.3 * conf_entropy
            + 0.2 * conf_variance
            + 0.2 * conf_visits
        )

        return max(0.0, min(1.0, confidence))

    def evaluate_sacrifice(
        self,
        action: int,
        state: dict[str, Any],
        mcts_stats: dict[str, Any],
        context: GameContext,
    ) -> tuple[bool, UtilityPriority, str]:
        """Evaluate if an action involves strategic sacrifice.

        This implements the "sacrifice logic" - the agent will
        recommend costly actions if they serve higher priorities.

        Args:
            action: Action to evaluate.
            state: Current game state.
            mcts_stats: MCTS statistics.
            context: Strategic context.

        Returns:
            Tuple of (involves_sacrifice, priority_served, reasoning).
        """
        if not self.config.enable_sacrifice_logic:
            return False, UtilityPriority.TACTICAL_EFFICIENCY, ""

        # Check if action involves unit loss
        action_cost = mcts_stats.get("action_cost", 0.0)
        expected_value_gain = mcts_stats.get("expected_value", 0.0)

        involves_sacrifice = action_cost > 0.2  # Significant cost

        if not involves_sacrifice:
            return False, UtilityPriority.TACTICAL_EFFICIENCY, ""

        # Determine what priority this serves
        reasoning_parts = []

        # Victory condition check
        if context.turn_pressure > 0.8 and expected_value_gain > 0.5:
            priority = UtilityPriority.VICTORY_CONDITION
            reasoning_parts.append("Time pressure demands decisive action")

        # Survival check
        elif context.threat_level == ThreatLevel.CRITICAL:
            priority = UtilityPriority.SURVIVAL
            reasoning_parts.append("Survival requires this sacrifice")

        # Strategic advantage check
        elif context.opportunity_detected and expected_value_gain > self.config.sacrifice_threshold:
            priority = UtilityPriority.STRATEGIC_ADVANTAGE
            reasoning_parts.append(
                f"Expected gain ({expected_value_gain:.2f}) exceeds sacrifice threshold"
            )

        else:
            # Sacrifice not justified
            priority = UtilityPriority.RESOURCE_PRESERVATION
            reasoning_parts.append("Sacrifice not strategically justified")
            involves_sacrifice = False

        reasoning = "; ".join(reasoning_parts)

        if involves_sacrifice:
            self._sacrifices_recommended += 1
            self._store_observation(
                trigger=f"sacrifice_action_{action}",
                observation=reasoning,
                pattern_type="sacrifice_decision",
                confidence=0.9,
            )

        return involves_sacrifice, priority, reasoning

    def evaluate_position(
        self,
        state: dict[str, Any],
        mcts_stats: dict[str, Any],
        policy: list[float] | None = None,
    ) -> StrategicDecision:
        """Evaluate position and produce strategic decision.

        Main entry point that combines all analysis components:
        1. Context analysis
        2. Uncertainty quantification
        3. Sacrifice evaluation
        4. Decision synthesis

        Args:
            state: Game state dictionary.
            mcts_stats: MCTS search statistics.
            policy: Optional policy distribution.

        Returns:
            StrategicDecision with full transparency.
        """
        self._decisions_made += 1

        # 1. Analyze context
        context = self.analyze_context(state)

        # 2. Quantify uncertainty
        uncertainty = self.quantify_uncertainty(mcts_stats, policy)

        # 3. Get recommended action
        action = mcts_stats.get("best_action", 0)

        # 4. Evaluate sacrifice
        sacrifice_involved, priority, sacrifice_reasoning = self.evaluate_sacrifice(
            action, state, mcts_stats, context
        )

        # 5. Build reasoning
        reasoning_parts = []

        if context.phase == GamePhase.CRISIS:
            reasoning_parts.append(f"CRISIS mode: {context.threat_level.name} threat")

        if context.opportunity_detected:
            patterns = ", ".join(context.detected_patterns[:3])
            reasoning_parts.append(f"Opportunity patterns: {patterns}")

        if sacrifice_involved:
            reasoning_parts.append(f"Sacrifice: {sacrifice_reasoning}")

        # Add honesty disclosure if uncertain
        if not uncertainty.is_confident(self.config.confidence_threshold):
            self._uncertainty_disclosures += 1
            if self.config.enable_uncertainty_disclosure:
                reasoning_parts.append(f"UNCERTAINTY: {uncertainty.explain()}")

        reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Standard tactical evaluation"

        # 6. Identify alternatives
        alternatives = self._identify_alternatives(mcts_stats)

        return StrategicDecision(
            recommended_action=action,
            confidence=uncertainty.confidence_score,
            uncertainty_report=uncertainty,
            reasoning=reasoning,
            alternatives=alternatives,
            sacrifice_involved=sacrifice_involved,
            utility_priority=priority,
            context=context,
        )

    def _identify_alternatives(self, mcts_stats: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify alternative actions from MCTS stats."""
        alternatives = []

        action_values = mcts_stats.get("action_values", {})
        action_visits = mcts_stats.get("action_visits", {})
        best_action = mcts_stats.get("best_action", 0)

        # Sort by visits (most explored alternatives)
        sorted_actions = sorted(
            action_visits.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for action, visits in sorted_actions[:5]:
            if action != best_action and visits > 0:
                alternatives.append({
                    "action": action,
                    "visits": visits,
                    "value": action_values.get(action, 0.0),
                })

        return alternatives

    def check_honesty_constraint(
        self,
        response: str,
        confidence: float,
    ) -> str:
        """Apply honesty constraint to a response.

        If confidence is below threshold, modifies response
        to disclose uncertainty. Cannot state uncertain
        things as fact.

        Args:
            response: Draft response.
            confidence: Confidence score.

        Returns:
            Response with uncertainty disclosure if needed.
        """
        if confidence >= self.config.confidence_threshold:
            return response

        # Must disclose uncertainty
        return (
            f"[Confidence: {confidence:.2%} - below threshold {self.config.confidence_threshold:.2%}] "
            f"I cannot state this with certainty. "
            f"Based on available analysis: {response}"
        )

    def imitate_strategy(self, pattern_name: str) -> str | None:
        """Attempt to replicate a known strategy pattern.

        This implements "style transfer" - analyzing and
        reconstructing strategic patterns algorithmically.

        Args:
            pattern_name: Name of pattern to imitate.

        Returns:
            Strategy description or None if pattern unknown.
        """
        if not self.rag or not self.rag.initialized:
            return None

        try:
            results = self.rag.query_strategy(pattern_name, top_k=1)
            if results and results[0].score >= self.config.pattern_match_threshold:
                content = results[0].content

                # Extract and reconstruct the pattern
                return (
                    f"Executing pattern: {pattern_name}\n"
                    f"Source: {results[0].source}\n"
                    f"Confidence: {results[0].score:.2%}\n"
                    f"Strategy: {content[:500]}"
                )
        except Exception as e:
            logger.debug(f"Pattern imitation failed: {e}")

        return None

    def _store_observation(
        self,
        trigger: str,
        observation: str,
        pattern_type: str,
        confidence: float,
    ) -> None:
        """Store an observation in strategic memory."""
        obs = StrategicObservation(
            trigger=trigger,
            observation=observation,
            pattern_type=pattern_type,
            confidence=confidence,
        )

        self._strategic_memory.append(obs)

        # Trim memory if over limit
        if len(self._strategic_memory) > self.config.observation_memory_size:
            self._strategic_memory = self._strategic_memory[-self.config.observation_memory_size:]

        # Store in RAG if available
        if self.rag and self.rag.initialized:
            try:
                self.rag.store_training_insight(
                    f"[{pattern_type}] {observation}",
                    metadata={"trigger": trigger, "confidence": confidence},
                )
            except Exception:
                pass  # Non-critical

    def get_memory(self, pattern_type: str | None = None) -> list[StrategicObservation]:
        """Retrieve observations from strategic memory.

        Args:
            pattern_type: Filter by pattern type, or None for all.

        Returns:
            List of matching observations.
        """
        if pattern_type is None:
            return list(self._strategic_memory)

        return [
            obs for obs in self._strategic_memory
            if obs.pattern_type == pattern_type
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "name": self.name,
            "decisions_made": self._decisions_made,
            "sacrifices_recommended": self._sacrifices_recommended,
            "uncertainty_disclosures": self._uncertainty_disclosures,
            "memory_size": len(self._strategic_memory),
            "confidence_threshold": self.config.confidence_threshold,
            "has_rag": self.rag is not None,
        }

    def __repr__(self) -> str:
        return (
            f"LogicAgent("
            f"name={self.name!r}, "
            f"decisions={self._decisions_made}, "
            f"threshold={self.config.confidence_threshold})"
        )
