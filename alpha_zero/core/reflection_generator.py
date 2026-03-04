"""LLM-based reflection generation for failed games.

This module generates verbal self-critiques for lost games using the existing
LLM client infrastructure. These critiques can be stored in ReflectionMemory
to help avoid repeating mistakes.

Usage:
    from alpha_zero.core.reflection_generator import (
        ReflectionGenerator,
        ReflectionGeneratorConfig,
    )

    # Create generator
    generator = ReflectionGenerator()

    # Generate critique for a lost game
    critique = await generator.generate_critique(trajectory)
    print(critique.summary)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from alpha_zero.distributed.shared_storage import GameTrajectory

if TYPE_CHECKING:
    from alpha_zero.envs.microciv import MicroCivActionSpace
    from alpha_zero.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ReflectionGeneratorConfig:
    """Configuration for LLM-based reflection generation."""

    # LLM settings
    llm_provider: str = "auto"  # "auto", "openai", "anthropic"
    model: str | None = None  # None = use provider default
    temperature: float = 0.3  # Lower for more focused critique
    max_tokens: int = 500

    # Prompt settings
    include_full_trajectory: bool = False  # Full game or just key moments
    max_trajectory_states: int = 10  # Truncate long games
    include_game_context: bool = True

    # Retry settings
    max_retries: int = 3
    timeout: float = 30.0

    # Caching
    cache_critiques: bool = True
    cache_ttl: float = 3600.0  # 1 hour

    # Fallback
    use_fallback_on_failure: bool = True

    @classmethod
    def from_env(cls) -> ReflectionGeneratorConfig:
        """Load configuration from environment variables."""
        import os

        return cls(
            llm_provider=os.getenv("CIV6AZ_REFLECTION_LLM_PROVIDER", "auto"),
            temperature=float(os.getenv("CIV6AZ_REFLECTION_LLM_TEMP", "0.3")),
            max_tokens=int(os.getenv("CIV6AZ_REFLECTION_LLM_MAX_TOKENS", "500")),
            max_trajectory_states=int(os.getenv("CIV6AZ_REFLECTION_MAX_STATES", "10")),
            cache_critiques=os.getenv("CIV6AZ_REFLECTION_CACHE", "true").lower() == "true",
        )


@dataclass
class GeneratedCritique:
    """Structured critique from LLM analysis."""

    summary: str = ""  # One-line summary of what went wrong
    key_mistake: str = ""  # Description of the critical error
    mistake_action: int | None = None  # Action index if identified
    suggested_alternative: str = ""  # What should have been done
    strategic_lesson: str = ""  # General principle to remember
    confidence: float = 0.5  # LLM's confidence in analysis (0-1)
    raw_response: str = ""  # Full LLM response for debugging

    @property
    def full_critique(self) -> str:
        """Get full critique text."""
        parts = []
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.key_mistake:
            parts.append(f"Key Mistake: {self.key_mistake}")
        if self.suggested_alternative:
            parts.append(f"Alternative: {self.suggested_alternative}")
        if self.strategic_lesson:
            parts.append(f"Lesson: {self.strategic_lesson}")
        return "\n".join(parts) or self.raw_response


class GameStateDescriber(Protocol):
    """Protocol for converting game state to text."""

    def describe_state(self, obs: dict[str, np.ndarray]) -> str:
        """Convert observation to human-readable description."""
        ...

    def describe_action(self, action: int) -> str:
        """Convert action index to human-readable description."""
        ...

    def describe_trajectory(self, trajectory: GameTrajectory) -> str:
        """Convert full trajectory to text summary."""
        ...


class DefaultGameStateDescriber:
    """Default implementation for describing game states."""

    def __init__(self, action_space: MicroCivActionSpace | None = None) -> None:
        """Initialize describer.

        Args:
            action_space: Optional action space for decoding actions.
        """
        self.action_space = action_space

    def describe_state(self, obs: dict[str, np.ndarray]) -> str:
        """Convert observation to human-readable description.

        Args:
            obs: Observation with 'vector' and optional 'spatial' keys.

        Returns:
            Text description of the state.
        """
        if "vector" not in obs:
            return "Unknown state"

        vector = obs["vector"]

        # Extract key features (assuming standard MicroCiv encoding)
        # Indices based on typical observation encoding
        try:
            turn_progress = float(vector[0]) if len(vector) > 0 else 0
            resources = float(vector[1]) if len(vector) > 1 else 0
            own_units = float(vector[2]) if len(vector) > 2 else 0
            enemy_units = float(vector[3]) if len(vector) > 3 else 0
            own_cities = float(vector[4]) if len(vector) > 4 else 0
            enemy_cities = float(vector[5]) if len(vector) > 5 else 0

            return (
                f"Turn progress: {turn_progress:.0%}, "
                f"Resources: {resources * 100:.0f}, "
                f"Units: {own_units * 20:.0f} vs {enemy_units * 20:.0f}, "
                f"Cities: {own_cities * 5:.0f} vs {enemy_cities * 5:.0f}"
            )
        except (IndexError, TypeError):
            return "State with limited visibility"

    def describe_action(self, action: int) -> str:
        """Convert action index to description.

        Args:
            action: Action index.

        Returns:
            Text description of the action.
        """
        if self.action_space is None:
            return f"Action {action}"

        try:
            # Decode action using action space
            if action == self.action_space.end_turn_offset:
                return "End Turn"

            if action < self.action_space.move_offset:
                return f"Action {action}"

            if action < self.action_space.attack_offset:
                rel = action - self.action_space.move_offset
                unit_idx = rel // 8
                direction = rel % 8
                directions = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"]
                return f"Move unit {unit_idx} {directions[direction]}"

            if action < self.action_space.skip_offset:
                return f"Attack action {action}"

            if action < self.action_space.found_city_offset:
                return "Skip unit"

            if action < self.action_space.produce_offset:
                return "Found city"

            return f"Produce action {action}"

        except Exception:
            return f"Action {action}"

    def describe_trajectory(
        self,
        trajectory: GameTrajectory,
        max_states: int = 10,
    ) -> str:
        """Convert trajectory to text summary.

        Args:
            trajectory: Game trajectory.
            max_states: Maximum states to include.

        Returns:
            Text summary of the game.
        """
        if not trajectory.examples:
            return "Empty game"

        lines = [
            f"Game {trajectory.game_id}: {trajectory.game_length} turns",
            f"Winner: Player {trajectory.winner}" if trajectory.winner is not None else "Draw",
            "",
            "Key moments:",
        ]

        # Sample key moments from the trajectory
        n_examples = len(trajectory.examples)
        if n_examples <= max_states:
            indices = list(range(n_examples))
        else:
            # Sample evenly + include first and last
            step = n_examples // (max_states - 2)
            indices = [
                0,
                *list(range(step, n_examples - 1, step))[: max_states - 2],
                n_examples - 1,
            ]

        for _i, idx in enumerate(indices):
            example = trajectory.examples[idx]
            obs = {"vector": example.vector, "spatial": example.spatial}
            state_desc = self.describe_state(obs)

            # Get action from policy
            action = int(np.argmax(example.policy))
            action_desc = self.describe_action(action)

            lines.append(f"  [{idx}] {state_desc} -> {action_desc}")

        return "\n".join(lines)


# Default prompt template
CRITIQUE_PROMPT_TEMPLATE = """You are analyzing a failed game in a simplified Civilization-style strategy game.

GAME SUMMARY:
{trajectory_description}

FINAL OUTCOME: Player lost

{game_context}

Analyze this game and provide a structured critique in the following format:

SUMMARY: [One sentence describing what went wrong]

KEY_MISTAKE: [The critical error that led to the loss]

ALTERNATIVE: [What should have been done instead]

LESSON: [A general strategic principle to remember]

Be specific and actionable. Focus on the turning point of the game.
"""


class ReflectionGenerator:
    """Generate verbal self-critiques for failed games using LLM."""

    def __init__(
        self,
        config: ReflectionGeneratorConfig | None = None,
        llm_client: LLMClient | None = None,
        state_describer: GameStateDescriber | None = None,
    ) -> None:
        """Initialize reflection generator.

        Args:
            config: Generator configuration. Uses defaults if None.
            llm_client: LLM client for critique generation. Created from env if None.
            state_describer: Custom state describer for game text conversion.
        """
        self.config = config or ReflectionGeneratorConfig()
        self._llm = llm_client  # Lazy initialization
        self._describer = state_describer or DefaultGameStateDescriber()
        self._logger = logging.getLogger(__name__)
        self._cache: dict[str, GeneratedCritique] = {}

    def _get_llm_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._llm is None:
            try:
                from alpha_zero.utils.llm_client import create_llm_client

                self._llm = create_llm_client()
            except Exception as e:
                self._logger.warning(f"Failed to create LLM client: {e}")
                raise RuntimeError("LLM client not available") from e
        return self._llm

    async def generate_critique(
        self,
        trajectory: GameTrajectory,
        game_context: str | None = None,
    ) -> GeneratedCritique:
        """Generate critique for a lost game.

        Args:
            trajectory: The failed game trajectory.
            game_context: Optional additional context about the game.

        Returns:
            Structured critique from LLM analysis.
        """
        # Check cache
        cache_key = self._compute_cache_key(trajectory)
        if self.config.cache_critiques and cache_key in self._cache:
            self._logger.debug("Using cached critique")
            return self._cache[cache_key]

        # Build prompt
        prompt = self._build_prompt(trajectory, game_context)

        # Generate critique
        try:
            llm = self._get_llm_client()
            response = await llm.generate(
                prompt,
                system_prompt="You are a game analysis expert providing constructive feedback on strategy game performance.",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            raw_response = response.content

        except Exception as e:
            self._logger.warning(f"LLM generation failed: {e}")
            if self.config.use_fallback_on_failure:
                return self._generate_fallback_critique(trajectory)
            raise

        # Parse response
        critique = self._parse_response(raw_response)

        # Cache result
        if self.config.cache_critiques:
            self._cache[cache_key] = critique

        return critique

    def generate_critique_sync(
        self,
        trajectory: GameTrajectory,
        game_context: str | None = None,
    ) -> GeneratedCritique:
        """Synchronous wrapper for generate_critique."""
        return asyncio.run(self.generate_critique(trajectory, game_context))

    def _compute_cache_key(self, trajectory: GameTrajectory) -> str:
        """Compute cache key for trajectory."""
        hasher = hashlib.md5(usedforsecurity=False)
        hasher.update(str(trajectory.game_id).encode())
        hasher.update(str(trajectory.game_length).encode())
        hasher.update(str(trajectory.winner).encode())

        if trajectory.examples:
            # Include first and last state
            hasher.update(trajectory.examples[0].vector.tobytes())
            hasher.update(trajectory.examples[-1].vector.tobytes())

        return hasher.hexdigest()

    def _build_prompt(
        self,
        trajectory: GameTrajectory,
        game_context: str | None,
    ) -> str:
        """Build analysis prompt."""
        # Describe trajectory
        trajectory_description = self._describer.describe_trajectory(  # type: ignore[call-arg]
            trajectory,
            max_states=self.config.max_trajectory_states,
        )

        # Format game context
        context_str = ""
        if game_context and self.config.include_game_context:
            context_str = f"\nADDITIONAL CONTEXT:\n{game_context}"

        return CRITIQUE_PROMPT_TEMPLATE.format(
            trajectory_description=trajectory_description,
            game_context=context_str,
        )

    def _parse_response(self, response: str) -> GeneratedCritique:
        """Parse LLM response into structured critique.

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed GeneratedCritique.
        """
        critique = GeneratedCritique(raw_response=response)

        # Extract sections using regex
        patterns = {
            "summary": r"SUMMARY:\s*(.+?)(?=\n\n|\nKEY_MISTAKE:|$)",
            "key_mistake": r"KEY_MISTAKE:\s*(.+?)(?=\n\n|\nALTERNATIVE:|$)",
            "suggested_alternative": r"ALTERNATIVE:\s*(.+?)(?=\n\n|\nLESSON:|$)",
            "strategic_lesson": r"LESSON:\s*(.+?)(?=\n\n|$)",
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                setattr(critique, field, value)

        # Extract action number if mentioned
        action_match = re.search(r"action\s+(\d+)", response, re.IGNORECASE)
        if action_match:
            critique.mistake_action = int(action_match.group(1))

        # Estimate confidence based on response completeness
        filled_fields = sum(
            1
            for f in [
                critique.summary,
                critique.key_mistake,
                critique.suggested_alternative,
                critique.strategic_lesson,
            ]
            if f
        )
        critique.confidence = filled_fields / 4.0

        # If no structured content found, use raw response as summary
        if not critique.summary and response:
            # Take first sentence or first 100 chars
            first_line = response.split("\n")[0].strip()
            critique.summary = first_line[:200] if len(first_line) > 200 else first_line

        return critique

    def _generate_fallback_critique(
        self,
        trajectory: GameTrajectory,
    ) -> GeneratedCritique:
        """Generate a basic critique without LLM.

        Args:
            trajectory: The failed game trajectory.

        Returns:
            Simple heuristic-based critique.
        """
        self._logger.info("Using fallback critique generation")

        summary = f"Lost after {trajectory.game_length} turns"

        # Analyze trajectory for simple patterns
        if trajectory.examples:
            # Check for early game vs late game loss
            if trajectory.game_length < 50:
                key_mistake = "Early game setback - possibly insufficient expansion or defense"
                alternative = "Focus on building units and cities in the early game"
            else:
                key_mistake = "Late game collapse - possibly overextension or weak economy"
                alternative = "Maintain balanced growth between military and economy"

            lesson = "Adapt strategy based on opponent behavior"
        else:
            key_mistake = "Unable to analyze - no game data available"
            alternative = "Collect more game data for analysis"
            lesson = "Learn from each game by reviewing key decisions"

        return GeneratedCritique(
            summary=summary,
            key_mistake=key_mistake,
            suggested_alternative=alternative,
            strategic_lesson=lesson,
            confidence=0.3,  # Low confidence for fallback
            raw_response="[Fallback critique - LLM unavailable]",
        )

    def clear_cache(self) -> None:
        """Clear the critique cache."""
        self._cache.clear()
        self._logger.debug("Critique cache cleared")


def create_reflection_generator(
    config: ReflectionGeneratorConfig | None = None,
    llm_client: LLMClient | None = None,
) -> ReflectionGenerator:
    """Create a ReflectionGenerator instance.

    Args:
        config: Generator configuration. Uses env defaults if None.
        llm_client: Optional LLM client to use.

    Returns:
        Configured ReflectionGenerator.
    """
    if config is None:
        config = ReflectionGeneratorConfig.from_env()
    return ReflectionGenerator(config=config, llm_client=llm_client)
