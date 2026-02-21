"""Patent-novel features adapted for the Atari demo.

Integrates two Tier-1 novel techniques from the core Civ6 project:

1. **RAG-Enhanced MCTS** — Retrieval-augmented generation injected into the
   MCTS search loop.  Domain knowledge documents (Atari game strategies) are
   ingested into a vector store and queried during search to bias policy priors
   toward known-good strategies.

2. **LATS-Style Reflection Memory** — Failed episodes are analysed to produce
   structured critiques.  The critiques are embedded and stored so that during
   future MCTS evaluations, similar failure states can be detected and their
   corresponding bad actions penalised.

Both features use lightweight, in-memory backends by default so no external
services are required for demo usage.

Example:
    >>> from demos.atari.novel_features import (
    ...     AtariReflectionMemory, AtariRAGAdvisor, NovelFeaturesConfig,
    ... )
    >>> cfg = NovelFeaturesConfig(reflection_enabled=True, rag_enabled=True)
    >>> reflection = AtariReflectionMemory(cfg)
    >>> rag = AtariRAGAdvisor(cfg)
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class NovelFeaturesConfig:
    """Configuration for patent-novel features in the Atari demo.

    Attributes:
        reflection_enabled: Enable LATS-style reflection memory.
        rag_enabled: Enable RAG-enhanced MCTS.
        reflection_max_entries: Maximum stored reflection entries.
        reflection_warning_penalty: Value penalty for bad actions.
        reflection_similarity_threshold: Cosine similarity threshold for retrieval.
        reflection_max_retrievals: Maximum reflections returned per query.
        rag_top_k: Number of RAG results to retrieve.
        rag_prior_boost: How much to boost policy priors from RAG.
        embedding_dimension: Dimension of state embeddings.
    """

    reflection_enabled: bool = False
    rag_enabled: bool = False
    reflection_max_entries: int = 5000
    reflection_warning_penalty: float = 0.1
    reflection_similarity_threshold: float = 0.6
    reflection_max_retrievals: int = 5
    rag_top_k: int = 3
    rag_prior_boost: float = 0.1
    embedding_dimension: int = 128


# ============================================================================
# Reflection Entry
# ============================================================================


@dataclass
class AtariReflectionEntry:
    """A stored reflection about a failed Atari episode.

    Attributes:
        id: Unique identifier.
        episode_reward: Total reward of the failed episode.
        failure_step: Step index where the most critical mistake occurred.
        critique: Structured critique string.
        action_taken: The bad action at failure_step.
        state_embedding: Embedding of the failure state.
        state_hash: Hash for deduplication.
        game_name: Atari game name.
        metadata: Additional information.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    episode_reward: float = 0.0
    failure_step: int = 0
    critique: str = ""
    action_taken: int = -1
    state_embedding: np.ndarray | None = None
    state_hash: str = ""
    game_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# State Encoder (lightweight — no external dependencies)
# ============================================================================


class AtariStateEncoder:
    """Encodes Atari observations into fixed-size embeddings.

    Uses a simple projection (average-pool + linear hash) for O(1) encoding
    without requiring a trained encoder network.

    Args:
        embedding_dim: Target embedding dimension.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = embedding_dim
        # Fixed random projection matrix (reproducible via seed)
        rng = np.random.RandomState(42)  # fixed seed for reproducibility
        self._proj: np.ndarray | None = None
        self._rng = rng

    def encode_state(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Encode an observation dict into a fixed-size vector.

        Args:
            obs: Dict with ``spatial`` (C, H, W) and ``vector`` (V,) keys.

        Returns:
            Embedding of shape ``(embedding_dim,)``.
        """
        spatial = obs.get("spatial", np.zeros(1))
        vector = obs.get("vector", np.zeros(1))

        # Flatten and concatenate
        flat = np.concatenate([spatial.ravel(), vector.ravel()]).astype(np.float32)

        # Lazy-init projection matrix to match input size
        if self._proj is None or self._proj.shape[1] != flat.shape[0]:
            self._proj = self._rng.randn(self.embedding_dim, flat.shape[0]).astype(np.float32)
            self._proj /= np.linalg.norm(self._proj, axis=1, keepdims=True) + 1e-8

        # Project and normalise
        emb = self._proj @ flat
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb

    @staticmethod
    def state_hash(obs: dict[str, np.ndarray]) -> str:
        """Compute a fast content hash for deduplication.

        Args:
            obs: Observation dict.

        Returns:
            Hex digest string.
        """
        data = obs.get("spatial", np.zeros(1)).tobytes()
        return hashlib.md5(data, usedforsecurity=False).hexdigest()


# ============================================================================
# Reflection Memory (in-memory, no external deps)
# ============================================================================


class AtariReflectionMemory:
    """LATS-style reflection memory for Atari games.

    Stores critiques of failed episodes and retrieves similar failures
    during MCTS to penalise previously-failed actions.

    This is a lightweight, in-memory implementation that demonstrates the
    patent-novel concept without requiring external vector databases.

    Args:
        config: Novel features configuration.
    """

    def __init__(self, config: NovelFeaturesConfig | None = None) -> None:
        self.config = config or NovelFeaturesConfig()
        self._encoder = AtariStateEncoder(self.config.embedding_dimension)
        self._entries: deque[AtariReflectionEntry] = deque(
            maxlen=self.config.reflection_max_entries
        )
        self.initialized = self.config.reflection_enabled

        logger.info(
            "AtariReflectionMemory: enabled=%s, max_entries=%d",
            self.config.reflection_enabled,
            self.config.reflection_max_entries,
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add_failed_episode(
        self,
        trajectory: list[dict[str, Any]],
        episode_reward: float,
        game_name: str = "",
    ) -> AtariReflectionEntry | None:
        """Analyse and store a failed episode.

        A "failed" episode is one where the reward is below zero or
        significantly below the running average.

        Args:
            trajectory: List of ``{obs, action, reward}`` dicts per step.
            episode_reward: Total episode reward.
            game_name: Name of the Atari game.

        Returns:
            The stored reflection entry, or ``None`` if skipped.
        """
        if not self.config.reflection_enabled or not trajectory:
            return None

        # Identify the worst step (lowest cumulative advantage)
        failure_step, failure_action = self._find_failure_point(trajectory)
        failure_obs = trajectory[failure_step].get("obs", {})

        # Generate heuristic critique
        critique = self._generate_critique(trajectory, failure_step, episode_reward)

        # Create embedding
        embedding = self._encoder.encode_state(failure_obs) if failure_obs else None
        state_hash = self._encoder.state_hash(failure_obs) if failure_obs else ""

        # Deduplicate by state hash
        for existing in self._entries:
            if existing.state_hash == state_hash:
                logger.debug("Duplicate reflection skipped (hash=%s)", state_hash[:8])
                return None

        entry = AtariReflectionEntry(
            episode_reward=episode_reward,
            failure_step=failure_step,
            critique=critique,
            action_taken=failure_action,
            state_embedding=embedding,
            state_hash=state_hash,
            game_name=game_name,
        )
        self._entries.append(entry)

        logger.debug(
            "Reflection stored: step=%d, action=%d, reward=%.2f, critique=%s",
            failure_step,
            failure_action,
            episode_reward,
            critique[:80],
        )
        return entry

    # ------------------------------------------------------------------
    # Retrieval & value adjustment
    # ------------------------------------------------------------------

    def get_value_adjustment(
        self,
        obs: dict[str, np.ndarray],
        action: int,
    ) -> float:
        """Compute a value penalty if the (state, action) resembles a past failure.

        This method is called during MCTS evaluation to penalise actions that
        led to failures in similar states.

        Args:
            obs: Current observation.
            action: Candidate action.

        Returns:
            Negative adjustment (penalty) or 0.0.
        """
        if not self.config.reflection_enabled or not self._entries:
            return 0.0

        query_emb = self._encoder.encode_state(obs)
        penalty = 0.0
        matches = 0

        for entry in self._entries:
            if entry.state_embedding is None:
                continue
            if entry.action_taken != action:
                continue

            sim = float(np.dot(query_emb, entry.state_embedding))
            if sim >= self.config.reflection_similarity_threshold:
                penalty -= self.config.reflection_warning_penalty * sim
                matches += 1
                if matches >= self.config.reflection_max_retrievals:
                    break

        if matches > 0:
            logger.debug(
                "Reflection penalty: action=%d, matches=%d, penalty=%.4f",
                action,
                matches,
                penalty,
            )
        return penalty

    def get_value_adjustment_sync(
        self,
        state: dict[str, Any],
        action: int,
    ) -> float:
        """Synchronous version compatible with ``Civ6MCTS`` reflection protocol.

        Args:
            state: State dict with ``spatial`` and ``vector`` numpy arrays.
            action: Candidate action.

        Returns:
            Value adjustment (negative = penalty).
        """
        obs = {
            "spatial": state.get("spatial", np.zeros(1)),
            "vector": state.get("vector", np.zeros(1)),
        }
        return self.get_value_adjustment(obs, action)

    def query_similar_failures(
        self,
        obs: dict[str, np.ndarray],
        top_k: int | None = None,
    ) -> list[AtariReflectionEntry]:
        """Retrieve reflections most similar to the given observation.

        Args:
            obs: Current observation.
            top_k: Maximum results (defaults to config).

        Returns:
            List of matching reflection entries, sorted by similarity.
        """
        if not self._entries:
            return []

        k = top_k or self.config.reflection_max_retrievals
        query_emb = self._encoder.encode_state(obs)

        scored: list[tuple[float, AtariReflectionEntry]] = []
        for entry in self._entries:
            if entry.state_embedding is None:
                continue
            sim = float(np.dot(query_emb, entry.state_embedding))
            if sim >= self.config.reflection_similarity_threshold:
                scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    # ------------------------------------------------------------------
    # Critique generation (heuristic — no LLM required)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_failure_point(trajectory: list[dict[str, Any]]) -> tuple[int, int]:
        """Identify the step with the most negative reward delta."""
        worst_idx = 0
        worst_reward = float("inf")
        for i, step in enumerate(trajectory):
            r = step.get("reward", 0.0)
            if r < worst_reward:
                worst_reward = r
                worst_idx = i
        action = trajectory[worst_idx].get("action", 0)
        return worst_idx, action

    @staticmethod
    def _generate_critique(
        trajectory: list[dict[str, Any]],
        failure_step: int,
        episode_reward: float,
    ) -> str:
        """Generate a structured heuristic critique.

        In production, this would call an LLM.  For the demo, we produce
        a structured string from the trajectory statistics.
        """
        total_steps = len(trajectory)
        failure_pct = failure_step / max(total_steps, 1) * 100
        action = trajectory[failure_step].get("action", "?")
        reward_at_fail = trajectory[failure_step].get("reward", 0)

        return (
            f"Episode failed (total_reward={episode_reward:.1f}). "
            f"Critical mistake at step {failure_step}/{total_steps} "
            f"({failure_pct:.0f}% through episode). "
            f"Action {action} yielded reward {reward_at_fail}. "
            f"Consider alternative actions at this game state."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of stored reflections."""
        return len(self._entries)

    def reset(self) -> None:
        """Clear all stored reflections."""
        self._entries.clear()


# ============================================================================
# RAG Advisor (in-memory knowledge retrieval)
# ============================================================================


@dataclass
class KnowledgeEntry:
    """A single piece of domain knowledge for RAG.

    Attributes:
        text: The knowledge text.
        embedding: Pre-computed embedding.
        source: Origin of the knowledge.
        game_name: Associated game (or empty for universal).
        action_hints: Action indices this knowledge recommends.
    """

    text: str
    embedding: np.ndarray
    source: str = ""
    game_name: str = ""
    action_hints: list[int] = field(default_factory=list)


class AtariRAGAdvisor:
    """RAG-enhanced MCTS advisor for Atari games.

    Maintains an in-memory knowledge base of game strategies and retrieves
    relevant entries during MCTS search to boost policy priors toward
    known-good actions.

    This demonstrates the patent-novel concept of injecting external
    knowledge into the MCTS search loop via retrieval-augmented generation.

    Args:
        config: Novel features configuration.
    """

    def __init__(self, config: NovelFeaturesConfig | None = None) -> None:
        self.config = config or NovelFeaturesConfig()
        self._encoder = AtariStateEncoder(self.config.embedding_dimension)
        self._knowledge: list[KnowledgeEntry] = []
        self._initialised = False

        logger.info("AtariRAGAdvisor: enabled=%s", self.config.rag_enabled)

    # ------------------------------------------------------------------
    # Knowledge ingestion
    # ------------------------------------------------------------------

    def ingest_knowledge(self, text: str, source: str = "", game_name: str = "",
                         action_hints: list[int] | None = None) -> None:
        """Add a piece of domain knowledge to the store.

        Args:
            text: Strategy description.
            source: Origin of the knowledge (e.g. "expert", "wiki").
            game_name: Associated game.
            action_hints: Action indices this strategy recommends.
        """
        # Create a pseudo-embedding from the text (hash-based, no ML model)
        emb = self._text_to_embedding(text)
        self._knowledge.append(
            KnowledgeEntry(
                text=text,
                embedding=emb,
                source=source,
                game_name=game_name,
                action_hints=action_hints or [],
            )
        )
        logger.debug("Knowledge ingested: %s (source=%s)", text[:60], source)

    def ingest_default_strategies(self, game_name: str) -> None:
        """Ingest built-in strategy hints for common Atari games.

        Args:
            game_name: Atari game name.
        """
        strategies = _DEFAULT_STRATEGIES.get(game_name, [])
        for strategy in strategies:
            self.ingest_knowledge(
                text=strategy["text"],
                source="built-in",
                game_name=game_name,
                action_hints=strategy.get("action_hints", []),
            )
        self._initialised = True
        logger.info("Ingested %d built-in strategies for %s", len(strategies), game_name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        obs: dict[str, np.ndarray],
        game_name: str = "",
        top_k: int | None = None,
    ) -> list[KnowledgeEntry]:
        """Retrieve relevant knowledge entries for the current state.

        Args:
            obs: Current observation.
            game_name: Filter by game (empty = all).
            top_k: Maximum results.

        Returns:
            List of matching knowledge entries.
        """
        if not self.config.rag_enabled or not self._knowledge:
            return []

        k = top_k or self.config.rag_top_k
        query_emb = self._encoder.encode_state(obs)

        scored: list[tuple[float, KnowledgeEntry]] = []
        for entry in self._knowledge:
            if game_name and entry.game_name and entry.game_name != game_name:
                continue
            sim = float(np.dot(query_emb, entry.embedding))
            scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    def get_policy_boost(
        self,
        obs: dict[str, np.ndarray],
        num_actions: int,
        game_name: str = "",
    ) -> np.ndarray:
        """Compute a policy-prior boost vector from retrieved knowledge.

        The returned vector is a small additive adjustment to be applied
        to the MCTS root policy priors before search begins.

        Args:
            obs: Current observation.
            num_actions: Size of the action space.
            game_name: Filter by game.

        Returns:
            Boost vector of shape ``(num_actions,)``, in [0, boost_scale].
        """
        boost = np.zeros(num_actions, dtype=np.float32)
        if not self.config.rag_enabled:
            return boost

        entries = self.query(obs, game_name=game_name)
        if not entries:
            return boost

        for entry in entries:
            for action_idx in entry.action_hints:
                if 0 <= action_idx < num_actions:
                    boost[action_idx] += self.config.rag_prior_boost

        return boost

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Create a pseudo-embedding from text content.

        Uses deterministic hashing to produce a fixed-size vector.
        In production, this would use a sentence-transformer model.
        """
        # Hash-based embedding (no ML model required)
        dim = self.config.embedding_dimension
        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)  # deterministic per-text seed
        emb = rng.randn(dim).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8
        return emb

    @property
    def size(self) -> int:
        """Number of stored knowledge entries."""
        return len(self._knowledge)


# ============================================================================
# Default game strategies (built-in knowledge base)
# ============================================================================


_DEFAULT_STRATEGIES: dict[str, list[dict[str, Any]]] = {
    "Pong": [
        {
            "text": "In Pong, track the ball vertically and move the paddle to intercept. "
            "When the ball is approaching, position the paddle center to align with "
            "the ball trajectory. Use the paddle edges for angled returns.",
            "action_hints": [2, 3],  # UP, DOWN
        },
        {
            "text": "Aggressive Pong strategy: aim returns at the corners of the opponent "
            "side by hitting the ball with the edge of the paddle. This makes it "
            "harder for the opponent to return.",
            "action_hints": [2, 3],
        },
        {
            "text": "Defensive Pong strategy: keep the paddle centered and make small "
            "adjustments. Avoid large movements that might overshoot the ball.",
            "action_hints": [0],  # NOOP — stay centered
        },
    ],
    "Breakout": [
        {
            "text": "In Breakout, the optimal strategy is to tunnel through one side of the "
            "bricks to reach the top layer. Once the ball is above the bricks, it "
            "bounces repeatedly for high scores.",
            "action_hints": [2, 3],  # LEFT, RIGHT
        },
        {
            "text": "Keep the paddle directly under the ball at all times. Anticipate "
            "the ball angle after bouncing off bricks or walls.",
            "action_hints": [2, 3],
        },
        {
            "text": "At game start, use the FIRE action to launch the ball. Time the "
            "launch to aim toward the corner of the brick wall.",
            "action_hints": [1],  # FIRE
        },
    ],
    "SpaceInvaders": [
        {
            "text": "In Space Invaders, prioritise shooting the bottom row to slow the "
            "descent. Move constantly to avoid enemy fire.",
            "action_hints": [1, 2, 3],  # FIRE, LEFT, RIGHT
        },
        {
            "text": "Use the shields strategically — take cover behind them when "
            "invaders fire, and shoot from behind shield cover.",
            "action_hints": [2, 3],  # positioning actions
        },
        {
            "text": "The mystery ship (bonus) appears periodically at the top. "
            "Shooting it gives bonus points. Time your shots when it appears.",
            "action_hints": [1],  # FIRE
        },
    ],
    "MsPacman": [
        {
            "text": "In Ms. Pac-Man, learn the ghost movement patterns. Each ghost has "
            "a different strategy. Clear nearby dots before moving to new areas.",
            "action_hints": [1, 2, 3, 4],  # all movement directions
        },
        {
            "text": "Power pellets are most valuable when multiple ghosts are nearby. "
            "Save them until ghosts cluster, then eat the power pellet and pursue.",
            "action_hints": [],
        },
    ],
}
