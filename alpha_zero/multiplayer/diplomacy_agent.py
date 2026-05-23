"""LLM-based diplomatic reasoning agent for multiplayer games.

Provides three cooperating components:

* :class:`DiplomacyAgent` -- evaluates and generates diplomatic proposals
  using an LLM backend (OpenAI / Anthropic via :func:`create_llm_client`).
* :class:`CoalitionManager` -- tracks alliances and enmities between players.
* :class:`DiplomacyHistory` -- append-only log of diplomatic events with
  trust-score computation.

All LLM interaction is async-first with a synchronous fallback for testing.
When the LLM client is unavailable the agent degrades gracefully and returns
neutral / empty results.

Usage:
    from alpha_zero.multiplayer.diplomacy_config import DiplomacyConfig
    from alpha_zero.multiplayer.diplomacy_agent import (
        DiplomacyAgent,
        CoalitionManager,
        DiplomacyHistory,
    )

    config = DiplomacyConfig.enabled_default()
    agent = DiplomacyAgent(config)

    evaluation = await agent.evaluate_proposal(game_state, proposal)
    proposal   = await agent.generate_proposal(game_state, player_id=0, target_id=1)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from alpha_zero.multiplayer.diplomacy_config import DiplomacyConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LLM import -- graceful degradation
# ---------------------------------------------------------------------------
try:
    from alpha_zero.utils.llm_client import (
        BaseLLMClient,
        LLMResponse,
        create_llm_client,
    )

    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    BaseLLMClient = None  # type: ignore[misc,assignment]
    LLMResponse = None  # type: ignore[misc,assignment]
    create_llm_client = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DiplomaticProposal:
    """A diplomatic proposal between two players.

    Attributes:
        proposer: Player ID of the proposer.
        target: Player ID of the target.
        action_type: Category of the proposal (e.g. ``"alliance"``).
        terms: Arbitrary key-value terms of the proposal.
        rationale: Human-readable reasoning behind the proposal.
    """

    proposer: int
    target: int
    action_type: str
    terms: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "proposer": self.proposer,
            "target": self.target,
            "action_type": self.action_type,
            "terms": self.terms,
            "rationale": self.rationale,
        }


@dataclass
class ProposalEvaluation:
    """Result of evaluating a diplomatic proposal.

    Attributes:
        score: Acceptance score in ``[0, 1]``.
        reasoning: Explanation for the score.
        counter_terms: Optional modified terms for a counter-proposal.
    """

    score: float
    reasoning: str
    counter_terms: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# CoalitionManager
# ---------------------------------------------------------------------------


class CoalitionManager:
    """Tracks alliances and enmities between players.

    Maintains two symmetric adjacency sets: *allies* and *enemies*.
    Forming an alliance automatically removes any existing enmity (and
    vice-versa).

    Attributes:
        num_players: Total number of players in the game.
    """

    def __init__(self, num_players: int) -> None:
        self.num_players = num_players
        self._allies: dict[int, set[int]] = {p: set() for p in range(num_players)}
        self._enemies: dict[int, set[int]] = {p: set() for p in range(num_players)}

    # -- Alliance operations ------------------------------------------------

    def form_alliance(self, player_a: int, player_b: int) -> None:
        """Form a mutual alliance between two players.

        If the players are currently enemies, the enmity is removed first.
        """
        self._validate_players(player_a, player_b)
        # Remove enmity if present
        self._enemies[player_a].discard(player_b)
        self._enemies[player_b].discard(player_a)
        # Add alliance
        self._allies[player_a].add(player_b)
        self._allies[player_b].add(player_a)

    def break_alliance(self, player_a: int, player_b: int) -> None:
        """Break the alliance between two players (if any)."""
        self._validate_players(player_a, player_b)
        self._allies[player_a].discard(player_b)
        self._allies[player_b].discard(player_a)

    def declare_enemy(self, player_a: int, player_b: int) -> None:
        """Declare mutual enmity between two players.

        If the players are currently allies, the alliance is broken first.
        """
        self._validate_players(player_a, player_b)
        self._allies[player_a].discard(player_b)
        self._allies[player_b].discard(player_a)
        self._enemies[player_a].add(player_b)
        self._enemies[player_b].add(player_a)

    def remove_enemy(self, player_a: int, player_b: int) -> None:
        """Remove enmity between two players (if any)."""
        self._validate_players(player_a, player_b)
        self._enemies[player_a].discard(player_b)
        self._enemies[player_b].discard(player_a)

    # -- Query operations ---------------------------------------------------

    def get_allies(self, player: int) -> set[int]:
        """Return the set of allies for *player*."""
        self._validate_player(player)
        return set(self._allies[player])

    def get_enemies(self, player: int) -> set[int]:
        """Return the set of enemies for *player*."""
        self._validate_player(player)
        return set(self._enemies[player])

    def is_allied(self, player_a: int, player_b: int) -> bool:
        """Check whether two players are allies."""
        self._validate_players(player_a, player_b)
        return player_b in self._allies[player_a]

    def is_enemy(self, player_a: int, player_b: int) -> bool:
        """Check whether two players are enemies."""
        self._validate_players(player_a, player_b)
        return player_b in self._enemies[player_a]

    # -- Validation ---------------------------------------------------------

    def _validate_player(self, player: int) -> None:
        if player < 0 or player >= self.num_players:
            raise ValueError(f"Player {player} out of range [0, {self.num_players})")

    def _validate_players(self, player_a: int, player_b: int) -> None:
        self._validate_player(player_a)
        self._validate_player(player_b)
        if player_a == player_b:
            raise ValueError("Cannot form relationship with self")


# ---------------------------------------------------------------------------
# DiplomacyHistory
# ---------------------------------------------------------------------------


@dataclass
class _HistoryEntry:
    """Internal entry in the diplomacy log."""

    proposal: DiplomaticProposal
    accepted: bool
    turn: int = 0


class DiplomacyHistory:
    """Append-only log of diplomatic events with trust-score computation.

    Trust between two players is computed as:

        trust = (agreements_kept) / max(total_interactions, 1)

    where *agreements_kept* counts accepted proposals that were not subsequently
    broken, and *total_interactions* counts every recorded event between the pair.
    """

    def __init__(self) -> None:
        self._events: list[_HistoryEntry] = []

    def record_event(
        self,
        event: DiplomaticProposal,
        accepted: bool,
        turn: int = 0,
    ) -> None:
        """Record a diplomatic event.

        Args:
            event: The proposal that was made.
            accepted: Whether the target accepted.
            turn: Current game turn (for ordering).
        """
        self._events.append(_HistoryEntry(proposal=event, accepted=accepted, turn=turn))

    def get_history(self, player: int | None = None) -> list[dict[str, Any]]:
        """Return recorded events, optionally filtered to a single player.

        Args:
            player: If given, only events involving this player are returned.

        Returns:
            List of serialized event dicts.
        """
        results: list[dict[str, Any]] = []
        for entry in self._events:
            if player is not None:
                if entry.proposal.proposer != player and entry.proposal.target != player:
                    continue
            results.append(
                {
                    **entry.proposal.to_dict(),
                    "accepted": entry.accepted,
                    "turn": entry.turn,
                }
            )
        return results

    def get_trust_score(self, player_a: int, player_b: int) -> float:
        """Compute a trust score between two players.

        The score is the ratio of accepted proposals to total interactions
        between the pair, in the range ``[0.0, 1.0]``.  When there are no
        interactions the score defaults to ``0.5`` (neutral).
        """
        total = 0
        accepted = 0
        for entry in self._events:
            pair = {entry.proposal.proposer, entry.proposal.target}
            if pair == {player_a, player_b}:
                total += 1
                if entry.accepted:
                    accepted += 1
        if total == 0:
            return 0.5  # neutral default
        return accepted / total

    @property
    def event_count(self) -> int:
        """Total number of recorded events."""
        return len(self._events)


# ---------------------------------------------------------------------------
# DiplomacyAgent
# ---------------------------------------------------------------------------


class DiplomacyAgent:
    """LLM-powered agent for evaluating and generating diplomatic proposals.

    Uses :func:`alpha_zero.utils.llm_client.create_llm_client` under the
    hood.  If the LLM client cannot be initialised (missing API keys,
    missing packages, etc.) the agent falls back to simple heuristics.

    Attributes:
        config: Diplomacy configuration.
        llm_client: Underlying LLM client (may be ``None``).
    """

    def __init__(self, config: DiplomacyConfig | None = None) -> None:
        self.config = config or DiplomacyConfig.default()
        self.llm_client: Any = None

        if self.config.enabled and _LLM_AVAILABLE and create_llm_client is not None:
            try:
                self.llm_client = create_llm_client()
                logger.info("DiplomacyAgent: LLM client initialised")
            except Exception as exc:
                logger.warning("DiplomacyAgent: LLM client unavailable (%s)", exc)

    # -- Public async API ---------------------------------------------------

    async def evaluate_proposal(
        self,
        game_state: dict[str, Any],
        proposal: DiplomaticProposal,
    ) -> ProposalEvaluation:
        """Evaluate a diplomatic proposal using the LLM.

        Falls back to a neutral score if the LLM is unavailable.

        Args:
            game_state: Serialised game state for context.
            proposal: The proposal to evaluate.

        Returns:
            A :class:`ProposalEvaluation`.
        """
        if not self.config.enabled or self.llm_client is None:
            return ProposalEvaluation(
                score=0.5,
                reasoning="Diplomacy agent disabled or LLM unavailable",
            )

        prompt = self.config.evaluation_prompt_template.format(
            game_state=self._format_game_state_prompt(game_state, proposal.target),
            proposal=json.dumps(proposal.to_dict(), indent=2),
        )

        try:
            response = await self.llm_client.generate(
                prompt,
                system_prompt="You are a diplomatic strategy advisor for Civilization 6.",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return self._parse_evaluation_response(response.content)
        except Exception as exc:
            logger.error("DiplomacyAgent.evaluate_proposal failed: %s", exc)
            return ProposalEvaluation(score=0.5, reasoning=f"Evaluation failed: {exc}")

    async def generate_proposal(
        self,
        game_state: dict[str, Any],
        player_id: int,
        target_id: int,
    ) -> DiplomaticProposal | None:
        """Generate a diplomatic proposal using the LLM.

        Returns ``None`` when the LLM is unavailable or when no sensible
        proposal can be formed.

        Args:
            game_state: Serialised game state for context.
            player_id: Proposing player.
            target_id: Target player.

        Returns:
            A :class:`DiplomaticProposal` or ``None``.
        """
        if not self.config.enabled or self.llm_client is None:
            return None

        categories = ", ".join(self.config.proposal_categories)
        prompt = self.config.generation_prompt_template.format(
            game_state=self._format_game_state_prompt(game_state, player_id),
            player_id=player_id,
            target_id=target_id,
            categories=categories,
        )

        try:
            response = await self.llm_client.generate(
                prompt,
                system_prompt="You are a diplomatic strategy advisor for Civilization 6.",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return self._parse_proposal_response(
                response.content,
                proposer=player_id,
                target=target_id,
            )
        except Exception as exc:
            logger.error("DiplomacyAgent.generate_proposal failed: %s", exc)
            return None

    # -- Synchronous wrappers (convenience for tests) -----------------------

    def evaluate_proposal_sync(
        self,
        game_state: dict[str, Any],
        proposal: DiplomaticProposal,
    ) -> ProposalEvaluation:
        """Synchronous wrapper around :meth:`evaluate_proposal`."""
        import asyncio

        return asyncio.run(self.evaluate_proposal(game_state, proposal))

    def generate_proposal_sync(
        self,
        game_state: dict[str, Any],
        player_id: int,
        target_id: int,
    ) -> DiplomaticProposal | None:
        """Synchronous wrapper around :meth:`generate_proposal`."""
        import asyncio

        return asyncio.run(self.generate_proposal(game_state, player_id, target_id))

    # -- Internal helpers ---------------------------------------------------

    def _format_game_state_prompt(
        self,
        game_state: dict[str, Any],
        player_id: int,
    ) -> str:
        """Format a game state dictionary into a prompt-friendly string.

        Args:
            game_state: Raw game state dictionary.
            player_id: The player whose perspective to render.

        Returns:
            Multi-line string summary of the game state.
        """
        parts: list[str] = [f"Player perspective: {player_id}"]

        if "turn" in game_state:
            parts.append(f"Turn: {game_state['turn']}")
        if "num_players" in game_state:
            parts.append(f"Players: {game_state['num_players']}")

        # Player-specific information
        players_info = game_state.get("players", {})
        if isinstance(players_info, dict):
            for pid, info in players_info.items():
                label = "You" if str(pid) == str(player_id) else f"Player {pid}"
                parts.append(f"{label}: {json.dumps(info)}")

        # Relations
        relations = game_state.get("relations", {})
        if relations:
            parts.append(f"Relations: {json.dumps(relations)}")

        return "\n".join(parts)

    def _parse_evaluation_response(self, response: str) -> ProposalEvaluation:
        """Parse an LLM response into a :class:`ProposalEvaluation`.

        Handles malformed JSON gracefully by returning a neutral evaluation.
        """
        try:
            data = json.loads(self._extract_json(response))
            score = float(data.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            reasoning = str(data.get("reasoning", ""))
            counter_terms = data.get("counter_terms")
            if counter_terms is not None and not isinstance(counter_terms, dict):
                counter_terms = None
            return ProposalEvaluation(
                score=score,
                reasoning=reasoning,
                counter_terms=counter_terms,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse evaluation response: %s", exc)
            return ProposalEvaluation(
                score=0.5,
                reasoning=f"Failed to parse LLM response: {response[:200]}",
            )

    def _parse_proposal_response(
        self,
        response: str,
        proposer: int,
        target: int,
    ) -> DiplomaticProposal | None:
        """Parse an LLM response into a :class:`DiplomaticProposal`.

        Returns ``None`` if the response cannot be parsed.
        """
        try:
            data = json.loads(self._extract_json(response))
            action_type = str(data.get("action_type", ""))
            if action_type not in self.config.proposal_categories:
                logger.warning(
                    "LLM proposed unknown action_type '%s'; falling back to first category",
                    action_type,
                )
                action_type = self.config.proposal_categories[0]
            terms = data.get("terms", {})
            if not isinstance(terms, dict):
                terms = {}
            rationale = str(data.get("rationale", ""))
            return DiplomaticProposal(
                proposer=proposer,
                target=target,
                action_type=action_type,
                terms=terms,
                rationale=rationale,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse proposal response: %s", exc)
            return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract the first balanced JSON object from *text*.

        Handles markdown code fences and ignores braces that appear inside
        JSON string literals (including escaped quotes), so payloads like
        ``{"reasoning": "Alliance {clause} is good"}`` are extracted intact
        instead of being truncated at the inner ``}``.
        """
        # Strip markdown code fences
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove opening fence (with optional language tag)
            lines = stripped.split("\n", 1)
            if len(lines) > 1:
                stripped = lines[1]
            if stripped.endswith("```"):
                stripped = stripped[: -len("```")]
            stripped = stripped.strip()

        # Find first { ... } pair using a string-aware scanner
        start = stripped.find("{")
        if start == -1:
            return stripped

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if in_string:
                # Skip the character following a backslash (handles \" \\ etc.)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return stripped[start : i + 1]

        # Fallback: return from first brace onward (caller will fail to parse,
        # which is the correct signal that the LLM emitted truncated JSON).
        return stripped[start:]
