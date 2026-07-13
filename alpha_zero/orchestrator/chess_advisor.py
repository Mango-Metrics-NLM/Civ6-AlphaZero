"""LLM-backed opening prior + commentary for chess training.

This module exposes :class:`ChessAdvisor` — a small layer that calls a
chat-completion LLM (typically Qwen 14B via LM Studio) to produce an
opening-prior probability distribution over legal moves, plus optional
natural-language commentary suitable for demos.

Design decisions:

- **Latency-conscious**: MCTS expansion happens many thousand times per
  game. We therefore (a) cache priors per board FEN with a configurable
  LRU and (b) gate calls behind ``max_ply_for_prior`` so the LLM is only
  consulted in the opening, where its language-model prior actually has
  signal.
- **Graceful degradation**: any error (HTTP, parsing, unknown UCI move)
  is logged once and the advisor returns ``None`` — callers should treat
  ``None`` as "no prior available, use the network policy unchanged."
- **No magic strings**: all prompts are built from configurable templates
  passed into the dataclass.
- **Synchronous façade over async LLM**: the MCTS hot path is synchronous,
  so we use the underlying ``LLMClient.generate_sync`` wrapper.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import chess
import numpy as np

from alpha_zero.envs.chess_actions import ACTION_SPACE_SIZE, move_to_index
from alpha_zero.utils.llm_client import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = (
    "You are a chess expert acting as an opening-book prior for a tree-search agent. "
    "Given a FEN and the list of legal moves in UCI notation, produce a normalised "
    "probability distribution over those moves reflecting their strategic strength."
)


_DEFAULT_PROMPT_TEMPLATE = (
    "Position FEN: {fen}\n"
    "Side to move: {side}\n"
    "Legal moves (UCI): {legal_moves}\n\n"
    "Return ONLY a single JSON object mapping each UCI move to a non-negative "
    'weight (need not sum to 1; we will renormalise). Example: {{"e2e4": 4, '
    '"d2d4": 4, "g1f3": 3, ...}}. Do not include any other text.'
)


@dataclass
class ChessAdvisorConfig:
    """All knobs for :class:`ChessAdvisor`. No magic numbers in code."""

    #: Master enable switch. When False, the advisor returns ``None`` for
    #: every query (i.e., no prior). The orchestrator can therefore wire
    #: the advisor in unconditionally.
    enabled: bool = False
    #: Blend weight in ``[0, 1]`` passed to ``Civ6MCTS(llm_prior_weight=...)``.
    prior_weight: float = 0.3
    #: Ply cap (inclusive). After this many plies, the advisor returns ``None``.
    max_ply_for_prior: int = 24
    #: LRU cache size keyed on board FEN.
    cache_size: int = 4096
    #: Prompt templates.
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    prompt_template: str = _DEFAULT_PROMPT_TEMPLATE
    #: Sampling controls forwarded to the LLM client.
    max_tokens: int = 512
    temperature: float = 0.2
    #: When True, log the LLM's commentary at INFO level.
    log_commentary: bool = False
    #: Extra keyword arguments forwarded to ``LLMClient.generate``.
    llm_kwargs: dict[str, Any] = field(default_factory=dict)


class ChessAdvisor:
    """Chess-specific opening-prior advisor backed by an LLM."""

    def __init__(
        self,
        config: ChessAdvisorConfig,
        llm_client: LLMClient | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        """Initialize.

        Args:
            config: Advisor configuration.
            llm_client: An already-constructed :class:`LLMClient`. Mutually
                exclusive with ``llm_config``.
            llm_config: An :class:`LLMConfig`; the advisor will construct
                its own :class:`LLMClient` from it. Mutually exclusive
                with ``llm_client``.
        """
        if llm_client is not None and llm_config is not None:
            raise ValueError("Provide either llm_client or llm_config, not both")
        if not 0.0 <= config.prior_weight <= 1.0:
            raise ValueError("prior_weight must be in [0, 1]")
        if config.max_ply_for_prior < 0:
            raise ValueError("max_ply_for_prior must be >= 0")
        if config.cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        self.config = config
        self._llm = (
            llm_client
            if llm_client is not None
            else (LLMClient(llm_config) if llm_config is not None else None)
        )
        self._error_logged: bool = False
        # Build the cached prior fetcher with the user-specified cache size.
        self._cached_prior = lru_cache(maxsize=config.cache_size)(self._raw_prior_for_fen)

    # ---------------------------------------------------------------- public API

    def prior_for_board(self, board: chess.Board) -> np.ndarray | None:
        """Return a normalised prior over the 4672 action indices, or ``None``.

        Returns ``None`` if the advisor is disabled, past its ply gate, has
        no LLM client, or encounters an error. The caller treats ``None`` as
        "no prior" and skips blending.
        """
        if not self.config.enabled or self._llm is None:
            return None
        if board.ply() > self.config.max_ply_for_prior:
            return None
        try:
            move_to_weight = self._cached_prior(board.fen())
        except Exception as exc:  # pragma: no cover - defensive
            self._log_once("Cached prior fetch failed", exc)
            return None
        if move_to_weight is None:
            return None
        legal_uci = {m.uci() for m in board.legal_moves}
        if not legal_uci:
            return None
        prior = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for uci, weight in move_to_weight.items():
            if uci not in legal_uci or weight <= 0:
                continue
            try:
                move = chess.Move.from_uci(uci)
                action = move_to_index(move)
            except (ValueError, AssertionError):
                continue
            prior[action] = max(prior[action], float(weight))
        total = prior.sum()
        if total <= 0:
            return None
        return prior / total

    def mcts_callback(self):  # type: ignore[no-untyped-def]  # noqa: ANN201
        """Return a callable suitable for ``Civ6MCTS(llm_prior_fn=...)``.

        The returned function accepts ``(env, action_mask)`` and returns a
        numpy prior array, or zeros when no prior is available (which the
        MCTS blender treats as a no-op).
        """

        def _callback(env: object, action_mask: np.ndarray) -> np.ndarray:
            board = getattr(env, "board", None)
            if board is None or not isinstance(board, chess.Board):
                return np.zeros_like(action_mask, dtype=np.float64)
            prior = self.prior_for_board(board)
            if prior is None:
                return np.zeros_like(action_mask, dtype=np.float64)
            return prior

        return _callback

    def reset(self) -> None:
        """Clear the per-position cache. Useful between training iterations."""
        self._cached_prior.cache_clear()

    # ---------------------------------------------------------------- internals

    def _raw_prior_for_fen(self, fen: str) -> dict[str, float] | None:
        """Call the LLM and parse its JSON response. Cached by FEN."""
        board = chess.Board(fen)
        legal = [m.uci() for m in board.legal_moves]
        if not legal:
            return None
        prompt = self.config.prompt_template.format(
            fen=fen,
            side="white" if board.turn == chess.WHITE else "black",
            legal_moves=", ".join(legal),
        )
        try:
            assert self._llm is not None  # checked in prior_for_board
            response = self._llm.generate_sync(
                prompt,
                system_prompt=self.config.system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **self.config.llm_kwargs,
            )
        except Exception as exc:
            self._log_once("LLM call failed", exc)
            return None
        if self.config.log_commentary:
            logger.info("ChessAdvisor commentary: %s", response.content)
        return self._parse_response(response.content, legal)

    def _parse_response(self, content: str, legal_moves: list[str]) -> dict[str, float] | None:
        """Pull a JSON object out of an LLM response; return ``None`` on failure."""
        match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        payload = match.group(0) if match else content
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            self._log_once("ChessAdvisor failed to parse JSON", exc)
            return None
        if not isinstance(obj, dict):
            self._log_once(
                "ChessAdvisor JSON was not an object",
                TypeError(type(obj).__name__),
            )
            return None
        legal_set = set(legal_moves)
        cleaned: dict[str, float] = {}
        for key, value in obj.items():
            uci = str(key).strip()
            if uci not in legal_set:
                continue
            try:
                weight = float(value)
            except (TypeError, ValueError):
                continue
            if weight < 0:
                continue
            cleaned[uci] = weight
        return cleaned or None

    def _log_once(self, message: str, exc: BaseException) -> None:
        if self._error_logged:
            return
        self._error_logged = True
        logger.warning("%s: %s", message, exc)
