"""Game session management for concurrent web games.

Each game session wraps a GameController + MicroCivEnv + AI agent,
managing the full lifecycle from creation to cleanup.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from alpha_zero.core.civ6_actions import MicroCivActionSpace
from alpha_zero.human_interface.game_controller import GameController
from alpha_zero.web import CHECKPOINT_MODEL_KEY, SESSION_ID_LENGTH
from alpha_zero.web.adapter import BaselineAgentAdapter, MCTSAIAgent, MicroCivGameAdapter
from alpha_zero.web.config import VALID_OPPONENT_TYPES, WebConfig

logger = logging.getLogger(__name__)


@dataclass
class GameSession:
    """A single game session with all associated state.

    Attributes:
        session_id: Unique session identifier.
        controller: GameController managing the game flow.
        adapter: MicroCivGameAdapter wrapping the environment.
        ai_agent: AI agent (MCTS or baseline adapter).
        action_space: Action space for decoding actions.
        created_at: When the session was created.
        last_activity: Last user interaction timestamp.
        config: Web configuration used for this session.
        human_player: Which player is human (0 or 1).
        opponent_type: Type of AI opponent.
    """

    session_id: str
    controller: GameController
    adapter: MicroCivGameAdapter
    ai_agent: MCTSAIAgent | BaselineAgentAdapter | None
    action_space: MicroCivActionSpace
    created_at: datetime
    last_activity: datetime
    config: WebConfig
    human_player: int = 0
    opponent_type: str = "random"
    _websockets: set[Any] = field(default_factory=set, repr=False)

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(UTC)

    @property
    def is_expired(self) -> bool:
        """Check if session has exceeded timeout."""
        elapsed = (datetime.now(UTC) - self.last_activity).total_seconds()
        return elapsed > self.config.session_timeout_seconds

    @property
    def age_seconds(self) -> float:
        """Seconds since session creation."""
        return (datetime.now(UTC) - self.created_at).total_seconds()


class GameSessionManager:
    """Thread-safe manager for concurrent game sessions.

    Handles session creation, lookup, removal, and expiry cleanup.
    """

    def __init__(self, config: WebConfig) -> None:
        """Initialize session manager.

        Args:
            config: Web configuration.
        """
        self._config = config
        self._sessions: dict[str, GameSession] = {}
        self._lock = threading.Lock()
        self._shared_network: Any | None = None
        logger.info(
            "GameSessionManager initialized (max_sessions=%d, timeout=%ds)",
            config.max_concurrent_sessions,
            int(config.session_timeout_seconds),
        )

    @property
    def active_count(self) -> int:
        """Number of active sessions."""
        with self._lock:
            return len(self._sessions)

    def create_session(
        self,
        human_player: int = 0,
        seed: int | None = None,
        opponent_type: str | None = None,
        board_size: int | None = None,
    ) -> GameSession:
        """Create a new game session.

        Args:
            human_player: Which player is human (0 or 1).
            seed: Random seed for reproducibility.
            opponent_type: AI opponent type (random, heuristic, neural).
            board_size: Override board size.

        Returns:
            Created GameSession with game already started.

        Raises:
            RuntimeError: If max concurrent sessions reached.
            ValueError: If opponent_type is invalid.
        """
        opponent_type = opponent_type or self._config.default_opponent

        if opponent_type not in VALID_OPPONENT_TYPES:
            msg = f"Invalid opponent_type: {opponent_type}. Must be one of {VALID_OPPONENT_TYPES}"
            raise ValueError(msg)

        if human_player not in (0, 1):
            msg = f"human_player must be 0 or 1, got {human_player}"
            raise ValueError(msg)

        with self._lock:
            if len(self._sessions) >= self._config.max_concurrent_sessions:
                # Try cleanup first
                self._cleanup_expired_locked()
                if len(self._sessions) >= self._config.max_concurrent_sessions:
                    msg = (
                        f"Maximum concurrent sessions ({self._config.max_concurrent_sessions}) "
                        "reached. Try again later."
                    )
                    raise RuntimeError(msg)

        # Create environment
        effective_board_size = board_size or self._config.board_size
        env = self._create_environment(effective_board_size)
        adapter = MicroCivGameAdapter(env)
        action_space = MicroCivActionSpace(board_size=effective_board_size)

        # Create AI agent
        ai_agent = self._create_ai_agent(opponent_type, env)

        # Create controller
        hi_config = self._config.to_human_interface_config()
        controller = GameController(
            config=hi_config,
            game_interface=adapter,
            ai_agent=ai_agent,
        )

        # Set env reference on agent for MCTS/baseline
        if isinstance(ai_agent, (MCTSAIAgent, BaselineAgentAdapter)):
            ai_agent.set_env(env)

        # Generate session ID
        session_id = uuid.uuid4().hex[:SESSION_ID_LENGTH]
        now = datetime.now(UTC)

        session = GameSession(
            session_id=session_id,
            controller=controller,
            adapter=adapter,
            ai_agent=ai_agent,
            action_space=action_space,
            created_at=now,
            last_activity=now,
            config=self._config,
            human_player=human_player,
            opponent_type=opponent_type,
        )

        # Start the game
        controller.start_game(human_player=human_player, seed=seed)

        with self._lock:
            self._sessions[session_id] = session

        logger.info(
            "Session %s created (opponent=%s, board=%d, human=P%d)",
            session_id,
            opponent_type,
            effective_board_size,
            human_player,
        )
        return session

    def get_session(self, session_id: str) -> GameSession | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            GameSession if found, None otherwise.
        """
        with self._lock:
            session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def get_session_or_error(self, session_id: str) -> GameSession:
        """Get a session by ID, raising if not found.

        Reusable lookup that avoids repeating the get+check pattern
        across all route handlers.

        Args:
            session_id: Session identifier.

        Returns:
            GameSession instance.

        Raises:
            KeyError: If session not found.
        """
        session = self.get_session(session_id)
        if session is None:
            msg = f"Session not found: {session_id}"
            raise KeyError(msg)
        return session

    def remove_session(self, session_id: str) -> bool:
        """Remove a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            True if session was found and removed.
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            logger.info("Session %s removed", session_id)
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove all expired sessions.

        Returns:
            Number of sessions removed.
        """
        with self._lock:
            return self._cleanup_expired_locked()

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all active sessions with summary info.

        Returns:
            List of session summary dicts.
        """
        with self._lock:
            sessions = list(self._sessions.values())

        return [
            {
                "session_id": s.session_id,
                "opponent_type": s.opponent_type,
                "human_player": s.human_player,
                "phase": s.controller.current_state.phase.value,
                "turn_number": s.controller.current_state.turn_number,
                "created_at": s.created_at.isoformat(),
                "age_seconds": round(s.age_seconds, 1),
            }
            for s in sessions
        ]

    def _cleanup_expired_locked(self) -> int:
        """Remove expired sessions (must hold lock)."""
        expired = [sid for sid, session in self._sessions.items() if session.is_expired]
        for sid in expired:
            del self._sessions[sid]
            logger.info("Session %s expired and removed", sid)
        return len(expired)

    def _create_environment(self, board_size: int) -> Any:
        """Create a MicroCiv environment instance.

        Args:
            board_size: Board size for the environment.

        Returns:
            MicroCivEnv instance.
        """
        from alpha_zero.envs.microciv import MicroCivEnv

        logger.debug(
            "Creating MicroCivEnv (board_size=%d, max_turns=%d)", board_size, self._config.max_turns
        )
        return MicroCivEnv(
            board_size=board_size,
            max_turns=self._config.max_turns,
        )

    def _create_ai_agent(
        self,
        opponent_type: str,
        env: Any,
    ) -> MCTSAIAgent | BaselineAgentAdapter:
        """Create an AI agent based on opponent type.

        Args:
            opponent_type: Type of opponent (random, heuristic, neural).
            env: Environment for baseline players.

        Returns:
            AI agent adapter.
        """
        logger.debug("Creating AI agent (type=%s)", opponent_type)
        if opponent_type == "random":
            return self._create_baseline_agent("random", env)

        if opponent_type == "heuristic":
            return self._create_baseline_agent("heuristic", env)

        if opponent_type == "neural":
            return self._create_neural_agent(env)

        msg = f"Unknown opponent type: {opponent_type}"
        raise ValueError(msg)

    def _create_baseline_agent(self, baseline_type: str, env: Any) -> BaselineAgentAdapter:
        """Create a baseline agent adapter.

        Args:
            baseline_type: Type of baseline (random or heuristic).
            env: Environment reference.

        Returns:
            BaselineAgentAdapter wrapping the baseline player.
        """
        if baseline_type == "random":
            from alpha_zero.evaluation.baselines import RandomPlayer

            baseline: Any = RandomPlayer(name="AI (Random)")
        elif baseline_type == "heuristic":
            from alpha_zero.evaluation.baselines import HeuristicPlayer

            baseline = HeuristicPlayer(name="AI (Heuristic)")
        else:
            msg = f"Unknown baseline type: {baseline_type}"
            raise ValueError(msg)

        return BaselineAgentAdapter(baseline=baseline, env=env)

    def _create_neural_agent(self, env: Any) -> MCTSAIAgent:
        """Create a neural network + MCTS agent.

        Uses shared network instance to save memory across sessions.

        Args:
            env: Environment reference for MCTS cloning.

        Returns:
            MCTSAIAgent wrapping the loaded network.

        Raises:
            RuntimeError: If no checkpoint path configured.
        """
        if self._config.checkpoint_path is None:
            msg = "No checkpoint_path configured for neural opponent"
            raise RuntimeError(msg)

        network = self._get_shared_network()

        agent = MCTSAIAgent(
            network=network,
            device=self._config.device,
            use_mcts=self._config.use_mcts,
            mcts_simulations=self._config.mcts_simulations,
            temperature=self._config.ai_temperature,
            board_size=self._config.board_size,
        )
        agent.set_env(env)
        return agent

    def _get_shared_network(self) -> Any:
        """Get or load the shared neural network.

        Loads the network once and shares across all sessions.

        Returns:
            Loaded neural network.
        """
        if self._shared_network is not None:
            logger.debug("Returning cached shared network")
            return self._shared_network

        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.utils.checkpoint import safe_load_checkpoint

        checkpoint_path = self._config.checkpoint_path
        if checkpoint_path is None:
            msg = "No checkpoint path configured"
            raise RuntimeError(msg)

        logger.info("Loading neural network from: %s", checkpoint_path)

        # Load checkpoint
        result = safe_load_checkpoint(checkpoint_path, map_location=self._config.device)
        state_dict = result.state_dict if hasattr(result, "state_dict") else result

        # Create network
        network = MicroCivNetwork(
            board_size=self._config.board_size,
            action_space_size=MicroCivActionSpace(board_size=self._config.board_size).total_size,
        )

        # Load weights
        if isinstance(state_dict, dict) and CHECKPOINT_MODEL_KEY in state_dict:
            network.load_state_dict(state_dict[CHECKPOINT_MODEL_KEY])
        elif isinstance(state_dict, dict):
            network.load_state_dict(state_dict)

        network.eval()
        network.to(self._config.device)

        self._shared_network = network
        logger.info("Neural network loaded successfully")
        return network
