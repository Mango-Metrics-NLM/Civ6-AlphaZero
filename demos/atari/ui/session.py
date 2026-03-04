"""Game session backend wrapping AtariEnv + AtariNetwork + MCTS."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from demos.atari.atari_env import AtariEnv
from demos.atari.atari_network import AtariNetwork
from demos.atari.config import AtariEnvConfig, AtariMCTSConfig, AtariNetworkConfig
from demos.atari.ui.config import AtariUIConfig
from demos.atari.ui.serializers import frame_to_base64

logger = logging.getLogger(__name__)


class AtariGameSession:
    """Manages a single Atari game session with AI support.

    Wraps environment, neural network, and MCTS into a coherent session
    for the web UI. Supports human play, AI autopilot, and step-by-step
    AI moves with tree inspection.
    """

    def __init__(self, ui_config: AtariUIConfig) -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.ui_config = ui_config
        self._created_at = time.time()
        self._last_touch = time.time()

        # Build environment with rgb_array rendering for UI
        env_config = AtariEnvConfig(
            game_name=ui_config.game,
            render_mode="rgb_array",
        )
        self.env = AtariEnv(env_config)

        # Build network
        net_config = AtariNetworkConfig()
        self.network = AtariNetwork(
            config=net_config,
            num_actions=self.env.action_space.n,
            spatial_size=env_config.spatial_size,
            spatial_channels=env_config.spatial_channels,
            vector_dim=env_config.vector_dim,
        )
        self.network.to(ui_config.device)
        self.network.eval()

        # Load checkpoint if provided
        if ui_config.model_path and Path(ui_config.model_path).exists():
            self.load_model(ui_config.model_path)

        # MCTS config
        self.mcts_config = AtariMCTSConfig(num_simulations=ui_config.mcts_simulations)

        # Episode recording
        self._episode_frames: list[np.ndarray] = []
        self._episode_actions: list[int] = []
        self._episode_rewards: list[float] = []
        self._episode_policies: list[np.ndarray] = []
        self._episode_values: list[float] = []

        # Last MCTS tree root (for visualization)
        self._last_mcts_root: Any = None

        # State
        self._total_reward: float = 0.0
        self._step_count: int = 0
        self._done: bool = False

        # Reset to start
        self.reset()
        logger.info(
            "Session %s created: game=%s, device=%s",
            self.session_id,
            ui_config.game,
            ui_config.device,
        )

    def touch(self) -> None:
        """Update last-activity timestamp."""
        self._last_touch = time.time()

    @property
    def idle_seconds(self) -> float:
        """Seconds since last activity."""
        return time.time() - self._last_touch

    def reset(self) -> dict[str, Any]:
        """Reset the environment and clear episode recording.

        Returns:
            Initial game state dict.
        """
        self.env.reset()
        self._total_reward = 0.0
        self._step_count = 0
        self._done = False
        self._episode_frames.clear()
        self._episode_actions.clear()
        self._episode_rewards.clear()
        self._episode_policies.clear()
        self._episode_values.clear()
        self._last_mcts_root = None

        # In Pong (and many Atari games), the ball doesn't appear until
        # FIRE (action 1) is pressed and a few frames elapse.  Send FIRE
        # followed by NOOPs so the initial frame shows ball + paddles.
        _FIRE = 1
        _NOOP = 0
        for action in [_FIRE, _NOOP, _NOOP, _NOOP, _NOOP, _NOOP,
                        _FIRE, _NOOP, _NOOP, _NOOP, _NOOP, _NOOP,
                        _NOOP, _NOOP, _NOOP, _NOOP, _NOOP, _NOOP,
                        _NOOP, _NOOP]:
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                self.env.reset()
                break

        # Capture initial frame
        raw_frame = self._get_raw_frame()
        if raw_frame is not None:
            self._episode_frames.append(raw_frame)

        return self.get_state()

    def human_step(self, action: int) -> dict[str, Any]:
        """Execute a human action.

        Args:
            action: Action index.

        Returns:
            Step result dict.
        """
        self.touch()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_reward += reward
        self._step_count += 1
        self._done = terminated or truncated

        # Record
        raw_frame = self._get_raw_frame()
        if raw_frame is not None:
            self._episode_frames.append(raw_frame)
        self._episode_actions.append(action)
        self._episode_rewards.append(reward)

        return {
            "action": action,
            "reward": float(reward),
            "done": self._done,
            "total_reward": float(self._total_reward),
            "step": int(self._step_count),
            "frame": frame_to_base64(raw_frame) if raw_frame is not None else "",
        }

    def ai_step(self) -> dict[str, Any]:
        """Run MCTS search and execute the best action.

        Returns:
            AI step result dict with action, probabilities, value, and frame.
        """
        self.touch()
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig

        mcts = Civ6MCTS(
            network=self.network,
            config=MCTSConfig(
                num_simulations=self.mcts_config.num_simulations,
                c_puct=self.mcts_config.c_puct,
                dirichlet_alpha=self.mcts_config.dirichlet_alpha,
                dirichlet_epsilon=self.mcts_config.dirichlet_epsilon,
            ),
            device=self.ui_config.device,
        )

        action_probs, root_value = mcts.search(self.env, add_noise=False)
        # MCTS tree root is a local in search(); not exposed via attribute
        self._last_mcts_root = getattr(mcts, "root", None)

        # Select action (greedy)
        action = int(np.argmax(action_probs))

        # Record policy + value
        self._episode_policies.append(action_probs)
        self._episode_values.append(float(root_value))

        # Execute
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_reward += reward
        self._step_count += 1
        self._done = terminated or truncated

        raw_frame = self._get_raw_frame()
        if raw_frame is not None:
            self._episode_frames.append(raw_frame)
        self._episode_actions.append(action)
        self._episode_rewards.append(reward)

        return {
            "action": action,
            "probs": action_probs.tolist(),
            "value": float(root_value),
            "reward": float(reward),
            "done": self._done,
            "total_reward": float(self._total_reward),
            "step": int(self._step_count),
            "frame": frame_to_base64(raw_frame) if raw_frame is not None else "",
        }

    def get_state(self) -> dict[str, Any]:
        """Get current game state for the UI.

        Returns:
            State dict with frame, score, step count, etc.
        """
        raw_frame = self._get_raw_frame()
        return {
            "session_id": self.session_id,
            "game": self.ui_config.game,
            "frame": frame_to_base64(raw_frame) if raw_frame is not None else "",
            "total_reward": float(self._total_reward),
            "step": int(self._step_count),
            "done": self._done,
            "num_actions": int(self.env.action_space.n),
        }

    def get_frame_base64(self) -> str:
        """Get current frame as base64 PNG."""
        raw_frame = self._get_raw_frame()
        if raw_frame is None:
            return ""
        return frame_to_base64(raw_frame)

    def get_mcts_root(self) -> Any:
        """Return the last MCTS root node (for tree visualization)."""
        return self._last_mcts_root

    def load_model(self, path: str) -> None:
        """Load a model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        try:
            loaded = AtariNetwork.load(
                path,
                spatial_channels=self.env.config.spatial_channels,
                vector_dim=self.env.config.vector_dim,
                device=self.ui_config.device,
            )
            self.network = loaded
            self.network.eval()
            logger.info("Model loaded from %s", path)
        except Exception:
            logger.exception("Failed to load model from %s", path)

    def save_episode(self, path: str | None = None) -> str:
        """Save the current episode recording as .npz.

        Args:
            path: Output file path. Auto-generated if None.

        Returns:
            Path to saved file.
        """
        if path is None:
            replay_dir = Path(self.ui_config.replay_dir)
            replay_dir.mkdir(parents=True, exist_ok=True)
            path = str(
                replay_dir / f"{self.ui_config.game}_{self.session_id}_{int(time.time())}.npz"
            )

        data: dict[str, Any] = {
            "game": self.ui_config.game,
            "actions": np.array(self._episode_actions),
            "rewards": np.array(self._episode_rewards),
            "total_reward": self._total_reward,
        }
        if self._episode_frames:
            data["frames"] = np.stack(self._episode_frames)
        if self._episode_policies:
            data["policies"] = np.stack(self._episode_policies)
        if self._episode_values:
            data["values"] = np.array(self._episode_values)

        np.savez_compressed(path, **data)
        logger.info("Episode saved to %s", path)
        return path

    def _get_raw_frame(self) -> np.ndarray | None:
        """Get the raw RGB frame from the environment."""
        try:
            frame = self.env._env.render()
            if frame is not None:
                return np.asarray(frame)
        except Exception:
            logger.debug("render() failed, falling back to last raw frame")
        # Fallback: use last raw frame if available
        if self.env._last_raw_frame is not None:
            return self.env._last_raw_frame
        return None

    def close(self) -> None:
        """Release resources."""
        self.env.close()


class SessionManager:
    """Manages multiple concurrent game sessions."""

    def __init__(self, max_sessions: int = 10) -> None:
        self._sessions: dict[str, AtariGameSession] = {}
        self.max_sessions = max_sessions

    def create_session(self, ui_config: AtariUIConfig) -> AtariGameSession:
        """Create a new game session.

        Args:
            ui_config: UI configuration.

        Returns:
            New session instance.
        """
        if len(self._sessions) >= self.max_sessions:
            # Evict oldest idle session
            oldest = min(self._sessions.values(), key=lambda s: s._last_touch)
            self.remove_session(oldest.session_id)

        session = AtariGameSession(ui_config)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> AtariGameSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        """Remove and close a session."""
        session = self._sessions.pop(session_id, None)
        if session:
            session.close()

    @property
    def active_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    def cleanup_expired(self, max_idle_seconds: float = 3600) -> int:
        """Remove sessions idle for too long.

        Args:
            max_idle_seconds: Maximum idle time before expiry.

        Returns:
            Number of sessions removed.
        """
        expired = [
            sid
            for sid, session in self._sessions.items()
            if session.idle_seconds > max_idle_seconds
        ]
        for sid in expired:
            self.remove_session(sid)
        return len(expired)
