"""Tests for ReplayBuffer and SelfPlayCollector."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.core.config import MCTSConfig
from src.core.training.replay_buffer import ReplayBuffer
from src.core.training.self_play import SelfPlayCollector
from src.core.types import GameOutcome, GameTrajectory, ObservationSpec, TrainingExample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_example(
    spatial_shape: tuple[int, ...] = (1, 2, 2),
    vector_dim: int = 2,
    action_size: int = 4,
    value: float = 0.5,
    *,
    rng: np.random.Generator | None = None,
) -> TrainingExample:
    """Create a minimal TrainingExample for testing."""
    rng = rng or np.random.default_rng(0)
    return TrainingExample(
        spatial=rng.random(spatial_shape, dtype=np.float32),
        vector=rng.random(vector_dim, dtype=np.float32),
        action_mask=np.ones(action_size, dtype=np.bool_),
        policy=np.ones(action_size, dtype=np.float32) / action_size,
        value=value,
    )


def _make_trajectory(
    num_examples: int = 3,
    outcome: GameOutcome = GameOutcome.win,
) -> GameTrajectory:
    """Create a minimal GameTrajectory for testing."""
    rng = np.random.default_rng(42)
    examples = [_make_example(rng=rng) for _ in range(num_examples)]
    return GameTrajectory(examples=examples, outcome=outcome)


# ---------------------------------------------------------------------------
# MockEnv for SelfPlayCollector
# ---------------------------------------------------------------------------


class MockEnv:
    """Minimal environment mock that satisfies the StrategicEnvironment protocol."""

    def __init__(self) -> None:
        self.action_space_size = 4
        self._terminal = False
        self._step_count = 0

    @property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(spatial_shape=(1, 2, 2), vector_dim=2)

    def reset(self, *, seed: int | None = None) -> dict[str, np.ndarray]:
        self._terminal = False
        self._step_count = 0
        return self.get_observation()

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "spatial": np.zeros((1, 2, 2), dtype=np.float32),
            "vector": np.zeros(2, dtype=np.float32),
        }

    def get_valid_actions(self) -> np.ndarray:
        return np.ones(4, dtype=np.bool_)

    def is_terminal(self) -> bool:
        return self._terminal

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        self._step_count += 1
        if self._step_count >= 5:
            self._terminal = True
        reward = 1.0 if self._terminal else 0.0
        return self.get_observation(), reward, self._terminal, False, {}

    def clone(self) -> MockEnv:
        return copy.deepcopy(self)


# ===================================================================
# ReplayBuffer tests
# ===================================================================


class TestReplayBufferConstruction:
    """Tests for ReplayBuffer construction and properties."""

    def test_construction_valid_capacity(self) -> None:
        buf = ReplayBuffer(capacity=100)
        assert buf.capacity == 100

    def test_capacity_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            ReplayBuffer(capacity=0)

    def test_capacity_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            ReplayBuffer(capacity=-5)

    def test_initially_empty(self) -> None:
        buf = ReplayBuffer(capacity=10)
        assert buf.size == 0
        assert not buf.is_full
        assert buf.total_added == 0


class TestReplayBufferAdd:
    """Tests for adding examples and trajectories."""

    def test_add_increases_size(self) -> None:
        buf = ReplayBuffer(capacity=10)
        buf.add(_make_example())
        assert buf.size == 1

    def test_add_trajectory_adds_all_examples(self) -> None:
        buf = ReplayBuffer(capacity=100)
        traj = _make_trajectory(num_examples=5)
        buf.add_trajectory(traj)
        assert buf.size == 5

    def test_capacity_limit_enforced(self) -> None:
        buf = ReplayBuffer(capacity=3)
        for _ in range(5):
            buf.add(_make_example())
        assert buf.size == 3

    def test_is_full_when_at_capacity(self) -> None:
        buf = ReplayBuffer(capacity=3)
        for _ in range(3):
            buf.add(_make_example())
        assert buf.is_full

    def test_total_added_tracks_all(self) -> None:
        buf = ReplayBuffer(capacity=3)
        for _ in range(7):
            buf.add(_make_example())
        assert buf.total_added == 7
        assert buf.size == 3  # only 3 retained


class TestReplayBufferSample:
    """Tests for sampling from the buffer."""

    def test_sample_returns_correct_batch_size(self) -> None:
        buf = ReplayBuffer(capacity=20, seed=0)
        for _ in range(10):
            buf.add(_make_example())
        batch = buf.sample(4)
        assert len(batch) == 4

    def test_sample_raises_when_batch_too_large(self) -> None:
        buf = ReplayBuffer(capacity=10)
        buf.add(_make_example())
        with pytest.raises(ValueError, match="Cannot sample"):
            buf.sample(5)

    def test_sample_batch_keys_and_shapes(self) -> None:
        buf = ReplayBuffer(capacity=50, seed=0)
        spatial_shape = (1, 2, 2)
        vector_dim = 2
        action_size = 4
        for _ in range(10):
            buf.add(
                _make_example(
                    spatial_shape=spatial_shape,
                    vector_dim=vector_dim,
                    action_size=action_size,
                )
            )

        batch = buf.sample_batch(4)
        assert set(batch.keys()) == {"spatial", "vector", "action_mask", "policy", "value"}
        assert batch["spatial"].shape == (4, *spatial_shape)
        assert batch["vector"].shape == (4, vector_dim)
        assert batch["action_mask"].shape == (4, action_size)
        assert batch["policy"].shape == (4, action_size)
        assert batch["value"].shape == (4,)

    def test_sample_with_seed_is_reproducible(self) -> None:
        examples = [_make_example(value=float(i)) for i in range(20)]

        buf1 = ReplayBuffer(capacity=20, seed=123)
        buf2 = ReplayBuffer(capacity=20, seed=123)
        for ex in examples:
            buf1.add(ex)
            buf2.add(ex)

        batch1 = buf1.sample(5)
        batch2 = buf2.sample(5)
        values1 = [e.value for e in batch1]
        values2 = [e.value for e in batch2]
        assert values1 == values2


class TestReplayBufferMisc:
    """Tests for clear, __len__, __repr__."""

    def test_clear_empties_buffer(self) -> None:
        buf = ReplayBuffer(capacity=10)
        for _ in range(5):
            buf.add(_make_example())
        buf.clear()
        assert buf.size == 0
        assert len(buf) == 0

    def test_len_matches_size(self) -> None:
        buf = ReplayBuffer(capacity=10)
        for _ in range(4):
            buf.add(_make_example())
        assert len(buf) == buf.size == 4

    def test_repr_contains_capacity_and_size(self) -> None:
        buf = ReplayBuffer(capacity=50)
        for _ in range(3):
            buf.add(_make_example())
        text = repr(buf)
        assert "50" in text
        assert "3" in text


# ===================================================================
# SelfPlayCollector tests
# ===================================================================


def _dummy_eval_fn(
    spatial: np.ndarray, vector: np.ndarray, action_mask: np.ndarray
) -> tuple[np.ndarray, float]:
    """Deterministic eval function that returns uniform policy and 0.0 value."""
    num_actions = len(action_mask)
    policy = np.ones(num_actions, dtype=np.float32) / num_actions
    return policy, 0.0


class TestSelfPlayCollectorTemperature:
    """Tests for temperature schedule lookup."""

    def test_default_schedule(self) -> None:
        collector = SelfPlayCollector(
            eval_fn=_dummy_eval_fn,
            mcts_config=MCTSConfig(num_simulations=2),
        )
        # Default schedule is {0: 1.0, 30: 0.1}
        assert collector._get_temperature(0) == 1.0
        assert collector._get_temperature(15) == 1.0
        assert collector._get_temperature(30) == pytest.approx(0.1)
        assert collector._get_temperature(100) == pytest.approx(0.1)

    def test_custom_schedule(self) -> None:
        schedule = {0: 2.0, 10: 1.0, 50: 0.5}
        collector = SelfPlayCollector(
            eval_fn=_dummy_eval_fn,
            mcts_config=MCTSConfig(num_simulations=2),
            temperature_schedule=schedule,
        )
        assert collector._get_temperature(0) == pytest.approx(2.0)
        assert collector._get_temperature(5) == pytest.approx(2.0)
        assert collector._get_temperature(10) == pytest.approx(1.0)
        assert collector._get_temperature(25) == pytest.approx(1.0)
        assert collector._get_temperature(50) == pytest.approx(0.5)
        assert collector._get_temperature(999) == pytest.approx(0.5)


class TestSelfPlayCollectorPlayGame:
    """Tests for single game play."""

    def test_play_game_returns_trajectory(self) -> None:
        collector = SelfPlayCollector(
            eval_fn=_dummy_eval_fn,
            mcts_config=MCTSConfig(num_simulations=2),
            max_steps=20,
        )
        env = MockEnv()
        traj = collector.play_game(env, add_noise=False, seed=0)

        assert isinstance(traj, GameTrajectory)
        assert len(traj.examples) > 0
        assert traj.outcome in list(GameOutcome)
        assert "steps" in traj.metadata

    def test_play_game_example_shapes(self) -> None:
        collector = SelfPlayCollector(
            eval_fn=_dummy_eval_fn,
            mcts_config=MCTSConfig(num_simulations=2),
            max_steps=20,
        )
        env = MockEnv()
        traj = collector.play_game(env, add_noise=False, seed=0)

        for ex in traj.examples:
            assert ex.spatial.shape == (1, 2, 2)
            assert ex.vector.shape == (2,)
            assert ex.action_mask.shape == (4,)
            assert ex.policy.shape == (4,)
            assert isinstance(ex.value, float)

    def test_play_game_respects_max_steps(self) -> None:
        # Use a MockEnv that never terminates on its own
        class NeverEndEnv(MockEnv):
            def step(self, action: int) -> tuple:
                self._step_count += 1
                return self.get_observation(), 0.0, False, False, {}

            def is_terminal(self) -> bool:
                return False

        collector = SelfPlayCollector(
            eval_fn=_dummy_eval_fn,
            mcts_config=MCTSConfig(num_simulations=2),
            max_steps=10,
        )
        env = NeverEndEnv()
        traj = collector.play_game(env, add_noise=False, seed=0)

        assert len(traj.examples) <= 10
        assert traj.outcome == GameOutcome.timeout

    def test_play_game_win_outcome(self) -> None:
        """When MockEnv terminates with reward > 0, outcome should be win."""
        collector = SelfPlayCollector(
            eval_fn=_dummy_eval_fn,
            mcts_config=MCTSConfig(num_simulations=2),
            max_steps=50,
        )
        env = MockEnv()  # terminates after 5 steps with reward=1.0
        traj = collector.play_game(env, add_noise=False, seed=0)

        assert traj.outcome == GameOutcome.win

    def test_play_game_loss_outcome(self) -> None:
        """When env terminates with reward < 0, outcome should be loss."""

        class LossEnv(MockEnv):
            """Mock env that terminates with negative reward."""

            def step(
                self, action: int
            ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
                self._step_count += 1
                if self._step_count >= 5:
                    self._terminal = True
                reward = -1.0 if self._terminal else 0.0
                return self.get_observation(), reward, self._terminal, False, {}

        collector = SelfPlayCollector(
            eval_fn=_dummy_eval_fn,
            mcts_config=MCTSConfig(num_simulations=2),
            max_steps=50,
        )
        env = LossEnv()
        traj = collector.play_game(env, add_noise=False, seed=0)

        assert traj.outcome == GameOutcome.loss
        # Value backfill: loss => final_value = -1.0
        # Last example should have value -1.0 (loser's perspective)
        assert traj.examples[-1].value == pytest.approx(-1.0)

    def test_play_game_values_are_filled_in(self) -> None:
        collector = SelfPlayCollector(
            eval_fn=_dummy_eval_fn,
            mcts_config=MCTSConfig(num_simulations=2),
            max_steps=20,
        )
        env = MockEnv()
        traj = collector.play_game(env, add_noise=False, seed=0)

        # Values should be non-zero for a win outcome (alternating +/- pattern)
        if traj.outcome == GameOutcome.win:
            # At least one example should have non-zero value
            has_nonzero = any(abs(ex.value) > 0 for ex in traj.examples)
            assert has_nonzero


class TestSelfPlayCollectorPlayGames:
    """Tests for playing multiple games."""

    def test_play_games_returns_correct_count(self) -> None:
        collector = SelfPlayCollector(
            eval_fn=_dummy_eval_fn,
            mcts_config=MCTSConfig(num_simulations=2),
            max_steps=20,
        )
        trajectories = collector.play_games(
            env_factory=MockEnv,
            num_games=3,
            seed_offset=0,
        )
        assert len(trajectories) == 3
        for traj in trajectories:
            assert isinstance(traj, GameTrajectory)
            assert len(traj.examples) > 0


class TestSelfPlayCollectorUpdateEvalFn:
    """Tests for the update_eval_fn public method."""

    def test_update_eval_fn_replaces_function(self) -> None:
        """update_eval_fn replaces the internal _eval_fn."""
        collector = SelfPlayCollector(
            eval_fn=_dummy_eval_fn,
            mcts_config=MCTSConfig(num_simulations=2),
        )
        assert collector._eval_fn is _dummy_eval_fn

        def new_eval_fn(
            spatial: np.ndarray, vector: np.ndarray, action_mask: np.ndarray
        ) -> tuple[np.ndarray, float]:
            return np.zeros(len(action_mask), dtype=np.float32), 0.0

        collector.update_eval_fn(new_eval_fn)
        assert collector._eval_fn is new_eval_fn

    def test_update_eval_fn_used_in_subsequent_games(self) -> None:
        """After update_eval_fn, subsequent games use the new function."""
        call_tracker: list[bool] = []

        def tracking_eval_fn(
            spatial: np.ndarray, vector: np.ndarray, action_mask: np.ndarray
        ) -> tuple[np.ndarray, float]:
            call_tracker.append(True)
            num_actions = len(action_mask)
            return np.ones(num_actions, dtype=np.float32) / num_actions, 0.0

        collector = SelfPlayCollector(
            eval_fn=_dummy_eval_fn,
            mcts_config=MCTSConfig(num_simulations=2),
            max_steps=10,
        )
        collector.update_eval_fn(tracking_eval_fn)
        collector.play_game(MockEnv(), add_noise=False, seed=0)
        assert len(call_tracker) > 0
