"""Integration tests for N-player training pipeline value support.

Tests the full flow of N-player self-play:
- TrainingExample with scalar vs array values
- GameTrajectory with num_players metadata
- Serialization/deserialization for both value types
- Network forward pass with NPlayerValueHead
- End-to-end: create N-player env, run steps, create training examples

Uses TestDefaults from tests/test_constants.py for configuration.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alpha_zero.distributed.shared_storage import (
    GameTrajectory,
    TrainingExample,
)
from tests.test_constants import TestDefaults, microciv_channels

# ---------------------------------------------------------------------------
# TrainingExample backward-compat tests
# ---------------------------------------------------------------------------


class TestTrainingExampleScalarValue:
    """TrainingExample with scalar float value (2-player backward compat)."""

    def _make_example(self, value: float = 1.0) -> TrainingExample:
        board = TestDefaults.SMALL_BOARD_SIZE
        channels = TestDefaults.MICROCIV_CHANNELS
        return TrainingExample(
            spatial=np.zeros((channels, board, board), dtype=np.float32),
            vector=np.zeros(TestDefaults.VECTOR_DIM, dtype=np.float32),
            action_mask=np.ones(TestDefaults.ACTION_SPACE_SIZE, dtype=np.float32),
            policy=np.ones(TestDefaults.ACTION_SPACE_SIZE, dtype=np.float32)
            / TestDefaults.ACTION_SPACE_SIZE,
            value=value,
        )

    def test_scalar_value_type(self) -> None:
        ex = self._make_example(1.0)
        assert isinstance(ex.value, float)

    def test_scalar_roundtrip_uncompressed(self) -> None:
        ex = self._make_example(-0.5)
        data = ex.to_bytes(compress=False)
        restored = TrainingExample.from_bytes(data, compressed=False)
        assert isinstance(restored.value, float)
        assert restored.value == pytest.approx(-0.5)
        np.testing.assert_array_equal(restored.spatial, ex.spatial)

    def test_scalar_roundtrip_compressed(self) -> None:
        ex = self._make_example(0.0)
        data = ex.to_bytes(compress=True)
        restored = TrainingExample.from_bytes(data, compressed=True)
        assert isinstance(restored.value, float)
        assert restored.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TrainingExample N-player tests
# ---------------------------------------------------------------------------


class TestTrainingExampleArrayValue:
    """TrainingExample with numpy array value (N-player)."""

    def _make_example(self, num_players: int = 4) -> TrainingExample:
        board = TestDefaults.SMALL_BOARD_SIZE
        channels = microciv_channels(num_players)
        values = np.array(
            [1.0 if i == 0 else -1.0 / (num_players - 1) for i in range(num_players)],
            dtype=np.float32,
        )
        return TrainingExample(
            spatial=np.zeros((channels, board, board), dtype=np.float32),
            vector=np.zeros(TestDefaults.VECTOR_DIM, dtype=np.float32),
            action_mask=np.ones(TestDefaults.ACTION_SPACE_SIZE, dtype=np.float32),
            policy=np.ones(TestDefaults.ACTION_SPACE_SIZE, dtype=np.float32)
            / TestDefaults.ACTION_SPACE_SIZE,
            value=values,
        )

    def test_array_value_type(self) -> None:
        ex = self._make_example(4)
        assert isinstance(ex.value, np.ndarray)
        assert ex.value.shape == (4,)

    def test_array_roundtrip_uncompressed(self) -> None:
        ex = self._make_example(4)
        data = ex.to_bytes(compress=False)
        restored = TrainingExample.from_bytes(data, compressed=False)
        assert isinstance(restored.value, np.ndarray)
        np.testing.assert_array_almost_equal(restored.value, ex.value)

    def test_array_roundtrip_compressed(self) -> None:
        ex = self._make_example(6)
        data = ex.to_bytes(compress=True)
        restored = TrainingExample.from_bytes(data, compressed=True)
        assert isinstance(restored.value, np.ndarray)
        assert restored.value.shape == (6,)
        np.testing.assert_array_almost_equal(restored.value, ex.value)


# ---------------------------------------------------------------------------
# GameTrajectory tests
# ---------------------------------------------------------------------------


class TestGameTrajectoryNPlayer:
    """GameTrajectory with N-player metadata."""

    def _make_trajectory(
        self,
        num_players: int = 4,
        length: int = 3,
    ) -> GameTrajectory:
        board = TestDefaults.SMALL_BOARD_SIZE
        channels = microciv_channels(num_players)
        values = np.array(
            [1.0 if i == 0 else -0.33 for i in range(num_players)],
            dtype=np.float32,
        )
        examples = []
        for _ in range(length):
            examples.append(
                TrainingExample(
                    spatial=np.random.randn(channels, board, board).astype(np.float32),
                    vector=np.random.randn(TestDefaults.VECTOR_DIM).astype(np.float32),
                    action_mask=np.ones(TestDefaults.ACTION_SPACE_SIZE, dtype=np.float32),
                    policy=np.ones(TestDefaults.ACTION_SPACE_SIZE, dtype=np.float32)
                    / TestDefaults.ACTION_SPACE_SIZE,
                    value=values,
                )
            )
        return GameTrajectory(
            examples=examples,
            game_id=1,
            actor_id=0,
            game_length=length,
            winner=0,
            num_players=num_players,
            rankings={0: 0, 1: 1, 2: 1, 3: 1},
            player_scores={0: 100.0, 1: 50.0, 2: 30.0, 3: 20.0},
        )

    def test_num_players_field(self) -> None:
        traj = self._make_trajectory(num_players=4)
        assert traj.num_players == 4

    def test_default_num_players(self) -> None:
        traj = GameTrajectory()
        assert traj.num_players == 2
        assert traj.rankings is None
        assert traj.player_scores is None

    def test_rankings_and_scores(self) -> None:
        traj = self._make_trajectory(4)
        assert traj.rankings is not None
        assert traj.rankings[0] == 0  # Winner
        assert traj.player_scores is not None
        assert traj.player_scores[0] == 100.0

    def test_roundtrip_uncompressed(self) -> None:
        traj = self._make_trajectory(4, length=2)
        data = traj.to_bytes(compress=False)
        restored = GameTrajectory.from_bytes(data, compressed=False)
        assert restored.num_players == 4
        assert restored.rankings == {0: 0, 1: 1, 2: 1, 3: 1}
        assert restored.player_scores == {0: 100.0, 1: 50.0, 2: 30.0, 3: 20.0}
        assert len(restored.examples) == 2
        assert isinstance(restored.examples[0].value, np.ndarray)

    def test_roundtrip_compressed(self) -> None:
        traj = self._make_trajectory(4, length=2)
        data = traj.to_bytes(compress=True)
        restored = GameTrajectory.from_bytes(data, compressed=True)
        assert restored.num_players == 4
        assert restored.winner == 0

    def test_legacy_deserialisation(self) -> None:
        """Deserializing a trajectory without N-player fields defaults safely."""
        import pickle

        # Simulate a legacy payload without num_players/rankings/player_scores
        legacy_data = pickle.dumps(
            {
                "examples": [],
                "game_id": 42,
                "actor_id": 0,
                "game_length": 10,
                "winner": 1,
            }
        )
        traj = GameTrajectory.from_bytes(legacy_data, compressed=False)
        assert traj.num_players == 2
        assert traj.rankings is None
        assert traj.player_scores is None
        assert traj.winner == 1


# ---------------------------------------------------------------------------
# Network N-player tests
# ---------------------------------------------------------------------------


class TestMicroCivNetworkNPlayer:
    """MicroCivNetwork with NPlayerValueHead."""

    def test_2player_scalar_output(self) -> None:
        """Default 2-player network produces scalar value (B,)."""
        from alpha_zero.core.civ6_network import MicroCivNetwork

        board = TestDefaults.SMALL_BOARD_SIZE
        net = MicroCivNetwork(
            board_size=board,
            action_space_size=TestDefaults.ACTION_SPACE_SIZE,
            num_res_blocks=TestDefaults.NUM_RES_BLOCKS,
            num_filters=TestDefaults.NUM_FILTERS,
        )
        spatial = torch.randn(2, 13, board, board)
        vector = torch.randn(2, TestDefaults.VECTOR_DIM)
        _policy, value = net(spatial, vector)
        assert value.shape == (2,)  # Scalar per sample

    def test_4player_vector_output(self) -> None:
        """4-player network produces per-player values (B, 4)."""
        from alpha_zero.core.civ6_network import MicroCivNetwork

        num_players = 4
        board = TestDefaults.SMALL_BOARD_SIZE
        channels = microciv_channels(num_players)

        net = MicroCivNetwork(
            board_size=board,
            action_space_size=TestDefaults.ACTION_SPACE_SIZE,
            num_res_blocks=TestDefaults.NUM_RES_BLOCKS,
            num_filters=TestDefaults.NUM_FILTERS,
            spatial_channels=channels,
            num_players=num_players,
        )
        assert net.num_players == 4

        spatial = torch.randn(2, channels, board, board)
        vector = torch.randn(2, TestDefaults.VECTOR_DIM)
        policy, value = net(spatial, vector)

        assert policy.shape == (2, TestDefaults.ACTION_SPACE_SIZE)
        assert value.shape == (2, num_players)
        # Values should be in [-1, 1] (tanh output)
        assert value.min().item() >= -1.0
        assert value.max().item() <= 1.0

    def test_8player_vector_output(self) -> None:
        """8-player network works with max player count."""
        from alpha_zero.core.civ6_network import MicroCivNetwork

        num_players = 8
        board = TestDefaults.SMALL_BOARD_SIZE
        channels = microciv_channels(num_players)

        net = MicroCivNetwork(
            board_size=board,
            action_space_size=TestDefaults.ACTION_SPACE_SIZE,
            num_res_blocks=TestDefaults.NUM_RES_BLOCKS,
            num_filters=TestDefaults.NUM_FILTERS,
            spatial_channels=channels,
            num_players=num_players,
        )
        spatial = torch.randn(1, channels, board, board)
        vector = torch.randn(1, TestDefaults.VECTOR_DIM)
        _policy, value = net(spatial, vector)
        assert value.shape == (1, 8)


# ---------------------------------------------------------------------------
# sample_batch N-player tests
# ---------------------------------------------------------------------------


class TestSampleBatchNPlayer:
    """DistributedReplayBuffer.sample_batch with N-player values."""

    def test_sample_batch_scalar_values(self) -> None:
        """sample_batch produces 1-D value tensor for scalar examples."""
        from alpha_zero.distributed.shared_storage import (
            DistributedReplayBuffer,
            ReplayBufferConfig,
        )

        config = ReplayBufferConfig(max_size=100, min_size=1)
        buf = DistributedReplayBuffer(config=config)

        # Add scalar-valued examples directly to buffer
        board = TestDefaults.SMALL_BOARD_SIZE
        for _ in range(5):
            ex = TrainingExample(
                spatial=np.zeros((13, board, board), dtype=np.float32),
                vector=np.zeros(TestDefaults.VECTOR_DIM, dtype=np.float32),
                action_mask=np.ones(TestDefaults.ACTION_SPACE_SIZE, dtype=np.float32),
                policy=np.ones(TestDefaults.ACTION_SPACE_SIZE, dtype=np.float32)
                / TestDefaults.ACTION_SPACE_SIZE,
                value=1.0,
            )
            buf._buffer.append(ex)

        batch = buf.sample_batch(batch_size=3)
        assert batch["value"].dim() == 1  # (B,)
        assert batch["value"].shape[0] == 3
        buf.shutdown()

    def test_sample_batch_array_values(self) -> None:
        """sample_batch produces 2-D value tensor for array examples."""
        from alpha_zero.distributed.shared_storage import (
            DistributedReplayBuffer,
            ReplayBufferConfig,
        )

        config = ReplayBufferConfig(max_size=100, min_size=1)
        buf = DistributedReplayBuffer(config=config)

        board = TestDefaults.SMALL_BOARD_SIZE
        num_players = 4
        channels = microciv_channels(num_players)
        values = np.array([1.0, -0.33, -0.33, -0.33], dtype=np.float32)

        for _ in range(5):
            ex = TrainingExample(
                spatial=np.zeros((channels, board, board), dtype=np.float32),
                vector=np.zeros(TestDefaults.VECTOR_DIM, dtype=np.float32),
                action_mask=np.ones(TestDefaults.ACTION_SPACE_SIZE, dtype=np.float32),
                policy=np.ones(TestDefaults.ACTION_SPACE_SIZE, dtype=np.float32)
                / TestDefaults.ACTION_SPACE_SIZE,
                value=values,
            )
            buf._buffer.append(ex)

        batch = buf.sample_batch(batch_size=3)
        assert batch["value"].dim() == 2  # (B, num_players)
        assert batch["value"].shape == (3, num_players)
        buf.shutdown()


# ---------------------------------------------------------------------------
# Value loss tests
# ---------------------------------------------------------------------------


class TestNPlayerValueLoss:
    """compute_nplayer_value_loss integrates correctly."""

    def test_nplayer_loss_basic(self) -> None:
        from alpha_zero.multiplayer.network_heads import compute_nplayer_value_loss

        predicted = torch.tensor([[0.5, -0.5, 0.0, 0.2]])
        target = torch.tensor([[1.0, -1.0, 0.0, 0.0]])
        loss = compute_nplayer_value_loss(predicted, target, num_players=4)
        assert loss.item() > 0
        assert loss.dim() == 0  # Scalar

    def test_scalar_fallback(self) -> None:
        """Legacy scalar inputs still work."""
        from alpha_zero.multiplayer.network_heads import compute_nplayer_value_loss

        predicted = torch.tensor([0.5, -0.3])
        target = torch.tensor([1.0, -1.0])
        loss = compute_nplayer_value_loss(predicted, target, num_players=2)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Scoring integration tests
# ---------------------------------------------------------------------------


class TestScoringIntegration:
    """Verify scoring module produces correct value arrays."""

    def test_compute_player_values_winner(self) -> None:
        from alpha_zero.multiplayer.scoring import (
            compute_player_values,
            compute_rankings,
        )

        rankings = compute_rankings(winner=0, num_players=4)
        values = compute_player_values(rankings, num_players=4)
        assert values[0] == pytest.approx(1.0)
        # All losers share rank 1, so value = 1 - 2*(1/3) = 0.333...
        for p in [1, 2, 3]:
            assert values[p] < values[0]

    def test_extract_game_result(self) -> None:
        from alpha_zero.multiplayer.scoring import extract_game_result

        info = {"winner": 2}
        values = extract_game_result(info, num_players=4)
        assert values[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNPlayerEndToEnd:
    """End-to-end: create 4-player env, run a few steps, create examples."""

    def test_4player_env_to_training_examples(self) -> None:
        """Run a 4-player MicroCiv game and generate training examples."""
        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.multiplayer.scoring import (
            compute_player_values,
            compute_rankings,
        )

        num_players = 4
        board_size = TestDefaults.SMALL_BOARD_SIZE
        channels = microciv_channels(num_players)

        # Create environment
        env = MicroCivEnv(
            board_size=board_size,
            max_turns=TestDefaults.MAX_TURNS,
            num_players=num_players,
        )

        # Create network
        net = MicroCivNetwork(
            board_size=board_size,
            action_space_size=env.action_space.n,
            num_res_blocks=TestDefaults.NUM_RES_BLOCKS,
            num_filters=TestDefaults.NUM_FILTERS,
            spatial_channels=channels,
            num_players=num_players,
        )
        net.eval()

        # Run a few steps
        obs, info = env.reset()
        history = []
        for _step in range(min(10, TestDefaults.MAX_TURNS)):
            mask = env.get_valid_actions()
            valid_actions = np.where(mask > 0)[0]
            if len(valid_actions) == 0:
                break

            # Create uniform policy over valid actions
            policy = np.zeros(env.action_space.n, dtype=np.float32)
            policy[valid_actions] = 1.0 / len(valid_actions)

            history.append(
                {
                    "spatial": obs["spatial"].copy(),
                    "vector": obs["vector"].copy(),
                    "action_mask": mask.copy(),
                    "policy": policy,
                    "player": env.current_player,
                }
            )

            action = int(np.random.choice(valid_actions))
            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        # Compute per-player values
        winner = info.get("winner")
        rankings = compute_rankings(winner=winner, num_players=num_players)
        player_values = compute_player_values(rankings, num_players=num_players)
        value_array = np.array(
            [player_values.get(p, 0.0) for p in range(num_players)],
            dtype=np.float32,
        )

        # Create training examples
        examples = []
        for h in history:
            examples.append(
                TrainingExample(
                    spatial=h["spatial"],
                    vector=h["vector"],
                    action_mask=h["action_mask"],
                    policy=h["policy"],
                    value=value_array,
                )
            )

        assert len(examples) > 0

        # Verify network can process these examples
        spatial_batch = torch.stack([torch.from_numpy(ex.spatial).float() for ex in examples])
        vector_batch = torch.stack([torch.from_numpy(ex.vector).float() for ex in examples])

        with torch.no_grad():
            policy_logits, values = net(spatial_batch, vector_batch)

        assert policy_logits.shape == (len(examples), env.action_space.n)
        assert values.shape == (len(examples), num_players)

        # Verify serialization
        traj = GameTrajectory(
            examples=examples,
            game_id=1,
            actor_id=0,
            game_length=len(history),
            winner=winner,
            num_players=num_players,
            rankings=dict(rankings),
        )
        data = traj.to_bytes(compress=True)
        restored = GameTrajectory.from_bytes(data, compressed=True)
        assert restored.num_players == num_players
        assert len(restored.examples) == len(examples)
        assert isinstance(restored.examples[0].value, np.ndarray)
        assert restored.examples[0].value.shape == (num_players,)

    def test_network_loss_with_nplayer_batch(self) -> None:
        """Verify training loss computation with N-player value targets."""
        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.multiplayer.network_heads import compute_nplayer_value_loss

        num_players = 4
        board = TestDefaults.SMALL_BOARD_SIZE
        channels = microciv_channels(num_players)
        batch_size = TestDefaults.BATCH_SIZE

        net = MicroCivNetwork(
            board_size=board,
            action_space_size=TestDefaults.ACTION_SPACE_SIZE,
            num_res_blocks=TestDefaults.NUM_RES_BLOCKS,
            num_filters=TestDefaults.NUM_FILTERS,
            spatial_channels=channels,
            num_players=num_players,
        )

        spatial = torch.randn(batch_size, channels, board, board)
        vector = torch.randn(batch_size, TestDefaults.VECTOR_DIM)
        target_values = torch.randn(batch_size, num_players).clamp(-1, 1)

        _policy_logits, pred_values = net(spatial, vector)
        assert pred_values.shape == (batch_size, num_players)

        loss = compute_nplayer_value_loss(pred_values, target_values, num_players=num_players)
        assert loss.item() > 0
        assert loss.requires_grad

        # Verify backward pass works
        loss.backward()
        grad_count = sum(
            1 for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0
