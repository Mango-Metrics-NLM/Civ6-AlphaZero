"""End-to-end integration tests for N-player training pipeline.

Tests the full flow:
- N-player self-play with MCTS generates correct training examples
- Training examples feed into network training with loss decrease
- Full iteration: self-play -> generate examples -> train network
- MCTS search correctness across different player counts
- Feature combinations (fog of war, scoring, opponent model config)
- Channel consistency between encoder and network
- Checkpoint roundtrip for N-player models
- GameTrajectory serialization roundtrip
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.distributed.shared_storage import GameTrajectory, TrainingExample
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.multiplayer.config import (
    MultiplayerConfig,
    NPlayerMCTSConfig,
    ScoringConfig,
)
from alpha_zero.multiplayer.mcts import NPlayerMCTS
from alpha_zero.multiplayer.scoring import (
    ScoringMethod,
    compute_player_values,
    compute_rankings,
)
from tests.test_constants import TestDefaults, microciv_channels

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOARD_SIZE = TestDefaults.SMALL_BOARD_SIZE
MAX_TURNS = TestDefaults.MAX_TURNS
NUM_SIMS = 10
NUM_RES = TestDefaults.NUM_RES_BLOCKS
NUM_FILTERS = TestDefaults.NUM_FILTERS


def _make_env(num_players: int = 4, **kwargs) -> MicroCivEnv:
    """Create a small N-player environment."""
    return MicroCivEnv(
        board_size=BOARD_SIZE,
        max_turns=kwargs.pop("max_turns", MAX_TURNS),
        num_players=num_players,
        seed=kwargs.pop("seed", 42),
        **kwargs,
    )


def _make_network(num_players: int = 4) -> MicroCivNetwork:
    """Create a small N-player network."""
    channels = microciv_channels(num_players)
    env = _make_env(num_players=num_players)
    return MicroCivNetwork(
        board_size=BOARD_SIZE,
        action_space_size=int(env.action_space.n),
        num_res_blocks=NUM_RES,
        num_filters=NUM_FILTERS,
        spatial_channels=channels,
        num_players=num_players,
    )


def _run_random_game(
    env: MicroCivEnv,
    max_steps: int = 30,
) -> list[dict]:
    """Play random actions and collect (obs, mask, policy, player) history."""
    obs, _info = env.reset()
    history: list[dict] = []
    for _ in range(max_steps):
        mask = env.get_valid_actions()
        valid = np.where(mask > 0)[0]
        if len(valid) == 0:
            break
        policy = np.zeros(int(env.action_space.n), dtype=np.float32)
        policy[valid] = 1.0 / len(valid)
        history.append(
            {
                "spatial": obs["spatial"].copy(),
                "vector": obs["vector"].copy(),
                "action_mask": mask.copy(),
                "policy": policy,
                "player": env.current_player,
            }
        )
        action = int(np.random.choice(valid))
        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    return history


# ===========================================================================
# TestNPlayerTrainingLoop
# ===========================================================================


@pytest.mark.integration
class TestNPlayerTrainingLoop:
    """Full N-player training loop end-to-end tests."""

    def test_4player_selfplay_generates_examples(self) -> None:
        """4-player MCTS self-play generates TrainingExamples with correct shapes."""
        num_players = 4
        channels = microciv_channels(num_players)
        env = _make_env(num_players=num_players)
        net = _make_network(num_players=num_players)
        net.eval()

        mcts = NPlayerMCTS(
            network=net,
            config=NPlayerMCTSConfig(num_simulations=NUM_SIMS),
            num_players=num_players,
            device="cpu",
        )

        _obs, _info = env.reset()
        examples: list[TrainingExample] = []

        for _ in range(5):
            if env.done:
                break
            action_probs, _values = mcts.search(env, add_noise=False)
            # Create training example
            obs_now = env.get_observation()
            value_array = np.zeros(num_players, dtype=np.float32)
            examples.append(
                TrainingExample(
                    spatial=obs_now["spatial"],
                    vector=obs_now["vector"],
                    action_mask=obs_now["action_mask"],
                    policy=action_probs,
                    value=value_array,
                )
            )
            # Pick action
            valid = np.where(env.get_valid_actions() > 0)[0]
            if len(valid) == 0:
                break
            action = NPlayerMCTS.select_action(action_probs, temperature=1.0)
            env.step(action)

        assert len(examples) > 0
        for ex in examples:
            assert ex.spatial.shape == (channels, BOARD_SIZE, BOARD_SIZE)
            assert ex.vector.shape == (TestDefaults.VECTOR_DIM,)
            assert ex.policy.shape == (env.action_space.n,)
            assert isinstance(ex.value, np.ndarray)
            assert ex.value.shape == (num_players,)

    def test_training_examples_to_network_training(self) -> None:
        """Training examples train the network; loss should decrease over 5 epochs."""
        num_players = 4
        channels = microciv_channels(num_players)
        net = _make_network(num_players=num_players)
        env = _make_env(num_players=num_players)

        # Create fixed training batch
        batch_size = 8
        spatial = torch.randn(batch_size, channels, BOARD_SIZE, BOARD_SIZE)
        vector = torch.randn(batch_size, TestDefaults.VECTOR_DIM)
        action_mask = torch.ones(batch_size, int(env.action_space.n))
        target_policy = torch.softmax(torch.randn(batch_size, int(env.action_space.n)), dim=-1)
        target_values = torch.zeros(batch_size, num_players)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        initial_loss = None
        final_loss = None

        for epoch in range(20):
            net.train()
            optimizer.zero_grad()
            policy_logits, pred_values = net(spatial, vector, action_mask)

            # Policy loss (cross entropy)
            log_probs = torch.log_softmax(policy_logits, dim=-1)
            policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()
            # Value loss (MSE)
            value_loss = torch.nn.functional.mse_loss(pred_values, target_values)
            loss = policy_loss + value_loss

            if epoch == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        assert initial_loss is not None
        assert final_loss is not None
        assert final_loss < initial_loss, (
            f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    @pytest.mark.slow
    def test_full_iteration_selfplay_train(self) -> None:
        """Full iteration: 4-player self-play -> examples -> train -> loss improves."""
        num_players = 4
        net = _make_network(num_players=num_players)

        mcts = NPlayerMCTS(
            network=net,
            config=NPlayerMCTSConfig(num_simulations=NUM_SIMS),
            num_players=num_players,
            device="cpu",
        )

        # --- Self-play: 2 games ---
        all_examples: list[TrainingExample] = []
        for game_seed in range(2):
            game_env = _make_env(num_players=num_players, seed=game_seed)
            game_env.reset()
            history: list[dict] = []

            for _ in range(15):
                if game_env.done:
                    break
                action_probs, _vals = mcts.search(game_env, add_noise=True)
                obs_now = game_env.get_observation()
                history.append(
                    {
                        "spatial": obs_now["spatial"],
                        "vector": obs_now["vector"],
                        "action_mask": obs_now["action_mask"],
                        "policy": action_probs,
                    }
                )
                action = NPlayerMCTS.select_action(action_probs, temperature=1.0)
                _obs, _r, term, trunc, info = game_env.step(action)
                if term or trunc:
                    break

            # Compute final values
            winner = info.get("winner")
            rankings = compute_rankings(winner=winner, num_players=num_players)
            pvalues = compute_player_values(rankings, num_players=num_players)
            value_array = np.array(
                [pvalues.get(p, 0.0) for p in range(num_players)],
                dtype=np.float32,
            )

            for h in history:
                all_examples.append(
                    TrainingExample(
                        spatial=h["spatial"],
                        vector=h["vector"],
                        action_mask=h["action_mask"],
                        policy=h["policy"],
                        value=value_array,
                    )
                )

        assert len(all_examples) > 0

        # --- Train ---
        optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
        spatial_batch = torch.stack([torch.from_numpy(e.spatial).float() for e in all_examples])
        vector_batch = torch.stack([torch.from_numpy(e.vector).float() for e in all_examples])
        mask_batch = torch.stack([torch.from_numpy(e.action_mask).float() for e in all_examples])
        policy_batch = torch.stack([torch.from_numpy(e.policy).float() for e in all_examples])
        value_batch = torch.stack([torch.from_numpy(e.value).float() for e in all_examples])

        initial_loss = None
        final_loss = None
        for epoch in range(10):
            net.train()
            optimizer.zero_grad()
            pred_logits, pred_values = net(spatial_batch, vector_batch, mask_batch)
            log_probs = torch.log_softmax(pred_logits, dim=-1)
            p_loss = -(policy_batch * log_probs).sum(dim=-1).mean()
            v_loss = torch.nn.functional.mse_loss(pred_values, value_batch)
            loss = p_loss + v_loss
            if epoch == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        assert initial_loss is not None and final_loss is not None
        assert final_loss < initial_loss


# ===========================================================================
# TestNPlayerMCTSSearch
# ===========================================================================


@pytest.mark.integration
class TestNPlayerMCTSSearch:
    """MCTS search correctness across player counts."""

    def _run_search(self, num_players: int) -> tuple[np.ndarray, dict[int, float]]:
        """Run MCTS search for a given player count."""
        env = _make_env(num_players=num_players)
        net = _make_network(num_players=num_players)
        net.eval()
        mcts = NPlayerMCTS(
            network=net,
            config=NPlayerMCTSConfig(num_simulations=NUM_SIMS),
            num_players=num_players,
            device="cpu",
        )
        env.reset()
        return mcts.search(env, add_noise=False)

    def test_mcts_search_2player(self) -> None:
        """NPlayerMCTS.search with 2 players returns valid action probs."""
        probs, values = self._run_search(2)
        assert np.all(probs >= 0)
        assert abs(probs.sum() - 1.0) < 1e-4
        assert len(values) == 2

    def test_mcts_search_3player(self) -> None:
        """NPlayerMCTS.search with 3 players returns valid action probs."""
        probs, values = self._run_search(3)
        assert np.all(probs >= 0)
        assert abs(probs.sum() - 1.0) < 1e-4
        assert len(values) == 3

    def test_mcts_search_4player(self) -> None:
        """NPlayerMCTS.search with 4 players returns valid action probs."""
        probs, values = self._run_search(4)
        assert np.all(probs >= 0)
        assert abs(probs.sum() - 1.0) < 1e-4
        assert len(values) == 4

    def test_mcts_action_probs_valid(self) -> None:
        """Probs sum to ~1, no negatives, only valid actions have nonzero prob."""
        num_players = 4
        env = _make_env(num_players=num_players)
        net = _make_network(num_players=num_players)
        net.eval()
        mcts = NPlayerMCTS(
            network=net,
            config=NPlayerMCTSConfig(num_simulations=NUM_SIMS),
            num_players=num_players,
            device="cpu",
        )
        env.reset()
        action_mask = env.get_valid_actions()
        probs, _values = mcts.search(env, add_noise=False)

        # No negative probabilities
        assert np.all(probs >= 0), "Found negative action probabilities"
        # Sums to approximately 1
        assert abs(probs.sum() - 1.0) < 1e-4, f"Probs sum to {probs.sum()}, not 1.0"
        # Only valid actions should have nonzero probability
        invalid_actions = np.where(action_mask == 0)[0]
        for idx in invalid_actions:
            assert probs[idx] == pytest.approx(0.0, abs=1e-6), (
                f"Invalid action {idx} has prob {probs[idx]}"
            )


# ===========================================================================
# TestFeatureCombination
# ===========================================================================


@pytest.mark.integration
class TestFeatureCombination:
    """Test combinations of fog of war, scoring, and opponent model config."""

    def test_fog_of_war_with_nplayer(self) -> None:
        """Enable fog_of_war with N-player game, verify masked observations."""
        num_players = 4
        env = _make_env(num_players=num_players, fog_of_war=True)
        obs, _info = env.reset()

        # Play a few steps
        for _ in range(10):
            if env.done:
                break
            mask = env.get_valid_actions()
            valid = np.where(mask > 0)[0]
            if len(valid) == 0:
                break
            action = int(np.random.choice(valid))
            obs, _r, term, trunc, _i = env.step(action)
            if term or trunc:
                break

        # Fog of war should produce valid observations
        assert obs["spatial"].shape[0] == microciv_channels(num_players)
        assert np.all(np.isfinite(obs["spatial"]))
        assert np.all(np.isfinite(obs["vector"]))

    def test_scoring_with_nplayer(self) -> None:
        """Different scoring methods with N-player game produce valid values."""
        num_players = 4
        rankings = compute_rankings(winner=2, num_players=num_players)

        for method in ScoringMethod:
            scores = (
                {p: float((p + 1) * 10) for p in range(num_players)}
                if method == ScoringMethod.PROPORTIONAL
                else None
            )
            values = compute_player_values(
                rankings,
                num_players=num_players,
                method=method,
                player_scores=scores,
            )
            assert len(values) == num_players
            # For non-proportional methods, winner should have the highest value.
            # For proportional, the scores determine the ranking (not the winner).
            if method != ScoringMethod.PROPORTIONAL:
                assert values[2] == max(values.values()), (
                    f"Winner should have max value for {method.value}"
                )
            # All values in [-1, 1]
            for v in values.values():
                assert -1.0 <= v <= 1.0, f"Value {v} out of range for {method}"

    def test_all_features_enabled(self) -> None:
        """Enable fog, scoring config, opponent model config -- no crashes."""
        num_players = 4
        mp_config = MultiplayerConfig(
            num_players=num_players,
            mcts=NPlayerMCTSConfig(num_simulations=NUM_SIMS),
            scoring=ScoringConfig(method="winner_take_all"),
            fog_of_war=True,
        )

        env = _make_env(
            num_players=num_players,
            fog_of_war=mp_config.fog_of_war,
        )
        net = _make_network(num_players=num_players)
        net.eval()

        mcts = NPlayerMCTS(
            network=net,
            config=mp_config.mcts,
            num_players=num_players,
            device="cpu",
        )

        env.reset()
        # Run a few MCTS searches without crashing
        for _ in range(3):
            if env.done:
                break
            probs, _values = mcts.search(env, add_noise=False)
            assert np.all(probs >= 0)
            action = NPlayerMCTS.select_action(probs, temperature=1.0)
            env.step(action)


# ===========================================================================
# TestChannelConsistency
# ===========================================================================


@pytest.mark.integration
@pytest.mark.parametrize("num_players", [2, 3, 4, 6])
class TestChannelConsistency:
    """Verify encoder output channels match network expected spatial_channels."""

    def test_encoder_matches_network_channels(self, num_players: int) -> None:
        """Encoder output channels == network expected spatial_channels."""
        from alpha_zero.envs.civ6_encoder import MicroCivEncoder

        encoder = MicroCivEncoder(board_size=BOARD_SIZE, num_players=num_players)
        expected_channels = microciv_channels(num_players)

        # Encoder total_channels must match formula
        assert encoder.total_channels == expected_channels, (
            f"Encoder has {encoder.total_channels} channels but expected "
            f"{expected_channels} for {num_players} players"
        )

        # Network must accept exactly this many channels
        net = MicroCivNetwork(
            board_size=BOARD_SIZE,
            action_space_size=TestDefaults.ACTION_SPACE_SIZE,
            num_res_blocks=NUM_RES,
            num_filters=NUM_FILTERS,
            spatial_channels=expected_channels,
            num_players=num_players,
        )

        spatial = torch.randn(1, expected_channels, BOARD_SIZE, BOARD_SIZE)
        vector = torch.randn(1, TestDefaults.VECTOR_DIM)
        with torch.no_grad():
            policy, value = net(spatial, vector)
        assert policy.shape == (1, TestDefaults.ACTION_SPACE_SIZE)
        if num_players > 2:
            assert value.shape == (1, num_players)
        else:
            assert value.shape == (1,)


# ===========================================================================
# TestCheckpointRoundtrip
# ===========================================================================


@pytest.mark.integration
class TestCheckpointRoundtrip:
    """Save/load N-player model and verify predictions match."""

    def test_save_load_nplayer_model(self) -> None:
        """Save N-player model state_dict, load into matching model, verify predictions."""
        num_players = 4
        channels = microciv_channels(num_players)
        net = _make_network(num_players=num_players)
        net.eval()

        spatial = torch.randn(2, channels, BOARD_SIZE, BOARD_SIZE)
        vector = torch.randn(2, TestDefaults.VECTOR_DIM)

        with torch.no_grad():
            orig_policy, orig_value = net(spatial, vector)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            # Save state dict directly (MicroCivNetwork.load doesn't preserve
            # num_players in the config, so we recreate manually)
            torch.save(net.state_dict(), path)
            loaded = _make_network(num_players=num_players)
            loaded.load_state_dict(torch.load(path, weights_only=True))
            loaded.eval()

            with torch.no_grad():
                loaded_policy, loaded_value = loaded(spatial, vector)

            assert torch.allclose(orig_policy, loaded_policy, atol=1e-5)
            assert torch.allclose(orig_value, loaded_value, atol=1e-5)
        finally:
            import os

            os.unlink(path)

    def test_game_trajectory_serialize_deserialize(self) -> None:
        """GameTrajectory roundtrip: serialize -> deserialize -> data matches."""
        num_players = 4
        channels = microciv_channels(num_players)
        board = BOARD_SIZE

        values = np.array([1.0, -0.33, -0.33, -0.33], dtype=np.float32)
        examples = []
        for _ in range(3):
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

        traj = GameTrajectory(
            examples=examples,
            game_id=42,
            actor_id=1,
            game_length=3,
            winner=0,
            num_players=num_players,
            rankings={0: 0, 1: 1, 2: 1, 3: 1},
            player_scores={0: 100.0, 1: 50.0, 2: 30.0, 3: 20.0},
        )

        data = traj.to_bytes(compress=True)
        restored = GameTrajectory.from_bytes(data, compressed=True)

        assert restored.num_players == num_players
        assert restored.game_id == 42
        assert restored.winner == 0
        assert restored.rankings == {0: 0, 1: 1, 2: 1, 3: 1}
        assert restored.player_scores == {0: 100.0, 1: 50.0, 2: 30.0, 3: 20.0}
        assert len(restored.examples) == 3

        for orig, rest in zip(examples, restored.examples, strict=False):
            np.testing.assert_array_almost_equal(rest.spatial, orig.spatial)
            np.testing.assert_array_almost_equal(rest.vector, orig.vector)
            np.testing.assert_array_almost_equal(rest.policy, orig.policy)
            assert isinstance(rest.value, np.ndarray)
            np.testing.assert_array_almost_equal(rest.value, orig.value)
