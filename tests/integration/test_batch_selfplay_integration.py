"""Integration tests for batch self-play pipeline.

Validates the `run_batch_selfplay()` function from `selfplay_worker.py`,
which uses `BatchGumbelMCTS` + `VectorizedMicroCivEnv` for parallel
game generation.

Tests cover:
  - Trajectory key/shape validation
  - Batch vs sequential schema consistency
  - Edge cases (single game, zero games)
  - Vectorized env lifecycle

Run with:
    pytest tests/integration/test_batch_selfplay_integration.py -v --timeout=120
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from alpha_zero.core.batch_mcts import BatchMCTSConfig
from alpha_zero.envs.vectorized_env import VectorizedEnvConfig, VectorizedMicroCivEnv

# Small config for fast tests
_BOARD_SIZE = 8
_MCTS_SIMS = 4  # Very few sims for speed
_GUMBEL_ACTIONS = 4
_MAX_BATCH_SIZE = 8
_MAX_TURNS = 15


# ============================================================================
# Fixtures
# ============================================================================


# Weight seed for `small_model`. The tests below seed `run_batch_selfplay`
# because they assert on game *outcomes* -- but the network driving the search
# was left unseeded, so the policy differed on every run and the outcomes with
# it. `test_value_targets_are_not_uniformly_degenerate` failed in CI on exactly
# that: it asserts at least one game reaches a decisive result, and with a
# documented ~73.5% draw rate an unlucky random policy can draw all six games
# across all three env seeds. Seeding the environment but not the policy makes a
# test only half-deterministic, which reads as a flake.
_MODEL_SEED = 1337


@contextlib.contextmanager
def _seed_global_rngs(seed: int):
    """Seed the RNGs `run_batch_selfplay(seed=...)` does not, then restore them.

    That parameter seeds the *environments* only. `selfplay_worker` says so at
    the config construction: "the environments are only half the randomness --
    gumbel_sample_actions draws from the *global* numpy RNG, so a caller wanting
    full reproducibility must seed that too."

    These tests assert on game outcomes, so half-seeding left them depending on
    how the dice fell -- which is what made
    `test_value_targets_are_not_uniformly_degenerate` fail intermittently in CI
    while passing locally.

    Restoring matters as much as seeding. The first version of this helper
    seeded the process-global RNGs and left them there, so every test collected
    afterwards in the same worker inherited state from this file -- exactly the
    hidden coupling the `small_model` fixture uses `fork_rng` to avoid,
    reintroduced two functions below it. Seed inside the block, unchanged
    outside it.
    """
    import numpy as np
    import torch

    np_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            np.random.set_state(np_state)


@pytest.fixture
def small_model():
    """A small transformer with deterministic weights.

    Weights come from a forked RNG so the seed cannot leak into any other test's
    randomness -- the fixture is deterministic without making the rest of the
    suite implicitly depend on this seed.
    """
    import torch

    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    config = TransformerConfig(
        spatial_channels=13,
        spatial_size=_BOARD_SIZE,
        vector_dim=64,
        action_space_size=376,
        num_layers=1,
        embed_dim=32,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_MODEL_SEED)
        model = Civ6TransformerNetwork(config=config)
    model.eval()
    return model


@pytest.fixture
def vec_env():
    """Create vectorized environment with 4 parallel envs."""
    config = VectorizedEnvConfig(
        num_envs=4,
        board_size=_BOARD_SIZE,
        max_turns=_MAX_TURNS,
    )
    return VectorizedMicroCivEnv(config=config)


# ============================================================================
# Test: Vectorized Environment
# ============================================================================


class TestVectorizedEnvLifecycle:
    """Tests for VectorizedMicroCivEnv basic lifecycle."""

    def test_env_creation(self, vec_env):
        """Vectorized env initializes with correct count."""
        assert vec_env.num_envs == 4
        assert vec_env.board_size == _BOARD_SIZE

    def test_reset_all(self, vec_env):
        """Reset all returns valid observations."""
        vec_env.reset_all()
        spatial, vector, action_mask = vec_env.get_observations()
        assert spatial.shape[0] == 4  # num_envs
        assert vector.shape[0] == 4
        assert action_mask.shape[0] == 4

    def test_step_all(self, vec_env):
        """Step all with valid actions."""
        vec_env.reset_all()
        _, _, action_mask = vec_env.get_observations()

        # Pick first valid action for each env
        actions = np.zeros(4, dtype=np.int32)
        for i in range(4):
            valid = np.where(action_mask[i] > 0)[0]
            actions[i] = valid[0] if len(valid) > 0 else 0

        rewards, terminated, truncated, _infos = vec_env.step_all(actions)
        assert rewards.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)

    def test_clone_env(self, vec_env):
        """Clone creates independent copy."""
        vec_env.reset_all()
        original = vec_env.get_env(0)
        clone = vec_env.clone_env(0)

        # Should have same state
        orig_obs = original.get_observation()
        clone_obs = clone.get_observation()
        np.testing.assert_array_equal(orig_obs["spatial"], clone_obs["spatial"])

    def test_active_count(self, vec_env):
        """Active count starts at num_envs."""
        vec_env.reset_all()
        assert vec_env.active_count == 4


# ============================================================================
# Test: Batch MCTS Configuration
# ============================================================================


class TestBatchMCTSConfig:
    """Tests for BatchMCTSConfig validation."""

    def test_default_config(self):
        """Default config is valid."""
        config = BatchMCTSConfig()
        assert config.num_simulations > 0
        assert config.num_sampled_actions > 0

    def test_custom_config(self):
        """Custom config values are stored."""
        config = BatchMCTSConfig(
            num_simulations=_MCTS_SIMS,
            num_sampled_actions=_GUMBEL_ACTIONS,
            max_batch_size=_MAX_BATCH_SIZE,
        )
        assert config.num_simulations == _MCTS_SIMS
        assert config.num_sampled_actions == _GUMBEL_ACTIONS
        assert config.max_batch_size == _MAX_BATCH_SIZE

    def test_invalid_simulations_raises(self):
        """Zero simulations should raise ValueError."""
        with pytest.raises(ValueError):
            BatchMCTSConfig(num_simulations=0)


# ============================================================================
# Test: Batch Self-Play Trajectories
# ============================================================================


class TestBatchSelfplayTrajectories:
    """Tests for run_batch_selfplay trajectory output."""

    @pytest.mark.timeout(120)
    def test_batch_selfplay_produces_trajectories(self, small_model):
        """run_batch_selfplay returns non-empty trajectory list."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay

        trajectories = run_batch_selfplay(
            model=small_model,
            num_games=2,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            num_parallel_games=2,
            max_batch_size=_MAX_BATCH_SIZE,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        assert len(trajectories) > 0, "Should produce at least 1 trajectory"

    @pytest.mark.timeout(120)
    def test_trajectory_keys(self, small_model):
        """Trajectories contain required keys."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay

        trajectories = run_batch_selfplay(
            model=small_model,
            num_games=1,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            num_parallel_games=1,
            max_batch_size=_MAX_BATCH_SIZE,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        required_keys = {
            "spatial",
            "vector",
            "action_masks",
            "policies",
            "actions",
            "rewards",
        }
        for traj in trajectories:
            assert required_keys.issubset(traj.keys()), (
                f"Missing keys: {required_keys - set(traj.keys())}"
            )

    @pytest.mark.timeout(120)
    def test_trajectory_lengths_consistent(self, small_model):
        """All trajectory arrays have same length within each trajectory."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay

        trajectories = run_batch_selfplay(
            model=small_model,
            num_games=1,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            num_parallel_games=1,
            max_batch_size=_MAX_BATCH_SIZE,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        for traj in trajectories:
            n_steps = len(traj["spatial"])
            assert n_steps > 0, "Trajectory should have at least 1 step"
            assert len(traj["vector"]) == n_steps
            assert len(traj["action_masks"]) == n_steps
            assert len(traj["policies"]) == n_steps
            assert len(traj["actions"]) == n_steps
            assert len(traj["rewards"]) == n_steps
            assert len(traj["players"]) == n_steps


# ============================================================================
# Test: Schema Consistency
# ============================================================================


class TestBatchVsSequentialConsistency:
    """Tests that batch and sequential self-play produce compatible schemas."""

    @pytest.mark.timeout(180)
    def test_same_trajectory_keys(self, small_model):
        """Batch and sequential produce trajectories with identical key sets."""
        from alpha_zero.cloud.gcp.selfplay_worker import (
            run_batch_selfplay,
            run_selfplay,
        )

        batch_trajs = run_batch_selfplay(
            model=small_model,
            num_games=1,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            num_parallel_games=1,
            max_batch_size=_MAX_BATCH_SIZE,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        seq_trajs = run_selfplay(
            model=small_model,
            num_games=1,
            mcts_sims=_MCTS_SIMS,
            mcts_type="gumbel",
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        # Assert non-emptiness BEFORE comparing. These assertions previously sat
        # inside `if batch_trajs and seq_trajs:`, so the test passed silently
        # whenever either path returned nothing -- which is exactly what happens
        # when self-play breaks.
        assert batch_trajs, "batch self-play produced no trajectories"
        assert seq_trajs, "sequential self-play produced no trajectories"

        # Full key-set equality, not issubset: a subset check cannot detect a
        # path that omits a field (this is how the missing action_masks in the
        # batch path went unnoticed).
        assert set(batch_trajs[0]) == set(seq_trajs[0]), (
            f"key sets diverge: batch-only={set(batch_trajs[0]) - set(seq_trajs[0])}, "
            f"sequential-only={set(seq_trajs[0]) - set(batch_trajs[0])}"
        )


# ============================================================================
# Test: Action-mask integrity (regression guard for the all-ones fallback)
# ============================================================================


class TestBatchActionMasks:
    """The batch path must persist real action masks, not nothing.

    When ``action_masks`` is absent the trainer substitutes an all-ones mask
    (``training_worker.prepare_dataset``), which silently disables masked
    policy loss: the network is trained to put probability mass on illegal
    actions. Shape and length assertions alone cannot catch that, because an
    all-ones mask satisfies every one of them -- hence the density assertion
    below.
    """

    @pytest.mark.timeout(120)
    def test_masks_are_present_and_informative(self, small_model):
        """Stored masks must be binary, correctly sized, and actually masking."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay

        trajectories = run_batch_selfplay(
            model=small_model,
            num_games=1,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            num_parallel_games=1,
            max_batch_size=_MAX_BATCH_SIZE,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        assert trajectories, "batch self-play produced no trajectories"

        all_masks = []
        for traj in trajectories:
            assert traj["action_masks"], "trajectory carries no action masks"
            for step, mask in enumerate(traj["action_masks"]):
                # Width must match the policy the mask is meant to constrain.
                assert len(mask) == len(traj["policies"][step]), (
                    f"mask width {len(mask)} != policy width "
                    f"{len(traj['policies'][step])} at step {step}"
                )
                assert set(mask) <= {0.0, 1.0}, "masks must be binary"
                assert sum(mask) >= 1.0, "every position needs >=1 legal action"
            all_masks.extend(traj["action_masks"])

        # The assertion that distinguishes "fixed" from "still broken": an
        # all-ones mask would pass every check above.
        density = float(np.mean([np.mean(m) for m in all_masks]))
        assert 0.0 < density < 1.0, (
            f"mask density {density} implies masks are degenerate "
            "(1.0 means masking is effectively disabled)"
        )

    @pytest.mark.timeout(60)
    def test_terminates_when_a_batch_makes_no_progress(self, small_model, monkeypatch):
        """A batch that produces nothing must stop the run, not repeat forever.

        The outer loop's only advance is ``games_completed += batch_games``. A
        deterministic failure on a batch's *first* search leaves ``batch_games``
        at zero, so the identical batch would be retried indefinitely -- on a
        cloud worker, a job burning GPU hours in silence.

        The sibling alignment test below deliberately fails on the *second*
        call, because it needs a committed step to go ragged from. That is the
        right shape for what it asserts, and it is also why it cannot see this:
        its first batch always succeeds and the loop always advances.
        """
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay
        from alpha_zero.core.batch_mcts import BatchGumbelMCTS

        def always_fails(self, envs, add_noise=False):
            raise RuntimeError("induced deterministic search failure")

        monkeypatch.setattr(BatchGumbelMCTS, "batch_search", always_fails)

        # The timeout mark is the real assertion: before the no-progress guard
        # this call never returned.
        trajectories = run_batch_selfplay(
            model=small_model,
            num_games=4,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            num_parallel_games=2,
            max_batch_size=_MAX_BATCH_SIZE,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        assert trajectories == [], (
            f"a run whose every search failed should yield no trajectories, got {len(trajectories)}"
        )

    @pytest.mark.timeout(120)
    def test_trajectories_stay_aligned_when_search_fails(self, small_model, monkeypatch):
        """A mid-game search failure must not leave per-step fields ragged.

        The batch loop reads observations before searching. If those were
        appended eagerly, a failed search would leave ``spatial`` longer than
        ``policies``/``players``, and the trainer -- which iterates ``spatial``
        and indexes the others -- would raise IndexError on real data.
        """
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay
        from alpha_zero.core.batch_mcts import BatchGumbelMCTS

        real_search = BatchGumbelMCTS.batch_search
        calls = {"n": 0}

        def flaky_search(self, envs, add_noise=False):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise RuntimeError("induced batch search failure")
            return real_search(self, envs, add_noise=add_noise)

        monkeypatch.setattr(BatchGumbelMCTS, "batch_search", flaky_search)

        trajectories = run_batch_selfplay(
            model=small_model,
            num_games=1,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            num_parallel_games=1,
            max_batch_size=_MAX_BATCH_SIZE,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        assert calls["n"] >= 2, "the induced failure never triggered"

        per_step_fields = (
            "spatial",
            "vector",
            "action_masks",
            "policies",
            "actions",
            "players",
            "rewards",
        )
        for traj in trajectories:
            lengths = {field: len(traj[field]) for field in per_step_fields}
            assert len(set(lengths.values())) == 1, f"ragged trajectory: {lengths}"


class TestBatchGamesReachTheirOwnEnd:
    """Batch games must be bounded by the environment, not by an action counter.

    The loop's step counter counts batched *actions* while ``max_turns`` is the
    environment's *game-turn* limit, and MicroCiv spans many actions per turn.
    Comparing the two directly stopped every game at roughly a sixth of its
    length, before any environment could truncate itself -- so ``winner`` was
    always ``None`` and every position was labelled 0.0. Shape and length
    assertions all passed while the value targets were uniformly degenerate.
    """

    @pytest.mark.timeout(300)
    def test_games_are_not_cut_off_at_max_turns_actions(self, small_model):
        """Trajectories must exceed max_turns steps, since a turn spans actions."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay

        max_turns = 12
        with _seed_global_rngs(17):
            trajectories = run_batch_selfplay(
                model=small_model,
                num_games=2,
                mcts_sims=2,
                board_size=_BOARD_SIZE,
                max_turns=max_turns,
                num_parallel_games=2,
                max_batch_size=_MAX_BATCH_SIZE,
                gumbel_num_actions=2,
                device="cpu",
                seed=17,
            )

        assert trajectories, "batch self-play produced no trajectories"
        lengths = [len(t["spatial"]) for t in trajectories]
        # Deliberately only the inequality: a game can legitimately end at
        # exactly max_turns steps, so asserting no length equals it would fail
        # on a valid outcome. Exceeding the turn limit in *steps* is the real
        # signal, since a game turn spans several actions.
        assert max(lengths) > max_turns, (
            f"no trajectory exceeded {max_turns} steps {lengths}; a game turn "
            "spans several actions, so real games are longer than the turn limit"
        )

    @pytest.mark.timeout(300)
    def test_value_targets_are_not_uniformly_degenerate(self, small_model):
        """Across games, value targets must carry signal rather than all zeros.

        A single game may legitimately draw (0.0 throughout). Every game drawing,
        every time, is the signature of games being cut off before they finish.
        """
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay

        # Fully seeded: the model fixture, the environments, and the global
        # RNGs the Gumbel action sampling draws from. Seeding only the
        # environments -- which is all `run_batch_selfplay(seed=...)` does --
        # left this asserting on outcomes it could not reproduce. Several seeds
        # are tried because a single seed can legitimately produce an all-draw
        # batch; what must not happen is every game of every batch drawing.
        rewards: set[float] = set()
        for seed in (11, 23, 42):
            with _seed_global_rngs(seed):
                trajectories = run_batch_selfplay(
                    model=small_model,
                    num_games=2,
                    mcts_sims=2,
                    board_size=_BOARD_SIZE,
                    max_turns=12,
                    num_parallel_games=2,
                    max_batch_size=_MAX_BATCH_SIZE,
                    gumbel_num_actions=2,
                    device="cpu",
                    seed=seed,
                )
            for traj in trajectories:
                rewards.update(traj["rewards"])
            if rewards - {0.0}:
                break

        assert rewards, "no trajectories were produced at all"
        assert rewards - {0.0}, (
            "every value target across every game was 0.0 — games are not "
            "reaching a decisive outcome, so the value head has nothing to learn"
        )
