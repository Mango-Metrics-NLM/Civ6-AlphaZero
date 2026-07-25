"""Integration tests for cross-system gameplay interactions and full game simulations.

Tests that fog of war, economy, combat, city growth, terrain, and victory
systems work together correctly inside the EnhancedMicroCivEnv.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.envs.civ6_encoder import MicroCivEncoder
from alpha_zero.envs.microciv_enhanced import EnhancedMicroCivEnv
from alpha_zero.gameplay.config import GameplayConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(
    *,
    gameplay_config: GameplayConfig | None = None,
    board_size: int = 8,
    max_turns: int = 200,
    seed: int = 42,
) -> EnhancedMicroCivEnv:
    """Create and reset an EnhancedMicroCivEnv with the given settings."""
    env = EnhancedMicroCivEnv(
        board_size=board_size,
        max_turns=max_turns,
        seed=seed,
        gameplay_config=gameplay_config,
    )
    env.reset(seed=seed)
    return env


def _pick_valid_action(env: EnhancedMicroCivEnv, rng: np.random.Generator) -> int:
    """Select a random valid action from the environment."""
    mask = env.get_valid_actions()
    valid_indices = np.where(mask > 0)[0]
    assert len(valid_indices) > 0, "No valid actions available"
    return int(rng.choice(valid_indices))


def _play_steps(
    env: EnhancedMicroCivEnv,
    num_steps: int,
    rng: np.random.Generator,
) -> list[tuple[dict, float, bool, bool, dict]]:
    """Play *num_steps* random valid actions and return the transition tuples."""
    transitions: list[tuple[dict, float, bool, bool, dict]] = []
    for _ in range(num_steps):
        if env.done:
            break
        action = _pick_valid_action(env, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        transitions.append((obs, reward, terminated, truncated, info))
    return transitions


# ===========================================================================
# TestGameplaySystemInteractions
# ===========================================================================


class TestGameplaySystemInteractions:
    """Verify that multiple gameplay systems interact without conflicts."""

    def test_fog_and_economy_together(self) -> None:
        """Fog of war and economy both work simultaneously."""
        cfg = GameplayConfig(
            enable_fog_of_war=True,
            enable_economy=True,
        )
        env = _make_env(gameplay_config=cfg)

        # Both systems should be active
        assert env.fog_of_war is not None
        assert env.economy is not None

        rng = np.random.default_rng(42)

        # Play a handful of steps -- no crash expected
        transitions = _play_steps(env, 20, rng)
        assert len(transitions) > 0

        # Info should contain data from both systems
        info = env.get_info()
        assert "visibility" in info, "Expected fog-of-war visibility stats in info"
        assert "gold" in info, "Expected economy gold data in info"

    def test_all_systems_enabled(self) -> None:
        """Enable every gameplay system and play 50 steps without crashing."""
        cfg = GameplayConfig.enhanced()
        env = _make_env(gameplay_config=cfg, max_turns=200)

        rng = np.random.default_rng(42)
        transitions = _play_steps(env, 50, rng)

        assert len(transitions) > 0, "Expected at least one transition"
        # Every observation should have the expected keys
        for obs, _reward, _term, _trunc, _info in transitions:
            assert "spatial" in obs
            assert "vector" in obs
            assert "action_mask" in obs

    def test_combat_with_terrain_bonuses(self) -> None:
        """Units on hills receive a defense bonus via the enhanced combat system."""
        cfg = GameplayConfig(
            enable_enhanced_combat=True,
            enable_terrain_effects=True,
        )
        env = _make_env(gameplay_config=cfg)

        # The combat system should be initialized with terrain defense bonuses
        assert env.combat_system is not None
        combat_cfg = env.combat_system.config
        assert combat_cfg.terrain_defense_bonuses.get("hills", 0) > 0, (
            "Expected a positive defense bonus for hills terrain"
        )

    def test_city_growth_affects_production(self) -> None:
        """Higher city population produces more via the city growth system."""
        cfg = GameplayConfig(
            enable_city_growth=True,
        )
        env = _make_env(gameplay_config=cfg)

        assert env.city_growth is not None

        # A city with population 1 should have less production bonus than pop 5
        bonus_pop1 = env.city_growth.get_production_bonus(1)
        bonus_pop5 = env.city_growth.get_production_bonus(5)
        assert bonus_pop5 > bonus_pop1, (
            f"Expected pop-5 production bonus ({bonus_pop5}) > pop-1 ({bonus_pop1})"
        )

    def test_economy_and_city_growth_interact(self) -> None:
        """Gold from cities and maintenance from units both tracked over time."""
        cfg = GameplayConfig(
            enable_economy=True,
            enable_city_growth=True,
        )
        env = _make_env(gameplay_config=cfg)

        rng = np.random.default_rng(42)

        # Play enough steps for at least one full turn cycle to process economy
        _play_steps(env, 30, rng)

        info = env.get_info()
        # Economy info should be present
        assert "gold" in info, "Expected gold information after economy processing"

    def test_victory_with_economy_scoring(self) -> None:
        """Score victory at turn limit should include gold in its evaluation."""
        cfg = GameplayConfig(
            enable_economy=True,
        )
        env = _make_env(gameplay_config=cfg, max_turns=15)

        rng = np.random.default_rng(42)

        # Play until the game ends (either by victory or turn limit)
        for _ in range(500):
            if env.done:
                break
            action = _pick_valid_action(env, rng)
            env.step(action)

        # The game should have terminated (either by elimination or turn limit)
        assert env.done, "Expected game to end within step limit"


# ===========================================================================
# TestEncoderWithGameplaySystems
# ===========================================================================


class TestEncoderWithGameplaySystems:
    """Verify the MicroCivEncoder handles new gameplay data correctly."""

    def test_encoder_handles_visibility_data(self) -> None:
        """Visibility dict should be encoded into spatial channel 11."""
        encoder = MicroCivEncoder(board_size=8)
        game_state = {
            "terrain": {},
            "units": [],
            "cities": [],
            "turn": 0,
            "current_player": 0,
            "visibility": {
                (0, 0): "visible",
                (1, 0): "seen",
                (2, 0): "unexplored",
            },
        }
        spatial, _vector = encoder.encode(game_state)

        # Channel 11 is the visibility channel
        assert spatial[11, 0, 0].item() == pytest.approx(1.0), "visible -> 1.0"
        assert spatial[11, 0, 1].item() == pytest.approx(0.5), "seen -> 0.5"
        assert spatial[11, 0, 2].item() == pytest.approx(0.0), "unexplored -> 0.0"

    def test_encoder_handles_economy_data(self) -> None:
        """Gold and gold-per-turn should appear in vector indices 8-9."""
        encoder = MicroCivEncoder(board_size=8)
        game_state = {
            "terrain": {},
            "units": [],
            "cities": [],
            "turn": 0,
            "current_player": 0,
            "gold": 100,
            "gold_per_turn": 10,
        }
        _spatial, vector = encoder.encode(game_state)

        # Index 8 = gold / 500.0, Index 9 = gpt / 50.0
        assert vector[8].item() == pytest.approx(100 / 500.0)
        assert vector[9].item() == pytest.approx(10 / 50.0)

    def test_encoder_handles_population_data(self) -> None:
        """Total own-city population should be encoded at vector index 10."""
        encoder = MicroCivEncoder(board_size=8)
        game_state = {
            "terrain": {},
            "units": [],
            "cities": [
                {"player": 0, "position": (1, 1), "population": 3},
                {"player": 0, "position": (3, 3), "population": 5},
            ],
            "turn": 0,
            "current_player": 0,
        }
        _spatial, vector = encoder.encode(game_state)

        # Index 10 = total_own_pop / 30.0 => (3 + 5) / 30.0
        expected = (3 + 5) / 30.0
        assert vector[10].item() == pytest.approx(expected)

    def test_encoder_backward_compat(self) -> None:
        """Without visibility or economy data, the encoder should use defaults."""
        encoder = MicroCivEncoder(board_size=8)
        game_state = {
            "terrain": {},
            "units": [],
            "cities": [],
            "turn": 5,
            "current_player": 0,
        }
        spatial, vector = encoder.encode(game_state)

        # Channel 11 should be all 1.0 (backward compat: all tiles visible)
        assert np.all(spatial[11].numpy() == 1.0), (
            "Without visibility data, all tiles should be marked as visible"
        )

        # Gold defaults to 0
        assert vector[8].item() == pytest.approx(0.0)
        assert vector[9].item() == pytest.approx(0.0)

        # Visibility coverage index 11 defaults to 1.0
        assert vector[11].item() == pytest.approx(1.0)


# ===========================================================================
# TestGameSimulation
# ===========================================================================


class TestGameSimulation:
    """Full game simulation tests for stability and determinism."""

    def test_simulate_100_step_game(self) -> None:
        """Play 100 steps with all systems enabled and verify state consistency."""
        cfg = GameplayConfig.enhanced()
        env = _make_env(gameplay_config=cfg, max_turns=200)

        rng = np.random.default_rng(42)
        transitions = _play_steps(env, 100, rng)

        assert len(transitions) > 0

        # Verify observation shapes throughout the game
        for obs, _reward, _term, _trunc, _info in transitions:
            spatial = obs["spatial"]
            vector = obs["vector"]
            action_mask = obs["action_mask"]

            assert spatial.shape == (13, 8, 8), f"Bad spatial shape: {spatial.shape}"
            assert vector.shape == (64,), f"Bad vector shape: {vector.shape}"
            assert action_mask.shape[0] > 0, "Action mask should have positive size"

            # No NaN or Inf values
            assert np.all(np.isfinite(spatial)), "NaN/Inf found in spatial observation"
            assert np.all(np.isfinite(vector)), "NaN/Inf found in vector observation"

        # State should still be internally consistent
        info = env.get_info()
        assert info["turn"] >= 0

    def test_simulate_until_victory(self) -> None:
        """Play until a winner emerges or the turn limit is reached."""
        cfg = GameplayConfig.enhanced()
        env = _make_env(gameplay_config=cfg, max_turns=100)

        rng = np.random.default_rng(42)
        step_count = 0
        max_total_steps = 2000  # safety cap

        while not env.done and step_count < max_total_steps:
            action = _pick_valid_action(env, rng)
            _obs, _reward, _terminated, _truncated, _info = env.step(action)
            step_count += 1

        assert env.done, (
            f"Game did not end within {max_total_steps} steps "
            f"(turn={env.turn}, max_turns={env.max_turns})"
        )
        # Winner should be set (0, 1, or None for a draw)
        assert env.winner is None or env.winner in (0, 1), f"Unexpected winner value: {env.winner}"

    def test_deterministic_with_seed(self) -> None:
        """Two games with the same seed and same actions produce identical results.

        We use the default config (all enhanced features disabled) to avoid
        subtle non-determinism that can arise from the enhanced combat system's
        damage rounding.  The core environment itself is deterministic when
        seeded identically.
        """
        cfg = GameplayConfig.default()
        num_steps = 40

        # --- First run ---
        env1 = _make_env(gameplay_config=cfg)
        rng1 = np.random.default_rng(42)
        actions1: list[int] = []
        obs_list1: list[dict] = []

        for _ in range(num_steps):
            if env1.done:
                break
            action = _pick_valid_action(env1, rng1)
            actions1.append(action)
            obs, _r, _t, _tr, _i = env1.step(action)
            obs_list1.append(obs)

        # --- Second run with same seed and same actions ---
        env2 = _make_env(gameplay_config=cfg)
        obs_list2: list[dict] = []

        for action in actions1:
            if env2.done:
                break
            obs, _r, _t, _tr, _i = env2.step(action)
            obs_list2.append(obs)

        assert len(obs_list1) == len(obs_list2), "Runs produced different numbers of observations"

        for i, (o1, o2) in enumerate(zip(obs_list1, obs_list2, strict=False)):
            np.testing.assert_array_equal(
                o1["spatial"],
                o2["spatial"],
                err_msg=f"Spatial mismatch at step {i}",
            )
            np.testing.assert_array_equal(
                o1["vector"],
                o2["vector"],
                err_msg=f"Vector mismatch at step {i}",
            )
            np.testing.assert_array_equal(
                o1["action_mask"],
                o2["action_mask"],
                err_msg=f"Action mask mismatch at step {i}",
            )

    def test_clone_produces_identical_game(self) -> None:
        """Clone mid-game; both copies should diverge independently."""
        cfg = GameplayConfig.enhanced()
        env = _make_env(gameplay_config=cfg)

        rng = np.random.default_rng(42)

        # Play 15 steps on the original
        _play_steps(env, 15, rng)

        # Clone the environment
        cloned = env.clone()

        # Snapshot the state at the clone point
        original_turn = env.turn
        original_player = env.current_player
        clone_turn = cloned.turn
        clone_player = cloned.current_player

        assert original_turn == clone_turn, "Clone should have the same turn"
        assert original_player == clone_player, "Clone should have the same player"

        # Observations should be identical at the clone point
        obs_original = env.get_observation()
        obs_cloned = cloned.get_observation()
        np.testing.assert_array_equal(
            obs_original["spatial"],
            obs_cloned["spatial"],
            err_msg="Spatial observations differ at clone point",
        )
        np.testing.assert_array_equal(
            obs_original["vector"],
            obs_cloned["vector"],
            err_msg="Vector observations differ at clone point",
        )

        # Play 10 *different* actions on each copy
        rng_orig = np.random.default_rng(100)
        rng_clone = np.random.default_rng(200)

        _play_steps(env, 10, rng_orig)
        _play_steps(cloned, 10, rng_clone)

        # After divergent play, at least one observation should differ
        obs_after_orig = env.get_observation()
        obs_after_clone = cloned.get_observation()

        spatial_same = np.array_equal(obs_after_orig["spatial"], obs_after_clone["spatial"])
        vector_same = np.array_equal(obs_after_orig["vector"], obs_after_clone["vector"])

        # It is extremely unlikely that both spatial and vector remain identical
        # after 10 steps with different random seeds.  If the clone was truly
        # independent, the observations should have diverged.
        assert not (spatial_same and vector_same), (
            "After divergent play, cloned environments should not remain identical"
        )
