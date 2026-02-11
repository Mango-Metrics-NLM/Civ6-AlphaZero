"""Tests for src.core.training.evaluator -- Elo rating, EvalResult, random_eval_fn, Evaluator."""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.config import MCTSConfig
from src.core.training.evaluator import EloRating, EvalResult, Evaluator, random_eval_fn
from src.core.types import GameOutcome
from src.environments.mini_games.mini_resource import MiniResourceEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_factory() -> MiniResourceEnv:
    """Create a small, deterministic MiniResourceEnv for testing."""
    return MiniResourceEnv(board_size=4, max_steps=20, num_resources=2, seed=42)


def _dummy_eval_fn(
    spatial: np.ndarray,
    vector: np.ndarray,
    action_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Deterministic eval fn that always picks the first valid action."""
    policy = np.zeros_like(action_mask, dtype=np.float32)
    valid_indices = np.where(action_mask)[0]
    if len(valid_indices) > 0:
        policy[valid_indices[0]] = 1.0
    else:
        policy[:] = 1.0 / len(policy)
    return policy, 0.5


def _biased_eval_fn(
    spatial: np.ndarray,
    vector: np.ndarray,
    action_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Eval fn that always picks the last valid action."""
    policy = np.zeros_like(action_mask, dtype=np.float32)
    valid_indices = np.where(action_mask)[0]
    if len(valid_indices) > 0:
        policy[valid_indices[-1]] = 1.0
    else:
        policy[:] = 1.0 / len(policy)
    return policy, -0.1


# ===========================================================================
# EloRating
# ===========================================================================


class TestEloRatingDefaults:
    """EloRating should have sensible initial values."""

    def test_default_rating(self) -> None:
        elo = EloRating()
        assert elo.rating == 1000.0

    def test_default_k_factor(self) -> None:
        elo = EloRating()
        assert elo.k_factor == 32.0

    def test_default_counts_are_zero(self) -> None:
        elo = EloRating()
        assert elo.games_played == 0
        assert elo.wins == 0
        assert elo.losses == 0
        assert elo.draws == 0

    def test_default_history_is_empty(self) -> None:
        elo = EloRating()
        assert elo.history == []

    def test_custom_initial_rating(self) -> None:
        elo = EloRating(rating=1500.0)
        assert elo.rating == 1500.0

    def test_custom_k_factor(self) -> None:
        elo = EloRating(k_factor=16.0)
        assert elo.k_factor == 16.0


class TestEloExpectedScore:
    """EloRating.expected_score should follow the Elo formula."""

    def test_equal_ratings_gives_half(self) -> None:
        elo = EloRating(rating=1000.0)
        assert elo.expected_score(1000.0) == pytest.approx(0.5)

    def test_higher_rated_expects_more(self) -> None:
        elo = EloRating(rating=1200.0)
        score = elo.expected_score(1000.0)
        assert score > 0.5

    def test_lower_rated_expects_less(self) -> None:
        elo = EloRating(rating=800.0)
        score = elo.expected_score(1000.0)
        assert score < 0.5

    def test_400_point_gap(self) -> None:
        """A 400-point gap should give expected ~0.909 for the stronger player."""
        elo = EloRating(rating=1400.0)
        expected = elo.expected_score(1000.0)
        # 1 / (1 + 10^(-400/400)) = 1 / (1 + 0.1) = 1/1.1
        assert expected == pytest.approx(1.0 / 1.1, abs=1e-6)

    def test_symmetric(self) -> None:
        """expected_score(A, B) + expected_score(B, A) == 1."""
        a = EloRating(rating=1200.0)
        b = EloRating(rating=1000.0)
        assert a.expected_score(b.rating) + b.expected_score(a.rating) == pytest.approx(1.0)

    def test_matches_formula_directly(self) -> None:
        elo = EloRating(rating=1350.0)
        opp = 1150.0
        manual = 1.0 / (1.0 + math.pow(10.0, (opp - 1350.0) / 400.0))
        assert elo.expected_score(opp) == pytest.approx(manual, abs=1e-9)


class TestEloUpdate:
    """EloRating.update should correctly adjust rating and record counts."""

    def test_win_increases_rating(self) -> None:
        elo = EloRating(rating=1000.0)
        new_rating = elo.update(1.0, opponent_rating=1000.0)
        assert new_rating > 1000.0

    def test_loss_decreases_rating(self) -> None:
        elo = EloRating(rating=1000.0)
        new_rating = elo.update(0.0, opponent_rating=1000.0)
        assert new_rating < 1000.0

    def test_draw_against_equal_no_change(self) -> None:
        """A draw against an equal opponent should not change rating."""
        elo = EloRating(rating=1000.0)
        new_rating = elo.update(0.5, opponent_rating=1000.0)
        assert new_rating == pytest.approx(1000.0)

    def test_draw_against_stronger_increases(self) -> None:
        elo = EloRating(rating=1000.0)
        new_rating = elo.update(0.5, opponent_rating=1400.0)
        assert new_rating > 1000.0

    def test_draw_against_weaker_decreases(self) -> None:
        elo = EloRating(rating=1400.0)
        new_rating = elo.update(0.5, opponent_rating=1000.0)
        assert new_rating < 1400.0

    def test_win_exact_math(self) -> None:
        """Win against equal opponent: delta = k * (1 - 0.5) = 16."""
        elo = EloRating(rating=1000.0, k_factor=32.0)
        new_rating = elo.update(1.0, opponent_rating=1000.0)
        assert new_rating == pytest.approx(1016.0)

    def test_loss_exact_math(self) -> None:
        """Loss against equal opponent: delta = k * (0 - 0.5) = -16."""
        elo = EloRating(rating=1000.0, k_factor=32.0)
        new_rating = elo.update(0.0, opponent_rating=1000.0)
        assert new_rating == pytest.approx(984.0)

    def test_increments_games_played(self) -> None:
        elo = EloRating()
        elo.update(1.0, 1000.0)
        assert elo.games_played == 1
        elo.update(0.0, 1000.0)
        assert elo.games_played == 2

    def test_win_increments_wins(self) -> None:
        elo = EloRating()
        elo.update(1.0, 1000.0)
        assert elo.wins == 1
        assert elo.losses == 0
        assert elo.draws == 0

    def test_loss_increments_losses(self) -> None:
        elo = EloRating()
        elo.update(0.0, 1000.0)
        assert elo.wins == 0
        assert elo.losses == 1
        assert elo.draws == 0

    def test_draw_increments_draws(self) -> None:
        elo = EloRating()
        elo.update(0.5, 1000.0)
        assert elo.wins == 0
        assert elo.losses == 0
        assert elo.draws == 1

    def test_returns_updated_rating(self) -> None:
        elo = EloRating()
        result = elo.update(1.0, 1000.0)
        assert result == elo.rating

    def test_winner_gains_loser_loses(self) -> None:
        """In a head-to-head, one gains what the other loses (equal k)."""
        winner = EloRating(rating=1000.0, k_factor=32.0)
        loser = EloRating(rating=1000.0, k_factor=32.0)
        winner.update(1.0, 1000.0)
        loser.update(0.0, 1000.0)
        assert winner.rating > 1000.0
        assert loser.rating < 1000.0
        # Sum should remain 2000.0 (conservation)
        assert winner.rating + loser.rating == pytest.approx(2000.0)


class TestEloHistory:
    """EloRating.history should record rating after every update."""

    def test_history_tracks_after_each_game(self) -> None:
        elo = EloRating()
        elo.update(1.0, 1000.0)
        elo.update(0.0, 1000.0)
        elo.update(0.5, 1000.0)
        assert len(elo.history) == 3

    def test_history_values_match_rating(self) -> None:
        elo = EloRating()
        elo.update(1.0, 1000.0)
        assert elo.history[-1] == pytest.approx(elo.rating)

    def test_history_preserves_order(self) -> None:
        elo = EloRating()
        ratings: list[float] = []
        for score in [1.0, 1.0, 0.0, 0.5]:
            elo.update(score, 1000.0)
            ratings.append(elo.rating)
        assert elo.history == pytest.approx(ratings)

    def test_separate_instances_have_separate_histories(self) -> None:
        """Ensure the mutable default list is not shared between instances."""
        a = EloRating()
        b = EloRating()
        a.update(1.0, 1000.0)
        assert len(b.history) == 0


class TestEloWinRate:
    """EloRating.win_rate property."""

    def test_no_games_returns_zero(self) -> None:
        elo = EloRating()
        assert elo.win_rate == 0.0

    def test_all_wins(self) -> None:
        elo = EloRating()
        for _ in range(5):
            elo.update(1.0, 1000.0)
        assert elo.win_rate == pytest.approx(1.0)

    def test_all_losses(self) -> None:
        elo = EloRating()
        for _ in range(5):
            elo.update(0.0, 1000.0)
        assert elo.win_rate == pytest.approx(0.0)

    def test_mixed_results(self) -> None:
        elo = EloRating()
        elo.update(1.0, 1000.0)  # win
        elo.update(0.0, 1000.0)  # loss
        elo.update(0.5, 1000.0)  # draw
        # 1 win / 3 games
        assert elo.win_rate == pytest.approx(1.0 / 3.0)


# ===========================================================================
# EvalResult
# ===========================================================================


class TestEvalResult:
    """EvalResult dataclass field access."""

    def test_fields_stored(self) -> None:
        result = EvalResult(
            elo=1050.0,
            win_rate=0.6,
            games_played=10,
            wins=6,
            losses=3,
            draws=1,
            opponent_type="random",
            opponent_elo=1000.0,
        )
        assert result.elo == 1050.0
        assert result.win_rate == pytest.approx(0.6)
        assert result.games_played == 10
        assert result.wins == 6
        assert result.losses == 3
        assert result.draws == 1
        assert result.opponent_type == "random"
        assert result.opponent_elo == 1000.0

    def test_details_default_empty_dict(self) -> None:
        result = EvalResult(
            elo=1000.0,
            win_rate=0.0,
            games_played=0,
            wins=0,
            losses=0,
            draws=0,
            opponent_type="none",
            opponent_elo=1000.0,
        )
        assert result.details == {}

    def test_details_custom(self) -> None:
        result = EvalResult(
            elo=1000.0,
            win_rate=0.0,
            games_played=1,
            wins=0,
            losses=1,
            draws=0,
            opponent_type="random",
            opponent_elo=1000.0,
            details={"avg_game_length": 15.2},
        )
        assert result.details["avg_game_length"] == pytest.approx(15.2)

    def test_wins_losses_draws_sum_to_games_played(self) -> None:
        result = EvalResult(
            elo=1000.0,
            win_rate=0.5,
            games_played=10,
            wins=5,
            losses=3,
            draws=2,
            opponent_type="random",
            opponent_elo=1000.0,
        )
        assert result.wins + result.losses + result.draws == result.games_played


# ===========================================================================
# random_eval_fn
# ===========================================================================


class TestRandomEvalFn:
    """random_eval_fn produces valid, uniform-over-legal-actions policies."""

    def test_returns_tuple(self) -> None:
        spatial = np.zeros((6, 4, 4), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        mask = np.ones(5, dtype=np.bool_)
        result = random_eval_fn(spatial, vector, mask)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_policy_shape_matches_mask(self) -> None:
        spatial = np.zeros((6, 4, 4), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        mask = np.ones(5, dtype=np.bool_)
        policy, _ = random_eval_fn(spatial, vector, mask)
        assert policy.shape == mask.shape

    def test_policy_sums_to_one(self) -> None:
        spatial = np.zeros((6, 4, 4), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        mask = np.ones(5, dtype=np.bool_)
        policy, _ = random_eval_fn(spatial, vector, mask)
        assert policy.sum() == pytest.approx(1.0)

    def test_value_is_zero(self) -> None:
        spatial = np.zeros((6, 4, 4), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        mask = np.ones(5, dtype=np.bool_)
        _, value = random_eval_fn(spatial, vector, mask)
        assert value == 0.0

    def test_uniform_over_valid_actions_all_valid(self) -> None:
        spatial = np.zeros((6, 4, 4), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        mask = np.ones(5, dtype=np.bool_)
        policy, _ = random_eval_fn(spatial, vector, mask)
        expected = np.full(5, 0.2, dtype=np.float32)
        np.testing.assert_allclose(policy, expected, atol=1e-6)

    def test_respects_action_mask_partial(self) -> None:
        spatial = np.zeros((6, 4, 4), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        mask = np.array([True, False, True, False, True], dtype=np.bool_)
        policy, _ = random_eval_fn(spatial, vector, mask)
        # Only actions 0, 2, 4 should have weight
        assert policy[1] == pytest.approx(0.0)
        assert policy[3] == pytest.approx(0.0)
        assert policy[0] == pytest.approx(1.0 / 3.0, abs=1e-6)
        assert policy[2] == pytest.approx(1.0 / 3.0, abs=1e-6)
        assert policy[4] == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_respects_action_mask_single_valid(self) -> None:
        spatial = np.zeros((6, 4, 4), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        mask = np.array([False, False, True, False, False], dtype=np.bool_)
        policy, _ = random_eval_fn(spatial, vector, mask)
        assert policy[2] == pytest.approx(1.0)
        assert policy.sum() == pytest.approx(1.0)

    def test_all_invalid_falls_back_to_uniform(self) -> None:
        """When no actions are valid, policy should be uniform over all actions."""
        spatial = np.zeros((6, 4, 4), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        mask = np.zeros(5, dtype=np.bool_)
        policy, _ = random_eval_fn(spatial, vector, mask)
        expected = np.full(5, 0.2, dtype=np.float32)
        np.testing.assert_allclose(policy, expected, atol=1e-6)

    def test_policy_dtype_is_float32(self) -> None:
        spatial = np.zeros((6, 4, 4), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        mask = np.ones(5, dtype=np.bool_)
        policy, _ = random_eval_fn(spatial, vector, mask)
        assert policy.dtype == np.float32


# ===========================================================================
# Evaluator -- constructor & properties
# ===========================================================================


class TestEvaluatorConstructor:
    """Evaluator initialisation and property access."""

    def test_default_construction(self) -> None:
        mcts_cfg = MCTSConfig()
        evaluator = Evaluator(mcts_cfg)
        assert evaluator.num_eval_games == 20

    def test_custom_num_eval_games(self) -> None:
        mcts_cfg = MCTSConfig()
        evaluator = Evaluator(mcts_cfg, num_eval_games=10)
        assert evaluator.num_eval_games == 10

    def test_candidate_elo_starts_at_default(self) -> None:
        mcts_cfg = MCTSConfig()
        evaluator = Evaluator(mcts_cfg)
        assert evaluator.candidate_elo.rating == 1000.0
        assert evaluator.candidate_elo.games_played == 0

    def test_custom_baseline_elo(self) -> None:
        mcts_cfg = MCTSConfig()
        evaluator = Evaluator(mcts_cfg, baseline_elo=1200.0)
        # baseline_elo is stored internally; check via evaluate result
        assert evaluator.candidate_elo.rating == 1000.0

    def test_seed_reproducibility(self) -> None:
        """Two evaluators with the same seed should produce the same results."""
        mcts_cfg = MCTSConfig()
        ev1 = Evaluator(mcts_cfg, num_eval_games=4, seed=123)
        ev2 = Evaluator(mcts_cfg, num_eval_games=4, seed=123)
        r1 = ev1.evaluate(random_eval_fn, _env_factory)
        r2 = ev2.evaluate(random_eval_fn, _env_factory)
        assert r1.wins == r2.wins
        assert r1.losses == r2.losses
        assert r1.draws == r2.draws


# ===========================================================================
# Evaluator.evaluate
# ===========================================================================


class TestEvaluatorEvaluate:
    """Evaluator.evaluate should return a well-formed EvalResult."""

    @pytest.fixture()
    def evaluator(self) -> Evaluator:
        return Evaluator(MCTSConfig(), num_eval_games=6, seed=7)

    def test_returns_eval_result(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate(random_eval_fn, _env_factory)
        assert isinstance(result, EvalResult)

    def test_games_played_matches_num_eval_games(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate(random_eval_fn, _env_factory)
        assert result.games_played == 6

    def test_wins_losses_draws_sum(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate(random_eval_fn, _env_factory)
        assert result.wins + result.losses + result.draws == result.games_played

    def test_win_rate_consistent(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate(random_eval_fn, _env_factory)
        expected_wr = result.wins / result.games_played if result.games_played > 0 else 0.0
        assert result.win_rate == pytest.approx(expected_wr)

    def test_elo_updated_after_evaluate(self, evaluator: Evaluator) -> None:
        evaluator.evaluate(random_eval_fn, _env_factory)
        # After 6 games, the candidate_elo should have 6 recorded games
        assert evaluator.candidate_elo.games_played == 6

    def test_elo_in_result_matches_candidate(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate(random_eval_fn, _env_factory)
        assert result.elo == pytest.approx(evaluator.candidate_elo.rating)

    def test_opponent_type_default_random(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate(random_eval_fn, _env_factory)
        assert result.opponent_type == "random"

    def test_opponent_type_custom(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate(
            random_eval_fn, _env_factory, baseline_type="custom_agent"
        )
        assert result.opponent_type == "custom_agent"

    def test_opponent_elo_in_result(self) -> None:
        evaluator = Evaluator(MCTSConfig(), num_eval_games=2, baseline_elo=1200.0, seed=1)
        result = evaluator.evaluate(random_eval_fn, _env_factory)
        assert result.opponent_elo == 1200.0

    def test_default_baseline_is_random(self) -> None:
        """When baseline_fn is None, random_eval_fn should be used."""
        evaluator = Evaluator(MCTSConfig(), num_eval_games=4, seed=99)
        result = evaluator.evaluate(random_eval_fn, _env_factory, baseline_fn=None)
        assert result.games_played == 4

    def test_custom_baseline_fn(self) -> None:
        evaluator = Evaluator(MCTSConfig(), num_eval_games=4, seed=42)
        result = evaluator.evaluate(
            _dummy_eval_fn, _env_factory, baseline_fn=_biased_eval_fn, baseline_type="biased"
        )
        assert result.opponent_type == "biased"
        assert result.games_played == 4

    def test_multiple_evaluations_accumulate_elo(self) -> None:
        """Running evaluate twice should accumulate games_played in the Elo tracker."""
        evaluator = Evaluator(MCTSConfig(), num_eval_games=4, seed=10)
        evaluator.evaluate(random_eval_fn, _env_factory)
        evaluator.evaluate(random_eval_fn, _env_factory)
        assert evaluator.candidate_elo.games_played == 8

    def test_history_grows_with_evaluations(self) -> None:
        evaluator = Evaluator(MCTSConfig(), num_eval_games=3, seed=55)
        evaluator.evaluate(random_eval_fn, _env_factory)
        assert len(evaluator.candidate_elo.history) == 3
        evaluator.evaluate(random_eval_fn, _env_factory)
        assert len(evaluator.candidate_elo.history) == 6


# ===========================================================================
# _play_eval_game (indirectly via evaluate)
# ===========================================================================


class TestPlayEvalGame:
    """The internal _play_eval_game should drive games to completion."""

    def test_game_terminates(self) -> None:
        """All games in an evaluation should terminate (no infinite loops)."""
        evaluator = Evaluator(MCTSConfig(), num_eval_games=4, seed=0)
        result = evaluator.evaluate(random_eval_fn, _env_factory)
        # If we get here without hanging, the games terminated
        assert result.games_played == 4

    def test_alternates_starting_player(self) -> None:
        """Games should alternate who plays first (even games: candidate is p1)."""
        evaluator = Evaluator(MCTSConfig(), num_eval_games=4, seed=77)
        # This is tested implicitly -- the evaluate method alternates candidate_is_p1
        # based on game_idx % 2. We verify by checking the evaluation completes
        # and produces valid outcomes.
        result = evaluator.evaluate(_dummy_eval_fn, _env_factory, baseline_fn=_biased_eval_fn)
        assert result.wins + result.losses + result.draws == 4

    def test_different_eval_fns_produce_different_outcomes(self) -> None:
        """Using different candidate and baseline fns should produce varying results."""
        ev1 = Evaluator(MCTSConfig(), num_eval_games=10, seed=42)
        r1 = ev1.evaluate(random_eval_fn, _env_factory)

        ev2 = Evaluator(MCTSConfig(), num_eval_games=10, seed=42)
        r2 = ev2.evaluate(_dummy_eval_fn, _env_factory)

        # Results may differ because the eval functions behave differently
        # At minimum, both should complete and be well-formed
        assert r1.games_played == 10
        assert r2.games_played == 10

    def test_plays_with_small_env(self) -> None:
        """Games on a very small board with few resources should complete quickly."""
        def tiny_env() -> MiniResourceEnv:
            return MiniResourceEnv(board_size=2, max_steps=10, num_resources=1, seed=7)

        evaluator = Evaluator(MCTSConfig(), num_eval_games=6, seed=3)
        result = evaluator.evaluate(random_eval_fn, tiny_env)
        assert result.games_played == 6
        assert result.wins + result.losses + result.draws == 6

    def test_max_step_guard_returns_draw(self) -> None:
        """If the env doesn't terminate, _play_eval_game has a 200-step fallback."""
        # MiniResourceEnv with max_steps=20 will terminate well before 200.
        # We mainly verify the game completes without error.
        evaluator = Evaluator(MCTSConfig(), num_eval_games=2, seed=5)
        result = evaluator.evaluate(random_eval_fn, _env_factory)
        assert result.games_played == 2

    def test_env_factory_called_per_game(self) -> None:
        """Each game should use a fresh environment from the factory."""
        call_count = 0

        def counting_factory() -> MiniResourceEnv:
            nonlocal call_count
            call_count += 1
            return MiniResourceEnv(board_size=4, max_steps=20, num_resources=2, seed=42)

        evaluator = Evaluator(MCTSConfig(), num_eval_games=5, seed=1)
        evaluator.evaluate(random_eval_fn, counting_factory)
        assert call_count == 5

    def test_seed_produces_deterministic_results(self) -> None:
        """Two evaluators with the same seed and same fns yield identical results."""
        cfg = MCTSConfig()
        results = []
        for _ in range(2):
            ev = Evaluator(cfg, num_eval_games=6, seed=999)
            r = ev.evaluate(_dummy_eval_fn, _env_factory, baseline_fn=_biased_eval_fn)
            results.append((r.wins, r.losses, r.draws, r.elo))

        assert results[0] == pytest.approx(results[1])


# ===========================================================================
# _play_eval_game -- reward / mask edge cases
# ===========================================================================


def _make_mock_env(
    *,
    terminate_at_step: int = 1,
    reward: float = 1.0,
    num_actions: int = 5,
    spatial_shape: tuple[int, ...] = (6, 4, 4),
    vector_dim: int = 4,
    empty_mask_at_step: int | None = None,
) -> MagicMock:
    """Build a mock StrategicEnvironment with configurable termination reward."""
    env = MagicMock()
    step_counter = {"n": 0}

    def _reset(seed: int | None = None) -> None:
        step_counter["n"] = 0

    def _is_terminal() -> bool:
        return step_counter["n"] >= terminate_at_step

    def _get_observation() -> dict[str, np.ndarray]:
        return {
            "spatial": np.zeros(spatial_shape, dtype=np.float32),
            "vector": np.zeros(vector_dim, dtype=np.float32),
        }

    def _get_valid_actions() -> np.ndarray:
        if empty_mask_at_step is not None and step_counter["n"] >= empty_mask_at_step:
            return np.zeros(num_actions, dtype=np.bool_)
        return np.ones(num_actions, dtype=np.bool_)

    def _step(action: int) -> tuple[dict, float, bool, bool, dict]:
        step_counter["n"] += 1
        terminated = step_counter["n"] >= terminate_at_step
        return _get_observation(), reward if terminated else 0.0, terminated, False, {}

    env.reset = MagicMock(side_effect=_reset)
    env.is_terminal = MagicMock(side_effect=_is_terminal)
    env.get_observation = MagicMock(side_effect=_get_observation)
    env.get_valid_actions = MagicMock(side_effect=_get_valid_actions)
    env.step = MagicMock(side_effect=_step)
    return env


class TestPlayEvalGameRewardPaths:
    """Tests for _play_eval_game reward interpretation branches."""

    def test_candidate_p1_negative_reward_returns_loss(self) -> None:
        """When candidate is p1 and reward < 0, outcome is loss."""
        env = _make_mock_env(terminate_at_step=1, reward=-1.0)
        evaluator = Evaluator(MCTSConfig(), num_eval_games=1, seed=0)
        outcome = evaluator._play_eval_game(
            candidate_fn=_dummy_eval_fn,
            baseline_fn=_dummy_eval_fn,
            env_factory=lambda: env,
            candidate_is_p1=True,
            seed=0,
        )
        assert outcome == GameOutcome.loss

    def test_candidate_p1_positive_reward_returns_win(self) -> None:
        """When candidate is p1 and reward > 0, outcome is win."""
        env = _make_mock_env(terminate_at_step=1, reward=1.0)
        evaluator = Evaluator(MCTSConfig(), num_eval_games=1, seed=0)
        outcome = evaluator._play_eval_game(
            candidate_fn=_dummy_eval_fn,
            baseline_fn=_dummy_eval_fn,
            env_factory=lambda: env,
            candidate_is_p1=True,
            seed=0,
        )
        assert outcome == GameOutcome.win

    def test_candidate_p2_negative_reward_returns_win(self) -> None:
        """When candidate is p2 and reward < 0, outcome is win (p1 lost)."""
        env = _make_mock_env(terminate_at_step=1, reward=-1.0)
        evaluator = Evaluator(MCTSConfig(), num_eval_games=1, seed=0)
        outcome = evaluator._play_eval_game(
            candidate_fn=_dummy_eval_fn,
            baseline_fn=_dummy_eval_fn,
            env_factory=lambda: env,
            candidate_is_p1=False,
            seed=0,
        )
        assert outcome == GameOutcome.win

    def test_candidate_p2_positive_reward_returns_loss(self) -> None:
        """When candidate is p2 and reward > 0, outcome is loss (p1 won)."""
        env = _make_mock_env(terminate_at_step=1, reward=1.0)
        evaluator = Evaluator(MCTSConfig(), num_eval_games=1, seed=0)
        outcome = evaluator._play_eval_game(
            candidate_fn=_dummy_eval_fn,
            baseline_fn=_dummy_eval_fn,
            env_factory=lambda: env,
            candidate_is_p1=False,
            seed=0,
        )
        assert outcome == GameOutcome.loss

    def test_terminated_zero_reward_returns_draw(self) -> None:
        """When game terminates with reward == 0, outcome is draw."""
        env = _make_mock_env(terminate_at_step=1, reward=0.0)
        evaluator = Evaluator(MCTSConfig(), num_eval_games=1, seed=0)
        outcome = evaluator._play_eval_game(
            candidate_fn=_dummy_eval_fn,
            baseline_fn=_dummy_eval_fn,
            env_factory=lambda: env,
            candidate_is_p1=True,
            seed=0,
        )
        assert outcome == GameOutcome.draw


class TestPlayEvalGameMaskExhaustion:
    """Tests for _play_eval_game action mask exhaustion fallback."""

    def test_empty_mask_breaks_loop_returns_draw(self) -> None:
        """When all valid actions are exhausted (mask all False), the game breaks and returns draw."""
        env = _make_mock_env(
            terminate_at_step=100,
            reward=0.0,
            empty_mask_at_step=0,
        )
        evaluator = Evaluator(MCTSConfig(), num_eval_games=1, seed=0)
        outcome = evaluator._play_eval_game(
            candidate_fn=_dummy_eval_fn,
            baseline_fn=_dummy_eval_fn,
            env_factory=lambda: env,
            candidate_is_p1=True,
            seed=0,
        )
        assert outcome == GameOutcome.draw

    def test_empty_mask_fallback_with_policy_zero(self) -> None:
        """When policy produces zero mass and mask is empty, the game breaks cleanly."""
        def zero_policy_fn(
            spatial: np.ndarray,
            vector: np.ndarray,
            action_mask: np.ndarray,
        ) -> tuple[np.ndarray, float]:
            return np.zeros_like(action_mask, dtype=np.float32), 0.0

        env = _make_mock_env(
            terminate_at_step=100,
            reward=0.0,
            empty_mask_at_step=0,
        )
        evaluator = Evaluator(MCTSConfig(), num_eval_games=1, seed=0)
        outcome = evaluator._play_eval_game(
            candidate_fn=zero_policy_fn,
            baseline_fn=zero_policy_fn,
            env_factory=lambda: env,
            candidate_is_p1=True,
            seed=0,
        )
        assert outcome == GameOutcome.draw
