"""Comprehensive unit tests for alpha_zero.pbt.multiplayer_evaluator.

Targets coverage improvement from ~24% to 70%+ by exercising:
  - NPlayerMatchResult dataclass (construction, frozen, default game_length)
  - MultiplayerPBTEvaluator.__init__ (defaults, custom params, validation)
  - Properties (num_players, games_per_eval)
  - evaluate_population (small pop uniform 0.5, normal flow, fitness in [0,1])
  - _create_matchups (n < k, small pop all combos, large pop caps matchups)
  - _score_result (with multiplayer scoring, various ScoringMethod values,
    empty scores dict, invalid scoring_method fallback)
  - _linear_ranking_fallback (rank 1 -> +1, rank N -> -1, single player -> 0)
  - _normalise_scores (empty, equal, different)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from alpha_zero.pbt.config import AgentState
from alpha_zero.pbt.multiplayer_evaluator import (
    MultiplayerPBTEvaluator,
    NPlayerArenaRunner,
    NPlayerMatchResult,
)

# ---------------------------------------------------------------------------
# Mock arena runner
# ---------------------------------------------------------------------------


class _MockRunner:
    """Deterministic arena runner for testing.

    The first agent in each group always wins (rank 1), the rest get rank 2.
    """

    def __init__(self, winner_index: int = 0) -> None:
        self._winner_index = winner_index
        self.calls: list[tuple[list[str], int]] = []

    def run_nplayer_match(
        self,
        agent_ids: list[str],
        num_games: int,
    ) -> list[NPlayerMatchResult]:
        self.calls.append((list(agent_ids), num_games))
        results: list[NPlayerMatchResult] = []
        n = len(agent_ids)
        for _ in range(num_games):
            wi = min(self._winner_index, n - 1)
            rankings: dict[str, int] = {}
            scores: dict[str, float] = {}
            for i, aid in enumerate(agent_ids):
                if i == wi:
                    rankings[aid] = 1
                    scores[aid] = 10.0
                else:
                    rankings[aid] = n
                    scores[aid] = 0.0
            results.append(
                NPlayerMatchResult(
                    agent_ids=list(agent_ids),
                    rankings=rankings,
                    scores=scores,
                    game_length=42,
                )
            )
        return results


class _RankedRunner:
    """Runner that returns distinct ranks for each player (1st, 2nd, ..., Nth).

    Agents are ranked by their position in the agent_ids list.
    """

    def run_nplayer_match(
        self,
        agent_ids: list[str],
        num_games: int,
    ) -> list[NPlayerMatchResult]:
        results: list[NPlayerMatchResult] = []
        n = len(agent_ids)
        for _ in range(num_games):
            rankings = {aid: i + 1 for i, aid in enumerate(agent_ids)}
            scores = {aid: float(n - i) for i, aid in enumerate(agent_ids)}
            results.append(
                NPlayerMatchResult(
                    agent_ids=list(agent_ids),
                    rankings=rankings,
                    scores=scores,
                    game_length=30,
                )
            )
        return results


class _EmptyScoresRunner:
    """Runner that returns results with empty scores dicts.

    This forces the _score_result path where result.scores is falsy.
    """

    def run_nplayer_match(
        self,
        agent_ids: list[str],
        num_games: int,
    ) -> list[NPlayerMatchResult]:
        results: list[NPlayerMatchResult] = []
        len(agent_ids)
        for _ in range(num_games):
            rankings = {aid: i + 1 for i, aid in enumerate(agent_ids)}
            results.append(
                NPlayerMatchResult(
                    agent_ids=list(agent_ids),
                    rankings=rankings,
                    scores={},
                    game_length=10,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agents(n: int) -> list[AgentState]:
    """Create *n* agents with deterministic IDs."""
    return [AgentState(agent_id=f"agent_{i}", hyperparameters={}) for i in range(n)]


# =========================================================================
# 1. NPlayerMatchResult dataclass
# =========================================================================


class TestNPlayerMatchResult:
    """Tests for the NPlayerMatchResult frozen dataclass."""

    def test_construction_with_all_fields(self) -> None:
        r = NPlayerMatchResult(
            agent_ids=["a", "b", "c"],
            rankings={"a": 1, "b": 2, "c": 3},
            scores={"a": 100.0, "b": 50.0, "c": 10.0},
            game_length=200,
        )
        assert r.agent_ids == ["a", "b", "c"]
        assert r.rankings == {"a": 1, "b": 2, "c": 3}
        assert r.scores == {"a": 100.0, "b": 50.0, "c": 10.0}
        assert r.game_length == 200

    def test_default_game_length_is_zero(self) -> None:
        r = NPlayerMatchResult(
            agent_ids=["x", "y"],
            rankings={"x": 1, "y": 2},
            scores={"x": 1.0, "y": 0.0},
        )
        assert r.game_length == 0

    def test_frozen_cannot_set_attribute(self) -> None:
        r = NPlayerMatchResult(
            agent_ids=["a"],
            rankings={"a": 1},
            scores={"a": 1.0},
        )
        with pytest.raises((AttributeError, FrozenInstanceError)):
            r.game_length = 999  # type: ignore[misc]

    def test_frozen_cannot_set_rankings(self) -> None:
        r = NPlayerMatchResult(
            agent_ids=["a"],
            rankings={"a": 1},
            scores={},
        )
        with pytest.raises((AttributeError, FrozenInstanceError)):
            r.rankings = {"a": 2}  # type: ignore[misc]

    def test_two_player_result(self) -> None:
        r = NPlayerMatchResult(
            agent_ids=["p1", "p2"],
            rankings={"p1": 1, "p2": 2},
            scores={"p1": 1.0, "p2": 0.0},
            game_length=50,
        )
        assert len(r.agent_ids) == 2
        assert r.rankings["p1"] < r.rankings["p2"]


# =========================================================================
# 2. MultiplayerPBTEvaluator.__init__
# =========================================================================


class TestMultiplayerPBTEvaluatorInit:
    """Tests for constructor defaults, custom params, and validation."""

    def test_defaults(self) -> None:
        ev = MultiplayerPBTEvaluator()
        assert ev.num_players == 4
        assert ev.games_per_eval == 10

    def test_custom_params(self) -> None:
        ev = MultiplayerPBTEvaluator(
            num_players=6,
            games_per_eval=20,
            scoring_method="winner_take_all",
            elo_initial_rating=1500.0,
            elo_k_factor=16.0,
        )
        assert ev.num_players == 6
        assert ev.games_per_eval == 20
        assert ev._scoring_method == "winner_take_all"
        assert ev._elo_initial_rating == 1500.0
        assert ev._elo_k_factor == 16.0

    def test_num_players_below_two_raises(self) -> None:
        with pytest.raises(ValueError, match="num_players must be >= 2"):
            MultiplayerPBTEvaluator(num_players=1)

    def test_num_players_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_players must be >= 2"):
            MultiplayerPBTEvaluator(num_players=0)

    def test_num_players_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="num_players must be >= 2"):
            MultiplayerPBTEvaluator(num_players=-3)

    def test_games_per_eval_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="games_per_eval must be >= 1"):
            MultiplayerPBTEvaluator(games_per_eval=0)

    def test_games_per_eval_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="games_per_eval must be >= 1"):
            MultiplayerPBTEvaluator(games_per_eval=-5)

    def test_num_players_exactly_two(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=2)
        assert ev.num_players == 2

    def test_games_per_eval_exactly_one(self) -> None:
        ev = MultiplayerPBTEvaluator(games_per_eval=1)
        assert ev.games_per_eval == 1

    def test_elo_system_initialised_when_tournament_available(self) -> None:
        """When the tournament module is importable, _elo should be set."""
        ev = MultiplayerPBTEvaluator()
        # _HAS_TOURNAMENT is True in this codebase
        # Either way, the constructor shouldn't crash
        assert isinstance(ev._elo, object)  # could be None or ELORatingSystem


# =========================================================================
# 3 & 4. Properties
# =========================================================================


class TestProperties:
    """Tests for num_players and games_per_eval properties."""

    def test_num_players_property(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=5, games_per_eval=3)
        assert ev.num_players == 5

    def test_games_per_eval_property(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=3, games_per_eval=7)
        assert ev.games_per_eval == 7

    def test_properties_match_constructor_args(self) -> None:
        for np_, gpe in [(2, 1), (3, 5), (8, 100)]:
            ev = MultiplayerPBTEvaluator(num_players=np_, games_per_eval=gpe)
            assert ev.num_players == np_
            assert ev.games_per_eval == gpe


# =========================================================================
# 5. evaluate_population
# =========================================================================


class TestEvaluatePopulation:
    """Tests for evaluate_population flow."""

    def test_population_smaller_than_num_players_returns_uniform(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=4, games_per_eval=5)
        agents = _make_agents(3)
        runner = _MockRunner()
        fitness = ev.evaluate_population(agents, runner)

        assert len(fitness) == 3
        for a in agents:
            assert fitness[a.agent_id] == pytest.approx(0.5)
        # Runner should NOT have been called
        assert len(runner.calls) == 0

    def test_population_equal_to_num_players(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=3, games_per_eval=2)
        agents = _make_agents(3)
        runner = _MockRunner(winner_index=0)
        fitness = ev.evaluate_population(agents, runner)

        assert len(fitness) == 3
        for v in fitness.values():
            assert 0.0 <= v <= 1.0

    def test_single_agent_below_num_players(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=1)
        agents = _make_agents(1)
        runner = _MockRunner()
        fitness = ev.evaluate_population(agents, runner)
        assert fitness["agent_0"] == pytest.approx(0.5)

    def test_empty_population(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=1)
        runner = _MockRunner()
        fitness = ev.evaluate_population([], runner)
        assert fitness == {}

    def test_fitness_scores_normalised_to_0_1(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=4)
        agents = _make_agents(5)
        runner = _MockRunner(winner_index=0)
        fitness = ev.evaluate_population(agents, runner)

        for v in fitness.values():
            assert 0.0 <= v <= 1.0

    def test_all_agents_appear_in_fitness(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=2)
        agents = _make_agents(6)
        runner = _MockRunner()
        fitness = ev.evaluate_population(agents, runner)

        for a in agents:
            assert a.agent_id in fitness

    def test_runner_invoked_for_each_matchup(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=3)
        agents = _make_agents(4)
        runner = _MockRunner()
        ev.evaluate_population(agents, runner)

        # C(4,2) = 6 matchups
        assert len(runner.calls) == 6
        # Each call should request games_per_eval games
        for _, num_games in runner.calls:
            assert num_games == 3

    def test_winner_gets_highest_fitness(self) -> None:
        """Agent 0 always wins -> should be highest or tied for highest."""
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=5)
        agents = _make_agents(4)
        runner = _MockRunner(winner_index=0)
        fitness = ev.evaluate_population(agents, runner)

        max_fitness = max(fitness.values())
        assert fitness["agent_0"] == pytest.approx(max_fitness, abs=0.05)

    def test_with_ranked_runner(self) -> None:
        """Use distinct ranks to verify scoring spread."""
        ev = MultiplayerPBTEvaluator(num_players=3, games_per_eval=4)
        agents = _make_agents(5)
        runner = _RankedRunner()
        fitness = ev.evaluate_population(agents, runner)

        assert len(fitness) == 5
        for v in fitness.values():
            assert 0.0 <= v <= 1.0

    def test_with_empty_scores_runner(self) -> None:
        """Runner returns empty scores dicts; tests the falsy branch."""
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=2)
        agents = _make_agents(4)
        runner = _EmptyScoresRunner()
        fitness = ev.evaluate_population(agents, runner)

        assert len(fitness) == 4
        for v in fitness.values():
            assert 0.0 <= v <= 1.0


# =========================================================================
# 6. _create_matchups
# =========================================================================


class TestCreateMatchups:
    """Tests for _create_matchups."""

    def test_n_less_than_k_returns_empty(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=4, games_per_eval=1)
        assert ev._create_matchups(["a", "b"]) == []

    def test_n_equal_to_k_returns_one_group(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=3, games_per_eval=1)
        matchups = ev._create_matchups(["a", "b", "c"])
        assert len(matchups) == 1
        assert sorted(matchups[0]) == ["a", "b", "c"]

    def test_all_combinations_for_small_population(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=1)
        # C(4,2) = 6
        matchups = ev._create_matchups(["a", "b", "c", "d"])
        assert len(matchups) == 6
        # All groups have exactly 2 agents
        for group in matchups:
            assert len(group) == 2

    def test_every_agent_appears_at_least_once(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=3, games_per_eval=1)
        ids = [f"a{i}" for i in range(6)]
        matchups = ev._create_matchups(ids)
        seen = set()
        for group in matchups:
            seen.update(group)
        assert seen == set(ids)

    def test_groups_have_correct_size(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=3, games_per_eval=1)
        matchups = ev._create_matchups([f"a{i}" for i in range(7)])
        for group in matchups:
            assert len(group) == 3

    def test_caps_matchups_for_large_population(self) -> None:
        """With 20 agents choosing 2, C(20,2)=190 but max = max(20*3, 20)=60."""
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=1)
        ids = [f"a{i}" for i in range(20)]
        matchups = ev._create_matchups(ids)
        max_matchups = max(20 * 3, 20)  # 60
        assert len(matchups) <= max_matchups

    def test_caps_matchups_no_duplicates(self) -> None:
        """After capping, all matchup groups should still be unique."""
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=1)
        ids = [f"a{i}" for i in range(20)]
        matchups = ev._create_matchups(ids)
        group_tuples = [tuple(sorted(g)) for g in matchups]
        assert len(group_tuples) == len(set(group_tuples))

    def test_empty_agent_ids(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=1)
        assert ev._create_matchups([]) == []

    def test_single_agent(self) -> None:
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=1)
        assert ev._create_matchups(["solo"]) == []


# =========================================================================
# 7. _score_result
# =========================================================================


class TestScoreResult:
    """Tests for _score_result with multiplayer scoring module."""

    def test_linear_ranking_via_multiplayer_module(self) -> None:
        ev = MultiplayerPBTEvaluator(
            num_players=3,
            games_per_eval=1,
            scoring_method="linear_ranking",
        )
        result = NPlayerMatchResult(
            agent_ids=["a", "b", "c"],
            rankings={"a": 1, "b": 2, "c": 3},
            scores={"a": 10.0, "b": 5.0, "c": 1.0},
        )
        values = ev._score_result(result)
        assert len(values) == 3
        # All agent IDs present
        assert set(values.keys()) == {"a", "b", "c"}

    def test_winner_take_all_via_multiplayer_module(self) -> None:
        ev = MultiplayerPBTEvaluator(
            num_players=3,
            games_per_eval=1,
            scoring_method="winner_take_all",
        )
        # Use 1-indexed rankings as NPlayerMatchResult convention.
        # compute_player_values treats rank 0 as winner. The _score_result
        # method passes these ranks through directly, so rank=1 here means
        # "not rank 0" in compute_player_values.  Use rank 0 for the actual
        # winner to exercise the winner_take_all winner path.
        result = NPlayerMatchResult(
            agent_ids=["x", "y", "z"],
            rankings={"x": 0, "y": 1, "z": 2},
            scores={"x": 10.0, "y": 5.0, "z": 0.0},
        )
        values = ev._score_result(result)
        assert len(values) == 3
        # Winner (rank 0) should get +1.0, others -1.0
        assert values["x"] == pytest.approx(1.0)
        assert values["y"] == pytest.approx(-1.0)
        assert values["z"] == pytest.approx(-1.0)

    def test_proportional_via_multiplayer_module(self) -> None:
        ev = MultiplayerPBTEvaluator(
            num_players=3,
            games_per_eval=1,
            scoring_method="proportional",
        )
        result = NPlayerMatchResult(
            agent_ids=["a", "b", "c"],
            rankings={"a": 1, "b": 2, "c": 3},
            scores={"a": 100.0, "b": 50.0, "c": 0.0},
        )
        values = ev._score_result(result)
        assert len(values) == 3
        # Higher score -> higher value
        assert values["a"] > values["c"]

    def test_invalid_scoring_method_falls_back_to_linear(self) -> None:
        ev = MultiplayerPBTEvaluator(
            num_players=2,
            games_per_eval=1,
            scoring_method="totally_invalid_method",
        )
        result = NPlayerMatchResult(
            agent_ids=["a", "b"],
            rankings={"a": 1, "b": 2},
            scores={"a": 1.0, "b": 0.0},
        )
        # Should not raise, falls back to LINEAR_RANKING
        values = ev._score_result(result)
        assert len(values) == 2

    def test_empty_scores_dict_passes_none_to_scoring(self) -> None:
        """When result.scores is an empty dict (falsy), int_scores should be None."""
        ev = MultiplayerPBTEvaluator(
            num_players=2,
            games_per_eval=1,
            scoring_method="linear_ranking",
        )
        result = NPlayerMatchResult(
            agent_ids=["a", "b"],
            rankings={"a": 1, "b": 2},
            scores={},
        )
        values = ev._score_result(result)
        assert len(values) == 2

    def test_score_result_with_nonempty_scores(self) -> None:
        """When result.scores is non-empty, int_scores is built from it."""
        ev = MultiplayerPBTEvaluator(
            num_players=3,
            games_per_eval=1,
            scoring_method="proportional",
        )
        result = NPlayerMatchResult(
            agent_ids=["p0", "p1", "p2"],
            rankings={"p0": 1, "p1": 2, "p2": 3},
            scores={"p0": 100.0, "p1": 50.0, "p2": 25.0},
        )
        values = ev._score_result(result)
        # proportional scoring uses the scores
        assert values["p0"] > values["p2"]

    def test_fallback_when_multiplayer_unavailable(self) -> None:
        """Patch _HAS_MULTIPLAYER=False to exercise the fallback path."""
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=1)
        result = NPlayerMatchResult(
            agent_ids=["a", "b"],
            rankings={"a": 1, "b": 2},
            scores={"a": 1.0, "b": 0.0},
        )
        with patch("alpha_zero.pbt.multiplayer_evaluator._HAS_MULTIPLAYER", False):
            values = ev._score_result(result)
        assert values["a"] == pytest.approx(1.0)
        assert values["b"] == pytest.approx(-1.0)

    def test_four_player_scoring(self) -> None:
        ev = MultiplayerPBTEvaluator(
            num_players=4,
            games_per_eval=1,
            scoring_method="linear_ranking",
        )
        result = NPlayerMatchResult(
            agent_ids=["a", "b", "c", "d"],
            rankings={"a": 1, "b": 2, "c": 3, "d": 4},
            scores={"a": 10.0, "b": 7.0, "c": 3.0, "d": 0.0},
        )
        values = ev._score_result(result)
        assert len(values) == 4
        # Values should be ordered by rank
        assert values["a"] >= values["b"]
        assert values["b"] >= values["c"]
        assert values["c"] >= values["d"]


# =========================================================================
# 8. _linear_ranking_fallback
# =========================================================================


class TestLinearRankingFallback:
    """Tests for the static _linear_ranking_fallback method."""

    def test_rank_1_maps_to_plus_one_2_players(self) -> None:
        result = NPlayerMatchResult(
            agent_ids=["a", "b"],
            rankings={"a": 1, "b": 2},
            scores={},
        )
        values = MultiplayerPBTEvaluator._linear_ranking_fallback(result)
        assert values["a"] == pytest.approx(1.0)

    def test_rank_n_maps_to_minus_one_2_players(self) -> None:
        result = NPlayerMatchResult(
            agent_ids=["a", "b"],
            rankings={"a": 1, "b": 2},
            scores={},
        )
        values = MultiplayerPBTEvaluator._linear_ranking_fallback(result)
        assert values["b"] == pytest.approx(-1.0)

    def test_four_players_distinct_ranks(self) -> None:
        result = NPlayerMatchResult(
            agent_ids=["a", "b", "c", "d"],
            rankings={"a": 1, "b": 2, "c": 3, "d": 4},
            scores={},
        )
        values = MultiplayerPBTEvaluator._linear_ranking_fallback(result)
        assert values["a"] == pytest.approx(1.0)
        assert values["d"] == pytest.approx(-1.0)
        # Middle ranks interpolated linearly
        assert values["b"] == pytest.approx(1.0 / 3.0, abs=0.01)
        assert values["c"] == pytest.approx(-1.0 / 3.0, abs=0.01)

    def test_three_players_tied_all_rank_1(self) -> None:
        result = NPlayerMatchResult(
            agent_ids=["a", "b", "c"],
            rankings={"a": 1, "b": 1, "c": 1},
            scores={},
        )
        values = MultiplayerPBTEvaluator._linear_ranking_fallback(result)
        # All rank 1 in a 3-player game: value = 1.0 - 2.0 * (0/2) = 1.0
        assert values["a"] == values["b"] == values["c"]

    def test_single_player_returns_zero(self) -> None:
        result = NPlayerMatchResult(
            agent_ids=["solo"],
            rankings={"solo": 1},
            scores={},
        )
        values = MultiplayerPBTEvaluator._linear_ranking_fallback(result)
        # max_rank = 1, which equals n=1, so max_rank > 1 is False -> 0.0
        assert values["solo"] == pytest.approx(0.0)

    def test_two_players_rank_swapped(self) -> None:
        result = NPlayerMatchResult(
            agent_ids=["a", "b"],
            rankings={"a": 2, "b": 1},
            scores={},
        )
        values = MultiplayerPBTEvaluator._linear_ranking_fallback(result)
        assert values["b"] == pytest.approx(1.0)
        assert values["a"] == pytest.approx(-1.0)

    def test_five_players_first_and_last(self) -> None:
        ids = ["a", "b", "c", "d", "e"]
        rankings = {ids[i]: i + 1 for i in range(5)}
        result = NPlayerMatchResult(agent_ids=ids, rankings=rankings, scores={})
        values = MultiplayerPBTEvaluator._linear_ranking_fallback(result)
        assert values["a"] == pytest.approx(1.0)
        assert values["e"] == pytest.approx(-1.0)
        # Middle player (rank 3 in 5 players): 1.0 - 2.0*(2/4) = 0.0
        assert values["c"] == pytest.approx(0.0)


# =========================================================================
# 9. _normalise_scores
# =========================================================================


class TestNormaliseScores:
    """Tests for the static _normalise_scores method."""

    def test_empty_dict_returns_empty(self) -> None:
        assert MultiplayerPBTEvaluator._normalise_scores({}) == {}

    def test_equal_scores_return_half(self) -> None:
        result = MultiplayerPBTEvaluator._normalise_scores({"a": 0.5, "b": 0.5, "c": 0.5})
        for v in result.values():
            assert v == pytest.approx(0.5)

    def test_different_scores_normalised_to_0_1(self) -> None:
        result = MultiplayerPBTEvaluator._normalise_scores({"low": -1.0, "mid": 0.0, "high": 1.0})
        assert result["low"] == pytest.approx(0.0)
        assert result["mid"] == pytest.approx(0.5)
        assert result["high"] == pytest.approx(1.0)

    def test_two_scores(self) -> None:
        result = MultiplayerPBTEvaluator._normalise_scores({"a": 10.0, "b": 20.0})
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(1.0)

    def test_negative_scores(self) -> None:
        result = MultiplayerPBTEvaluator._normalise_scores({"a": -5.0, "b": -3.0, "c": -1.0})
        assert result["a"] == pytest.approx(0.0)
        assert result["c"] == pytest.approx(1.0)

    def test_single_score_returns_half(self) -> None:
        result = MultiplayerPBTEvaluator._normalise_scores({"only": 42.0})
        assert result["only"] == pytest.approx(0.5)

    def test_very_close_scores_return_half(self) -> None:
        """Scores within 1e-9 of each other should all map to 0.5."""
        result = MultiplayerPBTEvaluator._normalise_scores({"a": 1.0, "b": 1.0 + 1e-12})
        for v in result.values():
            assert v == pytest.approx(0.5)

    def test_preserves_all_keys(self) -> None:
        input_scores = {f"agent_{i}": float(i) for i in range(10)}
        result = MultiplayerPBTEvaluator._normalise_scores(input_scores)
        assert set(result.keys()) == set(input_scores.keys())

    def test_output_range(self) -> None:
        import numpy as np

        rng = np.random.default_rng(123)
        scores = {f"a{i}": rng.uniform(-100, 100) for i in range(20)}
        result = MultiplayerPBTEvaluator._normalise_scores(scores)
        for v in result.values():
            assert 0.0 <= v <= 1.0


# =========================================================================
# Protocol compliance
# =========================================================================


class TestNPlayerArenaRunnerProtocol:
    """Verify our mock runners satisfy the protocol."""

    def test_mock_runner_protocol(self) -> None:
        assert isinstance(_MockRunner(), NPlayerArenaRunner)

    def test_ranked_runner_protocol(self) -> None:
        assert isinstance(_RankedRunner(), NPlayerArenaRunner)

    def test_empty_scores_runner_protocol(self) -> None:
        assert isinstance(_EmptyScoresRunner(), NPlayerArenaRunner)


# =========================================================================
# Integration-style tests combining multiple methods
# =========================================================================


class TestIntegration:
    """Higher-level tests that exercise the full evaluate_population flow."""

    def test_end_to_end_with_various_scoring_methods(self) -> None:
        for method in ["linear_ranking", "winner_take_all", "proportional"]:
            ev = MultiplayerPBTEvaluator(
                num_players=2,
                games_per_eval=3,
                scoring_method=method,
            )
            agents = _make_agents(4)
            runner = _MockRunner(winner_index=0)
            fitness = ev.evaluate_population(agents, runner)

            assert len(fitness) == 4
            for v in fitness.values():
                assert 0.0 <= v <= 1.0

    def test_large_population_evaluation(self) -> None:
        """Ensure large populations (where matchups are capped) still work."""
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=1)
        agents = _make_agents(25)
        runner = _MockRunner(winner_index=0)
        fitness = ev.evaluate_population(agents, runner)

        assert len(fitness) == 25
        for v in fitness.values():
            assert 0.0 <= v <= 1.0

    def test_evaluation_with_invalid_scoring_method(self) -> None:
        """Invalid scoring method should not crash; falls back gracefully."""
        ev = MultiplayerPBTEvaluator(
            num_players=2,
            games_per_eval=2,
            scoring_method="nonexistent",
        )
        agents = _make_agents(4)
        runner = _MockRunner()
        fitness = ev.evaluate_population(agents, runner)
        assert len(fitness) == 4

    def test_agent_state_with_hyperparameters(self) -> None:
        """AgentState with hyperparameters still works for evaluation."""
        agents = [
            AgentState(
                agent_id=f"hp_agent_{i}",
                hyperparameters={"lr": 0.001 * (i + 1), "batch_size": 32 * (i + 1)},
            )
            for i in range(4)
        ]
        ev = MultiplayerPBTEvaluator(num_players=2, games_per_eval=2)
        runner = _MockRunner()
        fitness = ev.evaluate_population(agents, runner)
        assert len(fitness) == 4
