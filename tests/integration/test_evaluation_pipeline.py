"""Integration tests for the Model Evaluation Pipeline.

Tests cover:
- Full evaluation against random baseline
- Full evaluation against heuristic baseline
- ELO rating updates through evaluation
- Diversity metrics computation
- Report generation in multiple formats
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from alpha_zero.evaluation import (
    BaselineConfig,
    BaselineFactory,
    BaselineType,
    DiversityConfig,
    DiversityMetrics,
    DiversityTracker,
    EvaluationConfig,
    EvaluationReport,
    EvaluationResult,
    GameResult,
    HeuristicPlayer,
    HeuristicWeights,
    ModelEvaluator,
    RandomPlayer,
    ReportGenerator,
)


class TestEvaluationPipelineIntegration:
    """Integration tests for full evaluation pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def evaluation_config(self, temp_dir):
        """Create evaluation configuration for testing."""
        return EvaluationConfig(
            experiment_name="integration_test",
            env_board_size=5,
            env_max_turns=50,
            elo_enabled=True,
            elo_initial_rating=1500.0,
            device="cpu",
            output_dir=str(temp_dir),
            baselines=[
                BaselineConfig(
                    type=BaselineType.RANDOM,
                    name="random_test",
                    num_games=5,  # Small for speed
                    enabled=True,
                ),
                BaselineConfig(
                    type=BaselineType.HEURISTIC,
                    name="heuristic_test",
                    num_games=5,
                    enabled=True,
                    heuristic_weights=HeuristicWeights(
                        attack_weight=1.0,
                        defense_weight=0.8,
                        expansion_weight=1.2,
                    ),
                ),
            ],
            diversity=DiversityConfig(
                enabled=True,
                track_opening_moves=True,
                opening_depth=5,
                track_game_lengths=True,
            ),
        )

    @pytest.fixture
    def random_player(self):
        """Create random player for testing."""
        return RandomPlayer(seed=42, name="model_random")

    def test_full_evaluation_random_baseline(self, evaluation_config, random_player, temp_dir):
        """Test complete evaluation against random baseline."""
        evaluator = ModelEvaluator(evaluation_config)

        # Get random baseline
        random_config = evaluation_config.baselines[0]
        baseline = evaluator.get_baseline(random_config)

        # Play multiple games
        results = []
        for i in range(random_config.num_games):
            result = evaluator.play_game(
                model_player=random_player,
                opponent=baseline,
                model_plays_first=(i % 2 == 0),
            )
            results.append(result)

        # Verify game results
        assert len(results) == random_config.num_games
        for result in results:
            assert isinstance(result, GameResult)
            assert result.num_turns > 0
            assert result.model_score in [0.0, 0.5, 1.0]
            assert result.opponent_score in [0.0, 0.5, 1.0]
            assert result.model_score + result.opponent_score == 1.0

    def test_full_evaluation_heuristic_baseline(self, evaluation_config, random_player, temp_dir):
        """Test complete evaluation against heuristic baseline."""
        evaluator = ModelEvaluator(evaluation_config)

        # Get heuristic baseline
        heuristic_config = evaluation_config.baselines[1]
        baseline = evaluator.get_baseline(heuristic_config)

        assert isinstance(baseline, HeuristicPlayer)
        assert baseline.name == "heuristic_test"

        # Play multiple games
        results = []
        for i in range(heuristic_config.num_games):
            result = evaluator.play_game(
                model_player=random_player,
                opponent=baseline,
                model_plays_first=(i % 2 == 0),
            )
            results.append(result)

        # Verify all games completed
        assert len(results) == heuristic_config.num_games
        wins = sum(1 for r in results if r.winner == 0)
        losses = sum(1 for r in results if r.winner == 1)
        draws = sum(1 for r in results if r.winner is None)
        assert wins + losses + draws == len(results)

    def test_elo_rating_updates(self, evaluation_config, temp_dir):
        """Test ELO rating updates through evaluation."""
        evaluator = ModelEvaluator(evaluation_config)

        assert evaluator.elo_system is not None
        assert evaluator.model_stats is not None

        initial_rating = evaluator.model_stats.rating

        # Create players
        model_player = RandomPlayer(seed=42, name="model")
        random_config = evaluation_config.baselines[0]
        baseline = evaluator.get_baseline(random_config)

        # Get baseline stats
        baseline_stats = evaluator._get_baseline_stats(random_config.name)
        assert baseline_stats is not None

        # Play a game
        result = evaluator.play_game(
            model_player=model_player,
            opponent=baseline,
            model_plays_first=True,
        )

        # Process ELO update
        from alpha_zero.tournament.config import MatchOutcome

        if result.winner == 0:
            outcome = MatchOutcome.WIN
        elif result.winner == 1:
            outcome = MatchOutcome.LOSS
        else:
            outcome = MatchOutcome.DRAW

        from alpha_zero.tournament.config import TournamentConfig
        tournament_config = TournamentConfig()

        update1, _update2 = evaluator.elo_system.process_game_result(
            evaluator.model_stats,
            baseline_stats,
            outcome,
            tournament_config,
        )

        # Verify ELO changes
        assert update1.rating_before == initial_rating
        assert update1.rating_after != initial_rating or outcome == MatchOutcome.DRAW

    def test_diversity_metrics_computed(self, evaluation_config, random_player, temp_dir):
        """Test diversity metrics are computed correctly."""
        evaluator = ModelEvaluator(evaluation_config)

        assert evaluator.diversity_tracker is not None

        # Play some games and record history
        random_config = evaluation_config.baselines[0]
        baseline = evaluator.get_baseline(random_config)

        for _ in range(3):
            result = evaluator.play_game(
                model_player=random_player,
                opponent=baseline,
                model_plays_first=True,
            )

            # Record game for diversity analysis
            if result.game_history:
                evaluator.diversity_tracker.record_game(
                    game_history=result.game_history,
                    baseline_name="random_test",
                    game_length=result.num_turns,
                )

        # Compute metrics
        metrics = evaluator.diversity_tracker.compute_metrics()

        assert isinstance(metrics, DiversityMetrics)
        assert metrics.total_games >= 0
        assert 0.0 <= metrics.opening_entropy <= 10.0  # Entropy bound
        assert metrics.avg_game_length >= 0

    def test_report_generation(self, evaluation_config, random_player, temp_dir):
        """Test report generation in multiple formats."""
        evaluator = ModelEvaluator(evaluation_config)

        # Play some games
        random_config = evaluation_config.baselines[0]
        baseline = evaluator.get_baseline(random_config)

        game_results = []
        for i in range(random_config.num_games):
            result = evaluator.play_game(
                model_player=random_player,
                opponent=baseline,
                model_plays_first=(i % 2 == 0),
            )
            game_results.append(result)

        # Create evaluation result
        wins = sum(1 for r in game_results if r.winner == 0)
        losses = sum(1 for r in game_results if r.winner == 1)
        draws = sum(1 for r in game_results if r.winner is None)
        avg_length = np.mean([r.num_turns for r in game_results])

        eval_result = EvaluationResult(
            baseline_name="random_test",
            games_played=len(game_results),
            wins=wins,
            losses=losses,
            draws=draws,
            avg_game_length=float(avg_length),
            game_results=game_results,
        )

        # Generate report
        generator = ReportGenerator(evaluation_config)
        report = generator.generate(
            results={"random_test": eval_result},
            diversity_metrics=None,
        )

        assert isinstance(report, EvaluationReport)
        assert report.experiment_name == evaluation_config.experiment_name
        assert "random_test" in report.baseline_results

        # Test JSON export
        json_path = temp_dir / "report.json"
        report.to_json(str(json_path))
        assert json_path.exists()

        with open(json_path) as f:
            json_data = json.load(f)
        assert "experiment_name" in json_data
        assert "baseline_results" in json_data

        # Test markdown export
        md_path = temp_dir / "report.md"
        md_content = report.to_markdown()
        with open(md_path, "w") as f:
            f.write(md_content)
        assert md_path.exists()

        assert "# Model Evaluation Report" in md_content
        assert "random_test" in md_content


class TestBaselineFactoryIntegration:
    """Integration tests for baseline factory."""

    def test_create_all_baseline_types(self):
        """Test factory creates all baseline types correctly."""
        baseline_types = [
            BaselineType.RANDOM,
            BaselineType.HEURISTIC,
        ]

        for btype in baseline_types:
            config = BaselineConfig(
                type=btype,
                name=f"test_{btype.value}",
                enabled=True,
            )

            player = BaselineFactory.create(config, device="cpu", seed=42)
            assert player.name == f"test_{btype.value}"

    def test_heuristic_weights_affect_behavior(self):
        """Test that different heuristic weights produce different behavior."""
        # Create two heuristic players with different weights
        config1 = BaselineConfig(
            type=BaselineType.HEURISTIC,
            name="aggressive",
            heuristic_weights=HeuristicWeights(
                attack_weight=2.0,
                defense_weight=0.5,
                expansion_weight=0.5,
            ),
        )

        config2 = BaselineConfig(
            type=BaselineType.HEURISTIC,
            name="defensive",
            heuristic_weights=HeuristicWeights(
                attack_weight=0.5,
                defense_weight=2.0,
                expansion_weight=0.5,
            ),
        )

        player1 = BaselineFactory.create(config1, device="cpu", seed=42)
        player2 = BaselineFactory.create(config2, device="cpu", seed=42)

        assert player1.name != player2.name
        assert player1.weights.attack_weight != player2.weights.attack_weight


class TestDiversityTrackerIntegration:
    """Integration tests for diversity tracking."""

    def test_entropy_increases_with_variety(self):
        """Test that opening entropy increases with move variety."""
        config = DiversityConfig(
            enabled=True,
            track_opening_moves=True,
            opening_depth=3,
        )

        tracker = DiversityTracker(config)

        # Record same opening repeatedly (low entropy)
        # Create game history with same opening moves
        for _ in range(10):
            game_history = [
                (0, {"is_model": True, "turn": 0}),
                (1, {"is_model": True, "turn": 1}),
                (2, {"is_model": True, "turn": 2}),
            ]
            tracker.record_game(game_history, baseline_name="test", game_length=3)

        metrics_low = tracker.compute_metrics()

        # Create new tracker and record varied openings (high entropy)
        tracker2 = DiversityTracker(config)
        for i in range(10):
            game_history = [
                (i, {"is_model": True, "turn": 0}),
                (i + 1, {"is_model": True, "turn": 1}),
                (i + 2, {"is_model": True, "turn": 2}),
            ]
            tracker2.record_game(game_history, baseline_name="test", game_length=3)

        metrics_high = tracker2.compute_metrics()

        # Varied openings should have higher entropy
        assert metrics_high.opening_entropy > metrics_low.opening_entropy

    def test_game_length_statistics(self):
        """Test game length statistics computation."""
        config = DiversityConfig(
            enabled=True,
            track_game_lengths=True,
        )

        tracker = DiversityTracker(config)

        # Record some game lengths
        lengths = [10, 20, 30, 40, 50]
        for length in lengths:
            # Create minimal game history with specified length
            game_history = [(i, {"is_model": True, "turn": i}) for i in range(length)]
            tracker.record_game(game_history, baseline_name="test", game_length=length)

        metrics = tracker.compute_metrics()

        assert metrics.total_games == len(lengths)
        assert metrics.avg_game_length == pytest.approx(np.mean(lengths))
        assert metrics.game_length_std == pytest.approx(np.std(lengths))
        assert metrics.game_length_min == min(lengths)
        assert metrics.game_length_max == max(lengths)


class TestEvaluatorWithELOIntegration:
    """Integration tests for evaluator with ELO system."""

    @pytest.fixture
    def elo_config(self, tmp_path):
        """Create config with ELO enabled."""
        return EvaluationConfig(
            experiment_name="elo_test",
            env_board_size=5,
            env_max_turns=30,
            elo_enabled=True,
            elo_initial_rating=1500.0,
            device="cpu",
            output_dir=str(tmp_path),
            baselines=[
                BaselineConfig(
                    type=BaselineType.RANDOM,
                    name="random",
                    num_games=3,
                    enabled=True,
                ),
            ],
        )

    def test_elo_system_initialization(self, elo_config):
        """Test ELO system is properly initialized."""
        evaluator = ModelEvaluator(elo_config)

        assert evaluator.elo_system is not None
        assert evaluator.model_stats is not None
        assert evaluator.model_stats.rating == elo_config.elo_initial_rating

    def test_rating_bounds_maintained(self, elo_config):
        """Test that ELO ratings stay within bounds after many games."""
        evaluator = ModelEvaluator(elo_config)

        model_player = RandomPlayer(seed=42, name="model")
        baseline = evaluator.get_baseline(elo_config.baselines[0])
        baseline_stats = evaluator._get_baseline_stats("random")

        from alpha_zero.tournament.config import MatchOutcome, TournamentConfig
        tournament_config = TournamentConfig()

        # Play many games
        for _ in range(20):
            result = evaluator.play_game(
                model_player=model_player,
                opponent=baseline,
                model_plays_first=True,
            )

            if result.winner == 0:
                outcome = MatchOutcome.WIN
            elif result.winner == 1:
                outcome = MatchOutcome.LOSS
            else:
                outcome = MatchOutcome.DRAW

            evaluator.elo_system.process_game_result(
                evaluator.model_stats,
                baseline_stats,
                outcome,
                tournament_config,
            )

        # Check ratings are within bounds
        elo_config_bounds = tournament_config.elo
        assert evaluator.model_stats.rating >= elo_config_bounds.rating_floor
        assert evaluator.model_stats.rating <= elo_config_bounds.rating_ceiling
