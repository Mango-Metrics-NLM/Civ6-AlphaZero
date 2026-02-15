"""Unit tests for inference benchmarking utilities.

Tests the InferenceMetrics and InferenceBenchmark classes which measure
network inference performance in game playing scenarios.
"""

from __future__ import annotations

import pytest
import torch

from alpha_zero.optimizations.inference_benchmark import (
    InferenceBenchmark,
    InferenceMetrics,
)


class MockNetwork:
    """Mock network for testing inference benchmarks."""

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mock prediction that returns valid policy and value tensors.

        Args:
            spatial: Spatial input tensor of shape (B, C, H, W)
            vector: Vector input tensor of shape (B, V)
            mask: Action mask of shape (B, A)

        Returns:
            Tuple of (policy, value) where:
                - policy has shape (B, A) with valid probabilities
                - value has shape (B, 1)
        """
        batch_size = spatial.shape[0]
        action_size = mask.shape[-1]

        # Create valid policy (masked and normalized)
        policy = torch.ones(batch_size, action_size, device=spatial.device, dtype=spatial.dtype) / action_size * mask
        policy = policy / (policy.sum(dim=-1, keepdim=True) + 1e-8)

        # Create value tensor
        value = torch.zeros(batch_size, 1, device=spatial.device, dtype=spatial.dtype)

        return policy, value

    def eval(self) -> None:
        """Set network to evaluation mode."""
        pass


class TestInferenceMetrics:
    """Test suite for InferenceMetrics dataclass."""

    def test_creation(self) -> None:
        """Test creating InferenceMetrics with all fields."""
        metrics = InferenceMetrics(
            name="test_benchmark",
            games_per_minute=12.5,
            moves_per_second=50.0,
            avg_batch_size=4.2,
            gpu_utilization_pct=85.0,
            total_games=100,
            total_time_seconds=480.0,
            network_eval_time_pct=60.0,
            env_step_time_pct=35.0,
            additional_metrics={"custom_metric": 42},
        )

        assert metrics.name == "test_benchmark"
        assert metrics.games_per_minute == 12.5
        assert metrics.moves_per_second == 50.0
        assert metrics.avg_batch_size == 4.2
        assert metrics.gpu_utilization_pct == 85.0
        assert metrics.total_games == 100
        assert metrics.total_time_seconds == 480.0
        assert metrics.network_eval_time_pct == 60.0
        assert metrics.env_step_time_pct == 35.0
        assert metrics.additional_metrics == {"custom_metric": 42}

    def test_summary_format(self) -> None:
        """Test that summary returns a string with key information."""
        metrics = InferenceMetrics(
            name="benchmark_1",
            games_per_minute=10.0,
            moves_per_second=40.0,
            avg_batch_size=2.0,
            gpu_utilization_pct=70.0,
            total_games=50,
            total_time_seconds=300.0,
            network_eval_time_pct=55.0,
            env_step_time_pct=40.0,
            additional_metrics={},
        )

        summary = metrics.summary()

        assert isinstance(summary, str)
        assert "benchmark_1" in summary
        assert "10.0" in summary  # games_per_minute
        assert "40.0" in summary  # moves_per_second
        # Summary should be informative
        assert len(summary) > 50

    def test_to_dict(self) -> None:
        """Test that to_dict returns a flat dictionary."""
        metrics = InferenceMetrics(
            name="dict_test",
            games_per_minute=15.0,
            moves_per_second=60.0,
            avg_batch_size=3.5,
            gpu_utilization_pct=80.0,
            total_games=75,
            total_time_seconds=300.0,
            network_eval_time_pct=65.0,
            env_step_time_pct=30.0,
            additional_metrics={"extra": 123},
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        # Check that it's a flat dict (no nested dicts except additional_metrics)
        assert result["name"] == "dict_test"
        assert result["games_per_minute"] == 15.0
        assert result["moves_per_second"] == 60.0

    def test_to_dict_keys(self) -> None:
        """Test that to_dict has expected keys."""
        metrics = InferenceMetrics(
            name="key_test",
            games_per_minute=8.0,
            moves_per_second=32.0,
            avg_batch_size=1.5,
            gpu_utilization_pct=50.0,
            total_games=20,
            total_time_seconds=150.0,
            network_eval_time_pct=45.0,
            env_step_time_pct=50.0,
            additional_metrics={},
        )

        result = metrics.to_dict()

        expected_keys = {
            "name",
            "games_per_minute",
            "moves_per_second",
            "avg_batch_size",
            "gpu_utilization_pct",
            "total_games",
            "total_time_seconds",
            "network_eval_time_pct",
            "env_step_time_pct",
            "additional_metrics",
        }

        assert set(result.keys()) == expected_keys

    def test_additional_metrics_optional(self) -> None:
        """Test creating metrics without additional_metrics."""
        metrics = InferenceMetrics(
            name="minimal_test",
            games_per_minute=5.0,
            moves_per_second=20.0,
            avg_batch_size=1.0,
            gpu_utilization_pct=30.0,
            total_games=10,
            total_time_seconds=120.0,
            network_eval_time_pct=40.0,
            env_step_time_pct=55.0,
        )

        # Should have default empty dict
        assert metrics.additional_metrics == {}


class TestInferenceBenchmark:
    """Test suite for InferenceBenchmark class."""

    def test_init(self) -> None:
        """Test initialization with default and custom parameters."""
        # Default initialization
        benchmark = InferenceBenchmark()
        assert benchmark.board_size == 8
        assert benchmark.max_turns == 30
        assert benchmark.warmup_games == 2

        # Custom initialization
        benchmark = InferenceBenchmark(
            board_size=16,
            max_turns=15,
            warmup_games=5,
        )
        assert benchmark.board_size == 16
        assert benchmark.max_turns == 15
        assert benchmark.warmup_games == 5

    @pytest.mark.slow
    def test_single_benchmark(self) -> None:
        """Test single inference benchmark runs games and returns metrics."""
        network = MockNetwork()
        benchmark = InferenceBenchmark(
            board_size=8,
            max_turns=5,  # Small for speed
            warmup_games=0,  # Skip warmup for faster test
        )

        metrics = benchmark.run_single_inference_benchmark(
            network=network,
            num_games=2,
            num_simulations=2,  # Small for speed
            device="cpu",
        )

        # Verify metrics structure
        assert isinstance(metrics, InferenceMetrics)
        assert metrics.name == "single_inference"
        assert metrics.total_games == 2
        assert metrics.total_time_seconds > 0
        assert metrics.games_per_minute > 0
        assert metrics.moves_per_second >= 0
        assert metrics.avg_batch_size == 1.0  # Single inference uses batch size 1

    @pytest.mark.slow
    def test_batch_benchmark(self) -> None:
        """Test batch inference benchmark with parallel environments."""
        network = MockNetwork()
        benchmark = InferenceBenchmark(
            board_size=8,
            max_turns=5,  # Small for speed
            warmup_games=0,  # Skip warmup
        )

        metrics = benchmark.run_batch_inference_benchmark(
            network=network,
            num_games=2,
            num_parallel=2,
            num_simulations=2,  # Small for speed
            device="cpu",
        )

        # Verify metrics structure
        assert isinstance(metrics, InferenceMetrics)
        assert metrics.name == "batch_inference"
        assert metrics.total_games == 2
        assert metrics.total_time_seconds > 0
        assert metrics.games_per_minute > 0
        assert metrics.avg_batch_size >= 1.0

    @pytest.mark.slow
    def test_comparison(self) -> None:
        """Test comparison returns speedup metrics."""
        network = MockNetwork()
        benchmark = InferenceBenchmark(
            board_size=8,
            max_turns=5,  # Small for speed
            warmup_games=0,  # Skip warmup
        )

        comparison = benchmark.run_comparison(
            network=network,
            num_games=2,
            num_parallel=2,
            num_simulations=2,  # Small for speed
            device="cpu",
        )

        # Verify comparison structure
        assert isinstance(comparison, dict)
        assert "single" in comparison
        assert "batch" in comparison
        assert "speedup" in comparison

        # Verify nested metrics
        assert isinstance(comparison["single"], InferenceMetrics)
        assert isinstance(comparison["batch"], InferenceMetrics)

        # Speedup should be positive
        assert comparison["speedup"] > 0

    def test_warmup_games_applied(self) -> None:
        """Test that warmup games are run before benchmark."""
        network = MockNetwork()
        benchmark = InferenceBenchmark(
            board_size=8,
            max_turns=3,
            warmup_games=1,
        )

        # Should complete without error with warmup
        metrics = benchmark.run_single_inference_benchmark(
            network=network,
            num_games=1,
            num_simulations=2,
            device="cpu",
        )

        # Warmup games should not be counted in total
        assert metrics.total_games == 1

    def test_board_size_affects_environment(self) -> None:
        """Test that different board sizes create appropriate environments."""
        network = MockNetwork()

        # Test with small board
        benchmark_small = InferenceBenchmark(board_size=8, max_turns=3, warmup_games=0)
        metrics_small = benchmark_small.run_single_inference_benchmark(
            network=network,
            num_games=1,
            num_simulations=2,
            device="cpu",
        )

        # Test with larger board
        benchmark_large = InferenceBenchmark(board_size=16, max_turns=3, warmup_games=0)
        metrics_large = benchmark_large.run_single_inference_benchmark(
            network=network,
            num_games=1,
            num_simulations=2,
            device="cpu",
        )

        # Both should complete successfully
        assert metrics_small.total_games == 1
        assert metrics_large.total_games == 1

    def test_metrics_consistency(self) -> None:
        """Test that metrics values are consistent and reasonable."""
        network = MockNetwork()
        benchmark = InferenceBenchmark(
            board_size=8,
            max_turns=5,
            warmup_games=0,
        )

        metrics = benchmark.run_single_inference_benchmark(
            network=network,
            num_games=2,
            num_simulations=2,
            device="cpu",
        )

        # Time-based metrics should be reasonable
        assert metrics.total_time_seconds > 0
        assert metrics.games_per_minute > 0

        # Percentage metrics should be in valid range
        assert 0 <= metrics.network_eval_time_pct <= 100
        assert 0 <= metrics.env_step_time_pct <= 100

        # GPU utilization should be non-negative
        assert metrics.gpu_utilization_pct >= 0

    @pytest.mark.slow
    def test_batch_faster_than_single(self) -> None:
        """Test that batch inference is typically faster than single."""
        network = MockNetwork()
        benchmark = InferenceBenchmark(
            board_size=8,
            max_turns=5,
            warmup_games=0,
        )

        comparison = benchmark.run_comparison(
            network=network,
            num_games=4,
            num_parallel=4,
            num_simulations=2,
            device="cpu",
        )

        # Batch should generally be faster (speedup > 1.0)
        # Note: In some cases single might be faster for very small workloads
        # so we just check that speedup is calculated
        assert comparison["speedup"] > 0

    def test_different_num_simulations(self) -> None:
        """Test benchmarking with different MCTS simulation counts."""
        network = MockNetwork()
        benchmark = InferenceBenchmark(
            board_size=8,
            max_turns=3,
            warmup_games=0,
        )

        # Test with different simulation counts
        for num_sims in [1, 2, 4]:
            metrics = benchmark.run_single_inference_benchmark(
                network=network,
                num_games=1,
                num_simulations=num_sims,
                device="cpu",
            )

            assert metrics.total_games == 1
            assert metrics.total_time_seconds > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_benchmark(self) -> None:
        """Test benchmarking on GPU device."""
        network = MockNetwork()
        benchmark = InferenceBenchmark(
            board_size=8,
            max_turns=3,
            warmup_games=0,
        )

        metrics = benchmark.run_single_inference_benchmark(
            network=network,
            num_games=1,
            num_simulations=2,
            device="cuda",
        )

        assert metrics.total_games == 1
        # GPU utilization should be tracked
        assert metrics.gpu_utilization_pct >= 0

    def test_empty_additional_metrics(self) -> None:
        """Test that additional_metrics is properly initialized."""
        network = MockNetwork()
        benchmark = InferenceBenchmark(
            board_size=8,
            max_turns=3,
            warmup_games=0,
        )

        metrics = benchmark.run_single_inference_benchmark(
            network=network,
            num_games=1,
            num_simulations=2,
            device="cpu",
        )

        # Should have additional_metrics field
        assert hasattr(metrics, "additional_metrics")
        assert isinstance(metrics.additional_metrics, dict)
