"""peak_gpu_memory_mb field on InferenceMetrics."""
from __future__ import annotations

from alpha_zero.optimizations.inference_benchmark import InferenceMetrics


def _make_metrics(**overrides):
    """Build an InferenceMetrics with sensible defaults for all required fields."""
    kwargs = {
        "name": "test",
        "games_per_minute": 1.0,
        "moves_per_second": 1.0,
        "avg_batch_size": 1.0,
        "gpu_utilization_pct": 0.0,
        "total_games": 1,
        "total_time_seconds": 1.0,
        "network_eval_time_pct": 50.0,
        "env_step_time_pct": 50.0,
    }
    kwargs.update(overrides)
    return InferenceMetrics(**kwargs)


def test_metrics_has_peak_gpu_memory_field():
    m = _make_metrics()
    assert hasattr(m, "peak_gpu_memory_mb")
    assert m.peak_gpu_memory_mb is None or isinstance(m.peak_gpu_memory_mb, float)


def test_metrics_default_peak_gpu_memory_is_none():
    m = _make_metrics()
    assert m.peak_gpu_memory_mb is None


def test_metrics_accepts_explicit_peak_gpu_memory():
    m = _make_metrics(peak_gpu_memory_mb=512.0)
    assert m.peak_gpu_memory_mb == 512.0
