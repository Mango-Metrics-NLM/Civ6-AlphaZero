"""Unit tests for worker_autoscaler module."""
from __future__ import annotations

import os

import pytest

from alpha_zero.cloud.gcp.worker_autoscaler import (
    ScalingAction,
    ScalingDecision,
    WorkerAutoscaler,
    WorkerAutoscalerConfig,
    WorkerMetrics,
    get_worker_autoscaler,
)


class TestScalingAction:
    def test_scaling_action_values(self) -> None:
        assert ScalingAction.SCALE_UP.value == "scale_up"
        assert ScalingAction.SCALE_DOWN.value == "scale_down"
        assert ScalingAction.MAINTAIN.value == "maintain"


class TestWorkerMetrics:
    def test_creation(self) -> None:
        metrics = WorkerMetrics(
            worker_id=1,
            games_completed=100,
            avg_game_duration_seconds=300.0,
            preempted=False,
        )
        assert metrics.worker_id == 1
        assert metrics.games_completed == 100
        assert metrics.avg_game_duration_seconds == 300.0
        assert metrics.preempted is False

    def test_default_values(self) -> None:
        metrics = WorkerMetrics(worker_id=0)
        assert metrics.worker_id == 0
        assert metrics.games_completed == 0
        assert metrics.avg_game_duration_seconds == 0.0
        assert metrics.preempted is False
        assert metrics.last_update > 0

    def test_to_dict(self) -> None:
        metrics = WorkerMetrics(
            worker_id=5,
            games_completed=50,
            avg_game_duration_seconds=250.0,
            preempted=True,
        )
        data = metrics.to_dict()
        assert data["worker_id"] == 5
        assert data["games_completed"] == 50
        assert data["avg_game_duration_seconds"] == 250.0
        assert data["preempted"] is True
        assert "last_update" in data


class TestScalingDecision:
    def test_creation(self) -> None:
        decision = ScalingDecision(
            target_workers=15,
            current_workers=10,
            action=ScalingAction.SCALE_UP,
            reason="High demand",
            confidence=0.85,
        )
        assert decision.target_workers == 15
        assert decision.current_workers == 10
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.reason == "High demand"
        assert decision.confidence == 0.85

    def test_default_values(self) -> None:
        decision = ScalingDecision(
            target_workers=10,
            current_workers=10,
            action=ScalingAction.MAINTAIN,
            reason="Optimal",
        )
        assert decision.confidence == 1.0
        assert decision.factors == {}

    def test_to_dict(self) -> None:
        decision = ScalingDecision(
            target_workers=10,
            current_workers=10,
            action=ScalingAction.MAINTAIN,
            reason="Optimal",
            confidence=0.9,
        )
        data = decision.to_dict()
        assert data["action"] == "maintain"
        assert data["target_workers"] == 10
        assert data["current_workers"] == 10
        assert data["reason"] == "Optimal"
        assert data["confidence"] == 0.9


class TestWorkerAutoscalerConfig:
    def test_default_values(self) -> None:
        # Clear env vars that might interfere
        env_vars = [
            "CIV6AZ_AUTOSCALE_ENABLED",
            "CIV6AZ_AUTOSCALE_BASE_WORKERS",
            "CIV6AZ_AUTOSCALE_MIN_WORKERS",
            "CIV6AZ_AUTOSCALE_MAX_WORKERS",
        ]
        old_values = {}
        for var in env_vars:
            if var in os.environ:
                old_values[var] = os.environ.pop(var)

        try:
            config = WorkerAutoscalerConfig()
            assert config.enabled is True
            assert config.base_workers == 10
            assert config.min_workers == 2
            assert config.max_workers == 50
        finally:
            for var, val in old_values.items():
                os.environ[var] = val

    def test_from_env(self) -> None:
        os.environ["CIV6AZ_AUTOSCALE_ENABLED"] = "true"
        os.environ["CIV6AZ_AUTOSCALE_BASE_WORKERS"] = "8"
        try:
            config = WorkerAutoscalerConfig.from_env()
            assert config.enabled is True
            assert config.base_workers == 8
        finally:
            del os.environ["CIV6AZ_AUTOSCALE_ENABLED"]
            del os.environ["CIV6AZ_AUTOSCALE_BASE_WORKERS"]

    def test_validation_min_less_than_max(self) -> None:
        config = WorkerAutoscalerConfig(min_workers=5, max_workers=20)
        config.validate()  # Should not raise

    def test_validation_invalid(self) -> None:
        with pytest.raises(Exception):  # ConfigurationError
            WorkerAutoscalerConfig(min_workers=50, max_workers=10)

    def test_to_dict(self) -> None:
        config = WorkerAutoscalerConfig(
            enabled=True,
            base_workers=10,
            min_workers=5,
            max_workers=20,
        )
        data = config.to_dict()
        assert data["enabled"] is True
        assert data["base_workers"] == 10
        assert data["min_workers"] == 5
        assert data["max_workers"] == 20


class TestWorkerAutoscaler:
    def test_init(self) -> None:
        config = WorkerAutoscalerConfig(enabled=True)
        autoscaler = WorkerAutoscaler(config)
        assert autoscaler.config.enabled is True

    def test_get_optimal_worker_count_disabled(self) -> None:
        config = WorkerAutoscalerConfig(enabled=False, base_workers=10)
        autoscaler = WorkerAutoscaler(config)
        count = autoscaler.get_optimal_worker_count(iteration=1)
        assert count == 10  # Returns base_workers when disabled

    def test_get_optimal_worker_count_enabled(self) -> None:
        config = WorkerAutoscalerConfig(
            enabled=True,
            base_workers=10,
            min_workers=5,
            max_workers=20,
        )
        autoscaler = WorkerAutoscaler(config)
        count = autoscaler.get_optimal_worker_count(iteration=1)
        assert 5 <= count <= 20

    def test_get_scaling_decision(self) -> None:
        config = WorkerAutoscalerConfig(
            enabled=True,
            base_workers=10,
            min_workers=2,
            max_workers=50,
        )
        autoscaler = WorkerAutoscaler(config)
        decision = autoscaler.get_scaling_decision(
            current_workers=10,
            iteration=1,
        )
        assert isinstance(decision, ScalingDecision)
        assert decision.action in [
            ScalingAction.SCALE_UP,
            ScalingAction.SCALE_DOWN,
            ScalingAction.MAINTAIN,
        ]

    def test_get_scaling_decision_disabled(self) -> None:
        config = WorkerAutoscalerConfig(enabled=False, base_workers=10)
        autoscaler = WorkerAutoscaler(config)
        decision = autoscaler.get_scaling_decision(current_workers=10)
        assert decision.action == ScalingAction.MAINTAIN
        assert decision.reason == "Autoscaling disabled"

    def test_record_worker_metrics(self) -> None:
        config = WorkerAutoscalerConfig(enabled=True)
        autoscaler = WorkerAutoscaler(config)
        autoscaler.record_worker_metrics(
            worker_id=1,
            games_completed=10,
            avg_game_duration=300.0,
            preempted=False,
        )
        stats = autoscaler.get_statistics()
        assert stats["tracked_workers"] == 1

    def test_get_statistics(self) -> None:
        config = WorkerAutoscalerConfig(enabled=True)
        autoscaler = WorkerAutoscaler(config)
        stats = autoscaler.get_statistics()
        assert "enabled" in stats
        assert "base_workers" in stats
        assert "min_workers" in stats
        assert "max_workers" in stats
        assert "in_cooldown" in stats

    def test_reset(self) -> None:
        config = WorkerAutoscalerConfig(enabled=True)
        autoscaler = WorkerAutoscaler(config)
        autoscaler.record_worker_metrics(
            worker_id=1,
            games_completed=10,
            avg_game_duration=300.0,
        )
        assert autoscaler.get_statistics()["tracked_workers"] == 1

        autoscaler.reset()
        assert autoscaler.get_statistics()["tracked_workers"] == 0


class TestGetWorkerAutoscaler:
    def test_returns_autoscaler(self) -> None:
        autoscaler = get_worker_autoscaler()
        assert isinstance(autoscaler, WorkerAutoscaler)
