"""Unit tests for spot_availability module."""
from __future__ import annotations

import os
from datetime import datetime

from alpha_zero.cloud.gcp.spot_availability import (
    AvailabilityLevel,
    PreemptionEvent,
    SpotAvailabilityConfig,
    SpotAvailabilityTracker,
    SpotPattern,
    get_spot_tracker,
)


class TestAvailabilityLevel:
    def test_availability_values(self) -> None:
        assert AvailabilityLevel.HIGH.value == "high"
        assert AvailabilityLevel.MEDIUM.value == "medium"
        assert AvailabilityLevel.LOW.value == "low"

class TestSpotPattern:
    def test_creation(self) -> None:
        pattern = SpotPattern(
            hour_start=0,
            hour_end=9,
            days=[0, 1, 2, 3, 4],
            level=AvailabilityLevel.HIGH,
            worker_multiplier=1.5,
        )
        assert pattern.hour_start == 0
        assert pattern.hour_end == 9
        assert pattern.level == AvailabilityLevel.HIGH

    def test_matches_within_range(self) -> None:
        pattern = SpotPattern(
            hour_start=9,
            hour_end=17,
            days=[0, 1, 2, 3, 4],
            level=AvailabilityLevel.LOW,
            worker_multiplier=0.5,
        )
        # 10 AM on Monday
        dt = datetime(2024, 1, 1, 10, 0)  # Monday
        assert pattern.matches(dt) is True

class TestPreemptionEvent:
    def test_creation(self) -> None:
        event = PreemptionEvent(
            timestamp="2024-01-01T12:00:00Z",
            region="us-central1",
            gpu_type="NVIDIA_TESLA_T4",
            worker_id=1,
        )
        assert event.region == "us-central1"
        assert event.gpu_type == "NVIDIA_TESLA_T4"

    def test_to_dict(self) -> None:
        event = PreemptionEvent(
            timestamp="2024-01-01T12:00:00Z",
            region="us-west1",
            gpu_type="NVIDIA_L4",
            worker_id=2,
        )
        data = event.to_dict()
        assert data["region"] == "us-west1"
        assert data["gpu_type"] == "NVIDIA_L4"

class TestSpotAvailabilityConfig:
    def test_default_values(self) -> None:
        config = SpotAvailabilityConfig()
        assert config.history_hours == 168
        assert config.high_availability_multiplier == 1.5
        assert config.low_availability_multiplier == 0.5

    def test_from_env(self) -> None:
        os.environ["CIV6AZ_SPOT_HISTORY_HOURS"] = "72"
        try:
            config = SpotAvailabilityConfig.from_env()
            assert config.history_hours == 72
        finally:
            del os.environ["CIV6AZ_SPOT_HISTORY_HOURS"]

class TestSpotAvailabilityTracker:
    def test_init(self) -> None:
        config = SpotAvailabilityConfig()
        tracker = SpotAvailabilityTracker(config)
        assert tracker.config == config

    def test_get_current_availability(self) -> None:
        config = SpotAvailabilityConfig()
        tracker = SpotAvailabilityTracker(config)
        level = tracker.get_current_availability("us-central1", "NVIDIA_TESLA_T4")
        assert isinstance(level, AvailabilityLevel)

    def test_get_recommended_worker_count(self) -> None:
        config = SpotAvailabilityConfig(
            high_availability_multiplier=1.5,
            low_availability_multiplier=0.5,
            min_workers=2,
            max_workers=20,
        )
        tracker = SpotAvailabilityTracker(config)
        count = tracker.get_recommended_worker_count(10)
        assert 2 <= count <= 20

    def test_record_preemption(self) -> None:
        config = SpotAvailabilityConfig()
        tracker = SpotAvailabilityTracker(config)
        tracker.record_preemption("us-central1", "NVIDIA_TESLA_T4", worker_id=1)
        stats = tracker.get_statistics()
        assert stats["total_preemptions_tracked"] == 1

    def test_get_statistics(self) -> None:
        config = SpotAvailabilityConfig()
        tracker = SpotAvailabilityTracker(config)
        stats = tracker.get_statistics()
        assert "current_availability" in stats
        assert "total_preemptions_tracked" in stats

class TestGetSpotTracker:
    def test_returns_tracker(self) -> None:
        tracker = get_spot_tracker()
        assert isinstance(tracker, SpotAvailabilityTracker)
