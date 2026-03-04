"""Metrics collection for RAG system.

Provides:
- Timer for measuring operation duration
- Counter for counting events
- Gauge for tracking values
- MetricsCollector for aggregating metrics
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from alpha_zero.utils.compat import UTC


@dataclass
class Timer:
    """Timer for measuring operation duration.

    Example:
        timer = Timer("search_latency")
        with timer:
            # perform operation
            pass
        print(f"Duration: {timer.duration_ms}ms")
    """

    name: str
    start_time: float | None = None
    end_time: float | None = None
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Get duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def start(self) -> Timer:
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> Timer:
        """Stop the timer."""
        self.end_time = time.perf_counter()
        return self

    def __enter__(self) -> Timer:
        """Context manager entry."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Context manager exit."""
        self.stop()


@dataclass
class Counter:
    """Counter for counting events.

    Example:
        counter = Counter("documents_processed")
        counter.inc()
        counter.inc(5)
        print(f"Total: {counter.value}")
    """

    name: str
    value: int = 0
    labels: dict[str, str] = field(default_factory=dict)

    def inc(self, amount: int = 1) -> int:
        """Increment counter.

        Args:
            amount: Amount to increment by

        Returns:
            New value
        """
        self.value += amount
        return self.value

    def reset(self) -> None:
        """Reset counter to zero."""
        self.value = 0


@dataclass
class Gauge:
    """Gauge for tracking values.

    Example:
        gauge = Gauge("active_connections")
        gauge.set(10)
        gauge.inc()
        gauge.dec(2)
    """

    name: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def set(self, value: float) -> None:
        """Set gauge value.

        Args:
            value: Value to set
        """
        self.value = value

    def inc(self, amount: float = 1.0) -> float:
        """Increment gauge.

        Args:
            amount: Amount to increment by

        Returns:
            New value
        """
        self.value += amount
        return self.value

    def dec(self, amount: float = 1.0) -> float:
        """Decrement gauge.

        Args:
            amount: Amount to decrement by

        Returns:
            New value
        """
        self.value -= amount
        return self.value


class MetricsCollector:
    """Collector for aggregating metrics.

    Example:
        collector = MetricsCollector()
        collector.timer("search_latency").start()
        # ... operation
        collector.timer("search_latency").stop()
        collector.counter("documents_processed").inc(10)

        print(collector.get_summary())
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._timers: dict[str, list[Timer]] = defaultdict(list)
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._start_time = datetime.now(UTC)

    def timer(self, name: str, **labels: str) -> Timer:
        """Get or create a timer.

        Args:
            name: Timer name
            **labels: Labels for the timer

        Returns:
            Timer instance
        """
        timer = Timer(name=name, labels=labels)
        self._timers[name].append(timer)
        return timer

    def counter(self, name: str, **labels: str) -> Counter:
        """Get or create a counter.

        Args:
            name: Counter name
            **labels: Labels for the counter

        Returns:
            Counter instance
        """
        if name not in self._counters:
            self._counters[name] = Counter(name=name, labels=labels)
        return self._counters[name]

    def gauge(self, name: str, **labels: str) -> Gauge:
        """Get or create a gauge.

        Args:
            name: Gauge name
            **labels: Labels for the gauge

        Returns:
            Gauge instance
        """
        if name not in self._gauges:
            self._gauges[name] = Gauge(name=name, labels=labels)
        return self._gauges[name]

    def get_timer_stats(self, name: str) -> dict[str, float]:
        """Get statistics for a timer.

        Args:
            name: Timer name

        Returns:
            Dictionary with count, min, max, avg, p50, p95, p99
        """
        timers = self._timers.get(name, [])
        durations = [t.duration_ms for t in timers if t.duration_ms is not None]

        if not durations:
            return {"count": 0}

        durations.sort()
        count = len(durations)

        return {
            "count": count,
            "min_ms": min(durations),
            "max_ms": max(durations),
            "avg_ms": sum(durations) / count,
            "p50_ms": durations[int(count * 0.5)],
            "p95_ms": durations[int(count * 0.95)] if count >= 20 else durations[-1],
            "p99_ms": durations[int(count * 0.99)] if count >= 100 else durations[-1],
        }

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all metrics.

        Returns:
            Dictionary with all metric values
        """
        uptime = (datetime.now(UTC) - self._start_time).total_seconds()

        summary = {
            "uptime_seconds": uptime,
            "timers": {},
            "counters": {},
            "gauges": {},
        }

        # Timer stats
        for name in self._timers:
            summary["timers"][name] = self.get_timer_stats(name)  # type: ignore[index]

        # Counter values
        for name, counter in self._counters.items():
            summary["counters"][name] = {  # type: ignore[index]
                "value": counter.value,
                "labels": counter.labels,
            }

        # Gauge values
        for name, gauge in self._gauges.items():
            summary["gauges"][name] = {  # type: ignore[index]
                "value": gauge.value,
                "labels": gauge.labels,
            }

        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        self._timers.clear()
        self._counters.clear()
        self._gauges.clear()
        self._start_time = datetime.now(UTC)
