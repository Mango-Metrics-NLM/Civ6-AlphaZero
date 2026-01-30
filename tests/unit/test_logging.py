"""Unit tests for the logging infrastructure.

Tests for:
- MetricsTracker: statistics aggregation and thread safety
- TrainingLogger: training metrics logging
- GameLogger: game state recording and analysis
- PerformanceProfiler: timing and profiling
- setup_logging: configuration and initialization
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from alpha_zero.utils.logging import (
    CircularBuffer,
    GameLogger,
    LogConfig,
    MetricsTracker,
    PerformanceProfiler,
    TrainingLogger,
    get_logger,
    get_profiler,
    setup_logging,
    timed_section,
    timer,
)


class TestCircularBuffer:
    """Tests for CircularBuffer class."""

    def test_basic_append_and_get(self) -> None:
        """Test basic append and retrieval operations."""
        buffer = CircularBuffer(maxsize=5)
        for i in range(3):
            buffer.append({"value": i})

        recent = buffer.get_recent()
        assert len(recent) == 3
        assert recent[0]["value"] == 0
        assert recent[2]["value"] == 2

    def test_max_size_enforcement(self) -> None:
        """Test that buffer respects max size."""
        buffer = CircularBuffer(maxsize=3)
        for i in range(5):
            buffer.append({"value": i})

        recent = buffer.get_recent()
        assert len(recent) == 3
        # Should have the last 3 items
        assert recent[0]["value"] == 2
        assert recent[2]["value"] == 4

    def test_get_recent_n(self) -> None:
        """Test getting specific number of recent items."""
        buffer = CircularBuffer(maxsize=10)
        for i in range(10):
            buffer.append({"value": i})

        recent_3 = buffer.get_recent(3)
        assert len(recent_3) == 3
        assert recent_3[0]["value"] == 7
        assert recent_3[2]["value"] == 9

    def test_clear(self) -> None:
        """Test clearing the buffer."""
        buffer = CircularBuffer(maxsize=5)
        buffer.append({"value": 1})
        buffer.clear()
        assert len(buffer) == 0

    def test_thread_safety(self) -> None:
        """Test thread-safe operations."""
        buffer = CircularBuffer(maxsize=1000)

        def append_items(start: int, count: int) -> None:
            for i in range(start, start + count):
                buffer.append({"value": i})

        threads = [
            threading.Thread(target=append_items, args=(i * 100, 100))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(buffer) == 500


class TestMetricsTracker:
    """Tests for MetricsTracker class."""

    def test_basic_update_and_summary(self) -> None:
        """Test basic metric update and summary."""
        tracker = MetricsTracker()
        tracker.update("loss", 1.0)
        tracker.update("loss", 2.0)
        tracker.update("loss", 3.0)

        summary = tracker.get_summary("loss")
        assert summary["count"] == 3
        assert summary["mean"] == 2.0
        assert summary["min"] == 1.0
        assert summary["max"] == 3.0
        assert summary["last"] == 3.0

    def test_window_size(self) -> None:
        """Test sliding window for running statistics."""
        tracker = MetricsTracker(window_size=3)

        for i in range(10):
            tracker.update("value", float(i))

        summary = tracker.get_summary("value")
        # Window should contain last 3 values: 7, 8, 9
        assert summary["count"] == 3
        assert summary["mean"] == 8.0

    def test_update_batch(self) -> None:
        """Test batch metric update."""
        tracker = MetricsTracker()
        tracker.update_batch({
            "loss": 0.5,
            "accuracy": 0.9,
            "lr": 0.001,
        })

        assert tracker.get_summary("loss")["last"] == 0.5
        assert tracker.get_summary("accuracy")["last"] == 0.9
        assert tracker.get_summary("lr")["last"] == 0.001

    def test_running_average(self) -> None:
        """Test running average calculation."""
        tracker = MetricsTracker(window_size=5)
        for i in range(1, 6):
            tracker.update("x", float(i))

        avg = tracker.get_running_average("x")
        assert avg == 3.0  # (1+2+3+4+5)/5

    def test_get_all_metrics(self) -> None:
        """Test getting all tracked metrics."""
        tracker = MetricsTracker()
        tracker.update("a", 1.0)
        tracker.update("b", 2.0)

        all_metrics = tracker.get_all_metrics()
        assert "a" in all_metrics
        assert "b" in all_metrics
        assert all_metrics["a"]["mean"] == 1.0
        assert all_metrics["b"]["mean"] == 2.0

    def test_get_history(self) -> None:
        """Test getting full metric history."""
        tracker = MetricsTracker(window_size=3)
        for i in range(5):
            tracker.update("x", float(i))

        history = tracker.get_history("x")
        # History should contain all values, not just window
        assert len(history) == 5
        assert history == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_reset_single_metric(self) -> None:
        """Test resetting a single metric."""
        tracker = MetricsTracker()
        tracker.update("a", 1.0)
        tracker.update("b", 2.0)
        tracker.reset("a")

        assert tracker.get_summary("a")["count"] == 0
        assert tracker.get_summary("b")["count"] == 1

    def test_reset_all_metrics(self) -> None:
        """Test resetting all metrics."""
        tracker = MetricsTracker()
        tracker.update("a", 1.0)
        tracker.update("b", 2.0)
        tracker.reset()

        all_metrics = tracker.get_all_metrics()
        assert len(all_metrics) == 0

    def test_thread_safety(self) -> None:
        """Test thread-safe metric updates."""
        tracker = MetricsTracker()

        def update_metrics(name: str, count: int) -> None:
            for i in range(count):
                tracker.update(name, float(i))

        threads = [
            threading.Thread(target=update_metrics, args=(f"metric_{i}", 100))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each metric should have 100 updates
        for i in range(5):
            assert tracker.get_summary(f"metric_{i}")["count"] == 100


class TestPerformanceProfiler:
    """Tests for PerformanceProfiler class."""

    def test_section_timing(self) -> None:
        """Test timing a code section."""
        profiler = PerformanceProfiler()

        with profiler.section("test"):
            time.sleep(0.01)

        summary = profiler.get_summary("test")
        assert summary["count"] == 1
        assert summary["mean"] >= 0.01

    def test_multiple_sections(self) -> None:
        """Test timing multiple sections."""
        profiler = PerformanceProfiler()

        for _ in range(3):
            with profiler.section("fast"):
                pass
            with profiler.section("slow"):
                time.sleep(0.01)

        fast_summary = profiler.get_summary("fast")
        slow_summary = profiler.get_summary("slow")

        assert fast_summary["count"] == 3
        assert slow_summary["count"] == 3
        assert slow_summary["mean"] > fast_summary["mean"]

    def test_manual_start_stop(self) -> None:
        """Test manual start/stop timing."""
        profiler = PerformanceProfiler()

        profiler.start("manual")
        time.sleep(0.01)
        elapsed = profiler.stop("manual")

        assert elapsed >= 0.01
        assert profiler.get_summary("manual")["count"] == 1

    def test_stop_without_start(self) -> None:
        """Test stopping without starting."""
        profiler = PerformanceProfiler()
        elapsed = profiler.stop("nonexistent")
        assert elapsed == 0.0

    def test_reset_single(self) -> None:
        """Test resetting single section timing."""
        profiler = PerformanceProfiler()

        with profiler.section("a"):
            pass
        with profiler.section("b"):
            pass

        profiler.reset("a")

        assert profiler.get_summary("a") == {}
        assert profiler.get_summary("b")["count"] == 1

    def test_reset_all(self) -> None:
        """Test resetting all timing data."""
        profiler = PerformanceProfiler()

        with profiler.section("a"):
            pass
        with profiler.section("b"):
            pass

        profiler.reset()

        assert profiler.get_summary() == {}

    def test_report_generation(self) -> None:
        """Test report generation."""
        profiler = PerformanceProfiler()

        with profiler.section("test_section"):
            time.sleep(0.01)

        report = profiler.report()
        assert "test_section" in report
        assert "Calls:" in report
        assert "Total:" in report


class TestTimingHelpers:
    """Tests for timing helper functions."""

    def test_timed_section(self) -> None:
        """Test timed_section context manager."""
        with timed_section("test") as timing:
            time.sleep(0.01)

        assert "elapsed" in timing
        assert timing["elapsed"] >= 0.01

    def test_timer_decorator(self) -> None:
        """Test timer decorator."""
        @timer("decorated_test")
        def test_func() -> str:
            time.sleep(0.01)
            return "result"

        result = test_func()
        assert result == "result"

        profiler = get_profiler()
        summary = profiler.get_summary("decorated_test")
        assert summary["count"] >= 1


class TestTrainingLogger:
    """Tests for TrainingLogger class."""

    def test_log_training_step(self) -> None:
        """Test logging a training step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(
                log_dir=tmpdir,
                tensorboard=False,
                json_log=True,
            )

            logger.log_training_step(
                iteration=1,
                loss=0.5,
                policy_loss=0.3,
                value_loss=0.2,
                learning_rate=0.001,
            )
            logger.close()

            # Check JSON log was created
            jsonl_files = list(Path(tmpdir).glob("*.jsonl"))
            assert len(jsonl_files) == 1

            with open(jsonl_files[0]) as f:
                entry = json.loads(f.readline())

            assert entry["type"] == "training_step"
            assert entry["iteration"] == 1
            assert entry["loss"] == 0.5

    def test_log_evaluation(self) -> None:
        """Test logging evaluation results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(
                log_dir=tmpdir,
                tensorboard=False,
                json_log=True,
            )

            logger.log_evaluation(
                iteration=100,
                win_rate=0.75,
                games_played=20,
                opponent="random",
            )
            logger.close()

            jsonl_files = list(Path(tmpdir).glob("*.jsonl"))
            with open(jsonl_files[0]) as f:
                entry = json.loads(f.readline())

            assert entry["type"] == "evaluation"
            assert entry["win_rate"] == 0.75
            assert entry["opponent"] == "random"

    def test_log_selfplay(self) -> None:
        """Test logging self-play statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(
                log_dir=tmpdir,
                tensorboard=False,
                json_log=True,
            )

            logger.log_selfplay(
                iteration=10,
                num_games=100,
                total_examples=5000,
                avg_game_length=50.0,
                time_elapsed=120.0,
            )
            logger.close()

            jsonl_files = list(Path(tmpdir).glob("*.jsonl"))
            with open(jsonl_files[0]) as f:
                entry = json.loads(f.readline())

            assert entry["type"] == "selfplay"
            assert entry["num_games"] == 100

    def test_metrics_summary(self) -> None:
        """Test getting metrics summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(
                log_dir=tmpdir,
                tensorboard=False,
                json_log=False,
            )

            for i in range(5):
                logger.log_training_step(
                    iteration=i,
                    loss=0.5 - i * 0.1,
                    policy_loss=0.3,
                    value_loss=0.2,
                )

            summary = logger.get_metrics_summary()
            logger.close()

            assert "loss" in summary
            assert summary["loss"]["count"] == 5


class TestGameLogger:
    """Tests for GameLogger class."""

    def test_basic_game_logging(self) -> None:
        """Test basic game logging flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = GameLogger(log_dir=tmpdir)

            logger.start_game(game_id=1, config={"board_size": 8})
            logger.log_state(
                turn=1,
                player=0,
                action=5,
                value_estimate=0.3,
            )
            logger.log_state(
                turn=2,
                player=1,
                action=10,
                value_estimate=-0.2,
            )
            game_data = logger.end_game(winner=0, total_turns=2)

            assert game_data["game_id"] == 1
            assert len(game_data["turns"]) == 2
            assert game_data["winner"] == 0

    def test_action_probs_logging(self) -> None:
        """Test logging action probabilities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = GameLogger(log_dir=tmpdir)

            logger.start_game(game_id=1)
            action_probs = np.array([0.1, 0.2, 0.3, 0.4])
            logger.log_state(
                turn=1,
                player=0,
                action=3,
                action_probs=action_probs,
            )
            game_data = logger.end_game(winner=0, total_turns=1)

            turn_data = game_data["turns"][0]
            assert "top_actions" in turn_data
            assert turn_data["action_prob"] == 0.4

    def test_save_game(self) -> None:
        """Test saving game to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = GameLogger(log_dir=tmpdir)

            logger.start_game(game_id=42)
            logger.log_state(turn=1, player=0, action=1)
            game_data = logger.end_game(winner=1, total_turns=1)

            saved_path = logger.save_game(game_data)
            assert saved_path.exists()

            with open(saved_path) as f:
                loaded = json.load(f)
            assert loaded["game_id"] == 42

    def test_recent_games_buffer(self) -> None:
        """Test recent games circular buffer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = GameLogger(log_dir=tmpdir, max_games_in_memory=3)

            for i in range(5):
                logger.start_game(game_id=i)
                logger.end_game(winner=0, total_turns=1)

            recent = logger.get_recent_games()
            assert len(recent) == 3
            # Should have games 2, 3, 4
            assert recent[0]["game_id"] == 2
            assert recent[2]["game_id"] == 4

    def test_game_analysis(self) -> None:
        """Test game analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = GameLogger(log_dir=tmpdir)

            logger.start_game(game_id=1)
            for i in range(10):
                logger.log_state(
                    turn=i,
                    player=i % 2,
                    action=i % 3,  # Use valid action index for size-3 array
                    value_estimate=0.1 * i,
                    action_probs=np.array([0.5, 0.3, 0.2]),
                )
            game_data = logger.end_game(winner=0, total_turns=10)

            analysis = logger.analyze_game(game_data)

            assert analysis["total_turns"] == 10
            assert "value_estimates" in analysis
            assert analysis["value_estimates"]["min"] == 0.0

    def test_mcts_search_logging(self) -> None:
        """Test logging MCTS search statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = GameLogger(log_dir=tmpdir)

            logger.start_game(game_id=1)
            logger.log_state(turn=1, player=0, action=5)
            logger.log_mcts_search(
                turn=1,
                root_visits=800,
                max_depth=15,
                search_time=1.5,
                nodes_expanded=2000,
                q_values={5: 0.3, 10: 0.2},
            )
            game_data = logger.end_game(winner=0, total_turns=1)

            turn_data = game_data["turns"][0]
            assert "mcts_search" in turn_data
            assert turn_data["mcts_search"]["root_visits"] == 800


class TestSetupLogging:
    """Tests for logging setup functions."""

    def test_basic_setup(self) -> None:
        """Test basic logging setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(
                level="DEBUG",
                log_dir=tmpdir,
                console_output=False,
                file_output=False,
            )

            logger = get_logger("test_module")
            assert logger is not None
            logger.info("Test message")

    def test_config_object(self) -> None:
        """Test setup with LogConfig object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(
                level="WARNING",
                log_dir=tmpdir,
                console_output=False,
                file_output=False,
            )
            setup_logging(config=config)

            logger = get_logger("test")
            assert logger is not None

    def test_hierarchical_loggers(self) -> None:
        """Test hierarchical logger creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir, console_output=False, file_output=False)

            logger1 = get_logger("module.submodule")
            logger2 = get_logger("alpha_zero.training")

            assert "alpha_zero.module.submodule" in logger1.name
            assert logger2.name == "alpha_zero.training"
