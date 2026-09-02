"""Unit tests for the cloud training-job protocol types.

Covers ``TrainingJobInfo`` and ``JobStatus`` from ``alpha_zero.cloud.protocols``.
The ``VertexTrainer`` tests that used to live here went with
``alpha_zero/cloud/gcp/vertex_trainer.py`` (dead code, removed 2026-09-02).
"""

from __future__ import annotations

from datetime import datetime

from alpha_zero.cloud.protocols import JobStatus, TrainingJobInfo


class TestTrainingJobInfo:
    """Tests for TrainingJobInfo dataclass."""

    def test_is_terminal_succeeded(self) -> None:
        """Test terminal state check for succeeded job."""
        job = TrainingJobInfo(
            job_id="test",
            job_name="test",
            status=JobStatus.SUCCEEDED,
        )
        assert job.is_terminal() is True
        assert job.is_successful() is True

    def test_is_terminal_failed(self) -> None:
        """Test terminal state check for failed job."""
        job = TrainingJobInfo(
            job_id="test",
            job_name="test",
            status=JobStatus.FAILED,
            error_message="Out of memory",
        )
        assert job.is_terminal() is True
        assert job.is_successful() is False

    def test_is_terminal_running(self) -> None:
        """Test terminal state check for running job."""
        job = TrainingJobInfo(
            job_id="test",
            job_name="test",
            status=JobStatus.RUNNING,
        )
        assert job.is_terminal() is False

    def test_duration_seconds(self) -> None:
        """Test duration calculation."""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 12, 30, 0)

        job = TrainingJobInfo(
            job_id="test",
            job_name="test",
            status=JobStatus.SUCCEEDED,
            started_at=start,
            ended_at=end,
        )

        duration = job.duration_seconds()
        assert duration == 2.5 * 3600  # 2.5 hours in seconds

    def test_duration_none_when_incomplete(self) -> None:
        """Test duration is None when job hasn't completed."""
        job = TrainingJobInfo(
            job_id="test",
            job_name="test",
            status=JobStatus.RUNNING,
            started_at=datetime.now(),
        )

        assert job.duration_seconds() is None
