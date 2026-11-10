"""Unit tests for scripts/monitor_build.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.monitor_build import (
    TERMINAL_STATUSES,
    fetch_build,
    fetch_build_log_tail,
    monitor,
    parse_args,
    print_step_summary,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_build_response(
    status: str = "WORKING",
    num_steps: int = 9,
    step_status: str = "SUCCESS",
    failed_step: int | None = None,
    logs_bucket: str = "gs://test-logs",
) -> dict:
    """Build a synthetic Cloud Build API response."""
    steps = []
    for i in range(num_steps):
        s = step_status
        if failed_step is not None and i == failed_step:
            s = "FAILURE"
        steps.append({"id": f"step-{i}", "status": s})
    return {
        "status": status,
        "steps": steps,
        "logsBucket": logs_bucket,
        "logUrl": "https://console.cloud.google.com/cloud-build/builds/test-id",
    }


def _mock_urlopen(response_data: dict) -> MagicMock:
    """Create a mock urlopen that returns the given dict as JSON."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_data).encode()
    return mock_resp


# =============================================================================
# parse_args tests
# =============================================================================


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_build_id_required(self) -> None:
        """build_id is a required positional argument (project supplied explicitly)."""
        args = parse_args(["abc-123", "--project", "my-proj"])
        assert args.build_id == "abc-123"

    def test_project_is_required_without_env(self) -> None:
        """No built-in project id: with the env var unset, --project is mandatory."""
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("CIV6AZ_VERTEX_PROJECT", None)
            with pytest.raises(SystemExit):
                parse_args(["abc-123"])

    def test_project_default_env_var(self) -> None:
        """--project falls back to CIV6AZ_VERTEX_PROJECT env var."""
        with patch.dict("os.environ", {"CIV6AZ_VERTEX_PROJECT": "my-proj"}):
            args = parse_args(["abc-123"])
            assert args.project == "my-proj"

    def test_project_explicit(self) -> None:
        """Explicit --project overrides env var."""
        args = parse_args(["abc-123", "--project", "other-proj"])
        assert args.project == "other-proj"

    def test_poll_interval_default(self) -> None:
        args = parse_args(["abc-123"])
        assert args.poll_interval == 15

    def test_no_show_log(self) -> None:
        args = parse_args(["abc-123", "--no-show-log"])
        assert args.show_log is False


# =============================================================================
# print_step_summary tests
# =============================================================================


class TestStepSummary:
    """Tests for step summary output."""

    def test_all_success(self) -> None:
        """All steps SUCCESS returns (9, 9)."""
        data = _make_build_response(num_steps=9, step_status="SUCCESS")
        passed, total = print_step_summary(data["steps"])
        assert passed == 9
        assert total == 9

    def test_partial_failure(self) -> None:
        """One failed step reduces passed count."""
        data = _make_build_response(num_steps=9, step_status="SUCCESS", failed_step=3)
        passed, total = print_step_summary(data["steps"])
        assert passed == 8
        assert total == 9

    def test_all_queued(self) -> None:
        """Non-SUCCESS steps count as not passed."""
        data = _make_build_response(num_steps=5, step_status="QUEUED")
        passed, total = print_step_summary(data["steps"])
        assert passed == 0
        assert total == 5


# =============================================================================
# fetch_build tests
# =============================================================================


class TestFetchBuild:
    """Tests for the REST API fetch."""

    @patch("scripts.monitor_build.urllib.request.urlopen")
    def test_returns_parsed_json(self, mock_urlopen_fn: MagicMock) -> None:
        expected = _make_build_response(status="SUCCESS")
        mock_urlopen_fn.return_value = _mock_urlopen(expected)

        result = fetch_build("build-123", "proj-1", "fake-token")
        assert result["status"] == "SUCCESS"
        assert len(result["steps"]) == 9


# =============================================================================
# fetch_build_log_tail tests
# =============================================================================


class TestFetchBuildLogTail:
    """Tests for log tail fetching."""

    @patch("scripts.monitor_build.urllib.request.urlopen")
    def test_returns_last_n_lines(self, mock_urlopen_fn: MagicMock) -> None:
        log_text = "\n".join(f"line-{i}" for i in range(100))
        mock_resp = MagicMock()
        mock_resp.read.return_value = log_text.encode()
        mock_urlopen_fn.return_value = mock_resp

        tail = fetch_build_log_tail("gs://bucket", "build-1", "token", lines=5)
        assert tail is not None
        assert len(tail.split("\n")) == 5

    @patch("scripts.monitor_build.urllib.request.urlopen")
    def test_returns_none_on_error(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.side_effect = Exception("network error")
        tail = fetch_build_log_tail("gs://bucket", "build-1", "token")
        assert tail is None


# =============================================================================
# Terminal status detection
# =============================================================================


class TestTerminalStatuses:
    """Verify the set of terminal statuses."""

    @pytest.mark.parametrize(
        "status",
        ["SUCCESS", "FAILURE", "TIMEOUT", "CANCELLED", "EXPIRED", "INTERNAL_ERROR"],
    )
    def test_is_terminal(self, status: str) -> None:
        assert status in TERMINAL_STATUSES

    @pytest.mark.parametrize("status", ["WORKING", "QUEUED", "STATUS_UNKNOWN"])
    def test_is_not_terminal(self, status: str) -> None:
        assert status not in TERMINAL_STATUSES


# =============================================================================
# monitor integration tests
# =============================================================================


class TestMonitor:
    """Tests for the poll loop."""

    @patch("scripts.monitor_build._get_auth_token", return_value="fake-token")
    @patch("scripts.monitor_build.fetch_build")
    def test_success_returns_zero(
        self,
        mock_fetch: MagicMock,
        _mock_token: MagicMock,
    ) -> None:
        """Immediate success → exit 0."""
        mock_fetch.return_value = _make_build_response(
            status="SUCCESS", num_steps=9, step_status="SUCCESS"
        )
        result = monitor("build-1", "proj", poll_interval=0)
        assert result == 0

    @patch("scripts.monitor_build._get_auth_token", return_value="fake-token")
    @patch("scripts.monitor_build.fetch_build")
    def test_failure_returns_one(
        self,
        mock_fetch: MagicMock,
        _mock_token: MagicMock,
    ) -> None:
        """Build FAILURE → exit 1."""
        mock_fetch.return_value = _make_build_response(
            status="FAILURE", num_steps=9, step_status="SUCCESS", failed_step=5
        )
        result = monitor("build-1", "proj", poll_interval=0, show_log_on_failure=False)
        assert result == 1

    @patch("scripts.monitor_build._get_auth_token", return_value="fake-token")
    @patch("scripts.monitor_build.fetch_build")
    @patch("scripts.monitor_build.time.sleep")
    def test_polls_until_terminal(
        self,
        mock_sleep: MagicMock,
        mock_fetch: MagicMock,
        _mock_token: MagicMock,
    ) -> None:
        """Poll loop runs until terminal status."""
        working = _make_build_response(status="WORKING", step_status="QUEUED")
        success = _make_build_response(status="SUCCESS", step_status="SUCCESS")
        mock_fetch.side_effect = [working, working, success]
        result = monitor("build-1", "proj", poll_interval=1)
        assert result == 0
        assert mock_sleep.call_count == 2
