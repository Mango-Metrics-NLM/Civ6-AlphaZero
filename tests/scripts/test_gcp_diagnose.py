"""Tests for scripts/gcp_diagnose.py.

Covers all check functions, report aggregation, fix application,
and CLI output modes. Uses mocks for external I/O (subprocess, network).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
# The script lives in scripts/ which may not be on sys.path, so import
# it via importlib to keep things flexible.
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import gcp_diagnose as diag  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure test isolation — remove GCP env vars."""
    for var in (
        diag._DEFAULT_PROJECT_ENV,
        diag._DEFAULT_BUCKET_ENV,
        diag._DEFAULT_REGION_ENV,
        diag._GCLOUD_UPDATE_CHECK_ENV,
        diag._GCLOUD_DISABLE_PROMPTS_ENV,
    ):
        monkeypatch.delenv(var, raising=False)


# ===========================================================================
# Unit Tests — CheckResult / DiagnosticReport
# ===========================================================================


class TestCheckResult:
    """Tests for the CheckResult dataclass."""

    def test_creation(self) -> None:
        r = diag.CheckResult(name="test", status=diag.CheckStatus.OK, message="all good")
        assert r.name == "test"
        assert r.status == diag.CheckStatus.OK
        assert r.details == {}

    def test_creation_with_details(self) -> None:
        r = diag.CheckResult(
            name="test",
            status=diag.CheckStatus.WARN,
            message="warning",
            details={"key": "value"},
        )
        assert r.details["key"] == "value"


class TestDiagnosticReport:
    """Tests for the DiagnosticReport dataclass."""

    def test_add_ok_keeps_overall_ok(self) -> None:
        report = diag.DiagnosticReport()
        report.add(diag.CheckResult("a", diag.CheckStatus.OK, "fine"))
        assert report.overall_status == diag.CheckStatus.OK

    def test_add_warn_upgrades_overall(self) -> None:
        report = diag.DiagnosticReport()
        report.add(diag.CheckResult("a", diag.CheckStatus.OK, "fine"))
        report.add(diag.CheckResult("b", diag.CheckStatus.WARN, "hmm"))
        assert report.overall_status == diag.CheckStatus.WARN

    def test_add_fail_upgrades_overall(self) -> None:
        report = diag.DiagnosticReport()
        report.add(diag.CheckResult("a", diag.CheckStatus.WARN, "hmm"))
        report.add(diag.CheckResult("b", diag.CheckStatus.FAIL, "bad"))
        assert report.overall_status == diag.CheckStatus.FAIL

    def test_fail_not_downgraded_by_warn(self) -> None:
        report = diag.DiagnosticReport()
        report.add(diag.CheckResult("a", diag.CheckStatus.FAIL, "bad"))
        report.add(diag.CheckResult("b", diag.CheckStatus.WARN, "hmm"))
        assert report.overall_status == diag.CheckStatus.FAIL

    def test_to_dict(self) -> None:
        report = diag.DiagnosticReport(timestamp="2026-01-01")
        report.add(diag.CheckResult("x", diag.CheckStatus.OK, "good"))
        d = report.to_dict()
        assert d["overall_status"] == "ok"
        assert d["checks"][0]["status"] == "ok"
        assert d["timestamp"] == "2026-01-01"


# ===========================================================================
# Unit Tests — Individual Checks (mocked I/O)
# ===========================================================================


class TestCheckNetworkConnectivity:
    """Tests for check_network_connectivity."""

    @mock.patch("urllib.request.urlopen")
    def test_all_reachable(self, mock_urlopen: mock.Mock) -> None:
        mock_urlopen.return_value = mock.Mock()
        result = diag.check_network_connectivity()
        assert result.status == diag.CheckStatus.OK
        assert len(result.details["reachable"]) == len(diag._NETWORK_TEST_URLS)

    @mock.patch("urllib.request.urlopen")
    def test_http_errors_still_reachable(self, mock_urlopen: mock.Mock) -> None:
        """HTTP 400/404 errors count as reachable."""
        from urllib.error import HTTPError

        mock_urlopen.side_effect = HTTPError(
            url="",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,  # type: ignore[arg-type]
        )
        result = diag.check_network_connectivity()
        assert result.status == diag.CheckStatus.OK


class TestCheckGcloudExecutable:
    """Tests for check_gcloud_executable."""

    @mock.patch("subprocess.run")
    def test_found(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="C:\\gcloud.cmd\n"
        )
        result = diag.check_gcloud_executable()
        assert result.status == diag.CheckStatus.OK
        assert "C:\\gcloud.cmd" in result.message

    @mock.patch("subprocess.run")
    def test_not_found(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="")
        result = diag.check_gcloud_executable()
        assert result.status == diag.CheckStatus.FAIL

    @mock.patch("subprocess.run")
    def test_timeout(self, mock_run: mock.Mock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="", timeout=5)
        result = diag.check_gcloud_executable()
        assert result.status == diag.CheckStatus.FAIL


class TestCheckGcloudAuth:
    """Tests for check_gcloud_auth."""

    @mock.patch("subprocess.run")
    def test_active_account(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='[{"account": "user@example.com", "status": "ACTIVE"}]',
        )
        result = diag.check_gcloud_auth()
        assert result.status == diag.CheckStatus.OK
        assert "user@example.com" in result.message

    @mock.patch("subprocess.run")
    def test_no_active_account(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='[{"account": "user@example.com", "status": "DISABLED"}]',
        )
        result = diag.check_gcloud_auth()
        assert result.status == diag.CheckStatus.FAIL

    @mock.patch("subprocess.run")
    def test_timeout(self, mock_run: mock.Mock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="", timeout=10)
        result = diag.check_gcloud_auth()
        assert result.status == diag.CheckStatus.WARN
        assert "timed out" in result.message


class TestCheckPythonAuth:
    """Tests for check_python_auth."""

    @mock.patch("google.auth.default")
    def test_adc_available(self, mock_default: mock.Mock) -> None:
        mock_creds = mock.Mock()
        mock_creds.valid = True
        mock_default.return_value = (mock_creds, "test-project")
        result = diag.check_python_auth()
        assert result.status == diag.CheckStatus.OK
        assert "test-project" in result.message

    @mock.patch.dict("sys.modules", {"google.auth": None})
    def test_import_error(self) -> None:
        # When google.auth is set to None in sys.modules, import will
        # raise ImportError
        with mock.patch.dict("sys.modules", {"google": None}):
            result = diag.check_python_auth()
            # Either SKIP or FAIL depending on import behavior
            assert result.status in (
                diag.CheckStatus.SKIP,
                diag.CheckStatus.FAIL,
            )


class TestCheckEnvVars:
    """Tests for check_env_vars."""

    def test_all_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(diag._DEFAULT_PROJECT_ENV, "my-project")
        monkeypatch.setenv(diag._DEFAULT_BUCKET_ENV, "my-bucket")
        result = diag.check_env_vars()
        assert result.status == diag.CheckStatus.OK
        assert "my-project" in result.message

    def test_missing_project(self) -> None:
        result = diag.check_env_vars()
        assert result.status == diag.CheckStatus.WARN
        assert diag._DEFAULT_PROJECT_ENV in result.message

    def test_missing_bucket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(diag._DEFAULT_PROJECT_ENV, "my-project")
        result = diag.check_env_vars()
        assert result.status == diag.CheckStatus.WARN
        assert diag._DEFAULT_BUCKET_ENV in result.message


class TestCheckUpdateCheckSetting:
    """Tests for check_update_check_setting."""

    def test_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(diag._GCLOUD_UPDATE_CHECK_ENV, "1")
        result = diag.check_update_check_setting()
        assert result.status == diag.CheckStatus.OK

    def test_not_set(self) -> None:
        result = diag.check_update_check_setting()
        assert result.status == diag.CheckStatus.WARN


# ===========================================================================
# Integration Tests — run_diagnostics / apply_fixes
# ===========================================================================


class TestRunDiagnostics:
    """Tests for the run_diagnostics orchestrator."""

    @mock.patch("gcp_diagnose.check_vertex_api")
    @mock.patch("gcp_diagnose.check_gcs_connectivity")
    @mock.patch("gcp_diagnose.check_env_vars")
    @mock.patch("gcp_diagnose.check_python_auth")
    @mock.patch("gcp_diagnose.check_gcloud_auth")
    @mock.patch("gcp_diagnose.check_update_check_setting")
    @mock.patch("gcp_diagnose.check_gcloud_executable")
    @mock.patch("gcp_diagnose.check_network_connectivity")
    def test_all_ok(self, *mocks: mock.Mock) -> None:
        for i, m in enumerate(mocks):
            m.__name__ = f"mock_check_{i}"
            m.return_value = diag.CheckResult("mocked", diag.CheckStatus.OK, "ok")
        report = diag.run_diagnostics()
        assert report.overall_status == diag.CheckStatus.OK
        assert len(report.checks) == 8

    @mock.patch("gcp_diagnose.check_vertex_api")
    @mock.patch("gcp_diagnose.check_gcs_connectivity")
    @mock.patch("gcp_diagnose.check_env_vars")
    @mock.patch("gcp_diagnose.check_python_auth")
    @mock.patch("gcp_diagnose.check_gcloud_auth")
    @mock.patch("gcp_diagnose.check_update_check_setting")
    @mock.patch("gcp_diagnose.check_gcloud_executable")
    @mock.patch("gcp_diagnose.check_network_connectivity")
    def test_exception_in_check(self, *mocks: mock.Mock) -> None:
        for i, m in enumerate(mocks):
            m.__name__ = f"mock_check_{i}"
        mocks[0].side_effect = RuntimeError("boom")
        for m in mocks[1:]:
            m.return_value = diag.CheckResult("mocked", diag.CheckStatus.OK, "ok")
        report = diag.run_diagnostics()
        assert report.overall_status == diag.CheckStatus.FAIL
        failed = [c for c in report.checks if c.status == diag.CheckStatus.FAIL]
        assert len(failed) == 1
        assert "boom" in failed[0].message


class TestApplyFixes:
    """Tests for apply_fixes."""

    def test_sets_update_check_env(self) -> None:
        report = diag.DiagnosticReport()
        report.add(
            diag.CheckResult(
                "update_check_disabled",
                diag.CheckStatus.WARN,
                "not set",
            )
        )
        fixes = diag.apply_fixes(report)
        assert len(fixes) == 1
        assert os.environ.get(diag._GCLOUD_UPDATE_CHECK_ENV) == "1"

    def test_recommends_auth_login(self) -> None:
        report = diag.DiagnosticReport()
        report.add(
            diag.CheckResult(
                "gcloud_auth",
                diag.CheckStatus.FAIL,
                "no account",
            )
        )
        diag.apply_fixes(report)
        assert any("gcloud auth login" in r for r in report.recommendations)


# ===========================================================================
# CLI / Output Tests
# ===========================================================================


class TestPrintReport:
    """Tests for print_report."""

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = diag.DiagnosticReport(timestamp="2026-01-01")
        report.add(diag.CheckResult("x", diag.CheckStatus.OK, "good"))
        diag.print_report(report, as_json=True)
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed["overall_status"] == "ok"

    def test_text_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = diag.DiagnosticReport(timestamp="2026-01-01")
        report.add(diag.CheckResult("x", diag.CheckStatus.FAIL, "bad"))
        report.recommendations.append("fix it")
        diag.print_report(report)
        output = capsys.readouterr().out
        assert "[FAIL]" in output
        assert "fix it" in output
        assert "Diagnostic Report" in output


class TestMain:
    """Tests for main() entry point."""

    @mock.patch("gcp_diagnose.run_diagnostics")
    def test_returns_zero_on_ok(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = diag.DiagnosticReport(overall_status=diag.CheckStatus.OK)
        with mock.patch("sys.argv", ["gcp_diagnose.py"]):
            assert diag.main() == 0

    @mock.patch("gcp_diagnose.run_diagnostics")
    def test_returns_one_on_fail(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = diag.DiagnosticReport(overall_status=diag.CheckStatus.FAIL)
        with mock.patch("sys.argv", ["gcp_diagnose.py"]):
            assert diag.main() == 1

    @mock.patch("gcp_diagnose.run_diagnostics")
    def test_json_flag(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = diag.DiagnosticReport(overall_status=diag.CheckStatus.OK)
        with mock.patch("sys.argv", ["gcp_diagnose.py", "--json"]):
            assert diag.main() == 0
            mock_run.assert_called_once()

    @mock.patch("gcp_diagnose.run_diagnostics")
    def test_fix_flag(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = diag.DiagnosticReport(overall_status=diag.CheckStatus.OK)
        with mock.patch("sys.argv", ["gcp_diagnose.py", "--fix"]):
            diag.main()
            mock_run.assert_called_once_with(apply_fix=True)
